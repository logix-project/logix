import os

import torch

from analog.utils import nested_dict, get_world_size, get_rank, get_logger


class AnaLogState:
    def __init__(self) -> None:
        self._states = set()
        self._states_to_synchronize = set()
        self._states_to_save = set()
        self._states_to_normalize = set()

        self.register_state("log_state")

        self.register_state("hessian_state", synchronize=True, save=True)
        self.register_state("hessian_counter", synchronize=True, save=False)

        self.register_normalize_pair("hessian_state", "hessian_counter")

    def register_state(
        self, state_name: str, synchronize: bool = False, save: bool = False
    ):
        """
        Register a state to be logged.
        """
        if state_name in self._states:
            return

        assert state_name not in self._states
        setattr(self, state_name, nested_dict())
        self._states.add(state_name)
        if synchronize:
            self._states_to_synchronize.add(state_name)
        if save:
            self._states_to_save.add(state_name)

    def register_normalize_pair(self, state_name: str, counter_name: str):
        """
        Add a normalization to the state.
        """
        if (state_name, counter_name) in self._states_to_normalize:
            return

        self._states_to_normalize.add((state_name, counter_name))

    @torch.no_grad()
    def hessian_svd(self) -> None:
        """
        Compute the SVD of the covariance.
        """
        self.register_state("hessian_eigval_state", save=True)
        self.register_state("hessian_eigvec_state", save=True)

        for module_name, module_state in self.hessian_state.items():
            for mode, covariance in module_state.items():
                eigvals, eigvecs = torch.linalg.eigh(covariance)
                self.hessian_eigval_state[module_name][mode] = eigvals
                self.hessian_eigvec_state[module_name][mode] = eigvecs

    @torch.no_grad()
    def hessian_inverse(self, set_attr: bool = False) -> None:
        """
        Compute the inverse of the covariance.
        """
        self.register_state("hessian_inverse_state", save=True)

        for module_name, module_state in self.hessian_state.items():
            for mode, covariance in module_state.items():
                self.hessian_inverse_state[module_name][mode] = torch.inverse(
                    covariance
                    + 0.01
                    * torch.trace(covariance)
                    * torch.eye(covariance.shape[0])
                    / covariance.shape[0]
                )

    def get_hessian_state(self):
        return self.hessian_state

    def get_hessian_inverse_state(self):
        if not hasattr(self, "hessian_inverse_state"):
            self.hessian_inverse()
        return self.hessian_inverse_state

    def get_hessian_svd_state(self):
        if not hasattr(self, "hessian_eigval_state") or not hasattr(
            self, "hessian_eigvec_state"
        ):
            self.hessian_svd()
        if hasattr(self, "ekfac_eigval_state"):
            return self.ekfac_eigval_state, self.hessian_eigvec_state
        return self.hessian_eigval_state, self.hessian_eigvec_state

    def synchronize(self) -> None:
        """
        Synchronize the Hessian state across processes.
        """

        # synchronize helper function
        def _synchronize(state_dict):
            for key in state_dict:
                if not isinstance(state_dict[key], torch.Tensor):
                    _synchronize(state_dict[key])
                else:
                    # we need to move the state to the GPU before all-reducing
                    # across processes as the communication is only
                    # supported on GPU tensors in most PyTorch backends.
                    if not isinstance(state_dict[key], torch.Tensor):
                        state_dict[key] = torch.tensor(state_dict[key])
                    state_gpu = state_dict[key].cuda()
                    dist.all_reduce(state_gpu, op=dist.ReduceOp.SUM)
                    state_dict[key].copy_(state_gpu.cpu())

        for state_name in self._states_to_synchronize:
            state_dict = getattr(self, state_name)
            _synchronize(state_dict)

    def normalize(self) -> None:
        """
        Normalize the state.
        """

        # normalize helper function
        def _normalize(state_dict, counter_dict):
            for key in state_dict:
                if not isinstance(state_dict[key], torch.Tensor):
                    _normalize(state_dict[key], counter_dict[key])
                else:
                    state_dict[key] /= counter_dict[key]

        for state_name, counter_name in self._states_to_normalize:
            state_dict = getattr(self, state_name)
            counter_dict = getattr(self, counter_name)
            _normalize(state_dict, counter_dict)

    def finalize(self) -> None:
        """
        Finalize the state.
        """
        if get_world_size() > 1:
            self.synchronize()
        self.normalize()

    def save_state(self, log_dir: str) -> None:
        """
        Save Hessian state to disk.
        """
        state_log_dir = os.path.join(log_dir, "state")
        if not os.path.exists(state_log_dir) and get_rank() == 0:
            os.makedirs(state_log_dir)

        for state_name in self._states_to_save:
            state_dict = getattr(self, state_name)
            torch.save(state_dict, os.path.join(state_log_dir, f"{state_name}.pt"))

        others = {
            "_states_to_synchronize": self._states_to_synchronize,
            "_states_to_save": self._states_to_save,
            "_states_to_normalize": self._states_to_normalize,
        }
        torch.save(others, os.path.join(state_log_dir, "others.pt"))

    def load_state(self, log_dir: str) -> None:
        """
        Load Hessian state from disk.
        """
        state_log_dir = os.path.join(log_dir, "state")

        others = torch.load(os.path.join(state_log_dir, "others.pt"))
        for key, value in others.items():
            setattr(self, key, value)

        for state_name in self._states_to_save:
            state_dict = torch.load(os.path.join(state_log_dir, f"{state_name}.pt"))
            setattr(self, state_name, state_dict)

    def clear_log_state(self) -> None:
        """
        Clear the log state.
        """
        self.log_state = nested_dict()

    def clear(self) -> None:
        for state_name in self._states:
            state = getattr(self, state_name)
            state.clear()
