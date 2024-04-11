import logging
import os

import torch
import torch.distributed as dist

from logix.utils import nested_dict, get_world_size, get_rank


class LogIXState:
    """
    LogIXState stores all these relevant log states that are used for
    communication between different handlers. All states in LogIXState
    follow the nested dictionary data structure.
    """

    def __init__(self) -> None:
        self._states = set()
        self._states_to_synchronize = []
        self._states_to_save = []
        self._states_to_normalize = []
        self._states_to_not_clear = set()

        self.register_state("mean_state", synchronize=True, save=True)
        self.register_state("mean_counter", synchronize=True, save=False)
        self.register_state("covariance_state", synchronize=True, save=True)
        self.register_state("covariance_counter", synchronize=True, save=False)
        self.register_state(
            "model_module", synchronize=False, save=True, not_clear=True
        )

        self.register_normalize_pair("mean_state", "mean_counter")
        self.register_normalize_pair("covariance_state", "covariance_counter")

    def register_state(
        self,
        state_name: str,
        synchronize: bool = False,
        save: bool = False,
        not_clear: bool = False,
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
            self._states_to_synchronize.append(state_name)
        if save:
            self._states_to_save.append(state_name)
        if not_clear:
            self._states_to_not_clear.add(state_name)

    def register_normalize_pair(self, state_name: str, counter_name: str):
        """
        Add a normalization pair to the state.
        """
        if (state_name, counter_name) in self._states_to_normalize:
            return

        self._states_to_normalize.append((state_name, counter_name))

    @torch.no_grad()
    def covariance_svd(self) -> None:
        """
        Compute the SVD of the covariance.
        """
        self.register_state("covariance_eigval_state", save=True)
        self.register_state("covariance_eigvec_state", save=True)

        for module_name, module_state in self.covariance_state.items():
            for mode, covariance in module_state.items():
                dtype = covariance.dtype
                eigvals, eigvecs = torch.linalg.eigh(covariance.double())
                self.covariance_eigval_state[module_name][mode] = eigvals.to(
                    dtype=dtype
                )
                self.covariance_eigvec_state[module_name][mode] = eigvecs.to(
                    dtype=dtype
                )

    @torch.no_grad()
    def covariance_inverse(self, set_attr: bool = False) -> None:
        """
        Compute the inverse of the covariance.
        """
        self.register_state("covariance_inverse_state", save=True)

        for module_name, module_state in self.covariance_state.items():
            for mode, covariance in module_state.items():
                self.covariance_inverse_state[module_name][mode] = torch.inverse(
                    covariance
                    + 0.1
                    * torch.trace(covariance)
                    * torch.eye(covariance.shape[0]).to(device=covariance.device)
                    / covariance.shape[0]
                )

    def get_covariance_state(self):
        """
        Return the covariance state.
        """
        return self.covariance_state

    def get_covariance_inverse_state(self):
        """
        Return the covariance inverse state. If the state is not computed, compute
        it first.
        """
        if not hasattr(self, "covariance_inverse_state"):
            self.covariance_inverse()
        return self.covariance_inverse_state

    def get_covariance_svd_state(self):
        """
        Return the covariance SVD state. If the state is not computed, compute
        it first.
        """
        if not hasattr(self, "covariance_eigval_state") or not hasattr(
            self, "covariance_eigvec_state"
        ):
            self.covariance_svd()
        if hasattr(self, "ekfac_eigval_state"):
            return self.ekfac_eigval_state, self.covariance_eigvec_state
        return self.covariance_eigval_state, self.covariance_eigvec_state

    def synchronize(self) -> None:
        """
        Synchronize all synchronizable states across processes.
        """

        # synchronize helper function
        def _synchronize(state_dict):
            for key in state_dict:
                if isinstance(state_dict[key], dict):
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

                    torch.cuda.synchronize()

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
        Save state to disk.
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
        Load state from disk.
        """
        state_log_dir = os.path.join(log_dir, "state")

        others = torch.load(os.path.join(state_log_dir, "others.pt"))
        for key, value in others.items():
            setattr(self, key, value)

        for state_name in self._states_to_save:
            state_dict = torch.load(os.path.join(state_log_dir, f"{state_name}.pt"))
            setattr(self, state_name, state_dict)

    def set_state(self, state_name, **kwargs):
        """
        set_state sets the state for the given state_name with input kwargs.
        """
        if state_name not in self._states:
            raise ValueError("state name {} not in registered state".format(state_name))
        for key, value in kwargs.items():
            state = getattr(self, state_name)
            state[key] = value

    def get_state(self, state_name):
        return getattr(self, state_name)

    def clear_log_state(self) -> None:
        """
        Clear the log state.
        """
        # self.log_state = nested_dict()
        self.log_state.clear()

    def clear(self) -> None:
        for state_name in self._states:
            if state_name in self._states_to_not_clear:
                continue
            state = getattr(self, state_name)
            state.clear()
