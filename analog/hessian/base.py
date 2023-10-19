from abc import ABC, abstractmethod

from analog.utils import nested_dict


class HessianHandlerBase(ABC):
    def __init__(self, config):
        self.config = config
        self.hessian_state = nested_dict()
        self.sample_counter = nested_dict()

        self.hessian_inverse_with_override = False

        self.parse_config()

    @abstractmethod
    def parse_config(self):
        """
        Parse the configuration parameters.
        """
        pass

    @abstractmethod
    def update_hessian(self, module, module_name, mode, data, mask):
        """
        Compute the covariance for given data.
        """
        pass

    @abstractmethod
    def finalize(self):
        """
        Finalize the covariance computation.
        """
        pass

    def get_hessian_state(self, name: str = None):
        """
        Get the Hessian state.
        """
        if name is None:
            return self.hessian_state
        assert name in self.hessian_state
        return self.hessian_state[name]

    def get_sample_size(self, data, mask) -> int:
        """
        Get the sample size for the given data.
        """
        if mask is not None:
            return mask.sum().item()
        return data.size(0)

    def clear(self) -> None:
        """
        Clear the Hessian state.
        """
        self.hessian_state.clear()
        self.sample_counter.clear()
