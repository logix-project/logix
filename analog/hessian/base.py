from abc import ABC, abstractmethod

from analog.utils import nested_dict


class HessianHandlerBase(ABC):
    def __init__(self, config):
        self.config = config
        self.hessian_state = nested_dict()
        self.sample_counter = nested_dict()

        self.parse_config()

    @abstractmethod
    def parse_config(self):
        """
        Parse the configuration parameters.
        """
        pass

    @abstractmethod
    def update_hessian(self, module, module_name, mode, data):
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

    def get_hessian_state(self):
        """
        Get the Hessian state.
        """
        return self.hessian_state
