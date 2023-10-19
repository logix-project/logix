from abc import ABC, abstractmethod


class AnalysisBase(ABC):
    def __init__(self, config, storage_handler, hessian_handler):
        self.storage_handler = storage_handler
        self.hessian_handler = hessian_handler

        self.config = config
        self.parse_config()

    @abstractmethod
    def parse_config(self) -> None:
        """
        Returns a dictionary of method names and their callable functions.
        """
        pass
