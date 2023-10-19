from abc import ABC, abstractmethod


class AnalysisBase(ABC):
    def __init__(self, analog):
        self.analog = analog

        self.storage_handler = analog.storage_handler
        self.hessian_handler = analog.hessian_handler

        self.config = analog.config.get_analysis_config()
        self.parse_config()

    @abstractmethod
    def parse_config(self) -> None:
        """
        Returns a dictionary of method names and their callable functions.
        """
        pass
