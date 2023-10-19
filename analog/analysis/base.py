from abc import ABC, abstractmethod
from typing import Dict

from analog.storage import StorageHandlerBase
from analog.hessian import HessianHandlerBase


class AnalysisBase(ABC):
    def __init__(
        self,
        config: Dict,
        storage_handler: StorageHandlerBase,
        hessian_handler: HessianHandlerBase,
    ):
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
