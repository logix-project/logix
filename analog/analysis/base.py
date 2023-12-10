from abc import ABC, abstractmethod
from typing import Dict, Any

from analog.state import AnaLogState


class AnalysisBase(ABC):
    def __init__(
            self,
            config: Dict[str, Any],
            state: AnaLogState,
    ):
        self.config = config
        self._state = state

        self.parse_config()

    @abstractmethod
    def parse_config(self) -> None:
        """
        Returns a dictionary of method names and their callable functions.
        """
        pass
