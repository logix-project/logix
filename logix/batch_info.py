from typing import Iterable, Optional, Any

import torch

from logix.utils import nested_dict


class BatchInfo:
    def __init__(self):
        self.data_id: Optional[Iterable[Any]] = None
        self.mask: Optional[torch.Tensor] = None
        self.log = nested_dict()

    def clear(self):
        self.data_id = None
        self.mask = None
        self.log.clear()
