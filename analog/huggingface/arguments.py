from typing import List
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class AnaLogArguments:
    project: str = field(
        default="tmp_analog", metadata={"help": "The name of the project."}
    )
    config: str = field(
        default="config.yaml", metadata={"help": "The path to the config file."}
    )
    lora: bool = field(default=False, metadata={"help": "Enable LoRA."})
    ekfac: bool = field(default=False, metadata={"help": "Enable EKFAC."})
    initialize_from_log: bool = field(
        default=False, metadata={"help": "Initialize from the log."}
    )
    name_filter: List[str] = field(
        default_factory=list, metadata={"help": "Filter for layer names."}
    )
    type_filter: List[nn.Module] = field(
        default_factory=list, metadata={"help": "Filter for layer types."}
    )
    log_batch_size: int = field(
        default=16, metadata={"help": "The batch size for log dataloader."}
    )
    log_num_workers: int = field(
        default=1, metadata={"help": "The number of workers for log dataloader."}
    )

    def __post_init__(self):
        self.mode = "log"
