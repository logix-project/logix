from typing import List
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class LogIXArgument:
    project: str = field(
        default="tmp_logix", metadata={"help": "The name of the project."}
    )
    config: str = field(
        default="config.yaml", metadata={"help": "The path to the config file."}
    )
    lora: bool = field(default=False, metadata={"help": "Enable LoRA."})
    hessian: str = field(default="none", metadata={"help": "Hessian type."})
    save: str = field(default="none", metadata={"help": "Save type."})
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
    input_key: str = field(
        default="input_ids", metadata={"help": "The dictionary key for 'input_ids'."}
    )
    influence_damping: float = field(
        default=None, metadata={"help": "A damping term in influence functions."}
    )
    influence_mode: str = field(
        default="dot", metadata={"help": "Influence function mode."}
    )
    label_key: str = field(
        default="labels", metadata={"help": "The dictionary key for 'labels'."}
    )
    attention_key: str = field(
        default="attention_mask",
        metadata={"help": "The dictionary key for 'attention_mask'."},
    )
    data_id: str = field(
        default="detokenize", metadata={"help": "The 'data_id' generation logic."}
    )
    ignore_idx: int = field(
        default=-100, metadata={"help": "The index to be ignored in loss computation."}
    )

    def __post_init__(self):
        self.mode = "log"
