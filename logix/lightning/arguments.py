# Copyright 2023-present the LogIX team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List

import torch.nn as nn


@dataclass
class LogIXArguments:
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
    influence_damping: float = field(
        default=None, metadata={"help": "A damping term in influence functions."}
    )
    influence_mode: str = field(
        default="dot", metadata={"help": "Influence function mode."}
    )
    model_key: str = field(
        default="model", metadata={"help": "The dictionary key for 'model'."}
    )

    def __post_init__(self):
        self.mode = "log"
