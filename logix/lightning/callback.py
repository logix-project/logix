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

from typing import Callable, Optional

from lightning.pytorch.callbacks import Callback

from logix import LogIX, LogIXScheduler
from logix.lightning.arguments import LogIXArguments
from logix.utils import DataIDGenerator


class LogIXCallback(Callback):
    def __init__(
        self,
        logix: LogIX,
        logix_scheduler: LogIXScheduler,
        logix_args: LogIXArguments,
        data_id_extractor: Optional[Callable] = None,
        mask_extractor: Optional[Callable] = None,
    ) -> None:
        self.logix = logix
        self.scheduler = logix_scheduler
        self.args = logix_args

        self.data_id_extractor = data_id_extractor or DataIDGenerator()
        self.mask_extractor = mask_extractor

        self.counter = 0  # maybe use for gradient accumulation later

    def on_train_epoch_start(self, trainer, pl_module):
        if self.args.mode == "log":
            next(self.scheduler)
        else:
            self.logix.setup({"grad": ["log"]})
            self.logix.eval()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.args.mode == "log":
            self.logix.finalize()

        max_epochs = len(self.scheduler) if self.args.mode == "log" else 1
        if trainer.current_epoch >= max_epochs - 1:
            trainer.should_stop = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        data_id = self.data_id_extractor(batch)
        mask = None if self.mask_extractor is None else self.mask_extractor(batch)

        self.logix.start(data_id=data_id, mask=mask)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.logix.end()

        if self.args.mode == "influence":
            batch_log = self.logix.get_log()
            log_dataloader = self.logix.build_log_dataloader(
                num_workers=self.args.log_num_workers,
                batch_size=self.args.log_batch_size,
            )
            self.logix.influence.compute_influence_all(
                batch_log,
                log_dataloader,
                mode=self.args.influence_mode,
                damping=self.args.influence_damping,
                save=True,
            )
        elif self.args.mode == "self_influence":
            batch_log = self.logix.get_log()
            log_dataloader = self.logix.build_log_dataloader(
                num_workers=self.args.log_num_workers,
                batch_size=self.args.log_batch_size,
            )
            self.logix.influence.compute_self_influence(
                batch_log, damping=self.args.influence_damping
            )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        optimizer.zero_grad()
