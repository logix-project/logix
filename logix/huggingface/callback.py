from typing import Optional

from transformers.trainer import TrainerCallback

from logix import LogIX, LogIXScheduler
from logix.utils import merge_logs
from logix.huggingface.arguments import LogIXArguments


class LogIXCallback(TrainerCallback):
    def __init__(
        self,
        logix: LogIX,
        logix_scheduler: LogIXScheduler,
        args: LogIXArguments,
    ):
        self.logix = logix
        self.logix_scheduler = iter(logix_scheduler)
        self.args = args

        self.accumulated_log = []

        self._log_dataloader = None

    def on_init_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        self.logix.watch(
            model=model,
            name_filter=self.args.name_filter,
            type_filter=self.args.type_filter,
        )

        if self.args.lora:
            self.logix.add_lora()

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.args.mode == "log":
            next(self.logix_scheduler)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.args.mode == "log":
            self.logix.finalize()

    def on_train_begin(self, args, state, control, **kwargs):
        if self.args.initialize_from_log:
            self.logix.initialize_from_log()

        if self.args.mode in ["influence", "self_influence"]:
            self.logix.setup({"grad": ["log"]})
            self.logix.eval()

            state.epoch = 0
            state.num_train_epochs = 1

    def on_step_end(self, args, state, control, **kwargs):
        if self.args.mode == "influence":
            self.accumulated_log.append(self.logix.get_log(copy=True))
            accumulated_log = merge_logs(self.accumulated_log)

            self.logix.influence.compute_influence_all(
                accumulated_log,
                self.log_dataloader(),
                mode=self.args.influence_mode,
                damping=self.args.influence_damping,
                save=True,
            )

            self.accumulated_log = []
        elif self.args.mode == "self_influence":
            self.accumulated_log.append(self.logix.get_log(copy=True))
            accumulated_log = merge_logs(self.accumulated_log)

            self.logix.influence.compute_self_influence(
                accumulated_log, damping=self.args.influence_damping
            )

            self.accumulated_log = []

    def on_substep_end(self, args, state, control, **kwargs):
        if self.args.mode in ["influence", "self_influence"]:
            self.accumulated_log.append(self.logix.get_log(copy=True))

    def log_dataloader(self):
        if self._log_dataloader is None:
            self._log_dataloader = self.logix.build_log_dataloader(
                num_workers=self.args.log_num_workers,
                batch_size=self.args.log_batch_size,
            )
        return self._log_dataloader
