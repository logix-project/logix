from typing import Optional

from transformers.trainer import TrainerCallback

from logix import LogiX, LogiXScheduler
from logix.huggingface.arguments import AnaLogArguments


class AnalogCallback(TrainerCallback):
    def __init__(
        self,
        analog: LogiX,
        analog_scheduler: LogiXScheduler,
        args: AnaLogArguments,
    ):
        self.analog = analog
        self.analog_scheduler = iter(analog_scheduler)
        self.args = args

        self._log_dataloader = None

    def on_init_end(self, args, state, control, **kwargs):
        if self.args.lora:
            model = kwargs["model"]
            self.analog.add_lora(model, watch=False)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.args.mode == "log":
            next(self.analog_scheduler)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.args.mode == "log":
            self.analog.finalize()

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        self.analog.watch(
            model=model,
            name_filter=self.args.name_filter,
            type_filter=self.args.type_filter,
        )

        if self.args.initialize_from_log:
            self.analog.initialize_from_log()

        if self.args.mode in ["influence", "self_influence"]:
            self.analog.setup({"log": "grad"})
            self.analog.eval()

            state.epoch = 0
            state.num_train_epochs = 1

    def on_step_end(self, args, state, control, **kwargs):
        if self.args.mode == "influence":
            test_log = self.analog.get_log()
            self.analog.compute_influence_all(test_log, self.log_dataloader())
        elif self.args.mode == "self_influence":
            test_log = self.analog.get_log()
            self.analog.compute_self_influence(test_log)

    def on_substep_end(self, args, state, control, **kwargs):
        if self.args.mode == "influence":
            test_log = self.analog.get_log()
            self.analog.compute_influence_all(test_log, self.log_dataloader())
        elif self.args.mode == "self_influence":
            test_log = self.analog.get_log()
            self.analog.compute_self_influence(test_log)

    def log_dataloader(self):
        if self._log_dataloader is None:
            self._log_dataloader = self.analog.build_log_dataloader(
                num_workers=self.args.log_num_workers,
                batch_size=self.args.log_batch_size,
            )
        return self._log_dataloader
