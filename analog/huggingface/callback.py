from typing import Optional

from transformers.trainer import TrainerCallback

from analog import AnaLog, AnaLogScheduler


class AnalogCallback(TrainerCallback):
    def __init__(self, run: AnaLog, scheduler: Optional[AnaLogScheduler] = None):
        self.run = run
        self.scheduler = iter(scheduler) if scheduler is not None else None

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.scheduler is not None:
            next(self.scheduler)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.run.finalize()

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        self.run.watch(model)
