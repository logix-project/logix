from typing import List, Optional

import torch.nn as nn
from transformers import Trainer, TrainerCallback

from analog import AnaLog, AnaLogScheduler
from analog.huggingface.utils import prepare_model, training_step


def extract_log_from_trainer(
    trainer: Trainer,
    analog_instance: AnaLog,
    scheduler: AnaLogScheduler,
    type_filter: Optional[List[nn.Module]] = None,
    name_filter: Optional[List[str]] = None,
):
    """Documentation (Hwijeen)

    Args:
        trainer (Trainer): Huggingface Trainer instantiated by the user
        analog_instance (AnaLog): AnaLog instance instantiated by the user
        scheduler (AnaLogScheduler): AnaLogScheduler instance instantiated by the user
    """
    model = prepare_model(trainer)
    model.eval()
    data_loader = trainer.get_train_dataloader()

    analog_instance.watch(model, type_filter=type_filter, name_filter=name_filter)
    epochs = scheduler if scheduler is not None else range(1)

    for _ in epochs:
        for inputs in data_loader:
            data_id = trainer.tokenizer.batch_decode(inputs["input_ids"])
            mask = inputs.get("attention_mask", None)
            with analog_instance(data_id, mask=mask):
                training_step(trainer, model, inputs)
        analog_instance.finalize()


# NOTE: WIP
class AnaLogCallback(TrainerCallback):
    def __init__(self, analog_instance: AnaLog, scheduler: AnaLogScheduler):
        self.analog_instance = analog_instance
        self.scheduler = scheduler

    def on_epoch_begin(self, args, state, control, **kwargs):
        next(iter(self.scheduler))

    def on_epoch_end(self, args, state, control, **kwargs):
        self.analog_instance.finalize()

    def on_step_begin(self, args, state, control, **kwargs):
        data_id = None # FIXME: need a way to get the data_id
        mask = None
        # TODO: add `logging.start` interface to Analog class
        self.analog_instance.binfo.clear()
        self.analog_instance.binfo.data_id = data_id
        self.analog_instance.binfo.mask = mask
        self.analog_instance.logger.clear(
            hook=True, module=True, buffer=False
        )
        self.analog_instance.logger.register_all_module_hooks()

    def on_step_end(self, args, state, control, **kwargs):
        # TODO: `logging.end`
        self.analog_instance.update()

        optimizer = kwargs["optimizer"]
        optimizer.zero_grad()  # to nullfy optim.step performed by trainer

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        self.analog_instance.watch(model) # TODO: add filters

        # TODO: turn off huggingface logger
        # https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
