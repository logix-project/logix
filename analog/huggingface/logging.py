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


class AnaLogCallback(TrainerCallback):
    def on_train_end(
        self, args, state, control, model=None, train_dataloader=None, **kwargs
    ):
        raise NotImplementedError
