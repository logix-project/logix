import torch.nn as nn
from transformers import Trainer


def prepare_model(trainer: Trainer):
    """
    Replicate the model preparation process in transformers.Trainer._inner_training_loop.
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1565-L1691
    """
    args = trainer.args

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        if args.gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        else:
            gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs
        # use_reentrant must be set to False for hooks to work propertly in AnaLog
        gradient_checkpointing_kwargs.update({"use_reentrant": False})

        trainer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    model = trainer._wrap_model(trainer.model_wrapped)

    # as the model is wrapped, don't use `accelerator.prepare`
    # this is for unhandled cases such as
    # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    use_accelerator_prepare = True if model is trainer.model else False

    # prepare using `accelerator` prepare
    if use_accelerator_prepare:
        trainer.model.train()
        model = trainer.accelerator.prepare(trainer.model)

    if trainer.is_fsdp_enabled:
        trainer.model = trainer.model_wrapped = model

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not trainer.model:
        trainer.model_wrapped = model

    # important: at this point:
    # self.model         is the Transformers Model
    # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    return model


def training_step(trainer: Trainer, model: nn.Module, inputs):
    """
    Replicate the training step process in transformers.Trainer._training_step.
    """
    inputs = trainer._prepare_inputs(inputs)

    model.zero_grad()

    with trainer.compute_loss_context_manager():
        loss = trainer.compute_loss(model, inputs)

    if trainer.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training

    assert not trainer.use_apex, "APEX is not supported."
    trainer.accelerator.backward(loss)
