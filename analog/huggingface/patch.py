from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.trainer import *

from analog import AnaLog, AnaLogScheduler
from analog.utils import DataIDGenerator
from analog.huggingface.callback import AnalogCallback
from analog.huggingface.arguments import AnaLogArguments


def patch_trainer(TrainerClass):
    """
    Patch the (variant of) Huggingface Trainer class with the AnaLog functionalities.

    Args:
        TrainerClass: The Trainer class to patch
    """

    class PatchedTrainer(TrainerClass):
        def __init__(
            self,
            analog_args: Optional[AnaLogArguments] = None,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[
                torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
            ] = (None, None),
            preprocess_logits_for_metrics: Optional[
                Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            ] = None,
        ):
            # Initialize AnaLog
            self.analog_args = analog_args
            self.analog = AnaLog(project=analog_args.project, config=analog_args.config)
            self.analog_scheduler = AnaLogScheduler(
                self.analog, ekfac=analog_args.ekfac
            )
            self.data_id_generator = DataIDGenerator()
            analog_callback = AnalogCallback(
                self.analog, self.analog_scheduler, self.analog_args
            )

            # Patch TrainingArguments
            if args is None:
                output_dir = "tmp_trainer"
                args = TrainingArguments(output_dir=output_dir)
            args.num_train_epochs = len(self.analog_scheduler)
            args.report_to = []

            super().__init__(
                model,
                args,
                data_collator,
                train_dataset,
                eval_dataset,
                tokenizer,
                model_init,
                compute_metrics,
                (
                    [analog_callback]
                    if callbacks is None
                    else [analog_callback] + callbacks
                ),
                optimizers,
                preprocess_logits_for_metrics,
            )

        def extract_log(self, *args, **kwargs):
            self.analog_args.mode = "log"
            self.train(*args, **kwargs)

        def influence(self, *args, **kwargs):
            self.analog_args.mode = "influence"
            self.train(*args, **kwargs)

        def self_influence(self, *args, **kwargs):
            self.analog_args.mode = "self_influence"
            self.train(*args, **kwargs)

        def create_optimizer_and_scheduler(self, num_training_steps: int):
            self.create_optimizer()
            optimizer = self.optimizer
            self.create_scheduler(
                num_training_steps=num_training_steps, optimizer=optimizer
            )

        def create_optimizer(self):
            class DummyOptimizer:
                def __init__(self, params):
                    pass

                def step(self):
                    pass

                def zero_grad(self):
                    pass

                def state_dict(self):
                    return dict()

            self.optimizer = DummyOptimizer(self.model.parameters())

        def create_scheduler(
            self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
        ):
            class DummyScheduler:
                def __init__(self):
                    pass

                def step(self):
                    pass

                def get_last_lr(self):
                    return [0]

                def state_dict(self):
                    return dict()

            self.lr_scheduler = DummyScheduler()
            return self.lr_scheduler

        def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
        ) -> torch.Tensor:
            model.eval()
            inputs = self._prepare_inputs(inputs)

            if self.tokenizer is not None:
                data_id = self.tokenizer.batch_decode(
                    inputs["input_ids"], skip_special_tokens=True
                )
            else:
                data_id = self.data_id_generator(inputs["input_ids"])
            mask = inputs.get("attention_mask", None)
            with self.analog(data_id=data_id, mask=mask):
                if is_sagemaker_mp_enabled():
                    loss_mb = smp_forward_backward(
                        model, inputs, self.args.gradient_accumulation_steps
                    )
                    return loss_mb.reduce_mean().detach().to(self.args.device)

                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)

                if self.args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training

                # loss reduction with sum instead of mean
                sum_scale = (inputs["labels"] != -100).sum().item()
                loss = loss * sum_scale
                if self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss)

            return loss.detach() / self.args.gradient_accumulation_steps

    return PatchedTrainer
