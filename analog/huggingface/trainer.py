from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.trainer import *
from analog.utils import DataIDGenerator
from analog.huggingface.callback import AnalogCallback

from analog import AnaLog, AnaLogScheduler


class AnaLogTrainer(Trainer):
    def __init__(
        self,
        run: AnaLog,
        scheduler: AnaLogScheduler,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            args = TrainingArguments(output_dir=output_dir)
        args.num_train_epochs = len(scheduler)
        args.report_to = []

        self.run = run
        self.scheduler = scheduler
        self.data_id_generator = DataIDGenerator()
        analog_callback = AnalogCallback(run, scheduler)
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            [analog_callback] if callbacks is None else [analog_callback] + callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        optimizer = self.optimizer
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=optimizer
        )

    def create_optimizer(self):
        class DummyOptimizer:
            def __init__(self):
                pass

            def step(self):
                pass

            def zero_grad(self):
                pass

        self.optimizer = DummyOptimizer()
        return self.optimizer

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

        self.lr_scheduler = DummyScheduler()
        return self.lr_scheduler

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.tokenizer is not None:
            data_id = self.tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
        else:
            data_id = self.data_id_generator(inputs["input_ids"])
        mask = inputs.get("attention_mask", None)
        with self.run(data_id=data_id, mask=mask):
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(
                    model, inputs, self.args.gradient_accumulation_steps
                )
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss = (
                loss * inputs["labels"].numel()
            )  # loss reduction with mean instead of sum
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
