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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers.trainer import *

from logix import LogIX, LogIXScheduler
from logix.huggingface.arguments import LogIXArguments
from logix.huggingface.callback import LogIXCallback
from logix.utils import DataIDGenerator


def patch_trainer(TrainerClass):
    """
    Patch the (variant of) Huggingface Trainer class with the LogIX functionalities.

    Args:
        TrainerClass: The Trainer class to patch
    """

    class PatchedTrainer(TrainerClass):
        def __init__(
            self,
            logix_args: LogIXArguments,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: Optional[TrainingArguments] = None,
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
            # Initialize LogIX
            self.logix_args = logix_args
            self.logix = LogIX(project=logix_args.project, config=logix_args.config)
            self.logix_scheduler = LogIXScheduler(
                self.logix, hessian=logix_args.hessian, save=logix_args.save
            )
            self.data_id_generator = DataIDGenerator()
            logix_callback = LogIXCallback(
                self.logix, self.logix_scheduler, self.logix_args
            )

            self.logix_input_key = logix_args.input_key
            self.logix_label_key = logix_args.label_key
            self.logix_attention_key = logix_args.attention_key
            self.logix_ignore_idx = logix_args.ignore_idx
            self.data_id_logic = logix_args.data_id

            # Patch TrainingArguments
            if args is None:
                output_dir = "tmp_trainer"
                args = TrainingArguments(output_dir=output_dir)
            args.num_train_epochs = len(self.logix_scheduler)
            args.report_to = []
            args.save_strategy = "no"

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
                    [logix_callback]
                    if callbacks is None
                    else [logix_callback] + callbacks
                ),
                optimizers,
                preprocess_logits_for_metrics,
            )

        def extract_log(self, *args, **kwargs):
            self.logix_args.mode = "log"
            self.train(*args, **kwargs)

        def influence(self, *args, **kwargs):
            self.logix_args.mode = "influence"
            self.train(*args, **kwargs)

            return self.logix.influence.get_influence_scores()

        def self_influence(self, *args, **kwargs):
            self.logix_args.mode = "self_influence"
            self.train(*args, **kwargs)

            return

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

            data_id = self.get_data_id(inputs)
            mask = inputs.get(self.logix_attention_key, None)
            sum_scale = self.get_sum_scale(inputs)

            with self.logix(data_id=data_id, mask=mask):
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
                loss = loss * sum_scale
                if self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss)

            return loss.detach() / self.args.gradient_accumulation_steps

        def get_data_id(self, inputs):
            ipt = inputs.get(self.logix_input_key)
            if self.data_id_logic == "detokenize":
                assert self.tokenizer is not None
                return self.tokenizer.batch_decode(ipt, skip_special_tokens=True)
            elif self.data_id_logic == "hash":
                return self.data_id_generator(ipt)

        def get_sum_scale(self, inputs):
            labels = inputs.get(self.logix_label_key)
            return (labels != self.logix_ignore_idx).sum().item()

    return PatchedTrainer
