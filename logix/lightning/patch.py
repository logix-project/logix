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

from logix import LogIX, LogIXScheduler
from logix.lightning.arguments import LogIXArguments
from logix.lightning.callback import LogIXCallback


def patch(
    module_cls,
    trainer_cls,
    logix_args: LogIXArguments,
    data_id_extractor: Optional[Callable] = None,
    mask_extractor: Optional[Callable] = None,
) -> LogIXCallback:
    run = LogIX(project=logix_args.project, config=logix_args.config)
    logix_scheduler = LogIXScheduler(
        run, hessian=logix_args.hessian, save=logix_args.save
    )

    logix_callback = LogIXCallback(
        logix=run,
        logix_scheduler=logix_scheduler,
        logix_args=logix_args,
        data_id_extractor=data_id_extractor,
        mask_extractor=mask_extractor,
    )

    class PatchedModule(module_cls):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            self.logix = run
            self.logix_args = logix_args

            self.logix.watch(
                self._get_model(),
                name_filter=logix_args.name_filter,
                type_filter=logix_args.type_filter,
            )

            if self.logix_args.lora:
                self.logix.add_lora()

            if self.logix_args.initialize_from_log:
                run.initialize_from_log()

        def _get_model(self):
            return getattr(self, self.logix_args.model_key)

    class PatchedTrainer(trainer_cls):
        def __init__(
            self,
            accelerator="auto",
            strategy="auto",
            devices="auto",
            num_nodes=1,
            precision=None,
            logger=None,
            callbacks=None,
            fast_dev_run=False,
            max_epochs=None,
            min_epochs=None,
            max_steps=-1,
            min_steps=None,
            max_time=None,
            limit_train_batches=None,
            limit_val_batches=None,
            limit_test_batches=None,
            limit_predict_batches=None,
            overfit_batches=0.0,
            val_check_interval=None,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=None,
            log_every_n_steps=None,
            enable_checkpointing=None,
            enable_progress_bar=None,
            enable_model_summary=None,
            accumulate_grad_batches=1,
            gradient_clip_val=None,
            gradient_clip_algorithm=None,
            deterministic=None,
            benchmark=None,
            inference_mode=True,
            use_distributed_sampler=True,
            profiler=None,
            detect_anomaly=False,
            barebones=False,
            plugins=None,
            sync_batchnorm=False,
            reload_dataloaders_every_n_epochs=0,
            default_root_dir=None,
        ) -> None:
            # add logix callback
            if callbacks is None:
                callbacks = [logix_callback]
            else:
                callbacks = callbacks + [logix_callback]

            # disable validation
            check_val_every_n_epoch = 1_000_000_000

            # disable checkpointing
            enable_checkpointing = False

            # disable logging
            logger = False

            super().__init__(
                accelerator,
                strategy,
                devices,
                num_nodes,
                precision,
                logger,
                callbacks,
                fast_dev_run,
                max_epochs,
                min_epochs,
                max_steps,
                min_steps,
                max_time,
                limit_train_batches,
                limit_val_batches,
                limit_test_batches,
                limit_predict_batches,
                overfit_batches,
                val_check_interval,
                check_val_every_n_epoch,
                num_sanity_val_steps,
                log_every_n_steps,
                enable_checkpointing,
                enable_progress_bar,
                enable_model_summary,
                accumulate_grad_batches,
                gradient_clip_val,
                gradient_clip_algorithm,
                deterministic,
                benchmark,
                inference_mode,
                use_distributed_sampler,
                profiler,
                detect_anomaly,
                barebones,
                plugins,
                sync_batchnorm,
                reload_dataloaders_every_n_epochs,
                default_root_dir,
            )

            self.logix = run
            self.logix_args = logix_args

        def extract_log(self, *args, **kwargs):
            self.logix_args.mode = "log"
            self.fit(*args, **kwargs)

        def influence(self, *args, **kwargs):
            self.logix_args.mode = "influence"
            self.fit(*args, **kwargs)

            return self.logix.influence.get_influence_scores()

        def self_influence(self, *args, **kwargs):
            self.logix_args.mode = "self_influence"
            self.fit(*args, **kwargs)

            return

    return PatchedModule, PatchedTrainer
