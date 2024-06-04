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

from logix import LogIX
from logix.statistic import CorrectedEigval, Covariance, Log


class LogIXScheduler:
    def __init__(
        self,
        logix: LogIX,
        lora: str = "none",
        hessian: str = "none",
        save: str = "none",
    ):
        self._logix = logix

        self._lora = lora
        self._hessian = hessian
        self._save = save

        self._epoch = -1
        self._logix_state_schedule = []

        self.sanity_check(lora, hessian)
        self.configure_schedule(lora, hessian, save)

        self._schedule_iterator = iter(self._logix_state_schedule)

    def sanity_check(self, lora: str, hessian: str):
        assert lora in ["none", "random", "pca"]
        assert hessian in ["none", "raw", "kfac", "ekfac"]

    def get_lora_epoch(self, lora: str) -> int:
        if lora == "random":
            return 0
        elif lora == "pca":
            return 1
        return -1

    def get_save_epoch(self, save: str) -> int:
        if save != "none":
            return len(self) - 1
        return -1

    def configure_schedule(self, lora: str, hessian: str, save: str) -> None:
        if lora == "pca":
            self._logix_state_schedule.append(
                {"forward": [Covariance], "backward": [Covariance]}
            )

        if hessian == "ekfac":
            self._logix_state_schedule.append(
                {"forward": [Covariance], "backward": [Covariance]}
            )

        last_state = {"forward": [], "backward": [], "grad": []}
        if save != "none":
            last_state[save].append(Log)
        if hessian == "kfac":
            last_state["forward"].append(Covariance)
            last_state["backward"].append(Covariance)
        elif hessian == "ekfac":
            if Log not in last_state["grad"]:
                last_state["grad"].append(Log)
            last_state["grad"].append(CorrectedEigval)
        elif hessian == "raw":
            if Log not in last_state["grad"]:
                last_state["grad"].append(Log)
            last_state["grad"].append(Covariance)
        self._logix_state_schedule.append(last_state)

    def __iter__(self):
        return self

    def __next__(self) -> int:
        logix_state = next(self._schedule_iterator)
        self._epoch += 1

        # maybe add lora
        if self._epoch == self.get_lora_epoch(self._lora):
            self._logix.add_lora()

        # maybe setup save
        if self._epoch == self.get_save_epoch(self._save):
            self._logix.save(True)

        self._logix.setup(logix_state)

        return self._epoch

    def __len__(self) -> int:
        return len(self._logix_state_schedule)
