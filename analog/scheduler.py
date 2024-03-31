from analog import AnaLog
from analog.statistic import Covariance
from analog.utils import get_logger


class AnaLogScheduler:
    def __init__(
        self,
        analog: AnaLog,
        lora: str = "none",
        hessian: str = "none",
        save: str = "none",
    ):
        self.analog = analog

        self._epoch = -1
        self._lora_epoch = -1
        self._analog_state_schedule = []

        self.sanity_check()
        self.configure_lora_epoch(lora)
        self.configure_schedule(ekfac, lora, sample)

        self._schedule_iterator = iter(self._analog_state_schedule)

    def sanity_check(self, lora: str, hessian: str, save: str):
        assert lora in ["none", "random", "pca"]
        assert hessian in ["none", "raw", "kfac", "ekfac"]
        assert save in ["none", "grad"]

    def configure_lora_epoch(self, lora: str):
        if lora == "random":
            self._lora_epoch = 0
        elif lora == "pca":
            self._lora_epoch = 1

    def configure_schedule(self, lora: str, hessian: str, save: str):
        # (log, hessian, save) for analog
        if lora == "pca":
            self._analog_state_schedule.append({"statistic": "kfac"})

        if hessian == "ekfac":
            self._analog_state_schedule.append({"statistic": "kfac"})

        last_state = {}
        # log
        if save in ["grad"] or hessian in ["raw", "ekfac"]:
            last_state["log"] = "grad"
        # statistic
        if hessian in ["kfac", "ekfac"]:
            last_state["statistic"] = hessian
        elif hessian in ["raw"]:
            last_state["statistic"] = {
                "grad": [Covariance],
                "forward": [],
                "backward": [],
            }
        # save
        if save in ["grad"]:
            last_state["save"] = save
        self._analog_state_schedule.append(last_state)

    def __iter__(self):
        return self

    def __next__(self):
        analog_state = next(self._schedule_iterator)
        self._epoch += 1
        if self._epoch == self._lora_epoch:
            self.analog.add_lora()
        self.analog.setup(analog_state)
        return self._epoch

    def __len__(self):
        return len(self.analog_state_schedule)
