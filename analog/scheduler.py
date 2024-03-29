from analog import AnaLog
from analog.utils import get_logger


class AnaLogScheduler:
    def __init__(
        self,
        analog: AnaLog,
        ekfac: bool = False,
        lora: bool = False,
        sample: bool = False,
    ):
        self.analog = analog

        self._epoch = -1
        self.analog_state_schedule = []
        self.execution_schedule = {"ekfac": -1, "lora": -1}

        self.generate_schedule(ekfac, lora, sample)
        self.schedule_iterator = iter(self.analog_state_schedule)

    def generate_schedule(
        self, ekfac: bool = False, lora: bool = False, sample: bool = False
    ):
        lora = self.analog.config.logging_options["use_lora"]
        ekfac = self.analog.config.logging_options["ekfac"]
        sample = self.analog.config.logging_options["sample"]
        use_full_covariance = self.analog.config.logging_options["use_full_covariance"]
        to_save = "grad" if self.analog.config.other_options["save_gradient"] else None

        if lora:
            self.execution_schedule["lora"] = 1

        if use_full_covariance:
            self.analog_state_schedule = [
                {"statistic": "kfac"},
                {"log": "grad", "save": to_save, "statistic": "full_covariance"},
            ]
            return

        # (log, hessian, save) for analog
        if ekfac and lora and sample:
            self.analog_state_schedule = [
                {"statistic": "kfac"},
                {"statistic": "kfac"},
                {"log": "grad", "statistic": "ekfac"},
                {"log": "grad", "save": to_save},
            ]
        elif ekfac and lora and not sample:
            self.analog_state_schedule = [
                {"statistic": "kfac"},
                {"statistic": "kfac"},
                {"log": "grad", "save": to_save, "statistic": "ekfac"},
            ]
        elif ekfac and not lora and sample:
            self.analog_state_schedule = [
                {"statistic": "kfac"},
                {"log": "grad", "statistic": "ekfac"},
                {"log": "grad", "save": to_save},
            ]
        elif ekfac and not lora and not sample:
            self.analog_state_schedule = [
                {"statistic": "kfac"},
                {"log": "grad", "save": to_save, "statistic": "ekfac"},
            ]
        elif not ekfac and lora and sample:
            self.analog_state_schedule = [
                {"statistic": "kfac"},
                {"statistic": "kfac"},
                {"log": "grad", "save": to_save},
            ]
        elif not ekfac and lora and not sample:
            self.analog_state_schedule = [
                {"statistic": "kfac"},
                {"log": "grad", "save": to_save, "statistic": "kfac"},
            ]
        elif not ekfac and not lora and sample:
            self.analog_state_schedule = [
                {"statistic": "kfac"},
                {"log": "grad", "save": to_save},
            ]
        elif not ekfac and not lora and not sample:
            self.analog_state_schedule = [
                {"log": "grad", "save": to_save, "statistic": "kfac"},
            ]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            analog_state = next(self.schedule_iterator)
            self._epoch += 1
            self.analog.setup(analog_state)
            if self._epoch == self.execution_schedule["lora"]:
                self.analog.add_lora()
            return self._epoch
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.analog_state_schedule)
