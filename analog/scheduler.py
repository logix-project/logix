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
        if lora:
            self.execution_schedule["lora"] = 1
        if ekfac:
            self.execution_schedule["ekfac"] = 1 + int(lora)

        # (log, hessian, save) for analog
        if ekfac and lora and sample:
            self.analog_state_schedule = [
                {
                    "log": [],
                    "hessian": True,
                    "save": False,
                },  # collect covariances using sampling
                {
                    "log": ["grad"],
                    "hessian": True,
                    "save": False,
                },  # fit lambda for ekfac using sampling
                {
                    "log": ["grad"],
                    "hessian": False,
                    "save": True,
                },  # collect LoRA gradients
            ]
        elif ekfac and lora and not sample:
            self.analog_state_schedule = [
                {"log": [], "hessian": True, "save": False},  # collect covariances
                {
                    "log": ["grad"],
                    "hessian": True,
                    "save": True,
                },  # fit lambda for ekfac while collectng LoRA gradient
            ]
        elif ekfac and not lora and sample:
            self.analog_state_schedule = [
                {
                    "log": [],
                    "hessian": True,
                    "save": False,
                },  # collect covariances using sampling
                {
                    "log": ["grad"],
                    "hessian": True,
                    "save": False,
                },  # fit lambda for ekfac using sampling
                {"log": ["grad"], "hessian": False, "save": True},  # collect gradients
            ]
        elif ekfac and not lora and not sample:
            self.analog_state_schedule = [
                {"log": [], "hessian": True, "save": False},  # collect covariances
                {
                    "log": ["grad"],
                    "hessian": True,
                    "save": True,
                },  # fit lambda for ekfac while collecting gradients
            ]
        elif not ekfac and lora and sample:
            self.analog_state_schedule = [
                {
                    "log": [],
                    "hessian": True,
                    "save": False,
                },  # collect covariances using sampling
                {
                    "log": ["grad"],
                    "hessian": False,
                    "save": True,
                },  # collect LoRA gradients
            ]
        elif not ekfac and lora and not sample:
            self.analog_state_schedule = [
                {"log": [], "hessian": True, "save": False},  # collect covariances
                {
                    "log": ["grad"],
                    "hessian": False,
                    "save": True,
                },  # collect LoRA gradients
            ]
        elif not ekfac and not lora and sample:
            self.analog_state_schedule = [
                {
                    "log": [],
                    "hessian": True,
                    "save": False,
                },  # collect covariances using sampling
                {"log": ["grad"], "hessian": False, "save": True},  # collect gradients
            ]
        elif not ekfac and not lora and not sample:
            self.analog_state_schedule = [
                {"log": [], "hessian": True, "save": False},  # collect covariances
                {"log": ["grad"], "hessian": True, "save": True},  # collect gradients
            ]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            analog_state = next(self.schedule_iterator)
            self._epoch += 1
            self.analog.update(analog_state)
            if self._epoch == self.execution_schedule["ekfac"]:
                self.analog.ekfac()
            if self._epoch == self.execution_schedule["lora"]:
                self.analog.add_lora()
            return self._epoch
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.analog_state_schedule)
