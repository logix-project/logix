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
                ([], True, False),
                ([], True, False),
                (["grad"], True, False),
                (["grad"], False, True),
            ]
        elif ekfac and lora and not sample:
            self.analog_state_schedule = [
                ([], True, False),
                ([], True, False),
                (["grad"], True, True),
            ]
        elif ekfac and not lora and sample:
            self.analog_state_schedule = [
                ([], True, False),
                (["grad"], True, False),
                (["grad"], False, True),
            ]
        elif ekfac and not lora and not sample:
            self.analog_state_schedule = [
                ([], True, False),
                (["grad"], True, True),
            ]
        elif not ekfac and lora and sample:
            self.analog_state_schedule = [
                ([], True, False),
                ([], True, False),
                ([grad], False, True),
            ]
        elif not ekfac and lora and not sample:
            self.analog_state_schedule = [
                ([], True, False),
                (["grad"], True, True),
            ]
        elif not ekfac and not lora and sample:
            self.analog_state_schedule = [
                ([], True, False),
                (["grad"], False, True),
            ]
        elif not ekfac and not lora and not sample:
            self.analog_state_schedule = [
                (["grad"], True, True),
            ]

    def __iter__(self):
        return self

    def __next__(self):
        self._epoch += 1
        if self._epoch < len(self.analog_state_schedule):
            self.analog.set_default_state(*self.analog_state_schedule[self._epoch])
            if self._epoch == self.execution_schedule["ekfac"]:
                self.analog.ekfac()
            if self._epoch == self.execution_schedule["lora"]:
                self.analog.add_lora()
            return self._epoch
        raise StopIteration

    def __len__(self):
        return len(self.analog_state_schedule)
