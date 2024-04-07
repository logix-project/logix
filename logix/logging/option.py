from typing import Any

from logix.statistic import Covariance, CorrectedEigval
from logix.utils import get_logger


class LogOption:
    def __init__(self):
        self._log = {}
        self._save = {}
        self._statistic = {}

        self.clear()

    def setup(self, log_option_kwargs):
        """
        Update logging configurations.

        Args:
            log: Logging configurations.
            save: Saving configurations.
            statistic: Statistic configurations.
        """
        log = log_option_kwargs.get("log", None)
        save = log_option_kwargs.get("save", None)
        statistic = log_option_kwargs.get("statistic", None)
        self.clear()

        if log is not None:
            if isinstance(log, str):
                self._log[log] = True
            elif isinstance(log, list):
                for l in log:
                    self._log[l] = True
            elif isinstance(log, dict):
                self._log = log
            else:
                raise ValueError(f"Unsupported log type: {type(log)}")

        if save is not None:
            if isinstance(save, str):
                self._save[save] = True
            elif isinstance(save, list):
                for s in save:
                    self._save[s] = True
            elif isinstance(save, dict):
                self._save = save
            else:
                raise ValueError(f"Unsupported save type: {type(save)}")

        if statistic is not None:
            if isinstance(statistic, str):
                if statistic == "kfac":
                    statistic = {
                        "forward": [Covariance],
                        "backward": [Covariance],
                        "grad": [],
                    }
                elif statistic == "ekfac":
                    statistic = {
                        "forward": [],
                        "backward": [],
                        "grad": [CorrectedEigval],
                    }
                else:
                    raise ValueError(f"Unknown statistic: {statistic}")

            assert isinstance(statistic, dict)
            self._statistic = statistic

        self._sanity_check()

    def _sanity_check(self):
        # forward
        if self._save["forward"] and not self._log["forward"]:
            get_logger().warning(
                "Saving forward activations without logging it is not allowed. "
                + "Setting log['forward'] to True automatically."
            )
            self._log["forward"] = True

        # backward
        if self._save["backward"] and not self._log["backward"]:
            get_logger().warning(
                "Saving backward error signals without logging it is not allowed. "
                + "Setting log['backward'] to True automatically."
            )
            self._log["backward"] = True

        # grad
        if (self._save["grad"] or len(self._statistic["grad"]) > 0) and not self._log[
            "grad"
        ]:
            get_logger().warning(
                "Saving gradients or computing statistic without logging it "
                + "is not allowed. Setting log['grad'] to True automatically."
            )
            self._log["grad"] = True

    def eval(self):
        """
        Enable the evaluation mode. This will turn of saving and updating
        statistic.
        """
        self.clear(log=False, save=True, statistic=True)

    def clear(self, log=True, save=True, statistic=True):
        """
        Clear all logging configurations.
        """
        if log:
            self._log = {"forward": False, "backward": False, "grad": False}
        if save:
            self._save = {"forward": False, "backward": False, "grad": False}
        if statistic:
            self._statistic = {"forward": [], "backward": [], "grad": []}

    @property
    def log(self):
        return self._log

    @property
    def save(self):
        return self._save

    @property
    def statistic(self):
        return self._statistic
