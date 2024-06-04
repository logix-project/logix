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

from typing import Any, List

from logix.statistic import CorrectedEigval, Covariance, Log, Mean, Variance
from logix.utils import get_logger

_PLUGIN_MAPPING = {
    "log": Log,
    "mean": Mean,
    "variance": Variance,
    "covariance": Covariance,
    "corrected_eigval": CorrectedEigval,
}
_PLUGIN_LIST = [Log, Mean, Variance, Covariance, CorrectedEigval]


def _reorder_plugins(plugins):
    """
    Reorder the plugins to ensure that the plugins are in the correct order. Especially,
    it is important to ensure that the Log plugin is the first plugin.
    Args:
        plugins: List of plugins.
    Returns:
        List of plugins in the correct order.
    """
    order = [Log, Mean, Variance, Covariance, CorrectedEigval]
    ordered_plugins = []
    for plugin in order:
        if plugin in plugins:
            ordered_plugins.append(plugin)
    return ordered_plugins


def _to_plugins(plugins: List[Any], is_grad: bool = False):
    """
    Convert and reorder the list of plugins to the actual plugins.
    """
    # Convert the string plugins to the actual plugins.
    for idx, plugin in enumerate(plugins):
        if isinstance(plugin, str):
            assert plugin in _PLUGIN_MAPPING
            plugins[idx] = _PLUGIN_MAPPING[plugin]
        assert plugins[idx] in _PLUGIN_LIST

    # reorder the plugins to ensure that the plugins are in the correct order.
    plugins = _reorder_plugins(plugins)

    if is_grad:
        # Ensure that the Log plugin is the first plugin.
        if len(plugins) > 0 and Log not in plugins:
            get_logger().warning(
                "The `Log` plugin is not in the list of plugins. "
                "The `Log` plugin will be inserted at the beginning of the list."
            )
            plugins.insert(0, Log)

    return plugins


class LogOption:
    def __init__(self):
        self.forward = []
        self.backward = []
        self.grad = []

        self.clear()

    def setup(self, log_option_kwargs):
        """
        Update logging configurations.

        Args:
            log: Logging configurations.
            save: Saving configurations.
            statistic: Statistic configurations.
        """
        self.clear()

        forward = log_option_kwargs.get("forward", [])
        backward = log_option_kwargs.get("backward", [])
        grad = log_option_kwargs.get("grad", [])

        self.forward = _to_plugins(forward)
        self.backward = _to_plugins(backward)
        self.grad = _to_plugins(grad)

    def clear(self):
        """
        Clear all logging configurations.
        """
        self.forward = []
        self.backward = []
        self.grad = []
