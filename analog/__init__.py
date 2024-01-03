from analog.analog import AnaLog
from analog.scheduler import AnaLogScheduler
from analog.utils import get_logger
from analog.huggingface import AnaLogTrainer


__version__ = "0.1.0"

_ANALOG_INSTANCE = None


def init(project: str, config: str = "./config.yaml"):
    """Initialize AnaLog.

    Args:
        project (str): The name of the project.
        config (dict): The configuration dictionary.
    """
    global _ANALOG_INSTANCE

    if _ANALOG_INSTANCE is not None:
        get_logger().warning(
            "AnaLog is already initialized. If you want to initialize "
            + "additional AnaLog instances, please use analog.AnaLog instead."
        )
        return

    run = AnaLog(project, config)

    _ANALOG_INSTANCE = run

    return run


def add_analysis(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.add_analysis(*args, **kwargs)


def add_lora(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.add_lora(*args, **kwargs)


def build_log_dataloader(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.build_log_dataloader(*args, **kwargs)


def clear(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.clear(*args, **kwargs)


def eval(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.eval(*args, **kwargs)


def finalize(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.finalize(*args, **kwargs)


def get_log(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.get_log(*args, **kwargs)


def initialize_from_log(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.initialize_from_log(*args, **kwargs)


def log(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.log(*args, **kwargs)


def setup(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.setup(*args, **kwargs)


def watch(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.watch(*args, **kwargs)


def watch_activation(*args, **kwargs):
    if _ANALOG_INSTANCE is None:
        raise RuntimeError(
            "AnaLog is not initialized. You must call analog.init() first."
        )
    return _ANALOG_INSTANCE.watch_activation(*args, **kwargs)
