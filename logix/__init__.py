from logix.logix import LogIX
from logix.scheduler import LogIXScheduler
from logix.utils import get_logger

__version__ = "0.1.1"

_LOGIX_INSTANCE = None


def init(project: str, config: str = "./config.yaml"):
    """Initialize LogIX.

    Args:
        project (str): The name of the project.
        config (dict): The configuration dictionary.
    """
    global _LOGIX_INSTANCE

    if _LOGIX_INSTANCE is not None:
        get_logger().warning(
            "LogIX is already initialized. If you want to initialize "
            + "additional LogIX instances, please use logix.LogIX instead."
        )
        return

    run = LogIX(project, config)

    _LOGIX_INSTANCE = run

    return run


def add_analysis(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.add_analysis(*args, **kwargs)


def add_lora(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.add_lora(*args, **kwargs)


def build_log_dataloader(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.build_log_dataloader(*args, **kwargs)


def clear(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.clear(*args, **kwargs)


def eval(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.eval(*args, **kwargs)


def finalize(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.finalize(*args, **kwargs)


def get_log(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.get_log(*args, **kwargs)


def initialize_from_log(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.initialize_from_log(*args, **kwargs)


def log(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.log(*args, **kwargs)


def setup(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.setup(*args, **kwargs)


def watch(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.watch(*args, **kwargs)


def watch_activation(*args, **kwargs):
    if _LOGIX_INSTANCE is None:
        raise RuntimeError(
            "LogIX is not initialized. You must call logix.init() first."
        )
    return _LOGIX_INSTANCE.watch_activation(*args, **kwargs)
