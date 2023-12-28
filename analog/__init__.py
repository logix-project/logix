from analog.analog import AnaLog
from analog.scheduler import AnaLogScheduler
from analog.utils import get_logger


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
            "AnaLog is already initialized. If you want to initialize " +
            "additional AnaLog instances, please use analog.AnaLog instead."
        )
        return

    run = AnaLog(project, config)

    _ANALOG_INSTANCE = run

    return run


if _ANALOG_INSTANCE is not None:
    add_analysis = _ANALOG_INSTANCE.add_analysis
    add_lora = _ANALOG_INSTANCE.add_lora
    build_log_dataloader = _ANALOG_INSTANCE.build_log_dataloader
    clear = _ANALOG_INSTANCE.clear
    eval = _ANALOG_INSTANCE.eval
    finalize = _ANALOG_INSTANCE.finalize
    log = _ANALOG_INSTANCE.log
    setup = _ANALOG_INSTANCE.setup
    watch = _ANALOG_INSTANCE.watch
    watch_activation = _ANALOG_INSTANCE.watch_activation
