from analog.analog import AnaLog
from analog.scheduler import AnaLogScheduler


__version__ = "0.1.0"

_ANALOG_INSTANCES = {}


def init(project: str, config: str = "./config.yaml"):
    """Initialize AnaLog.

    Args:
        project (str): The name of the project.
        config (dict): The configuration dictionary.
    """
    run = AnaLog(project, config)

    global _analog_instances
    _ANALOG_INSTANCES[project] = run

    return run


def is_initialized_single():
    """Check if AnaLog is initialized.

    Returns:
        bool: True if AnaLog is initialized, False otherwise.
    """
    return len(_ANALOG_INSTANCES) == 1


def watch(model, type_filter=None, name_filter=None):
    """Watch a model.

    Args:
        model (torch.nn.Module): The model to be watched.
        type_filter (list, optional): The types of modules to be watched. Defaults to None.
        name_filter (list, optional): The names of modules to be watched. Defaults to None.
    """
    if is_initialized_single():
        for run in _ANALOG_INSTANCES.values():
            run.watch(model, type_filter, name_filter)
    else:
        raise RuntimeError("Multiple AnaLog instances are not supported.")


def log(data_id, mask=None):
    """Log data.

    Args:
        data_id (str): The id of the data.
        mask (dict, optional): The mask of the data. Defaults to None.
    """
    if is_initialized_single():
        for run in _ANALOG_INSTANCES.values():
            run.log(data_id, mask)
    else:
        raise RuntimeError("Multiple AnaLog instances are not supported.")
