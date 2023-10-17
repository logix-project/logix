from analog.config import Config

from .base import HessianHandlerBase
from .kfac import KFACHessianHandler


def init_hessian_handler_from_config(config: Config) -> HessianHandlerBase:
    """
    Initialize a Hessian handler from the configuration.

    Args:
        config (dict): The configuration for the Hessian handler.

    Returns:
        The initialized Hessian handler.
    """
    hessian_handler = None
    hessian_config = config.get_hessian_config()
    hessian_type = hessian_config.get("type", "kfac")
    if hessian_type == "kfac":
        hessian_handler = KFACHessianHandler(hessian_config)
    else:
        raise ValueError(f"Unknown Hessian type: {hessian_type}")
    return hessian_handler
