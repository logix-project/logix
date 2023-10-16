import sys
import abc
import logging


_logger = None


def get_logger():
    """
    Get global logger.
    """
    global _logger
    if _logger:
        return _logger
    logger = logging.getLogger("AnaLog")
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    logger.propagate = False
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    _logger = logger
    return _logger
