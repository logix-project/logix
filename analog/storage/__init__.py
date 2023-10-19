from analog.config import Config

from .base import StorageHandlerBase
from .default import DefaultStorageHandler
from .mongo_db import MongoDBStorageHandler


def init_storage_handler_from_config(storage_config: Config) -> StorageHandlerBase:
    """
    Initialize a storage handler from the configuration.

    Args:
        config (dict): The configuration for the storage handler.

    Returns:
        The initialized storage handler.
    """
    storage_handler = None
    storage_type = storage_config.get("type", "default")
    if storage_type == "default":
        storage_handler = DefaultStorageHandler(storage_config)
    elif storage_type == "mongodb":
        storage_handler = MongoDBStorageHandler(storage_config)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
    return storage_handler
