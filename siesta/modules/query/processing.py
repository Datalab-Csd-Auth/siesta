from typing import Any, Dict
from siesta.core.interfaces import StorageManager
from siesta.model.StorageModel import MetaData


def load_metadata(query_config: Dict[str, Any], storage: StorageManager) -> MetaData:
    metadata = MetaData(
        storage_namespace=query_config.get("storage_namespace", "siesta"),
        log_name=query_config.get("log_name", "default_log"),
        storage_type=query_config.get("storage_type", "s3"),
    )
    return storage.read_metadata_table(metadata)
