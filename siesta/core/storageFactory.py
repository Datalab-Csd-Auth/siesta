from typing import Dict, Any, Optional
from siesta.core.interfaces import StorageManager
from siesta.storage.S3.S3Manager import S3Manager
import logging
logger = logging.getLogger(__name__)


# Global storage manager instance
_storage_manager: Optional[StorageManager] = None

def get_storage_manager() -> StorageManager:
    """Get the global storage manager instance.
    
    Returns:
        The initialized StorageManager instance.
    """
    if _storage_manager is None:
        raise RuntimeError("StorageManager not initialized. Call startup() first.")
    return _storage_manager


def set_storage_manager(manager: StorageManager) -> None:
    """Set the global storage manager instance.
    
    Args:
        manager: The StorageManager instance to set as global.
    """
    global _storage_manager
    _storage_manager = manager


class StorageManagerFactory:
    """
    Factory class for creating storage manager instances based on configuration.
    
    This allows the framework to dynamically select the appropriate storage backend
    (S3, Cassandra, PostgreSQL, etc.) without hardcoding dependencies.
    """
    
    _registry: Dict[str, type[StorageManager]] = {
        "s3": S3Manager,
        # Add more storage managers here as they are implemented
        # "cassandra": CassandraManager,
        # "postgres": PostgresManager,
    }
    
    @classmethod
    def register_storage_manager(cls, name: str, manager_class: type[StorageManager]) -> None:
        """
        Register a new storage manager type.
        
        Args:
            name: Identifier for the storage manager (e.g., "s3", "cassandra")
            manager_class: Class that implements StorageManager interface
        """
        if not issubclass(manager_class, StorageManager):
            raise TypeError(f"{manager_class} must be a subclass of StorageManager")
        cls._registry[name.lower()] = manager_class
        logger.info(f"Registered storage manager: {name} -> {manager_class.__name__}")
    
    @classmethod
    def create_storage_manager(cls, config: Dict[str, Any]) -> StorageManager:
        """
        Create and return a storage manager instance based on configuration.
        
        Args:
            config: Configuration dictionary that must contain a "storage_type" key
            spark_manager: The spark manager instance to pass to the storage manager
            
        Returns:
            An instance of the appropriate StorageManager implementation
            
        Raises:
            ValueError: If storage_type is not specified or not recognized
            RuntimeError: If initialization fails
        """
        try:
            storage_type = config.get("storage_type", "s3").lower()
            
            if not storage_type:
                raise ValueError(
                    "Configuration must specify 'storage_type'. "
                    f"Available options: {', '.join(cls._registry.keys())}"
                )
            
            manager_class = cls._registry.get(storage_type)
            
            if manager_class is None:
                raise ValueError(
                    f"Unknown storage type: '{storage_type}'. "
                    f"Available options: {', '.join(cls._registry.keys())}"
                )
            
            logger.info(f"Creating storage manager: {storage_type} ({manager_class.name} v{manager_class.version})")
            return manager_class()
            
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error initializing storage manager: {e}")
            logger.error(f"Available storage types: {', '.join(cls.get_available_storage_types())}")
            raise
    
    @classmethod
    def get_available_storage_types(cls) -> list[str]:
        """
        Get a list of all registered storage manager types.
        
        Returns:
            List of storage type identifiers
        """
        return list(cls._registry.keys())
