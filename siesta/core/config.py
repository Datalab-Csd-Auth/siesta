"""
Global configuration accessor for Siesta Framework.

This module provides a centralized way to access the system configuration
from anywhere in the framework without needing to pass config through
multiple layers.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
import json
import logging
logger = logging.getLogger(__name__)


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    storage_type: str = Field(default="s3", description="Storage backend type")
    api: dict = Field(default_factory=lambda: {"host": "0.0.0.0", "port": 8000}, description="API host and port")
    s3_access_key: str = Field(default="minioadmin", description="S3/MinIO access key")
    s3_secret_key: str = Field(default="minioadmin", description="S3/MinIO secret key")
    s3_endpoint: str = Field(default="http://localhost:9000", description="S3/MinIO endpoint URL")
    s3_region: str = Field(default="us-east-1", description="S3/MinIO region")
    empty_namespace: bool = Field(default=False, description="Clear the storage namespace on startup")
    enable_streaming: bool = Field(default=True, description="Enable Kafka streaming support")
    enable_timing: bool = Field(default=True, description="Log execution timing for timed operations")
    spark_master: str = Field(default="spark://localhost:7077", description="Spark master URL")
    spark_app_name: str = Field(default="SiestaFramework", description="Spark application name")
    kafka_bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    raw_events_dir: str = Field(default="raw_events", description="Directory for raw streaming events")
    checkpoint_dir: str = Field(default="checkpoints", description="Directory for Spark streaming checkpoints")


DEFAULT_SYSTEM_CONFIG: Dict[str, Any] = SystemConfig().model_dump()

_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a JSON file and merge with defaults.
    
    Args:
        config_path: Path to the configuration JSON file. If None, returns DEFAULT_CONFIG.
        
    Returns:
        Dictionary containing configuration merged with defaults
    """
    # Start with default config
    config = DEFAULT_SYSTEM_CONFIG.copy()
    
    if not config_path:
        return config
    
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found. Using default config.")
        return config
    
    try:
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            # Merge user config with defaults
            config.update(user_config)
            logger.info(f"Configuration loaded from {config_path} and merged with defaults")
            return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}. Using default config.")
        return config


def initialize_config(config: Dict[str, Any]) -> None:
    """Initialize the global configuration.
    
    Should be called once during application startup (in app.py).
    
    Args:
        config: Configuration dictionary to make globally available
    """
    global _config
    _config = config


def get_system_config() -> Dict[str, Any]:
    """Get the current system configuration.
    
    Returns:
        Configuration dictionary. Returns DEFAULT_CONFIG if not initialized.
    """
    if _config is None:
        return DEFAULT_SYSTEM_CONFIG
    return _config


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific configuration value.
    
    Args:
        key: Configuration key to retrieve
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config = get_system_config()
    return config.get(key, default)


def is_initialized() -> bool:
    """Check if configuration has been initialized.
    
    Returns:
        True if config is initialized, False otherwise
    """
    return _config is not None
