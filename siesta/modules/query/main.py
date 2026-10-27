import argparse
from pathlib import Path
from typing import Annotated, Any, Dict
from fastapi import Body
from pydantic import BaseModel, ConfigDict, Field
from siesta.core.interfaces import SiestaModule, StorageManager
from pyspark.sql import SparkSession
from siesta.core.storageFactory import get_storage_manager
from siesta.core.config import get_system_config
from siesta.core.logger import timed
from siesta.model.StorageModel import MetaData
import json
import logging
from siesta.modules.query.processors.detection_query import process_detection_query
from siesta.modules.query.processors.exploration_query import process_exploration_query
from siesta.modules.query.processors.stats_query import process_stats_query
from siesta.modules.query.processing import load_metadata


logger = logging.getLogger(__name__)


class QueryMethodInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    pattern: str = Field(default="", description="Pattern string, e.g. 'A B* C' or 'A[pos=?1]+ B[pos=?1+5]'")
    explore_mode: str = Field(default="accurate", description="'accurate', 'fast', or 'hybrid'. Used by exploration only")
    explore_k: int = Field(default=10, description="Candidates for hybrid exploration (0 = fast mode)")


class QueryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field(default="example_log", description="Name of the indexed log")
    storage_namespace: str = Field(default="siesta", description="Storage namespace")
    method: str = Field(default="statistics", description="'statistics', 'detection', or 'exploration'. Ignored by API endpoints - set automatically.")
    query: QueryMethodInput = Field(default_factory=QueryMethodInput, description="Query-specific parameters")
    support_threshold: float = Field(default=0.0, description="Minimum support fraction [0,1] for results")


DEFAULT_QUERY_CONFIG: Dict[str, Any] = QueryConfig().model_dump()


class Querying(SiestaModule):
    def __init__(self):
        super().__init__()

    name = "executor"
    version = "1.0.0"
    spark: SparkSession
    storage: StorageManager
    siesta_config: Dict[str, Any]
    query_config: Dict[str, Any]
    metadata: MetaData

    def startup(self):
        logger.info(f"{self.name} v{self.version} initialized.")

    def register_routes(self) -> SiestaModule.ApiRoutes:
        return {
            "statistics":  ("POST", self.api_statistics),
            "detection":   ("POST", self.api_detection),
            "exploration": ("POST", self.api_exploration),
        }

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def cli_run(self, args: Any, **kwargs: Any) -> Any:
        logger.info(f"{self.name} is running with args: {args} and kwargs: {kwargs}")

        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        parser = argparse.ArgumentParser(description="Siesta Query module")
        parser.add_argument('--query_config', type=str, help='Path to configuration JSON file', required=False)

        parsed_args, _ = parser.parse_known_args(args)

        if not parsed_args.query_config:
            raise RuntimeError("Config not provided. Use --query_config <path>")

        config_path = parsed_args.query_config
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file {config_path} not found.")

        try:
            with open(config_path, 'r') as f:
                self._load_query_config(json.load(f))
                logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading config from {config_path}: {e}")

        result = self._dissect_query(self.query_config)
        if result is not None:
            logger.info(f"Query result: {result}")
        return result

    # ------------------------------------------------------------------
    # API entry points
    # ------------------------------------------------------------------

    def api_statistics(self, query_config: Annotated[QueryConfig, Body(
        openapi_examples={
            "pair_support": {
                "summary": "Support for a directly-follows pair",
                "value": {"log_name": "example_log", "query": {"pattern": "A B"}, "support_threshold": 0.0},
            },
            "chain_support": {
                "summary": "Support for a longer chain",
                "value": {"log_name": "example_log", "query": {"pattern": "A B C"}, "support_threshold": 0.05},
            },
        }
    )]) -> Any | None:
        """Compute occurrence counts and support statistics for an activity pattern.

        Splits the pattern into directly-following pairs and retrieves their
        occurrence counts and support fractions from the pre-built count table.

        **Request body (`QueryConfig`):**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log.
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `query.pattern` *(str)* - pattern string, e.g. `"A B"` or `"A B C"`.
        - `support_threshold` *(float [0,1], default: `0.0`)* - minimum support fraction to include a pair in results.
        """
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = query_config.model_dump()
        config["method"] = "statistics"
        try:
            self._load_query_config(config)
            self._load_metadata()
        except Exception as e:
            logger.exception(f"Error processing query config: {e}")
            return {"code": 400, "error": f"Invalid query configuration: {e}"}
        
        return timed(process_stats_query, "Stats Query: ", self.query_config, self.metadata)

    def api_detection(self, query_config: Annotated[QueryConfig, Body(
        openapi_examples={
            "simple_sequence": {
                "summary": "Detect traces containing A directly followed by B",
                "value": {"log_name": "example_log", "query": {"pattern": "A B"}, "support_threshold": 0.0},
            },
            "with_quantifiers": {
                "summary": "Detect traces with positional constraints",
                "value": {"log_name": "example_log", "query": {"pattern": "A B* C"}, "support_threshold": 0.1},
            },
        }
    )]) -> Any | None:
        """Retrieve all traces that satisfy a temporal or positional pattern.

        Evaluates the pattern against the sequence index and returns matching
        trace IDs together with the matched positions and an overall support fraction.

        **Request body (`QueryConfig`):**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log.
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `query.pattern` *(str)* - pattern to detect, e.g. `"A B* C"` or `"A[pos=?1]+ B[pos=?1+5]"`.
        - `support_threshold` *(float [0,1], default: `0.0`)* - minimum per-trace support to include a match.
        """
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = query_config.model_dump()
        config["method"] = "detection"
        try:
            self._load_query_config(config)
            self._load_metadata()
        except Exception as e:
            logger.exception(f"Error processing query config: {e}")
            return {"code": 400, "error": f"Invalid query configuration: {e}"}

        return timed(process_detection_query, "Detection Query: ", self.query_config, self.metadata)

    def api_exploration(self, query_config: Annotated[QueryConfig, Body(
        openapi_examples={
            "accurate": {
                "summary": "Accurate exploration after a prefix",
                "value": {"log_name": "example_log", "query": {"pattern": "A B", "explore_mode": "accurate"}},
            },
            "hybrid": {
                "summary": "Hybrid exploration with candidate cap",
                "value": {"log_name": "example_log", "query": {"pattern": "A", "explore_mode": "hybrid", "explore_k": 5}},
            },
        }
    )]) -> Any | None:
        """Find the most likely activity continuations after a partial pattern.

        Evaluates which activities most commonly follow the given prefix pattern
        and ranks them by support. Three modes trade accuracy for speed.

        **Request body (`QueryConfig`):**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log.
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `query.pattern` *(str)* - prefix pattern to continue from, e.g. `"A B"`.
        - `query.explore_mode` *(str, default: `"accurate"`)* - `"accurate"` (full index scan), `"fast"` (count table only), or `"hybrid"` (fast pre-filter + accurate top-k).
        - `query.explore_k` *(int, default: `10`)* - number of candidates to evaluate accurately in `"hybrid"` mode (`0` = pure fast mode).
        - `support_threshold` *(float [0,1], default: `0.0`)* - minimum support fraction to include a continuation.
        """
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = query_config.model_dump()
        config["method"] = "exploration"
        try:
            self._load_query_config(config)
            self._load_metadata()
        except Exception as e:
            logger.exception(f"Error processing query config: {e}")
            return {"code": 400, "error": f"Invalid query configuration: {e}"}
        
        return timed(process_exploration_query, "Exploration Query: ", self.query_config, self.metadata)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _load_query_config(self, config: Dict[str, Any]):
        if not config.get("method"):
            raise ValueError("Query method not specified in config.")
        if config.get("log_name") is None:
            raise ValueError("Log name not specified in config.")
        if config.get("query", {}).get("pattern") is None:
            raise ValueError("Query pattern not specified in config.")
        self.query_config = DEFAULT_QUERY_CONFIG.copy()
        self.query_config = self.query_config | config
        logger.info(self.query_config)

    def _load_metadata(self):
        self.metadata = load_metadata(self.query_config, self.storage)

    def _dissect_query(self, config: Dict[str, Any]):
        self._load_metadata()
        match config.get("method", "").lower():
            case "statistics":
                return timed(process_stats_query, "Stats Query: ", config, self.metadata)
            case "detection":
                return timed(process_detection_query, "Detection Query: ", config, self.metadata)
            case "exploration":
                return timed(process_exploration_query, "Exploration Query: ", config, self.metadata)
