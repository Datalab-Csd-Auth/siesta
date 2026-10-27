import argparse
import datetime
import time
from pathlib import Path
from typing import Annotated, Any, Dict
from fastapi import Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from siesta.model.StorageModel import MetaData
from siesta.core.interfaces import SiestaModule, StorageManager
from siesta.core.config import get_system_config
from siesta.core.storageFactory import get_storage_manager
from siesta.modules.mine.processing import mine, stream_mining_results
from pyspark.sql import SparkSession

import json
import logging
logger = logging.getLogger(__name__)


class MiningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field("siesta", description="Storage namespace")
    categories: list[str] = Field(["*"], description="Constraint categories: 'positional', 'existential', 'ordered', 'unordered', 'negation', or '*' for all")
    grouping: str = Field("trace", description="Grouping strategy: 'trace' or 'window'")
    window_size: int = Field(30, description="Position-based window size when grouping='window'")
    support_threshold: float = Field(0.0, description="Minimum support fraction [0,1] to retain constraints")
    confidence_threshold: float = Field(0.0, description="Minimum confidence fraction [0,1] to retain constraints")
    interest_threshold: float = Field(0.0, description="Minimum interest fraction [0,1] to retain constraints")
    include_trace_lists: bool = Field(False, description="Append a pipe-delimited trace_ids column per constraint")
    force_recompute: bool = Field(False, description="Remine all traces ignoring previous mining state")
    branching_type: str = Field("none", description="Branching mode: 'none' (off), 'target' (branch over targets per source), or 'source' (branch over sources per target)")
    branching_policy: str = Field("or", description="Logical relation for merging singular constraints: 'and' (intersection), 'or' (union), or 'xor' (exclusive)")
    branching_bound: int = Field(0, description="Maximum number of activities in a branched set; 0 = unbounded (stop on support drop)")
    branching_approach: str = Field("auto", description="Greedy merge direction: 'bottomup', 'topdown', or 'auto' (topdown for OR, bottomup for AND/XOR)")
    output_path: str = Field("output/example_log", description="Local path prefix for the output CSV")


DEFAULT_MINING_CONFIG: Dict[str, Any] = MiningConfig().model_dump()


class Mining(SiestaModule):
        
    name = "miner"
    version = "1.0.0"
    spark: SparkSession
    storage: StorageManager
    siesta_config: Dict[str, Any]

    mining_config: Dict[str, Any]

    metadata: MetaData | None

    def __init__(self):
        super().__init__()
        self.mining_config = {}
        self.metadata = None

    def register_routes(self) -> SiestaModule.ApiRoutes|None:
        return {"run": ('POST', self.api_run)}

    def startup(self):
        logger.info("Startup complete.")

    def api_run(self, mining_config: Annotated[MiningConfig, Body(openapi_examples={
        "default": {
            "summary": "Mine all constraint categories with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "categories": ["*"],
                "grouping": "trace",
                "window_size": 30,
                "support_threshold": 0.0,
                "include_trace_lists": False,
                "force_recompute": False,
            },
        },
    })]) -> Any:
        """Mine declarative constraints from an indexed event log.

        Performs incremental constraint discovery across the selected categories. Only
        traces that evolved since the last mining run are processed unless
        `force_recompute` is set. Results are written to a CSV file and returned as a
        list of rows.

        **Config fields:**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `categories` *(list, default: `["*"]`)* - constraint categories to mine.
            `"*"` = all. Options: `"positional"`, `"existential"`, `"ordered"`, `"unordered"`, `"negation"`.
        - `grouping` *(str, default: `"trace"`)* - grouping strategy: `"trace"` or `"window"`.
        - `window_size` *(int, default: `30`)* - position-based window size when `grouping="window"`.
        - `support_threshold` *(float [0,1], default: `0.0`)* - minimum support fraction to retain constraints.
        - `confidence_threshold` *(float [0,1], default: `0.0`)* - minimum confidence fraction to retain constraints.
        - `interest_threshold` *(float [0,1], default: `0.0`)* - minimum interest fraction to retain constraints.
        - `include_trace_lists` *(bool, default: `false`)* - append a pipe-delimited `trace_ids` column per constraint.
        - `force_recompute` *(bool, default: `false`)* - remine all traces ignoring previous mining state.
        - `branching_type` *(str, default: `"none"`)* - branch singular constraints into branched ones:
            `"none"` = off, `"target"` = branch over targets sharing a source, `"source"` = branch over sources sharing a target.
        - `branching_policy` *(str, default: `"or"`)* - logical relation merging the singular trace sets:
            `"and"` (intersection), `"or"` (union), or `"xor"` (exclusive/symmetric difference).
        - `branching_bound` *(int, default: `0`)* - max activities in a branched set; `0` = unbounded (stop on support drop).
        - `branching_approach` *(str, default: `"auto"`)* - greedy merge direction: `"bottomup"`, `"topdown"`,
            or `"auto"` (top-down for OR, bottom-up for AND/XOR). Branched constraints report support only
            (confidence/interest are undefined for an activity set) and replace the singular constraints they absorb.
        """
        logger.info(f"{self.name} is running via API request.")

        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()
        
        try:
            self._load_mining_config(mining_config.model_dump())
        except ValueError as e:
            logger.error(f"Invalid mining config: {e}")
            return {"code": 400, "message": str(e)}

        logger.info(f"Running mining with args: {mining_config}")
        
        start_time = time.time()
        logger.info("Beginning mining process initiated by api.")
        self.metadata = mine(self.mining_config, self.storage)
        end_time = time.time()

        logger.info(f"Completed in {end_time - start_time} seconds. Results available at {self.mining_config['output_path']}.")

        return StreamingResponse(
            stream_mining_results(self.mining_config["output_path"], end_time - start_time),
            media_type="application/json",
        )


    def cli_run(self, args: Any, **kwargs: Any) -> Any:
        """
        Entry point for Mining via the command line.
        """
        logger.info(f"{self.name} is running with args: {args} and kwargs: {kwargs}")

        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        parser = argparse.ArgumentParser(description="Siesta Mining module")
        parser.add_argument('--mining_config', type=str, help='Path to configuration JSON file', required=False)

        parsed_args, _ = parser.parse_known_args(args)
        
        # Check if a config path is provided
        if parsed_args.mining_config:
            config_path = parsed_args.mining_config
            # Check if the provided path exists
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Config file {config_path} not found.")

            # Load configuration
            try:
                with open(config_path, 'r') as f:
                    user_mining_config = json.load(f)
                    
                    self._load_mining_config(user_mining_config)
                    self.storage.initialize_db(self.mining_config)

                    logger.info(f"Loaded config from {config_path}: {user_mining_config}")
            except Exception as e:
                logger.error(f"Failed to load mining config from {config_path}: {e}")
                raise ValueError(f"Invalid config file: {e}")

        start_time = time.time()
        logger.info("Beginning mining process initiated by cli.")
        self.metadata = mine(self.mining_config, self.storage)
        end_time = time.time()
        logger.info(f"Completed in {end_time - start_time:.2f} seconds. Results available at {self.mining_config['output_path']}.")
        
        return self.mining_config["output_path"]

    def _load_mining_config(self, config: Dict[str, Any]):
        # Validate that the specified log exists in storage before proceeding with mining. 
        if not self.storage.log_exists(config):
            log_name = config.get("log_name")
            logger.exception(f"Log '{log_name}' does not exist in storage. Run preprocessing first.")
            raise ValueError(f"Log '{log_name}' does not exist in storage. Run preprocessing first.")
        
        if config.get("log_name") is None:
            raise ValueError("Log name not specified in config.")
    
        self.mining_config = DEFAULT_MINING_CONFIG.copy()
        self.mining_config.update(config)

        # Ensure that the specified categories are valid before proceeding with mining.
        valid_categories = {"positional", "existential", "ordered", "unordered", "negation", "*"}
        if not set(self.mining_config["categories"]).issubset(valid_categories):
            raise ValueError(f"Invalid categories specified in mining_config: {self.mining_config['categories']}. Valid options are: {valid_categories}.")

        # Validate branching options. Branching merges singular constraints into
        # branched ones over the source or target activity side.
        branching_type = str(self.mining_config.get("branching_type", "none")).lower()
        if branching_type not in {"none", "source", "target"}:
            raise ValueError(f"Invalid branching_type: {self.mining_config.get('branching_type')}. Valid options are: none, source, target.")
        self.mining_config["branching_type"] = branching_type

        branching_policy = str(self.mining_config.get("branching_policy", "or")).lower()
        if branching_policy not in {"and", "or", "xor"}:
            raise ValueError(f"Invalid branching_policy: {self.mining_config.get('branching_policy')}. Valid options are: and, or, xor.")
        self.mining_config["branching_policy"] = branching_policy

        branching_approach = str(self.mining_config.get("branching_approach", "auto")).lower()
        if branching_approach not in {"auto", "bottomup", "topdown"}:
            raise ValueError(f"Invalid branching_approach: {self.mining_config.get('branching_approach')}. Valid options are: auto, bottomup, topdown.")
        self.mining_config["branching_approach"] = branching_approach

        if int(self.mining_config.get("branching_bound", 0)) < 0:
            raise ValueError("branching_bound must be >= 0 (0 = unbounded).")

        # If the caller left output_path at its default, derive it from log_name instead.
        if config.get("output_path") is None or config.get("output_path") == "output/example_log":
            config["output_path"] = "output/" + config.get("log_name", "mining_results")

        # Ensure output_path is unique for each run to avoid overwriting results
        given_output_path = config.get("output_path")
        Path(given_output_path).parent.mkdir(parents=True, exist_ok=True)
        self.mining_config["output_path"] = given_output_path + "_" + str(datetime.datetime.now().timestamp()) + ".csv"
