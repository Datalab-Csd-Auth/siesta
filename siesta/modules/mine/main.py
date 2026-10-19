import argparse
import datetime
import re
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any, Dict
from fastapi import Body
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field
from siesta.model.StorageModel import MetaData
from siesta.core.interfaces import SiestaModule, StorageManager
from siesta.core.config import get_system_config, get_config_value
from siesta.core.logger import timed
from siesta.core.storageFactory import get_storage_manager
from siesta.core.sparkManager import cleanup as spark_cleanup
from siesta.modules.mine.existential import discover_existential
from siesta.modules.mine.positional import discover_positional
from siesta.modules.mine.ordered import discover_ordered
from siesta.modules.mine.unordered import discover_unordered
from siesta.modules.mine.negations import discover_negations
from pyspark.sql import SparkSession, DataFrame, functions as F

import csv
import json
import logging
logger = logging.getLogger(__name__)


class MiningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field(default_factory=lambda: get_config_value("storage_namespace_default", "siesta"), description="Storage namespace")
    categories: list[str] = Field(["*"], description="Constraint categories: 'positional', 'existential', 'ordered', 'unordered', 'negation', or '*' for all")
    grouping: str = Field("trace", description="Grouping strategy: 'trace' or 'window'")
    window_size: int = Field(30, description="Position-based window size when grouping='window'")
    support_threshold: float = Field(0.02, description="Minimum support fraction [0,1] to retain constraints")
    max_support_threshold: float | None = Field(0.999, description="Maximum support fraction [0,1] to retain constraints; null = no upper bound")
    confidence_threshold: float = Field(0.95, description="Minimum confidence fraction [0,1] to retain constraints")
    max_confidence_threshold: float | None = Field(0.999, description="Maximum confidence fraction [0,1] to retain constraints; null = no upper bound")
    interest_threshold: float = Field(0.0, description="Minimum interest fraction [0,1] to retain constraints")
    include_trace_lists: bool = Field(False, description="Append a pipe-delimited trace_ids column per constraint")
    force_recompute: bool = Field(False, description="Remine all traces ignoring previous mining state")
    output_path: str = Field("output/example_log", description="Local path prefix for the output CSV. CLI runs keep the file; API runs discard it once the results are in the response")


DEFAULT_MINING_CONFIG: Dict[str, Any] = MiningConfig().model_dump()


# Declare templates produced by each miner, keyed by the category under which they are
# stored. The miners are the source of truth: the ordered miner also emits the two
# not_* ordered templates, so those live in the 'ordered' category rather than 'negation'.
# Existence and absence are derived from the stored 'exactly' rows, exactly as
# discover_existential does, so they are queryable even though only 'exactly' is persisted.
CATEGORY_TEMPLATES: Dict[str, tuple[str, ...]] = {
    "positional": ("init", "end"),
    "existential": ("exactly", "existence", "absence"),
    "ordered": (
        "response", "precedence", "succession",
        "alternate_response", "alternate_precedence",
        "chain_response", "chain_precedence", "chain_succession",
        "not_succession", "not_chain_succession",
    ),
    "unordered": ("coexistence", "choice", "exclusive_choice"),
    "negation": ("not_coexistence",),
}

# Canonical mining order: the negation miner consumes the coexistence constraints the
# unordered miner writes, so the categories must always be mined in this sequence.
CONSTRAINT_CATEGORIES: tuple[str, ...] = ("positional", "existential", "ordered", "unordered", "negation")

TEMPLATE_CATEGORY: Dict[str, str] = {
    template: category
    for category, templates in CATEGORY_TEMPLATES.items()
    for template in templates
}

CONSTRAINT_COLUMNS: list[str] = ["category", "template", "source", "target", "occurrences", "trace_id"]


class ConstraintQueryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field(default_factory=lambda: get_config_value("storage_namespace_default", "siesta"), description="Storage namespace")
    trace_id: str = Field(..., description="Trace whose mined constraints are returned")
    category: str | None = Field(None, description=f"Restrict to one constraint category: {list(CATEGORY_TEMPLATES)}. null or '*' = all categories")
    template: str | None = Field(None, description=f"Restrict to one declare template, e.g. 'response'. null = every template of the selected category. Options: {sorted(TEMPLATE_CATEGORY)}")
    output_format: str = Field("json", description="'json' (default) returns the constraints inline; 'csv' returns a CSV file download")


class RuleTraceQueryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field(default_factory=lambda: get_config_value("storage_namespace_default", "siesta"), description="Storage namespace")
    source: str = Field(..., description="Source activity of the rule whose backing traces are returned")
    target: str | None = Field(None, description="Target activity of the rule. null matches rules with no target (e.g. positional / existential)")
    category: str | None = Field(None, description=f"Restrict to one constraint category: {list(CATEGORY_TEMPLATES)}. null or '*' = all categories")
    template: str | None = Field(None, description=f"Restrict to one declare template, e.g. 'coexistence'. null = every template of the selected category. Options: {sorted(TEMPLATE_CATEGORY)}")
    output_format: str = Field("json", description="'json' (default) returns the matched rules and their traces inline; 'csv' returns a CSV file download")


UNORDERED_TEMPLATES: tuple[str, ...] = ("coexistence", "choice", "exclusive_choice")
VIOLATION_MODES: tuple[str, ...] = ("XOR", "NAND", "NOR")


class ViolationQueryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field(default_factory=lambda: get_config_value("storage_namespace_default", "siesta"), description="Storage namespace")
    source: str = Field(..., description="Source activity of the rule")
    target: str = Field(..., description="Target activity of the rule")
    template: str | None = Field(None, description=f"Restrict to one unordered template: {list(UNORDERED_TEMPLATES)}. null = all applicable")
    mode: str = Field("XOR", description="Violation mode: 'XOR' (activated but not fulfilled), 'NAND' (not fulfilled), 'NOR' (not activated)")
    output_format: str = Field("json", description="'json' (default) returns the violations inline; 'csv' returns a CSV file download")


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
        return {
            "run": ('POST', self.api_run),
            "constraints": ('POST', self.api_constraints),
            "traces": ('POST', self.api_traces),
            "violations": ('POST', self.api_violations),
        }

    def startup(self):
        logger.info("Startup complete.")

    def api_run(self, mining_config: Annotated[MiningConfig, Body(openapi_examples={
        "default": {
            "summary": "Mine all constraint categories with default settings",
            "value": {
                "log_name": "example_log",
                "categories": ["*"],
                "grouping": "trace",
                "window_size": 30,
                "support_threshold": 0.02,
                "max_support_threshold": 0.999,
                "confidence_threshold": 0.95,
                "max_confidence_threshold": 0.999,
                "include_trace_lists": False,
                "force_recompute": False,
            },
        },
    })]) -> Any:
        """Mine declarative constraints from an indexed event log.

        Performs incremental constraint discovery across the selected categories. Only
        traces that evolved since the last mining run are processed unless
        `force_recompute` is set. Results are returned as a list of rows; the CSV file the
        miner writes on the driver is removed once it has been read into the response, since
        the response already carries it. CLI runs keep their file at `output_path`.

        **Config fields:**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: system config `storage_namespace_default`)* - storage namespace.
        - `categories` *(list, default: `["*"]`)* - constraint categories to mine.
            `"*"` = all. Options: `"positional"`, `"existential"`, `"ordered"`, `"unordered"`, `"negation"`.
        - `grouping` *(str, default: `"trace"`)* - grouping strategy: `"trace"` or `"window"`.
        - `window_size` *(int, default: `30`)* - position-based window size when `grouping="window"`.
        - `support_threshold` *(float [0,1], default: `0.02`)* - minimum support fraction to retain constraints.
        - `max_support_threshold` *(float [0,1] | null, default: `0.999`)* - maximum support fraction to retain constraints. `null` = no upper bound.
        - `confidence_threshold` *(float [0,1], default: `0.95`)* - minimum confidence fraction to retain constraints.
        - `max_confidence_threshold` *(float [0,1] | null, default: `0.999`)* - maximum confidence fraction to retain constraints. `null` = no upper bound.
        - `interest_threshold` *(float [0,1], default: `0.0`)* - minimum interest fraction to retain constraints.
        - `include_trace_lists` *(bool, default: `false`)* - append a pipe-delimited `trace_ids` column per constraint.
        - `force_recompute` *(bool, default: `false`)* - remine all traces ignoring previous mining state.
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
        self.mine(caller="api")
        end_time = time.time()

        logger.info(f"Completed in {end_time - start_time} seconds. Results available at {self.mining_config['output_path']}.")

        try:
            with open(self.mining_config["output_path"], 'r', newline="") as f:
                mined = list(csv.DictReader(f))
        except Exception:
            logger.error(f"Failed to parse mining results from {self.mining_config['output_path']}. Check if the file is a valid CSV and inspect logs for details.")
            return {"code": 500, "message": f"Cannot parse mining results. Check logs and {self.mining_config['output_path']} for details."}

        # The rows are in the response now, so the file on the driver has no reader left.
        self._discard_output_file(self.mining_config["output_path"])

        return {"code": 200, "mined": mined, "time": end_time - start_time}


    def api_constraints(self, query_config: Annotated[ConstraintQueryConfig, Body(openapi_examples={
        "all": {
            "summary": "All stored constraints of a trace",
            "value": {
                "log_name": "example_log",
                "trace_id": "1",
                "output_format": "json",
            },
        },
        "category": {
            "summary": "Only the ordered constraints of a trace",
            "value": {
                "log_name": "example_log",
                "trace_id": "1",
                "category": "ordered",
                "output_format": "json",
            },
        },
        "template": {
            "summary": "Only one declare template, as a CSV download",
            "value": {
                "log_name": "example_log",
                "trace_id": "1",
                "template": "response",
                "output_format": "csv",
            },
        },
    })]) -> Any:
        """Return the mined declarative constraints stored for a single trace.

        Reads the constraint tables persisted by the miner and returns the rows belonging to
        `trace_id`, optionally narrowed to one category and/or one declare template. When a
        requested category has never been mined for this log, mining is run for the missing
        categories first (with `force_recompute`, so the freshly created table covers every
        trace and not only the ones that evolved since the last mining run), and the
        constraints are then read back.

        Note that already-stored constraints are returned as-is: mining is only triggered when
        a category is missing altogether, so call `/mining/run` to refresh constraints for a log
        that has been re-indexed since it was last mined. A query covering `negation` can still
        trigger mining after a `/mining/run`, since not-coexistence is only persisted per trace
        when that run had `include_trace_lists` set.

        The `existence` and `absence` templates are derived here from the stored `exactly` rows,
        the same way the existential miner derives them.

        **Config fields:**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: system config `storage_namespace_default`)* - storage namespace.
        - `trace_id` *(str)* - trace whose constraints are returned. **Required.**
        - `category` *(str, optional)* - one of `"positional"`, `"existential"`, `"ordered"`,
            `"unordered"`, `"negation"`. `null` or `"*"` returns every category.
        - `template` *(str, optional)* - a single declare template, e.g. `"response"`. Must belong
            to `category` when both are given. `null` returns every template of the selected category.
        - `output_format` *(str, default: `"json"`)* - `"json"` returns the constraints inline,
            `"csv"` returns them as a CSV file download. The CSV is built in a temporary file
            that is deleted once the response has been sent.
        """
        logger.info(f"{self.name} is fetching constraints via API request: {query_config}")

        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        try:
            categories, template = self._resolve_constraint_scope(query_config.category, query_config.template)
        except ValueError as e:
            logger.error(f"Invalid constraint query: {e}")
            return {"code": 400, "message": str(e)}

        output_format = (query_config.output_format or "json").lower()
        if output_format not in ("json", "csv"):
            return {"code": 400, "message": f"Invalid output_format '{query_config.output_format}'. Valid options are: ['json', 'csv']."}

        if not self.storage.log_exists({"log_name": query_config.log_name, "storage_namespace": query_config.storage_namespace}):
            message = f"Log '{query_config.log_name}' does not exist in namespace '{query_config.storage_namespace}'. Run preprocessing first."
            logger.error(message)
            return {"code": 404, "message": message}

        metadata = MetaData(
            storage_namespace=query_config.storage_namespace,
            log_name=query_config.log_name,
            storage_type=getattr(query_config, "storage_type", "s3"),
        )
        metadata = self.storage.read_metadata_table(metadata)

        # Mine on demand whatever the query needs but storage does not hold yet.
        mined_now = self._mine_missing_categories(query_config, metadata, categories)
        if mined_now:
            metadata = self.metadata

        constraints = self._collect_trace_constraints(metadata, categories, template, query_config.trace_id)
        logger.info(f"Found {len(constraints)} constraint(s) for trace '{query_config.trace_id}' of log '{query_config.log_name}'.")

        if output_format == "csv":
            download_name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{query_config.log_name}_{query_config.trace_id}_constraints")
            with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv", prefix=download_name + "_", delete=False) as f:
                writer = csv.DictWriter(f, fieldnames=CONSTRAINT_COLUMNS)
                writer.writeheader()
                writer.writerows(constraints)
                output_path = f.name
            logger.info(f"Constraints written to {output_path}.")
            # The file only exists to be streamed back, so drop it once the response is out.
            return FileResponse(
                output_path,
                media_type="text/csv",
                filename=download_name + ".csv",
                background=BackgroundTask(self._discard_output_file, output_path),
            )

        return {
            "code": 200,
            "log_name": query_config.log_name,
            "storage_namespace": query_config.storage_namespace,
            "trace_id": query_config.trace_id,
            "categories": categories,
            "template": template,
            "mined_now": mined_now,
            "constraint_count": len(constraints),
            "constraints": constraints,
        }


    def api_traces(self, query_config: Annotated[RuleTraceQueryConfig, Body(openapi_examples={
        "unordered": {
            "summary": "Traces backing a coexistence rule between two activities",
            "value": {
                "log_name": "example_log",
                "source": "A",
                "target": "B",
                "template": "coexistence",
                "output_format": "json",
            },
        },
        "any": {
            "summary": "Every rule between two activities, across categories",
            "value": {
                "log_name": "example_log",
                "source": "A",
                "target": "B",
                "output_format": "json",
            },
        },
    })]) -> Any:
        """Return the traces backing the mined rule(s) with a given source and target.

        The inverse of `/mining/constraints`: instead of "which constraints does this
        trace satisfy", this answers "which traces satisfy this rule". A rule is keyed by
        its `source` and `target` activity strings (not a hashed id), so it resolves
        regardless of how the caller obtained the rule. The lookup can be narrowed to one
        category and/or one declare template; without them every stored rule matching the
        source/target pair is returned, one entry per (category, template, source, target).

        Missing categories are mined on demand, exactly as `/mining/constraints` does, so a
        query for a category never mined yet triggers a `force_recompute` run of it first.
        Already-stored constraints are returned as-is; call `/mining/run` to refresh a log
        re-indexed since it was last mined.

        **Config fields:**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: system config `storage_namespace_default`)* - storage namespace.
        - `source` *(str)* - source activity of the rule. **Required.**
        - `target` *(str, optional)* - target activity. `null` matches rules with no target
            (positional / existential). For unordered rules both are always present.
        - `category` *(str, optional)* - one of `"positional"`, `"existential"`, `"ordered"`,
            `"unordered"`, `"negation"`. `null` or `"*"` searches every category.
        - `template` *(str, optional)* - a single declare template, e.g. `"coexistence"`. Must
            belong to `category` when both are given. `null` searches every template of the category.
        - `output_format` *(str, default: `"json"`)* - `"json"` returns the matched rules and
            their traces inline, `"csv"` returns them as a CSV file download (one row per rule,
            trace ids pipe-delimited). The CSV is a temporary file removed once the response is sent.
        """
        logger.info(f"{self.name} is fetching rule traces via API request: {query_config}")

        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        try:
            categories, template = self._resolve_constraint_scope(query_config.category, query_config.template)
        except ValueError as e:
            logger.error(f"Invalid rule-trace query: {e}")
            return {"code": 400, "message": str(e)}

        output_format = (query_config.output_format or "json").lower()
        if output_format not in ("json", "csv"):
            return {"code": 400, "message": f"Invalid output_format '{query_config.output_format}'. Valid options are: ['json', 'csv']."}

        if not self.storage.log_exists({"log_name": query_config.log_name, "storage_namespace": query_config.storage_namespace}):
            message = f"Log '{query_config.log_name}' does not exist in namespace '{query_config.storage_namespace}'. Run preprocessing first."
            logger.error(message)
            return {"code": 404, "message": message}

        metadata = MetaData(
            storage_namespace=query_config.storage_namespace,
            log_name=query_config.log_name,
            storage_type=getattr(query_config, "storage_type", "s3"),
        )
        metadata = self.storage.read_metadata_table(metadata)

        # Mine on demand whatever the query needs but storage does not hold yet.
        mined_now = self._mine_missing_categories(query_config, metadata, categories)
        if mined_now:
            metadata = self.metadata

        rules = self._collect_rule_traces(metadata, categories, template, query_config.source, query_config.target)
        trace_ids = sorted({trace_id for rule in rules for trace_id in rule["trace_ids"]})
        logger.info(
            f"Found {len(rules)} rule(s) and {len(trace_ids)} distinct trace(s) for "
            f"source '{query_config.source}' target '{query_config.target}' of log '{query_config.log_name}'."
        )

        if output_format == "csv":
            columns = ["category", "template", "source", "target", "occurrences", "trace_count", "trace_ids"]
            download_name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{query_config.log_name}_{query_config.source}_{query_config.target}_traces")
            with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv", prefix=download_name + "_", delete=False) as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for rule in rules:
                    row = {col: rule.get(col) for col in columns}
                    row["trace_ids"] = "|".join(rule["trace_ids"])
                    writer.writerow(row)
                output_path = f.name
            logger.info(f"Rule traces written to {output_path}.")
            return FileResponse(
                output_path,
                media_type="text/csv",
                filename=download_name + ".csv",
                background=BackgroundTask(self._discard_output_file, output_path),
            )

        return {
            "code": 200,
            "log_name": query_config.log_name,
            "storage_namespace": query_config.storage_namespace,
            "source": query_config.source,
            "target": query_config.target,
            "categories": categories,
            "template": template,
            "mined_now": mined_now,
            "rule_count": len(rules),
            "trace_count": len(trace_ids),
            "trace_ids": trace_ids,
            "rules": rules,
        }


    def api_violations(self, query_config: Annotated[ViolationQueryConfig, Body(openapi_examples={
        "coexistence_xor": {
            "summary": "Traces that activate coexistence(A,B) but do not fulfill it",
            "value": {
                "log_name": "example_log",
                "source": "A",
                "target": "B",
                "template": "coexistence",
                "mode": "XOR",
                "output_format": "json",
            },
        },
        "nand": {
            "summary": "All traces not fulfilling any unordered rule between A and B",
            "value": {
                "log_name": "example_log",
                "source": "A",
                "target": "B",
                "mode": "NAND",
                "output_format": "json",
            },
        },
    })]) -> Any:
        """Return traces that violate mined unordered rule(s) for a given activity pair.

        - **XOR** — activated but not fulfilled: the trace contains one of the two
          activities but not the other.
        - **NAND** — not fulfilled: every trace that does not satisfy the constraint.
        - **NOR** — not activated: the trace contains neither activity.

        Currently limited to unordered constraints (coexistence, choice, exclusive_choice).

        **Config fields:**
        - `log_name` *(str, default: `"example_log"`)* — name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: system config `storage_namespace_default`)* — storage namespace.
        - `source` *(str)* — source activity of the rule. **Required.**
        - `target` *(str)* — target activity of the rule. **Required.**
        - `template` *(str, optional)* — one of `"coexistence"`, `"choice"`, `"exclusive_choice"`.
            `null` checks every unordered template.
        - `mode` *(str, default: `"XOR"`)* — violation mode: `"XOR"`, `"NAND"`, or `"NOR"`.
        - `output_format` *(str, default: `"json"`)* — `"json"` or `"csv"`.
        """
        logger.info(f"{self.name} is fetching violations via API request: {query_config}")

        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        # Debugging
        if query_config.template is not None and query_config.template not in UNORDERED_TEMPLATES:
            return {"code": 400, "message": f"Invalid template '{query_config.template}'. Violations are only supported for unordered templates: {list(UNORDERED_TEMPLATES)}."}

        mode = (query_config.mode or "XOR").upper()
        if mode not in VIOLATION_MODES:
            return {"code": 400, "message": f"Invalid mode '{query_config.mode}'. Valid options are: {list(VIOLATION_MODES)}."}

        output_format = (query_config.output_format or "json").lower()
        if output_format not in ("json", "csv"):
            return {"code": 400, "message": f"Invalid output_format '{query_config.output_format}'. Valid options are: ['json', 'csv']."}

        if not self.storage.log_exists({"log_name": query_config.log_name, "storage_namespace": query_config.storage_namespace}):
            message = f"Log '{query_config.log_name}' does not exist in namespace '{query_config.storage_namespace}'. Run preprocessing first."
            logger.error(message)
            return {"code": 404, "message": message}

        metadata = MetaData(
            storage_namespace=query_config.storage_namespace,
            log_name=query_config.log_name,
            storage_type=getattr(query_config, "storage_type", "s3"),
        )
        metadata = self.storage.read_metadata_table(metadata)


        # Update mined if it doesn't exist
        mined_now = self._mine_missing_categories(query_config, metadata, ["unordered"])
        if mined_now:
            metadata = self.metadata

        # Apothikevontai alphabetically
        source, target = sorted([query_config.source, query_config.target])

        requested_templates = [query_config.template] if query_config.template else list(UNORDERED_TEMPLATES)
        violations = self._collect_violations(metadata, source, target, requested_templates, mode)
        all_trace_ids = sorted({tid for v in violations for tid in v["violating_trace_ids"]})

        logger.info(
            f"Found {len(all_trace_ids)} distinct violating trace(s) across {len(violations)} template(s) for "
            f"source '{source}' target '{target}' mode '{mode}' of log '{query_config.log_name}'."
        )


        # Make output
        if output_format == "csv":
            columns = ["template", "source", "target", "mode", "violation_count", "violating_trace_ids"]
            download_name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{query_config.log_name}_{source}_{target}_violations")
            with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv", prefix=download_name + "_", delete=False) as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for v in violations:
                    row = {col: v.get(col) for col in columns}
                    row["mode"] = mode
                    row["violating_trace_ids"] = "|".join(v["violating_trace_ids"])
                    writer.writerow(row)
                output_path = f.name
            logger.info(f"Violations written to {output_path}.")
            return FileResponse(
                output_path,
                media_type="text/csv",
                filename=download_name + ".csv",
                background=BackgroundTask(self._discard_output_file, output_path),
            )

        return {
            "code": 200,
            "log_name": query_config.log_name,
            "storage_namespace": query_config.storage_namespace,
            "source": source,
            "target": target,
            "mode": mode,
            "mined_now": mined_now,
            "violation_count": len(all_trace_ids),
            "trace_ids": all_trace_ids,
            "violations": violations,
        }


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
        self.mine(caller="cli")
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

        for metric in ("support", "confidence"):
            low = self.mining_config.get(f"{metric}_threshold", 0.0)
            high = self.mining_config.get(f"max_{metric}_threshold")
            if high is not None and high < low:
                raise ValueError(f"max_{metric}_threshold ({high}) is below {metric}_threshold ({low}).")

        # If the caller left output_path at its default, derive it from log_name instead.
        if config.get("output_path") is None or config.get("output_path") == "output/example_log":
            config["output_path"] = "output/" + config.get("log_name", "mining_results")

        # Ensure output_path is unique for each run to avoid overwriting results
        given_output_path = config.get("output_path")
        Path(given_output_path).parent.mkdir(parents=True, exist_ok=True)
        self.mining_config["output_path"] = given_output_path + "_" + str(datetime.datetime.now().timestamp()) + ".csv"


    @staticmethod
    def _resolve_constraint_scope(category: str | None, template: str | None) -> tuple[list[str], str | None]:
        """
        Validate a category/template pair and resolve which stored categories have to be read.

        :param category: requested category, "*" or None for all of them.
        :param template: requested declare template, or None for all templates of the category.
        :return: the categories to read and the template to filter on (None for all).
        :raises ValueError: if either value is unknown, or if the template is not part of the category.
        """
        if category == "*":
            category = None

        if category is not None and category not in CATEGORY_TEMPLATES:
            raise ValueError(f"Invalid category '{category}'. Valid options are: {sorted(CATEGORY_TEMPLATES)}.")

        if template is not None and template not in TEMPLATE_CATEGORY:
            raise ValueError(f"Invalid template '{template}'. Valid options are: {sorted(TEMPLATE_CATEGORY)}.")

        if category is not None and template is not None and TEMPLATE_CATEGORY[template] != category:
            raise ValueError(
                f"Template '{template}' belongs to category '{TEMPLATE_CATEGORY[template]}', not '{category}'."
            )

        # A template already pins down its category, so only that one table needs reading.
        if template is not None:
            return [TEMPLATE_CATEGORY[template]], template
        if category is not None:
            return [category], None
        return list(CONSTRAINT_CATEGORIES), None


    def _mine_missing_categories(self, query_config: ConstraintQueryConfig | RuleTraceQueryConfig | ViolationQueryConfig, metadata: MetaData, categories: list[str]) -> list[str]:
        """
        Mine the requested categories that hold no constraints in storage yet.

        :param query_config: the constraint query being served.
        :param metadata: metadata of the queried log.
        :param categories: the categories the query needs.
        :return: the categories that were missing and have now been mined (empty if nothing was mined).
        """
        missing = [category for category in categories if not self.storage.constraints_exist(metadata, category)]
        if not missing:
            return []

        # Negation is the complement of the coexistence constraints the unordered miner writes,
        # so it can only be mined once those are in storage.
        to_mine = set(missing)
        if "negation" in to_mine and not self.storage.constraints_exist(metadata, "unordered"):
            to_mine.add("unordered")
        mining_categories = [category for category in CONSTRAINT_CATEGORIES if category in to_mine]

        logger.info(
            f"No stored constraints for {missing} in log '{query_config.log_name}'. "
            f"Mining {mining_categories} before answering the query."
        )
        previous_mined_timestamp = metadata.last_mined_timestamp
        self._load_mining_config({
            "log_name": query_config.log_name,
            "storage_namespace": query_config.storage_namespace,
            "categories": mining_categories,
            # Negation constraints are only persisted per trace in trace-list mode, which is
            # what a per-trace lookup needs; the other miners always write per-trace rows.
            "include_trace_lists": "negation" in mining_categories,
            # These categories have never been mined, so the incremental path would only cover
            # traces that evolved since the last mining run of the other categories.
            "force_recompute": True,
        })
        self.mine(caller="api")

        # This query answers from storage, so nothing ever reads the CSV that mining wrote.
        self._discard_output_file(self.mining_config["output_path"])

        # Mining moves last_mined_timestamp forward, which would make a later incremental run
        # skip traces for the categories left untouched here. The categories mined above were
        # recomputed over every trace, so rewinding the timestamp costs them nothing and keeps
        # the log's mining state exactly as this query found it.
        if self.metadata is not None and self.metadata.last_mined_timestamp != previous_mined_timestamp:
            self.metadata.last_mined_timestamp = previous_mined_timestamp
            self.storage.write_metadata_table(self.metadata)

        return missing


    def _collect_trace_constraints(self, metadata: MetaData, categories: list[str], template: str | None, trace_id: str) -> list[Dict[str, Any]]:
        """
        Read the stored constraints of a single trace across the given categories.

        :param metadata: metadata of the queried log.
        :param categories: categories to read, in canonical order.
        :param template: declare template to keep, or None to keep all of them.
        :param trace_id: the trace whose constraints are collected.
        :return: constraint rows as dicts keyed by CONSTRAINT_COLUMNS.
        """
        readers = {
            "positional": self.storage.read_positional_constraints,
            "existential": self.storage.read_existential_constraints,
            "ordered": self.storage.read_ordered_constraints,
            "unordered": self.storage.read_unordered_constraints,
            "negation": self.storage.read_negation_constraints,
        }

        constraints: list[Dict[str, Any]] = []
        for category in categories:
            constraints_df = readers[category](metadata).where(F.col("trace_id") == F.lit(trace_id))

            if category == "existential":
                constraints_df = self._derive_existential_templates(constraints_df)

            if template is not None:
                constraints_df = constraints_df.where(F.col("template") == F.lit(template))

            # Each reader projects only the columns its category carries; pad the rest.
            for col_name in ["target", "occurrences"]:
                if col_name not in constraints_df.columns:
                    constraints_df = constraints_df.withColumn(col_name, F.lit(None).cast("string"))

            constraints_df = constraints_df.select(
                F.lit(category).alias("category"),
                F.col("template"),
                F.col("source"),
                F.col("target"),
                F.col("occurrences"),
                F.col("trace_id"),
            ).orderBy("template", "source", "target")

            constraints.extend(
                {col_name: row[col_name] for col_name in CONSTRAINT_COLUMNS}
                for row in constraints_df.toLocalIterator(prefetchPartitions=True)
            )

        return constraints


    def _collect_rule_traces(self, metadata: MetaData, categories: list[str], template: str | None, source: str, target: str | None) -> list[Dict[str, Any]]:
        """
        Read the stored traces that back a rule, across the given categories.

        The mirror of _collect_trace_constraints: rows are filtered by the rule's source
        and target rather than by a trace, and grouped back into one entry per distinct
        (category, template, source, target, occurrences) rule, each carrying its traces.

        :param metadata: metadata of the queried log.
        :param categories: categories to read, in canonical order.
        :param template: declare template to keep, or None to keep all of them.
        :param source: source activity the rule must carry.
        :param target: target activity the rule must carry, or None to match rules with no target.
        :return: one dict per matched rule, keyed by CONSTRAINT_COLUMNS minus trace_id, plus
                 sorted 'trace_ids' and their 'trace_count'.
        """
        readers = {
            "positional": self.storage.read_positional_constraints,
            "existential": self.storage.read_existential_constraints,
            "ordered": self.storage.read_ordered_constraints,
            "unordered": self.storage.read_unordered_constraints,
            "negation": self.storage.read_negation_constraints,
        }

        rules: Dict[tuple, Dict[str, Any]] = {}
        for category in categories:
            constraints_df = readers[category](metadata)

            if category == "existential":
                constraints_df = self._derive_existential_templates(constraints_df)

            if template is not None:
                constraints_df = constraints_df.where(F.col("template") == F.lit(template))

            # Each reader projects only the columns its category carries; pad the rest so
            # the source/target filter and the projection below are uniform.
            for col_name in ["target", "occurrences"]:
                if col_name not in constraints_df.columns:
                    constraints_df = constraints_df.withColumn(col_name, F.lit(None).cast("string"))

            constraints_df = constraints_df.where(F.col("source") == F.lit(source))
            if target is None:
                constraints_df = constraints_df.where(F.col("target").isNull() | (F.col("target") == F.lit("")))
            else:
                constraints_df = constraints_df.where(F.col("target") == F.lit(target))

            constraints_df = constraints_df.select(
                F.lit(category).alias("category"),
                F.col("template"),
                F.col("source"),
                F.col("target"),
                F.col("occurrences"),
                F.col("trace_id"),
            )

            for row in constraints_df.toLocalIterator(prefetchPartitions=True):
                key = (row["category"], row["template"], row["source"], row["target"], row["occurrences"])
                rule = rules.get(key)
                if rule is None:
                    rule = {
                        "category": row["category"],
                        "template": row["template"],
                        "source": row["source"],
                        "target": row["target"],
                        "occurrences": row["occurrences"],
                        "trace_ids": set(),
                    }
                    rules[key] = rule
                if row["trace_id"] is not None:
                    rule["trace_ids"].add(row["trace_id"])

        result = []
        for rule in rules.values():
            trace_ids = sorted(rule["trace_ids"])
            rule["trace_ids"] = trace_ids
            rule["trace_count"] = len(trace_ids)
            result.append(rule)
        result.sort(key=lambda r: (r["category"], r["template"] or "", r["source"] or "", r["target"] or ""))
        return result


    def _collect_violations(self, metadata: MetaData, source: str, target: str, templates: list[str], mode: str) -> list[Dict[str, Any]]:
        """
        Find traces that violate unordered rules for a given activity pair.

        :param metadata: metadata of the queried log.
        :param source: source activity (alphabetically first).
        :param target: target activity (alphabetically second).
        :param templates: unordered templates whose violations to compute.
        :param mode: one of 'XOR', 'NAND', 'NOR'.
        :return: one dict per template with violation_count and sorted violating_trace_ids.
        """
        activity_index = self.storage.read_activity_index(metadata=metadata)
        all_traces = activity_index.select("trace_id").distinct()
        traces_with_source = activity_index.where(F.col("activity") == F.lit(source)).select("trace_id").distinct()
        traces_with_target = activity_index.where(F.col("activity") == F.lit(target)).select("trace_id").distinct()
        traces_with_both = traces_with_source.intersect(traces_with_target)
        traces_with_either = traces_with_source.union(traces_with_target).distinct()
        traces_with_neither = all_traces.subtract(traces_with_either)
        traces_with_xor = traces_with_either.subtract(traces_with_both)

        violation_map = {
            "coexistence": {
                "XOR": traces_with_xor,
                "NAND": all_traces.subtract(traces_with_both),
                "NOR": traces_with_neither,
            },
            "exclusive_choice": {
                "XOR": traces_with_both,
                "NAND": traces_with_both.union(traces_with_neither),
                "NOR": traces_with_neither,
            },
            "choice": {
                "XOR": None,
                "NAND": traces_with_neither,
                "NOR": traces_with_neither,
            },
        }

        results = []
        for tpl in templates:
            violations_df = violation_map.get(tpl, {}).get(mode)
            if violations_df is None:
                results.append({
                    "template": tpl,
                    "source": source,
                    "target": target,
                    "violation_count": 0,
                    "violating_trace_ids": [],
                })
                continue

            trace_ids = sorted(
                row["trace_id"] for row in violations_df.toLocalIterator(prefetchPartitions=True)
            )
            results.append({
                "template": tpl,
                "source": source,
                "target": target,
                "violation_count": len(trace_ids),
                "violating_trace_ids": trace_ids,
            })

        return results


    @staticmethod
    def _derive_existential_templates(constraints_df: DataFrame) -> DataFrame:
        """
        Add the Existence and Absence rows that the miner derives from the stored Exactly rows.

        Only Exactly constraints are persisted, since Existence(a,n) holds for the mined n and
        Absence(a,n) holds for n+1 - the same derivation discover_existential applies.

        :param constraints_df: stored existential constraints of a trace.
        :return: the input rows plus their derived Existence and Absence counterparts.
        """
        exactly_df = constraints_df.where(F.col("template") == F.lit("exactly"))
        existence_df = exactly_df.withColumn("template", F.lit("existence"))
        absence_df = exactly_df.withColumn("template", F.lit("absence")) \
            .withColumn("occurrences", F.col("occurrences") + 1)
        return constraints_df.unionByName(existence_df).unionByName(absence_df)


    @staticmethod
    def _discard_output_file(output_path: str) -> None:
        """
        Delete a result file that only existed to build an API response.

        API callers receive the results in the response itself, so keeping the file would
        just pile up unread CSVs on the driver. CLI runs never call this: they are handed
        the path of their file and keep it.

        :param output_path: path of the file to remove.
        """
        try:
            Path(output_path).unlink(missing_ok=True)
            logger.info(f"Discarded API output file {output_path}.")
        except OSError as e:
            logger.warning(f"Could not remove output file {output_path}: {e}")


    def mine(self, caller: str):
        """
        Permorms incremental mining on the log data based on the provided mining configuration and metadata.
        The method loads evolved traces since the last mining, discovers new constraints and keeps only valid 
        old ones and new ones in storage (by overwrite mode), and outputs the results to a CSV file on the driver's local filesystem.

        :param caller: a string indicating the caller of the mining process (e.g. "cli", "api") for logging purposes.
        """

        logger.info(f"Beginning mining process initiated by {caller}.")

        # Load metadata if available, and evolved traces since last mining from storage
        self.metadata = MetaData(
            storage_namespace=self.mining_config.get("storage_namespace", get_config_value("storage_namespace_default", "siesta")),
            log_name=self.mining_config.get("log_name", "default_log"),
            storage_type=self.mining_config.get("storage_type", "s3")
        )

        self.metadata = self.storage.read_metadata_table(self.metadata) 
        evolved_df = self.storage.read_sequence_table(self.metadata, filter_out="mined" if not self.mining_config.get("force_recompute", False) else None)
        evolved_df.cache()  # Cache evolved traces as they will be used multiple times during mining

        # Perform mining based on the specified categories in the mining configuration. 
        # Each miner function returns a DataFrame with a common schema, and we union them together 
        # while adding a "category" column to identify the source of each constraint.
        miners = []
        include_trace_lists = self.mining_config.get("include_trace_lists", False)
        for category in self.mining_config["categories"]:
            if category in ["positional", "*"]:
                miners.append(("positional", discover_positional))
            if category in ["existential", "*"]:
                miners.append(("existential", discover_existential))
            if category in ["ordered", "*"]:
                miners.append(("ordered", discover_ordered))
            if category in ["unordered", "*"]:
                miners.append(("unordered", discover_unordered))
            if category in ["negation", "*"]:
                miners.append(("negation", lambda e, m: discover_negations(e, m, include_trace_lists)))
        
        raw_miner_results = []
        constraints_df_list = []
        for category, miner_func in miners:
            result = miner_func(evolved_df, self.metadata)
            raw_miner_results.append(result)
            constraints_df_list.append(result.withColumn("category", F.lit(category)))

        constraints_df = constraints_df_list[0]
        for constaint_df in constraints_df_list[1:]:
            constraints_df = constraints_df.unionByName(constaint_df, allowMissingColumns=True)

        for col_name in ["target", "occurrences"]:
            if col_name not in constraints_df.columns:
                constraints_df = constraints_df.withColumn(col_name, F.lit(None).cast("string"))

        # Update metadata with new last mining timestamp based on the max timestamp of the evolved traces
        self.metadata.last_mined_timestamp = evolved_df.agg({"start_timestamp": "max"}).collect()[0][0] if not evolved_df.rdd.isEmpty() else self.metadata.last_mined_timestamp
        evolved_df.unpersist()
        self.storage.write_metadata_table(self.metadata)

        # Output the discovered constraints to a CSV file on the driver's local filesystem 
        # based on the specified output path in the mining configuration.
        self._output_constraints(constraints_df, self.metadata.trace_count)

        # Release cached miner results now that output is written
        for df in raw_miner_results:
            df.unpersist(blocking=False)

        # Release Delta metadata and any remaining cached data
        spark_cleanup()


    def _output_constraints(self, constraints_df: DataFrame, trace_count: int):
        """
        Outputs the discovered constraints to a CSV file on the driver's local filesystem 
        based on the specified output path in the mining configuration.
        This method collects the results from the Spark executors and writes them incrementally to avoid driver memory issues.
        
        :param constraints: DataFrame based on ConstraintEntry schema (template, source, target, occurrences, trace_id)
        """
        # Aggregate trace_ids for the same (template, source, target, occurrences) tuples
        # Handle pre-aggregated constraints (support-only negation mode) separately
        has_precomputed = "_support_count" in constraints_df.columns

        if has_precomputed:
            precomputed = constraints_df.filter(F.col("_support_count").isNotNull())
            trace_level = constraints_df.filter(F.col("_support_count").isNull()).drop("_support_count")
        else:
            trace_level = constraints_df
            precomputed = None

        grouped_constraints = trace_level.groupBy(
            "category", "template", "source", "target", "occurrences"
        ).agg(
            F.collect_list("trace_id").alias("trace_ids")
        )

        # Track the actual match count separately from trace_ids, since precomputed
        # negation rows have no trace_id list (only an aggregate support count).
        grouped_constraints = grouped_constraints.withColumn(
            "match_count", F.size(F.col("trace_ids"))
        )

        # Calculate support: len(trace_ids) / trace_count
        grouped_constraints = grouped_constraints.withColumn(
            "support",
            (F.col("match_count") / F.lit(trace_count))
        )

        if precomputed is not None:
            grouped_pre = precomputed.select(
                F.col("category"), F.col("template"), F.col("source"),
                F.col("target"), F.col("occurrences"),
                (F.col("_support_count") / F.lit(trace_count)).alias("support"),
                F.array().cast("array<string>").alias("trace_ids"),
                F.col("_support_count").alias("match_count"),
            )
            grouped_constraints = grouped_constraints.unionByName(grouped_pre)

        grouped_constraints = grouped_constraints.filter(F.col("support") >= self.mining_config.get("support_threshold", 0.0))
        max_support = self.mining_config.get("max_support_threshold")
        if max_support is not None:
            grouped_constraints = grouped_constraints.filter(F.col("support") <= max_support)

        # Calculate confidence:
        #   for ordered: #traces_ab / #traces_a                          (P(target|source))
        #   for unordered: #traces_ab / sqrt(#traces_a * #traces_b)      (geometric mean of both directed confidences)
        #   for negation: match_count is the *complement* of coexistence (N - #traces_ab), so the
        #     coexistence count must be recovered as N - match_count before reusing the directed
        #     formulas, applied to the negated events P(not target|source) and P(not source|target).
        activity_counts = self.storage.read_activity_index(metadata=self.metadata).groupBy("activity").agg(F.count_distinct("trace_id").alias("activity_trace_count"))
        grouped_constraints = grouped_constraints.join(
            activity_counts.withColumnRenamed("activity", "source").withColumnRenamed("activity_trace_count", "source_trace_count"),
            on="source",
            how="left"
        ).join(
            activity_counts.withColumnRenamed("activity", "target").withColumnRenamed("activity_trace_count", "target_trace_count"),
            on="target",
            how="left"
        )

        coexist_count = F.lit(trace_count) - F.col("match_count")
        negation_confidence = F.sqrt(
            ((F.col("source_trace_count") - coexist_count) / F.col("source_trace_count"))
            * ((F.col("target_trace_count") - coexist_count) / F.col("target_trace_count"))
        )

        grouped_constraints = grouped_constraints.withColumn(
            "confidence",
            F.when(
                F.col("category") == "unordered",
                F.col("match_count") / F.sqrt(F.col("source_trace_count") * F.col("target_trace_count"))
            ).when(
                F.col("category") == "negation",
                negation_confidence
            ).when(
                (F.col("category") == "ordered") | (F.col("category") == "positional") | (F.col("category") == "existential"),
                F.col("match_count") / F.col("source_trace_count")
            )
            .otherwise(F.lit(None))
        )

        grouped_constraints = grouped_constraints.filter(F.col("confidence") >= self.mining_config.get("confidence_threshold", 0.0))
        max_confidence = self.mining_config.get("max_confidence_threshold")
        if max_confidence is not None:
            grouped_constraints = grouped_constraints.filter(F.col("confidence") <= max_confidence)

        # Calculate interest = support(rule) / (expected support under independence).
        # For positive-event categories the independence baseline is P(source) * P(target).
        # For negation the mined event is "source and target do NOT coexist", whose independence
        # baseline is 1 - P(source) * P(target), not P(source) * P(target).
        independence_support = (F.col("source_trace_count") / F.lit(trace_count)) * (F.col("target_trace_count") / F.lit(trace_count))
        expected_support = F.when(F.col("category") == "negation", F.lit(1.0) - independence_support).otherwise(independence_support)
        grouped_constraints = grouped_constraints.withColumn(
            "interest",
            F.when(
                (F.col("source_trace_count") != 0) & (F.col("target_trace_count") != 0) & (expected_support != 0),
                F.col("support") / expected_support
            )
            .otherwise(F.lit(None))
        )

        grouped_constraints = grouped_constraints.filter(F.col("interest") >= self.mining_config.get("interest_threshold", 0.0))

        # Prepare a CSV-friendly DataFrame
        select_cols = [
            F.col("category"),
            F.col("template"),
            F.col("source"),
            F.col("target"),
            F.col("occurrences").cast("string"),
            F.col("support").cast("string"),
            F.col("confidence").cast("string"),
            F.col("interest").cast("string"),
        ]
        
        # Optionally include the list of trace_ids supporting each constraint, serialized as a pipe-delimited string. 
        if self.mining_config.get("include_trace_lists", False):
            select_cols.append(F.concat_ws("|", F.col("trace_ids")).alias("trace_ids"))
        
        constraints_csv = grouped_constraints.select(*select_cols)

        # Stream rows partition-by-partition from executors to the driver and write
        # them incrementally into a single CSV file on the driver's local filesystem.
        output_path = self.mining_config["output_path"]
        col_names = constraints_csv.columns

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            for row in constraints_csv.toLocalIterator(prefetchPartitions=True):
                writer.writerow([row[c] for c in col_names])
