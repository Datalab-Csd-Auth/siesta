import argparse
import csv
import datetime
import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict
from fastapi import Body
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import SparkSession

from siesta.core.config import get_system_config
from siesta.core.interfaces import SiestaModule, StorageManager
from siesta.core.storageFactory import get_storage_manager
from siesta.model.StorageModel import MetaData

from siesta.modules.analyser.attribute_deviations import ALL_STEPS
from siesta.modules.analyser.processing import (
    run_directly_follows, run_loop_detection, write_loop_detection_json, run_durations,
    run_attribute_deviations, write_attribute_deviations_json, run_ngrams, run_rare_rules,
    run_targeted_rules, run_process_model, run_bottlenecks, run_temporal_deviations,
)


class DirectlyFollowsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field("siesta", description="Storage namespace")
    min_timestamp: str | None = Field(None, description="Lower bound on start_timestamp as ISO 8601 datetime with millisecond precision (e.g. '2024-01-15T10:30:45.123Z')")
    end_time: str | None = Field(None, description="Attribute key for event end timestamp. null = transition time (next_start - start)")
    support_threshold: float | None = Field(None, description="Min support fraction [0,1]; null = no filtering")
    filter_out: bool = Field(False, description="When true, keeps pairs with support ≤ threshold instead")
    include_traces: bool = Field(False, description="Append a trace_ids column listing traces that contain each pair")
    return_csv: bool = Field(False, description="Return a CSV file download instead of a JSON list")
    output_path: str = Field("output/example_log", description="Local path prefix for the output file")


class LoopDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field("siesta", description="Storage namespace")
    grouping_key: str | list[str] | None = Field(None, description="Attribute key(s) to group by; null = trace_id")
    grouping_value: str | list[str] | dict | None = Field(None, description="Restrict to groups with matching key value(s)")
    min_timestamp: str | None = Field(None, description="Lower bound on start_timestamp as ISO 8601 datetime with millisecond precision (e.g. '2024-01-15T10:30:45.123Z')")
    support_threshold: float | None = Field(None, description="Min support fraction [0,1]; null = no filtering")
    filter_out: bool = Field(False, description="When true, keeps rare loops (support ≤ threshold)")
    top_k: int | None = Field(None, description="Keep only the k most-supported loops; null = all")
    trace_based: bool = Field(False, description="Add trace_ids list to each loop entry (only when grouping by trace_id)")
    max_pattern_length: int | str = Field(8, description="Longest repeated-pattern block to search for: an int length, or \"auto\" to derive clamp(round(avg_group_length / 2), 2, 8) from the data")
    include_repeats: bool = Field(True, description="When false, skip the repeated-pattern search (self-loops and non-self-loops are always detected)")
    with_time_stats: bool = Field(True, description="Include per-pattern occurrence/cost statistics (avg/median events, time, and their share of the group's length/span)")
    with_distribution: bool = Field(False, description="Additionally emit the spread of per-trace time consumed by each pattern (min/p10/p25/p50/p75/p90/max/stddev), in absolute seconds and as a share of the trace; requires with_time_stats")
    output_path: str = Field("output/example_log", description="Local path prefix for the output file")


class DurationsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field("siesta", description="Storage namespace")
    min_timestamp: str | None = Field(None, description="Lower bound on start_timestamp as ISO 8601 datetime with millisecond precision (e.g. '2024-01-15T10:30:45.123Z')")
    duration_mode: str = Field("activity", description="'activity' (per activity type) or 'group' (per group instance)")
    end_time: str | None = Field(None, description="Attribute key for event end timestamp. null = transition / span time")
    grouping_key: str | list[str] | None = Field(None, description="Attribute key(s) defining groups; null = trace_id")
    grouping_value: str | list[str] | None = Field(None, description="Restrict to groups with matching key value(s)")
    per_group: bool = Field(False, description="Activity mode only: produce one row per (group, activity) instead of globally")
    return_csv: bool = Field(False, description="Return a CSV file download instead of a JSON list")
    output_path: str = Field("output/example_log", description="Local path prefix for the output file")


class AttributeDeviationsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log. **Required.**")
    storage_namespace: str = Field("siesta", description="Storage namespace.")
    min_timestamp: str | None = Field(None, description="Lower bound on start_timestamp as ISO 8601 datetime with millisecond precision (e.g. '2024-01-15T10:30:45.123Z')")
    steps: list[int] = Field(
        list(ALL_STEPS),
        description=(
            "Which analysis steps to run. Subset of [0,1,2,3,4]. "
            "0=value frequency (inter+intra-trace), 1=activity×attribute, "
            "2=position-conditioned, 3=n-gram context, 4=value transitions."
        ),
    )
    excluded_attributes: list[str] | None = Field(
        None,
        description="Attribute keys to skip in addition to auto-detected timestamp keys.",
    )
    surprise_threshold: float = Field(
        4.0,
        description="Anomaly threshold for categorical attributes (−log₂ score). Default 4.0 ≈ support ≤ 6.25 %.",
    )
    zscore_threshold: float = Field(
        3.5,
        description="Anomaly threshold for numeric attributes (|robust z-score| via MAD).",
    )
    ngram_n: int = Field(2, description="N-gram length for step 3 (window of activities ending at the event).")
    min_group_size: int = Field(5, description="Minimum observations required to score a group in steps 3 and 4.")
    n_buckets: int = Field(5, description="Number of relative-position buckets for step 2.")
    support_threshold: float | None = Field(
        None,
        description=(
            "Filter output by inter-trace support of the flagged value [0,1]. "
            "null = no filter. Combined with filter_out."
        ),
    )
    filter_out: bool = Field(
        False,
        description="When true, keep deviations where support ≤ threshold (rare values). "
                    "When false (default), keep deviations where support ≥ threshold.",
    )
    on_rare: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Support threshold [0,1] used to enable rare-mode. When set, discover ordered constraints "
            "and run deviations only on traces that violate at least one ordered constraint with "
            "support >= on_rare. null disables rare-mode."
        ),
    )
    output_format: str = Field("json", description="Output format: 'json' (default), 'csv', or 'html'.")
    output_path: str = Field("output/example_log", description="Local path prefix for csv/html output files.")


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field("siesta", description="Storage namespace")
    method_params: dict = Field(default_factory=lambda: {"n": 2}, description="Method-specific params. ngrams: {n, vis}. targeted_rules: {target_label, filtering_support}")
    separating_key: str = Field("activity", description="Column or attribute key used to label traces into groups")
    separating_groups: list[list[str]] = Field(default_factory=list, description="Group definitions, e.g. [['fail','error']] splits into listed values vs. all others")
    support_threshold: float = Field(0.0, description="Minimum support fraction [0,1] for results")
    output_path: str = Field("output/example_log", description="Local path prefix for the output file")


class ProcessModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field("siesta", description="Storage namespace")
    min_timestamp: str | None = Field(None, description="Lower bound on start_timestamp as ISO 8601 datetime with millisecond precision (e.g. '2024-01-15T10:30:45.123Z')")
    end_time: str | None = Field(None, description="Attribute key for event end timestamp, used only to annotate visualizations with average per-activity durations. null = no duration annotation")
    noise_threshold: float = Field(0.0, description="Fraction of infrequent DFG paths to prune before discovery (0.0 = keep all, 1.0 = keep only the most frequent path)")
    output_format: str = Field("model", description="File returned by the endpoint: 'model' (native file - .xml/.bpmn/.pnml), 'png' (process map image), or 'html' (interactive graph)")
    output_path: str = Field("output/example_log", description="Local path prefix for the output file")


class BottlenecksConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field("siesta", description="Storage namespace")
    min_timestamp: str | None = Field(None, description="Lower bound on start_timestamp as ISO 8601 datetime with millisecond precision (e.g. '2024-01-15T10:30:45.123Z')")
    end_time: str | None = Field(None, description="Attribute key for event end timestamp. null = transition time (next_start - start)")
    grouping_key: str | list[str] | None = Field(None, description="Attribute key(s) defining what counts as 'adjacent' events; null = trace_id")
    grouping_value: str | list[str] | None = Field(None, description="Restrict to groups with matching key value(s)")
    zscore_threshold: float = Field(3.5, description="Robust (MAD-based) z-score threshold above which a pair's average duration is flagged as anomalous relative to the rest of the process (independent of support_threshold/sigma_multiplier/is_bottleneck; kept for backward compatibility)")
    top_k: int | None = Field(None, description="Keep only the k pairs ranked first by order_by; null = all")
    return_csv: bool = Field(False, description="Return a CSV file download instead of a JSON list")
    support_threshold: float = Field(0.0, description="Minimum fraction of groups a pair must occur in to count as 'common'; pairs below this are dropped entirely. Default 0.0 keeps every pair")
    sigma_multiplier: float = Field(3.0, description="How many standard deviations above the mean avg_duration_sec (over the support_threshold-filtered population) a pair must exceed to be flagged is_bottleneck")
    order_by: str = Field("impact_score", description="Column to rank/truncate by: 'impact_score' (default), 'avg_duration_sec', or 'median_duration_sec'")
    output_path: str = Field("output/example_log", description="Local path prefix for the output file")


class TemporalDeviationsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field("siesta", description="Storage namespace")
    min_timestamp: str | None = Field(None, description="Lower bound on start_timestamp as ISO 8601 datetime with millisecond precision (e.g. '2024-01-15T10:30:45.123Z')")
    separating_key: str = Field("activity", description="Column or attribute key used to label traces into groups (e.g. a 'decision' attribute)")
    separating_groups: list[list[str]] = Field(default_factory=list, description="Group definitions, e.g. [['rejected']] splits into listed values (label 1) vs. all others (label 0)")
    activity_pairs: list[list[str]] | None = Field(None, description="Explicit [source, target] activity pairs to check for duration-based discrimination. null = auto-derive the top max_auto_pairs most frequent pairs")
    max_auto_pairs: int = Field(50, description="Upper bound on auto-derived candidate pairs when activity_pairs is null")
    min_group_size: int = Field(5, description="Minimum combined (count_1 + count_0) required to keep a (pair, threshold) result")
    top_k_per_pair: int | None = Field(None, description="Keep only the k best (by |balance|) thresholds per activity pair; null = keep all decile candidates")
    return_csv: bool = Field(False, description="Return a CSV file download instead of a JSON list")
    output_path: str = Field("output/example_log", description="Local path prefix for the output file")


import logging

logger = logging.getLogger(__name__)


DEFAULT_ANALYSER_CONFIG: Dict[str, Any] = {
    **DirectlyFollowsConfig().model_dump(),
    **LoopDetectionConfig().model_dump(),
    **DurationsConfig().model_dump(),
    **AttributeDeviationsConfig().model_dump(),
    **ComparisonConfig().model_dump(),
    **ProcessModelConfig().model_dump(),
    **BottlenecksConfig().model_dump(),
    **TemporalDeviationsConfig().model_dump(),
    "method": "directly_follows"
}


class Analyser(SiestaModule):

    name = "analyser"
    version = "1.0.0"
    spark: SparkSession
    storage: StorageManager
    siesta_config: Dict[str, Any]
    analyser_config: Dict[str, Any]
    metadata: MetaData | None

    def __init__(self):
        super().__init__()
        self.analyser_config = {}
        self.metadata = None

    def startup(self):
        logger.info("Analyser startup complete.")

    def register_routes(self) -> SiestaModule.ApiRoutes | None:
        return {
            "directly_follows":      ("POST", self.api_directly_follows),
            "loop_detection":        ("POST", self.api_loop_detection),
            "durations":             ("POST", self.api_durations),
            "attribute_deviations":  ("POST", self.api_attribute_deviations),
            "ngrams":                ("POST", self.api_ngrams),
            "rare_rules":            ("POST", self.api_rare_rules),
            "targeted_rules":        ("POST", self.api_targeted_rules),
            "dfg_model":             ("POST", self.api_dfg_model),
            "bpmn":                  ("POST", self.api_bpmn),
            "petri_net":             ("POST", self.api_petri_net),
            "bottlenecks":           ("POST", self.api_bottlenecks),
            "temporal_deviations":   ("POST", self.api_temporal_deviations),
        }

    # ------------------------------------------------------------------
    # API entry points - core analysis (unchanged from the former `analyse` module)
    # ------------------------------------------------------------------

    def api_directly_follows(self, analyser_config: Annotated[DirectlyFollowsConfig, Body(openapi_examples={
        "default": {
            "summary": "Find directly-following pairs with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "min_timestamp": None,
                "end_time": None,
                "support_threshold": None,
                "filter_out": False,
                "include_traces": False,
                "return_csv": False,
            },
        },
    })]) -> Any:
        """Find directly-following activity pairs in an indexed event log.

        Returns pairs of consecutive activities with support (fraction of traces where A
        is directly followed by B) and duration statistics in seconds.

        **Config fields:**
        - `log_name` *(str)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp` as ISO 8601 with millisecond precision.
        - `end_time` *(str | null, default: `null`)* - attribute key for event end timestamp.
            If set, duration = `end_time - start_timestamp` (activity duration).
            If null, duration = `next_start - start_timestamp` (transition time).
        - `support_threshold` *(float [0,1] | null, default: `null`)* - keep pairs with support ≥ threshold. `null` = no filtering.
        - `filter_out` *(bool, default: `false`)* - when `true`, keeps pairs with support ≤ threshold instead.
        - `include_traces` *(bool, default: `false`)* - append a `trace_ids` column to the output.
        - `return_csv` *(bool, default: `false`)* - return a CSV file download instead of a JSON list.
            """
        logger.info(f"{self.name} running directly_follows via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "directly_follows"
        try:
            self._load_analyser_config(config)
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        return self._csv_response(run_directly_follows(self.analyser_config, self.storage), self.analyser_config.get("return_csv", False))

    def api_durations(self, analyser_config: Annotated[DurationsConfig, Body(openapi_examples={
        "default": {
            "summary": "Compute duration statistics with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "min_timestamp": None,
                "duration_mode": "activity",
                "end_time": None,
                "grouping_key": None,
                "grouping_value": None,
                "per_group": False,
                "return_csv": False,
            },
        },
    })]) -> Any:
        """Compute duration statistics for activities or groups in an indexed event log.

        Two modes controlled by `duration_mode`:
        - **`activity`**: avg / min / max duration per activity type; optionally per group when `per_group=true`.
        - **`group`**: total duration per group instance (e.g. per trace).

        **Config fields:**
        - `log_name` *(str)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp`.
        - `duration_mode` *(str, default: `"activity"`)* - `"activity"` or `"group"`.
        - `end_time` *(str | null, default: `null`)* - attribute key for event end timestamp.
            Activity mode: `end_time - start_timestamp`; group mode: sum of per-event durations.
            If null - activity mode uses transition time; group mode uses `last_start - first_start`.
        - `grouping_key` *(str | list | null, default: `null`)* - attribute key(s) defining groups. `null` = `trace_id`.
        - `grouping_value` *(str | list | null, default: `null`)* - restrict to groups with matching key value(s).
        - `per_group` *(bool, default: `false`)* - `activity` mode only: produce one row per (group, activity).
        - `return_csv` *(bool, default: `false`)* - return a CSV file download instead of a JSON list.
        """
        logger.info(f"{self.name} running durations via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "durations"
        try:
            self._load_analyser_config(config)
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        return self._csv_response(run_durations(self.analyser_config, self.storage), self.analyser_config.get("return_csv", False))

    def api_loop_detection(self, analyser_config: Annotated[LoopDetectionConfig, Body(openapi_examples={
        "default": {
            "summary": "Detect loops with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "grouping_key": None,
                "grouping_value": None,
                "min_timestamp": None,
                "support_threshold": None,
                "filter_out": False,
                "top_k": None,
                "trace_based": False,
                "max_pattern_length": 8,
                "include_repeats": True,
                "with_time_stats": True,
                "with_distribution": False,
            },
        },
    })]) -> Any:
        """Detect self-loops, non-self-loops and repeated patterns in an indexed event log.

        A **self-loop** is an activity immediately followed by itself.
        A **non-self-loop** is a minimal cycle A -> … -> A where A does not appear in the body.
        A **repeated pattern** is any contiguous block of 2..max_pattern_length activities
        (e.g. A -> B -> C) that recurs verbatim, non-overlapping, at least twice in a trace;
        only maximal patterns are reported (a shorter pattern fully explained by a longer,
        equally-recurring one is omitted).

        Returns JSON with `self_loops`, `non_self_loops` and `repeated_patterns` arrays. Each
        entry contains the activity pattern, its support fraction across groups, and - when
        `with_time_stats` is true - how much of the trace it typically consumes: average/median
        occurrence count, events consumed (absolute and as a share of the trace length), and
        time consumed (absolute seconds and as a share of the trace's span).

        **Config fields:**
        - `log_name` *(str)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `grouping_key` *(str | list | null, default: `null`)* - attribute key(s) to group by. `null` = `trace_id`.
        - `grouping_value` *(str | list | dict | null, default: `null`)* - restrict to specific group values.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp`.
        - `support_threshold` *(float [0,1] | null, default: `null`)* - keep loops with support ≥ threshold. `null` = no filtering.
        - `filter_out` *(bool, default: `false`)* - when `true`, keeps loops with support ≤ threshold (rare loops).
        - `top_k` *(int | null, default: `null`)* - keep only the k most-supported loops across all three types. `null` = all.
        - `trace_based` *(bool, default: `false`)* - add a `trace_ids` list to each loop entry (only when grouping by `trace_id`).
        - `max_pattern_length` *(int | "auto", default: `8`)* - longest repeated-pattern block to search for.
            `"auto"` derives `clamp(round(avg_group_length / 2), 2, 8)` from the data.
        - `include_repeats` *(bool, default: `true`)* - when `false`, skip the repeated-pattern search.
        - `with_time_stats` *(bool, default: `true`)* - include the occurrence/cost statistics above.
        - `with_distribution` *(bool, default: `false`)* - additionally emit `time_distribution_sec` and
            `pct_trace_time_distribution` per entry: the min/p10/p25/p50/p75/p90/max/stddev spread of the
            per-trace time each pattern consumes, in absolute seconds and as a share of the trace. Requires
            `with_time_stats`.
        """
        logger.info(f"{self.name} running loop_detection via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "loop_detection"
        try:
            self._load_analyser_config(config)
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        return {"code": 200, **run_loop_detection(self.analyser_config, self.storage)}

    def api_attribute_deviations(self, analyser_config: Annotated[AttributeDeviationsConfig, Body(openapi_examples={
        "default": {
            "summary": "Detect attribute deviations with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "min_timestamp": None,
                "steps": [0, 1, 2, 3, 4],
                "excluded_attributes": None,
                "surprise_threshold": 4.0,
                "zscore_threshold": 3.5,
                "ngram_n": 2,
                "min_group_size": 5,
                "n_buckets": 5,
                "support_threshold": None,
                "filter_out": False,
                "on_rare": None,
                "output_format": "json",
            },
        },
    })]) -> Any:
        """Detect anomalous attribute values in an indexed event log using a multi-step pipeline.

        Five steps (all on by default; select via `steps`):
        - **Step 0** - Value frequency: flags values that are globally rare across traces
            (`value_freq_inter`) or appear unusually often within a single trace (`value_freq_intra`).
        - **Step 1** - Activity × Attribute: flags values whose distribution within an activity type
            is anomalous (Laplace surprise for categorical, MAD z-score for numeric).
        - **Step 2** - Position-conditioned: same as step 1 but conditioned on relative position
            within the trace (bucketed).
        - **Step 3** - N-gram context: same as step 1 but conditioned on the n-gram of activities
            ending at this event.
        - **Step 4** - Value transitions: flags rare (prev_value -> curr_value) transitions
            within a trace (categorical attributes only).

        **Config fields:**
        - `log_name` *(str)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp`.
        - `steps` *(list[int], default: `[0,1,2,3,4]`)* - which steps to run.
        - `excluded_attributes` *(list[str] | null)* - attribute keys to skip (auto-excludes timestamp keys).
        - `surprise_threshold` *(float, default: `4.0`)* - categorical anomaly threshold (−log₂ score).
        - `zscore_threshold` *(float, default: `3.5`)* - numeric anomaly threshold (|robust z-score|).
        - `ngram_n` *(int, default: `2`)* - n-gram length for step 3.
        - `min_group_size` *(int, default: `5`)* - minimum group size for steps 3 and 4.
        - `n_buckets` *(int, default: `5`)* - position buckets for step 2.
        - `support_threshold` *(float | null)* - filter output by inter-trace support [0,1].
        - `filter_out` *(bool, default: `false`)* - when true, keep deviations with support ≤ threshold.
        - `on_rare` *(float [0,1] | null, default: `null`)* - rare-mode threshold.
                When set, run analysis only on traces violating at least one ordered constraint
                whose support is ≥ `on_rare`.
        - `output_format` *(str, default: `"json"`)* - `"json"`, `"csv"`, or `"html"`.
        """
        logger.info(f"{self.name} running attribute_deviations via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "attribute_deviations"
        try:
            self._load_analyser_config(config)
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        result, output_path = run_attribute_deviations(self.analyser_config, self.storage)
        output_format = self.analyser_config.get("output_format", "json")
        if output_format == "csv":
            return FileResponse(output_path, media_type="text/csv", filename=Path(output_path).name)
        if output_format == "html":
            return FileResponse(output_path, media_type="text/html", filename=Path(output_path).name)
        return result

    # ------------------------------------------------------------------
    # API entry points - comparison (from the former `compare` module)
    # ------------------------------------------------------------------

    def api_ngrams(self, analyser_config: Annotated[ComparisonConfig, Body(
        openapi_examples={
            "default": {
                "summary": "Compare n-gram frequencies with default settings",
                "value": {
                    "log_name": "example_log",
                    "storage_namespace": "siesta",
                    "method_params": {"n": 2},
                    "separating_key": "activity",
                    "separating_groups": [],
                    "support_threshold": 0.0,
                },
            },
        }
    )]) -> Any:
        """Compare n-gram frequency distributions between two trace groups.

        Splits traces into two groups based on `separating_key`/`separating_groups` and
        computes n-gram frequency differences (balance/confidence/support statistics).

        **Config fields:**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log.
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `method_params` *(object, default: `{"n": 2}`)* - `n` (int) = gram length; `vis` (bool) = generate HTML network.
        - `separating_key` *(str, default: `"activity"`)* - column or attribute key used to label traces into groups.
        - `separating_groups` *(list[list[str]])* - group definitions, e.g. `[["fail", "error"]]`.
        - `support_threshold` *(float [0,1], default: `0.0`)* - minimum support fraction for results.
        """
        logger.info(f"{self.name} running ngrams via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "ngrams"
        try:
            self._load_analyser_config(config)
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        return self._csv_response(run_ngrams(self.analyser_config, self.storage), return_csv=False)

    def api_rare_rules(self, analyser_config: Annotated[ComparisonConfig, Body(
        openapi_examples={
            "default": {
                "summary": "Discover rare rules with default settings",
                "value": {
                    "log_name": "example_log",
                    "storage_namespace": "siesta",
                    "separating_key": "activity",
                    "separating_groups": [],
                    "support_threshold": 0.1,
                },
            },
        }
    )]) -> Any:
        """Discover directly-following rules that are rare overall but appear in target traces.

        Compares ordered constraints across the two trace groups and returns rules
        whose support is low globally (≤ `support_threshold`) but present in target-group traces.

        **Config fields:**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log.
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `separating_key` *(str, default: `"activity"`)* - column or attribute key used to label traces into groups.
        - `separating_groups` *(list[list[str]])* - group definitions, e.g. `[["fail", "error"]]`.
        - `support_threshold` *(float [0,1], default: `0.1`)* - global-support ceiling defining "rare".
        """
        logger.info(f"{self.name} running rare_rules via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "rare_rules"
        try:
            self._load_analyser_config(config)
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        result_list, _ = run_rare_rules(self.analyser_config, self.storage)
        return {"code": 200, "data": result_list}

    def api_targeted_rules(self, analyser_config: Annotated[ComparisonConfig, Body(
        openapi_examples={
            "default": {
                "summary": "Discover targeted rules with default settings",
                "value": {
                    "log_name": "example_log",
                    "storage_namespace": "siesta",
                    "method_params": {"target_label": 1, "filtering_support": 1.0},
                    "separating_key": "activity",
                    "separating_groups": [],
                    "support_threshold": 0.8,
                },
            },
        }
    )]) -> Any:
        """Discover directly-following rules strongly associated with a target trace group.

        Evaluates which rules are characteristic of the target group using a support-based filter.

        **Config fields:**
        - `log_name` *(str, default: `"example_log"`)* - name of the indexed log.
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `method_params` *(object)* - `target_label` (int, default: `1`), `filtering_support` (float, default: `1`).
        - `separating_key` *(str, default: `"activity"`)* - column or attribute key used to label traces into groups.
        - `separating_groups` *(list[list[str]])* - group definitions, e.g. `[["fail", "error"]]`.
        - `support_threshold` *(float [0,1], default: `0.8`)* - min fraction of target-group traces a rule must cover.
        """
        logger.info(f"{self.name} running targeted_rules via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "targeted_rules"
        try:
            self._load_analyser_config(config)
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        result_list, _ = run_targeted_rules(self.analyser_config, self.storage)
        return {"code": 200, "data": result_list}

    # ------------------------------------------------------------------
    # API entry points - process model discovery (from the former `model` module)
    # ------------------------------------------------------------------

    def api_dfg_model(self, analyser_config: Annotated[ProcessModelConfig, Body(openapi_examples={
        "default": {
            "summary": "Discover DFG model with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "end_time": None,
                "noise_threshold": 0.0,
                "output_format": "model",
            },
        },
    })]) -> Any:
        """Discover a Directly-Follows Graph model from an indexed event log.

        Computes the DFG with our own distributed Spark aggregation (same statistics as
        `directly_follows`, but exported as a process-map model file/visualization
        instead of a stats table).

        **Config fields:**
        - `log_name` *(str)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp`.
        - `end_time` *(str | null, default: `null`)* - attribute key for activity end timestamp;
            if set, average activity duration is computed and shown inside each node.
        - `noise_threshold` *(float, default: `0.0`)* - fraction of infrequent paths to remove.
        - `output_format` *(str, default: `"model"`)* - `"model"` (native `.xml`), `"png"`, or `"html"`.
        """
        logger.info(f"{self.name} running dfg_model via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "dfg_model"
        try:
            self._load_analyser_config(config)
        except ValueError as e:
            logger.error(f"Invalid config for dfg_model: {e}")
            return {"code": 400, "message": str(e)}

        return self._model_api_response(*run_process_model(self.analyser_config, self.storage, algo="dfg"))

    def api_bpmn(self, analyser_config: Annotated[ProcessModelConfig, Body(openapi_examples={
        "default": {
            "summary": "Discover BPMN model with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "end_time": None,
                "noise_threshold": 0.0,
                "output_format": "model",
            },
        },
    })]) -> Any:
        """Discover a BPMN process model via pm4py's Inductive Miner, fed by our own
        scalable Spark DFG (no full-log materialization).

        **Config fields:** same as `dfg_model`. `output_format`: `"model"` (native `.bpmn`), `"png"`, or `"html"`.
        """
        logger.info(f"{self.name} running bpmn via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "bpmn"
        try:
            self._load_analyser_config(config)
        except ValueError as e:
            logger.error(f"Invalid config for bpmn: {e}")
            return {"code": 400, "message": str(e)}

        return self._model_api_response(*run_process_model(self.analyser_config, self.storage, algo="bpmn"))

    def api_petri_net(self, analyser_config: Annotated[ProcessModelConfig, Body(openapi_examples={
        "default": {
            "summary": "Discover Petri net model with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "end_time": None,
                "noise_threshold": 0.0,
                "output_format": "model",
            },
        },
    })]) -> Any:
        """Discover a (sound, block-structured) Petri net process model via pm4py's
        Inductive Miner, fed by our own scalable Spark DFG (no full-log materialization).

        **Config fields:** same as `dfg_model`. `output_format`: `"model"` (native `.pnml`), `"png"`, or `"html"`.
        """
        logger.info(f"{self.name} running petri_net via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "petri_net"
        try:
            self._load_analyser_config(config)
        except ValueError as e:
            logger.error(f"Invalid config for petri_net: {e}")
            return {"code": 400, "message": str(e)}

        return self._model_api_response(*run_process_model(self.analyser_config, self.storage, algo="petri_net"))

    # ------------------------------------------------------------------
    # API entry points - new analytics
    # ------------------------------------------------------------------

    def api_bottlenecks(self, analyser_config: Annotated[BottlenecksConfig, Body(openapi_examples={
        "default": {
            "summary": "Detect bottleneck activity pairs with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "end_time": None,
                "grouping_key": None,
                "grouping_value": None,
                "zscore_threshold": 3.5,
                "top_k": None,
                "support_threshold": 0.0,
                "sigma_multiplier": 3.0,
                "order_by": "impact_score",
            },
        },
    })]) -> Any:
        """Surface activity-pair "regions" that are both common among traces and take
        unusually long relative to other common regions.

        A pair passes two clauses to be flagged `is_bottleneck`: (1) `support` (fraction of
        groups containing the pair) at or above `support_threshold`, i.e. "common"; and
        (2) `avg_duration_sec` above `mean + sigma_multiplier * stddev`, computed over the
        pairs that passed clause 1 - i.e. "unusually long relative to other common regions".
        The response's `bottleneck_cutoff_sec` records that computed threshold.

        **Config fields:**
        - `log_name` *(str)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp`.
        - `end_time` *(str | null, default: `null`)* - attribute key for event end timestamp.
        - `grouping_key` *(str | list | null, default: `null`)* - attribute key(s) defining what counts as
            "adjacent" events; `null` = `trace_id`.
        - `grouping_value` *(str | list | null, default: `null`)* - restrict to groups with matching key value(s).
        - `zscore_threshold` *(float, default: `3.5`)* - robust (MAD-based) z-score threshold for the separate
            `flagged`/`zscore` columns (kept for backward compatibility; independent of `is_bottleneck`).
        - `top_k` *(int | null, default: `null`)* - keep only the k pairs ranked first by `order_by`.
        - `return_csv` *(bool, default: `false`)* - return a CSV file download instead of a JSON list.
        - `support_threshold` *(float [0,1], default: `0.0`)* - minimum support for a pair to count as "common".
        - `sigma_multiplier` *(float, default: `3.0`)* - standard deviations above the mean required for `is_bottleneck`.
        - `order_by` *(str, default: `"impact_score"`)* - `"impact_score"`, `"avg_duration_sec"`, or `"median_duration_sec"`.
        """
        logger.info(f"{self.name} running bottlenecks via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "bottlenecks"
        try:
            self._load_analyser_config(config)
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        return self._csv_response(run_bottlenecks(self.analyser_config, self.storage), self.analyser_config.get("return_csv", False))

    def api_temporal_deviations(self, analyser_config: Annotated[TemporalDeviationsConfig, Body(openapi_examples={
        "default": {
            "summary": "Discover duration-based discriminating rules with default settings",
            "value": {
                "log_name": "example_log",
                "storage_namespace": "siesta",
                "separating_key": "activity",
                "separating_groups": [["rejected"]],
                "activity_pairs": None,
                "max_auto_pairs": 50,
                "min_group_size": 5,
                "top_k_per_pair": 3,
            },
        },
    })]) -> Any:
        """Discover duration-based discriminating rules between groups of traces.

        For instance, this may reveal that applications in which a manual review and a
        credit reassessment occur within five minutes of each other are significantly
        more likely to be rejected - a pattern conditioned on timing, not just sequence.

        **Config fields:**
        - `log_name` *(str)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: `"siesta"`)* - storage namespace.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp`.
        - `separating_key` *(str, default: `"activity"`)* - column or attribute key used to label traces into groups.
        - `separating_groups` *(list[list[str]])* - group definitions, e.g. `[["rejected"]]`.
        - `activity_pairs` *(list[[str,str]] | null)* - explicit pairs to check; `null` = auto-derive top `max_auto_pairs`.
        - `max_auto_pairs` *(int, default: `50`)* - cap on auto-derived candidate pairs.
        - `min_group_size` *(int, default: `5`)* - minimum combined trace count required to keep a result.
        - `top_k_per_pair` *(int | null, default: `null`)* - keep only the k best thresholds per pair.
        - `return_csv` *(bool, default: `false`)* - return a CSV file download instead of a JSON list.
        """
        logger.info(f"{self.name} running temporal_deviations via API.")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        config = analyser_config.model_dump()
        config["method"] = "temporal_deviations"
        try:
            self._load_analyser_config(config)
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        return self._csv_response(run_temporal_deviations(self.analyser_config, self.storage), self.analyser_config.get("return_csv", False))

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def cli_run(self, args: Any, **kwargs: Any) -> Any:
        logger.info(f"{self.name} is running with args: {args}")
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        parser = argparse.ArgumentParser(description="Siesta Analyser module")
        parser.add_argument("--analyser_config", type=str, required=False,
                            help="Path to analyser configuration JSON file")
        parsed_args, _ = parser.parse_known_args(args)

        if not parsed_args.analyser_config:
            raise RuntimeError("Config not provided. Use --analyser_config <path>")

        config_path = parsed_args.analyser_config
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file {config_path} not found.")

        with open(config_path, "r") as f:
            user_config = json.load(f)

        try:
            self._load_analyser_config(user_config)
        except Exception as e:
            raise ValueError(f"Error loading analyser config: {e}")

        self.storage.initialize_db(self.analyser_config)

        method = self.analyser_config.get("method", "directly_follows")
        match method:
            case "directly_follows":
                return run_directly_follows(self.analyser_config, self.storage)
            case "loop_detection":
                return write_loop_detection_json(run_loop_detection(self.analyser_config, self.storage), self.analyser_config)
            case "durations":
                return run_durations(self.analyser_config, self.storage)
            case "attribute_deviations":
                result, output_path = run_attribute_deviations(self.analyser_config, self.storage)
                if result is None:
                    return output_path
                return write_attribute_deviations_json(result, self.analyser_config)
            case "ngrams":
                return run_ngrams(self.analyser_config, self.storage)
            case "rare_rules":
                _, output_path = run_rare_rules(self.analyser_config, self.storage)
                return output_path
            case "targeted_rules":
                _, output_path = run_targeted_rules(self.analyser_config, self.storage)
                return output_path
            case "dfg_model":
                model_out, _, _ = run_process_model(self.analyser_config, self.storage, algo="dfg")
                return model_out
            case "bpmn":
                model_out, _, _ = run_process_model(self.analyser_config, self.storage, algo="bpmn")
                return model_out
            case "petri_net":
                model_out, _, _ = run_process_model(self.analyser_config, self.storage, algo="petri_net")
                return model_out
            case "bottlenecks":
                return run_bottlenecks(self.analyser_config, self.storage)
            case "temporal_deviations":
                return run_temporal_deviations(self.analyser_config, self.storage)
            case _:
                raise ValueError(f"Unknown analyser method: '{method}'")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _load_analyser_config(self, config: Dict[str, Any]):
        if not self.storage.log_exists(config):
            logger.error(
                f"Log '{config.get("log_name")}' does not exist in storage. Run indexing first."
            )
            raise ValueError(f"Log '{config.get("log_name")}' not found in storage.")

        if config.get("log_name") is None:
            raise ValueError("Log name not specified in config.")

        self.analyser_config = DEFAULT_ANALYSER_CONFIG.copy()
        self.analyser_config.update(config)

        if config.get("output_path") is not None and config.get("output_path") == "output/example_log":
            config["output_path"] = "output/" + config.get("log_name", "analyser_results")

        given_output = config.get("output_path")

        Path(given_output).parent.mkdir(parents=True, exist_ok=True)
        self.analyser_config["output_path"] = (
            given_output + "_" + str(datetime.datetime.now().timestamp())
        )

    # ------------------------------------------------------------------
    # Response shaping
    # ------------------------------------------------------------------

    def _csv_response(self, output_path: str, return_csv: bool) -> Any:
        if return_csv:
            return FileResponse(output_path, media_type="text/csv", filename=Path(output_path).name)
        with open(output_path, "r", newline="") as f:
            try:
                return {"code": 200, "data": list(csv.DictReader(f))}
            except Exception:
                logger.error(f"Failed to parse results from {output_path}.")
                return {"code": 500, "message": f"Failed to parse results from {output_path}."}

    def _model_api_response(self, model_out, png_out, html_out):
        fmt = self.analyser_config.get("output_format", "model")
        if fmt == "png" and os.path.exists(png_out):
            return FileResponse(png_out, media_type="image/png", filename=Path(png_out).name)
        if fmt == "html" and os.path.exists(html_out):
            return FileResponse(html_out, media_type="text/html", filename=Path(html_out).name)
        return FileResponse(model_out, media_type="application/octet-stream",
                            filename=Path(model_out).name)
