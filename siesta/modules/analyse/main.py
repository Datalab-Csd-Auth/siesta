import argparse
import csv
import datetime
import json
from pathlib import Path
from typing import Annotated, Any, Dict
from fastapi import Body, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import SparkSession, functions as F

from siesta.core.config import get_system_config, get_config_value
from siesta.core.interfaces import SiestaModule, StorageManager
from siesta.core.storageFactory import get_storage_manager
from siesta.model.StorageModel import MetaData
from siesta.modules.analyse.directly_follows import compute_directly_follows
from siesta.modules.analyse.loop_detection import DEFAULT_MAX_PATTERN_WINDOW, compute_loop_detection
from siesta.modules.analyse.durations import compute_activity_durations, compute_group_durations
from siesta.modules.analyse.attribute_deviations import (
    compute_attribute_deviations, render_html, ALL_STEPS,
)
from siesta.modules.analyse.form_analysis import DATE_GROUPINGS, run_form_analysis
from siesta.modules.analyse.postprocess_csv import build_rules_html
from siesta.modules.mine.ordered import discover_ordered
from siesta.modules.index.main import Indexing
from siesta.modules.mine.main import Mining


class DirectlyFollowsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field(default_factory=lambda: get_config_value("storage_namespace_default", "siesta"), description="Storage namespace")
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
    storage_namespace: str = Field(default_factory=lambda: get_config_value("storage_namespace_default", "siesta"), description="Storage namespace")
    grouping_key: str | list[str] | None = Field(None, description="Attribute key(s) to group by; null = trace_id")
    grouping_value: str | list[str] | dict | None = Field(None, description="Restrict to groups with matching key value(s)")
    min_timestamp: str | None = Field(None, description="Lower bound on start_timestamp as ISO 8601 datetime with millisecond precision (e.g. '2024-01-15T10:30:45.123Z')")
    support_threshold: float | None = Field(None, description="Min support fraction [0,1]; null = no filtering")
    filter_out: bool = Field(False, description="When true, keeps rare loops (support ≤ threshold)")
    top_k: int | None = Field(None, description="Keep only the k most-supported loops; null = all")
    trace_based: bool = Field(False, description="Add trace_ids list to each loop entry (only when grouping by trace_id)")
    repeated_patterns: bool = Field(False, description="Also detect repeated patterns: blocks recurring later in the group, gaps allowed (A B C x y z A B C)")
    max_pattern_window: int = Field(DEFAULT_MAX_PATTERN_WINDOW, ge=2, description="Longest block considered as a repeated pattern")
    output_path: str = Field("output/example_log", description="Local path prefix for the output file")


class DurationsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("example_log", description="Name of the indexed log")
    storage_namespace: str = Field(default_factory=lambda: get_config_value("storage_namespace_default", "siesta"), description="Storage namespace")
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
    storage_namespace: str = Field(default_factory=lambda: get_config_value("storage_namespace_default", "siesta"), description="Storage namespace.")
    min_timestamp: str | None = Field(None, description="Lower bound on start_timestamp as ISO 8601 datetime with millisecond precision (e.g. '2024-01-15T10:30:45.123Z')")
    steps: list[int] = Field(
        list(ALL_STEPS),
        description=(
            "Which analysis steps to run. Subset of [0,1,2,3,4]. "
            "0=value frequency (inter+intra-trace), 1=activity*attribute, "
            "2=position-conditioned, 3=n-gram context, 4=value transitions."
        ),
    )
    excluded_attributes: list[str] | None = Field(
        None,
        description="Attribute keys to skip in addition to auto-detected timestamp keys.",
    )
    surprise_threshold: float = Field(
        4.0,
        description="Anomaly threshold for categorical attributes (-log₂ score). Default 4.0 ≈ support ≤ 6.25 %.",
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


class FormAnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_name: str = Field("form_log", description="Name used for the produced trace log")
    input_path: str | None = Field(None, description="Path to the local .xlsx file (CLI only; the API takes an upload)")
    header_row: int = Field(1, description="Zero-indexed row of the sheet holding the column names")
    sheet_name: str | int = Field(0, description="Sheet name or zero-indexed sheet position")
    date_grouping: str = Field("dow", description=f"Bucketing applied to date columns. One of {list(DATE_GROUPINGS)}")
    numeric_bins: int = Field(5, description="Number of equal-width buckets for numeric columns")
    retain_threshold: float = Field(0.05, ge=0.0, le=1.0, description="Columns whose distinct-value ratio exceeds this are collapsed to filled/unfilled")
    drop_constant: bool = Field(True, description="Drop columns holding a single distinct value")
    drop_unique: bool = Field(True, description="Drop string columns where every row holds a different value")
    return_csv: bool = Field(True, description="Return the CSV file; when false, return the preprocessing report and a preview")
    output_path: str = Field("output/form_log", description="Local path prefix for the output file")
    mine: bool = Field(False, description="After preprocessing, index the resulting log and mine unordered relations from it")
    clear_existing: bool = Field(False, description="mine=true only: drop and rebuild any existing indexed data for log_name before indexing")
    mined_format: str = Field("json", description="mine=true only: format of the mined constraints - 'json', 'csv' (the rules file as-is), or 'html' (interactive report)")
    include_trace_lists: bool = Field(True, description="mine=true only: include the supporting trace_ids alongside each mined constraint")
    api_base_url: str | None = Field(None, description="Base URL of the Siesta API for the HTML report (e.g. 'http://localhost:8000'). null = relative paths (same origin)")


import logging

logger = logging.getLogger(__name__)


def _parse_min_timestamp(iso_str: str | None) -> int | None:
    """Parse an ISO 8601 datetime string to epoch milliseconds, or return None."""
    if iso_str is None:
        return None
    parsed = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(parsed.timestamp() * 1000)


DEFAULT_ANALYSER_CONFIG: Dict[str, Any] = {
    **DirectlyFollowsConfig().model_dump(),
    **LoopDetectionConfig().model_dump(),
    **DurationsConfig().model_dump(),
    **AttributeDeviationsConfig().model_dump(),
    "method": "directly_follows"
}


class Analysing(SiestaModule):

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
            "form_analysis":         ("POST", self.api_form_analysis),
        }

    # ------------------------------------------------------------------
    # API entry points
    # ------------------------------------------------------------------

    def api_directly_follows(self, analyser_config: Annotated[DirectlyFollowsConfig, Body(openapi_examples={
        "default": {
            "summary": "Find directly-following pairs with default settings",
            "value": {
                "log_name": "example_log",
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
        - `storage_namespace` *(str, default: system config `storage_namespace_default`)* - storage namespace.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp` as ISO 8601 with millisecond precision (e.g. `"2024-01-15T10:30:45.123Z"`). Events before this datetime are excluded.
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
        
        return self._run_directly_follows(caller="api")

    def api_durations(self, analyser_config: Annotated[DurationsConfig, Body(openapi_examples={
        "default": {
            "summary": "Compute duration statistics with default settings",
            "value": {
                "log_name": "example_log",
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
        - `storage_namespace` *(str, default: system config `storage_namespace_default`)* - storage namespace.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp` as ISO 8601 with millisecond precision (e.g. `"2024-01-15T10:30:45.123Z"`). Events before this datetime are excluded.
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
        
        return self._run_durations(caller="api")

    def api_loop_detection(self, analyser_config: Annotated[LoopDetectionConfig, Body(openapi_examples={
        "default": {
            "summary": "Detect loops with default settings",
            "value": {
                "log_name": "example_log",
                "grouping_key": None,
                "grouping_value": None,
                "min_timestamp": None,
                "support_threshold": None,
                "filter_out": False,
                "top_k": None,
                "trace_based": False,
                "repeated_patterns": False,
                "max_pattern_window": 8,
            },
        },
    })]) -> Any:
        """Detect self-loops, non-self-loops and repeated patterns in an indexed event log.

        A **self-loop** is an activity immediately followed by itself.
        A **non-self-loop** is a minimal cycle A -> … -> A where A does not appear in the body.
        A **repeated pattern** is a block of activities that recurs later in the group, with
        anything at all in between the occurrences: `A B C x y z A B C` holds the repeated
        pattern `A -> B -> C`. Only detected when `repeated_patterns` is `true`.

        Returns JSON with `self_loops`, `non_self_loops` and `repeated_patterns` arrays. Each
        entry contains the activity pattern and its support fraction across groups; repeated
        patterns additionally carry `max_occurrences`, the highest number of non-overlapping
        occurrences of the block in any single group.

        **Config fields:**
        - `log_name` *(str)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: system config `storage_namespace_default`)* - storage namespace.
        - `grouping_key` *(str | list | null, default: `null`)* - attribute key(s) to group by. `null` = `trace_id`.
        - `grouping_value` *(str | list | dict | null, default: `null`)* - restrict to specific group values.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp` as ISO 8601 with millisecond precision (e.g. `"2024-01-15T10:30:45.123Z"`). Events before this datetime are excluded.
        - `support_threshold` *(float [0,1] | null, default: `null`)* - keep loops with support ≥ threshold. `null` = no filtering.
        - `filter_out` *(bool, default: `false`)* - when `true`, keeps loops with support ≤ threshold (rare loops).
        - `top_k` *(int | null, default: `null`)* - keep only the k most-supported loops. `null` = all.
        - `trace_based` *(bool, default: `false`)* - add a `trace_ids` list to each loop entry (only when grouping by `trace_id`).
        - `repeated_patterns` *(bool, default: `false`)* - also detect repeated patterns, i.e. blocks of
            2+ activities that recur later in the group with anything in between the occurrences
            (`A B C x y z A B C` -> `A -> B -> C`). Only maximal blocks are kept: a block whose
            occurrences can all be extended left or right by the same activity is reported as that
            longer block instead (so no `A -> B` or `B -> C` above), and a block that is itself a
            repetition of a shorter one (`A B A B`) is reported under that shorter block.
        - `max_pattern_window` *(int, default: `8`)* - longest block considered as a repeated pattern.
            Blocks of length 1 are always skipped since those are self-loops; a repeat longer than the
            window is reported truncated to it rather than dropped.
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
        

        return self._run_loop_detection(caller="api")

    def api_attribute_deviations(self, analyser_config: Annotated[AttributeDeviationsConfig, Body(openapi_examples={
        "default": {
            "summary": "Detect attribute deviations with default settings",
            "value": {
                "log_name": "example_log",
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
        - **Step 1** - Activity * Attribute: flags values whose distribution within an activity type
            is anomalous (Laplace surprise for categorical, MAD z-score for numeric).
        - **Step 2** - Position-conditioned: same as step 1 but conditioned on relative position
            within the trace (bucketed).
        - **Step 3** - N-gram context: same as step 1 but conditioned on the n-gram of activities
            ending at this event.
        - **Step 4** - Value transitions: flags rare (prev_value -> curr_value) transitions
            within a trace (categorical attributes only).

        **Config fields:**
        - `log_name` *(str)* - name of the indexed log. **Required.**
        - `storage_namespace` *(str, default: system config `storage_namespace_default`)* - storage namespace.
        - `min_timestamp` *(str | null, default: `null`)* - lower bound on `start_timestamp` as ISO 8601 with millisecond precision (e.g. `"2024-01-15T10:30:45.123Z"`). Events before this datetime are excluded.
        - `steps` *(list[int], default: `[0,1,2,3,4]`)* - which steps to run.
        - `excluded_attributes` *(list[str] | null)* - attribute keys to skip (auto-excludes timestamp keys).
        - `surprise_threshold` *(float, default: `4.0`)* - categorical anomaly threshold (-log₂ score).
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
        
        return self._run_attribute_deviations(caller="api")

    def api_form_analysis(
        self,
        analyser_config: Annotated[str, Form(
            description="JSON configuration object - see endpoint description for all supported fields.",
            openapi_examples={
                "dow": {
                    "summary": "Bucket date columns by day of week",
                    "value": '{"log_name": "forms", "date_grouping": "dow", "numeric_bins": 5}',
                },
                "month": {
                    "summary": "Bucket date columns by month, keep rarer values",
                    "value": '{"log_name": "forms", "date_grouping": "month", "retain_threshold": 0.1}',
                },
            },
        )],
        form_file: UploadFile | None = None,
    ) -> Any:
        """ Extracting pairwise rules from form data inputs.

        Columns are reduced first: constant and all-distinct columns are dropped,
        date columns are bucketed according to `date_grouping`, numeric columns are
        cut into `numeric_bins` equal-width buckets, and columns with more than
        `retain_threshold` distinct values collapse to `filled` / unfilled.

        Columns scoped to a country contribute an event only when the cell carries a
        `yes` or a digit, keeping the bulk of "not registered here" answers out of the
        traces. Country names are matched against `siesta/modules/analyse/countries.txt`.

        Returns the resulting CSV (`trace_id,position,activity,timestamp`) as a download, or
        the per-column report when `return_csv` is `false`.

        When `form_file` is omitted, no preprocessing happens - instead `log_name` must
        already be indexed (e.g. from a prior call to this endpoint with a file). The
        existing log is mined directly and the mined constraints are returned per
        `mined_format`, exactly as if `mine=true` had produced them.

        **Form fields:**
        - `form_file` *(file, optional)* - the `.xlsx` file to preprocess. When omitted,
            `log_name` must refer to an already-indexed log; mining runs against it directly.
        - `analyser_config` *(JSON string)* - configuration; fields below.

        **Config fields (`analyser_config`):**
        - `log_name` *(str, default: `"form_log"`)* - name used for the produced trace log.
        - `header_row` *(int, default: `1`)* - zero-indexed row of the sheet holding the column names.
        - `sheet_name` *(str | int, default: `0`)* - sheet name or zero-indexed sheet position.
        - `date_grouping` *(str, default: `"dow"`)* - bucketing for date columns:
            `"dow"`, `"month"`, `"quarter"`, `"year"`, `"hour"`, `"date"`, or `"none"` (drops them).
        - `numeric_bins` *(int, default: `5`)* - equal-width buckets for numeric columns.
        - `retain_threshold` *(float [0,1], default: `0.05`)* - above this distinct-value ratio a column
            collapses to `filled` / unfilled; at or below it the values are kept verbatim.
        - `drop_constant` *(bool, default: `true`)* - drop columns holding a single distinct value.
        - `drop_unique` *(bool, default: `true`)* - drop string columns where every row differs.
        - `return_csv` *(bool, default: `true`)* - when `false`, return the per-column report plus a 20-row preview.
            Ignored when `mine` is `true` (the response is always JSON so the mined constraints can be included).
        - `mine` *(bool, default: `false`)* - `form_file` given only: after preprocessing, index the
            resulting log under `log_name` and mine unordered relations from it. The mined constraints
            are then returned per `mined_format`, instead of the usual form-analysis CSV/report. Ignored
            (mining always runs) when `form_file` is omitted.
        - `clear_existing` *(bool, default: `false`)* - `mine=true` only: drop and rebuild any existing indexed
            data for `log_name` before indexing the freshly produced log.
        - `mined_format` *(str, default: `"json"`)* - `mine=true`, or no `form_file`, only: `"json"` returns
            the form-analysis summary with a `mined` list of the discovered constraints (just the `mined`
            list when no `form_file` was given); `"csv"` returns the rules CSV file as-is; `"html"` returns
            a self-contained interactive rules viewer built from those same rows.
        - `include_trace_lists` *(bool, default: `true`)* - `mine=true`, or no `form_file`, only: include the
            supporting `trace_ids` for each mined constraint. In `"html"`, trace ids are hidden behind a
            per-row expander rather than shown inline, since a constraint can be backed by many traces.
        """
        logger.info(f"{self.name} running form_analysis via API.")

        try:
            self._load_form_analysis_config(json.loads(analyser_config))
        except Exception as e:
            logger.exception(f"Error loading analyser config: {e}")
            return {"code": 400, "message": f"Invalid config: {e}"}

        has_file = form_file is not None and bool(form_file.filename)

        try:
            if has_file:
                return self._run_form_analysis(caller="api", source=form_file.file)
            return self._run_form_analysis_existing(caller="api")
        except Exception as e:
            logger.exception(f"Error running form_analysis: {e}")
            return {"code": 400, "message": f"Form analysis failed: {e}"}

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

        if user_config.get("method") == "form_analysis":
            self._load_form_analysis_config(user_config)
            return self._run_form_analysis(caller="cli")

        try:
            self._load_analyser_config(user_config)
        except Exception as e:
            raise ValueError(f"Error loading analyser config: {e}")

        self.storage.initialize_db(self.analyser_config)

        method = self.analyser_config.get("method", "directly_follows")
        match method:
            case "directly_follows":
                return self._run_directly_follows(caller="cli")
            case "loop_detection":
                return self._run_loop_detection(caller="cli")
            case "durations":
                return self._run_durations(caller="cli")
            case "attribute_deviations":
                return self._run_attribute_deviations(caller="cli")
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

    def _load_form_analysis_config(self, config: Dict[str, Any]):
        self.analyser_config = FormAnalysisConfig().model_dump()
        self.analyser_config.update(config)
        self.analyser_config["method"] = "form_analysis"

        given_output = self.analyser_config.get("output_path") or "output/form_log"
        if given_output == "output/form_log":
            given_output = "output/" + self.analyser_config.get("log_name", "form_log")

        Path(given_output).parent.mkdir(parents=True, exist_ok=True)
        self.analyser_config["output_path"] = (
            given_output + "_" + str(datetime.datetime.now().timestamp()) + ".csv"
        )

    def _require_form_analysis_completed(self, config: Dict[str, Any]) -> Dict[str, Any] | None:
        """Ensure `form_analysis` has fully run (indexed) for `config["log_name"]`.

        Returns a ready-to-send error dict when the log is missing, or `None` when
        the previous form-analysis run has completed and the caller may proceed.
        """
        self.siesta_config = get_system_config()
        self.storage = get_storage_manager()

        log_name = config.get("log_name")
        if not log_name:
            return {"code": 400, "message": "Log name not specified in config."}

        if not self.storage.log_exists(config):
            logger.error(f"Form analysis has not completed for log '{log_name}'.")
            return {
                "code": 409,
                "message": (
                    f"Form analysis has not completed for log '{log_name}'. "
                    "Run the form_analysis endpoint with mine=true first."
                ),
            }
        return None

    def _load_metadata(self):
        self.metadata = MetaData(
            storage_namespace=self.analyser_config.get("storage_namespace", get_config_value("storage_namespace_default", "siesta")),
            log_name=self.analyser_config.get("log_name", "default_log"),
            storage_type=self.analyser_config.get("storage_type", "s3"),
        )
        self.metadata = self.storage.read_metadata_table(self.metadata)

    # ------------------------------------------------------------------
    # Method implementations
    # ------------------------------------------------------------------

    def _run_directly_follows(self, caller: str) -> Any:
        logger.info(f"Running directly_follows initiated by {caller}.")
        self._load_metadata()

        events_df = self.storage.read_sequence_table(self.metadata)
        min_ts = _parse_min_timestamp(self.analyser_config.get("min_timestamp"))
        if min_ts is not None:
            events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
        events_df.cache()

        result_df =compute_directly_follows(
            events_df=events_df,
            trace_count=self.metadata.trace_count,
            end_time=self.analyser_config.get("end_time"),
            support_threshold=self.analyser_config.get("support_threshold"),
            filter_out=self.analyser_config.get("filter_out", False),
            include_traces=self.analyser_config.get("include_traces", False),
        )

        output_path = self.analyser_config["output_path"] + ".csv"
        self.analyser_config["output_path"] = output_path

        result_pd = result_df.toPandas()
        if self.analyser_config.get("include_traces", False) and "trace_ids" in result_pd.columns:
            result_pd["trace_ids"] = result_pd["trace_ids"].apply(
                lambda x: ",".join(sorted(x)) if x else ""
            )
        result_pd.to_csv(output_path, index=False)

        events_df.unpersist()
        logger.info(f"Completed. Results written to {output_path}.")

        if caller == "api":
            if self.analyser_config.get("return_csv", False):
                return FileResponse(
                    output_path,
                    media_type="text/csv",
                    filename=Path(output_path).name,
                )
            with open(output_path, "r", newline="") as f:
                try:
                    return {"code": 200, "data": list(csv.DictReader(f))}
                except Exception:
                    logger.error(f"Failed to parse results from {output_path}.")
                    return {"code": 500, "message": f"Failed to parse results from {output_path}."}

        return output_path

    def _run_loop_detection(self, caller: str) -> Any:
        logger.info(f"Running loop_detection initiated by {caller}.")
        self._load_metadata()

        events_df = self.storage.read_sequence_table(self.metadata)
        events_df.cache()

        min_ts = _parse_min_timestamp(self.analyser_config.get("min_timestamp"))
        if min_ts is not None:
            events_df = events_df.filter(F.col("start_timestamp") >= min_ts)

        result = compute_loop_detection(
            events_df=events_df,
            grouping_key=self.analyser_config.get("grouping_key"),
            grouping_value=self.analyser_config.get("grouping_value"),
            support_threshold=self.analyser_config.get("support_threshold"),
            filter_out=self.analyser_config.get("filter_out", False),
            top_k=self.analyser_config.get("top_k"),
            trace_based=self.analyser_config.get("trace_based", False),
            repeated_patterns=self.analyser_config.get("repeated_patterns", False),
            max_pattern_window=self.analyser_config.get("max_pattern_window") or DEFAULT_MAX_PATTERN_WINDOW,
        )

        events_df.unpersist()
        logger.info(
            f"Completed. Found {len(result['self_loops'])} self-loop type(s), "
            f"{len(result['non_self_loops'])} non-self-loop type(s) and "
            f"{len(result['repeated_patterns'])} repeated pattern(s)."
        )

        if caller == "cli":
            output_path = self.analyser_config["output_path"] + ".json"
            self.analyser_config["output_path"] = output_path
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results written to {output_path}.")
            return output_path

        return {"code": 200, **result}

    def _run_durations(self, caller: str) -> Any:
        logger.info(f"Running durations initiated by {caller}.")
        self._load_metadata()

        events_df = self.storage.read_sequence_table(self.metadata)
        min_ts = _parse_min_timestamp(self.analyser_config.get("min_timestamp"))
        if min_ts is not None:
            events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
        events_df.cache()

        mode         = self.analyser_config.get("duration_mode", "activity")
        end_time     = self.analyser_config.get("end_time")
        grouping_key = self.analyser_config.get("grouping_key")
        grouping_val = self.analyser_config.get("grouping_value")

        if mode == "group":
            result_df = compute_group_durations(
                events_df=events_df,
                end_time=end_time,
                grouping_key=grouping_key,
                grouping_value=grouping_val,
            )
        else:
            result_df = compute_activity_durations(
                events_df=events_df,
                end_time=end_time,
                grouping_key=grouping_key,
                grouping_value=grouping_val,
                per_group=self.analyser_config.get("per_group", False),
            )

        output_path = self.analyser_config["output_path"] + ".csv"
        self.analyser_config["output_path"] = output_path

        result_df.toPandas().to_csv(output_path, index=False)

        events_df.unpersist()
        logger.info(f"Completed. Results written to {output_path}.")

        if caller == "api":
            if self.analyser_config.get("return_csv", False):
                return FileResponse(
                    output_path,
                    media_type="text/csv",
                    filename=Path(output_path).name,
                )
            with open(output_path, "r", newline="") as f:
                try:
                    return {"code": 200, "data": list(csv.DictReader(f))}
                except Exception:
                    logger.error(f"Failed to parse results from {output_path}.")
                    return {"code": 500, "message": f"Failed to parse results from {output_path}."}

        return output_path

    def _run_attribute_deviations(self, caller: str) -> Any:
        logger.info(f"Running attribute_deviations initiated by {caller}.")
        self._load_metadata()

        events_df = self.storage.read_sequence_table(self.metadata)
        min_ts = _parse_min_timestamp(self.analyser_config.get("min_timestamp"))
        if min_ts is not None:
            events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
        events_df.cache()
        total_traces = self.metadata.trace_count

        rare_support_threshold = self.analyser_config.get("on_rare")
        if rare_support_threshold is not None:
            logger.info(
                "on_rare set: discovering ordered constraints and filtering to non-satisfying traces "
                f"for support >= {rare_support_threshold}."
            )

            ordered_constraints = discover_ordered(events_df, self.metadata).select(
                "template", "source", "target", "trace_id"
            ).cache()

            high_supported = (
                ordered_constraints
                .groupBy("template", "source", "target")
                .agg(F.countDistinct("trace_id").alias("support_count"))
                .withColumn("support", F.col("support_count") / F.lit(self.metadata.trace_count))
                .filter(F.col("support") >= F.lit(float(rare_support_threshold)))
                .select("template", "source", "target")
            )

            high_constraint_count = high_supported.count()
            if high_constraint_count > 0:
                satisfied_counts = (
                    ordered_constraints
                    .join(F.broadcast(high_supported), on=["template", "source", "target"], how="inner")
                    .select("trace_id", "template", "source", "target")
                    .distinct()
                    .groupBy("trace_id")
                    .agg(F.count(F.lit(1)).alias("satisfied_high_constraints"))
                )

                rare_trace_ids = (
                    events_df
                    .select("trace_id")
                    .distinct()
                    .join(satisfied_counts, on="trace_id", how="left")
                    .fillna({"satisfied_high_constraints": 0})
                    .filter(F.col("satisfied_high_constraints") < F.lit(high_constraint_count))
                    .select("trace_id")
                )

                filtered_events_df = events_df.join(F.broadcast(rare_trace_ids), on="trace_id", how="inner")
                filtered_events_df.cache()
                events_df.unpersist()
                events_df = filtered_events_df

                total_traces = rare_trace_ids.count()
                logger.info(
                    f"on_rare set: selected {total_traces} non-satisfying trace(s) "
                    f"out of {self.metadata.trace_count}."
                )
            else:
                logger.info(
                    "on_rare set: no high-support ordered constraints found; "
                    "running deviations on all traces."
                )

            ordered_constraints.unpersist()

        records, active_steps = compute_attribute_deviations(
            events_df=events_df,
            total_traces=total_traces,
            steps=self.analyser_config.get("steps", list(ALL_STEPS)),
            excluded_keys=self.analyser_config.get("excluded_attributes"),
            surprise_threshold=self.analyser_config.get("surprise_threshold", 4.0),
            zscore_threshold=self.analyser_config.get("zscore_threshold", 3.5),
            n_buckets=self.analyser_config.get("n_buckets", 5),
            ngram_n=self.analyser_config.get("ngram_n", 2),
            min_group_size=self.analyser_config.get("min_group_size", 5),
            support_threshold=self.analyser_config.get("support_threshold"),
            filter_out=self.analyser_config.get("filter_out", False),
            # When on_rare is active the trace pool has already been filtered to
            # non-conforming traces.  We pivot to low-surprise perspective so the
            # report shows what is *typical* about these traces rather than further
            # flagging anomalies inside them.
            low_surprise_mode=(rare_support_threshold is not None),
        )

        events_df.unpersist()
        logger.info(f"Completed. {len(records)} deviation records found.")

        output_format = self.analyser_config.get("output_format", "json")
        log_name = self.analyser_config.get("log_name", "log")

        if output_format == "csv":
            import csv as _csv
            output_path = self.analyser_config["output_path"] + "_deviations.csv"
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if records:
                    all_step_keys: set = set()
                    for rec in records:
                        all_step_keys.update(rec.get("scores", {}).keys())
                    flat = []
                    for rec in records:
                        row = {k: v for k, v in rec.items() if k not in ("flagged_by", "scores")}
                        row["flagged_by"] = "|".join(rec.get("flagged_by", []))
                        for sk in sorted(all_step_keys):
                            row[f"score_{sk}"] = rec["scores"].get(sk, "")
                        flat.append(row)
                    writer = _csv.DictWriter(f, fieldnames=list(flat[0].keys()))
                    writer.writeheader()
                    writer.writerows(flat)
            if caller == "api":
                return FileResponse(output_path, media_type="text/csv", filename=Path(output_path).name)
            return output_path

        if output_format == "html":
            output_path = self.analyser_config["output_path"] + "_deviations.html"
            html_content = render_html(
                records,
                log_name,
                active_steps,
                low_surprise_mode=(rare_support_threshold is not None),
                on_rare_threshold=rare_support_threshold,
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            if caller == "api":
                return FileResponse(output_path, media_type="text/html", filename=Path(output_path).name)
            return output_path

        # Default: JSON
        result = {
            "code": 200,
            "log_name": log_name,
            "total_deviations": len(records),
            "deviations": records,
        }
        if caller == "cli":
            output_path = self.analyser_config["output_path"] + "_deviations.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results written to {output_path}.")
            return output_path
        return result

    def _run_form_analysis(self, caller: str, source: Any = None) -> Any:
        logger.info(f"Running form_analysis initiated by {caller}.")

        if source is None:
            input_path = self.analyser_config.get("input_path")
            if not input_path:
                raise ValueError("No input file given. Set 'input_path' in the config.")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input file {input_path} not found.")
            source = input_path

        summary = run_form_analysis(
            source,
            self.analyser_config["output_path"],
            header_row=self.analyser_config.get("header_row", 1),
            sheet_name=self.analyser_config.get("sheet_name", 0),
            date_grouping=self.analyser_config.get("date_grouping", "dow"),
            numeric_bins=self.analyser_config.get("numeric_bins", 5),
            retain_threshold=self.analyser_config.get("retain_threshold", 0.05),
            drop_constant=self.analyser_config.get("drop_constant", True),
            drop_unique=self.analyser_config.get("drop_unique", True),
        )

        output_path = summary["output_path"]
        logger.info(
            f"Completed. {summary['event_count']} event(s) across {summary['trace_count']} trace(s) "
            f"written to {output_path}."
        )

        mined_path, mined_header, mined_rows = None, None, None
        if self.analyser_config.get("mine", False):
            mined_path, mined_header, mined_rows = self._index_and_mine_form_log(output_path, caller=caller)

        if mined_path is not None:
            rendered = self._render_mined(mined_path, mined_header, mined_rows, caller=caller)
            if rendered is not None:
                return rendered

        if caller == "api":
            if self.analyser_config.get("return_csv", True) and mined_path is None:
                return FileResponse(
                    output_path,
                    media_type="text/csv",
                    filename=Path(output_path).name,
                )
            with open(output_path, "r", newline="") as f:
                preview = [row for _, row in zip(range(20), csv.DictReader(f))]
            response = {"code": 200, "log_name": self.analyser_config.get("log_name"), **summary, "preview": preview}
            if mined_path is not None:
                response["mined"] = [dict(zip(mined_header, row)) for row in mined_rows]
            return response

        if mined_path is not None:
            return {"output_path": output_path, "mined": [dict(zip(mined_header, row)) for row in mined_rows]}
        return output_path

    def _run_form_analysis_existing(self, caller: str) -> Any:
        """Mine an already-indexed log directly, skipping preprocessing.

        Used when the API is called without `form_file`: rather than erroring, we check
        the log is indexed (raising if it isn't - there's nothing to mine yet), mine it,
        and hand the rules straight to postprocess_csv for the html/csv report, exactly
        like the `mine=true` path does after preprocessing a freshly uploaded file.
        """
        logger.info(f"Running form_analysis initiated by {caller} without a file; mining the existing indexed log.")

        mined_path, mined_header, mined_rows = self._index_and_mine_form_log(None, caller=caller)

        rendered = self._render_mined(mined_path, mined_header, mined_rows, caller=caller)
        if rendered is not None:
            return rendered

        if caller == "api":
            return {
                "code": 200,
                "log_name": self.analyser_config.get("log_name"),
                "mined": [dict(zip(mined_header, row)) for row in mined_rows],
            }

        return {"mined": [dict(zip(mined_header, row)) for row in mined_rows]}

    def _render_mined(
        self, mined_path: str, mined_header: list[str], mined_rows: list[list[str]], caller: str
    ) -> Any:
        """Render mined rules per `mined_format`, honoring `caller` for how the result is delivered.

        Returns None for `mined_format == "json"` so the caller falls through to its own
        JSON response; for "csv"/"html" the API gets a FileResponse/HTMLResponse while the
        CLI gets the path of the file already written to disk (mirroring the CSV, which
        is already on disk at `mined_path`; the HTML is written here since mining doesn't
        produce it).
        """
        mined_format = self.analyser_config.get("mined_format", "json")

        if mined_format == "csv":
            if caller == "api":
                return FileResponse(mined_path, media_type="text/csv", filename=Path(mined_path).name)
            return mined_path

        if mined_format == "html":
            api_base = (self.analyser_config.get("api_base_url") or "").rstrip("/")
            module_prefix = f"/{Mining.__name__.lower()}"
            html, _stats = build_rules_html(
                mined_header, mined_rows, src_name=Path(mined_path).name,
                log_name=self.analyser_config.get("log_name"),
                storage_namespace=self.analyser_config.get(
                    "storage_namespace", get_config_value("storage_namespace_default", "siesta")
                ),
                trace_api=f"{api_base}{module_prefix}/traces",
                violation_api=f"{api_base}{module_prefix}/violations",
            )
            if caller == "api":
                return HTMLResponse(html)
            html_path = str(Path(mined_path).with_suffix(".html"))
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"form_analysis: mined rules HTML written to {html_path}.")
            return html_path

        return None

    def _index_and_mine_form_log(
        self, trace_log_path: str | None, caller: str
    ) -> tuple[str, list[str], list[list[str]]]:
        """Index the trace log produced by form_analysis, then mine unordered relations from it.

        When `trace_log_path` is None, indexing is skipped and mining runs directly against
        whatever is already indexed under `log_name` (raising if nothing is indexed yet) -
        this is the no-file API path where the caller wants mining output for a log that was
        already turned into traces by an earlier form_analysis call.

        Returns (rules_csv_path, header, rows) with rows as raw csv.reader rows, read once,
        so the caller can serve the CSV as-is, build an HTML report from those same in-memory
        rows, or convert them to JSON, without re-reading the file more than once.
        """
        log_name = self.analyser_config.get("log_name", "form_log")
        storage_namespace = self.analyser_config.get(
            "storage_namespace", get_config_value("storage_namespace_default", "siesta")
        )

        storage = get_storage_manager()

        if trace_log_path is not None:
            logger.info(f"form_analysis: mine=true, indexing '{log_name}' before mining.")
            indexer = Indexing()
            indexer.siesta_config = get_system_config()
            indexer.storage = storage
            indexer._load_index_config({
                "log_name": log_name,
                "log_path": trace_log_path,
                "storage_namespace": storage_namespace,
                "clear_existing": self.analyser_config.get("clear_existing", False),
            })
            indexer.storage.initialize_db(indexer.index_config)
            indexer.begin_builders(caller=caller)
        elif not storage.log_exists({"log_name": log_name, "storage_namespace": storage_namespace}):
            raise ValueError(
                f"No file provided and log '{log_name}' was not found in storage namespace "
                f"'{storage_namespace}'. Upload a file, or point log_name at an already-indexed log."
            )

        logger.info(f"form_analysis: mining unordered relations for '{log_name}'.")
        miner = Mining()
        miner.siesta_config = get_system_config()
        miner.storage = storage
        miner._load_mining_config({
            "log_name": log_name,
            "storage_namespace": storage_namespace,
            "categories": ["unordered"],
            "include_trace_lists": self.analyser_config.get("include_trace_lists", True),
        })
        miner.mine(caller=caller)

        rules_csv_path = miner.mining_config["output_path"]
        with open(rules_csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        return rules_csv_path, header, rows
