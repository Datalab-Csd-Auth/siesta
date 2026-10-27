import csv
import datetime
import json
import os
import shutil
from typing import Any, Dict
from pyspark.sql import functions as F

from siesta.core.interfaces import StorageManager
from siesta.model.StorageModel import MetaData

from siesta.modules.analyser.directly_follows import compute_directly_follows
from siesta.modules.analyser.loop_detection import compute_loop_detection
from siesta.modules.analyser.durations import compute_activity_durations, compute_group_durations
from siesta.modules.analyser.attribute_deviations import (
    compute_attribute_deviations, render_html, ALL_STEPS,
)
from siesta.modules.analyser.ngrams import discover_ngrams, save_ngram_results, create_network
from siesta.modules.analyser.dm import discover_rare_rules, discover_targeted_rules, save_dm_results
from siesta.modules.analyser.trace_labels import resolve_trace_labels
from siesta.modules.analyser.process_model import discover_process_model
from siesta.modules.analyser.bottlenecks import compute_bottlenecks
from siesta.modules.analyser.temporal_deviations import compute_temporal_deviations
from siesta.modules.mine.ordered import discover_ordered

import logging

logger = logging.getLogger(__name__)


def parse_min_timestamp(iso_str: str | None) -> int | None:
    """Parse an ISO 8601 datetime string to epoch milliseconds, or return None."""
    if iso_str is None:
        return None
    parsed = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(parsed.timestamp() * 1000)


def load_metadata(config: Dict[str, Any], storage: StorageManager) -> MetaData:
    metadata = MetaData(
        storage_namespace=config.get("storage_namespace", "siesta"),
        log_name=config.get("log_name", "default_log"),
        storage_type=config.get("storage_type", "s3"),
    )
    return storage.read_metadata_table(metadata)


def run_directly_follows(config: Dict[str, Any], storage: StorageManager) -> str:
    logger.info("Running directly_follows.")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata)
    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
    events_df.cache()

    result_df = compute_directly_follows(
        events_df=events_df,
        trace_count=metadata.trace_count,
        end_time=config.get("end_time"),
        support_threshold=config.get("support_threshold"),
        filter_out=config.get("filter_out", False),
        include_traces=config.get("include_traces", False),
    )

    output_path = config["output_path"] + ".csv"
    config["output_path"] = output_path

    result_pd = result_df.toPandas()
    if config.get("include_traces", False) and "trace_ids" in result_pd.columns:
        result_pd["trace_ids"] = result_pd["trace_ids"].apply(
            lambda x: ",".join(sorted(x)) if x else ""
        )
    result_pd.to_csv(output_path, index=False)

    events_df.unpersist()
    logger.info(f"Completed. Results written to {output_path}.")

    return output_path


def run_loop_detection(config: Dict[str, Any], storage: StorageManager) -> Dict[str, Any]:
    logger.info("Running loop_detection.")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata)
    events_df.cache()

    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)

    result = compute_loop_detection(
        events_df=events_df,
        grouping_key=config.get("grouping_key"),
        grouping_value=config.get("grouping_value"),
        support_threshold=config.get("support_threshold"),
        filter_out=config.get("filter_out", False),
        top_k=config.get("top_k"),
        trace_based=config.get("trace_based", False),
        max_pattern_length=config.get("max_pattern_length", 8),
        include_repeats=config.get("include_repeats", True),
        with_time_stats=config.get("with_time_stats", True),
        with_distribution=config.get("with_distribution", False),
    )

    events_df.unpersist()
    logger.info(
        f"Completed. Found {len(result['self_loops'])} self-loop type(s), "
        f"{len(result['non_self_loops'])} non-self-loop type(s), and "
        f"{len(result['repeated_patterns'])} repeated-pattern type(s)."
    )

    return result


def write_loop_detection_json(result: Dict[str, Any], config: Dict[str, Any]) -> str:
    output_path = config["output_path"] + ".json"
    config["output_path"] = output_path
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results written to {output_path}.")
    return output_path


def run_durations(config: Dict[str, Any], storage: StorageManager) -> str:
    logger.info("Running durations.")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata)
    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
    events_df.cache()

    mode         = config.get("duration_mode", "activity")
    end_time     = config.get("end_time")
    grouping_key = config.get("grouping_key")
    grouping_val = config.get("grouping_value")

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
            per_group=config.get("per_group", False),
        )

    output_path = config["output_path"] + ".csv"
    config["output_path"] = output_path

    result_df.toPandas().to_csv(output_path, index=False)

    events_df.unpersist()
    logger.info(f"Completed. Results written to {output_path}.")

    return output_path


def run_attribute_deviations(config: Dict[str, Any], storage: StorageManager) -> tuple[Dict[str, Any] | None, str | None]:
    logger.info("Running attribute_deviations.")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata)
    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
    events_df.cache()
    total_traces = metadata.trace_count

    rare_support_threshold = config.get("on_rare")
    if rare_support_threshold is not None:
        logger.info(
            "on_rare set: discovering ordered constraints and filtering to non-satisfying traces "
            f"for support >= {rare_support_threshold}."
        )

        ordered_constraints = discover_ordered(events_df, metadata).select(
            "template", "source", "target", "trace_id"
        ).cache()

        high_supported = (
            ordered_constraints
            .groupBy("template", "source", "target")
            .agg(F.countDistinct("trace_id").alias("support_count"))
            .withColumn("support", F.col("support_count") / F.lit(metadata.trace_count))
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
                f"out of {metadata.trace_count}."
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
        steps=config.get("steps", list(ALL_STEPS)),
        excluded_keys=config.get("excluded_attributes"),
        surprise_threshold=config.get("surprise_threshold", 4.0),
        zscore_threshold=config.get("zscore_threshold", 3.5),
        n_buckets=config.get("n_buckets", 5),
        ngram_n=config.get("ngram_n", 2),
        min_group_size=config.get("min_group_size", 5),
        support_threshold=config.get("support_threshold"),
        filter_out=config.get("filter_out", False),
    )

    events_df.unpersist()
    logger.info(f"Completed. {len(records)} deviation records found.")

    output_format = config.get("output_format", "json")
    log_name = config.get("log_name", "log")

    if output_format == "csv":
        output_path = config["output_path"] + "_deviations.csv"
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
                writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
                writer.writeheader()
                writer.writerows(flat)
        return None, output_path

    if output_format == "html":
        output_path = config["output_path"] + "_deviations.html"
        html_content = render_html(records, log_name, active_steps)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return None, output_path

    # Default: JSON
    result = {
        "code": 200,
        "log_name": log_name,
        "total_deviations": len(records),
        "deviations": records,
    }
    return result, None


def write_attribute_deviations_json(result: Dict[str, Any], config: Dict[str, Any]) -> str:
    output_path = config["output_path"] + "_deviations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results written to {output_path}.")
    return output_path


def run_ngrams(config: Dict[str, Any], storage: StorageManager) -> str:
    logger.info("Running ngrams.")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata).dropDuplicates(
        ["trace_id", "activity", "start_timestamp"]
    )
    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
    events_df.cache()

    trace_labels = resolve_trace_labels(
        events_df,
        config.get("separating_key", "activity"),
        config.get("separating_groups", []),
    )
    params = config.get("method_params", {})

    results = discover_ngrams(events=events_df, trace_labels=trace_labels, n=params.get("n", 2))

    output_path = config["output_path"] + ".csv"
    config["output_path"] = output_path
    save_ngram_results(results, output_path, fmt="csv")

    if params.get("vis", False):
        with open(output_path.replace(".csv", ".html"), "w") as f:
            f.write(create_network(output_path))

    events_df.unpersist()
    logger.info(f"Completed. Results written to {output_path}.")

    return output_path


def run_rare_rules(config: Dict[str, Any], storage: StorageManager) -> tuple[list, str]:
    logger.info("Running rare_rules.")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata).dropDuplicates(
        ["trace_id", "activity", "start_timestamp"]
    )
    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
    events_df.cache()

    trace_labels = resolve_trace_labels(
        events_df,
        config.get("separating_key", "activity"),
        config.get("separating_groups", []),
    )

    ordered_constraints_df = discover_ordered(events_df, metadata)
    result_list = discover_rare_rules(
        ordered_constraints_df=ordered_constraints_df,
        trace_labels=trace_labels,
        trace_count=metadata.trace_count,
        support_pct=config.get("support_threshold", 0.1),
    )

    output_path = config["output_path"] + ".json"
    config["output_path"] = output_path
    save_dm_results(result_list, output_path)

    events_df.unpersist()
    logger.info(f"Completed. Results written to {output_path}.")

    return result_list, output_path


def run_targeted_rules(config: Dict[str, Any], storage: StorageManager) -> tuple[list, str]:
    logger.info("Running targeted_rules.")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata).dropDuplicates(
        ["trace_id", "activity", "start_timestamp"]
    )
    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
    events_df.cache()

    trace_labels = resolve_trace_labels(
        events_df,
        config.get("separating_key", "activity"),
        config.get("separating_groups", []),
    )
    params = config.get("method_params", {})

    ordered_constraints_df = discover_ordered(events_df, metadata)
    result_list = discover_targeted_rules(
        ordered_constraints_df=ordered_constraints_df,
        trace_labels=trace_labels,
        target_label=params.get("target_label", 1),
        support_threshold=config.get("support_threshold", 0.8),
        filtering_support=params.get("filtering_support", 1),
    )

    output_path = config["output_path"] + ".json"
    config["output_path"] = output_path
    save_dm_results(result_list, output_path)

    events_df.unpersist()
    logger.info(f"Completed. Results written to {output_path}.")

    return result_list, output_path


def _copy_outputs(config, model_path, fmt, png_path, html_path):
    base = config["output_path"]
    model_out = base + "." + fmt
    png_out   = base + ".png"
    html_out  = base + ".html"
    try:
        for src, dst in [(model_path, model_out), (png_path, png_out), (html_path, html_out)]:
            if src and os.path.exists(src):
                shutil.copy(src, dst)
    except Exception:
        logger.exception("Failed to copy model files to output location.")
    return model_out, png_out, html_out


def run_process_model(config: Dict[str, Any], storage: StorageManager, algo: str) -> tuple[str, str, str]:
    logger.info(f"Running process model discovery ({algo}).")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata)
    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
    events_df.cache()

    try:
        model_path, fmt, png_path, html_path = discover_process_model(
            events_df,
            algo=algo,
            end_time=config.get("end_time"),
            noise_threshold=config.get("noise_threshold", 0.0),
        )
    except Exception:
        events_df.unpersist()
        logger.exception(f"Process model discovery ({algo}) failed.")
        raise

    model_out, png_out, html_out = _copy_outputs(config, model_path, fmt, png_path, html_path)
    events_df.unpersist()
    logger.info(f"{algo} model -> {model_out}  PNG -> {png_out}  HTML -> {html_out}")
    config["output_path"] = model_out

    return model_out, png_out, html_out


def run_bottlenecks(config: Dict[str, Any], storage: StorageManager) -> str:
    logger.info("Running bottlenecks.")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata)
    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
    events_df.cache()

    result_df = compute_bottlenecks(
        events_df=events_df,
        trace_count=metadata.trace_count,
        end_time=config.get("end_time"),
        grouping_key=config.get("grouping_key"),
        grouping_value=config.get("grouping_value"),
        zscore_threshold=config.get("zscore_threshold", 3.5),
        top_k=config.get("top_k"),
        support_threshold=config.get("support_threshold", 0.0),
        sigma_multiplier=config.get("sigma_multiplier", 3.0),
        order_by=config.get("order_by", "impact_score"),
    )

    output_path = config["output_path"] + ".csv"
    config["output_path"] = output_path
    result_df.toPandas().to_csv(output_path, index=False)

    events_df.unpersist()
    logger.info(f"Completed. Results written to {output_path}.")

    return output_path


def run_temporal_deviations(config: Dict[str, Any], storage: StorageManager) -> str:
    logger.info("Running temporal_deviations.")
    metadata = load_metadata(config, storage)

    events_df = storage.read_sequence_table(metadata)
    min_ts = parse_min_timestamp(config.get("min_timestamp"))
    if min_ts is not None:
        events_df = events_df.filter(F.col("start_timestamp") >= min_ts)
    events_df.cache()

    trace_labels = resolve_trace_labels(
        events_df,
        config.get("separating_key", "activity"),
        config.get("separating_groups", []),
    )

    result_df = compute_temporal_deviations(
        events_df=events_df,
        trace_labels=trace_labels,
        trace_count=metadata.trace_count,
        activity_pairs=config.get("activity_pairs"),
        max_auto_pairs=config.get("max_auto_pairs", 50),
        min_group_size=config.get("min_group_size", 5),
        top_k_per_pair=config.get("top_k_per_pair"),
    )

    output_path = config["output_path"] + ".csv"
    config["output_path"] = output_path
    result_df.toPandas().to_csv(output_path, index=False)

    events_df.unpersist()
    logger.info(f"Completed. Results written to {output_path}.")

    return output_path
