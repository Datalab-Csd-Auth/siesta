import csv
import json
from typing import Any, Dict
from siesta.model.StorageModel import MetaData
from siesta.core.interfaces import StorageManager
from siesta.core.sparkManager import cleanup as spark_cleanup
from siesta.modules.mine.existential import discover_existential
from siesta.modules.mine.positional import discover_positional
from siesta.modules.mine.ordered import discover_ordered
from siesta.modules.mine.unordered import discover_unordered
from siesta.modules.mine.negations import discover_negations
from siesta.modules.mine.branching import apply_branching
from pyspark.sql import DataFrame, functions as F

import logging
logger = logging.getLogger(__name__)


def mine(mining_config: Dict[str, Any], storage: StorageManager) -> MetaData:
    """
    Permorms incremental mining on the log data based on the provided mining configuration and metadata.
    The method loads evolved traces since the last mining, discovers new constraints and keeps only valid 
    old ones and new ones in storage (by overwrite mode), and outputs the results to a CSV file on the driver's local filesystem.
    """

    # Load metadata if available, and evolved traces since last mining from storage
    metadata = MetaData(
        storage_namespace=mining_config.get("storage_namespace", "siesta"),
        log_name=mining_config.get("log_name", "default_log"),
        storage_type=mining_config.get("storage_type", "s3")
    )

    metadata = storage.read_metadata_table(metadata) 
    evolved_df = storage.read_sequence_table(metadata, filter_out="mined" if not mining_config.get("force_recompute", False) else None)
    # Drop per-event attributes (and any other unused columns) right after the
    # fetch: no miner consumes them, but the sequence table's `attributes` map
    # is heavy and would otherwise be materialised by the cache below and
    # carried through every shuffle/UDF. Keep only what mining and the
    # last-mined-timestamp bookkeeping actually need. Projecting before the
    # cache also lets Catalyst prune `attributes` from the Delta scan.
    evolved_df = evolved_df.select("trace_id", "activity", "position", "start_timestamp")
    evolved_df.cache()  # Cache evolved traces as they will be used multiple times during mining

    # Perform mining based on the specified categories in the mining configuration. 
    # Each miner function returns a DataFrame with a common schema, and we union them together 
    # while adding a "category" column to identify the source of each constraint.
    miners = []
    include_trace_lists = mining_config.get("include_trace_lists", False)
    for category in mining_config["categories"]:
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
        result = miner_func(evolved_df, metadata)
        raw_miner_results.append(result)
        constraints_df_list.append(result.withColumn("category", F.lit(category)))

    constraints_df = constraints_df_list[0]
    for constaint_df in constraints_df_list[1:]:
        constraints_df = constraints_df.unionByName(constaint_df, allowMissingColumns=True)

    for col_name in ["target", "occurrences"]:
        if col_name not in constraints_df.columns:
            constraints_df = constraints_df.withColumn(col_name, F.lit(None).cast("string"))

    # Update metadata with new last mining timestamp based on the max timestamp of the evolved traces
    metadata.last_mined_timestamp = evolved_df.agg({"start_timestamp": "max"}).collect()[0][0] if not evolved_df.rdd.isEmpty() else metadata.last_mined_timestamp
    evolved_df.unpersist()
    storage.write_metadata_table(metadata)

    # Output the discovered constraints to a CSV file on the driver's local filesystem 
    # based on the specified output path in the mining configuration.
    _output_constraints(constraints_df, mining_config, storage, metadata, metadata.trace_count)

    # Release cached miner results now that output is written
    for df in raw_miner_results:
        df.unpersist(blocking=False)

    # Release Delta metadata and any remaining cached data
    spark_cleanup()

    return metadata


def _output_constraints(constraints_df: DataFrame, mining_config: Dict[str, Any], storage: StorageManager, metadata: MetaData, trace_count: int):
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

    # Optional branching stage: merge singular constraints that share a fixed
    # activity into a single branched constraint over the variable activity
    # side, under the configured logical policy. Pair constraints (ordered,
    # unordered) branch over the selected source/target side; existential
    # extends the activity per occurrence count; positional extends the
    # activity (OR/XOR only); negations are never branched. Merged singulars
    # are replaced by their branched constraint; unmerged singulars pass
    # through. Branched rows carry a multi-activity source/target ("A|B|C")
    # and report support only. When branching is off, every row is flagged
    # not-branched.
    branching_type = mining_config.get("branching_type", "none")
    if branching_type in ("source", "target"):
        grouped_constraints = apply_branching(
            grouped_constraints,
            branching_type=branching_type,
            branching_policy=mining_config.get("branching_policy", "or"),
            branching_approach=mining_config.get("branching_approach", "auto"),
            branching_bound=int(mining_config.get("branching_bound", 0)),
            trace_count=trace_count,
            support_threshold=mining_config.get("support_threshold", 0.0),
            include_trace_lists=mining_config.get("include_trace_lists", False),
        )
    else:
        grouped_constraints = grouped_constraints.withColumn("is_branched", F.lit(False))

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
            F.lit(False).alias("is_branched"),
        )
        grouped_constraints = grouped_constraints.unionByName(grouped_pre)

    grouped_constraints = grouped_constraints.filter(F.col("support") >= mining_config.get("support_threshold", 0.0))

    # Calculate confidence:
    #   for ordered: #traces_ab / #traces_a                          (P(target|source))
    #   for unordered: #traces_ab / sqrt(#traces_a * #traces_b)      (geometric mean of both directed confidences)
    #   for negation: match_count is the *complement* of coexistence (N - #traces_ab), so the
    #     coexistence count must be recovered as N - match_count before reusing the directed
    #     formulas, applied to the negated events P(not target|source) and P(not source|target).
    activity_counts = storage.read_activity_index(metadata=metadata).groupBy("activity").agg(F.count_distinct("trace_id").alias("activity_trace_count"))
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

    # Branched constraints combine multiple activities on one side, so the
    # single-activity confidence/interest formulas do not apply: they report
    # support only, with confidence/interest left null.
    grouped_constraints = grouped_constraints.withColumn(
        "confidence",
        F.when(
            F.col("is_branched"), F.lit(None)
        ).when(
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

    # A null confidence never satisfies `>=`; branched rows (and any other
    # null-confidence rows) must pass through rather than being dropped.
    grouped_constraints = grouped_constraints.filter(
        F.col("confidence").isNull() | (F.col("confidence") >= mining_config.get("confidence_threshold", 0.0))
    )

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

    # Interest is only defined for categories with a target (ordered/unordered/negation);
    # for unary categories (positional/existential) it's NULL by design, and a NULL
    # never satisfies `>=`, so those rows must pass through rather than being dropped.
    grouped_constraints = grouped_constraints.filter(
        F.col("interest").isNull() | (F.col("interest") >= mining_config.get("interest_threshold", 0.0))
    )

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
    if mining_config.get("include_trace_lists", False):
        select_cols.append(F.concat_ws("|", F.col("trace_ids")).alias("trace_ids"))

    constraints_csv = grouped_constraints.select(*select_cols)

    # Stream rows partition-by-partition from executors to the driver and write
    # them incrementally into a single CSV file on the driver's local filesystem.
    output_path = mining_config["output_path"]
    col_names = constraints_csv.columns

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(col_names)
        for row in constraints_csv.toLocalIterator(prefetchPartitions=True):
            writer.writerow([row[c] for c in col_names])


def stream_mining_results(output_path: str, elapsed: float):
    """Stream the mining CSV as a JSON body without holding the full result in memory.

    Constraints with high support can pack tens of thousands of trace IDs into a
    single `trace_ids` field, well past csv's default 128KB field-size guard, so
    that limit is raised here to a size comfortably above realistic field sizes.
    """
    csv.field_size_limit(10_000_000)
    yield f'{{"code": 200, "time": {json.dumps(elapsed)}, "mined": ['
    try:
        with open(output_path, "r", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                yield ("," if i else "") + json.dumps(row)
    except Exception:
        logger.exception(f"Failed to parse mining results from {output_path} while streaming.")
        raise
    yield "]}"
