from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField, StringType
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


DEFAULT_MAX_PATTERN_WINDOW = 8


# ---------------------------------------------------------------------------
# UDF schema: each detected loop is (pattern, loop_type, repetitions)
# ---------------------------------------------------------------------------

_LOOP_SCHEMA = ArrayType(StructType([
    StructField("pattern", StringType(), False),
    StructField("loop_type", StringType(), False),
    StructField("occurrences", IntegerType(), False),
]))


def _is_primitive(block: list) -> bool:
    """True when `block` is not itself a back-to-back repetition of a shorter block.

    "A B A B" is not primitive - it is "A B" twice - and is therefore left to the
    shorter window, which already reports it with the correct repetition count.
    """
    w = len(block)
    for d in range(1, w // 2 + 1):
        if w % d == 0 and block == block[:d] * (w // d):
            return False
    return True


def _same_neighbour(activities: list, positions: list, offset: int) -> bool:
    """True when every occurrence in `positions` has the same activity at `offset`.

    Used to test whether a block can be extended - on the left (offset -1) or on the
    right (offset = block length) - without losing any of its occurrences. An
    occurrence that runs off either end of the sequence blocks the extension.
    """
    n = len(activities)
    neighbour = None
    for i in positions:
        j = i + offset
        if j < 0 or j >= n:
            return False
        if neighbour is None:
            neighbour = activities[j]
        elif activities[j] != neighbour:
            return False
    return True


def _count_non_overlapping(positions: list, w: int) -> int:
    """Count occurrences of a width-`w` block, greedily skipping overlapping ones."""
    count = 0
    last_end = -1
    for i in positions:  # ascending
        if i >= last_end:
            count += 1
            last_end = i + w
    return count


def _find_repeated_patterns(activities: list, max_pattern_window: int) -> dict:
    """Find repeated patterns in an ordered activity sequence.

    A repeated pattern is a block of 2..`max_pattern_window` activities that recurs
    later in the sequence, with anything at all in between the occurrences:
    "A B C x y z A B C" holds the repeated pattern "A -> B -> C". Blocks of length
    one are skipped since those are self-loops, and occurrences must not overlap.

    Only maximal blocks are kept:
      - a block whose occurrences can all be extended by the same activity on the
        left or on the right is dropped in favour of that longer block, so
        "A B C x y z A B C" reports "A -> B -> C" and not "A -> B" or "B -> C";
      - a block that is itself a repetition of a shorter block ("A B A B") is
        dropped in favour of that shorter block.
    Right extension is not tested at the window cap, so a repeat longer than
    `max_pattern_window` is reported truncated to the cap rather than lost.

    Returns a dict mapping the block (as a tuple) to its number of non-overlapping
    occurrences in this sequence.
    """
    n = len(activities)
    found: dict = {}

    for w in range(2, max_pattern_window + 1):
        if 2 * w > n:  # no room for two non-overlapping occurrences
            break

        positions_by_block: dict = {}
        for i in range(n - w + 1):
            positions_by_block.setdefault(tuple(activities[i:i + w]), []).append(i)

        for block, positions in positions_by_block.items():
            if len(positions) < 2 or not _is_primitive(list(block)):
                continue
            if _same_neighbour(activities, positions, -1):
                continue  # left-extendible: the longer block carries the same occurrences
            if w < max_pattern_window and _same_neighbour(activities, positions, w):
                continue  # right-extendible

            count = _count_non_overlapping(positions, w)
            if count >= 2:
                found[block] = count

    return found


def _make_find_loops_udf(repeated_patterns: bool, max_pattern_window: int):
    """Build the per-group loop-detection UDF, closing over the detection options."""

    @F.udf(_LOOP_SCHEMA)
    def _find_loops_udf(sequence):
        """Detect loops in an ordered activity sequence.

        A self-loop is an activity immediately followed by itself.
        A minimal non-self-loop is a subsequence that starts and ends with the same
        activity, with at least one different activity in between, and the bounding
        activity does not appear anywhere within the middle part.
        A repeated pattern is a block of activities that recurs later in the group,
        with anything in between ("A B C x y z A B C"), only detected when enabled.

        Returns a list of (pattern, loop_type, occurrences) triples, deduplicated
        per group; `occurrences` is 1 for everything but repeated patterns.
        """
        if not sequence:
            return []

        activities = [row["activity"] for row in sorted(sequence, key=lambda r: r["position"])]
        n = len(activities)
        loops = set()

        for i in range(n - 1):
            # Self-loop: same activity at consecutive positions
            if activities[i] == activities[i + 1]:
                loops.add((activities[i], "self_loop"))

        for i in range(n - 1):
            for j in range(i + 2, n):
                if activities[i] == activities[j]:
                    # Minimal: bounding activity must not appear in the body
                    if activities[i] not in activities[i + 1:j]:
                        pattern = " -> ".join(activities[i:j + 1])
                        loops.add((pattern, "non_self_loop"))

        result = [(pattern, loop_type, 1) for pattern, loop_type in loops]

        if repeated_patterns:
            for block, count in _find_repeated_patterns(activities, max_pattern_window).items():
                result.append((" -> ".join(block), "repeated_pattern", count))

        return result

    return _find_loops_udf


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_loop_detection(
    events_df: DataFrame,
    grouping_key: Optional[Union[str, list]] = None,
    grouping_value: Optional[Union[str, list]] = None,
    support_threshold: Optional[float] = None,
    filter_out: bool = False,
    top_k: Optional[int] = None,
    trace_based: bool = False,
    repeated_patterns: bool = False,
    max_pattern_window: int = DEFAULT_MAX_PATTERN_WINDOW,
) -> dict:
    """Detect loops in an indexed event log using distributed Spark processing.

    Args:
        events_df: Sequence table DataFrame (activity, trace_id, position,
                   start_timestamp, attributes).
        grouping_key: Attribute key(s) to group by instead of trace_id. A string
                      means a single attribute key; a list means a composite key.
                      None defaults to trace_id grouping.
        grouping_value: Optional value(s) to keep after extracting the group
                        column. Rows whose group column value is not in this list
                        are dropped before detection. For a composite key, supply
                        a dict mapping each key to its allowed value(s).
        support_threshold: Fraction [0, 1] threshold; None = no filtering.
        filter_out: When True, keeps loops with support <= threshold (rare).
                    When False (default), keeps loops with support >= threshold
                    (frequent).
        top_k: Keep only the k most-supported loops (applied after threshold
               filtering). None = keep all.
        trace_based: When True and grouping by trace_id, adds a trace_ids field
                     to each loop entry listing the traces that contain it.
        repeated_patterns: When True, also detect repeated patterns - blocks of
                           activities that recur later in the group with anything
                           in between, as in "A B C x y z A B C". Off by default.
        max_pattern_window: Longest block considered as a repeated pattern
                            (default 8). Blocks of length 1 are always skipped
                            since they are self-loops.

    Returns:
        Dict with keys:
            total_groups      - number of distinct groups evaluated
            grouping_key      - effective grouping key(s) used
            self_loops        - list of loop dicts, sorted by support desc
            non_self_loops    - list of loop dicts, sorted by support desc
            repeated_patterns - list of loop dicts, sorted by support desc;
                                empty unless repeated_patterns=True

        Each loop dict contains:
            pattern         - activity name (self-loop), "A -> B -> ... -> A"
                              (non-self-loop) or the recurring block "A -> B -> C"
            support         - fraction of groups containing the loop [0, 1]
            group_count     - absolute count of groups
            max_occurrences - highest number of non-overlapping occurrences of the
                              block in any single group (repeated patterns only)
            trace_ids       - list of trace IDs (only when trace_based=True and
                              grouping by trace_id)
    """
    if repeated_patterns and max_pattern_window < 2:
        raise ValueError("max_pattern_window must be at least 2.")

    # --- 1. Resolve grouping columns ---
    keys = [] if grouping_key is None else (
        [grouping_key] if isinstance(grouping_key, str) else list(grouping_key)
    )
    is_trace_grouping = not keys or keys == ["trace_id"]

    if is_trace_grouping:
        group_cols = ["trace_id"]
    else:
        existing_cols = set(events_df.columns)
        group_cols = []
        for key in keys:
            if key in existing_cols:
                group_cols.append(key)           # top-level column (e.g. activity)
            else:
                col_name = f"_grp_{key}"
                events_df = events_df.withColumn(col_name, F.col("attributes").getItem(key))
                group_cols.append(col_name)

    # --- 2. Grouping-value filter ---
    if grouping_value is not None:
        if isinstance(grouping_value, dict):
            # Multi-key case: {key: value(s)}
            for key, vals in grouping_value.items():
                col_name = f"_grp_{key}"
                allowed = [vals] if isinstance(vals, str) else list(vals)
                events_df = events_df.filter(F.col(col_name).isin(allowed))
        else:
            # Single key: string or list of strings applied to the first group col
            allowed = [grouping_value] if isinstance(grouping_value, str) else list(grouping_value)
            events_df = events_df.filter(F.col(group_cols[0]).isin(allowed))

    # --- 3. Count total groups ---
    total_groups = events_df.select(*group_cols).distinct().count()

    if total_groups == 0:
        return {
            "total_groups": 0,
            "grouping_key": "trace_id" if is_trace_grouping else grouping_key,
            "self_loops": [],
            "non_self_loops": [],
            "repeated_patterns": [],
        }

    # --- 4. Collect ordered sequences per group and detect loops via UDF ---
    seq_df = events_df.groupBy(*group_cols).agg(
        F.collect_list(
            F.struct(F.col("activity"), F.col("position"))
        ).alias("sequence")
    )

    find_loops_udf = _make_find_loops_udf(repeated_patterns, max_pattern_window)
    loops_df = seq_df.withColumn("loops", find_loops_udf(F.col("sequence")))

    # Explode: one row per (group, loop_pattern, loop_type), already unique per
    # group since the UDF emits each (pattern, loop_type) at most once
    exploded = (
        loops_df
        .select(*group_cols, F.explode("loops").alias("loop"))
        .select(
            *group_cols,
            F.col("loop.pattern").alias("pattern"),
            F.col("loop.loop_type").alias("loop_type"),
            F.col("loop.occurrences").alias("occurrences"),
        )
        .distinct()
    )
    exploded.cache()

    # --- 5. Aggregate: count groups per loop ---
    agg_exprs = [
        F.count("*").alias("group_count"),
        F.max("occurrences").alias("max_occurrences"),
    ]
    if trace_based and is_trace_grouping:
        agg_exprs.append(F.collect_list(F.col("trace_id")).alias("trace_ids"))

    result_df = (
        exploded
        .groupBy("pattern", "loop_type")
        .agg(*agg_exprs)
        .withColumn("support", F.col("group_count") / F.lit(total_groups))
    )

    # --- 6. Apply support threshold ---
    if support_threshold is not None:
        if filter_out:
            result_df = result_df.filter(F.col("support") <= F.lit(support_threshold))
        else:
            result_df = result_df.filter(F.col("support") >= F.lit(support_threshold))

    # --- 7. Sort and limit ---
    result_df = result_df.orderBy(F.col("support").desc())
    if top_k is not None:
        result_df = result_df.limit(top_k)

    # --- 8. Collect to driver and build output dict ---
    rows = result_df.collect()
    exploded.unpersist()

    self_loops = []
    non_self_loops = []
    repeated = []

    for row in rows:
        entry = {
            "pattern": row["pattern"],
            "support": round(float(row["support"]), 6),
            "group_count": int(row["group_count"]),
        }
        if trace_based and is_trace_grouping:
            entry["trace_ids"] = sorted(row["trace_ids"])

        if row["loop_type"] == "self_loop":
            self_loops.append(entry)
        elif row["loop_type"] == "repeated_pattern":
            entry["max_occurrences"] = int(row["max_occurrences"])
            repeated.append(entry)
        else:
            non_self_loops.append(entry)

    return {
        "total_groups": int(total_groups),
        "grouping_key": "trace_id" if is_trace_grouping else grouping_key,
        "self_loops": self_loops,
        "non_self_loops": non_self_loops,
        "repeated_patterns": repeated,
    }
