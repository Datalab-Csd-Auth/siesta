"""
Branching post-processing for singular declarative constraints.

Singular mining produces one constraint per (template, source, target) with the
set of traces that satisfy it. *Branching* merges several of those singular
constraints into a single branched constraint whose variable side is a *set* of
activities, combined under a logical policy:

    - AND  -> intersection of the satisfying trace sets
    - OR   -> union of the satisfying trace sets
    - XOR  -> exclusive (symmetric) difference of the satisfying trace sets

The merge is greedy and apriori-based, in one of two directions:

    - bottomup : start from the best 2-activity pair and expand one activity at
                 a time while the combined support keeps improving (AND: while it
                 stays above the support floor).
    - topdown  : start from every activity and prune the least useful one at a
                 time until the bound / support floor is reached.

Bound semantics:
    - branching_bound > 0 : maximum number of activities in the branched set.
    - branching_bound == 0: unbounded - stop only when support would drop below
                            the support floor (support_threshold * trace_count).

Scope per category:
    - Pair categories (ordered, unordered) carry a real source -> target pair and
      are branched over whichever side ``branching_type`` selects.
    - Existential constraints (activity, #occurrences) are branched over the
      activity (source) only, grouped by (template, occurrences): activities that
      share the same occurrence count merge into one activity set.
    - Positional constraints (init/end of an activity) have no source/target, so
      the activity is simply extended into a set - one branched init and one
      branched end - and only OR/XOR branching is allowed.
    - Negation constraints are never branched.

Unary branching (existential, positional) always extends the activity side and
runs whenever branching is enabled, independent of ``branching_type``. Branched
constraints report support only; confidence and interest are undefined for an
activity *set* and left null, mirroring the standalone miners.
"""

from typing import Dict, List, Optional, Set

import pandas as pd
from siesta.modules.mine.pandas_udf_utils import (
    sanitize_udf_output, find_chunked_output_columns,
)
from pyspark.sql import DataFrame, Column, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType,
    BooleanType, ArrayType,
)

import logging
logger = logging.getLogger(__name__)


# Pair categories expose a genuine source -> target relationship and admit
# source/target branching over whichever side is selected.
PAIR_CATEGORIES = ("ordered", "unordered")

# Unary categories carry a single activity (in ``source``); branching extends
# that activity into a set. Positional additionally restricts the policy.
UNARY_CATEGORIES = ("existential", "positional")

# Policies allowed for positional constraints (init/end): union / exclusive only.
POSITIONAL_POLICIES = ("or", "xor")

# Per-policy default merge direction. OR coverage is best pruned from the full
# union downward; AND/XOR are best grown from the strongest pair upward.
_DEFAULT_APPROACH = {"or": "topdown", "and": "bottomup", "xor": "bottomup"}


def _branched_schema() -> StructType:
    """Output schema of the per-group branching UDF."""
    return StructType([
        StructField("category", StringType(), False),
        StructField("template", StringType(), False),
        StructField("source", StringType(), True),
        StructField("target", StringType(), True),
        StructField("occurrences", IntegerType(), True),
        StructField("trace_ids", ArrayType(StringType()), True),
        StructField("match_count", LongType(), True),
        StructField("is_branched", BooleanType(), False),
    ])


def _combine(sets: List[Set[str]], policy: str) -> Set[str]:
    """Fold a list of trace sets under the branching policy (associative)."""
    if not sets:
        return set()
    acc = set(sets[0])
    for s in sets[1:]:
        if policy == "and":
            acc &= s
        elif policy == "or":
            acc |= s
        else:  # xor
            acc ^= s
    return acc


def _branch_bottomup(
    elements: Dict[str, Set[str]], policy: str, bound: int, min_traces: int
) -> Optional[Set[str]]:
    """Grow the branched set from the strongest pair, one activity at a time."""
    keys = list(elements.keys())
    if len(keys) < 2:
        return None

    # Strongest starting pair (maximum combined support).
    best_pair = None
    best_support = -1
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            support = len(_combine([elements[keys[i]], elements[keys[j]]], policy))
            if support > best_support:
                best_support = support
                best_pair = (keys[i], keys[j])

    if best_pair is None or best_support <= min_traces:
        return None

    current = list(best_pair)
    current_set = _combine([elements[k] for k in current], policy)
    remaining = [k for k in keys if k not in current]

    while remaining:
        if bound and len(current) >= bound:
            break

        best_add = None
        best_new = -1
        best_new_set: Optional[Set[str]] = None
        for cand in remaining:
            new_set = _combine([current_set, elements[cand]], policy)
            new_support = len(new_set)
            # AND is restrictive: never let the intersection fall to/under the floor.
            if policy == "and" and new_support <= min_traces:
                continue
            if new_support > best_new:
                best_new = new_support
                best_add = cand
                best_new_set = new_set

        if best_add is None:
            break

        # OR/XOR: only keep expanding while the combination strictly improves.
        if policy in ("or", "xor") and best_new <= len(current_set):
            break

        current.append(best_add)
        current_set = best_new_set  # type: ignore[assignment]
        remaining.remove(best_add)

    if len(current) < 2 or len(current_set) <= min_traces:
        return None
    return set(current)


def _branch_topdown(
    elements: Dict[str, Set[str]], policy: str, bound: int, min_traces: int
) -> Optional[Set[str]]:
    """Prune the least useful activity from the full set, one at a time."""
    keys = list(elements.keys())
    if len(keys) < 2:
        return None

    current = list(keys)
    current_set = _combine([elements[k] for k in current], policy)
    if len(current_set) <= min_traces:
        return None

    # Unbounded prunes down to the minimum branchable size (2); the support
    # floor is what actually stops it earlier.
    target_bound = max(2, bound) if bound else 2

    while len(current) > target_bound:
        pick = None
        best_metric = None
        best_without: Optional[Set[str]] = None
        for k in current:
            without = _combine([elements[o] for o in current if o != k], policy)
            w = len(without)
            if policy == "or":
                # Remove the activity whose absence loses the fewest traces.
                metric = len(current_set) - w
                better = best_metric is None or metric < best_metric
            elif policy == "and":
                # Remove the activity whose absence gains the most traces.
                metric = w - len(current_set)
                better = best_metric is None or metric > best_metric
            else:  # xor
                # Remove the activity with the smallest impact on the set.
                metric = abs(w - len(current_set))
                better = best_metric is None or metric < best_metric
            if better:
                best_metric = metric
                pick = k
                best_without = without

        if pick is None:
            break
        # For OR/XOR a removal can drop support below the floor: stop first.
        if policy in ("or", "xor") and len(best_without) <= min_traces:  # type: ignore[arg-type]
            break

        current = [o for o in current if o != pick]
        current_set = best_without  # type: ignore[assignment]

    if len(current) < 2 or len(current_set) <= min_traces:
        return None
    return set(current)


def _select_branch(
    elements: Dict[str, Set[str]], policy: str, approach: str, bound: int, min_traces: int
) -> Optional[Set[str]]:
    """Dispatch to the configured (or policy-default) greedy direction."""
    if approach == "topdown":
        return _branch_topdown(elements, policy, bound, min_traces)
    return _branch_bottomup(elements, policy, bound, min_traces)


def _make_branching_udf(fixed_col: Optional[str], branch_col: str, policy: str,
                        approach: str, bound: int, min_traces: int,
                        emit_trace_ids: bool):
    """
    Build the pandas grouped-map function applied per branching group.

    ``branch_col`` is the activity side that varies (and is merged into a set);
    ``fixed_col`` is the activity side held constant across the group, or ``None``
    for unary categories that have no second activity.

    The full trace sets are always consumed *inside* the UDF to score the greedy
    merge, but they are only echoed back on the output when ``emit_trace_ids`` is
    set. High-support constraints pack millions of IDs into that list column, and
    a single group's output can exceed PyArrow's 2 GiB contiguous-buffer limit,
    which fails the ``StructArray.from_arrays`` step of Spark's Arrow serializer.
    Since ``match_count`` (computed here) drives support/confidence/interest
    downstream, the lists are dropped unless the caller asked to keep them.
    """
    def _udf(pdf: pd.DataFrame) -> pd.DataFrame:
        category = pdf["category"].iloc[0]
        template = pdf["template"].iloc[0]
        occ_raw = pdf["occurrences"].iloc[0]
        occ = None if pd.isna(occ_raw) else int(occ_raw)
        fixed_val = pdf[fixed_col].iloc[0] if fixed_col else None

        elements: Dict[str, Set[str]] = {}
        originals: Dict[str, pd.Series] = {}
        for _, row in pdf.iterrows():
            el = row[branch_col]
            traces = set(row["trace_ids"]) if row["trace_ids"] is not None else set()
            elements[el] = traces
            originals[el] = row

        selected = _select_branch(elements, policy, approach, bound, min_traces)

        rows: List[dict] = []
        if selected and len(selected) >= 2:
            combined = _combine([elements[k] for k in selected], policy)
            branch_label = "|".join(sorted(selected))
            branched = {
                "category": category,
                "template": template,
                "source": None,
                "target": None,
                "occurrences": occ,
                "trace_ids": sorted(combined) if emit_trace_ids else [],
                "match_count": len(combined),
                "is_branched": True,
            }
            if fixed_col:
                branched[fixed_col] = fixed_val
            branched[branch_col] = branch_label
            rows.append(branched)
            leftover = [k for k in elements if k not in selected]
        else:
            # No viable branch: every singular constraint passes through as-is.
            leftover = list(elements.keys())

        for k in leftover:
            r = originals[k]
            r_occ = r["occurrences"]
            rows.append({
                "category": category,
                "template": template,
                "source": r["source"],
                "target": r["target"],
                "occurrences": None if pd.isna(r_occ) else int(r_occ),
                "trace_ids": (list(r["trace_ids"]) if r["trace_ids"] is not None else []) if emit_trace_ids else [],
                "match_count": int(r["match_count"]),
                "is_branched": False,
            })

        result_df = pd.DataFrame(rows, columns=[
            "category", "template", "source", "target",
            "occurrences", "trace_ids", "match_count", "is_branched",
        ])

        # Unary categories (existential, positional) branch on their own stream
        # with target always None; pandas types that all-None column as float64,
        # and other string columns can carry a multi-chunk PyArrow backing. Both
        # serialise to a ChunkedArray that PySpark's Arrow output cannot handle,
        # so coerce the schema's string/array columns to safe object dtypes.
        # TEMP: log which column(s) would have tripped the serializer, to confirm
        # the fix in production; remove once validated.
        schema = _branched_schema()
        offenders = find_chunked_output_columns(result_df, schema)
        if offenders:
            logger.error(
                "branching-udf pre-sanitize ChunkedArray offenders category=%s "
                "template=%s fixed[%s]=%s rows=%d: %s",
                category, template, fixed_col, fixed_val, len(result_df),
                " | ".join(offenders),
            )
        return sanitize_udf_output(result_df, schema)

    return _udf


def _branch_stream(branchable: DataFrame, group_keys: List[str],
                   fixed_col: Optional[str], branch_col: str, policy: str,
                   approach: str, bound: int, min_traces: int,
                   emit_trace_ids: bool) -> DataFrame:
    """Run one branching UDF over a filtered, grouped constraint stream."""
    udf = _make_branching_udf(fixed_col, branch_col, policy, approach, bound,
                              min_traces, emit_trace_ids)
    return branchable.groupBy(*group_keys).applyInPandas(udf, _branched_schema())


def apply_branching(
    grouped_constraints: DataFrame,
    branching_type: str,
    branching_policy: str,
    branching_approach: Optional[str],
    branching_bound: int,
    trace_count: int,
    support_threshold: float,
    include_trace_lists: bool = False,
) -> DataFrame:
    """
    Merge singular constraints into branched ones over the aggregated set.

    :param grouped_constraints: aggregated constraints with columns
        (category, template, source, target, occurrences, trace_ids, match_count).
    :param branching_type: 'source' or 'target' - the pair-constraint side that
        varies. Unary categories always vary the activity regardless.
    :param branching_policy: 'and', 'or' or 'xor'.
    :param branching_approach: 'bottomup', 'topdown', or None for the policy default.
    :param branching_bound: max activities in a branched set, or 0 for unbounded.
    :param trace_count: total number of traces (for the support floor).
    :param support_threshold: minimum support fraction; the branching floor.
    :param include_trace_lists: keep the merged ``trace_ids`` list on branched and
        passed-through UDF output. Defaults to ``False``: only ``match_count`` is
        retained, which avoids materialising multi-gigabyte list columns that
        overflow PyArrow's contiguous-buffer limit during Arrow serialisation.
    :return: aggregated constraints where merged singulars are replaced by their
        branched constraint, carrying an ``is_branched`` boolean column.
    """
    policy = branching_policy.lower()
    approach = (branching_approach or _DEFAULT_APPROACH[policy]).lower()
    bound = branching_bound if branching_bound and branching_bound > 0 else 0
    min_traces = int(support_threshold * trace_count)

    large_var = grouped_constraints.sparkSession.conf.get(
        "spark.sql.execution.arrow.useLargeVarTypes", "false"
    )
    logger.info(
        "Applying %s-branching (policy=%s, approach=%s, bound=%s, floor=%d traces, "
        "include_trace_lists=%s, arrow.useLargeVarTypes=%s).",
        branching_type, policy, approach, bound if bound else "unbounded", min_traces,
        include_trace_lists, large_var,
    )
    if include_trace_lists and str(large_var).lower() != "true":
        logger.warning(
            "Branching will emit full trace_ids lists but arrow.useLargeVarTypes is "
            "off; large groups may overflow PyArrow's 2 GiB buffer and abort the "
            "Arrow output serializer. Restart the driver so the config takes effect."
        )

    # Normalise occurrences to a nullable integer so every branched/passthrough
    # stream shares a type when unioned (occurrences is added as a string
    # placeholder upstream when no category supplies it).
    grouped_constraints = grouped_constraints.withColumn(
        "occurrences", F.col("occurrences").cast("int")
    )

    has_traces = F.size(F.col("trace_ids")) > 0

    # --- Pair branching (ordered, unordered) over the selected side -----------
    pair_pred: Column = (
        F.col("category").isin(list(PAIR_CATEGORIES))
        & F.col("source").isNotNull()
        & F.col("target").isNotNull()
        & has_traces
    )
    if branching_type == "target":
        pair_fixed, pair_branch = "source", "target"
        pair_keys = ["category", "template", "source", "occurrences"]
    else:  # source branching
        pair_fixed, pair_branch = "target", "source"
        pair_keys = ["category", "template", "target", "occurrences"]

    pair_out = _branch_stream(
        grouped_constraints.filter(pair_pred), pair_keys,
        pair_fixed, pair_branch, policy, approach, bound, min_traces,
        include_trace_lists,
    )

    # --- Unary branching (existential, positional) over the activity ----------
    # Positional only admits OR/XOR; under AND it is left singular.
    unary_categories = ["existential"]
    if policy in POSITIONAL_POLICIES:
        unary_categories.append("positional")

    unary_pred: Column = (
        F.col("category").isin(unary_categories)
        & F.col("target").isNull()
        & has_traces
    )
    # Existential merges activities sharing an occurrence count; positional groups
    # by template alone (occurrences is null), yielding one branched init/end.
    unary_out = _branch_stream(
        grouped_constraints.filter(unary_pred),
        ["category", "template", "occurrences"],
        None, "source", policy, approach, bound, min_traces,
        include_trace_lists,
    )

    # --- Passthrough: everything not eligible for branching -------------------
    passthrough = grouped_constraints.filter(~(pair_pred | unary_pred)) \
        .withColumn("is_branched", F.lit(False)) \
        .withColumn("match_count", F.col("match_count").cast("long"))

    return pair_out.unionByName(unary_out, allowMissingColumns=True) \
                   .unionByName(passthrough, allowMissingColumns=True)
