from typing import Dict, List
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from siesta.model.StorageModel import ConstraintEntry, MetaData
from siesta.core.storageFactory import get_storage_manager
from siesta.modules.mine.pandas_udf_utils import sanitize_udf_output


# ---------------------------------------------------------------------------
# Pandas UDF: mine all ordered templates for a single trace
# ---------------------------------------------------------------------------
def _mine_trace_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Receives all events of ONE trace (columns: trace_id, activity, position).
    Returns a DataFrame with columns (trace_id, template, source, target).
    """
    trace_id = pdf["trace_id"].iloc[0]
    pdf = pdf.sort_values("position").reset_index(drop=True)
    activities = pdf["activity"].to_numpy()
    n = len(activities)

    if n < 2:
        return pd.DataFrame(columns=["template", "source", "trace_id", "target", "occurrences"])

    # Pre-compute per-activity position lists for O(1) lookups
    act_positions: Dict[str, np.ndarray] = {}
    for act in np.unique(activities):
        act_positions[act] = np.where(activities == act)[0]

    next_activity = np.empty(n, dtype=object)
    next_activity[:-1] = activities[1:]
    next_activity[-1] = None

    prev_activity = np.empty(n, dtype=object)
    prev_activity[1:] = activities[:-1]
    prev_activity[0] = None

    rows: List = []

    # Iterate over all ordered (source, target) distinct pairs
    unique_acts = np.unique(activities)
    for src_act in unique_acts:
        src_indices = act_positions[src_act]
        for tgt_act in unique_acts:
            tgt_indices = act_positions[tgt_act]

            # Check if source ever appears before target
            # (indices are sorted, so min(src) >= max(tgt) means no src < any tgt)
            any_src_before_tgt = src_indices[0] < tgt_indices[-1]

            # --- Not Succession: a never occurs before b ---
            is_not_succession = not any_src_before_tgt

            # --- Not Chain Succession: a is never immediately followed by b ---
            is_not_chain_succession = not any(
                next_activity[s] == tgt_act for s in src_indices
            )

            if not any_src_before_tgt:
                # No positive ordered constraints can hold,
                # but negation constraints might
                if is_not_succession:
                    rows.append(("not_succession", src_act, trace_id, tgt_act, None))
                if is_not_chain_succession:
                    rows.append(("not_chain_succession", src_act, trace_id, tgt_act, None))
                continue

            # --- Response ---
            # Every src occurrence must have at least one later tgt
            is_response = True
            is_alt_response = True
            for i, s_idx in enumerate(src_indices):
                later = tgt_indices[tgt_indices > s_idx]
                if len(later) == 0:
                    is_response = False
                    is_alt_response = False
                    break
                # AlternateResponse: closest tgt before next src
                if is_alt_response and i + 1 < len(src_indices):
                    next_src_idx = src_indices[i + 1]
                    if later[0] >= next_src_idx:
                        is_alt_response = False

            # --- Precedence ---
            is_precedence = True
            is_alt_precedence = True
            for j, t_idx in enumerate(tgt_indices):
                earlier = src_indices[src_indices < t_idx]
                if len(earlier) == 0:
                    is_precedence = False
                    is_alt_precedence = False
                    break
                # AlternatePrecedence: closest src after previous tgt
                if is_alt_precedence and j > 0:
                    prev_tgt_idx = tgt_indices[j - 1]
                    if earlier[-1] <= prev_tgt_idx:
                        is_alt_precedence = False

            # --- ChainResponse ---
            is_chain_response = all(
                next_activity[s] == tgt_act for s in src_indices
            )

            # --- ChainPrecedence ---
            is_chain_precedence = all(
                prev_activity[t] == src_act for t in tgt_indices
            )

            # Skip if nothing matches
            if not any([
                is_response, is_precedence,
                is_alt_response, is_alt_precedence,
                is_chain_response, is_chain_precedence,
                is_not_chain_succession,
            ]):
                continue

            if is_response:
                rows.append(("response", src_act, trace_id, tgt_act, None))
            if is_precedence:
                rows.append(("precedence", src_act, trace_id, tgt_act, None))
            if is_response and is_precedence:
                rows.append(("succession", src_act, trace_id, tgt_act, None))
            if is_alt_response:
                rows.append(("alternate_response", src_act, trace_id, tgt_act, None))
            if is_alt_precedence:
                rows.append(("alternate_precedence", src_act, trace_id, tgt_act, None))
            if is_chain_response:
                rows.append(("chain_response", src_act, trace_id, tgt_act, None))
            if is_chain_precedence:
                rows.append(("chain_precedence", src_act, trace_id, tgt_act, None))
            if is_chain_response and is_chain_precedence:
                rows.append(("chain_succession", src_act, trace_id, tgt_act, None))
            if is_not_chain_succession:
                rows.append(("not_chain_succession", src_act, trace_id, tgt_act, None))

    return sanitize_udf_output(
        pd.DataFrame(rows, columns=["template", "source", "trace_id", "target", "occurrences"]),
        ConstraintEntry.get_schema(),
    )


def discover_ordered(evolved_df: DataFrame, metadata: MetaData) -> DataFrame:
    """
    Mine ordered constraints (tight semantics: the relation must hold for ALL
    occurrences of the source/target pair within a given trace)

    Templates discovered (per activity-pair, per trace):
      1. Response(a,b)            every 'a' is eventually followed by a 'b'.
      2. Precedence(a,b)          every 'b' is preceded by an 'a'.
      3. Succession(a,b)          Response AND Precedence.
      4. AlternateResponse(a,b)   every 'a' is followed by 'b' before the next 'a'.
      5. AlternatePrecedence(a,b) every 'b' is preceded by 'a' after the previous 'b'.
      6. ChainResponse(a,b)       every 'a' is immediately followed by 'b'.
      7. ChainPrecedence(a,b)     every 'b' is immediately preceded by 'a'.
      8. ChainSuccession(a,b)     ChainResponse AND ChainPrecedence.

    Incremental: old constraints for traces that appear in ``evolved_df`` are
    invalidated and re-mined from scratch; untouched traces are kept as-is.

    :param evolved_df: DataFrame[Event] with events after the last mining timestamp.
    :param metadata:   MetaData with storage paths and log context.
    :return: DataFrame with columns (template, source, target, trace_id, occurrences).
    """
    storage = get_storage_manager()
    old_constraints_df = storage.read_ordered_constraints(metadata).cache()
    old_constraints_df.count()

    # Invalidate old constraints for traces that have evolved 
    evolved_trace_ids = evolved_df.select("trace_id").distinct()
    unchanged_constraints = old_constraints_df.join(
        evolved_trace_ids, on="trace_id", how="left_anti"
    )

    # Mine constraints for evolved traces
    new_constraints = evolved_df.select("trace_id", "activity", "position")\
        .groupBy("trace_id")\
        .applyInPandas(_mine_trace_pandas, ConstraintEntry.get_schema())


    ordered_constraints = unchanged_constraints.unionByName(new_constraints, allowMissingColumns=True)

    # Force materialization
    ordered_constraints = ordered_constraints.cache()
    ordered_constraints.count()
    old_constraints_df.unpersist()

    storage.write_ordered_constraints(metadata, ordered_constraints)

    return ordered_constraints
