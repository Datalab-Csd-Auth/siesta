"""
Functions relating to the generation and filtering of valid event pairs
"""
from collections import defaultdict
import re
from pyspark import RDD
from pyspark.sql import DataFrame
from typing import Iterable, List, Literal, Optional, Tuple, TypeAlias
from pyspark.sql.functions import col, sum as spark_sum, count, min as spark_min, max as spark_max, pow as spark_pow
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from siesta.model.DataModel import Last_Checked_table_schema, EventPair
import logging
from siesta.core.sparkManager import get_spark_session
logger = logging.getLogger(__name__)

# pair_schema = StructType([
#     StructField("eventA", StringType(), True),
#     StructField("eventB", StringType(), True),
#     StructField("trace_id", StringType(), True),
#     StructField("timestampA", IntegerType(), True),
#     StructField("timestampB", IntegerType(), True),
#     StructField("posA", IntegerType(), True),
#     StructField("posB", IntegerType(), True)
# ])

activity_index_schema = StructType([
    StructField("activity", StringType(), True),
    StructField("trace_id", StringType(), True),
    StructField("position", IntegerType(), True),
    StructField("start_timestamp", IntegerType(), True),
    StructField("attributes", StringType(), True)
])

_TIME_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(d|h|m|s)")
LookbackType = Literal["time", "index"]
pair_index_schema = EventPair.get_schema()
# Schema: Source, Target, trace_id, source_timestamp, target_timestamp, source_position, target_position, source_attributes, target_attributes


Trace_ID: TypeAlias = str
Activity_type: TypeAlias = str
Timestamp: TypeAlias = int
Position: TypeAlias = int
Attributes: TypeAlias = str

Event: TypeAlias = Tuple[Activity_type, Timestamp, Position, Attributes]
Trace: TypeAlias = Tuple[Trace_ID, Tuple[Event]]


def extract_last_checked_and_all_pairs(updated_sequence_table_DF: DataFrame, previous_last_checked: DataFrame | None, lookback: str, batch_min_ts: int, batch_min_pos: int) -> Tuple[DataFrame, DataFrame]:
    """
    Extracts the event type pairs from a DF that contains the complete traces. 

    If there are previously indexed pairs, this process will only calculate the new event type 
    pairs by utilizing the information stored in the LastChecked table. Additionally, 
    there will be no event type pair where the two events will have greater time difference 
    than the one described by the parameter lookback.

    The generated pairs follow a Skip-till-next-match policy without overlapping in time. 
    That is, in order to be two occurrences of the same event type pair in a trace, the 
    second one must start after the first one has ended.

    Example:
        The trace t1 = (a,b,c,a,b) contains 2 occurrences of the event type (a,b). 
        However, the trace t2 = (a,a,c,b,b) only contains one occurrence of the 
        previous event type pair.

    Args:
        activity_index_DF: The DataFrame that contains the complete activity index. activity (str)|trace_id (str)|position (int)|    start_timestamp (ts)|          attributes (json)
        previous_last_checked: The loaded values as a DataFrame from the last checked table (it should be None 
            if it is the first time that events are indexed in this log database).
        lookback: The parameter that describes the maximum time difference that two 
            events can have in order to create an event type pair.

    Returns:
        A DataFrame with the extracted event type pairs and their corresponding last timestamps per trace.
    """
    

    real_lookback = _parse_lookback(lookback)
    
    trace_rdd: RDD[Tuple[Trace_ID, Event]] = updated_sequence_table_DF.rdd.map(lambda row: (
        row.trace_id,
        (row.activity, row.start_timestamp, row.position, row.attributes)
    ))
    
    last_checkedRDD = previous_last_checked.rdd if previous_last_checked else None
    
    
    if not last_checkedRDD or last_checkedRDD.count() == 0:
        logger.info("No previously indexed pairs found. Extracting all pairs from scratch.")
        full = trace_rdd.groupByKey().map(lambda x: _calculate_pairs_stnm(x, None, real_lookback))
    else:
        last_checkedRDD_grouped = last_checkedRDD.map(lambda row: (
            row.trace_id,
            (row.source, row.target, row.last_checked_moment)
        )).groupByKey()
        
        full = trace_rdd.groupByKey().leftOuterJoin(last_checkedRDD_grouped).map(
            lambda x: _calculate_pairs_stnm((x[0], x[1][0]), x[1][1], real_lookback)
        )
    
    pairs = full.flatMap(lambda x: x[0])
    last_checked = full.flatMap(lambda x: x[1])
    spark = get_spark_session()
    pairs_df = spark.createDataFrame(pairs, schema=pair_index_schema)
    last_checked_df = spark.createDataFrame(last_checked, schema=Last_Checked_table_schema)
    merged_last_checked_df = update_last_checked(previous_last_checked, last_checked_df, batch_min_ts, batch_min_pos, real_lookback)
    
    logger.info(f"Extracted {pairs_df.count()} event pairs")
    logger.info(f"Extracted {last_checked_df.count()} last checked entries")

    return pairs_df, merged_last_checked_df

def _calculate_pairs_stnm(activity_index: Tuple[Trace_ID, Iterable[Event]], last: Optional[Iterable[Tuple[str, str, int]]] | None, lookback: Tuple[int, LookbackType]) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Extract the event type pairs from the activity index.

    Args:
        activity_index: The complete activity index.
        last: The list with all the last timestamps that correspond to this event.
        lookback: The parameter that describes the maximum time or position difference between 
            two events in a pair.
    
    Returns:
        tuple: A tuple where the first element is the extracted event type pairs 
            and the second element is the last timestamps for each event type.
    """
    trace_id: Trace_ID = activity_index[0]
    events: Iterable[Event] = activity_index[1]
    
    # Group events by event_name
    activity_index_map = defaultdict(list)
    for event_name, timestamp, position, attributes in events:
        activity_index_map[event_name].append((timestamp, position, attributes))
    
    for key in activity_index_map:
        activity_index_map[key].sort(key=lambda x: x[1])  # sort by position

    # Build lastMap from last_checked
    last_map = {}
    if last:
        for eventA, eventB, timestamp in last:
            last_map[(eventA, eventB)] = timestamp
    
    # Get all unique event types
    all_events = list(activity_index_map.keys())
    
    # Get all combinations
    combinations = _findCombinations(all_events)
    
    results = []
    new_last_checked = []
    
    for key1, key2 in combinations:
        ts1 = activity_index_map.get(key1, [])
        ts2 = activity_index_map.get(key2, [])
        
        if not ts1 or not ts2:
            continue
        
        last_checked_ts = last_map.get((key1, key2), None)
        
        # Detect all occurrences of this event type pair
        nres = createTuples(key1, key2, ts1, ts2, lookback, last_checked_ts, trace_id)
        
        # If there are any, append them and keep the last timestamp
        if nres:
            new_last_checked.append(nres[-1])
            results.extend(nres)
    
    # Convert to last_checked format: (eventA, eventB, trace_id, timestamp)
    last_checked_list = [
        (x[2], x[0], x[1], x[4])  # trace_id, eventA, eventB, timestampB 
        for x in new_last_checked
    ]
    
    return results, last_checked_list

    
def createTuples(
    key1: str, 
    key2: str, 
    e_source: List[Tuple[int, int, str]], 
    e_target: List[Tuple[int, int, str]],
    lookback: Tuple[int, LookbackType], 
    last_checked: Optional[int],
    trace_id: str
) -> List[Tuple[str, str, str, int, int, int, int, str, str]]:
    """
    Creates event type pair tuples following Skip-till-next-match policy.

    Returns:
        List of tuples: (eventA, eventB, trace_id, timestampA, timestampB, posA, posB, attrsA, attrsB)
    """
    pairs = []
    lookback_number = lookback[0]
    prev = None
    i = 0
    
    for ea_ts, ea_pos, ea_attr in e_source:
        # Evaluate based on previous and last_checked
        if ((prev is None or ea_ts >= prev) and 
            (last_checked is None or ea_ts > last_checked)):
            
            stop = False
            while i < len(e_target) and not stop:
                eb_ts, eb_pos, eb_attr = e_target[i]
                
                if ea_pos >= eb_pos:
                    # Event a is not before event b, remove it
                    i += 1
                else:
                    # Event a is before event b
                    if lookback[1] == "index":
                        if eb_pos - ea_pos <= lookback_number:
                            # Lookback satisfied, create pair
                            pairs.append((key1, key2, trace_id, ea_ts, eb_ts, ea_pos, eb_pos, ea_attr, eb_attr))
                            prev = eb_ts
                            i += 1
                    else:
                        if eb_ts - ea_ts <= lookback_number:
                            # Lookback satisfied, create pair
                            pairs.append((key1, key2, trace_id, ea_ts, eb_ts, ea_pos, eb_pos, ea_attr, eb_attr))
                            prev = eb_ts
                            i += 1
                    stop = True
    
    return pairs


def _findCombinations(activity_types: List[str]) -> List[Tuple[str, str]]:
    """
    Extracts all the possible event type pairs that can occur in a trace based on the unique event types

    Args:
        activity_types: The unique event types in this trace
    Returns:
        list: A list of all the possible event type pairs that can occur in this trace
    """
    return [(t1, t2) for t1 in activity_types for t2 in activity_types]


def extract_counts(pairs_index: DataFrame) -> DataFrame:
    """
    Extracts statistics for each event type pair.

    For every (eventA, eventB) pair, computes:
      - total_duration: sum of durations (in seconds) across all occurrences
      - total_completions: number of times this pair occurred
      - min_duration: minimum duration (seconds) across all occurrences
      - max_duration: maximum duration (seconds) across all occurrences
      - sum_squared_duration: sum of squared durations (for variance calculation)

    Args:
        pairs_index: DataFrame with schema matching EventPair (source, target,
            trace_id, source_timestamp, target_timestamp, ...)

    Returns:
        A DataFrame with columns:
            eventA, eventB, total_duration, total_completions,
            min_duration, max_duration, sum_squared_duration
    """

    # Timestamps are unix (milliseconds from the lookback arithmetic); convert to seconds
    duration_col = ((col("target_timestamp") - col("source_timestamp")) / 1000).alias("duration")

    count_table = (pairs_index
        .withColumn("duration", (col("target_timestamp").cast("long") - col("source_timestamp").cast("long")) / 1000)
        .groupBy("source", "target")
        .agg(
            spark_sum("duration").alias("total_duration").cast("float"),
            count("duration").alias("total_completions").cast("integer"),
            spark_min("duration").alias("min_duration").cast("float"),
            spark_max("duration").alias("max_duration").cast("float"),
            spark_sum(spark_pow(col("duration"), 2)).alias("sum_squared_duration"),
        ))

    return count_table

def _parse_lookback(lookback: str) -> Tuple[int, LookbackType]:
    """
    Returns a tuple(lookback_number, LookbackType)
    lookback_number is in ms for time and positions for index
    format: for time $d$m$s for index $i eg. 25d16m0.5s or 255i 
    """
    if "i" in lookback:
        return (int(lookback.split("i")[0]), 'index')
    else:
        time_matches = list(_TIME_PATTERN.finditer(lookback))
        if not time_matches:
            raise ValueError(f"Invalid lookback format: {lookback}")
        total_ms = 0.0
        for match in time_matches:
            value = float(match.group(1))
            unit = match.group(2)

            if unit == "d":
                total_ms += value * 24 * 60 * 60 
            elif unit == "h":
                total_ms += value * 60 * 60 
            elif unit == "m":
                total_ms += value * 60 
            elif unit == "s":
                total_ms += value 
            else:
                raise ValueError(f"Unsupported unit: {unit}")

        return int(total_ms), "time"

def update_last_checked(previous_last_checked: DataFrame|None, current_last_checked: DataFrame, batch_min_ts: int, batch_min_pos: int, real_lookback: Tuple[int, LookbackType]) -> DataFrame:
     # Merge old and new last_checked, prefer new record where available, keep old otherwise
    if previous_last_checked is not None:
        updated_trace_ids = current_last_checked.select("trace_id").distinct()
        unchanged = previous_last_checked.join(
            updated_trace_ids, on="trace_id", how="left_anti"
        )
        merged_last_checked_df = unchanged.union(current_last_checked)
    else:
        merged_last_checked_df = current_last_checked

    # Step 2: Prune records older than lookback relative to the minimum timestamp in the current batch
    # Only applicable for time-based lookback - index-based has no time dimension to prune on
    if real_lookback[1] == "time":
        return merged_last_checked_df.filter((col("last_checked_moment") >= batch_min_ts) | ((batch_min_ts - col("last_checked_moment")) < real_lookback[0] / 1000))
    else:
        return merged_last_checked_df.filter((col("last_checked_moment") >= batch_min_pos) | ((batch_min_pos - col("last_checked_moment")) < real_lookback[0]))
