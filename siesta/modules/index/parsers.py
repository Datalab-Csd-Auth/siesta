from typing import Any, Dict
import pandas as pd
from lxml import etree
import tempfile
from siesta.core.sparkManager import get_spark_session
from siesta.core.storageFactory import get_storage_manager
from siesta.model.DataModel import Event, EventConfig
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import StringType, IntegerType, MapType, StructType, StructField
from pyspark.sql.functions import monotonically_increasing_id, lit
from datetime import datetime, timezone
import os
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col
import logging

logger = logging.getLogger(__name__)

from siesta.model.StorageModel import MetaData

def update_event_positions(eventsDF: DataFrame, metadata: MetaData):

    storage_manager = get_storage_manager()
    eventsDF = eventsDF.sort("start_timestamp")
    # Cache the current trace metadata to prevent re-reading after overwrite
    trace_metadata_table = storage_manager.read_trace_metadata_table(metadata).cache()
    trace_metadata_table.count()  # Force materialization before overwrite
    updated_events_df = eventsDF\
        .join(trace_metadata_table, on="trace_id", how="left")\
        .withColumn("position", F.col("position") + F.coalesce(F.col("max_pos"), F.lit(-1)) + 1)\
        .drop("max_pos")\
        .localCheckpoint()  # Materialize to decouple from trace_metadata_table
    updates_to_apply = updated_events_df.groupBy("trace_id").agg(F.max("position").alias("max_pos"))

    updated_metadata_table = trace_metadata_table\
        .join(updates_to_apply, on="trace_id", how="outer")\
        .select("trace_id",F.coalesce(updates_to_apply["max_pos"], trace_metadata_table["max_pos"])\
        .alias("max_pos"))
    
    storage_manager.write_trace_metadata_table(updated_metadata_table, metadata)
    trace_metadata_table.unpersist()

    return updated_events_df
    
def process_events_batch(preprocess_config: Dict, batch_df, batch_id, metadata: MetaData) -> None:
    """
    Core processing logic for event batches; parses and stores events in Sequence Table.
    Args:
        batch_df: DataFrame containing the batch of events
        batch_id: Optional batch identifier for logging
    """
    if batch_df.isEmpty():
        return
    try:
        event_config = EventConfig.from_preprocess_config(preprocess_config, "json")
        events_df = _parse_rows(event_config, batch_df)

        events_df = update_event_positions(events_df, metadata)

        get_storage_manager().write_sequence_table(events_df, metadata)
        events_df.unpersist()  # Release checkpointed data
    except Exception as e:
        batch_info = f"batch {batch_id}" if batch_id is not None else "batch"
        logger.error(f"Error processing {batch_info}: {e}")



def _cast_value_by_schema(field_name: str, value, config: EventConfig):
    """Cast a value to the appropriate type based on Event schema.
    
    Args:
        field_name: Name of the Event field
        value: Raw value to cast
        config: EventConfig instance for timestamp field detection
    
    Returns:
        Casted value according to Event schema
    """
    if value is None:
        return None
    
    schema = Event.get_schema()
    field_type = None
    
    # Find the field type in schema
    for field in schema.fields:
        if field.name == field_name:
            field_type = field.dataType
            break
    
    if field_type is None:
        # Field not in schema, return as string
        return str(value)
    
    # Cast based on type
    if isinstance(field_type, IntegerType):
        if config.is_timestamp_field(field_name):
            try:
                if isinstance(value, datetime):
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=timezone.utc)
                    return int(value.timestamp())
                elif isinstance(value, str):
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return int(dt.timestamp())
            except (ValueError, TypeError):
                return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    elif isinstance(field_type, StringType):
        return str(value)
    elif isinstance(field_type, MapType):
        # For attributes, ensure it's a dict with string values
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        return {}
    
    return value


def process_event_log(preprocess_config: dict, metadata: MetaData) -> DataFrame:
    """
    Generic parsing function that determines the format and path from config.
    """
    log_path = preprocess_config.get("log_path")
    if not log_path:
        raise ValueError("Log path not specified in configuration")
    filename = f"{int(datetime.now().timestamp())}_" + os.path.basename(log_path)
    spark = get_spark_session()
    if spark is None:
        raise RuntimeError("Spark session is not initialized.")
    
    storage = get_storage_manager()
    if not storage:
        raise RuntimeError("Storage manager is not initialized.")
    
    # If local, verify file exists and upload
    if os.path.exists(log_path):
        logger.info(f"Uploading {log_path} to storage...")
        log_path = storage.upload_file(preprocess_config, log_path, filename)
        logger.info(f"File uploaded to: {log_path}")
    
    # Else, assume log_path is already in storage (e.g., s3a://...)

    _, ext = os.path.splitext(filename)
    log_format = ext.lower().lstrip('.')
    
    if log_format == 'xes':
        events_df = parse_xml(log_path, spark, preprocess_config)
    elif log_format == 'csv':
        events_df = parse_csv(log_path, spark, preprocess_config)
    else:
        raise ValueError(f"Unsupported log format: {log_format}")
    
    events_df = update_event_positions(events_df, metadata)

    storage.write_sequence_table(events_df, metadata)
    return events_df


def parse_xml(storage_path: str, spark: SparkSession, preprocess_config: dict) -> DataFrame:
    """
    Parse XES log file using lxml's iterparse for streaming, memory-efficient parsing.

    XML files are not splittable, so Spark's XML reader processes them in a single
    partition anyway while adding schema-inference overhead and expensive nested-array
    operations. lxml (a C library) streams through the same file 10-50x faster and
    produces a flat list of events that converts to a Spark DataFrame via Arrow.

    Args:
        storage_path: Path to the XES file (local path or s3a:// URI)
        spark: Active Spark Session
        preprocess_config: Preprocess configuration dictionary

    Returns:
        DataFrame matching Event.get_schema()
    """
    
    eventConfig = EventConfig.from_preprocess_config(preprocess_config, "xes")

    # --- Resolve file to a local path (lxml needs a local file or file-like object) ---
    local_path = _resolve_to_local_path(storage_path, spark)

    # --- Build field-to-XES-key lookups from EventConfig ---
    trace_fields = eventConfig.get_trace_fields()   # e.g. {'trace_id': 'concept:name'} or {'trace_id': ['concept:name', 'org:resource']}
    event_fields = eventConfig.get_event_fields()    # e.g. {'activity': 'concept:name', 'start_timestamp': 'time:timestamp', 'position': None}
    timestamp_fields = eventConfig.timestamp_fields  # e.g. {'start_timestamp'}
    schema_type_map = {f.name: f.dataType for f in Event.get_schema().fields}

    def _is_single_valued(field_name: str) -> bool:
        # A timestamp or non-string target only ever takes its first source component -
        # concatenating several XES attribute values makes no sense for those types.
        if eventConfig.is_timestamp_field(field_name):
            return True
        return isinstance(schema_type_map.get(field_name), IntegerType)

    # Ordered source keys per field (component index within a composite mapping)
    trace_components = {f: keys for f, v in trace_fields.items() if (keys := EventConfig.as_source_keys(v))}
    event_components = {f: keys for f, v in event_fields.items() if (keys := EventConfig.as_source_keys(v))}

    # Invert: XES key -> [(field_name, component_index), ...]
    trace_key_map: dict[str, list[tuple[str, int]]] = {}
    for field_name, keys in trace_components.items():
        for idx, k in enumerate(keys):
            trace_key_map.setdefault(k, []).append((field_name, idx))

    event_key_map: dict[str, list[tuple[str, int]]] = {}
    for field_name, keys in event_components.items():
        for idx, k in enumerate(keys):
            event_key_map.setdefault(k, []).append((field_name, idx))

    def _join_components(field_name: str, keys: list[str], parts: dict[int, Any]) -> Any:
        """Resolve the raw component values for one field into its final value.

        Missing components are skipped (not treated as a hard failure); a field with
        no components present at all yields None, same as an unmapped XES key today.
        """
        if not parts:
            return None
        if _is_single_valued(field_name):
            if len(keys) > 1 and len(parts) > 1:
                logger.warning(
                    f"Field '{field_name}' only supports a single source column in XES; "
                    f"using '{keys[0]}', ignoring {keys[1:]}."
                )
            v = parts.get(0)
            if v and field_name in timestamp_fields:
                v = v[:19]  # trim to seconds ISO
            return v
        present_vals = [parts[i] for i in sorted(parts)]
        return present_vals[0] if len(present_vals) == 1 else eventConfig.composite_separator.join(present_vals)

    # Mapped source keys to exclude from attributes
    mapped_source_keys = frozenset(eventConfig.all_source_keys())
    attrs_mapping = eventConfig.attributes_mapping or []
    collect_attrs = bool(attrs_mapping)
    attrs_wildcard = collect_attrs and "*" in attrs_mapping
    attrs_keep_keys = (
        frozenset(attrs_mapping) - mapped_source_keys
        if collect_attrs and not attrs_wildcard
        else frozenset()
    )

    # --- XES tag names that carry key/value attributes ---
    XES_ATTR_TAGS = frozenset(('string', 'date', 'int', 'float', 'boolean'))

    # --- Streaming parse with lxml iterparse ---
    logger.info(f"Parsing XES file with lxml iterparse: {storage_path}")

    activities = []
    trace_ids = []
    positions = []
    timestamps = []
    attr_dicts = []

    cur_trace_vals: dict = {}  # trace-level field values
    cur_position = 0


    # Bypassing namespace for lxml - This may be jank
    def iter_no_ns(path):
        for event, elem in etree.iterparse(path, events=('start', 'end')):
            # This is the "Face Value" trick: Remove the namespace prefix from the tag
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1] 
            yield event, elem

    # --- Streaming parse ---
    context = iter_no_ns(local_path)

    for action, elem in context:
        
        if action == 'start' and elem.tag == 'trace':
            cur_trace_vals = {}
            cur_position = 0
            continue

        if action == 'end' and elem.tag == 'event':
            evt_partial: dict[str, dict[int, Any]] = {}
            attrs: dict = {}

            for child in elem:
                if child.tag in XES_ATTR_TAGS:
                    k = child.get('key')
                    v = child.get('value')

                    if k in event_key_map:
                        for field_name, idx in event_key_map[k]:
                            evt_partial.setdefault(field_name, {})[idx] = v
                    elif collect_attrs:
                        if attrs_wildcard and k not in mapped_source_keys:
                            attrs[k] = str(v) if v is not None else None
                        elif k in attrs_keep_keys:
                            attrs[k] = str(v) if v is not None else None

            evt_vals = {
                field_name: _join_components(field_name, keys, evt_partial.get(field_name, {}))
                for field_name, keys in event_components.items()
            }

            activities.append(evt_vals.get('activity'))
            trace_ids.append(cur_trace_vals.get('trace_id'))
            positions.append(cur_position)
            timestamps.append(evt_vals.get('start_timestamp'))
            attr_dicts.append(attrs)
            cur_position += 1
            elem.clear()

        elif action == 'end' and elem.tag == 'trace':
            # Extract trace-level fields from trace's direct children
            if not cur_trace_vals:
                trace_partial: dict[str, dict[int, Any]] = {}
                found_keys = 0
                for child in elem:
                    if child.tag in XES_ATTR_TAGS:
                        k = child.get('key')
                        if k in trace_key_map:
                            found_keys += 1
                            for field_name, idx in trace_key_map[k]:
                                trace_partial.setdefault(field_name, {})[idx] = child.get('value')
                    if found_keys == len(trace_key_map):
                        break  # found all distinct trace-level source keys, stop scanning

                for field_name, keys in trace_components.items():
                    value = _join_components(field_name, keys, trace_partial.get(field_name, {}))
                    if value is not None:
                        cur_trace_vals[field_name] = value

                # Backfill trace-level values for all events in this trace
                trace_id = cur_trace_vals.get('trace_id')
                start = len(trace_ids) - cur_position
                for i in range(start, len(trace_ids)):
                    trace_ids[i] = trace_id

            elem.clear()

    logger.info(f"Parsed {len(activities)} events from XES file.")

    # --- Build pandas DataFrame and convert to Spark via Arrow ---
    pdf = pd.DataFrame({
        'activity': activities,
        'trace_id': trace_ids,
        'position': positions,
        'start_timestamp': timestamps,
        'attributes': attr_dicts,
    })

    # Free memory from lists immediately
    del activities, trace_ids, positions, timestamps, attr_dicts

    # Reassign positions based on timestamp order within each trace
    pdf = pdf.sort_values(['trace_id', 'start_timestamp'], na_position='last')
    pdf['position'] = pdf.groupby('trace_id').cumcount()

    # Convert to Spark DataFrame (uses Arrow serialization automatically)
    # Use intermediate schema with StringType for timestamp fields so Spark can parse them
    event_schema = Event.get_schema()
    intermediate_schema = StructType([
        StructField(f.name, StringType() if f.name in timestamp_fields else f.dataType, f.nullable)
        for f in event_schema.fields
    ])
    events_df = spark.createDataFrame(pdf, schema=intermediate_schema)
    # Convert timestamp strings to Unix seconds (integer)
    for ts_field in timestamp_fields:
        events_df = events_df.withColumn(
            ts_field,
            F.unix_timestamp(F.col(ts_field), "yyyy-MM-dd'T'HH:mm:ss").cast("int")
        )
    # Final cast to enforce Event schema types
    events_df = events_df.select(*[
        F.col(f.name).cast(f.dataType).alias(f.name) for f in event_schema.fields
    ])

    logger.info("Spark DataFrame created from parsed XES data.")
    return events_df


def _resolve_to_local_path(storage_path: str, spark: SparkSession) -> str:
    """Download a remote file (e.g. s3a://) to a local temp file for lxml, or
    return as-is if already local."""
    if os.path.exists(storage_path):
        return storage_path

    # Remote path - use Hadoop FileSystem API to download
    logger.info(f"Downloading remote file to local temp: {storage_path}")

    jvm = spark.sparkContext._jvm
    if jvm is None:
        raise RuntimeError("Spark JVM gateway unavailable")
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    path = jvm.org.apache.hadoop.fs.Path(storage_path)
    fs = path.getFileSystem(hadoop_conf)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xes")
    local_path = tmp.name
    tmp.close()

    fs.copyToLocalFile(False, path, jvm.org.apache.hadoop.fs.Path(local_path))
    logger.info(f"Downloaded to {local_path}")
    return local_path


def parse_csv(storage_path: str, spark: SparkSession, system_config: dict) -> DataFrame:
    """
    Parse_csv: Parses CSV log file using Spark CSV reader and creates a DataFrame of Event objects.
    
    Args:
        storage_path: Path to the log file in storage
        spark: Active Spark Session
        system_config: System configuration dictionary
    
    Returns:
        DataFrame containing Event objects
    """
    config = EventConfig.from_preprocess_config(system_config, "csv")
    
    # Read CSV into DataFrame
    df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(storage_path)
    
    return _parse_rows(config, df)

def parse_json(storage_path: str, spark: SparkSession, system_config: dict) -> DataFrame:
    """
    Parse_json: Parses JSON log file using Spark JSON reader and creates an DataFrame of Event objects.
    
    Args:
        storage_path: Path to the log file in storage
        spark: Active Spark Session
        system_config: System configuration dictionary
    
    Returns:
        DataFrame containing Event objects
    """
    config = EventConfig.from_preprocess_config(system_config, "json")
    
    # Read JSON into DataFrame
    df = spark.read.format("json") \
        .option("multiline", "true") \
        .load(storage_path) 
    
    return _parse_rows(config, df)

def _present_source_keys(config: EventConfig, field_name: str, df: DataFrame) -> list[str]:
    """Source keys of a field that actually exist in the DataFrame, in declared order.

    Missing components of a composite mapping are dropped with a warning rather than
    failing the whole parse, mirroring how a missing single column is tolerated.
    """
    keys = config.get_source_keys(field_name)
    available = [k for k in keys if k in df.columns]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        logger.warning(
            f"Field '{field_name}': source column(s) {missing} not found in the log; "
            f"{'using ' + str(available) if available else 'field will be left unmapped'}."
        )
    return available


def _composite_column(config: EventConfig, keys: list[str]):
    """Combine one or more source columns into a single string Column.

    A single key is returned as-is; several keys are joined with the configured
    composite separator so the resulting value stays one StringType column.
    """
    if not keys:
        return None
    if len(keys) == 1:
        return F.col(keys[0])
    return F.concat_ws(config.composite_separator, *[F.col(k).cast("string") for k in keys])


def _parse_rows(config: EventConfig, df: DataFrame) -> DataFrame:
    """Parse raw DataFrame rows into Event schema using pure DataFrame operations.
    Avoids RDD operations entirely for optimal PySpark performance."""
    # Event/trace fields are disjoint by construction (partitioned by trace_level_fields),
    # so a plain merge is equivalent to the previous set union but tolerates list values
    # (composite mappings), which a set of (key, value) tuples cannot hash.
    fields = {**config.get_event_fields(), **config.get_trace_fields()}
    source_keys = config.all_source_keys()

    # Resolve the trace_id column (possibly a combination of columns) for partitioning
    trace_id_keys = _present_source_keys(config, 'trace_id', df)
    trace_id_col = _composite_column(config, trace_id_keys)

    # Ordering must use a single column - a concatenation orders lexicographically, which
    # is meaningless for a timestamp or a position, so only the first component is used.
    position_keys = _present_source_keys(config, 'position', df)
    timestamp_field = next(iter(config.timestamp_fields), None)
    timestamp_keys = _present_source_keys(config, timestamp_field, df) if timestamp_field else []

    # Preserve original row order as stable tiebreaker
    df = df.withColumn("_row_idx", monotonically_increasing_id())

    if timestamp_keys:
        order_col = F.col(timestamp_keys[0]).cast("timestamp")
    elif position_keys:
        order_col = F.col(position_keys[0])
    else:
        order_col = F.col("_row_idx")

    if trace_id_col is not None:
        df = df.withColumn("position", row_number().over(
            Window.partitionBy(trace_id_col).orderBy(order_col, F.col("_row_idx"))
        ) - 1)
    else:
        df = df.withColumn("position", row_number().over(
            Window.orderBy(order_col, F.col("_row_idx"))
        ) - 1)

    df = df.drop("_row_idx")

    schema = Event.get_schema()
    schema_type_map = {f.name: f.dataType for f in schema.fields}
    result_df = df

    # Map source columns to Event field names with appropriate type casts
    for field_name in fields:
        keys = _present_source_keys(config, field_name, df)
        if keys:
            target_type = schema_type_map.get(field_name)
            if config.is_timestamp_field(field_name):
                # Parse timestamp string to Unix seconds (integer)
                if len(keys) > 1:
                    logger.warning(
                        f"Field '{field_name}' is a timestamp; only its first source column "
                        f"'{keys[0]}' is used (columns {keys[1:]} ignored)."
                    )
                result_df = result_df.withColumn(
                    field_name,
                    F.unix_timestamp(F.col(keys[0]).cast("timestamp")).cast("int")
                )
            elif target_type and isinstance(target_type, IntegerType):
                if len(keys) > 1:
                    logger.warning(
                        f"Field '{field_name}' is numeric; only its first source column "
                        f"'{keys[0]}' is used (columns {keys[1:]} ignored)."
                    )
                result_df = result_df.withColumn(field_name, F.col(keys[0]).cast("int"))
            else:
                result_df = result_df.withColumn(
                    field_name, _composite_column(config, keys).cast("string")
                )
        elif config.is_computed_field(field_name):
            if field_name != 'position':  # position already assigned above
                result_df = result_df.withColumn(field_name, F.lit(None))

    # Build attributes map from unmapped columns
    _EMPTY_MAP = F.create_map().cast(MapType(StringType(), StringType()))
    original_cols = set(df.columns) - {'position'}
    unmapped_cols = sorted(original_cols - source_keys)

    if config.attributes_mapping and unmapped_cols:
        if "*" in config.attributes_mapping:
            attr_cols = unmapped_cols
        else:
            attr_cols = [c for c in unmapped_cols if c in config.attributes_mapping]

        if attr_cols:
            kv_pairs = []
            for c in attr_cols:
                kv_pairs.extend([F.lit(c), F.col(c).cast("string")])
            result_df = result_df.withColumn("attributes", F.create_map(*kv_pairs))
        else:
            result_df = result_df.withColumn("attributes", _EMPTY_MAP)
    else:
        result_df = result_df.withColumn("attributes", _EMPTY_MAP)

    # read_trace_metadata_table
    final_projection_df = result_df.select(*[
        F.col(f.name).cast(f.dataType).alias(f.name) if f.name in result_df.columns
        else F.lit(None).cast(f.dataType).alias(f.name)
        for f in schema.fields
    ])

    # Final projection matching Event schema
    return final_projection_df


def upload_log_file_object(preprocess_config: dict, file: Any, destination_path: str) -> str:
    """
    Uploads an in-memory log file (UploadFile) to storage and returns the S3 path.
    """
    storage = get_storage_manager()
    if not storage:
        raise RuntimeError("Storage manager is not initialized.")
    destination_path = f"{int(datetime.now().timestamp())}_" + destination_path
    logger.info(f"Uploading file object to storage as {destination_path}...")
    s3_path = storage.upload_file_object(preprocess_config, file, destination_path)
    logger.info(f"File uploaded to: {s3_path}")
    return s3_path

