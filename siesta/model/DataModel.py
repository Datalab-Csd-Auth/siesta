from datetime import datetime
from typing import Optional
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, MapType, DoubleType, FloatType
import logging
logger = logging.getLogger(__name__)

DEFAULT_COMPOSITE_SEPARATOR = "::"


class EventConfig:
    """Configuration for mapping source data attributes to Event fields.
    
    Loads field mappings from system config. To customize mappings, edit config.json
    or pass a custom config to parse functions.
    """
    
    def __init__(self,
                 field_mappings: dict[str, Optional[str | list[str]]],
                 trace_level_fields: set[str],
                 timestamp_fields: set[str],
                 attributes_mapping: Optional[list[str]] = None,
                 composite_separator: str = DEFAULT_COMPOSITE_SEPARATOR):
        """
        Initialize EventConfig with field mappings.

        Args:
            field_mappings: Dict mapping Event field names to source attribute keys.
                          A single key (str) maps the field to one source column; a list of
                          keys maps it to the combination of those columns, joined with
                          composite_separator. Use None as value for computed fields.
            trace_level_fields: Set of fields extracted from trace level
            timestamp_fields: Set of fields containing timestamps
            attributes_mapping: List of source keys to include in attributes.
                              None/Empty means store nothing. ["*"] means store all unmapped.
            composite_separator: String used to join the values of a multi-column mapping.
        """
        self.field_mappings = field_mappings
        self.trace_level_fields = trace_level_fields
        self.timestamp_fields = timestamp_fields
        self.attributes_mapping = attributes_mapping
        self.composite_separator = composite_separator

    @staticmethod
    def from_preprocess_config(config: dict, log_format: str = 'xes') -> 'EventConfig':
        """Create EventConfig from preprocess configuration.
        
        Args:
            config: Preprocess configuration dictionary
            log_format: Log format to use ('xes', 'csv', 'json', or custom defined in config)
            
        Returns:
            EventConfig initialized from preprocess config
        """
        field_mappings = config.get('field_mappings', {}).get(log_format, {}).copy()

        attributes_mapping = field_mappings.pop('attributes', None)

        if not field_mappings:
            # Fallback to default XES for default Event class
            field_mappings = {
                'activity': 'concept:name',
                'trace_id': 'concept:name',
                'position': None,
                'start_timestamp': 'time:timestamp'
            }
        
        expected_attrs = set(dir(Event()))
        field_mappings_keys = set(field_mappings.keys())
        if not field_mappings_keys <= expected_attrs:
            logger.error(f"Given field mappings: {field_mappings_keys}")
            logger.error(f"Expected Event fields: {expected_attrs}")
            raise Exception("Incompatible field mappings given in Config")
        
        return EventConfig(
            field_mappings=field_mappings,
            trace_level_fields=set(config.get('trace_level_fields', ['trace_id'])),
            timestamp_fields=set(config.get('timestamp_fields', ['start_timestamp'])),
            attributes_mapping=attributes_mapping,
            composite_separator=config.get('composite_separator', DEFAULT_COMPOSITE_SEPARATOR)
        )

    @staticmethod
    def as_source_keys(mapping: Optional[str | list[str]]) -> list[str]:
        """Normalise a single mapping value into an ordered list of source keys.

        Accepts the single-column form (str), the composite form (list of str) and the
        computed form (None / empty), so callers never branch on the union type.
        """
        if mapping is None:
            return []
        if isinstance(mapping, str):
            return [mapping]
        return [k for k in mapping if k]

    def get_source_keys(self, field_name: str) -> list[str]:
        """Ordered source keys backing an Event field ([] for computed fields)."""
        return self.as_source_keys(self.field_mappings.get(field_name))

    def all_source_keys(self) -> set[str]:
        """Every source key referenced by any mapping, composite components included."""
        return {k for v in self.field_mappings.values() for k in self.as_source_keys(v)}

    def is_composite_field(self, field_name: str) -> bool:
        """Check if a field is built from more than one source column."""
        return len(self.get_source_keys(field_name)) > 1

    def get_event_fields(self) -> dict[str, Optional[str | list[str]]]:
        """Get mappings for event-level fields only."""
        return {k: v for k, v in self.field_mappings.items() if k not in self.trace_level_fields}

    def get_trace_fields(self) -> dict[str, Optional[str | list[str]]]:
        """Get mappings for trace-level fields only."""
        return {k: v for k, v in self.field_mappings.items() if k in self.trace_level_fields}

    def is_timestamp_field(self, field_name: str) -> bool:
        """Check if a field should be parsed as timestamp."""
        return field_name in self.timestamp_fields

    def is_computed_field(self, field_name: str) -> bool:
        """Check if a field is computed (not extracted from source)."""
        return not self.get_source_keys(field_name)

    def get_event_schema(self) -> StructType:
        """Return the Spark schema for Event based on field mappings."""
        return StructType([StructField(str(x), StringType(), True) for x in self.field_mappings.keys()])
    
    def get_source_schema(self) -> StructType:
        """Return the Spark schema using source field names for parsing raw data."""
        fields = []
        seen = set()
        for event_field, source_field in self.field_mappings.items():
            # Skips computed fields; a composite mapping contributes one field per component
            for source_key in self.as_source_keys(source_field):
                if source_key not in seen:
                    fields.append(StructField(str(source_key), StringType(), True))
                    seen.add(source_key)
        for attr in (self.attributes_mapping or []):
            if attr == "*" or attr in seen:
                continue
            fields.append(StructField(str(attr), StringType(), True))
            seen.add(attr)
        return StructType(fields)
    
    def __reduce__(self):
        return (self.__class__, (self.field_mappings, self.trace_level_fields,
                             self.timestamp_fields, self.attributes_mapping,
                             self.composite_separator))


class Event:
    activity: str
    
    trace_id: str
    position: int
    
    start_timestamp: Optional[int]

    attributes: Optional[dict[str, str | int | float | bool]]

    def __init__(self, activity: str = "", trace_id: str = "", position: int = -1,
                 start_timestamp: Optional[int] = None, attributes: Optional[dict] = None, **kwargs):
        self.activity = activity
        self.trace_id = trace_id
        self.position = position
        self.start_timestamp = start_timestamp
        self.attributes = attributes if attributes else {}
        
        # Support dynamic field assignment for extensibility
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        """Create Event from dictionary with dynamic field support."""
        return cls(**data)

    @staticmethod
    def get_schema() -> StructType:
        """Return the Spark schema for Event serialization."""
        return StructType([
            StructField("activity", StringType(), False),
            StructField("trace_id", StringType(), True),
            StructField("position", IntegerType(), False),
            StructField("start_timestamp", IntegerType(), True),
            StructField("attributes", MapType(StringType(), StringType()), True)
        ])

    def to_dict(self) -> dict:
        r = {
            "activity": self.activity,
            "trace_id": self.trace_id if self.trace_id else None,
            "position": self.position,
            "start_timestamp": self.start_timestamp,
            "attributes": self.attributes if self.attributes else {}
        }
        return r


class EventPair:
    source: Event
    target: Event
    @property
    def trace_id(self) -> str:
        return self.source.trace_id
    @property
    def start_position(self) -> int:
        return self.source.position
    @property
    def end_position(self) -> int:
        return self.target.position
    @property
    def start_timestamp(self) -> Optional[int]:
        return self.source.start_timestamp
    
    @property
    def position_diff(self) -> int:
        return self.target.position - self.source.position
    @property
    def start_timestamp_diff(self) -> Optional[int]:
        if self.source.start_timestamp and self.target.start_timestamp:
            return self.target.start_timestamp - self.source.start_timestamp
        return None

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict() if self.source else None,
            "target": self.target.to_dict() if self.target else None,
            "position_diff": self.position_diff,
            "start_timestamp_diff": self.start_timestamp_diff
        }
    
    @staticmethod
    def get_schema() -> StructType:
        return StructType([
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("trace_id", StringType(), False),
            StructField("source_timestamp", IntegerType(), False),
            StructField("target_timestamp", IntegerType(), False),
            StructField("source_position", IntegerType(), False),
            StructField("target_position", IntegerType(), False),
            StructField("source_attributes", MapType(StringType(), StringType()), True),
            StructField("target_attributes", MapType(StringType(), StringType()), True)
        ])



class Trace:
    events: list[Event]
    @property
    def trace_id(self) -> str:
        return self.events[0].trace_id
    @property
    def start_position(self) -> int:
        return 0
    @property
    def end_position(self) -> int:
        return len(self.events) - 1
    @property
    def start_timestamp(self) -> Optional[int]:
        if self.events[0].start_timestamp:
            return self.events[0].start_timestamp
        return None
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "events": [event.to_dict() for event in self.events] if self.events else [],
            "start_position": self.start_position,
            "end_position": self.end_position,
            "start_timestamp": self.start_timestamp
        }
    

Last_Checked_table_schema = StructType([
            StructField("trace_id", StringType(), True),
            StructField("source", StringType(), False),
            StructField("target", StringType(), False),
            StructField("last_checked_moment", IntegerType(), False)
])

count_table_schema = StructType([
    StructField("source", StringType(), False),
    StructField("target", StringType(), False),
    StructField("total_duration", FloatType(), False),
    StructField("total_completions", IntegerType(), False),
    StructField("min_duration", FloatType(), False),
    StructField("max_duration", FloatType(), False),
    StructField("sum_squared_duration", DoubleType(), False),
])

Trace_metadata_table_schema = StructType([
    StructField("trace_id", StringType(), False),
    StructField("max_pos", IntegerType(), True)
])