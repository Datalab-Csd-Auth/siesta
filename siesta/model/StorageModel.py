from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from . import DataModel
import xxhash

class MetaData:
    log_name: str
    storage_namespace: str
    storage_type: str
    
    trace_count: int
    event_count: int

    pair_count: int

    first_timestamp: int
    last_timestamp: int
    last_mined_timestamp: int

    # S3 Table paths
    @property
    def s3_pairs_index(self) -> str:
        return f"s3a://{self.storage_namespace}/{self.log_name}/pairs_index"
    @property
    def s3_sequence_table(self) -> str:
        return f"s3a://{self.storage_namespace}/{self.log_name}/sequence_table"
    @property
    def s3_metadata_table(self) -> str:
        return f"s3a://{self.storage_namespace}/{self.log_name}/metadata_table"
    @property
    def s3_activity_index(self) -> str:
        return f"s3a://{self.storage_namespace}/{self.log_name}/activity_index"
    @property
    def s3_count_table(self) -> str:
        return f"s3a://{self.storage_namespace}/{self.log_name}/count_table"
    @property
    def s3_last_checked_table(self) -> str:
        return f"s3a://{self.storage_namespace}/{self.log_name}/last_checked_table"
    @property
    def s3_trace_metadata_table(self) -> str:
        return f"s3a://{self.storage_namespace}/{self.log_name}/trace_metadata_table"
    @property
    def s3_mining(self) -> str:
        return f"s3a://{self.storage_namespace}/{self.log_name}/declare_constraints/"
    @property
    def s3_positional_constraints(self) -> str:
        return self.s3_mining + "positional.parquet"
    @property
    def s3_existential_constraints(self) -> str:
        return self.s3_mining + "existential.parquet"
    @property
    def s3_ordered_constraints(self) -> str:
        return self.s3_mining + "ordered.parquet"
    @property
    def s3_unordered_constraints(self) -> str:
        return self.s3_mining + "unordered.parquet"
    @property
    def s3_negation_constraints(self) -> str:
        return self.s3_mining + "negation.parquet"
    @property
    def s3_all_activity_pairs(self) -> str:
        return self.s3_mining + "all_activity_pairs"
    @property
    def s3_active_pairs_table(self) -> str:
        return f"s3a://{self.storage_namespace}/{self.log_name}/active_pairs_table"
    
    # Table paths for Storage Managers (extend for more compatibility)
    @property
    def pairs_index_path(self) -> str:
        return self.s3_pairs_index if self.storage_type == "s3" else ""
    @property
    def sequence_table_path(self) -> str:
        return self.s3_sequence_table if self.storage_type == "s3" else ""
    @property
    def metadata_table_path(self) -> str:
        return self.s3_metadata_table if self.storage_type == "s3" else ""
    @property
    def activity_index_path(self) -> str:
        return self.s3_activity_index if self.storage_type == "s3" else ""
    @property
    def count_table_path(self) -> str:
        return self.s3_count_table if self.storage_type == "s3" else ""
    @property
    def last_checked_table_path(self) -> str:
        return self.s3_last_checked_table if self.storage_type == "s3" else ""
    @property
    def trace_metadata_table_path(self) -> str:
        return self.s3_trace_metadata_table if self.storage_type == "s3" else ""

########################################################

    @property
    def mining_path(self) -> str:
        return self.s3_mining if self.storage_type == "s3" else ""
    @property
    def positional_constraints_path(self) -> str:
        return self.s3_positional_constraints if self.storage_type == "s3" else ""
    @property
    def existential_constraints_path(self) -> str:
        return self.s3_existential_constraints if self.storage_type == "s3" else ""
    @property
    def ordered_constraints_path(self) -> str:
        return self.s3_ordered_constraints if self.storage_type == "s3" else ""
    @property
    def unordered_constraints_path(self) -> str:
        return self.s3_mining + "unordered.parquet" if self.storage_type == "s3" else ""
    @property
    def negation_constraints_path(self) -> str:
        return self.s3_negation_constraints if self.storage_type == "s3" else ""
    @property
    def all_activity_pairs_path(self) -> str:
        return self.s3_all_activity_pairs if self.storage_type == "s3" else ""
    @property
    def active_pairs_table_path(self) -> str:
        return self.s3_active_pairs_table if self.storage_type == "s3" else ""


    def __init__(self, storage_namespace: str = "siesta", storage_type: str = "s3", log_name: str = "default_log"):
        self.storage_namespace = storage_namespace
        self.log_name = log_name
        self.storage_type = storage_type

    @staticmethod
    def get_schema() -> StructType:
        """Return the Spark schema for MetaData serialization."""
        return StructType([
            StructField("log_name", StringType(), False),
            StructField("storage_namespace", StringType(), False),
            StructField("trace_count", IntegerType(), False),
            StructField("event_count", IntegerType(), False),
            StructField("pair_count", IntegerType(), False),
            StructField("first_timestamp", IntegerType(), True),
            StructField("last_timestamp", IntegerType(), True),
            StructField("last_mined_timestamp", IntegerType(), True),
        ])

    def to_dict(self) -> dict:
        return {
            "log_name": self.log_name,
            "storage_namespace": self.storage_namespace,
            "trace_count": getattr(self, 'trace_count', 0),
            "event_count": getattr(self, 'event_count', 0),
            "pair_count": getattr(self, 'pair_count', 0),
            "first_timestamp": getattr(self, 'first_timestamp', None),
            "last_timestamp": getattr(self, 'last_timestamp', None),
            "last_mined_timestamp": getattr(self, 'last_mined_timestamp', None),
        }
    
    def __str__(self) -> str:
        return self.to_dict().__str__()


class IndexTableEntry(DataModel.EventPair):
    def to_dict(self) -> dict:
        return super().to_dict()
    # @property
    # def trace_id(self) -> str:
    #     if self.source.trace_id != self.target.trace_id:
    #         raise ValueError("Source and target events do not belong to the same trace.")
    #     return self.source.trace_id

class SequenceTableEntry(DataModel.Event):
    @staticmethod
    def get_schema() -> StructType:
        """Return the Spark schema for SequenceTableEntry serialization."""
        return DataModel.Event.get_schema()
    
    def to_dict(self) -> dict:
        return super().to_dict()


class ConstraintEntry:
    template: str
    source: str
    trace_id: str
    target: str | None
 
    occurrences: int | None
 

    def __init__(self):
        self.template = ""
        self.source = ""
        self.trace_id = ""
        self.target = None
        self.occurrences = None

    def to_dict(self) -> dict:
        return {
            "template": self.template,
            "source": self.source,
            "trace_id": self.trace_id,
            "target": self.target,
            "occurrences": self.occurrences,
        }
    
    @staticmethod
    def get_schema() -> StructType:
        return StructType([
            StructField("template", StringType(), False),
            StructField("source", StringType(), False),
            StructField("trace_id", StringType(), False),
            StructField("target", StringType(), True),
            StructField("occurrences", IntegerType(), True),
        ])

def hash_str(string: str) -> int:
    """
    Generate a consistent, deterministic 128-bit integer hash for a given string.
    """
    hash_bytes = xxhash.xxh128(string.encode('utf-8')).digest()
    return int.from_bytes(hash_bytes, byteorder='big')
