from typing import Annotated, Any, Callable, Dict, ClassVar, Literal, Tuple, TypeAlias
from abc import ABC, abstractmethod
from fastapi import UploadFile
from fastapi.params import Form
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType
from pyspark import RDD

from siesta.model.StorageModel import MetaData

class SiestaModule(ABC):
    """All Siesta modules must implement to be accessed by the framework.
       eg. ::
        class PreprocessorModule(SiestaModule):
            name = "preprocessor"

            def register_routes(self):
                return {
                    "preprocess": self.preprocess_endpoint
                }

            def startup(self):
                # Initialization code here
                pass
            ...
    """
    
    name: ClassVar[str] = "Unnamed Module"
    version: ClassVar[str] = "unversioned"

    ApiMethod: TypeAlias = Literal["GET", "POST", "PUT", "DELETE"]
    ApiRoute: TypeAlias = Tuple[ApiMethod, Callable] | Tuple[ApiMethod, Callable, Dict[str, Any]]
    ApiRoutes: TypeAlias = Dict[str, ApiRoute]

    # @abstractmethod
    def register_routes(self) -> ApiRoutes | None:
        """Return a dict of endpoint_name -> function."""
        pass

    def startup(self) -> None:
        """Lifecycle hook: Called when the framework starts."""
        pass

    def cli_run(self, args: Any, **kwargs: Any) -> Any:
        """Main execution method for the module."""
        pass



class StorageManager(ABC):

    name: ClassVar[str] = "Unnamed Storage Manager"
    version: ClassVar[str] = "unversioned"
    type: ClassVar[str] = "generic"

    """
    This abstract class defines the interface that each database connector should implement.
    
    There are 3 types of methods:
    - write and read methods: should be implemented by each database, as they might have different
      indexing or structure
    - combine methods: describe how the data already stored in the database are combined with newly
      calculated data
    - spark related methods: initialization, building appropriate tables, and termination
    
    The tables that are utilized are:
    - PairsIndexTable: contains the inverted index based on event type pairs
    - SequenceTable: contains the traces that are indexed
    - ActivityIndexTable: contains the activity inverted index (similar to Set-Containment)
    - ActivePairTable: contains the timestamp of the last completion of each event type per trace
    - CountTable: contains basic statistics (max duration, number of completions) for each event type pair
    - MetadataTable: contains information for each log database (compression algorithm, number of traces, etc.)
    """
    
    @abstractmethod
    def initialize_spark(self) -> None:
        """
        Initialize the Spark context depending on the database requirements.

        """
        pass
    
    @abstractmethod
    def initialize_db(self, preprocess_config: Dict[str, Any] = {}) -> None:
        """
        Create the appropriate tables and remove previous ones if necessary.
        
        """
        pass
    
    @abstractmethod
    def initialize_streaming_collector(self, preprocess_config: Dict[str, Any] = {}) -> Any:
        """
        Initialize streaming context if necessary.

        Args:
            preprocess_config: Configuration dictionary containing streaming settings
        """
        pass
    
    @abstractmethod
    def get_steaming_collector_path(self, preprocess_config: Dict[str, Any]) -> str:
        """
        Get the path where the streaming collector stores data.

        Args:
            preprocess_config: Configuration dictionary containing streaming settings

        Returns:
            Path as a string
        """
        pass
    
    @abstractmethod
    def get_checkpoint_location(self, metadata: MetaData, checkpoint_table: str = "table") -> str:
        """
        Get the S3 path for streaming checkpoint location.
        
        Args:
            metadata: MetaData object containing checkpoint settings
            checkpoint_type: Type of checkpoint (e.g., 'index', 'collector')
            
        Returns:
            Path as a string
        """
        pass
    
    @abstractmethod
    def list_namespaces(self) -> list[str]:
        """
        List all storage namespaces (e.g. S3 buckets) available in the storage backend.

        Returns:
            List of namespace names.
        """
        pass

    @abstractmethod
    def list_logs(self, storage_namespace: str) -> list[str]:
        """
        List all log names stored under a given storage namespace.

        Args:
            storage_namespace: Name of the storage namespace to list logs from.

        Returns:
            List of log names.
        """
        pass

    @abstractmethod
    def log_exists(self, task_config: Dict[str, Any]) -> bool:
        """
        Check whether a log dataset already exists in the storage backend.

        Args:
            task_config: Configuration dictionary containing at least 'log_name'
                         and 'storage_namespace'.

        Returns:
            True if the log exists, False otherwise.
        """
        pass

    @abstractmethod
    def delete_log(self, metadata: MetaData) -> bool:
        """
        Permanently delete all stored tables for a single log within a namespace.

        Args:
            metadata: MetaData object identifying the storage_namespace and log_name to delete.

        Returns:
            True if the deletion completed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def delete_namespace(self, storage_namespace: str) -> bool:
        """
        Permanently delete an entire storage namespace (e.g. an S3 bucket) and everything under it.

        Args:
            storage_namespace: Name of the namespace to delete.

        Returns:
            True if the deletion completed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def list_tables(self, metadata: MetaData) -> list[str]:
        """
        List the canonical table names available for a given log, for ad-hoc administrative
        access independent of the analytical pipelines.

        Args:
            metadata: MetaData object identifying the storage_namespace and log_name.

        Returns:
            List of table names that can be passed to `read_table`.
        """
        pass

    @abstractmethod
    def read_table(self, metadata: MetaData, table_name: str) -> DataFrame:
        """
        Load an arbitrary named table (as returned by `list_tables`) as a DataFrame, for
        ad-hoc SQL access independent of the analytical pipelines.

        Args:
            metadata: MetaData object identifying the storage_namespace and log_name.
            table_name: One of the names returned by `list_tables`.

        Returns:
            DataFrame containing the raw contents of the requested table.
        """
        pass

    @abstractmethod
    def read_metadata_table(self, metadata: MetaData) -> MetaData:
        """
        Construct metadata based on data already stored in the database and new configuration.
        
        Args:
            metadata: MetaData object to be updated with information from the database
        Returns:
            MetaData object containing the metadata
        """
        pass

    @abstractmethod
    def upload_file(self, preprocess_config: Dict[str, Any], local_path: str, destination_path: str) -> str:
        """
        Upload a local file to the storage system.
        
        Args:
            preprocess_config: Configuration dictionary passed during execution
            local_path: Path to the local file
            destination_path: Path/Name for the file in storage
            
        Returns:
            The URI to access the uploaded file (e.g., s3a://bucket/path)
        """
        pass

    @abstractmethod
    def upload_file_object(self, preprocess_config: Dict[str, Any], file_obj: Any, destination_path: str) -> str:
        """
        Upload a file-like object to the storage system.
        
        Args:
            preprocess_config: Configuration dictionary passed during execution
            file_obj: File-like object to upload
            destination_path: Path/Name for the file in storage
        Returns:
            The URI to access the uploaded file (e.g., s3a://bucket/path)
        """
        pass
    
    @abstractmethod
    def write_metadata_table(self, metadata: Any) -> None:
        """
        Persist metadata to the database.
        
        Args:
            metadata: MetaData object containing the metadata
        """
        pass
    
    
    @abstractmethod
    def read_sequence_table(self, metadata: MetaData, filter_out: Any | None = None) -> DataFrame:
        """
        Read data as an DataFrame from the SequenceTable.
        
        Args:
            metadata: MetaData object containing the metadata
            
        Returns:
            DataFrame containing EventTrait objects
        """
        pass
    
    @abstractmethod
    def read_activity_index(self, metadata: Any) -> DataFrame:
        """
        Load the activity index from the database (stored in activity_index).
        
        Args:
            metadata: MetaData object containing the metadata
            
        Returns:
            DataFrame containing Event objects
        """
        pass
    
    @abstractmethod
    def read_last_checked_table(self, metadata: Any) -> DataFrame:
        """
        Load data from the LastCheckedTable, containing the last timestamp per event type pair per trace.
        
        Args:
            metadata: MetaData object containing the metadata
            
        Returns:
            DataFrame with last timestamps per event type pair per trace
        """
        pass

    @abstractmethod
    def read_pairs_index(self, metadata: Any) -> DataFrame:
        """
        Load data from the Pairs Index

        Args: 
            metadata: MetaData object containing the metadata
        Returns:
            DataFrame indexed by Pairs
        """
    
    @abstractmethod
    def write_last_checked_table(self, last_checked: DataFrame, metadata: Any) -> None:
        """
        Store new records for last checked timestamps back in the database.
        
        Args:
            last_checked: DataFrame containing timestamp of last completion for each event type pair per trace
            metadata: MetaData object containing the metadata
        """
        pass
    
    @abstractmethod
    def read_count_table(self, metadata: MetaData) -> DataFrame:
        pass

    @abstractmethod
    def write_count_table(self, count_df: DataFrame, metadata: Any) -> None:
        """
        Write count statistics to the CountTable.
        
        Args:
            count_df: Dataframe containing calculated basic statistics per event type pair
            metadata: MetaData object containing the metadata
        """
        pass
    
    @abstractmethod
    def write_sequence_table(self, events_df: DataFrame, metadata: MetaData) -> None:
        """
        Write traces to the SequenceTable. Updates the metadata object.
        
        Args:
            events_df: DataFrame containing traces with new events
            metadata: MetaData object containing the metadata
        """
        pass
    
    @abstractmethod
    def write_activity_index(self, events_df: DataFrame, metadata: MetaData) -> None:
        """
        Write events to the activity index table. Updates the metadata object.
        
        Args:
            events_df: DataFrame containing newly indexed events in activity index index form
            metadata: MetaData object containing the metadata
        """
        pass
    
    @abstractmethod
    def write_pairs_index(self, new_pairs: DataFrame, metadata: Any) -> None:
        """
        Write the combined pairs back to the database, grouped by interval and first event.
        
        Args:
            new_pairs: DataFrame containing newly generated pairs
            metadata: MetaData object containing the metadata
        """
        pass
    
    @staticmethod
    def _complete_schema(df: DataFrame, target_schema: StructType) -> DataFrame:
        """
        Ensure a DataFrame conforms to the given target schema by adding any missing
        columns as nulls (cast to the correct type) and reordering to match schema order.

        Args:
            df: Input DataFrame that may be missing some columns.
            target_schema: The expected StructType schema.

        Returns:
            DataFrame with all schema fields present in order.
        """
        existing_cols = set(df.columns)
        for field in target_schema.fields:
            if field.name not in existing_cols:
                df = df.withColumn(field.name, lit(None).cast(field.dataType))
        return df.select([field.name for field in target_schema.fields])

    @abstractmethod
    def read_positional_constraints(self, metadata: MetaData, filter_out_df: DataFrame | None = None) -> DataFrame:
        """
        Read existing positional constraints from storage as flat ConstraintEntry rows.

        Args:
            metadata: MetaData object containing the metadata of the log dataset
            filter_out_df: Optional DataFrame of evolved traces; End constraints for those
                           traces are excluded, while Init constraints are always included.
        Returns:
            DataFrame[ConstraintEntry] containing the flat positional constraint rows.
        """
        pass


    @abstractmethod
    def read_trace_metadata_table(self, metadata: Any) -> DataFrame:
        """
        Read the trace metadata table
        """
    
    @abstractmethod
    def write_trace_metadata_table(self, trace_metadata_df: DataFrame, metadata: Any) -> None:
        """
        Write the Trace metadata table to the DB
        """
    
  ######################################################################################
    @abstractmethod
    def write_positional_constraints(self, metadata: MetaData, df: DataFrame) -> None:
        """
        Write positional constraints to storage.
        """
        pass

    @abstractmethod
    def write_existential_constraints(self, metadata: MetaData, df: DataFrame) -> None:
        """
        Write existential constraints to storage.
        """
        pass

    @abstractmethod
    def write_ordered_constraints(self, metadata: MetaData, df: DataFrame) -> None:
        """
        Write ordered constraints to storage.
        """
        pass

    @abstractmethod
    def read_existential_constraints(self, metadata: MetaData) -> DataFrame:
        """
        Read existing existential constraints from S3.
        
        Args:
            metadata: MetaData object containing the metadata of the log dataset
        Returns:
            DataFrame[Constraint] containing the existential constraints
        """
        pass

    @abstractmethod
    def read_ordered_constraints(self, metadata: MetaData) -> DataFrame:
        """
        Read existing ordered constraints from storage as flat ConstraintEntry rows.

        Args:
            metadata: MetaData object containing the metadata of the log dataset
        Returns:
            DataFrame with columns (template, source, target, trace_id)
        """
        pass

    @abstractmethod
    def read_unordered_constraints(self, metadata: MetaData) -> DataFrame:
        """
        Read existing unordered constraints from storage as flat ConstraintEntry rows.

        Args:
            metadata: MetaData object containing the metadata of the log dataset
        Returns:
            DataFrame with columns (template, source, target, trace_id, occurrences)
        """
        pass
    
    @abstractmethod
    def write_unordered_constraints(self, metadata: MetaData, df: DataFrame) -> None:
        """
        Write unordered constraints to storage.
        """
        pass

    @abstractmethod
    def read_negation_constraints(self, metadata: MetaData) -> DataFrame:
        """
        Read existing negation constraints from storage as flat ConstraintEntry rows.

        Args:
            metadata: MetaData object containing the metadata of the log dataset
        Returns:
            DataFrame with columns (template, source, target, trace_id)
        """
        pass

    @abstractmethod
    def write_negation_constraints(self, metadata: MetaData, df: DataFrame) -> None:
        """
        Write negation constraints to storage.
        """
        pass

    @abstractmethod
    def read_all_activity_pairs(self, metadata: MetaData) -> DataFrame:
        """
        Read the incrementally maintained table of all activity pairs (source < target).

        Returns:
            DataFrame with columns (source, target)
        """
        pass

    @abstractmethod
    def write_all_activity_pairs(self, metadata: MetaData, df: DataFrame) -> None:
        """
        Overwrite the all-activity-pairs table.
        """
        pass