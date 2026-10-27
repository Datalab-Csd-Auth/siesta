from typing import Any, Dict
from siesta.core.interfaces import StorageManager
from siesta.core.logger import timed
from siesta.core.sparkManager import get_spark_session, cleanup as spark_cleanup
from siesta.model.StorageModel import MetaData
from siesta.modules.index.builders import *


def build_index(index_config: Dict[str, Any], storage: StorageManager, caller: str = "cli") -> MetaData:
    # Create a metadata object that will overwrite existing metadata
    # according to new indexing task (based on the index_config)
    metadata = MetaData(
        storage_namespace=index_config.get("storage_namespace", "siesta"),
        log_name=index_config.get("log_name", "default_log"),
        storage_type=index_config.get("storage_type", "s3")
    )


    # # Load existing metadata from storage if available
    storage.read_metadata_table(metadata)

    seq_df = timed(build_sequence_table, "Indexing.", index_config=index_config, metadata=metadata)
    activity_index_df = timed(build_activity_index, "Indexing.", events_df=seq_df, metadata=metadata)

    # seq_df (checkpointed in update_event_positions) is no longer needed in batch mode
    if not isinstance(seq_df, StreamingQuery):
        seq_df.unpersist()

    if isinstance(activity_index_df, StreamingQuery):
        build_last_checked_index_and_count_streamed(index_config, metadata, batch_activity_index_df=activity_index_df)
    else:
        pairs_df, _ = timed(build_last_checked_table, "Indexing.", index_config, metadata, batch_activity_index_df=activity_index_df)
        # pairs_df is cached inside build_last_checked_table

        timed(build_pairs_index, "Indexing.", index_config, metadata, pairs_df)

        timed(build_count_table, "Indexing.", index_config, metadata, pairs_df)

        storage.write_metadata_table(metadata)

        # Release the memory occupied by pairs_df now it's not needed
        pairs_df.unpersist()

        # Release cached data and Delta metadata after batch processing
        spark_cleanup()

        # In CLI mode, we want to keep streaming jobs alive until termination
        if caller == "cli" and index_config.get("enable_streaming", False):
            get_spark_session().streams.awaitAnyTermination()

    return metadata
