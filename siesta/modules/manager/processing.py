import json
from typing import Any, Dict

from siesta.core.interfaces import StorageManager
from siesta.core.sparkManager import get_spark_session
from siesta.model.StorageModel import MetaData

import logging

logger = logging.getLogger(__name__)


def summarize_namespaces(storage: StorageManager) -> Dict[str, Any]:
    namespaces = storage.list_namespaces()
    namespace_logs = {
        namespace: storage.list_logs(namespace) for namespace in namespaces
    }

    return {
        "namespace_count": len(namespaces),
        "namespaces": [
            {
                "name": namespace,
                "log_count": len(logs),
                "logs": logs,
            }
            for namespace, logs in namespace_logs.items()
        ],
    }


def read_log_metadata(
    storage: StorageManager,
    metadata: MetaData,
    include_alphabet: bool,
    include_indexed_pairs: bool,
) -> Dict[str, Any]:
    metadata = storage.read_metadata_table(metadata)

    result = metadata.to_dict()

    if include_alphabet:
        result["activity_alphabet"] = sorted(
            row["activity"]
            for row in storage.read_activity_index(metadata).select("activity").distinct().collect()
        )
    if include_indexed_pairs:
        result["indexed_pairs"] = [
            {"source": row["source"], "target": row["target"]}
            for row in storage.read_pairs_index(metadata).select("source", "target").distinct().collect()
        ]

    return result


def run_adhoc_query(storage: StorageManager, metadata: MetaData, sql_text: str, row_limit: int) -> Dict[str, Any]:
    spark = get_spark_session()

    registered_tables = []
    for table_name in storage.list_tables(metadata):
        try:
            storage.read_table(metadata, table_name).createOrReplaceTempView(table_name)
            registered_tables.append(table_name)
        except Exception as e:
            logger.warning(f"Skipping table '{table_name}' - could not register as view: {e}")

    try:
        result_df = spark.sql(sql_text).limit(row_limit)
        rows = [json.loads(r) for r in result_df.toJSON().collect()]
    except Exception as e:
        logger.exception(f"Ad-hoc query failed: {e}")
        raise ValueError(f"Query execution failed: {e}")

    return {
        "columns": result_df.columns,
        "row_count": len(rows),
        "rows": rows,
        "available_tables": registered_tables,
    }
