from typing import Any, Dict
from pyspark.sql import SparkSession
import os
import sys
import subprocess
import zipfile
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


# Ensure PySpark uses its bundled Spark
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

spark_session = None

def get_docker_bridge_ip():
    """Get Docker bridge gateway IP dynamically."""
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show", "docker0"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'inet ' in line:
                    ip = line.strip().split()[1].split('/')[0]
                    return ip
    except:
        pass
    
    # Fallback to default
    return "172.17.0.1"

def startup(config: Dict[str, Any] = {}) -> None:
    """Initialize Spark session with configuration.
    
    Args:
        config: Configuration dictionary containing Spark settings.
                Falls back to environment variables or default if config is not provided.
    """

    spark_master_url = config.get("spark_master", os.getenv("SPARK_MASTER", "spark://spark-master:7077"))
    app_name = config.get("spark_app_name", "SiestaFramework")
    driver_memory = config.get("spark_driver_memory", os.getenv("SPARK_DRIVER_MEMORY", "8g"))
    executor_memory = config.get("spark_executor_memory", os.getenv("SPARK_EXECUTOR_MEMORY", "8g"))
    executor_cores = config.get("spark_executor_cores", os.getenv("SPARK_EXECUTOR_CORES"))
    cores_max = config.get("spark_cores_max", os.getenv("SPARK_CORES_MAX"))
    executor_memory_overhead = config.get("spark_executor_memory_overhead", os.getenv("SPARK_EXECUTOR_MEMORY_OVERHEAD"))
    shuffle_partitions = config.get("spark_shuffle_partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS"))
    driver_host = config.get("spark_driver_host", os.getenv("SPARK_DRIVER_HOST"))
    driver_port = os.getenv("SPARK_DRIVER_PORT")
    block_manager_port = os.getenv("SPARK_BLOCKMANAGER_PORT")
    # Local mode runs Python workers on this machine; cluster executors use the Spark image's python3.12
    default_python = sys.executable if spark_master_url.startswith("local") else "/usr/bin/python3.12"
    pyspark_python = os.getenv("PYSPARK_PYTHON", default_python)
    
    global spark_session 
    try:
        builder = SparkSession.builder \
            .appName(app_name) \
            .master(spark_master_url)
        
        # Resolve OS username for Hadoop UGI (container may have no /etc/passwd entry)
        hadoop_user = os.getenv("HADOOP_USER_NAME", os.getenv("SPARK_USER", "spark"))

        # Required JVM options for Java 17+ compatibility
        java_options = (
            f"-Duser.name={hadoop_user} "
            "--add-opens=java.base/java.lang=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
            "--add-opens=java.base/java.io=ALL-UNNAMED "
            "--add-opens=java.base/java.net=ALL-UNNAMED "
            "--add-opens=java.base/java.nio=ALL-UNNAMED "
            "--add-opens=java.base/java.util=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
            "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
            "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
            "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
        )
        
        # Hadoop AWS packages for S3 support and Delta Lake and Kafka
        packages = "org.apache.hadoop:hadoop-aws:3.4.1,com.amazonaws:aws-java-sdk-bundle:1.12.540,io.delta:delta-spark_2.13:4.0.0,io.delta:delta-storage:4.0.0,org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0"
        
        builder = builder \
            .config("spark.driver.extraJavaOptions", java_options) \
            .config("spark.executor.extraJavaOptions", java_options) \
            .config("spark.driver.memory", driver_memory) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.pyspark.python", pyspark_python) \
            .config("spark.pyspark.driver.python", os.getenv("PYSPARK_DRIVER_PYTHON", pyspark_python)) \
            .config("spark.executorEnv.PYSPARK_PYTHON", pyspark_python) \
            .config("spark.databricks.delta.optimizeWrite.enabled", "true") \
            .config("spark.databricks.delta.autoCompact.enabled", "true") \
            .config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.S3SingleDriverLogStore") \
            .config("spark.hadoop.fs.s3a.fast.upload", "true") \
            .config("spark.hadoop.fs.s3a.multipart.size", "104857600") \
            .config("spark.jars.packages", packages) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", str(cores_max or 6)) \
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")

        # Optional cluster sizing (used for scalability experiments; unset in local mode)
        if executor_cores:
            builder = builder.config("spark.executor.cores", str(executor_cores))
        if cores_max:
            builder = builder.config("spark.cores.max", str(cores_max))
        if executor_memory_overhead:
            builder = builder.config("spark.executor.memoryOverhead", executor_memory_overhead)
        if shuffle_partitions:
            builder = builder \
                .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
                .config("spark.default.parallelism", str(shuffle_partitions))

        # Driver networking: executors on other swarm nodes must reach the driver
        if driver_host:
            builder = builder \
                .config("spark.driver.host", driver_host) \
                .config("spark.driver.bindAddress", "0.0.0.0")
        if driver_port:
            builder = builder.config("spark.driver.port", driver_port)
        if block_manager_port:
            builder = builder.config("spark.blockManager.port", block_manager_port)

        if os.getenv("SPARK_IVY_DIR"):
            builder = builder.config("spark.jars.ivy", os.getenv("SPARK_IVY_DIR"))

        # Configure S3 credentials if provided
        if config and config.get("s3_access_key") and config.get("s3_secret_key"):
            builder = builder \
                .config("spark.hadoop.fs.s3a.access.key", config["s3_access_key"]) \
                .config("spark.hadoop.fs.s3a.secret.key", config["s3_secret_key"])
            logger.info(f"S3 credentials configured from config file.")
        
        # Configure S3 endpoint if provided
        if config and config.get("s3_endpoint"):
            s3_endpoint = config["s3_endpoint"]
            # Translate localhost to Docker bridge IP
            bridge_ip = get_docker_bridge_ip()
            executor_s3_endpoint = s3_endpoint.replace("localhost", bridge_ip).replace("127.0.0.1", bridge_ip)
            builder = builder.config("spark.hadoop.fs.s3a.endpoint", executor_s3_endpoint)
            logger.info(f"S3 endpoint configured: {executor_s3_endpoint} (Docker bridge: {bridge_ip})")
        
        spark_session = builder.getOrCreate()
        spark_session.sparkContext.setLogLevel("ERROR")
        logger.info(f"SparkSession initialized and connected to Spark Master at {spark_master_url}.")
        conf = spark_session.sparkContext.getConf()
        logger.info(
            "Spark resources: cores.max=%s, executor.cores=%s, executor.memory=%s, executor.memoryOverhead=%s, shuffle.partitions=%s",
            conf.get("spark.cores.max", "unset"),
            conf.get("spark.executor.cores", "unset"),
            conf.get("spark.executor.memory", "unset"),
            conf.get("spark.executor.memoryOverhead", "unset"),
            conf.get("spark.sql.shuffle.partitions", "unset"),
        )
        
        # Ship code to executors
        try:
            current_file = Path(__file__).resolve()
            package_dir = current_file.parent.parent # siesta_framework directory
            project_root = package_dir.parent # Project root containing siesta_framework
            
            zip_path = "/tmp/siesta_framework.zip"
            
            # Create zip with only .py files to reduce size
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                siesta_dir = project_root / "siesta"
                for py_file in siesta_dir.rglob("*.py"):
                    arcname = py_file.relative_to(project_root)
                    zipf.write(py_file, arcname)
            
            spark_session.sparkContext.addPyFile(zip_path)
            logger.info(f"Shipped code to executors: {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to ship code to executors: {e}")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize SparkSession: {e}")

def get_spark_session() -> SparkSession:
    """Get the active Spark session.
    Returns:
        Active SparkSession instance.
    """
    if spark_session is None:
        raise RuntimeError("SparkSession not initialized. Call startup() first.")
    return spark_session


def cleanup():
    """Release cached data and Delta metadata without stopping the session.
    
    Call this after completing a batch of work (e.g., preprocessing or mining)
    to free memory while keeping the session alive for subsequent requests.
    """
    global spark_session
    if spark_session is None:
        return

    # Clear all cached RDDs/DataFrames
    try:
        sc = spark_session.sparkContext
        for rdd_id in list(sc._jsc.getPersistentRDDs().keys()):
            sc._jsc.getPersistentRDDs().get(rdd_id).unpersist(True)
    except Exception:
        pass

    # Clear Delta Lake's transaction log cache
    try:
        spark_session._jvm.org.apache.spark.sql.delta.DeltaLog.clearCache()
    except Exception:
        pass


def shutdown():
    """Stop the Spark session and release all resources."""
    global spark_session
    if spark_session:
        cleanup()
        logger.info("Shutting down SparkSession...")
        spark_session.stop()
        spark_session = None
        logger.info("SparkSession stopped successfully.")