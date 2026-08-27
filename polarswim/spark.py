"""Optional PySpark path for the per-length aggregation.

Deliberately not the default. On a database this size — order 10^4 lengths — pandas
finishes in milliseconds and Spark's startup alone costs seconds, so using it here
would be slower and would add a JVM to the install. It exists for the case the
pipeline is pointed at a fleet's worth of data rather than one swimmer's: the same
aggregation, expressed against a Spark DataFrame, so the analysis logic does not
have to be rewritten when the length table stops fitting on one machine.

    pip install -r requirements-spark.txt
    python -m polarswim analyze --engine spark

`available()` is the guard the rest of the package uses; nothing here is imported
unless the caller asks for it.
"""

from __future__ import annotations

import pandas as pd

REFERENCE_LENGTH_M = 22.86


def available() -> bool:
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        return False


def session(app_name: str = "polarswim"):
    """Local Spark session sized for a single machine."""
    from pyspark.sql import SparkSession
    return (SparkSession.builder
            .appName(app_name)
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")   # tiny data; 200 is waste
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def set_aggregates(lengths_pdf: pd.DataFrame):
    """Per-set medians and spread, computed through Spark.

    Takes and returns pandas so it is a drop-in for the pandas path; a real
    deployment would read the source table with `spark.read.jdbc` instead of
    handing over a local frame, which is the only line that would change.
    """
    if not available():
        raise RuntimeError(
            "PySpark is not installed. `pip install -r requirements-spark.txt` "
            "(needs a JVM). The default pandas path requires neither.")

    from pyspark.sql import functions as F

    spark = session()
    try:
        sdf = spark.createDataFrame(
            lengths_pdf[["workout_id", "set_id", "idx", "pace_s", "hr_cost"]])
        agg = (sdf.groupBy("workout_id", "set_id")
               .agg(F.count("idx").alias("n"),
                    F.expr("percentile_approx(pace_s, 0.5)").alias("median_pace_s"),
                    F.stddev("pace_s").alias("sd_pace_s"),
                    F.mean("hr_cost").alias("mean_hr_cost"))
               .orderBy("workout_id", "set_id"))
        return agg.toPandas()
    finally:
        spark.stop()
