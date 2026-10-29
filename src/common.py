"""Shared config, schema, Spark session, cleaning, feature engineering, analytics."""
import os, sys, shutil
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR, OUT_DIR = ROOT / "data", ROOT / "output"
DEFAULT_INPUT = DATA_DIR / "transactions.csv"
BASE_DATE = "2025-01-01"          # step 1 = 2025-01-01 00:00 (synthetic calendar)

COLUMNS = ["txn_id", "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
           "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]


def fail(msg):
    print(f"\nERROR: {msg}\n", file=sys.stderr); sys.exit(1)


def check_input(path):
    path = Path(path)
    if not path.is_file():
        fail(f"Dataset not found: {path}\nGenerate it first, e.g.:  python src\\generate_data.py --rows 100000")
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if header != COLUMNS:
        fail(f"Invalid schema in {path}.\nExpected columns: {COLUMNS}\nFound: {header}")
    return path


def get_spark(master=None, app="FinancialTransactionAnalytics"):
    if not shutil.which("java"):
        fail("Java not found on PATH. Install JDK 17 (e.g. Eclipse Temurin) and reopen the terminal. Check with: java -version")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession
    tmp = OUT_DIR / "spark_tmp"; tmp.mkdir(parents=True, exist_ok=True)
    try:
        spark = (SparkSession.builder.appName(app)
                 .master(master or os.getenv("SPARK_MASTER", "local[*]"))
                 .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
                 .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "16"))
                 .config("spark.local.dir", str(tmp)).config("spark.ui.enabled", "false").config("spark.ui.showConsoleProgress", "false")
                 .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    except Exception as e:
        fail(f"Could not start Spark (is Java 8/11/17 installed and JAVA_HOME valid?): {e}")
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_csv(spark, path):
    from pyspark.sql import types as T
    schema = T.StructType([
        T.StructField("txn_id", T.LongType()), T.StructField("step", T.IntegerType()),
        T.StructField("type", T.StringType()), T.StructField("amount", T.DoubleType()),
        T.StructField("nameOrig", T.StringType()), T.StructField("oldbalanceOrg", T.DoubleType()),
        T.StructField("newbalanceOrig", T.DoubleType()), T.StructField("nameDest", T.StringType()),
        T.StructField("oldbalanceDest", T.DoubleType()), T.StructField("newbalanceDest", T.DoubleType()),
        T.StructField("isFraud", T.IntegerType())])
    # PERMISSIVE: malformed values become null and are removed in clean()
    return spark.read.csv(Path(path).resolve().as_uri(), header=True, schema=schema, mode="PERMISSIVE")


def clean(df):
    """Remove nulls, invalid values and duplicates."""
    from pyspark.sql import functions as F
    df = df.withColumn("type", F.upper(F.trim("type")))
    df = df.dropna(subset=["txn_id", "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud"])
    df = df.filter((F.col("amount") > 0) & (F.col("step") >= 1) & F.col("isFraud").isin(0, 1))
    return df.dropDuplicates(["txn_id"])


def features(df):
    from pyspark.sql import functions as F
    c = F.col
    return (df.withColumn("day", F.floor((c("step") - 1) / 24).cast("int") + 1)
            .withColumn("hour", ((c("step") - 1) % 24).cast("int"))
            .withColumn("date", F.date_add(F.lit(BASE_DATE).cast("date"), c("day") - 1))
            .withColumn("is_night", (c("hour") < 6).cast("int"))
            .withColumn("is_high_value", (c("amount") >= 100000).cast("int"))
            .withColumn("orig_emptied", ((c("newbalanceOrig") == 0) & (c("oldbalanceOrg") > 0)).cast("int"))
            .withColumn("err_orig", c("newbalanceOrig") + c("amount") - c("oldbalanceOrg"))
            .withColumn("err_dest", c("oldbalanceDest") + c("amount") - c("newbalanceDest"))
            .withColumn("amount_bucket", F.when(c("amount") < 1e3, "1) < 1K").when(c("amount") < 1e4, "2) 1K-10K")
                        .when(c("amount") < 1e5, "3) 10K-100K").when(c("amount") < 1e6, "4) 100K-1M")
                        .otherwise("5) >= 1M")))


def _agg(df, *keys):
    from pyspark.sql import functions as F
    return df.groupBy(*keys).agg(
        F.count("*").alias("txns"), F.sum("amount").alias("total_amount"),
        F.avg("amount").alias("avg_amount"), F.max("amount").alias("max_amount"),
        F.sum("isFraud").alias("fraud_txns"),
        F.sum(F.when(F.col("isFraud") == 1, F.col("amount")).otherwise(0.0)).alias("fraud_amount"))


def analyze(df):
    """Distributed aggregations on the feature DataFrame. Returns {collection: [documents]}.
    Only small aggregated tables are collected to the driver."""
    from pyspark.sql import functions as F
    rows = lambda d: [r.asDict() for r in d.collect()]
    fraud_profile = df.groupBy("isFraud").agg(
        F.count("*").alias("txns"), F.avg("amount").alias("avg_amount"),
        F.expr("percentile_approx(amount, 0.5)").alias("median_amount"),
        F.avg("orig_emptied").alias("share_origin_emptied"), F.avg("is_night").alias("share_night"),
        F.avg("is_high_value").alias("share_high_value"),
        F.avg(F.abs("err_dest")).alias("avg_abs_dest_balance_error"))
    emptied = _agg(df, "type", "orig_emptied")
    docs = {
        "transaction_summary": rows(_agg(df, "type")),
        "time_analysis": rows(_agg(df, "day", "hour", "type")),
        "amount_analysis": rows(_agg(df, "amount_bucket", "type")),
        "fraud_analysis": ([dict(kind="profile", **r) for r in rows(fraud_profile)] +
                           [dict(kind="type_origin_emptied", **r) for r in rows(emptied)]),
    }
    return docs
