"""PySpark pipeline: read -> clean -> features -> analytics -> MongoDB (+CSV copies in output/).
   python src\\pipeline.py [--input data\\transactions.csv]"""
import argparse, time
import pandas as pd
from common import DEFAULT_INPUT, OUT_DIR, check_input, get_spark, read_csv, clean, features, analyze, fail
from mongodb_utils import get_db, save_collections, save_metadata

ap = argparse.ArgumentParser(); ap.add_argument("--input", default=str(DEFAULT_INPUT)); a = ap.parse_args()
path = check_input(a.input)
db = get_db()                       # fail early if MongoDB is down
spark = get_spark()
t0 = time.time()
try:
    raw = read_csv(spark, path)
    raw_rows = raw.count()
    df = features(clean(raw)).persist()          # cached in memory/disk for the many aggregations below
    clean_rows = df.count()
    docs = analyze(df)
except Exception as e:
    if "OutOfMemory" in str(e):
        fail("Spark ran out of memory. Raise SPARK_DRIVER_MEMORY in .env (e.g. 6g/8g) or use fewer rows.")
    raise
secs = time.time() - t0
print(f"raw rows={raw_rows:,}  clean rows={clean_rows:,}  removed={raw_rows-clean_rows:,}  time={secs:.1f}s")
save_collections(db, docs)
save_metadata(db, input=str(path), raw_rows=raw_rows, clean_rows=clean_rows, removed_rows=raw_rows - clean_rows,
              seconds=round(secs, 2), spark_master=spark.sparkContext.master, spark_version=spark.version)
for name, items in docs.items():
    pd.DataFrame(items).to_csv(OUT_DIR / f"{name}.csv", index=False)
print("Saved to MongoDB collections:", ", ".join(docs), "+ run_metadata. CSV copies in output\\")
spark.stop()
