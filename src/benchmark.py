"""Measure real Spark processing time on different dataset sizes (and optionally local thread counts).
Each size is the first N rows of the main CSV (copied to data\\bench\\), so file sizes are real.
   python src\\benchmark.py --sizes 100000,500000,1000000 [--repeats 1] [--cores 1,2,4]"""
import argparse, os, time, csv
from datetime import datetime, timezone
from common import DEFAULT_INPUT, DATA_DIR, OUT_DIR, check_input, get_spark, read_csv, clean, features, analyze, fail
from mongodb_utils import get_db

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=str(DEFAULT_INPUT))
ap.add_argument("--sizes", default="100000,500000,1000000,5000000,10000000")
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--cores", default="", help="optional, e.g. 1,2,4: local threads on THIS machine at the largest size")
a = ap.parse_args()
src = check_input(a.input); db = get_db()
with open(src, "rb") as f: total = sum(1 for _ in f) - 1
sizes = [s for s in sorted({int(x) for x in a.sizes.split(",")}) if s <= total]
skipped = [s for s in a.sizes.split(",") if int(s) > total]
if skipped: print(f"Skipping sizes larger than dataset ({total:,} lines): {skipped}")
if not sizes: fail(f"No requested size fits the dataset ({total:,} rows). Generate more rows.")
bench = DATA_DIR / "bench"; bench.mkdir(exist_ok=True)


def subset(n):
    p = bench / f"tx_{n}.csv"
    if not p.exists():
        with open(src, "rb") as fi, open(p, "wb") as fo:
            for i, line in enumerate(fi):
                if i > n: break
                fo.write(line)
    return p


def timed(spark, path):
    t0 = time.perf_counter()
    df = features(clean(read_csv(spark, path))).persist()
    n = df.count(); analyze(df)
    secs = time.perf_counter() - t0; df.unpersist()
    return n, secs


results = []
def record(kind, master, n_req, path, n_clean, secs_list):
    secs = sorted(secs_list)[len(secs_list) // 2]
    r = dict(kind=kind, master=master, requested_rows=n_req, clean_rows=n_clean,
             file_mb=round(os.path.getsize(path) / 1e6, 1), seconds=round(secs, 2),
             records_per_sec=round(n_clean / secs), repeats=len(secs_list), logical_cpus=os.cpu_count(),
             ts=datetime.now(timezone.utc).isoformat())
    print(r); results.append(r)

try:
    spark = get_spark()
    print("Warm-up run (not recorded) ..."); timed(spark, subset(sizes[0]))
    for n in sizes:
        p = subset(n); runs = [timed(spark, p) for _ in range(a.repeats)]
        record("data_size", spark.sparkContext.master, n, p, runs[0][0], [r[1] for r in runs])
    spark.stop()
    if a.cores:
        p = subset(sizes[-1])
        for c in a.cores.split(","):
            sp = get_spark(f"local[{c.strip()}]"); timed(sp, subset(sizes[0]))
            runs = [timed(sp, p) for _ in range(a.repeats)]
            record("local_threads", f"local[{c.strip()}]", sizes[-1], p, runs[0][0], [r[1] for r in runs]); sp.stop()
except Exception as e:
    if "OutOfMemory" in str(e): fail("Out of memory during benchmark. Raise SPARK_DRIVER_MEMORY in .env or use smaller sizes.")
    raise
db["benchmark_results"].delete_many({}); db["benchmark_results"].insert_many([dict(r) for r in results])
with open(OUT_DIR / "benchmark_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
print("Saved to MongoDB (benchmark_results) and output\\benchmark_results.csv")
