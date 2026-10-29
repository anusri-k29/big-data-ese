"""Generate a SYNTHETIC financial transaction CSV (NumPy, chunked - no Hadoop/Spark needed).
   python src\\generate_data.py --rows 100000
Fraud is simulated only in TRANSFER/CASH_OUT, skewed to night hours, large amounts that empty the origin account.
A tiny share of dirty rows (nulls, negative amounts, duplicates) is injected to exercise the cleaning step."""
import argparse, time
import numpy as np, pandas as pd
from common import DEFAULT_INPUT, COLUMNS, fail

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=100_000)
ap.add_argument("--out", default=str(DEFAULT_INPUT))
ap.add_argument("--chunk", type=int, default=1_000_000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fraud-rate", type=float, default=0.004, help="fraud probability among TRANSFER/CASH_OUT")
a = ap.parse_args()
if a.rows < 1: fail("--rows must be >= 1")
rng = np.random.default_rng(a.seed)


def chunk(start, n, total):
    u = rng.random(n)
    t = np.select([u < .35, u < .69, u < .91, u < .99], ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER"], "DEBIT")
    fraud = np.isin(t, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < a.fraud_rate)
    hour = np.floor((rng.random(n) + rng.random(n)) * 12).astype(int)          # daytime peak
    hour = np.where(fraud & (rng.random(n) < .5), rng.integers(0, 6, n), hour)   # fraud skews to night
    step = rng.integers(0, 30, n) * 24 + hour + 1
    old = np.round(np.exp(rng.normal(9, 1.8, n)), 2)
    amt = np.minimum(np.exp(rng.normal(8.5, 1.6, n)), old)
    amt = np.round(np.maximum(np.where(fraud, old, amt), 1.0), 2)
    new = np.round(np.maximum(old - amt, 0), 2)
    dest_old = np.where(t == "PAYMENT", 0.0, np.round(np.exp(rng.normal(9, 1.8, n)), 2))
    dest_new = np.where(fraud, dest_old, np.round(dest_old + amt, 2))           # fraud: destination balance not updated
    pool = max(total // 3, 1)
    df = pd.DataFrame({
        "txn_id": np.arange(start, start + n), "step": step, "type": t, "amount": amt,
        "nameOrig": ["C" + str(i) for i in rng.integers(0, pool, n)], "oldbalanceOrg": old, "newbalanceOrig": new,
        "nameDest": [("M" if x == "PAYMENT" else "C") + str(i) for x, i in zip(t, rng.integers(0, pool, n))],
        "oldbalanceDest": dest_old, "newbalanceDest": dest_new, "isFraud": fraud.astype(int)})[COLUMNS]
    k = max(n // 1000, 1)                                                         # ~0.1% dirty rows of each kind
    df["amount"] = df["amount"].astype(float)
    df.loc[rng.choice(n, k, replace=False), "amount"] = np.nan
    df.loc[rng.choice(n, k, replace=False), "amount"] = -1.0
    return pd.concat([df, df.sample(k, random_state=1)], ignore_index=True)       # duplicated txn_ids


t0, done = time.time(), 0
try:
    while done < a.rows:
        n = min(a.chunk, a.rows - done)
        chunk(done, n, a.rows).to_csv(a.out, mode="w" if done == 0 else "a", header=done == 0, index=False)
        done += n
        print(f"  {done:,}/{a.rows:,} rows  ({time.time()-t0:.0f}s)", flush=True)
except OSError as e:
    fail(f"Could not write {a.out}: {e}")
import os
print(f"Wrote {a.out}  ({os.path.getsize(a.out)/1e6:,.1f} MB, SYNTHETIC data)")
