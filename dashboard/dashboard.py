import sys
from pathlib import Path
import pandas as pd, plotly.express as px, streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import os
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Financial Transaction Analytics", layout="wide")
st.title("Financial Transaction Analytics using Big Data")
st.caption("SYNTHETIC data - processed with PySpark (local mode), results served from MongoDB")


@st.cache_resource
def db():
    c = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=4000)
    c.admin.command("ping"); return c[os.getenv("MONGO_DATABASE", "financial_analytics")]


def load(name):
    return pd.DataFrame(list(db()[name].find({}, {"_id": 0})))


try:
    ts, tm, am, fa, bm = (load(n) for n in ["transaction_summary", "time_analysis", "amount_analysis", "fraud_analysis", "benchmark_results"])
    meta = load("run_metadata")
except Exception as e:
    st.error(f"Cannot reach MongoDB. Start it and check MONGO_URI in .env.\n\n{e}"); st.stop()
if ts.empty:
    st.warning("No analytics found. Run:  python src\\pipeline.py"); st.stop()

tab1, tab2 = st.tabs(["Transaction & fraud analytics", "Performance / scalability"])
with tab1:
    types = sorted(ts["type"].unique())
    sel = st.sidebar.multiselect("Transaction type", types, default=types)
    d0, d1 = int(tm["day"].min()), int(tm["day"].max())
    days = st.sidebar.slider("Day range (step/24)", d0, d1, (d0, d1))
    ts_f = tm[tm["type"].isin(sel) & tm["day"].between(*days)]
    am_f = am[am["type"].isin(sel)]
    tot = ts_f[["txns", "total_amount", "fraud_txns", "fraud_amount"]].sum()
    if tot["txns"] == 0: st.warning("No data for the selected filters."); st.stop()
    c = st.columns(5)
    c[0].metric("Total transactions", f"{int(tot.txns):,}")
    c[1].metric("Total amount", f"{tot.total_amount:,.0f}")
    c[2].metric("Average amount", f"{tot.total_amount / tot.txns:,.2f}")
    c[3].metric("Fraudulent transactions", f"{int(tot.fraud_txns):,}")
    c[4].metric("Fraud rate", f"{tot.fraud_txns / tot.txns:.3%}")

    bt = ts_f.groupby("type", as_index=False)[["txns", "total_amount", "fraud_txns"]].sum()
    bt["fraud_rate_%"] = bt.fraud_txns / bt.txns * 100
    a, b = st.columns(2)
    a.plotly_chart(px.bar(bt, x="type", y="txns", title="Transactions by type"), use_container_width=True)
    b.plotly_chart(px.bar(bt, x="type", y="total_amount", title="Total amount by type"), use_container_width=True)
    a, b = st.columns(2)
    daily = ts_f.groupby("day", as_index=False)[["txns", "fraud_txns"]].sum()
    a.plotly_chart(px.line(daily, x="day", y="txns", title="Transactions per day"), use_container_width=True)
    hourly = ts_f.groupby("hour", as_index=False)[["txns", "fraud_txns"]].sum()
    hourly["fraud_rate_%"] = hourly.fraud_txns / hourly.txns * 100
    b.plotly_chart(px.bar(hourly, x="hour", y="fraud_rate_%", title="Fraud rate by hour of day"), use_container_width=True)
    st.subheader("Fraud analysis")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(bt, x="type", y="fraud_txns", title="Fraud count by type"), use_container_width=True)
    pie = pd.DataFrame({"class": ["Fraudulent", "Legitimate"], "txns": [tot.fraud_txns, tot.txns - tot.fraud_txns]})
    b.plotly_chart(px.pie(pie, names="class", values="txns", title="Fraud vs legitimate"), use_container_width=True)
    prof = fa[fa.kind == "profile"].copy()
    if not prof.empty:
        prof["class"] = prof.isFraud.map({0: "Legitimate", 1: "Fraudulent"})
        a, b = st.columns(2)
        a.plotly_chart(px.bar(prof, x="class", y=["avg_amount", "median_amount"], barmode="group",
                              title="Amount: fraudulent vs legitimate"), use_container_width=True)
        b.plotly_chart(px.bar(prof, x="class", y=["share_origin_emptied", "share_night", "share_high_value"],
                              barmode="group", title="Behavioural flags (share of class)"), use_container_width=True)
        st.dataframe(prof.drop(columns=["kind", "isFraud"]), use_container_width=True)
    ab = am_f.groupby("amount_bucket", as_index=False)[["txns", "fraud_txns"]].sum().sort_values("amount_bucket")
    ab["fraud_rate_%"] = ab.fraud_txns / ab.txns * 100
    a, b = st.columns(2)
    a.plotly_chart(px.bar(ab, x="amount_bucket", y="txns", title="Transactions by amount bucket"), use_container_width=True)
    b.plotly_chart(px.bar(ab, x="amount_bucket", y="fraud_rate_%", title="Fraud rate by amount bucket"), use_container_width=True)
    if not meta.empty:
        m = meta.iloc[0]
        st.caption(f"Last run: {int(m.raw_rows):,} raw rows -> {int(m.clean_rows):,} clean ({int(m.removed_rows):,} removed) "
                   f"in {m.seconds}s on {m.spark_master}, Spark {m.spark_version}")
with tab2:
    if bm.empty:
        st.info("No benchmark yet. Run:  python src\\benchmark.py")
    else:
        st.caption("Measured on a single machine in Spark local mode (threads, not a cluster).")
        ds = bm[bm.kind == "data_size"].sort_values("clean_rows")
        if not ds.empty:
            a, b = st.columns(2)
            a.plotly_chart(px.line(ds, x="clean_rows", y="seconds", markers=True, title="Processing time vs records"), use_container_width=True)
            b.plotly_chart(px.line(ds, x="clean_rows", y="records_per_sec", markers=True, title="Throughput (records/sec)"), use_container_width=True)
        th = bm[bm.kind == "local_threads"]
        if not th.empty:
            st.plotly_chart(px.bar(th, x="master", y="seconds", title=f"Local threads at {int(th.clean_rows.iloc[0]):,} rows"), use_container_width=True)
        st.dataframe(bm.drop(columns=["ts"]), use_container_width=True)
