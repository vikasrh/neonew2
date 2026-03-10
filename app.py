import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

DB_PATH = "neo.db"
PRED_TABLE = "neo_predictions"
RUNS_TABLE = "pipeline_runs"
AUTO_REFRESH_SECONDS = 60

st.set_page_config(page_title="NEO Risk Dashboard", layout="wide")
st.title("Near-Earth Object Risk Dashboard")
st.caption("Railway worker fetches NASA NEO data, retrains the models, and refreshes predictions every hour.")


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def load_predictions() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {PRED_TABLE}", conn)


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def load_latest_run() -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(
                f"SELECT * FROM {RUNS_TABLE} ORDER BY run_time_utc DESC LIMIT 1",
                conn,
            )
    except Exception:
        return pd.DataFrame()


df = load_predictions()
latest_run_df = load_latest_run()
latest_run = latest_run_df.iloc[0] if not latest_run_df.empty else None

status_cols = st.columns(4)
with status_cols[0]:
    if latest_run is not None and latest_run.get("run_time_utc"):
        run_time = datetime.fromisoformat(str(latest_run["run_time_utc"]))
        st.metric("Last Pipeline Run", run_time.strftime("%Y-%m-%d %H:%M UTC"))
    else:
        st.metric("Last Pipeline Run", "No runs yet")

with status_cols[1]:
    st.metric("Pipeline Status", str(latest_run["status"]).upper() if latest_run is not None else "UNKNOWN")

with status_cols[2]:
    st.metric("Fetched Rows", int(latest_run["fetched_rows"]) if latest_run is not None else 0)

with status_cols[3]:
    st.metric("Predicted Rows", int(latest_run["predicted_rows"]) if latest_run is not None else 0)

if latest_run is not None and str(latest_run.get("status", "")).lower() != "ok":
    st.error(f"Latest pipeline run failed: {latest_run.get('error', 'Unknown error')}")

if df.empty:
    st.warning(
        "No predictions found in 'neo_predictions' yet. The Railway worker will populate this table after the next hourly pipeline run."
    )
    st.stop()

# cleanup
df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
df = df.dropna(subset=["risk_score"])

# parse date if present
if "date" in df.columns:
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
else:
    df["date_parsed"] = pd.NaT

# Sidebar controls
st.sidebar.header("Filters")
min_score = float(df["risk_score"].min())
max_score = float(df["risk_score"].max())
score_range = st.sidebar.slider("Risk score range", min_score, max_score, (min_score, max_score))
show_only_high = st.sidebar.checkbox("Show only HIGH risk", value=False)

if latest_run is not None:
    st.sidebar.caption(f"Latest worker status: {str(latest_run['status']).upper()}")

# filtered dataframe
df_f = df[(df["risk_score"] >= score_range[0]) & (df["risk_score"] <= score_range[1])].copy()
if show_only_high and "risk_label" in df_f.columns:
    df_f = df_f[df_f["risk_label"] == "HIGH"]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictions", len(df_f))
c2.metric("High risk", int((df_f["risk_label"] == "HIGH").sum()) if "risk_label" in df_f.columns else 0)
c3.metric("Avg score", f"{float(df_f['risk_score'].mean()) if len(df_f) else 0.0:.3f}")
c4.metric("Max score", f"{float(df_f['risk_score'].max()) if len(df_f) else 0.0:.3f}")

# Charts
st.subheader("Risk score distribution")
fig_hist = px.histogram(df_f, x="risk_score", nbins=30, color="risk_label" if "risk_label" in df_f.columns else None)
st.plotly_chart(fig_hist, use_container_width=True)

# Risk over time (date)
if df_f["date_parsed"].notna().any():
    tmp = df_f.dropna(subset=["date_parsed"]).sort_values("date_parsed")
    st.subheader("Risk score over date")
    fig_line = px.line(
        tmp,
        x="date_parsed",
        y="risk_score",
        hover_data=["name", "neo_reference_id", "risk_label"] if "risk_label" in tmp.columns else ["name", "neo_reference_id"],
    )
    st.plotly_chart(fig_line, use_container_width=True)

# Scatter: miss distance vs diameter
if "miss_distance_km" in df_f.columns and "diameter_m" in df_f.columns:
    st.subheader("Miss distance vs Diameter (colored by risk)")
    fig_scatter = px.scatter(
        df_f,
        x="miss_distance_km",
        y="diameter_m",
        color="risk_score",
        hover_data=["name", "date", "risk_label", "neo_reference_id"] if "risk_label" in df_f.columns else ["name", "date", "neo_reference_id"],
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Table
st.subheader("Top risky objects")
cols = [
    c
    for c in [
        "date",
        "name",
        "neo_reference_id",
        "orbiting_body",
        "diameter_m",
        "miss_distance_km",
        "velocity_kmh",
        "xgb_risk_prob",
        "isolation_anomaly_score",
        "is_anomaly",
        "risk_score",
        "risk_label",
        "hazardous",
        "prediction_time_utc",
    ]
    if c in df_f.columns
]
st.dataframe(df_f.sort_values("risk_score", ascending=False)[cols].head(100), use_container_width=True)

st.caption("The report refreshes from the latest neo_predictions rows. Use Railway logs to monitor the hourly worker cycle.")

if st.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()