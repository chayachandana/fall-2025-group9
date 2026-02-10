# app.py
import os
import math
import time
from datetime import datetime
from typing import Tuple, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Path setup ---
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIVARIATE_DIR = os.path.join(BASE_DIR, "src", "component", "univariate")
sys.path.insert(0, UNIVARIATE_DIR)

from forecasting import forecast_future_dates, next_days
from utils import load_and_aggregate_district, safe_time_split  
# --- Config / paths ---
CSV_PATH = os.path.join(BASE_DIR, "src", "Data", "Output", "meals_combined.csv")
DATE_COL = "date"
TARGET_COL = "production_cost_total"
MODEL_DIR = os.path.join(UNIVARIATE_DIR, "LSTM_models")
MODEL_STUB = os.path.join(MODEL_DIR, "LSTM.pth")

st.set_page_config(page_title="School Production Cost Forecasting", layout="wide")

# --- Helpers ---
@st.cache_data
def load_series(csv_path=CSV_PATH):
    dates, values, _, _ = load_and_aggregate_district(
        CSV_PATH=csv_path, DATE_COL=DATE_COL, TARGET_COL=TARGET_COL, dayfirst="auto", debug=False)
    records = [(school, meal, pd.to_datetime(dt), float(v)) for (school, meal, dt), v in zip(dates, values.reshape(-1))]
    df = pd.DataFrame(records, columns=["school_name", "meal_type", DATE_COL, TARGET_COL])
    df = df.sort_values(["school_name", "meal_type", DATE_COL]).reset_index(drop=True)
    return df

def get_unique_school_meal(df: pd.DataFrame) -> List[Tuple[str, str]]:
    uniq = df[['school_name', 'meal_type']].drop_duplicates()
    return [(r.school_name, r.meal_type) for r in uniq.itertuples(index=False)]

def agg_total_by_date(df: pd.DataFrame) -> pd.DataFrame:
    # produce a daily district total (sum over all schools & meals)
    daily = df.groupby(DATE_COL, sort=True)[TARGET_COL].sum().reset_index()
    return daily

def compute_school_loss_summary(
    df: pd.DataFrame,
    window_recent: int = 10,
    window_baseline: int = 10,
    pct_threshold: float = 0.10,
) -> pd.DataFrame:
    records = []

    for (school, meal), g in df.groupby(["school_name", "meal_type"]):
        g = g.sort_values("date")
        if len(g) < window_recent + window_baseline:
            continue

        tail = g.tail(window_recent + window_baseline)
        base_vals = tail.iloc[:window_baseline][TARGET_COL].values
        recent_vals = tail.iloc[window_baseline:][TARGET_COL].values
        baseline_mean = float(base_vals.mean())
        recent_mean = float(recent_vals.mean())
        if baseline_mean == 0:
            continue

        diff = recent_mean - baseline_mean
        pct_change = diff / baseline_mean * 100.0
        if diff > baseline_mean * pct_threshold:
            reason = "overproduction"
        elif diff < -baseline_mean * pct_threshold:
            reason = "underproduction"
        else:
            reason = "stable"

        records.append({"school_name": school,"meal_type": meal,"loss_amount": diff,"loss_pct": pct_change,"loss_reason": reason,})

    if not records:
        return pd.DataFrame(columns=["school_name", "meal_type", "loss_amount", "loss_pct", "loss_reason"])
    loss_df = pd.DataFrame(records)
    loss_df = loss_df[loss_df["loss_reason"] != "stable"].copy()
    loss_df = loss_df.reindex(loss_df["loss_amount"].abs().sort_values(ascending=False).index)
    return loss_df

def business_days_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Return number of business days from (start_date, exclusive) to end_date (inclusive).
    If end_date <= start_date -> 0
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if end <= start:
        return 0
    rng = pd.bdate_range(start + pd.offsets.BDay(1), end)
    return len(rng)

@st.cache_data(show_spinner=False)
def compute_all_school_forecasts_for_horizon(df: pd.DataFrame, k_steps: int, model_type="LSTM"):
    combos = df[['school_name', 'meal_type']].drop_duplicates()
    all_outs = []
    total_failed = 0
    for _, row in combos.iterrows():
        school = row['school_name']
        meal = row['meal_type']
        try:
            out, bt_true, bt_pred = forecast_future_dates(csv_path=CSV_PATH,date_col=DATE_COL,target_col=TARGET_COL,
                            k_steps=k_steps,model_type=model_type,model_path=MODEL_STUB,
                            school_name=school,meal_type=meal,)
            if out is not None and not out.empty:
                all_outs.append(out)
        except Exception as e:
            total_failed += 1
            continue
    
    if len(all_outs) == 0:
        return pd.DataFrame(), total_failed
    df_all = pd.concat(all_outs, ignore_index=True)
    return df_all, total_failed

def aggregate_forecasts_for_date(df_forecasts: pd.DataFrame, target_date: pd.Timestamp) -> float:
    """Sum target_col across all forecasts where forecast_date == target_date"""
    mask = pd.to_datetime(df_forecasts['forecast_date']).dt.date == pd.to_datetime(target_date).date()
    if mask.sum() == 0:
        return float('nan')
    total = df_forecasts.loc[mask, TARGET_COL].sum()
    return float(total)

# --- Main Page Layout ---
st.title("School Production Cost — Forecasting Dashboard")
df_series = load_series()
st.sidebar.title("📊 Project Dashboard")

selected_view = st.sidebar.radio(
    "Select a View:",
    (
        "Single School Forecast",
        "Historical vs Forecast Plot",
        "Top Production Cost Schools",
        "School Loss Analysis",
    ),
)

st.sidebar.markdown("---")

left_col, right_col = st.columns([2, 1])

if "forecast_df" not in st.session_state:
    st.session_state["forecast_df"] = None
    st.session_state["forecast_school"] = None
    st.session_state["forecast_meal"] = None
    st.session_state["forecast_k"] = None

with left_col:
    # --- SECTION 1: run forecast and show table ---
    st.header("Forecast a single School + Meal")
    school_names = sorted(df_series["school_name"].unique())
    selected_school = st.selectbox("Choose School", school_names)
    meal_types = sorted(df_series[df_series["school_name"] == selected_school]["meal_type"].unique())
    selected_meal = st.selectbox("Choose Meal Type", meal_types)
    k_steps = st.number_input("Days to forecast (business days)",min_value=1,max_value=60,value=10,step=1)
    run_forecast_btn = st.button("Run Forecast for this School")

    if run_forecast_btn:
        with st.spinner(
            f"Forecasting {selected_school} / {selected_meal} for {k_steps} days..."
        ):
            try:
                out, _, _ = forecast_future_dates(csv_path=CSV_PATH,date_col=DATE_COL,target_col=TARGET_COL,
                                        k_steps=k_steps,model_type="LSTM",model_path=MODEL_STUB,
                                        school_name=selected_school,meal_type=selected_meal)
                
                st.session_state["forecast_df"] = out
                st.session_state["forecast_school"] = selected_school
                st.session_state["forecast_meal"] = selected_meal
                st.session_state["forecast_k"] = k_steps

            except Exception as e:
                st.error(f"Forecast failed: {e}")

    if st.session_state["forecast_df"] is not None:
        st.success(
            f"Forecast produced for {st.session_state['forecast_school']} / "
            f"{st.session_state['forecast_meal']} for "
            f"{st.session_state['forecast_k']} business day(s)."
        )
        st.subheader("Forecast table")
        df_show = st.session_state["forecast_df"].copy()
        if "forecast_date" in df_show.columns:
            df_show["forecast_date"] = pd.to_datetime(df_show["forecast_date"]).dt.date

        st.dataframe(df_show)
    st.markdown("---")

    # --- SECTION 2: plot for the same school + meal ---
    st.header("Historical + Forecast plot for selected School + Meal")

    if st.session_state["forecast_df"] is None:
        st.info("Run a forecast above to see the plot for that school and meal here.")
    else:
        school = st.session_state["forecast_school"]
        meal = st.session_state["forecast_meal"]
        fc_df = st.session_state["forecast_df"]

        # historical series for this school + meal
        df_hist = df_series[(df_series["school_name"] == school)& (df_series["meal_type"] == meal)].copy()
        df_hist = df_hist.sort_values(DATE_COL)

        # build combined dataframe
        hist_plot = df_hist[[DATE_COL, TARGET_COL]].copy()
        hist_plot["source"] = "Historical"
        hist_plot = hist_plot.rename(columns={DATE_COL: "date_plot"})
        fc_plot = fc_df[["forecast_date", TARGET_COL]].copy()
        fc_plot["source"] = "Forecast"
        fc_plot = fc_plot.rename(columns={"forecast_date": "date_plot"})
        plot_df = pd.concat([hist_plot, fc_plot], ignore_index=True)

        fig = px.line(plot_df,x="date_plot",y=TARGET_COL,color="source",title=f"{school} – {meal}: Historical and Forecast Production Cost",)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    
    # SECTION 3: Top Production Cost Schools
    st.header("Top Production Cost Schools")
    top_n = st.slider("Top N schools to show (by mean production cost)",min_value=3,max_value=30,value=10,)
    school_agg = (df_series.groupby("school_name")[TARGET_COL].mean().sort_values(ascending=False).reset_index())
    top_schools = school_agg.head(top_n)
    fig = px.bar(top_schools,x="school_name",y=TARGET_COL,title=f"Top {top_n} Schools by Average Production Cost",)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # Section 4: School Loss Analysis
    st.header("🏫 School Loss Analysis")
    loss_threshold_pct = st.slider(
        "Percent change threshold to flag over/under production (%)",min_value=1,
        max_value=50,value=10,
        help="Compares recent average cost to a baseline window for each school + meal.",)

    loss_df = compute_school_loss_summary(df_series,window_recent=10,window_baseline=10,pct_threshold=loss_threshold_pct / 100.0,)

    if loss_df.empty:
        st.success("No schools show significant over- or under-production based on the selected threshold ✅")
    else:

        cols = [
            c
            for c in ["school_name", "meal_type", "loss_amount", "loss_pct", "loss_reason"]
            if c in loss_df.columns
        ]

        st.subheader("Loss summary by school and meal")
        st.dataframe(loss_df[cols], use_container_width=True)
        fig_loss = px.bar(loss_df,x="school_name",y="loss_amount",
                color="loss_reason",title="Schools with Over/Under Production (recent vs baseline)",
                labels={"loss_amount": "Change in average daily cost"},)
        fig_loss.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_loss, use_container_width=True)

with right_col:
    # --- DATASET OVERVIEW ---
    st.subheader("Dataset overview")

    n_schools = df_series["school_name"].nunique()
    n_meal_types = df_series["meal_type"].nunique()
    n_rows = len(df_series)
    min_date = pd.to_datetime(df_series[DATE_COL].min())
    max_date = pd.to_datetime(df_series[DATE_COL].max())
    date_span_days = pd.to_datetime(df_series[DATE_COL]).dt.date.nunique()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Number of schools", n_schools)
        st.metric("Meal types", n_meal_types)
    with c2:
        st.metric("Total records", n_rows)
        st.metric("Total days", date_span_days)
        st.caption(f"From {min_date.date()} to {max_date.date()}")

    st.markdown("---")

    # --- HISTORICAL TREND (ALL SCHOOLS) ---
    st.subheader("Overall School historical production cost")

    daily_totals = (
        df_series.groupby(DATE_COL, sort=True)[TARGET_COL]
        .sum()
        .reset_index()
    )
    st.line_chart(
        daily_totals.set_index(DATE_COL)[TARGET_COL],
        height=200,
    )
    total_hist_cost = float(daily_totals[TARGET_COL].sum())
    st.metric(
        "Historical total production cost (all schools)",
        f"${total_hist_cost:,.0f}",
    )
    st.caption("Total daily production cost across all schools and meals.")
    st.markdown("---")
    # --- SELECTED SCHOOL + MEAL SNAPSHOT ---
    st.subheader("Selected school + meal snapshot")
    df_sel = df_series[(df_series["school_name"] == selected_school)& (df_series["meal_type"] == selected_meal)].copy()
    if df_sel.empty:
        st.info("No historical data for the selected school and meal.")
    else:
        avg_cost = df_sel[TARGET_COL].mean()
        min_cost = df_sel[TARGET_COL].min()
        max_cost = df_sel[TARGET_COL].max()
        latest_row = df_sel.sort_values(DATE_COL).iloc[-1]
        latest_date = pd.to_datetime(latest_row[DATE_COL])
        latest_cost = latest_row[TARGET_COL]

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Average daily cost", f"${avg_cost:,.0f}")
            st.metric("Min daily cost", f"${min_cost:,.0f}")
        with m2:
            st.metric("Max daily cost", f"${max_cost:,.0f}")
            st.metric(
                "Most recent cost",
                f"${latest_cost:,.0f}",
                help=f"Last date in data: {latest_date.date()}",
            )
        # last 10 days mini chart
        df_last = (df_sel.sort_values(DATE_COL).tail(10)[[DATE_COL, TARGET_COL]].copy())
        df_last = df_last.set_index(DATE_COL)
        st.line_chart(df_last[TARGET_COL], height=180)
        st.caption("Last 10 historical days for this school and meal.")
        st.markdown("---")
        # --- TOP 5 OVER / UNDER PRODUCTION SUMMARY ---
        st.subheader("Top 5 over / under production schools")
        loss_df_small = compute_school_loss_summary(df_series,window_recent=10,window_baseline=10,pct_threshold=0.10,)
        if loss_df_small.empty:
            st.info("No significant over- or under-production detected.")
        else:
            over = loss_df_small[loss_df_small["loss_amount"] > 0].head(5)
            under = loss_df_small[loss_df_small["loss_amount"] < 0].head(5)
            c3, c4 = st.columns(2)
            with c3:
                st.caption("Overproduction (higher recent cost)")
                if over.empty:
                    st.text("None")
                else:
                    st.dataframe(over[["school_name", "meal_type", "loss_amount"]],use_container_width=True,height=200,)
            with c4:
                st.caption("Underproduction (lower recent cost)")
                if under.empty:
                    st.text("None")
                else:
                    st.dataframe(under[["school_name", "meal_type", "loss_amount"]],use_container_width=True,height=200,)
    
    # Total Forecasted Production Cost (sum over horizon) 
    st.header("Total Forecasted Production Cost — All Schools")
    horizon_all = st.slider(
        "Horizon (business days) to forecast across all schools",
        min_value=1,
        max_value=60,
        value=10,
    )

    if st.button("Compute total forecast for ALL schools"):
        with st.spinner(
            f"Running LSTM forecasts for all schools and meals for next {horizon_all} business day(s)..."
        ):
            df_all_forecasts, failed = compute_all_school_forecasts_for_horizon(df_series,k_steps=horizon_all,model_type="LSTM",)

        if df_all_forecasts.empty:
            st.error(
                "No forecasts were produced. This usually means per-school model files "
                "are missing for many schools."
            )
        else:

            total_cost_all = df_all_forecasts[TARGET_COL].sum()
            st.markdown("### 📦 TOTAL Forecasted Production Cost ")
            st.markdown(f"## ${total_cost_all:,.2f}")
    st.markdown("---")
st.sidebar.markdown("---")
