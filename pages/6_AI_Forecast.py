# pages/6_AI_Forecast.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from forecasting import determine_effective_projection_rates, calculate_ai_forecast_core
from calculations import calculate_avg_lag_generic
from constants import *
from helpers import load_css

# --- Theme Initialization ---
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# --- Page Configuration ---
st.set_page_config(page_title="AI Forecast", page_icon="🤖", layout="wide")

# --- Apply CSS ---
if st.session_state.theme == "light":
    load_css("style-light.css")
else:
    load_css("style-dark.css")

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")
    
    st.write("") # Spacer

    selected_theme = st.radio(
        "Theme",
        ["Dark", "Light"],
        captions=["", ""],
        index=0 if st.session_state.theme == "dark" else 1,
        key="theme_selector",
        horizontal=True,
    )

    new_theme_value = "light" if selected_theme == "Light" else "dark"
    if new_theme_value != st.session_state.theme:
        st.session_state.theme = new_theme_value
        st.rerun()

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data from Session State ---
processed_data = st.session_state.referral_data_processed
ordered_stages = st.session_state.ordered_stages
ts_col_map = st.session_state.ts_col_map
inter_stage_lags = st.session_state.inter_stage_lags
site_metrics = st.session_state.site_metrics_calculated
weights = st.session_state.weights_normalized
icf_variation = st.session_state.shared_icf_variation
cpql_inflation = st.session_state.ai_cpql_inflation
ql_vol_threshold = st.session_state.ai_ql_vol_threshold
ql_capacity_multiplier = st.session_state.ai_ql_capacity_multiplier
proj_horizon = st.session_state.proj_horizon

# --- Page-Specific Controls ---
with st.container(border=True):
    st.subheader("Define Your Goals & Assumptions")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        goal_lpi_date = st.date_input("Target LPI Date", value=datetime.now() + pd.DateOffset(months=12))
    with c2:
        goal_icf_num = st.number_input("Target Total ICFs", min_value=1, value=100, step=10)
    with c3:
        base_cpql = st.number_input("Base Estimated CPQL (POF)", min_value=1.0, value=75.0, step=5.0, format="%.2f")

    st.write("") # Spacer
    
    lag_method = st.radio(
        "ICF Landing Lag Assumption:",
        ("Use Overall Average POF->ICF Lag", "Use P25/P50/P75 Day Lag Distribution"),
        horizontal=True, key="ai_lag_method"
    )
    p25_lag, p50_lag, p75_lag = None, None, None
    if lag_method == "Use P25/P50/P75 Day Lag Distribution":
        l1, l2, l3 = st.columns(3)
        p25_lag = l1.number_input("P25 Lag (Days)", min_value=0, value=20, step=1)
        p50_lag = l2.number_input("P50 (Median) Lag (Days)", min_value=0, value=30, step=1)
        p75_lag = l3.number_input("P75 Lag (Days)", min_value=0, value=45, step=1)
        if not (p25_lag <= p50_lag <= p75_lag):
            st.warning("For logical distribution, ensure P25 <= P50 <= P75.")

    rate_method = st.radio(
        "Base Conversion Rates On:",
        options=('Manual Input Below', 'Rolling Historical Average'),
        index=1, key="ai_rate_method", horizontal=True
    )
    rolling_window = 0
    if rate_method == 'Rolling Historical Average':
        rolling_window = st.selectbox(
            "Select Rolling Window:", options=[1, 3, 6, 999], index=2,
            format_func=lambda x: "Overall Average" if x == 999 else f"{x}-Month",
            key='ai_rolling_window'
        )

    cr1, cr2 = st.columns(2)
    manual_rates = {
        f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}": cr1.slider("AI: POF -> PreScreen %", 0.0, 100.0, 90.0, format="%.1f%%", key='ai_cr_qps') / 100.0,
        f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}": cr1.slider("AI: PreScreen -> StS %", 0.0, 100.0, 25.0, format="%.1f%%", key='ai_cr_pssts') / 100.0,
        f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}": cr2.slider("AI: StS -> Appt %", 0.0, 100.0, 50.0, format="%.1f%%", key='ai_cr_sa') / 100.0,
        f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}": cr2.slider("AI: Appt -> ICF %", 0.0, 100.0, 60.0, format="%.1f%%", key='ai_cr_ai') / 100.0
    }

with st.expander("Optional Site Configurations"):
    # Site Activity Dates & Caps go here
    pass # Placeholder for your existing site config code

if st.button("🚀 Generate Auto Forecast", type="primary", use_container_width=True):
    # The main calculation logic remains the same
    pass # Placeholder for your existing forecast calculation and display logic