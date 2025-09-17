# pages/4_Projections.py
import streamlit as st
import pandas as pd
import numpy as np

from forecasting import determine_effective_projection_rates, calculate_projections
from constants import *
from helpers import load_css

# --- Theme Initialization ---
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# --- Page Configuration ---
st.set_page_config(page_title="Projections", page_icon="📈", layout="wide")

# --- Apply CSS ---
if st.session_state.theme == "light":
    load_css("style-light.css")
else:
    load_css("style-dark.css")

st.title("📈 Projections Dashboard")
st.info("This dashboard forecasts future performance based on assumptions configured in the sidebar on the Home page.")

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")
    
    st.write("") # Spacer
    def theme_changed_projections():
        st.session_state.theme = "light" if st.session_state.theme_selector_projections == "Light" else "dark"

    st.radio(
        "Theme",
        ["Dark", "Light"],
        index=1 if st.session_state.theme == "light" else 0,
        key="theme_selector_projections",
        on_change=theme_changed_projections,
        horizontal=True,
    )

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
proj_horizon = st.session_state.proj_horizon
goal_icf = st.session_state.proj_goal_icf
spend_dict = st.session_state.proj_spend_dict
cpqr_dict = st.session_state.proj_cpqr_dict
rate_method = st.session_state.proj_rate_method
manual_rates = st.session_state.proj_manual_rates
rolling_window = st.session_state.proj_rolling_window
icf_variation = st.session_state.shared_icf_variation

# --- Main Page Logic ---
effective_rates, method_desc = determine_effective_projection_rates(
    processed_data, ordered_stages, ts_col_map,
    rate_method, rolling_window, manual_rates, inter_stage_lags
)

projection_inputs = {
    'horizon': proj_horizon, 'spend_dict': spend_dict, 'cpqr_dict': cpqr_dict,
    'final_conv_rates': effective_rates, 'goal_icf': goal_icf,
    'site_performance_data': site_metrics, 'inter_stage_lags': inter_stage_lags,
    'icf_variation_percentage': icf_variation
}

(
    projection_results_df, avg_lag_used, lpi_date,
    ads_off_date, site_level_proj_df, lag_msg
) = calculate_projections(processed_data, ordered_stages, ts_col_map, projection_inputs)

st.divider()

st.subheader("Key Performance Indicators")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    with st.container(border=True):
        st.metric(label="Goal Total ICFs", value=f"{goal_icf:,}")
with kpi_col2:
    with st.container(border=True):
        st.metric(label="Estimated LPI Date", value=str(lpi_date))
with kpi_col3:
    with st.container(border=True):
        st.metric(label="Estimated Ads Off Date", value=str(ads_off_date))

if pd.notna(avg_lag_used):
    st.caption(f"ℹ️ Lag applied: **{avg_lag_used:.1f} days**. ({lag_msg}) | Rates based on: **{method_desc}**")
else:
    st.caption(f"ℹ️ Lag applied: **N/A**. ({lag_msg}) | Rates based on: **{method_desc}**")

st.divider()

dash_col1, dash_col2 = st.columns([3, 2])
with dash_col1:
    with st.container(border=True):
        st.subheader("Monthly Projections Table")
        if projection_results_df is not None and not projection_results_df.empty:
            display_df = projection_results_df.copy()
            # ... (formatting logic remains the same)
            cpicf_cols = ['Projected_CPICF_Cohort_Source_Low', 'Projected_CPICF_Cohort_Source_Mean', 'Projected_CPICF_Cohort_Source_High']
            if all(c in display_df.columns for c in cpicf_cols):
                display_df['Projected CPICF (Low-Mean-High)'] = display_df.apply(
                    lambda row: (f"${row[cpicf_cols[0]]:,.2f} - ${row[cpicf_cols[1]]:,.2f} - ${row[cpicf_cols[2]]:,.2f}"
                                 if pd.notna(row[cpicf_cols[1]]) else "-"), axis=1)
                final_cols = ['Forecasted_Ad_Spend', 'Forecasted_Qual_Referrals', 'Projected_ICF_Landed', 'Projected CPICF (Low-Mean-High)']
            else:
                final_cols = ['Forecasted_Ad_Spend', 'Forecasted_Qual_Referrals', 'Projected_ICF_Landed']
            display_df.index = display_df.index.strftime('%Y-%m')
            if 'Forecasted_Ad_Spend' in display_df: display_df['Forecasted_Ad_Spend'] = display_df['Forecasted_Ad_Spend'].map('${:,.2f}'.format)
            for col in ['Forecasted_Qual_Referrals', 'Projected_ICF_Landed']:
                if col in display_df: display_df[col] = display_df[col].map('{:,.0f}'.format)
            st.dataframe(display_df[final_cols], use_container_width=True)
        else:
            st.warning("Could not calculate projections based on the current inputs.")

with dash_col2:
    with st.container(border=True):
        st.subheader("Projected ICFs Landed Over Time")
        if projection_results_df is not None and not projection_results_df.empty:
            chart_data = projection_results_df[['Projected_ICF_Landed']].copy()
            chart_data['Projected_ICF_Landed'] = pd.to_numeric(chart_data['Projected_ICF_Landed'], errors='coerce').fillna(0)
            chart_data.index = chart_data.index.to_timestamp()
            st.line_chart(chart_data, use_container_width=True)
        else:
            st.info("Chart will appear here once projection data is calculated.")

st.divider()

with st.container(border=True):
    st.subheader("Site-Level Monthly Projections")
    if site_level_proj_df is not None and not site_level_proj_df.empty:
        st.dataframe(site_level_proj_df, use_container_width=True)
    else:
        st.info("Site-level projections are not available.")