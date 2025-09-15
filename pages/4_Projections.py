# pages/4_Projections.py
import streamlit as st
import pandas as pd
import numpy as np

# Direct imports from modules in the root directory
from forecasting import determine_effective_projection_rates, calculate_projections
from constants import *

st.set_page_config(
    page_title="Projections",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Projections Dashboard")
st.info("This dashboard forecasts future performance based on the planned ad spend and conversion rate assumptions you configure in the sidebar on the Home page.")

# Page Guard
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# Load Data from Session State
processed_data = st.session_state.referral_data_processed
ordered_stages = st.session_state.ordered_stages
ts_col_map = st.session_state.ts_col_map
inter_stage_lags = st.session_state.inter_stage_lags
site_metrics = st.session_state.site_metrics_calculated

# Load projection-specific settings
proj_horizon = st.session_state.proj_horizon
goal_icf = st.session_state.proj_goal_icf
spend_dict = st.session_state.proj_spend_dict
cpqr_dict = st.session_state.proj_cpqr_dict
rate_method = st.session_state.proj_rate_method
manual_rates = st.session_state.proj_manual_rates
rolling_window = st.session_state.proj_rolling_window
icf_variation = st.session_state.shared_icf_variation

# --- Main Page Logic ---

# 1. Determine the effective conversion rates
effective_rates, method_desc = determine_effective_projection_rates(
    processed_data, ordered_stages, ts_col_map,
    rate_method, rolling_window, manual_rates, inter_stage_lags
)

# 2. Assemble inputs for the calculation
projection_inputs = {
    'horizon': proj_horizon,
    'spend_dict': spend_dict,
    'cpqr_dict': cpqr_dict,
    'final_conv_rates': effective_rates,
    'goal_icf': goal_icf,
    'site_performance_data': site_metrics,
    'inter_stage_lags': inter_stage_lags,
    'icf_variation_percentage': icf_variation
}

# 3. Run the projection calculation
(
    projection_results_df,
    avg_lag_used,
    lpi_date,
    ads_off_date,
    site_level_proj_df,
    lag_msg
) = calculate_projections(processed_data, ordered_stages, ts_col_map, projection_inputs)

st.divider()

# --- NEW: KPI Row using st.columns ---
st.subheader("Key Performance Indicators")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

with kpi_col1:
    st.metric(label="Goal Total ICFs", value=f"{goal_icf:,}")

with kpi_col2:
    st.metric(label="Estimated LPI Date", value=str(lpi_date))

with kpi_col3:
    st.metric(label="Estimated Ads Off Date", value=str(ads_off_date))

if pd.notna(avg_lag_used):
    st.caption(f"ℹ️ Lag applied in projections: **{avg_lag_used:.1f} days**. ({lag_msg}) | Rates based on: **{method_desc}**")
else:
    st.caption(f"ℹ️ Lag applied in projections: **N/A**. ({lag_msg}) | Rates based on: **{method_desc}**")

st.divider()

# --- NEW: Dashboard Layout (Table on left, Chart on right) ---
dash_col1, dash_col2 = st.columns([3, 2]) # 3 parts width for col1, 2 parts for col2

with dash_col1:
    st.subheader("Monthly Projections Table")
    if projection_results_df is not None and not projection_results_df.empty:
        display_df = projection_results_df.copy()

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

        try:
            csv_data = projection_results_df.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button("Download Projections Data", csv_data, "spend_based_projections.csv", "text/csv", key='dl_projections')
        except Exception as e:
            st.warning(f"Could not prepare data for download: {e}")
    else:
        st.warning("Could not calculate projections based on the current inputs.")

with dash_col2:
    st.subheader("Projected ICFs Landed Over Time")
    if projection_results_df is not None and not projection_results_df.empty:
        chart_data = projection_results_df[['Projected_ICF_Landed']].copy()
        chart_data['Projected_ICF_Landed'] = pd.to_numeric(chart_data['Projected_ICF_Landed'], errors='coerce').fillna(0)
        chart_data.index = chart_data.index.to_timestamp()
        st.line_chart(chart_data, use_container_width=True)
    else:
        st.info("Chart will appear here once projection data is calculated.")

st.divider()

# Site-level projections remain below the main dashboard
st.subheader("Site-Level Monthly Projections")
if site_level_proj_df is not None and not site_level_proj_df.empty:
    st.dataframe(site_level_proj_df, use_container_width=True)
    try:
        csv_data = site_level_proj_df.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button("Download Site-Level Projections", csv_data, "site_level_projections.csv", "text/csv", key='dl_site_projections')
    except Exception as e:
        st.warning(f"Could not prepare site-level data for download: {e}")
else:
    st.info("Site-level projections are not available.")