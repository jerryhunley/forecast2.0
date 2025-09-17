# pages/4_Projections.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from forecasting import determine_effective_projection_rates, calculate_projections
from constants import *

st.set_page_config(
    page_title="Projections",
    page_icon="📈",
    layout="wide"
)

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("📈 Projections Dashboard")
st.info("This dashboard forecasts future performance based on the assumptions you configure below.")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- All assumption controls are now on this page ---
with st.expander("Configure Projection Assumptions", expanded=True):
    c1, c2, c3 = st.columns(3)
    proj_horizon = c1.number_input("Projection Horizon (Months)", 1, 48, 12, key="proj_horizon")
    goal_icf = c2.number_input("Goal Total ICFs", 1, 10000, 100, key="proj_goal_icf")
    icf_variation = c3.slider("Projected ICF Variation (+/- %)", 0, 50, 10, key="proj_icf_var")

    st.subheader("Future Spend & CPQR")
    last_hist_month = st.session_state.referral_data_processed["Submission_Month"].max()
    future_months = pd.period_range(start=last_hist_month + 1, periods=proj_horizon, freq='M')
    
    if 'proj_spend_df_cache' not in st.session_state or len(st.session_state.get('proj_spend_df_cache', [])) != len(future_months):
        st.session_state.proj_spend_df_cache = pd.DataFrame({'Month': future_months.strftime('%Y-%m'), 'Planned_Spend': [20000.0] * len(future_months)})
        st.session_state.proj_cpqr_df_cache = pd.DataFrame({'Month': future_months.strftime('%Y-%m'), 'Assumed_CPQR': [120.0] * len(future_months)})

    spend_col, cpqr_col = st.columns(2)
    with spend_col:
        st.write("Planned Spend ($)")
        edited_spend_df = st.data_editor(st.session_state.proj_spend_df_cache, key='proj_spend_editor_page')
    with cpqr_col:
        st.write("Assumed CPQR ($)")
        edited_cpqr_df = st.data_editor(st.session_state.proj_cpqr_df_cache, key='proj_cpqr_editor_page')
    
    spend_dict = {pd.Period(row['Month'], 'M'): row['Planned_Spend'] for _, row in edited_spend_df.iterrows()}
    cpqr_dict = {pd.Period(row['Month'], 'M'): row['Assumed_CPQR'] for _, row in edited_cpqr_df.iterrows()}
    
    st.subheader("Conversion Rates")
    rate_method = st.radio("Rate Assumption:", ('Manual Input Below', 'Rolling Historical Average'), key='proj_rate_method_radio_page')
    manual_rates = {}
    cols = st.columns(4)
    manual_rates[f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}"] = cols[0].slider("POF -> PreScreen %", 0.0, 100.0, 100.0, format="%.1f%%", key="proj_rate_1") / 100.0
    manual_rates[f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}"] = cols[1].slider("PreScreen -> StS %", 0.0, 100.0, 17.0, format="%.1f%%", key="proj_rate_2") / 100.0
    manual_rates[f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}"] = cols[2].slider("StS -> Appt %", 0.0, 100.0, 33.0, format="%.1f%%", key="proj_rate_3") / 100.0
    manual_rates[f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"] = cols[3].slider("Appt -> ICF %", 0.0, 100.0, 35.0, format="%.1f%%", key="proj_rate_4") / 100.0
    rolling_window = st.selectbox("Rolling Window:", [1, 3, 6, 999], index=1, format_func=lambda x: "Overall" if x == 999 else f"{x}-Month", key="proj_rolling")

# Load Data from Session State
processed_data = st.session_state.referral_data_processed
ordered_stages = st.session_state.ordered_stages
ts_col_map = st.session_state.ts_col_map
inter_stage_lags = st.session_state.inter_stage_lags
site_metrics = st.session_state.site_metrics_calculated

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
with kpi_col1, st.container(border=True):
    st.metric(label="Goal Total ICFs", value=f"{goal_icf:,}")
with kpi_col2, st.container(border=True):
    st.metric(label="Estimated LPI Date", value=str(lpi_date))
with kpi_col3, st.container(border=True):
    st.metric(label="Estimated Ads Off Date", value=str(ads_off_date))

if pd.notna(avg_lag_used):
    st.caption(f"ℹ️ Lag applied: **{avg_lag_used:.1f} days**. ({lag_msg}) | Rates based on: **{method_desc}**")
else:
    st.caption(f"ℹ️ Lag applied: **N/A**. ({lag_msg}) | Rates based on: **{method_desc}**")

st.divider()

dash_col1, dash_col2 = st.columns([3, 2])
with dash_col1, st.container(border=True):
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
    else:
        st.warning("Could not calculate projections.")

with dash_col2, st.container(border=True):
    st.subheader("Projected ICFs Landed Over Time")
    if projection_results_df is not None and not projection_results_df.empty:
        chart_data = projection_results_df[['Projected_ICF_Landed']].copy()
        chart_data['Projected_ICF_Landed'] = pd.to_numeric(chart_data['Projected_ICF_Landed'], errors='coerce').fillna(0)
        chart_data.index = chart_data.index.to_timestamp()
        st.line_chart(chart_data, use_container_width=True)
    else:
        st.info("Chart will appear after calculation.")

st.divider()

with st.container(border=True):
    st.subheader("Site-Level Monthly Projections")
    if site_level_proj_df is not None and not site_level_proj_df.empty:
        st.dataframe(site_level_proj_df, use_container_width=True)
    else:
        st.info("Site-level projections are not available.")