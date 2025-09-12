# pages/6_AI_Forecast.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Direct imports from modules in the root directory
from forecasting import determine_effective_projection_rates, calculate_ai_forecast_core
from calculations import calculate_avg_lag_generic
from constants import *

st.set_page_config(
    page_title="AI Forecast",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Forecast (Goal-Based)")
st.info("""
Define your recruitment goals. The tool will estimate a monthly plan to meet your Last Patient In (LPI) date.
Settings for CPQL Inflation and Monthly QL Capacity can be adjusted in the sidebar on the Home page.
""")

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
weights = st.session_state.weights_normalized

# Shared settings from sidebar
icf_variation = st.session_state.shared_icf_variation
cpql_inflation = st.session_state.ai_cpql_inflation
ql_vol_threshold = st.session_state.ai_ql_vol_threshold
ql_capacity_multiplier = st.session_state.ai_ql_capacity_multiplier
proj_horizon = st.session_state.proj_horizon


# Page-Specific Goal & Assumption Controls
st.subheader("Define Your Goals")
g1, g2, g3 = st.columns(3)
with g1:
    goal_lpi_date = st.date_input("Target LPI Date", value=datetime.now() + pd.DateOffset(months=12))
with g2:
    goal_icf_num = st.number_input("Target Total ICFs", min_value=1, value=100, step=10)
with g3:
    base_cpql = st.number_input("Base Estimated CPQL (POF)", min_value=1.0, value=75.0, step=5.0, format="%.2f")

st.divider()
st.subheader("Auto Forecast Assumptions")

# Lag Assumption
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

# Rate Assumption
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

st.divider()
st.subheader("Optional Site Configurations")

# Site Activity Dates
with st.expander("Site Activation/Deactivation Dates"):
    st.caption("Define when sites are active for QL allocation. Leave blank if always active.")
    site_activity_schedule = {}
    if site_metrics is not None and not site_metrics.empty:
        site_activity_df_data = [{"Site": s, "Activation Date": None, "Deactivation Date": None} for s in site_metrics['Site'].unique()]
        edited_activity_df = st.data_editor(
            pd.DataFrame(site_activity_df_data),
            column_config={
                "Site": st.column_config.TextColumn(disabled=True),
                "Activation Date": st.column_config.DateColumn(),
                "Deactivation Date": st.column_config.DateColumn(),
            },
            hide_index=True, key="site_activity_editor"
        )
        for _, row in edited_activity_df.iterrows():
            site_activity_schedule[row['Site']] = {
                'activation_period': pd.Period(row['Activation Date'], 'M') if pd.notna(row['Activation Date']) else None,
                'deactivation_period': pd.Period(row['Deactivation Date'], 'M') if pd.notna(row['Deactivation Date']) else None,
            }
    else:
        st.info("No site data available to configure.")


# Site QL Caps
with st.expander("Site-Specific Monthly QL Caps"):
    st.caption("Set a maximum number of 'Passed Online Form' (POF) leads a site can handle per month. Leave blank for no cap.")
    site_caps = {}
    if site_metrics is not None and not site_metrics.empty:
        site_caps_df_data = [{"Site": s, "Monthly POF Cap": None} for s in site_metrics['Site'].unique()]
        edited_caps_df = st.data_editor(
            pd.DataFrame(site_caps_df_data),
            column_config={
                "Site": st.column_config.TextColumn(disabled=True),
                "Monthly POF Cap": st.column_config.NumberColumn(min_value=0, step=1)
            },
            hide_index=True, key="site_caps_editor"
        )
        for _, row in edited_caps_df.iterrows():
            if pd.notna(row['Monthly POF Cap']):
                site_caps[row['Site']] = int(row['Monthly POF Cap'])
    else:
        st.info("No site data available to configure.")


st.divider()

if st.button("🚀 Generate Auto Forecast", type="primary"):
    # 1. Determine effective rates
    effective_rates, rates_method_desc = determine_effective_projection_rates(
        processed_data, ordered_stages, ts_col_map,
        rate_method, rolling_window, manual_rates, inter_stage_lags
    )

    # 2. Determine effective lag
    avg_pof_icf_lag = calculate_avg_lag_generic(processed_data, ts_col_map.get(STAGE_PASSED_ONLINE_FORM), ts_col_map.get(STAGE_SIGNED_ICF))
    if pd.isna(avg_pof_icf_lag): avg_pof_icf_lag = 30.0 # Default fallback

    # 3. Calculate baseline QL volume
    ts_pof_col = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
    baseline_ql_volume = 50.0 # Default fallback
    if ts_pof_col and ts_pof_col in processed_data.columns:
        pof_data = processed_data.dropna(subset=[ts_pof_col])
        if not pof_data.empty and 'Submission_Month' in pof_data.columns:
            monthly_counts = pof_data.groupby('Submission_Month').size()
            if not monthly_counts.empty:
                baseline_ql_volume = monthly_counts.nlargest(6).mean()
    
    # 4. Run primary forecast
    run_mode_primary = "primary"
    st.session_state.ai_forecast_primary_results = calculate_ai_forecast_core(
        goal_lpi_date_dt_orig=datetime.combine(goal_lpi_date, datetime.min.time()), goal_icf_number_orig=goal_icf_num, estimated_cpql_user=base_cpql,
        icf_variation_percent=icf_variation, processed_df=processed_data, ordered_stages=ordered_stages,
        ts_col_map=ts_col_map, effective_projection_conv_rates=effective_rates, avg_overall_lag_days=avg_pof_icf_lag,
        site_metrics_df=site_metrics, projection_horizon_months=proj_horizon, site_caps_input=site_caps,
        site_activity_schedule=site_activity_schedule, site_scoring_weights_for_ai=weights,
        cpql_inflation_factor_pct=cpql_inflation, ql_vol_increase_threshold_pct=ql_vol_threshold,
        run_mode=run_mode_primary, ai_monthly_ql_capacity_multiplier=ql_capacity_multiplier,
        ai_lag_method=lag_method, ai_lag_p25_days=p25_lag, ai_lag_p50_days=p50_lag, ai_lag_p75_days=p75_lag,
        baseline_monthly_ql_volume_override=baseline_ql_volume
    )
    
    # 5. If primary is unfeasible, run best-case scenario
    _, _, _, _, is_unfeasible, _ = st.session_state.ai_forecast_primary_results
    st.session_state.show_best_case = is_unfeasible
    if is_unfeasible:
        st.info("Initial forecast is unfeasible. Running a 'best-case' scenario with an extended LPI date...")
        run_mode_best_case = "best_case_extended_lpi"
        st.session_state.ai_forecast_best_case_results = calculate_ai_forecast_core(
            goal_lpi_date_dt_orig=datetime.combine(goal_lpi_date, datetime.min.time()), goal_icf_number_orig=goal_icf_num, estimated_cpql_user=base_cpql,
            icf_variation_percent=icf_variation, processed_df=processed_data, ordered_stages=ordered_stages,
            ts_col_map=ts_col_map, effective_projection_conv_rates=effective_rates, avg_overall_lag_days=avg_pof_icf_lag,
            site_metrics_df=site_metrics, projection_horizon_months=proj_horizon, site_caps_input=site_caps,
            site_activity_schedule=site_activity_schedule, site_scoring_weights_for_ai=weights,
            cpql_inflation_factor_pct=cpql_inflation, ql_vol_increase_threshold_pct=ql_vol_threshold,
            run_mode=run_mode_best_case, ai_monthly_ql_capacity_multiplier=ql_capacity_multiplier,
            ai_lag_method=lag_method, ai_lag_p25_days=p25_lag, ai_lag_p50_days=p50_lag, ai_lag_p75_days=p75_lag,
            baseline_monthly_ql_volume_override=baseline_ql_volume
        )


# --- Display Results ---
results_to_show = None
if st.session_state.get('show_best_case', False):
    results_to_show = st.session_state.get('ai_forecast_best_case_results')
else:
    results_to_show = st.session_state.get('ai_forecast_primary_results')

if results_to_show:
    df, site_df, ads_off, message, is_unfeasible, actual_icfs = results_to_show

    st.divider()
    st.subheader("Forecast Results")
    
    if is_unfeasible: st.error(f"**Feasibility:** {message}")
    else: st.success(f"**Feasibility:** {message}")

    r1, r2, r3 = st.columns(3)
    r1.metric("Target LPI Date (Original Goal)", goal_lpi_date.strftime("%Y-%m-%d"))
    r2.metric("Projected/Goal ICFs", f"{actual_icfs:,.0f} / {goal_icf_num:,}")
    r3.metric("Est. Ads Off Date (Generation)", ads_off)

    if df is not None and not df.empty:
        st.subheader("Forecasted Monthly Performance")
        df_display = df.copy()
        df_display.rename(columns={'Target_QLs_POF': 'Planned QLs (POF)'}, inplace=True)
        st.dataframe(df_display)

    if site_df is not None and not site_df.empty:
        st.subheader("Forecasted Site-Level Performance")
        st.dataframe(site_df)