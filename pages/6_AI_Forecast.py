# pages/6_AI_Forecast.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Direct imports from modules in the root directory
from forecasting import determine_effective_projection_rates, calculate_ai_forecast_core
from calculations import calculate_avg_lag_generic
from constants import *
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="AI Forecast", page_icon="🤖", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("🤖 AI Forecast (Goal-Based)")
st.info("""
Define your recruitment goals and assumptions below. The tool will estimate a monthly plan to meet your Last Patient In (LPI) date.
""")

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
weights = st.session_state.get("weights_normalized", {})

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

    st.divider()
    
    st.markdown("##### Global Model Assumptions")
    a1, a2, a3, a4 = st.columns(4)
    proj_horizon = a1.number_input("Projection Horizon (Months)", 1, 48, 12)
    icf_variation = a2.slider("Projected ICF Variation (+/- %)", 0, 50, 10)
    cpql_inflation = a3.slider("CPQL Inflation Factor (%)", 0.0, 25.0, 5.0, 0.5)
    ql_vol_threshold = a4.slider("QL Volume Increase Threshold (%)", 1.0, 50.0, 10.0, 1.0)
    ql_capacity_multiplier = st.slider("Monthly QL Capacity Multiplier", 1.0, 30.0, 3.0, 0.5, help="Controls how aggressively the AI plans monthly lead generation.")
        
    st.divider()

    st.markdown("##### Lag & Conversion Rate Assumptions")
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
    
    cr1, cr2, cr3, cr4 = st.columns(4)
    manual_rates = {
        f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}": cr1.slider("AI: POF -> PreScreen %", 0.0, 100.0, 90.0, format="%.1f%%", key='ai_cr_qps') / 100.0,
        f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}": cr2.slider("AI: PreScreen -> StS %", 0.0, 100.0, 25.0, format="%.1f%%", key='ai_cr_pssts') / 100.0,
        f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}": cr3.slider("AI: StS -> Appt %", 0.0, 100.0, 50.0, format="%.1f%%", key='ai_cr_sa') / 100.0,
        f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}": cr4.slider("AI: Appt -> ICF %", 0.0, 100.0, 60.0, format="%.1f%%", key='ai_cr_ai') / 100.0
    }

with st.expander("Optional Site Configurations"):
    site_activity_schedule = {}
    if site_metrics is not None and not site_metrics.empty:
        site_activity_df_data = [{"Site": s, "Activation Date": None, "Deactivation Date": None} for s in site_metrics['Site'].unique()]
        edited_activity_df = st.data_editor(pd.DataFrame(site_activity_df_data), hide_index=True, use_container_width=True, key="site_activity_editor")
        for _, row in edited_activity_df.iterrows():
            site_activity_schedule[row['Site']] = {
                'activation_period': pd.Period(row['Activation Date'], 'M') if pd.notna(row['Activation Date']) else None,
                'deactivation_period': pd.Period(row['Deactivation Date'], 'M') if pd.notna(row['Deactivation Date']) else None,
            }
    else:
        st.info("No site data available to configure.")

# --- EXECUTION LOGIC ---
if st.button("🚀 Generate Auto Forecast", type="primary", use_container_width=True):
    with st.spinner("Calculating forecast..."):
        effective_rates, _ = determine_effective_projection_rates(
            processed_data, ordered_stages, ts_col_map,
            rate_method, rolling_window, manual_rates, inter_stage_lags
        )

        avg_pof_icf_lag = calculate_avg_lag_generic(processed_data, ts_col_map.get(STAGE_PASSED_ONLINE_FORM), ts_col_map.get(STAGE_SIGNED_ICF))
        if pd.isna(avg_pof_icf_lag): avg_pof_icf_lag = 30.0

        ts_pof_col = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
        baseline_ql_volume = 50.0
        if ts_pof_col and ts_pof_col in processed_data.columns:
            pof_data = processed_data.dropna(subset=[ts_pof_col])
            if not pof_data.empty and 'Submission_Month' in pof_data.columns:
                monthly_counts = pof_data.groupby('Submission_Month').size()
                if not monthly_counts.empty:
                    baseline_ql_volume = monthly_counts.nlargest(6).mean()
        
        # Run primary forecast
        (
            df_primary, site_df_primary, ads_off_primary,
            message_primary, is_unfeasible_primary, actual_icfs_primary
        ) = calculate_ai_forecast_core(
            goal_lpi_date_dt_orig=datetime.combine(goal_lpi_date, datetime.min.time()), goal_icf_number_orig=goal_icf_num, estimated_cpql_user=base_cpql,
            icf_variation_percent=icf_variation, processed_df=processed_data, ordered_stages=ordered_stages,
            ts_col_map=ts_col_map, effective_projection_conv_rates=effective_rates, avg_overall_lag_days=avg_pof_icf_lag,
            site_metrics_df=site_metrics, projection_horizon_months=proj_horizon, site_caps_input={},
            site_activity_schedule=site_activity_schedule, site_scoring_weights_for_ai=weights,
            cpql_inflation_factor_pct=cpql_inflation, ql_vol_increase_threshold_pct=ql_vol_threshold,
            run_mode="primary",
            # --- THIS IS THE FIX ---
            ai_monthly_ql_capacity_multiplier=ql_capacity_multiplier,
            # ----------------------
            ai_lag_method=lag_method, ai_lag_p25_days=p25_lag, ai_lag_p50_days=p50_lag, ai_lag_p75_days=p75_lag,
            baseline_monthly_ql_volume_override=baseline_ql_volume
        )

        results_to_show = (df_primary, site_df_primary, ads_off_primary, message_primary, is_unfeasible_primary, actual_icfs_primary)

        if is_unfeasible_primary:
            st.info("Initial forecast is unfeasible. Running a 'best-case' scenario with an extended LPI date...")
            # Run best-case scenario
            (
                df_best, site_df_best, ads_off_best,
                message_best, is_unfeasible_best, actual_icfs_best
            ) = calculate_ai_forecast_core(
                goal_lpi_date_dt_orig=datetime.combine(goal_lpi_date, datetime.min.time()), goal_icf_number_orig=goal_icf_num, estimated_cpql_user=base_cpql,
                icf_variation_percent=icf_variation, processed_df=processed_data, ordered_stages=ordered_stages,
                ts_col_map=ts_col_map, effective_projection_conv_rates=effective_rates, avg_overall_lag_days=avg_pof_icf_lag,
                site_metrics_df=site_metrics, projection_horizon_months=proj_horizon, site_caps_input={},
                site_activity_schedule=site_activity_schedule, site_scoring_weights_for_ai=weights,
                cpql_inflation_factor_pct=cpql_inflation, ql_vol_increase_threshold_pct=ql_vol_threshold,
                run_mode="best_case_extended_lpi",
                # --- THIS IS THE FIX ---
                ai_monthly_ql_capacity_multiplier=ql_capacity_multiplier,
                # ----------------------
                ai_lag_method=lag_method, ai_lag_p25_days=p25_lag, ai_lag_p50_days=p50_lag, ai_lag_p75_days=p75_lag,
                baseline_monthly_ql_volume_override=baseline_ql_volume
            )
            results_to_show = (df_best, site_df_best, ads_off_best, message_best, is_unfeasible_best, actual_icfs_best)

    st.divider()
    st.subheader("Forecast Results")
    
    df_res, site_df_res, ads_off_res, message_res, is_unfeasible_res, actual_icfs_res = results_to_show
    
    if is_unfeasible_res: st.error(f"**Feasibility:** {message_res}")
    else: st.success(f"**Feasibility:** {message_res}")

    r1, r2, r3 = st.columns(3)
    with r1, st.container(border=True):
        st.metric("Target LPI Date (Original Goal)", goal_lpi_date.strftime("%Y-%m-%d"))
    with r2, st.container(border=True):
        st.metric("Projected/Goal ICFs", f"{actual_icfs_res:,.0f} / {goal_icf_num:,}")
    with r3, st.container(border=True):
        st.metric("Est. Ads Off Date (Generation)", ads_off_res)

    if df_res is not None and not df_res.empty:
        with st.container(border=True):
            st.subheader("Forecasted Monthly Performance")
            df_display = df_res.copy()
            df_display.rename(columns={'Target_QLs_POF': 'Planned QLs (POF)'}, inplace=True)
            st.dataframe(df_display, use_container_width=True)

    if site_df_res is not None and not site_df_res.empty:
        with st.container(border=True):
            st.subheader("Forecasted Site-Level Performance")
            st.dataframe(site_df_res, use_container_width=True)