# pages/5_Funnel_Analysis.py
import streamlit as st
import pandas as pd
from datetime import datetime

# Direct imports from modules in the root directory
from forecasting import determine_effective_projection_rates, calculate_pipeline_projection, generate_funnel_narrative
from constants import *
from helpers import format_performance_df # We still need this helper

# --- Page Configuration ---
st.set_page_config(page_title="Funnel Analysis", page_icon="🔬", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("🔬 Funnel Analysis (Based on Current Pipeline)")
st.info("""
This forecast shows the expected outcomes (**ICFs & Enrollments**) from the leads **already in your funnel**.
It answers the question: "If we stopped all new recruitment activities today, what results would we still see and when?"
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

# --- Page-Specific Assumption Controls ---
with st.container(border=True):
    st.subheader("Funnel Analysis Assumptions")
    rate_method = st.radio(
        "Base Funnel Conversion Rates On:",
        options=('Manual Input Below', 'Rolling Historical Average'),
        index=1,
        key="fa_rate_method",
        horizontal=True
    )
    rolling_window = 0
    if rate_method == 'Rolling Historical Average':
        rolling_window = st.selectbox(
            "Select Rolling Window:",
            options=[1, 3, 6, 999],
            index=1,
            format_func=lambda x: "Overall Average" if x == 999 else f"{x}-Month",
            key='fa_rolling_window'
        )

    cols_rate = st.columns(5)
    manual_rates = {
        f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}": cols_rate[0].slider("FA: POF -> PreScreen %", 0.0, 100.0, 95.0, key='fa_cr_qps', format="%.1f%%") / 100.0,
        f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}": cols_rate[1].slider("FA: PreScreen -> StS %", 0.0, 100.0, 20.0, key='fa_cr_pssts', format="%.1f%%") / 100.0,
        f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}": cols_rate[2].slider("FA: StS -> Appt %", 0.0, 100.0, 45.0, key='fa_cr_sa', format="%.1f%%") / 100.0,
        f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}": cols_rate[3].slider("FA: Appt -> ICF %", 0.0, 100.0, 55.0, key='fa_cr_ai', format="%.1f%%") / 100.0,
        f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}": cols_rate[4].slider("FA: ICF -> Enrolled %", 0.0, 100.0, 85.0, key='fa_cr_ie', format="%.1f%%") / 100.0
    }

if st.button("🔬 Analyze Current Funnel", type="primary", use_container_width=True):
    effective_rates, rates_method_desc = determine_effective_projection_rates(
        processed_data, ordered_stages, ts_col_map,
        rate_method, rolling_window, manual_rates, inter_stage_lags
    )
    st.session_state.funnel_analysis_results = calculate_pipeline_projection(
        _processed_df=processed_data,
        ordered_stages=ordered_stages,
        ts_col_map=ts_col_map,
        inter_stage_lags=inter_stage_lags,
        conversion_rates=effective_rates,
        lag_assumption_model=None
    )
    st.session_state.funnel_narrative_data = generate_funnel_narrative(
        st.session_state.funnel_analysis_results.get('in_flight_df_for_narrative', pd.DataFrame()),
        ordered_stages, effective_rates, inter_stage_lags
    )
    st.session_state.funnel_analysis_rates_desc = rates_method_desc

st.divider()

if 'funnel_analysis_results' in st.session_state and st.session_state.funnel_analysis_results:
    results = st.session_state.funnel_analysis_results
    rates_desc = st.session_state.get('funnel_analysis_rates_desc', "N/A")

    st.caption(f"**Projection Using: {rates_desc} Conversion Rates**")
    
    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1, st.container(border=True):
        st.metric("Total Expected ICF Yield from Funnel", f"{results['total_icf_yield']:,.1f}")
    with kpi_col2, st.container(border=True):
        st.metric("Total Expected Enrollment Yield from Funnel", f"{results['total_enroll_yield']:,.1f}")
            
    st.write("")

    # --- THIS IS THE RESTORED NARRATIVE/BREAKDOWN BLOCK ---
    narrative_steps = st.session_state.get('funnel_narrative_data', [])
    if narrative_steps:
        with st.container(border=True):
            st.subheader("Funnel Breakdown by Stage")
            for step in narrative_steps:
                if step['leads_at_stage'] > 0:
                    st.markdown(f"##### From '{step['current_stage']}'")
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Leads Currently At This Stage", f"{step['leads_at_stage']:,}")
                    c2.metric(f"Conversion to '{step['next_stage']}'", f"{step['conversion_rate']:.1%}" if step['conversion_rate'] is not None else "N/A")
                    c3.metric(f"Avg. Time to '{step['next_stage']}'", f"{step['lag_to_next_stage']:.1f} Days" if pd.notna(step.get('lag_to_next_stage')) else "N/A")
                    
                    if step['downstream_projections']:
                        with st.expander("View downstream projections from this group"):
                             for proj in step['downstream_projections']:
                                time_text = f" in an avg. of **{proj['cumulative_lag_days']:.1f} days**" if pd.notna(proj.get('cumulative_lag_days')) else ""
                                st.info(f"**~{proj['projected_count']:.1f}** will advance to **'{proj['stage_name']}'**{time_text}", icon="➡️")
                    st.divider()

    st.write("")

    with st.container(border=True):
        st.subheader("Projected Monthly Landings (Future)")
        results_df = results['results_df']
        if not results_df.empty:
            st.dataframe(results_df[['Projected_ICF_Landed', 'Projected_Enrollments_Landed']].style.format("{:,.0f}"), use_container_width=True)
            
            st.subheader("Cumulative Future Projections Over Time")
            chart_df = results_df[['Cumulative_ICF_Landed', 'Cumulative_Enrollments_Landed']].copy()
            chart_df.index = chart_df.index.to_timestamp()
            st.line_chart(chart_df)
        else:
            st.info("No future landings are projected from the current in-flight leads.")