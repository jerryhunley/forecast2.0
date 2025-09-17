# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import io
import os

# Utility and Constant Imports
from parsing import parse_funnel_definition
from processing import preprocess_referral_data
from calculations import calculate_overall_inter_stage_lags, calculate_site_metrics
from constants import *
from helpers import load_css

# --- Set a unique key for this page's widgets ---
st.session_state.page_key = "app"

# --- Theme Initialization and Page Config ---
if "theme_selector" not in st.session_state:
    st.session_state.theme_selector = "Dark"

st.set_page_config(
    page_title="Recruitment Forecasting Tool",
    page_icon="assets/favicon.png", 
    layout="wide"
)

if st.session_state.theme_selector == "Light":
    load_css("style-light.css")
else:
    load_css("style-dark.css")

# --- Session State Initialization for App Data ---
required_keys = [
    'data_processed_successfully', 'referral_data_processed', 'funnel_definition',
    'ordered_stages', 'ts_col_map', 'site_metrics_calculated', 'inter_stage_lags',
    'weights_normalized', 'historical_spend_df', 'ad_spend_input_dict',
    'proj_horizon', 'proj_goal_icf', 'proj_spend_dict', 'proj_cpqr_dict',
    'proj_rate_method', 'proj_manual_rates', 'proj_rolling_window',
    'shared_icf_variation', 'ai_cpql_inflation', 'ai_ql_vol_threshold',
    'ai_ql_capacity_multiplier'
]
default_values = {
    'data_processed_successfully': False,
    'historical_spend_df': pd.DataFrame([
        {'Month (YYYY-MM)': (datetime.now() - pd.DateOffset(months=2)).strftime('%Y-%m'), 'Historical Spend': 45000.0},
        {'Month (YYYY-MM)': (datetime.now() - pd.DateOffset(months=1)).strftime('%Y-%m'), 'Historical Spend': 60000.0}
    ]),
    'proj_horizon': 12, 'proj_goal_icf': 100, 'shared_icf_variation': 10,
    'ai_cpql_inflation': 5.0, 'ai_ql_vol_threshold': 10.0, 'ai_ql_capacity_multiplier': 3.0
}
for key in required_keys:
    if key not in st.session_state:
        st.session_state[key] = default_values.get(key, None)

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")
    st.write("") 

    # Determine the current index based on session state
    current_index = 1 if st.session_state.get("theme_selector") == "Light" else 0

    # Create the radio button
    selected_theme = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=current_index,
        key=f"theme_selector_{st.session_state.page_key}", # Unique key per page
        horizontal=True,
    )

    # If the user's selection has changed, update the state and rerun
    if selected_theme != st.session_state.get("theme_selector"):
        st.session_state.theme_selector = selected_theme
        st.rerun()
    
    st.header("⚙️ Setup")
    st.info("Start here by uploading your data files.")
    st.warning("🔒 **Privacy Notice:** Do not upload files containing PII.", icon="⚠️")
    pii_checkbox = st.checkbox("I confirm my files do not contain PII.")

    if pii_checkbox:
        uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"])
        uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (CSV/TSV)", type=["csv", "tsv"])
    else:
        uploaded_referral_file = None
        uploaded_funnel_def_file = None

    st.divider()

    with st.expander("Global Assumptions & Weights"):
        st.subheader("Historical Ad Spend")
        edited_df = st.data_editor(st.session_state.historical_spend_df, num_rows="dynamic", key="hist_spend_editor")
        temp_spend_dict = {}
        valid_entries = True
        for _, row in edited_df.iterrows():
            try:
                if row['Month (YYYY-MM)'] and pd.notna(row['Historical Spend']):
                    month_period = pd.Period(row['Month (YYYY-MM)'], freq='M')
                    temp_spend_dict[month_period] = float(row['Historical Spend'])
            except Exception:
                st.error(f"Invalid month format: {row['Month (YYYY-MM)']}. Please use YYYY-MM.")
                valid_entries = False
                break
        if valid_entries:
            st.session_state.ad_spend_input_dict = temp_spend_dict
            st.session_state.historical_spend_df = edited_df
        
        st.divider()
        st.subheader("Performance Scoring Weights")
        weights = {
            "Qual to Enrollment %": st.slider("Qual (POF) -> Enrollment %", 0, 100, 10),
            "ICF to Enrollment %": st.slider("ICF -> Enrollment %", 0, 100, 10),
            "Qual -> ICF %": st.slider("Qual (POF) -> ICF %", 0, 100, 20),
            "Avg TTC (Days)": st.slider("Avg Time to Contact", 0, 100, 25, help="Lower is better."),
            "Site Screen Fail %": st.slider("Site Screen Fail %", 0, 100, 5, help="Lower is better."),
            "StS -> Appt %": st.slider("StS -> Appt Sched %", 0, 100, 30),
            "Appt -> ICF %": st.slider("Appt Sched -> ICF %", 0, 100, 15),
            "Lag Qual -> ICF (Days)": st.slider("Lag Qual (POF) -> ICF (Days)", 0, 100, 0, help="Lower is better."),
            "Site Projection Lag (Days)": st.slider("Site Projection Lag (Days)", 0, 100, 0, help="Lower is better."),
            "Screen Fail % (from ICF)": st.slider("Generic Screen Fail %", 0, 100, 5, help="Lower is better."),
            "Projection Lag (Days)": st.slider("Generic Projection Lag (Days)", 0, 100, 0, help="Lower is better.")
        }
        total_weight = sum(abs(w) for w in weights.values())
        st.session_state.weights_normalized = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {}

    with st.expander("Projection & AI Assumptions"):
        st.subheader("General Settings")
        st.session_state.proj_horizon = st.number_input("Projection Horizon (Months)", 1, 48, 12)
        st.session_state.proj_goal_icf = st.number_input("Goal ICFs (for 'Projections' Page)", 1, 10000, 100)
        st.session_state.shared_icf_variation = st.slider("Projected ICF Variation (+/- %)", 0, 50, 10)

        st.subheader("Future Spend & CPQR ('Projections' Page)")
        last_hist_month = st.session_state.referral_data_processed["Submission_Month"].max() if st.session_state.data_processed_successfully and st.session_state.referral_data_processed is not None else pd.Period(datetime.now(), 'M')
        future_months = pd.period_range(start=last_hist_month + 1, periods=st.session_state.proj_horizon, freq='M')
        
        if 'proj_spend_df_cache' not in st.session_state or len(st.session_state.proj_spend_df_cache) != len(future_months):
            st.session_state.proj_spend_df_cache = pd.DataFrame({'Month': future_months.strftime('%Y-%m'), 'Planned_Spend': [20000.0] * len(future_months)})
            st.session_state.proj_cpqr_df_cache = pd.DataFrame({'Month': future_months.strftime('%Y-%m'), 'Assumed_CPQR': [120.0] * len(future_months)})

        edited_spend_df = st.data_editor(st.session_state.proj_spend_df_cache, key='proj_spend_editor')
        edited_cpqr_df = st.data_editor(st.session_state.proj_cpqr_df_cache, key='proj_cpqr_editor')
        st.session_state.proj_spend_dict = {pd.Period(row['Month'], 'M'): row['Planned_Spend'] for _, row in edited_spend_df.iterrows()}
        st.session_state.proj_cpqr_dict = {pd.Period(row['Month'], 'M'): row['Assumed_CPQR'] for _, row in edited_cpqr_df.iterrows()}

        st.subheader("Conversion Rates ('Projections' Page)")
        st.session_state.proj_rate_method = st.radio("Rate Assumption:", ('Manual Input Below', 'Rolling Historical Average'), key='proj_rate_method_radio')
        cols = st.columns(2)
        st.session_state.proj_manual_rates = {
            f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}": cols[0].slider("Manual: POF -> PreScreen %", 0.0, 100.0, 100.0, format="%.1f%%") / 100.0,
            f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}": cols[0].slider("Manual: PreScreen -> StS %", 0.0, 100.0, 17.0, format="%.1f%%") / 100.0,
            f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}": cols[1].slider("Manual: StS -> Appt %", 0.0, 100.0, 33.0, format="%.1f%%") / 100.0,
            f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}": cols[1].slider("Manual: Appt -> ICF %", 0.0, 100.0, 35.0, format="%.1f%%") / 100.0,
        }
        st.session_state.proj_rolling_window = st.selectbox("Rolling Window:", [1, 3, 6, 999], index=1, format_func=lambda x: "Overall" if x == 999 else f"{x}-Month")

        st.subheader("AI Forecast Settings")
        st.session_state.ai_cpql_inflation = st.slider("CPQL Inflation Factor (%)", 0.0, 25.0, 5.0, 0.5)
        st.session_state.ai_ql_vol_threshold = st.slider("QL Volume Increase Threshold (%)", 1.0, 50.0, 10.0, 1.0)
        st.session_state.ai_ql_capacity_multiplier = st.slider("Monthly QL Capacity Multiplier", 1.0, 30.0, 3.0, 0.5)

st.title("📊 Recruitment Forecasting Tool")
st.header("Home & Data Setup")

if uploaded_referral_file and uploaded_funnel_def_file:
    st.info("Files uploaded. Click the button below to process and load the data.")
    if st.button("Process Uploaded Data", type="primary"):
        with st.spinner("Parsing files and processing data..."):
            try:
                referral_bytes_data = uploaded_referral_file.getvalue()
                header_df = pd.read_csv(io.BytesIO(referral_bytes_data), nrows=0, low_memory=False)
                pii_cols = [c for c in ["notes", "first name", "last name", "name", "phone", "email"] if c in [str(h).lower().strip() for h in header_df.columns]]

                if pii_cols:
                    original_col_names = [col for col in header_df.columns if str(col).lower().strip() in pii_cols]
                    st.error(f"PII Detected in columns: {', '.join(original_col_names)}. Please remove them and re-upload.", icon="🚫")
                    st.stop()
                
                funnel_def, ordered_st, ts_map = parse_funnel_definition(uploaded_funnel_def_file)
                
                if funnel_def and ordered_st and ts_map:
                    raw_df = pd.read_csv(io.BytesIO(referral_bytes_data))
                    processed_data = preprocess_referral_data(raw_df, funnel_def, ordered_st, ts_map)

                    if processed_data is not None and not processed_data.empty:
                        st.session_state.funnel_definition = funnel_def
                        st.session_state.ordered_stages = ordered_st
                        st.session_state.ts_col_map = ts_map
                        st.session_state.referral_data_processed = processed_data
                        st.session_state.inter_stage_lags = calculate_overall_inter_stage_lags(processed_data, ordered_st, ts_map)
                        st.session_state.site_metrics_calculated = calculate_site_metrics(processed_data, ordered_st, ts_map)
                        st.session_state.data_processed_successfully = True
                        st.success("Data processed successfully!")
                        st.rerun()
                    else:
                        st.error("Data processing failed after preprocessing.")
                else:
                    st.error("Funnel definition parsing failed.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)

if st.session_state.data_processed_successfully:
    st.success("Data is loaded and ready.")
    st.info("👈 Please select an analysis page from the sidebar to view the results.")
else:
    st.info("👋 **Welcome to the Recruitment Forecasting Tool!**")
    st.markdown("""
        1.  **Confirm No PII**: Check the box in the sidebar.
        2.  **Upload Your Data**: Use the file uploaders in the sidebar.
        3.  **Process Data**: Click the "Process Uploaded Data" button.
        4.  **Explore**: Navigate to the analysis pages.
    """)