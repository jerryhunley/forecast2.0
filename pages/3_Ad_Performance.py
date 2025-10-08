# pages/3_Ad_Performance.py
import streamlit as st
import pandas as pd

from scoring import score_performance_groups
from helpers import format_performance_df

st.set_page_config(page_title="Ad Performance", page_icon="📢", layout="wide")

if 'ranked_ad_source_df' not in st.session_state:
    st.session_state.ranked_ad_source_df = pd.DataFrame()
if 'ranked_ad_combo_df' not in st.session_state:
    st.session_state.ranked_ad_combo_df = pd.DataFrame()

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("📢 Ad Channel Performance")
st.info("Performance metrics grouped by UTM parameters. Adjust weights and click 'Apply' to recalculate scores.")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

with st.expander("Adjust Ad Performance Scoring Weights"):
    st.markdown("Adjust the importance of different metrics in the overall ad channel score. Changes here do not affect the Site Performance page.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Conversion Weights")
        st.slider("Qualified to Enrollment %", 0, 100, key="w_ad_qual_to_enroll")
        st.slider("Qualified to ICF %", 0, 100, key="w_ad_qual_to_icf")
        st.slider("ICF to Enrollment %", 0, 100, key="w_ad_icf_to_enroll")
        st.slider("StS to Appt %", 0, 100, key="w_ad_sts_to_appt")
    with c2:
        st.subheader("Speed / Lag Weights")
        st.markdown("_Lower is better for these metrics._")
        st.slider("Average time to first site action", 0, 100, key="w_ad_avg_time_to_first_action")
        st.slider("Avg time from StS to Appt Sched.", 0, 100, key="w_ad_lag_sts_appt")
    with c3:
        st.subheader("Negative Outcome Weights")
        st.markdown("_Lower is better for these metrics._")
        st.slider("Screen Fail % (from Qualified)", 0, 100, key="w_ad_generic_sf")
    
    if st.button("Apply & Recalculate Score", type="primary", use_container_width=True, key="apply_ad_weights"):
        weights = {
            "Qualified to Enrollment %": st.session_state.w_ad_qual_to_enroll,
            "ICF to Enrollment %": st.session_state.w_ad_icf_to_enroll,
            "Qualified to ICF %": st.session_state.w_ad_qual_to_icf,
            "StS to Appt %": st.session_state.w_ad_sts_to_appt,
            "Average time to first site action": st.session_state.w_ad_avg_time_to_first_action,
            "Avg time from StS to Appt Sched.": st.session_state.w_ad_lag_sts_appt,
            "Screen Fail % (from Qualified)": st.session_state.w_ad_generic_sf,
        }
        total_weight = sum(abs(w) for w in weights.values())
        weights_normalized = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {}
        
        if not st.session_state.enhanced_ad_source_metrics_df.empty:
            st.session_state.ranked_ad_source_df = score_performance_groups(st.session_state.enhanced_ad_source_metrics_df, weights_normalized, "UTM Source")
        
        if not st.session_state.enhanced_ad_combo_metrics_df.empty:
            st.session_state.ranked_ad_combo_df = score_performance_groups(st.session_state.enhanced_ad_combo_metrics_df, weights_normalized, "UTM Source/Medium")

# --- Performance by UTM Source ---
with st.container(border=True):
    st.subheader("Performance by UTM Source")
    if not st.session_state.ranked_ad_source_df.empty:
        df_to_display = st.session_state.ranked_ad_source_df
        display_cols_ad = [
            'UTM Source', 'Score', 'Grade', 'Total Qualified', 'StS Count', 'Appt Count', 'ICF Count', 'Enrollment Count',
            'Qualified to StS %', 'StS to Appt %', 'Qualified to ICF %', 'Qualified to Enrollment %', 'ICF to Enrollment %',
            'Average time to first site action', 'Avg time from StS to Appt Sched.', 'Screen Fail % (from Qualified)'
        ]
        display_cols_exist = [col for col in display_cols_ad if col in df_to_display.columns]
        
        final_ad_display = df_to_display[display_cols_exist]
        formatted_df = format_performance_df(final_ad_display)
        st.dataframe(formatted_df, hide_index=True, use_container_width=True)
    else:
        st.info("Adjust weights and click 'Apply & Recalculate Score' to generate the ranking table.")

st.write("") 

# --- Performance by UTM Source & Medium ---
if "UTM Medium" in st.session_state.referral_data_processed.columns:
    with st.container(border=True):
        st.subheader("Performance by UTM Source & Medium")
        if not st.session_state.ranked_ad_combo_df.empty:
            ranked_utm_combo_df = st.session_state.ranked_ad_combo_df.copy()
            if 'UTM Source/Medium' in ranked_utm_combo_df.columns:
                split_cols = ranked_utm_combo_df['UTM Source/Medium'].str.split(' / ', n=1, expand=True)
                ranked_utm_combo_df['UTM Source'] = split_cols[0]
                ranked_utm_combo_df['UTM Medium'] = split_cols[1]

            display_cols_combo = [
                'UTM Source', 'UTM Medium', 'Score', 'Grade', 'Total Qualified', 'StS Count', 'Appt Count', 'ICF Count', 'Enrollment Count',
                'Qualified to StS %', 'StS to Appt %', 'Qualified to ICF %', 'Qualified to Enrollment %', 'ICF to Enrollment %',
                'Average time to first site action', 'Avg time from StS to Appt Sched.', 'Screen Fail % (from Qualified)'
            ]
            display_cols_combo_exist = [col for col in display_cols_combo if col in ranked_utm_combo_df.columns]
            
            final_combo_display = ranked_utm_combo_df[display_cols_combo_exist]
            formatted_df_combo = format_performance_df(final_combo_display)
            st.dataframe(formatted_df_combo, hide_index=True, use_container_width=True)
        else:
            st.info("Adjust weights and click 'Apply & Recalculate Score' to generate the ranking table.")