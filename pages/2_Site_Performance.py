# pages/2_Site_Performance.py
import streamlit as st
import pandas as pd
from scoring import score_sites
from helpers import format_performance_df

st.set_page_config(page_title="Site Performance", page_icon="🏆", layout="wide")

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("🏆 Site Performance Dashboard")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Synced Assumption Controls ---
with st.expander("Adjust Performance Scoring Weights"):
    # Each slider reads its default value from session_state and writes the new value back to session_state.
    # This keeps them in sync with the sliders on the Ad Performance page.
    st.session_state.w_qual_to_enroll = st.slider("Qual (POF) -> Enrollment %", 0, 100, st.session_state.w_qual_to_enroll, key="w_q_enr_site")
    st.session_state.w_icf_to_enroll = st.slider("ICF -> Enrollment %", 0, 100, st.session_state.w_icf_to_enroll, key="w_icf_enr_site")
    st.session_state.w_qual_to_icf = st.slider("Qual (POF) -> ICF %", 0, 100, st.session_state.w_qual_to_icf, key="w_q_icf_site")
    st.session_state.w_avg_ttc = st.slider("Avg Time to Contact (Sites)", 0, 100, st.session_state.w_avg_ttc, help="Lower is better.", key="w_ttc_site")
    st.session_state.w_site_sf = st.slider("Site Screen Fail %", 0, 100, st.session_state.w_site_sf, help="Lower is better.", key="w_ssf_site")
    st.session_state.w_sts_appt = st.slider("StS -> Appt Sched %", 0, 100, st.session_state.w_sts_appt, key="w_sts_appt_site")
    st.session_state.w_appt_icf = st.slider("Appt Sched -> ICF %", 0, 100, st.session_state.w_appt_icf, key="w_appt_icf_site")
    st.session_state.w_lag_q_icf = st.slider("Lag Qual -> ICF (Days)", 0, 100, st.session_state.w_lag_q_icf, help="Lower is better.", key="w_lag_site")
    st.session_state.w_generic_sf = st.slider("Generic Screen Fail % (Ads)", 0, 100, st.session_state.w_generic_sf, help="Lower is better.", key="w_gsf_site")
    st.session_state.w_proj_lag = st.slider("Generic Projection Lag (Ads)", 0, 100, st.session_state.w_proj_lag, help="Lower is better.", key="w_gpl_site")
    st.caption("Changes will apply automatically and be reflected on the Ad Performance page.")

# --- Calculation Logic ---
# Create the weights dictionary from the session state values, ensuring they exist.
weights = {
    "Qual to Enrollment %": st.session_state.w_qual_to_enroll,
    "ICF to Enrollment %": st.session_state.w_icf_to_enroll,
    "Qual -> ICF %": st.session_state.w_qual_to_icf,
    "Avg TTC (Days)": st.session_state.w_avg_ttc,
    "Site Screen Fail %": st.session_state.w_site_sf,
    "StS -> Appt %": st.session_state.w_sts_appt,
    "Appt -> ICF %": st.session_state.w_appt_icf,
    "Lag Qual -> ICF (Days)": st.session_state.w_lag_q_icf,
    "Screen Fail % (from ICF)": st.session_state.w_generic_sf,
    "Projection Lag (Days)": st.session_state.w_proj_lag,
}
total_weight = sum(abs(w) for w in weights.values())
weights_normalized = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {}

# Load Data from Session State
site_metrics = st.session_state.site_metrics_calculated

if site_metrics is not None and not site_metrics.empty and weights_normalized:
    ranked_sites_df = score_sites(site_metrics, weights_normalized)

    st.subheader("Key Performance Indicators (Overall)")
    
    total_qualified = ranked_sites_df['Total Qualified'].sum() if 'Total Qualified' in ranked_sites_df else 0
    total_enrollments = ranked_sites_df['Enrollment Count'].sum() if 'Enrollment Count' in ranked_sites_df else 0
    total_icfs = ranked_sites_df['ICF Count'].sum() if 'ICF Count' in ranked_sites_df else 0
    
    overall_qual_to_icf_rate = (total_icfs / total_qualified) * 100 if total_qualified > 0 else 0

    kpi_cols = st.columns(3)
    with kpi_cols[0], st.container(border=True):
        st.metric(label="Total Qualified Leads", value=f"{total_qualified:,}")
    with kpi_cols[1], st.container(border=True):
        st.metric(label="Total Enrollments", value=f"{total_enrollments:,}")
    with kpi_cols[2], st.container(border=True):
        st.metric(label="Overall Qualified to ICF Rate", value=f"{overall_qual_to_icf_rate:.1f}%")
            
    st.divider()

    with st.container(border=True):
        st.subheader("Site Performance Ranking")
        
        display_cols = [
            'Site', 'Score', 'Grade', 'Total Qualified', 'PSA Count', 'StS Count',
            'Appt Count', 'ICF Count', 'Enrollment Count', 'Qual to Enrollment %',
            'ICF to Enrollment %', 'Qual -> ICF %', 'POF -> PSA %', 'PSA -> StS %',
            'StS -> Appt %', 'Appt -> ICF %', 'Avg TTC (Days)',
            'Site Screen Fail %', 'Lag Qual -> ICF (Days)', 'Site Projection Lag (Days)'
        ]
        display_cols_exist = [col for col in display_cols if col in ranked_sites_df.columns]
        
        if display_cols_exist:
            final_display_df = ranked_sites_df[display_cols_exist]
            if not final_display_df.empty:
                formatted_df = format_performance_df(final_display_df)
                st.dataframe(formatted_df, hide_index=True, use_container_width=True)
            else:
                st.warning("Could not generate the site ranking table.")
        else:
            st.warning("None of the standard display columns were found.")
else:
    st.warning("Site metrics have not been calculated.")