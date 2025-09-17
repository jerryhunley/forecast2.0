# pages/2_Site_Performance.py
import streamlit as st
import pandas as pd
from scoring import score_sites
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="Site Performance", page_icon="🏆", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data from Session State ---
site_metrics = st.session_state.site_metrics_calculated
weights = st.session_state.weights_normalized

if site_metrics is not None and not site_metrics.empty and weights:
    ranked_sites_df = score_sites(site_metrics, weights)

    st.subheader("Key Performance Indicators (Overall)")
    
    total_qualified = ranked_sites_df['Total Qualified'].sum() if 'Total Qualified' in ranked_sites_df else 0
    total_enrollments = ranked_sites_df['Enrollment Count'].sum() if 'Enrollment Count' in ranked_sites_df else 0
    total_icfs = ranked_sites_df['ICF Count'].sum() if 'ICF Count' in ranked_sites_df else 0
    
    overall_qual_to_icf_rate = (total_icfs / total_qualified) * 100 if total_qualified > 0 else 0

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        with st.container(border=True):
            st.metric(label="Total Qualified Leads", value=f"{total_qualified:,}")
    with kpi_cols[1]:
        with st.container(border=True):
            st.metric(label="Total Enrollments", value=f"{total_enrollments:,}")
    with kpi_cols[2]:
        with st.container(border=True):
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