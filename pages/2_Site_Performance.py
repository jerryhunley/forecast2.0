# pages/2_Site_Performance.py
import streamlit as st
import pandas as pd

from scoring import score_sites
from helpers import format_performance_df

st.set_page_config(
    page_title="Site Performance",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Site Performance Ranking")
st.info("Metrics are scored based on the weights defined in the sidebar on the Home page.")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

site_metrics = st.session_state.site_metrics_calculated
weights = st.session_state.weights_normalized

if site_metrics is not None and not site_metrics.empty and weights:
    ranked_sites_df = score_sites(site_metrics, weights)
    st.subheader("Site Ranking Results")
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
            st.dataframe(formatted_df, hide_index=True)
            try:
                csv_data = final_display_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Site Ranking Data", csv_data, "site_performance_ranking.csv", "text/csv", key='download_site_perf')
            except Exception as e:
                st.warning(f"Could not prepare data for download: {e}")
        else:
            st.warning("Could not generate the site ranking table.")
    else:
        st.warning("None of the standard display columns were found in the calculated data.")
elif site_metrics is None or site_metrics.empty:
    st.warning("Site metrics not calculated. Check for a 'Site' column in your data.")
else:
    st.info("Waiting for data and weights to be configured.")