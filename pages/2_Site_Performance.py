# pages/2_Site_Performance.py
import streamlit as st
import pandas as pd

from utils.scoring import score_sites
from utils.helpers import format_performance_df # Use our new helper

st.set_page_config(
    page_title="Site Performance",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Site Performance Ranking")
st.info("Performance metrics are calculated for each site and then scored based on the weights defined in the sidebar on the Home page.")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data from Session State ---
site_metrics = st.session_state.site_metrics_calculated
weights = st.session_state.weights_normalized

# --- Main Page Logic ---
if site_metrics is not None and not site_metrics.empty and weights:
    ranked_sites_df = score_sites(site_metrics, weights)

    st.subheader("Site Ranking Results")

    # Define the columns to display in order
    display_cols = [
        'Site', 'Score', 'Grade', 'Total Qualified', 'PSA Count', 'StS Count',
        'Appt Count', 'ICF Count', 'Enrollment Count', 'Qual to Enrollment %',
        'ICF to Enrollment %', 'Qual -> ICF %', 'POF -> PSA %', 'PSA -> StS %',
        'StS -> Appt %', 'Appt -> ICF %', 'Avg TTC (Days)',
        'Site Screen Fail %', 'Lag Qual -> ICF (Days)', 'Site Projection Lag (Days)'
    ]

    # Filter out any columns that might not have been generated
    display_cols_exist = [col for col in display_cols if col in ranked_sites_df.columns]
    final_display_df = ranked_sites_df[display_cols_exist]

    if not final_display_df.empty:
        # Use the helper function to format the dataframe for display
        formatted_df = format_performance_df(final_display_df)
        st.dataframe(formatted_df, use_container_width=True, hide_index=True)

        # --- Download Button ---
        try:
            csv_data = final_display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Site Ranking Data",
                data=csv_data,
                file_name='site_performance_ranking.csv',
                mime='text/csv',
                key='download_site_perf'
            )
        except Exception as e:
            st.warning(f"Could not prepare data for download: {e}")
    else:
        st.warning("Could not generate the site ranking table after filtering columns.")

elif site_metrics is None or site_metrics.empty:
    st.warning("Site metrics have not been calculated. This usually means the 'Site' column was not found in your uploaded referral data.")
else:
    st.info("Waiting for data and weights to be configured.")
