# pages/3_Ad_Performance.py
import streamlit as st
import pandas as pd
import sys
import os

# --- Add the root directory to the Python path ---
# This is necessary for Streamlit Cloud to find the 'utils' module.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now the imports from your custom modules will work
from utils.calculations import calculate_grouped_performance_metrics
from utils.scoring import score_performance_groups
from utils.helpers import format_performance_df

st.set_page_config(
    page_title="Ad Performance",
    page_icon="📢",
    layout="wide"
)

st.title("📢 Ad Channel Performance")
st.info("Performance metrics grouped by UTM parameters, scored using the same weights as Site Performance (generic versions of metrics like Screen Fail % are used).")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data from Session State ---
processed_data = st.session_state.referral_data_processed
ordered_stages = st.session_state.ordered_stages
ts_col_map = st.session_state.ts_col_map
weights = st.session_state.weights_normalized

# --- Check for UTM Columns ---
if "UTM Source" not in processed_data.columns:
    st.warning("The 'UTM Source' column was not found in the uploaded data. Ad Performance cannot be calculated.")
    st.stop()

# --- Main Page Logic ---

# === Performance by UTM Source ===
st.subheader("Performance by UTM Source")

utm_source_metrics_df = calculate_grouped_performance_metrics(
    processed_data, ordered_stages, ts_col_map,
    grouping_col="UTM Source",
    unclassified_label="Unclassified Source"
)

if not utm_source_metrics_df.empty:
    ranked_utm_source_df = score_performance_groups(
        utm_source_metrics_df, weights,
        group_col_name="UTM Source"
    )

    display_cols_ad = [
        'UTM Source', 'Score', 'Grade', 'Total Qualified', 'PSA Count', 'StS Count',
        'Appt Count', 'ICF Count', 'Enrollment Count', 'Qual to Enrollment %',
        'ICF to Enrollment %', 'Qual -> ICF %', 'POF -> PSA %', 'PSA -> StS %',
        'StS -> Appt %', 'Appt -> ICF %', 'Lag Qual -> ICF (Days)',
        'Projection Lag (Days)', 'Screen Fail % (from ICF)'
    ]
    display_cols_exist = [col for col in display_cols_ad if col in ranked_utm_source_df.columns]
    
    if display_cols_exist:
        final_ad_display = ranked_utm_source_df[display_cols_exist]

        if not final_ad_display.empty:
            formatted_df = format_performance_df(final_ad_display)
            st.dataframe(formatted_df, use_container_width=True, hide_index=True)
            try:
                csv_data = final_ad_display.to_csv(index=False).encode('utf-8')
                st.download_button("Download UTM Source Performance", csv_data, "utm_source_performance.csv", "text/csv", key='dl_ad_source')
            except Exception as e:
                st.warning(f"Could not prepare data for download: {e}")
        else:
            st.info("No data to display for UTM Source performance.")
    else:
        st.info("Could not find any standard columns to display for UTM Source performance.")
else:
    st.info("Could not calculate performance metrics for UTM Source.")


st.divider()

# === Performance by UTM Source & Medium ===
st.subheader("Performance by UTM Source & Medium")

if "UTM Medium" in processed_data.columns:
    df_for_combo = processed_data.copy()
    df_for_combo['UTM Source/Medium'] = df_for_combo['UTM Source'].astype(str).fillna("Unclassified") + ' / ' + df_for_combo['UTM Medium'].astype(str).fillna("Unclassified")

    utm_combo_metrics_df = calculate_grouped_performance_metrics(
        df_for_combo, ordered_stages, ts_col_map,
        grouping_col="UTM Source/Medium",
        unclassified_label="Unclassified Combo"
    )

    if not utm_combo_metrics_df.empty:
        ranked_utm_combo_df = score_performance_groups(
            utm_combo_metrics_df, weights,
            group_col_name="UTM Source/Medium"
        )
        
        if 'UTM Source/Medium' in ranked_utm_combo_df.columns:
            split_cols = ranked_utm_combo_df['UTM Source/Medium'].str.split(' / ', n=1, expand=True)
            ranked_utm_combo_df['UTM Source'] = split_cols[0]
            ranked_utm_combo_df['UTM Medium'] = split_cols[1]

        display_cols_combo = [
            'UTM Source', 'UTM Medium', 'Score', 'Grade', 'Total Qualified', 'PSA Count',
            'StS Count', 'Appt Count', 'ICF Count', 'Enrollment Count', 'Qual to Enrollment %',
            'ICF to Enrollment %', 'Qual -> ICF %', 'POF -> PSA %', 'PSA -> StS %',
            'StS -> Appt %', 'Appt -> ICF %', 'Lag Qual -> ICF (Days)',
            'Projection Lag (Days)', 'Screen Fail % (from ICF)'
        ]
        display_cols_combo_exist = [col for col in display_cols_combo if col in ranked_utm_combo_df.columns]
        
        if display_cols_combo_exist:
            final_combo_display = ranked_utm_combo_df[display_cols_combo_exist]

            if not final_combo_display.empty:
                formatted_df = format_performance_df(final_combo_display)
                st.dataframe(formatted_df, use_container_width=True, hide_index=True)
                try:
                    csv_data = final_combo_display.to_csv(index=False).encode('utf-8')
                    st.download_button("Download UTM Source/Medium Performance", csv_data, "utm_source_medium_performance.csv", "text/csv", key='dl_ad_combo')
                except Exception as e:
                    st.warning(f"Could not prepare data for download: {e}")
            else:
                st.info("No data to display for UTM Source/Medium performance.")
        else:
            st.info("Could not find any standard columns to display for UTM Source/Medium performance.")
    else:
        st.info("Could not calculate performance metrics for UTM Source/Medium.")
else:
    st.warning("The 'UTM Medium' column was not found in the uploaded data. Combined Source/Medium performance cannot be calculated.")