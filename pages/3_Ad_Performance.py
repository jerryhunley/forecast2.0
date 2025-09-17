# pages/3_Ad_Performance.py
import streamlit as st
import pandas as pd

# Direct imports from modules in the root directory
from calculations import calculate_grouped_performance_metrics
from scoring import score_performance_groups
from helpers import format_performance_df

st.set_page_config(
    page_title="Ad Performance",
    page_icon="📢",
    layout="wide"
)

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("📢 Ad Channel Performance")
st.info("Performance metrics grouped by UTM parameters, scored using the weights defined below.")

# Page Guard
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Synced Assumption Controls ---
with st.expander("Adjust Performance Scoring Weights"):
    # Each slider reads its default value from session_state and writes the new value back to session_state.
    st.session_state.w_qual_to_enroll = st.slider("Qual (POF) -> Enrollment %", 0, 100, st.session_state.w_qual_to_enroll, key="w_q_enr_ad")
    st.session_state.w_icf_to_enroll = st.slider("ICF -> Enrollment %", 0, 100, st.session_state.w_icf_to_enroll, key="w_icf_enr_ad")
    st.session_state.w_qual_to_icf = st.slider("Qual (POF) -> ICF %", 0, 100, st.session_state.w_qual_to_icf, key="w_q_icf_ad")
    st.session_state.w_avg_ttc = st.slider("Avg Time to Contact (Sites)", 0, 100, st.session_state.w_avg_ttc, help="Lower is better.", key="w_ttc_ad")
    st.session_state.w_site_sf = st.slider("Site Screen Fail %", 0, 100, st.session_state.w_site_sf, help="Lower is better.", key="w_ssf_ad")
    st.session_state.w_sts_appt = st.slider("StS -> Appt Sched %", 0, 100, st.session_state.w_sts_appt, key="w_sts_appt_ad")
    st.session_state.w_appt_icf = st.slider("Appt Sched -> ICF %", 0, 100, st.session_state.w_appt_icf, key="w_appt_icf_ad")
    st.session_state.w_lag_q_icf = st.slider("Lag Qual -> ICF (Days)", 0, 100, st.session_state.w_lag_q_icf, help="Lower is better.", key="w_lag_ad")
    st.session_state.w_generic_sf = st.slider("Generic Screen Fail % (Ads)", 0, 100, st.session_state.w_generic_sf, help="Lower is better.", key="w_gsf_ad")
    st.session_state.w_proj_lag = st.slider("Generic Projection Lag (Ads)", 0, 100, st.session_state.w_proj_lag, help="Lower is better.", key="w_gpl_ad")
    st.caption("Changes will apply automatically and be reflected on the Site Performance page.")

# --- Calculation Logic ---
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
processed_data = st.session_state.referral_data_processed
ordered_stages = st.session_state.ordered_stages
ts_col_map = st.session_state.ts_col_map

# Check for UTM Columns
if "UTM Source" not in processed_data.columns:
    st.warning("The 'UTM Source' column was not found in the uploaded data. Ad Performance cannot be calculated.")
    st.stop()

# === Performance by UTM Source ===
with st.container(border=True):
    st.subheader("Performance by UTM Source")

    utm_source_metrics_df = calculate_grouped_performance_metrics(
        processed_data, ordered_stages, ts_col_map,
        grouping_col="UTM Source",
        unclassified_label="Unclassified Source"
    )

    if not utm_source_metrics_df.empty and weights_normalized:
        ranked_utm_source_df = score_performance_groups(
            utm_source_metrics_df, weights_normalized,
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
                st.dataframe(formatted_df, hide_index=True, use_container_width=True)
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

st.write("") # Spacer

# === Performance by UTM Source & Medium ===
if "UTM Medium" in processed_data.columns:
    with st.container(border=True):
        st.subheader("Performance by UTM Source & Medium")

        df_for_combo = processed_data.copy()
        df_for_combo['UTM Source/Medium'] = df_for_combo['UTM Source'].astype(str).fillna("Unclassified") + ' / ' + df_for_combo['UTM Medium'].astype(str).fillna("Unclassified")

        utm_combo_metrics_df = calculate_grouped_performance_metrics(
            df_for_combo, ordered_stages, ts_col_map,
            grouping_col="UTM Source/Medium",
            unclassified_label="Unclassified Combo"
        )

        if not utm_combo_metrics_df.empty and weights_normalized:
            ranked_utm_combo_df = score_performance_groups(
                utm_combo_metrics_df, weights_normalized,
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
                    st.dataframe(formatted_df, hide_index=True, use_container_width=True)
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
    st.warning("The 'UTM Medium' column was not found in the uploaded data.")