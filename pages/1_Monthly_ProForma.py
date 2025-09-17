# pages/1_Monthly_ProForma.py
import streamlit as st
import pandas as pd
from calculations import calculate_proforma_metrics
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="Monthly ProForma", page_icon="📅", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data from Session State ---
processed_data = st.session_state.referral_data_processed
ordered_stages = st.session_state.ordered_stages
ts_col_map = st.session_state.ts_col_map
ad_spend_dict = st.session_state.ad_spend_input_dict

# --- Main Page Logic ---
if processed_data is not None and not processed_data.empty and ad_spend_dict:
    proforma_df = calculate_proforma_metrics(
        processed_data,
        ordered_stages,
        ts_col_map,
        ad_spend_dict
    )

    if not proforma_df.empty:
        # --- NEW: Wrapped in a container for the card effect ---
        with st.container(border=True):
            st.subheader("Monthly Performance Data")
            proforma_display = proforma_df.transpose()
            proforma_display.columns = [str(col) for col in proforma_display.columns]

            format_dict = {
                idx: ("${:,.2f}" if 'Cost' in idx or 'Spend' in idx else
                      ("{:.1%}" if '%' in idx else "{:,.0f}"))
                for idx in proforma_display.index
            }

            st.dataframe(proforma_display.style.format(format_dict, na_rep='-'), use_container_width=True)

            try:
                csv_data = proforma_df.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download ProForma Data",
                    data=csv_data,
                    file_name='monthly_proforma.csv',
                    mime='text/csv',
                    key='download_proforma'
                )
            except Exception as e:
                st.warning(f"Could not prepare data for download: {e}")

    else:
        st.warning("Could not generate the ProForma table. This may be due to a mismatch between the months in your data and the historical ad spend entered.")
else:
    st.info("Waiting for data to be processed or historical ad spend to be entered.")