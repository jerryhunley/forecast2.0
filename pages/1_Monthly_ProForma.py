# pages/1_Monthly_ProForma.py
import streamlit as st
import pandas as pd
import sys
import os

# --- Add the root directory to the Python path ---
# This is necessary for Streamlit Cloud to find the 'utils' module.
# It gets the directory of the current script, goes up one level to the root,
# and adds that to the list of places Python looks for modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now the imports from your custom modules will work
from utils.calculations import calculate_proforma_metrics

st.set_page_config(
    page_title="Monthly ProForma",
    page_icon="📅",
    layout="wide"
)

st.title("📅 Monthly ProForma")
st.header("Historical Cohort Performance")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# ... (the rest of the file is the same) ...
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