# helpers.py
import streamlit as st
import pandas as pd

def format_performance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Applies standard formatting to a performance DataFrame for display."""
    if df.empty:
        return df

    formatted_df = df.copy()

    if 'Score' in formatted_df.columns:
        formatted_df['Score'] = formatted_df['Score'].round(1)

    for col in formatted_df.columns:
        # Format percentages
        if '%' in col and pd.api.types.is_numeric_dtype(formatted_df[col]):
            formatted_df[col] = (formatted_df[col] * 100).map('{:,.1f}%'.format).replace('nan%', '-')
        # Format days/lags/steps
        elif ('Lag' in col or 'TTC' in col or 'Steps' in col) and pd.api.types.is_numeric_dtype(formatted_df[col]):
             formatted_df[col] = formatted_df[col].map('{:,.1f}'.format).replace('nan', '-')
        # Format counts
        elif ('Count' in col or 'Qualified' in col) and pd.api.types.is_numeric_dtype(formatted_df[col]):
            formatted_df[col] = formatted_df[col].map('{:,.0f}'.format).replace('nan', '-')

    return formatted_df

def load_css(file_name):
    """A function to load a local CSS file into the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Please make sure it is in the project root directory.")