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
        if '%' in col and pd.api.types.is_numeric_dtype(formatted_df[col]):
            formatted_df[col] = (formatted_df[col] * 100).map('{:,.1f}%'.format).replace('nan%', '-')
        elif ('Lag' in col or 'TTC' in col or 'Steps' in col) and pd.api.types.is_numeric_dtype(formatted_df[col]):
             formatted_df[col] = formatted_df[col].map('{:,.1f}'.format).replace('nan', '-')
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

def format_days_to_dhm(days_float):
    """Converts a float number of days into a 'd h m' string format."""
    if pd.isna(days_float) or days_float < 0:
        return "N/A"
    
    # Convert float days to total seconds
    total_seconds = days_float * 24 * 60 * 60
    
    # Calculate days
    days = int(total_seconds // (24 * 3600))
    total_seconds %= (24 * 3600)
    
    # Calculate hours
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    
    # Calculate minutes
    minutes = int(total_seconds // 60)
    
    return f"{days} d {hours} h {minutes} m"