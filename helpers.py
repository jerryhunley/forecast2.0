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