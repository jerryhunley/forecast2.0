# utils/scoring.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def score_performance_groups(_performance_metrics_df, weights, group_col_name):
    if _performance_metrics_df is None or _performance_metrics_df.empty: return pd.DataFrame()

    df = _performance_metrics_df.copy()
    if group_col_name not in df.columns:
        st.error(f"Scoring Error: Grouping column '{group_col_name}' not found.")
        return df

    # Ensure no duplicate groups, which can happen with data issues
    df.drop_duplicates(subset=[group_col_name], keep='first', inplace=True)
    df.set_index(group_col_name, inplace=True)

    metrics_to_scale = list(weights.keys())
    lower_is_better = [m for m in metrics_to_scale if 'Lag' in m or 'TTC' in m or 'Fail' in m]

    scaled_data = df.reindex(columns=metrics_to_scale).copy()

    # Handle NaNs before scaling
    for col in metrics_to_scale:
        if col in scaled_data.columns:
            if col in lower_is_better:
                # For lower-is-better, fill NaNs with a "bad" value
                fill_value = (scaled_data[col].max() * 1.2) if pd.notna(scaled_data[col].max()) else 999
                scaled_data[col].fillna(fill_value, inplace=True)
            else:
                # For higher-is-better, fill NaNs with 0
                scaled_data[col].fillna(0, inplace=True)
        else:
             scaled_data[col] = 0.5 # If metric doesn't exist, give neutral score

    # Scale metrics from 0 to 1
    if not scaled_data.empty:
        scaler = MinMaxScaler()
        # Ensure we only scale if there's variance
        for col in scaled_data.columns:
            if scaled_data[col].min() == scaled_data[col].max():
                scaled_data[col] = 0.5
            else:
                scaled_data[[col]] = scaler.fit_transform(scaled_data[[col]])

    # Invert scores for "lower is better" metrics
    for col in lower_is_better:
        if col in scaled_data.columns:
            scaled_data[col] = 1 - scaled_data[col]

    # Calculate weighted score
    df['Score_Raw'] = 0.0
    total_weight = sum(abs(w) for w in weights.values())
    if total_weight > 0:
        for metric, weight in weights.items():
            if metric in scaled_data.columns:
                df['Score_Raw'] += scaled_data[metric] * weight
        df['Score'] = (df['Score_Raw'] / total_weight) * 100
    else:
        df['Score'] = 0.0


    df['Score'].fillna(0.0, inplace=True)

    # Assign grades
    def assign_grade(score):
        if pd.isna(score): return 'N/A'
        if score >= 90: return 'A'
        if score >= 80: return 'B'
        if score >= 70: return 'C'
        if score >= 60: return 'D'
        return 'F'
        
    if len(df) > 1:
        try:
            df['Grade'] = pd.qcut(df['Score'], q=[0, .2, .4, .6, .8, 1.], labels=['F', 'D', 'C', 'B', 'A'])
        except ValueError: # Handle cases with not enough unique values for qcut
            df['Grade'] = df['Score'].apply(assign_grade)
    else:
        df['Grade'] = df['Score'].apply(assign_grade)

    df['Grade'] = df['Grade'].astype(str).replace('nan', 'N/A')

    return df.reset_index().sort_values('Score', ascending=False)


def score_sites(_site_metrics_df, weights):
    return score_performance_groups(_site_metrics_df, weights, group_col_name="Site")