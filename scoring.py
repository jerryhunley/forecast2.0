# scoring.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def score_performance_groups(_performance_metrics_df, weights, group_col_name): 
    if _performance_metrics_df is None or _performance_metrics_df.empty:
        return pd.DataFrame()
    
    df = _performance_metrics_df.copy()
    if group_col_name not in df.columns:
        st.error(f"Scoring Error: Grouping column '{group_col_name}' not found.")
        return df

    df_with_scores = df.copy()
    df_with_scores.drop_duplicates(subset=[group_col_name], keep='first', inplace=True)
    df_with_scores.set_index(group_col_name, inplace=True)
    
    metrics_to_scale = list(weights.keys())
    
    # --- FIX: Make keyword matching for inversion case-insensitive and more comprehensive ---
    lower_is_better_keywords = ['lag', 'ttc', 'fail', 'lost', 'time', 'awaiting']
    lower_is_better = [m for m in metrics_to_scale if any(keyword in m.lower() for keyword in lower_is_better_keywords)]
    
    scaled_data = df_with_scores.reindex(columns=metrics_to_scale).copy()

    for col in metrics_to_scale:
        if col in scaled_data.columns:
            if col in lower_is_better:
                fill_value = (scaled_data[col].max() * 1.2) if pd.notna(scaled_data[col].max()) else 999
                scaled_data[col].fillna(fill_value, inplace=True)
            else:
                scaled_data[col].fillna(0, inplace=True)
        else:
             scaled_data[col] = 0.5

    if not scaled_data.empty:
        scaler = MinMaxScaler()
        for col in scaled_data.columns:
            if scaled_data[col].min() == scaled_data[col].max():
                scaled_data[col] = 0.5
            else:
                scaled_data[col] = scaler.fit_transform(scaled_data[[col]].values.reshape(-1, 1))

    for col in lower_is_better:
        if col in scaled_data.columns:
            scaled_data[col] = 1 - scaled_data[col]
            
    df_with_scores['Score_Raw'] = 0.0
    total_weight = sum(abs(w) for w in weights.values())
    if total_weight > 0:
        for metric, weight in weights.items():
            if metric in scaled_data.columns:
                df_with_scores['Score_Raw'] += scaled_data[metric] * weight
        df_with_scores['Score'] = (df_with_scores['Score_Raw'] / total_weight) * 100
    else:
        df_with_scores['Score'] = 0.0

    df_with_scores['Score'].fillna(0.0, inplace=True)
    
    def assign_grade(score):
        if pd.isna(score): return 'N/A'
        if score >= 90: return 'A'
        if score >= 80: return 'B'
        if score >= 70: return 'C'
        if score >= 60: return 'D'
        return 'F'
        
    if len(df_with_scores) > 1:
        try:
            df_with_scores['Grade'] = pd.qcut(df_with_scores['Score'], q=[0, .2, .4, .6, .8, 1.], labels=['F', 'D', 'C', 'B', 'A'])
        except ValueError:
            df_with_scores['Grade'] = df_with_scores['Score'].apply(assign_grade)
    else:
        df_with_scores['Grade'] = df_with_scores['Score'].apply(assign_grade)

    df_with_scores['Grade'] = df_with_scores['Grade'].astype(str).replace('nan', 'N/A')
    
    return df_with_scores.reset_index().sort_values('Score', ascending=False)


def score_sites(_site_metrics_df, weights):
    return score_performance_groups(_site_metrics_df, weights, group_col_name="Site")