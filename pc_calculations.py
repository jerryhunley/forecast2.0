# pc_calculations.py
import pandas as pd
import numpy as np
from collections import Counter

# Import the generic lag calculator for reuse
from calculations import calculate_avg_lag_generic

def calculate_heatmap_data(df, ts_col_map, status_history_col):
    """
    Prepares data for two heatmaps:
    1. Pre-Sent-to-Site Contact Attempts by day of week and hour.
    2. Sent to Site events by day of week and hour.
    """
    if df is None or df.empty or ts_col_map is None:
        return pd.DataFrame(), pd.DataFrame()
    contact_timestamps = []
    sts_ts_col = ts_col_map.get("Sent To Site")
    if status_history_col in df.columns and sts_ts_col in df.columns:
        for _, row in df.iterrows():
            sts_timestamp = row[sts_ts_col]
            history = row[status_history_col]
            if not isinstance(history, list): continue
            for event_name, event_dt in history:
                is_contact_attempt = "contact attempt" in event_name.lower()
                if is_contact_attempt and (pd.isna(sts_timestamp) or event_dt < sts_timestamp):
                    contact_timestamps.append(event_dt)
    sts_timestamps = df[sts_ts_col].dropna().tolist()
    def aggregate_timestamps(timestamps):
        if not timestamps:
            return pd.DataFrame(np.zeros((7, 24)), 
                                index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                                columns=range(24))
        ts_series = pd.Series(pd.to_datetime(timestamps))
        agg_df = pd.DataFrame({'day_of_week': ts_series.dt.dayofweek, 'hour': ts_series.dt.hour})
        heatmap_grid = pd.crosstab(agg_df['day_of_week'], agg_df['hour'])
        heatmap_grid = heatmap_grid.reindex(index=range(7), columns=range(24), fill_value=0)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_grid.index = heatmap_grid.index.map(lambda i: day_names[i])
        return heatmap_grid
    return aggregate_timestamps(contact_timestamps), aggregate_timestamps(sts_timestamps)

def calculate_average_time_metrics(df, ts_col_map, status_history_col):
    """
    Calculates key average time metrics for PC performance.
    """
    if df is None or df.empty or ts_col_map is None:
        return {"avg_time_to_first_contact": np.nan, "avg_time_between_contacts": np.nan, "avg_time_new_to_sts": np.nan}
    pof_ts_col = ts_col_map.get("Passed Online Form")
    psa_ts_col = ts_col_map.get("Pre-Screening Activities")
    sts_ts_col = ts_col_map.get("Sent To Site")
    avg_ttfc = calculate_avg_lag_generic(df, pof_ts_col, psa_ts_col)
    avg_new_to_sts = calculate_avg_lag_generic(df, pof_ts_col, sts_ts_col)
    all_contact_deltas = []
    if status_history_col in df.columns and sts_ts_col in df.columns:
        for _, row in df.iterrows():
            sts_timestamp = row[sts_ts_col]
            history = row[status_history_col]
            if not isinstance(history, list): continue
            attempt_timestamps = sorted([
                event_dt for event_name, event_dt in history 
                if "contact attempt" in event_name.lower() and (pd.isna(sts_timestamp) or event_dt < sts_timestamp)
            ])
            if len(attempt_timestamps) > 1:
                all_contact_deltas.extend(np.diff(attempt_timestamps))
    avg_between_contacts = pd.Series(all_contact_deltas).mean().total_seconds() / (60 * 60 * 24) if all_contact_deltas else np.nan
    return {"avg_time_to_first_contact": avg_ttfc, "avg_time_between_contacts": avg_between_contacts, "avg_time_new_to_sts": avg_new_to_sts}

def calculate_top_status_flows(df, ts_col_map, status_history_col, min_data_threshold=5):
    """
    Identifies the top 5 most common status flow paths leading to Sent To Site.
    """
    if df is None or df.empty or ts_col_map is None: return []
    sts_ts_col = ts_col_map.get("Sent To Site")
    if sts_ts_col not in df.columns or status_history_col not in df.columns: return []
    successful_referrals = df.dropna(subset=[sts_ts_col]).copy()
    if len(successful_referrals) < min_data_threshold: return []
    all_paths = []
    for _, row in successful_referrals.iterrows():
        sts_timestamp = row[sts_ts_col]
        history = row[status_history_col]
        if not isinstance(history, list): continue
        pre_sts_path = [event_name for event_name, event_dt in history if event_dt <= sts_timestamp]
        if pre_sts_path:
            all_paths.append(" -> ".join(pre_sts_path))
    if not all_paths: return []
    return Counter(all_paths).most_common(5)

def calculate_ttfc_effectiveness(df, ts_col_map):
    """
    Analyzes how the time to first contact impacts downstream conversion rates.
    """
    if df is None or df.empty or not ts_col_map: return pd.DataFrame()
    pof_ts_col = ts_col_map.get("Passed Online Form")
    if pof_ts_col not in df.columns: return pd.DataFrame()
    other_ts_cols = [v for k, v in ts_col_map.items() if k != "Passed Online Form" and v in df.columns]
    if not other_ts_cols: return pd.DataFrame()
    analysis_df = df.copy()
    start_ts = analysis_df[pof_ts_col]
    def find_first_action(row):
        future_events = row[row > start_ts.loc[row.name]]
        return future_events.min() if not future_events.empty else pd.NaT
    first_action_ts = analysis_df[other_ts_cols].apply(find_first_action, axis=1)
    analysis_df['ttfc_minutes'] = (first_action_ts - start_ts).dt.total_seconds() / 60
    bin_edges = [-np.inf, 5, 15, 30, 60, 3*60, 6*60, 12*60, 24*60, np.inf]
    bin_labels = ['<= 5 min', '5-15 min', '15-30 min', '30-60 min', '1-3 hours', '3-6 hours', '6-12 hours', '12-24 hours', '> 24 hours']
    analysis_df['ttfc_bin'] = pd.cut(analysis_df['ttfc_minutes'], bins=bin_edges, labels=bin_labels, right=True)
    sts_col = ts_col_map.get("Sent To Site")
    icf_col = ts_col_map.get("Signed ICF")
    enr_col = ts_col_map.get("Enrolled")
    result = analysis_df.groupby('ttfc_bin').agg(
        Attempts=('ttfc_bin', 'size'),
        Total_StS=(sts_col, lambda x: x.notna().sum()),
        Total_ICF=(icf_col, lambda x: x.notna().sum()),
        Total_Enrolled=(enr_col, lambda x: x.notna().sum())
    )
    result = result.reindex(bin_labels, fill_value=0).astype(int)
    result['StS_Rate'] = (result['Total_StS'] / result['Attempts'].replace(0, np.nan))
    result['ICF_Rate'] = (result['Total_ICF'] / result['Attempts'].replace(0, np.nan))
    result['Enrollment_Rate'] = (result['Total_Enrolled'] / result['Attempts'].replace(0, np.nan))
    result.reset_index(inplace=True)
    result.rename(columns={'ttfc_bin': 'Time to First Contact'}, inplace=True)
    return result

def calculate_contact_attempt_effectiveness(df, ts_col_map, status_history_col):
    """
    Analyzes how the number of pre-StS status changes impacts downstream conversions.
    """
    if df is None or df.empty or ts_col_map is None:
        return pd.DataFrame()

    sts_ts_col = ts_col_map.get("Sent To Site")
    if sts_ts_col not in df.columns or status_history_col not in df.columns:
        return pd.DataFrame()

    analysis_df = df.copy()

    def count_pre_sts_attempts(row):
        sts_timestamp = row[sts_ts_col]
        history = row[status_history_col]

        if not isinstance(history, list) or not history:
            return 0
        
        pre_sts_path = [
            event for event in history 
            if pd.isna(sts_timestamp) or event[1] < sts_timestamp
        ]
        return max(0, len(pre_sts_path) - 1)

    analysis_df['pre_sts_attempt_count'] = analysis_df.apply(count_pre_sts_attempts, axis=1)

    icf_col = ts_col_map.get("Signed ICF")
    enr_col = ts_col_map.get("Enrolled")

    result = analysis_df.groupby('pre_sts_attempt_count').agg(
        Referral_Count=('pre_sts_attempt_count', 'size'),
        Total_StS=(sts_ts_col, lambda x: x.notna().sum()),
        Total_ICF=(icf_col, lambda x: x.notna().sum()),
        Total_Enrolled=(enr_col, lambda x: x.notna().sum())
    )

    result['StS_Rate'] = (result['Total_StS'] / result['Referral_Count'].replace(0, np.nan))
    result['ICF_Rate'] = (result['Total_ICF'] / result['Referral_Count'].replace(0, np.nan))
    result['Enrollment_Rate'] = (result['Total_Enrolled'] / result['Referral_Count'].replace(0, np.nan))
    
    result.reset_index(inplace=True)
    
    result.rename(columns={
        'pre_sts_attempt_count': 'Number of Attempts',
        'Referral_Count': 'Total Referrals'
    }, inplace=True)
    
    return result

def calculate_performance_over_time(df, ts_col_map):
    """
    Calculates key PC performance metrics over time on a weekly basis,
    with transit-time adjustment for Sent to Site %.
    """
    if df is None or df.empty or 'Submitted On_DT' not in df.columns:
        return pd.DataFrame()

    pof_ts_col = ts_col_map.get("Passed Online Form")
    psa_ts_col = ts_col_map.get("Pre-Screening Activities")
    sts_ts_col = ts_col_map.get("Sent To Site")

    if not all(col in df.columns for col in [pof_ts_col, psa_ts_col, sts_ts_col]):
        return pd.DataFrame()

    avg_pof_to_sts_lag = calculate_avg_lag_generic(df, pof_ts_col, sts_ts_col)
    maturity_days = (avg_pof_to_sts_lag * 1.5) if pd.notna(avg_pof_to_sts_lag) else 30

    time_df = df.set_index('Submitted On_DT')

    weekly_summary = time_df.resample('W').apply(lambda week_df: pd.Series({
        'Total Qualified per Week': (
            len(week_df)
        ),
        'Sent to Site % (Transit-Time Adjusted)': (
            week_df[week_df.index + pd.Timedelta(days=maturity_days) < pd.Timestamp.now()]
            .pipe(lambda mature_df: mature_df[sts_ts_col].notna().sum() / len(mature_df) if len(mature_df) > 0 else 0)
        ),
        'Average Time to First Contact (Days)': calculate_avg_lag_generic(
            week_df, pof_ts_col, psa_ts_col
        ),
        'Average Sent to Site per Day': (
            week_df[sts_ts_col].notna().sum() / 7
        ),
        'Total Sent to Site per Week': (
            week_df[sts_ts_col].notna().sum()
        )
    }))

    weekly_summary['Sent to Site % (Transit-Time Adjusted)'] *= 100
    weekly_summary.fillna(method='ffill', inplace=True)

    return weekly_summary

# --- NEW FUNCTION ---
def analyze_heatmap_efficiency(contact_heatmap, sts_heatmap):
    """
    Analyzes the two heatmaps to find the best and worst times for contact attempts.
    """
    if contact_heatmap.empty or sts_heatmap.empty or contact_heatmap.sum().sum() == 0:
        return {}

    # Reshape data for analysis
    contacts_long = contact_heatmap.stack().reset_index()
    contacts_long.columns = ['Day', 'Hour', 'Contacts']
    sts_long = sts_heatmap.stack().reset_index()
    sts_long.columns = ['Day', 'Hour', 'StS']
    
    # Merge the two data sources
    merged_df = pd.merge(contacts_long, sts_long, on=['Day', 'Hour'])
    
    # Calculate efficiency (StS per Contact Attempt)
    merged_df['Efficiency'] = (merged_df['StS'] / merged_df['Contacts']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Define thresholds using quantiles for robustness
    high_contacts_threshold = merged_df['Contacts'].quantile(0.90)
    high_sts_threshold = merged_df['StS'].quantile(0.90)
    
    # Ensure there's variance in efficiency before calculating quantile
    if merged_df[merged_df['Contacts'] > 0]['Efficiency'].nunique() > 1:
        high_efficiency_threshold = merged_df[merged_df['Contacts'] > 0]['Efficiency'].quantile(0.90)
        low_efficiency_threshold = merged_df[merged_df['Contacts'] > high_contacts_threshold]['Efficiency'].quantile(0.25)
    else:
        high_efficiency_threshold = 0
        low_efficiency_threshold = 0

    # Filter for each category
    volume_best_df = merged_df[
        (merged_df['Contacts'] >= high_contacts_threshold) & 
        (merged_df['StS'] >= high_sts_threshold) &
        (merged_df['Contacts'] > 1) # Ensure some minimum activity
    ]
    
    most_efficient_df = merged_df[
        (merged_df['Efficiency'] >= high_efficiency_threshold) & 
        (merged_df['StS'] >= 1) & # Must have at least one success
        (merged_df['Efficiency'] > 0)
    ].sort_values(by='Efficiency', ascending=False)
    
    least_efficient_df = merged_df[
        (merged_df['Contacts'] >= high_contacts_threshold) & 
        (merged_df['Efficiency'] <= low_efficiency_threshold) &
        (merged_df['Efficiency'] < high_efficiency_threshold) # Must be less than the best
    ].sort_values(by='Efficiency', ascending=True)

    # Helper to format the output strings
    def format_hour(hour):
        if hour == 0: return "12 AM"
        if hour == 12: return "12 PM"
        if hour < 12: return f"{hour} AM"
        return f"{hour-12} PM"

    # Extract top N time slots for each category
    results = {
        "volume_best": [f"{row.Day}, {format_hour(row.Hour)}" for _, row in volume_best_df.head(5).iterrows()],
        "most_efficient": [f"{row.Day}, {format_hour(row.Hour)}" for _, row in most_efficient_df.head(5).iterrows()],
        "least_efficient": [f"{row.Day}, {format_hour(row.Hour)}" for _, row in least_efficient_df.head(5).iterrows()],
    }
    
    return results