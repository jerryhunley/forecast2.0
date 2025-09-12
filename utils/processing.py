# utils/processing.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from utils.parsing import parse_datetime_with_timezone, parse_history_string

def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}
    status_to_stage_map = {}
    if not funnel_def: return pd.Series(timestamps)

    for stage, statuses in funnel_def.items():
        for status in statuses:
            status_to_stage_map[status] = stage

    all_events = []
    stage_hist = row.get(parsed_stage_history_col, [])
    status_hist = row.get(parsed_status_history_col, [])

    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist if isinstance(name, str)])
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist if isinstance(name, str)])

    try:
        all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError:
        pass

    for event_name, event_dt in all_events:
        if pd.isna(event_dt): continue
        event_stage = None
        if event_name in ordered_stgs:
            event_stage = event_name
        elif event_name in status_to_stage_map:
            event_stage = status_to_stage_map[event_name]

        if event_stage and event_stage in ordered_stgs:
            ts_col_name = ts_col_mapping.get(event_stage)
            if ts_col_name and pd.isna(timestamps[ts_col_name]):
                timestamps[ts_col_name] = event_dt

    return pd.Series(timestamps, dtype='datetime64[ns]')

@st.cache_data
def preprocess_referral_data(_df_raw, funnel_def, ordered_stages, ts_col_map):
    if _df_raw is None or funnel_def is None or ordered_stages is None or ts_col_map is None: return None

    df = _df_raw.copy()

    if "Submitted On" in df.columns:
        submitted_on_col = "Submitted On"
    elif "Referral Date" in df.columns:
        df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
        submitted_on_col = "Submitted On"
    else:
        st.error("Missing 'Submitted On'/'Referral Date'. Cannot proceed.")
        return None

    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    initial_rows = len(df)
    df.dropna(subset=["Submitted On_DT"], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        st.warning(f"Dropped {rows_dropped} rows due to unparseable 'Submitted On' date.")
    if df.empty:
        st.error("No valid data remaining after date parsing.")
        return None

    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')

    history_cols_to_parse = ['Lead Stage History', 'Lead Status History']
    parsed_cols = {}
    for col_name in history_cols_to_parse:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string)
            parsed_cols[col_name] = parsed_col_name
        else:
            st.warning(f"History column '{col_name}' not found. Timestamps might be incomplete.")

    parsed_stage_hist_col = parsed_cols.get('Lead Stage History')
    parsed_status_hist_col = parsed_cols.get('Lead Status History')

    if not parsed_stage_hist_col and not parsed_status_hist_col:
        st.error("Neither 'Lead Stage History' nor 'Lead Status History' column found. Cannot determine stage progression.")
        return None

    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_stage_hist_col, parsed_status_hist_col, funnel_def, ordered_stages, ts_col_map), axis=1)

    df = pd.concat([df, timestamp_cols_df], axis=1)
    for stage, ts_col in ts_col_map.items():
         if ts_col in df.columns:
             df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

    # Ensure UTM columns exist for Ad Performance Tab
    if "UTM Source" not in df.columns: df["UTM Source"] = np.nan
    if "UTM Medium" not in df.columns: df["UTM Medium"] = np.nan

    return df
