# utils/parsing.py
import streamlit as st
import pandas as pd
import io
import re
from datetime import datetime

# In parsing.py
@st.cache_data
def parse_funnel_definition(uploaded_file):
    """
    Parses the funnel definition from an uploaded file, handling both CSV and TSV.
    """
    if uploaded_file is None: return None, None, None
    try:
        uploaded_file.seek(0)
        # --- FIX: Decode with 'utf-8-sig' to automatically handle the BOM character ---
        bytes_data = uploaded_file.getvalue()
        stringio = io.StringIO(bytes_data.decode("utf-8-sig", errors='replace'))
        df_funnel_def = pd.read_csv(stringio, sep=None, engine='python', header=None)

        parsed_funnel_definition = {}; parsed_ordered_stages = []; ts_col_map = {}
        for col_idx in df_funnel_def.columns:
            column_data = df_funnel_def[col_idx]
            stage_name = column_data.iloc[0]
            if pd.isna(stage_name) or str(stage_name).strip() == "": continue

            stage_name = str(stage_name).strip().replace('"', '')
            parsed_ordered_stages.append(stage_name)

            statuses = column_data.iloc[1:].dropna().astype(str).apply(lambda x: x.strip().replace('"', '')).tolist()
            statuses = [s for s in statuses if s]

            if stage_name not in statuses:
                statuses.append(stage_name)

            parsed_funnel_definition[stage_name] = statuses
            clean_ts_name = f"TS_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            ts_col_map[stage_name] = clean_ts_name

        if not parsed_ordered_stages:
            st.error("Could not parse stages from Funnel Definition. Check file format.")
            return None, None, None

        return parsed_funnel_definition, parsed_ordered_stages, ts_col_map
    except Exception as e:
        st.error(f"Error parsing Funnel Definition file: {e}")
        st.info("Please ensure the file is a standard CSV or TSV with stage names in the first row.")
        return None, None, None

def parse_datetime_with_timezone(dt_str):
    if pd.isna(dt_str): return pd.NaT
    dt_str_cleaned = str(dt_str).strip()
    tz_pattern = r'\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$'
    dt_str_no_tz = re.sub(tz_pattern, '', dt_str_cleaned)
    parsed_dt = pd.to_datetime(dt_str_no_tz, errors='coerce')
    return parsed_dt

def parse_history_string(history_str):
    if pd.isna(history_str) or str(history_str).strip() == "": return []
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[apAP][mM])?(?:\s+[A-Za-z]{3,}(?:T)?)?)")
    raw_lines = str(history_str).strip().split('\n')
    parsed_events = []
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        match = pattern.match(line)
        if match:
            name, dt_str = match.groups()
            name = name.strip()
            dt_obj = parse_datetime_with_timezone(dt_str.strip())
            if name and pd.notna(dt_obj):
                try:
                    py_dt = dt_obj.to_pydatetime()
                    parsed_events.append((name, py_dt))
                except AttributeError:
                    pass
    try:
        # --- FIX #2: Sort by the datetime object (index 1), not the whole tuple ---
        parsed_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError:
        # This pass is intentional as some malformed data might still cause issues
        pass
    return parsed_events