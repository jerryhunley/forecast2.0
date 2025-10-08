# calculations.py
import streamlit as st
import pandas as pd
import numpy as np

from constants import * # Import all stage names

def calculate_avg_lag_generic(df, col_from, col_to):
    if col_from is None or col_to is None or col_from not in df.columns or col_to not in df.columns:
        return np.nan

    if not all([pd.api.types.is_datetime64_any_dtype(df[col_from]),
                pd.api.types.is_datetime64_any_dtype(df[col_to])]):
        return np.nan

    valid_df = df.dropna(subset=[col_from, col_to])
    if valid_df.empty: return np.nan

    diff = pd.to_datetime(valid_df[col_to]) - pd.to_datetime(valid_df[col_from])
    diff_positive = diff[diff >= pd.Timedelta(days=0)]

    return diff_positive.mean().total_seconds() / (60 * 60 * 24) if not diff_positive.empty else np.nan


def calculate_overall_inter_stage_lags(_processed_df, ordered_stages, ts_col_map):
    if _processed_df is None or _processed_df.empty or not ordered_stages or not ts_col_map:
        return {}
    inter_stage_lags = {}
    for i in range(len(ordered_stages) - 1):
        stage_from, stage_to = ordered_stages[i], ordered_stages[i+1]
        ts_col_from, ts_col_to = ts_col_map.get(stage_from), ts_col_map.get(stage_to)
        
        avg_lag = calculate_avg_lag_generic(_processed_df, ts_col_from, ts_col_to)
        inter_stage_lags[f"{stage_from} -> {stage_to}"] = avg_lag
        
    return inter_stage_lags


def calculate_proforma_metrics(_processed_df, ordered_stages, ts_col_map, monthly_ad_spend_input):
    if _processed_df is None or _processed_df.empty: return pd.DataFrame()
    
    processed_df = _processed_df.copy()
    
    cohort_summary = processed_df.groupby("Submission_Month").size().reset_index(name="Total Qualified Referrals_Calc")
    cohort_summary = cohort_summary.set_index("Submission_Month")
    cohort_summary["Ad Spend"] = cohort_summary.index.map(monthly_ad_spend_input).fillna(0)
    
    reached_stage_cols_map = {}
    for stage_name in ordered_stages:
        ts_col = ts_col_map.get(stage_name)
        if ts_col and ts_col in processed_df.columns:
            reached_col = f"Reached_{stage_name.replace(' ', '_')}"
            reached_stage_cols_map[stage_name] = reached_col
            reached_stage_count = processed_df.dropna(subset=[ts_col]).groupby("Submission_Month").size()
            cohort_summary[reached_col] = reached_stage_count
    
    cohort_summary = cohort_summary.fillna(0)
    for col in cohort_summary.columns:
        if col != "Ad Spend": cohort_summary[col] = cohort_summary[col].astype(int)
    cohort_summary["Ad Spend"] = cohort_summary["Ad Spend"].astype(float)
    
    base_count_col = reached_stage_cols_map.get(STAGE_PASSED_ONLINE_FORM, "Total Qualified Referrals_Calc")
    if base_count_col in cohort_summary.columns:
        cohort_summary.rename(columns={base_count_col: "Pre-Screener Qualified"}, inplace=True)
    base_count_col_name = "Pre-Screener Qualified"
    
    proforma_metrics = pd.DataFrame(index=cohort_summary.index)
    if base_count_col_name in cohort_summary.columns:
        proforma_metrics["Ad Spend"] = cohort_summary["Ad Spend"]
        proforma_metrics["Pre-Screener Qualified"] = cohort_summary[base_count_col_name]
        proforma_metrics["Cost per Qualified Pre-screen"] = (cohort_summary["Ad Spend"] / cohort_summary[base_count_col_name].replace(0, np.nan)).round(2)

        for stage, reached_col in reached_stage_cols_map.items():
            metric_name = f"Total {stage}" if stage != STAGE_PASSED_ONLINE_FORM else "Pre-Screener Qualified"
            if reached_col in cohort_summary.columns:
                proforma_metrics[metric_name] = cohort_summary[reached_col]
        
        sts_col = reached_stage_cols_map.get(STAGE_SENT_TO_SITE)
        appt_col = reached_stage_cols_map.get(STAGE_APPOINTMENT_SCHEDULED)
        icf_col = reached_stage_cols_map.get(STAGE_SIGNED_ICF)

        if sts_col in cohort_summary: proforma_metrics["Qualified to StS %"] = (cohort_summary[sts_col] / cohort_summary[base_count_col_name].replace(0, np.nan))
        if sts_col in cohort_summary and appt_col in cohort_summary: proforma_metrics["StS to Appt Sched %"] = (cohort_summary[appt_col] / cohort_summary[sts_col].replace(0, np.nan))
        if appt_col in cohort_summary and icf_col in cohort_summary: proforma_metrics["Appt Sched to ICF %"] = (cohort_summary[icf_col] / cohort_summary[appt_col].replace(0, np.nan))
        if icf_col in cohort_summary:
            proforma_metrics["Qualified to ICF %"] = (cohort_summary[icf_col] / cohort_summary[base_count_col_name].replace(0, np.nan))
            proforma_metrics["Cost Per ICF"] = (cohort_summary["Ad Spend"] / cohort_summary[icf_col].replace(0, np.nan)).round(2)

    return proforma_metrics


def calculate_site_operational_kpis(df, ts_col_map, status_history_col, selected_site="Overall"):
    if df is None or df.empty:
        return {'avg_sts_to_first_action': np.nan, 'avg_time_between_site_contacts': np.nan, 'avg_sts_to_appt': np.nan}

    if selected_site != "Overall":
        if 'Site' not in df.columns:
            return {'avg_sts_to_first_action': np.nan, 'avg_time_between_site_contacts': np.nan, 'avg_sts_to_appt': np.nan}
        site_df = df[df['Site'] == selected_site].copy()
    else:
        site_df = df.copy()

    if site_df.empty:
        return {'avg_sts_to_first_action': np.nan, 'avg_time_between_site_contacts': np.nan, 'avg_sts_to_appt': np.nan}

    sts_ts_col = ts_col_map.get("Sent To Site")
    appt_ts_col = ts_col_map.get("Appointment Scheduled")

    all_ts_cols_after_sts = [v for k, v in ts_col_map.items() if k != "Sent To Site" and v in site_df.columns]
    
    def find_first_action_after_sts(row):
        sts_time = row[sts_ts_col]
        if pd.isna(sts_time):
            return pd.NaT
        future_events = row[all_ts_cols_after_sts][row[all_ts_cols_after_sts] > sts_time]
        return future_events.min() if not future_events.empty else pd.NaT

    first_actions = site_df.apply(find_first_action_after_sts, axis=1)
    time_to_first_action = (first_actions - site_df[sts_ts_col]).dt.total_seconds() / (60*60*24)
    avg_sts_to_first_action = time_to_first_action.mean()

    all_contact_deltas = []
    if status_history_col in site_df.columns:
        for _, row in site_df.iterrows():
            sts_time = row[sts_ts_col]
            appt_time = row[appt_ts_col]
            history = row[status_history_col]

            if pd.isna(sts_time) or not isinstance(history, list):
                continue

            start_window = sts_time
            end_window = appt_time if pd.notna(appt_time) else pd.Timestamp.max

            site_attempt_timestamps = sorted([
                event_dt for event_name, event_dt in history 
                if "contact attempt" in event_name.lower() and event_dt > start_window and event_dt < end_window
            ])
            
            if len(site_attempt_timestamps) > 1:
                all_contact_deltas.extend(np.diff(site_attempt_timestamps))

    avg_between_site_contacts = pd.Series(all_contact_deltas).mean().total_seconds() / (60 * 60 * 24) if all_contact_deltas else np.nan
    avg_sts_to_appt = calculate_avg_lag_generic(site_df, sts_ts_col, appt_ts_col)

    return {
        'avg_sts_to_first_action': avg_sts_to_first_action,
        'avg_time_between_site_contacts': avg_between_site_contacts,
        'avg_sts_to_appt': avg_sts_to_appt
    }

def calculate_site_ttfc_effectiveness(df, ts_col_map, selected_site="Overall"):
    if df is None or df.empty:
        return pd.DataFrame()

    if selected_site != "Overall":
        if 'Site' not in df.columns: return pd.DataFrame()
        site_df = df[df['Site'] == selected_site].copy()
    else:
        site_df = df.copy()

    sts_ts_col = ts_col_map.get("Sent To Site")
    site_df = site_df.dropna(subset=[sts_ts_col]).copy()

    if site_df.empty:
        return pd.DataFrame()

    all_ts_cols_after_sts = [
        v for k, v in ts_col_map.items() 
        if k not in ["Passed Online Form", "Pre-Screening Activities", "Sent To Site"] and v in site_df.columns
    ]
    
    def find_first_action_after_sts(row):
        sts_time = row[sts_ts_col]
        future_events = row[all_ts_cols_after_sts][row[all_ts_cols_after_sts] > sts_time]
        return future_events.min() if not future_events.empty else pd.NaT

    first_actions = site_df.apply(find_first_action_after_sts, axis=1)
    
    site_df['ttfc_hours'] = (first_actions - site_df[sts_ts_col]).dt.total_seconds() / 3600

    bin_edges = [-np.inf, 4, 8, 24, 48, 72, 120, 168, 336, np.inf]
    bin_labels = [
        '< 4 Hours', '4 - 8 Hours', '8 - 24 Hours', '1 - 2 Days', '2 - 3 Days',
        '3 - 5 Days', '5 - 7 Days', '7 - 14 Days', '> 14 Days'
    ]
    
    site_df['ttfc_bin'] = pd.cut(
        site_df['ttfc_hours'], 
        bins=bin_edges, 
        labels=bin_labels, 
        right=True
    )

    appt_col = ts_col_map.get("Appointment Scheduled")
    icf_col = ts_col_map.get("Signed ICF")
    enr_col = ts_col_map.get("Enrolled")

    result = site_df.groupby('ttfc_bin').agg(
        Attempts=('ttfc_bin', 'size'),
        Total_Appts=(appt_col, lambda x: x.notna().sum()),
        Total_ICF=(icf_col, lambda x: x.notna().sum()),
        Total_Enrolled=(enr_col, lambda x: x.notna().sum())
    )
    
    result = result.reindex(bin_labels, fill_value=0)

    result['Appt_Rate'] = (result['Total_Appts'] / result['Attempts'].replace(0, np.nan))
    result['ICF_Rate'] = (result['Total_ICF'] / result['Attempts'].replace(0, np.nan))
    result['Enrollment_Rate'] = (result['Total_Enrolled'] / result['Attempts'].replace(0, np.nan))
    
    result.reset_index(inplace=True)
    result.rename(columns={'ttfc_bin': 'Time to First Site Action'}, inplace=True)
    
    return result

def calculate_site_contact_attempt_effectiveness(df, ts_col_map, status_history_col, selected_site="Overall"):
    if df is None or df.empty or not ts_col_map or status_history_col not in df.columns:
        return pd.DataFrame()

    if selected_site != "Overall":
        if 'Site' not in df.columns: return pd.DataFrame()
        analysis_df = df[df['Site'] == selected_site].copy()
    else:
        analysis_df = df.copy()

    sts_ts_col = ts_col_map.get("Sent To Site")
    appt_ts_col = ts_col_map.get("Appointment Scheduled")
    lost_ts_col = ts_col_map.get("Lost")
    icf_ts_col = ts_col_map.get("Signed ICF")
    enr_ts_col = ts_col_map.get("Enrolled")

    for col in [sts_ts_col, appt_ts_col, lost_ts_col, icf_ts_col, enr_ts_col]:
        if col and col not in analysis_df.columns:
            analysis_df[col] = pd.NaT

    analysis_df = analysis_df.dropna(subset=[sts_ts_col]).copy()
    if analysis_df.empty:
        return pd.DataFrame()

    def count_logged_site_attempts(row):
        sts_time = row[sts_ts_col]
        appt_time = row.get(appt_ts_col)
        lost_time = row.get(lost_ts_col)
        history = row[status_history_col]

        if not isinstance(history, list):
            return 0

        end_window = pd.Timestamp.max
        if pd.notna(appt_time) and pd.notna(lost_time):
            end_window = min(appt_time, lost_time)
        elif pd.notna(appt_time):
            end_window = appt_time
        elif pd.notna(lost_time):
            end_window = lost_time

        attempt_count = 0
        for event_name, event_dt in history:
            if "contact attempt" in event_name.lower():
                if sts_time < event_dt < end_window:
                    attempt_count += 1
        return attempt_count

    analysis_df['site_attempt_count'] = analysis_df.apply(count_logged_site_attempts, axis=1)
    
    if appt_ts_col:
        mask = (analysis_df[appt_ts_col].notna()) & (analysis_df['site_attempt_count'] == 0)
        analysis_df.loc[mask, 'site_attempt_count'] = 1

    result = analysis_df.groupby('site_attempt_count').agg(
        Referral_Count=('site_attempt_count', 'size'),
        Total_Appts=(appt_ts_col, lambda x: x.notna().sum()),
        Total_ICF=(icf_ts_col, lambda x: x.notna().sum()),
        Total_Enrolled=(enr_ts_col, lambda x: x.notna().sum())
    )

    result['Appt_Rate'] = (result['Total_Appts'] / result['Referral_Count'].replace(0, np.nan))
    result['ICF_Rate'] = (result['Total_ICF'] / result['Referral_Count'].replace(0, np.nan))
    result['Enrollment_Rate'] = (result['Total_Enrolled'] / result['Referral_Count'].replace(0, np.nan))
    
    result.reset_index(inplace=True)
    
    result.rename(columns={
        'site_attempt_count': 'Number of Site Attempts',
        'Referral_Count': 'Total Referrals'
    }, inplace=True)
    
    return result

def calculate_site_performance_over_time(df, ts_col_map, status_history_col, selected_site="Overall"):
    if df is None or df.empty:
        return pd.DataFrame()

    sts_ts_col = ts_col_map.get(STAGE_SENT_TO_SITE)
    appt_ts_col = ts_col_map.get(STAGE_APPOINTMENT_SCHEDULED)
    icf_ts_col = ts_col_map.get(STAGE_SIGNED_ICF)
    enr_ts_col = ts_col_map.get(STAGE_ENROLLED)

    if not all(col in df.columns for col in [sts_ts_col, appt_ts_col, icf_ts_col, enr_ts_col]):
        return pd.DataFrame()

    if selected_site != "Overall":
        if 'Site' not in df.columns: return pd.DataFrame()
        site_df = df[df['Site'] == selected_site].copy()
    else:
        site_df = df.copy()

    site_df.dropna(subset=[sts_ts_col], inplace=True)
    if site_df.empty:
        return pd.DataFrame()

    avg_sts_to_icf_lag = calculate_avg_lag_generic(site_df, sts_ts_col, icf_ts_col)
    maturity_days = (avg_sts_to_icf_lag * 1.5) if pd.notna(avg_sts_to_icf_lag) else 45

    time_df = site_df.set_index(sts_ts_col)

    def get_weekly_metrics(week_df):
        is_mature = (week_df.index.max() + pd.Timedelta(days=maturity_days)) < pd.Timestamp.now()
        kpis = calculate_site_operational_kpis(week_df.reset_index(), ts_col_map, status_history_col, "Overall")
        
        rate_appt, rate_icf, rate_enr = np.nan, np.nan, np.nan
        if is_mature and len(week_df) > 0:
            rate_appt = week_df[appt_ts_col].notna().sum() / len(week_df)
            rate_icf = week_df[icf_ts_col].notna().sum() / len(week_df)
            rate_enr = week_df[enr_ts_col].notna().sum() / len(week_df)

        return pd.Series({
            'Total Sent to Site per Week': len(week_df),
            'Sent to Site -> Appointment %': rate_appt,
            'Sent to Site -> ICF %': rate_icf,
            'Sent to Site -> Enrollment %': rate_enr,
            'Total Appointments per Week': week_df[appt_ts_col].notna().sum(),
            'Average Time to First Site Action (Days)': kpis['avg_sts_to_first_action']
        })

    weekly_summary = time_df.resample('W').apply(get_weekly_metrics)

    for rate_col in ['Sent to Site -> Appointment %', 'Sent to Site -> ICF %', 'Sent to Site -> Enrollment %']:
        actual_col_name = f"{rate_col} (Actual)"
        projected_col_name = f"{rate_col} (Projected)"
        
        weekly_summary[actual_col_name] = weekly_summary[rate_col] * 100
        
        rolling_avg = weekly_summary[actual_col_name].rolling(window=4, min_periods=1).mean()
        
        last_valid_index = weekly_summary[actual_col_name].last_valid_index()
        if last_valid_index is not None:
            last_value = rolling_avg.loc[last_valid_index]
            
            projected_series = pd.Series(np.nan, index=weekly_summary.index)
            projected_series.loc[last_valid_index:] = last_value
            projected_series.ffill(inplace=True)
            projected_series.loc[last_valid_index] = np.nan
            
            weekly_summary[projected_col_name] = projected_series
        else:
            weekly_summary[projected_col_name] = np.nan

    return weekly_summary

def calculate_enhanced_site_metrics(_processed_df, ordered_stages, ts_col_map, status_history_col):
    if _processed_df is None or 'Site' not in _processed_df.columns:
        return pd.DataFrame()
        
    df = _processed_df.copy()
    df['Site'] = df['Site'].astype(str).str.strip().replace('', 'Unassigned Site').fillna('Unassigned Site')

    pof_ts_col = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
    psa_ts_col = ts_col_map.get(STAGE_PRE_SCREENING_ACTIVITIES)
    sts_ts_col = ts_col_map.get(STAGE_SENT_TO_SITE)
    appt_ts_col = ts_col_map.get(STAGE_APPOINTMENT_SCHEDULED)
    icf_ts_col = ts_col_map.get(STAGE_SIGNED_ICF)
    enr_ts_col = ts_col_map.get(STAGE_ENROLLED)
    lost_ts_col = ts_col_map.get(STAGE_LOST)
    sf_ts_col = ts_col_map.get(STAGE_SCREEN_FAILED)

    all_ts_cols_after_sts = [ts_col_map.get(s) for s in ordered_stages if ts_col_map.get(s) and ts_col_map.get(s) not in [pof_ts_col, psa_ts_col, sts_ts_col]]
    all_ts_cols_after_sts = [c for c in all_ts_cols_after_sts if c in df.columns]

    metrics_list = []
    for site_name, group_df in df.groupby('Site'):
        metrics = {'Site': site_name}
        
        pof_count = group_df[pof_ts_col].notna().sum()
        psa_count = group_df[psa_ts_col].notna().sum()
        sts_count = group_df[sts_ts_col].notna().sum()
        appt_count = group_df[appt_ts_col].notna().sum()
        icf_count = group_df[icf_ts_col].notna().sum()
        enr_count = group_df[enr_ts_col].notna().sum()
        lost_count = group_df[lost_ts_col].notna().sum()
        
        sf_or_lost_after_icf_mask = (group_df[icf_ts_col].notna()) & (
            ((group_df[sf_ts_col].notna()) & (group_df[sf_ts_col] > group_df[icf_ts_col])) |
            ((group_df[lost_ts_col].notna()) & (group_df[lost_ts_col] > group_df[icf_ts_col]))
        )
        sf_or_lost_after_icf_count = len(group_df[sf_or_lost_after_icf_mask])
        
        lost_after_sts_count = len(group_df[(group_df[sts_ts_col].notna()) & (group_df[lost_ts_col] > group_df[sts_ts_col])])

        metrics['Total Qualified'] = pof_count
        metrics['Pre-Screening Activities Count'] = psa_count
        metrics['StS Count'] = sts_count
        metrics['Appt Count'] = appt_count
        metrics['ICF Count'] = icf_count
        metrics['Enrollment Count'] = enr_count
        metrics['SF or Lost After ICF Count'] = sf_or_lost_after_icf_count
        metrics['Lost After StS'] = lost_after_sts_count
        metrics['Total Lost Count'] = lost_count

        if sts_count > 0:
            sts_df = group_df.dropna(subset=[sts_ts_col]).copy()
            
            def has_post_sts_action(row):
                sts_time = row[sts_ts_col]
                for ts_col in all_ts_cols_after_sts:
                    if pd.notna(row[ts_col]) and row[ts_col] > sts_time:
                        return True
                history = row.get(status_history_col, [])
                if isinstance(history, list):
                    for _, event_dt in history:
                        if pd.notna(event_dt) and event_dt > sts_time:
                            return True
                return False

            sts_df['has_action'] = sts_df.apply(has_post_sts_action, axis=1)
            awaiting_action_count = len(sts_df[~sts_df['has_action']])
            metrics['Total Referrals Awaiting First Site Action'] = awaiting_action_count
            
            ops_kpis = calculate_site_operational_kpis(group_df, ts_col_map, status_history_col, site_name)
            metrics['Avg. Time Between Site Contacts'] = ops_kpis.get('avg_time_between_site_contacts')
            metrics['Average time to first site action'] = ops_kpis.get('avg_sts_to_first_action')
            
            contact_attempts_df = calculate_site_contact_attempt_effectiveness(group_df, ts_col_map, status_history_col, site_name)
            total_attempts = (contact_attempts_df['Number of Site Attempts'] * contact_attempts_df['Total Referrals']).sum()
            metrics['Avg number of site contact attempts per referral'] = total_attempts / sts_count if sts_count > 0 else 0.0
            
            metrics['StS Contact Rate %'] = (sts_count - awaiting_action_count) / sts_count if sts_count > 0 else 0.0
        
        metrics['Qualified to StS %'] = sts_count / pof_count if pof_count > 0 else 0.0
        metrics['Qualified to Appt %'] = appt_count / pof_count if pof_count > 0 else 0.0
        metrics['Qualified to ICF %'] = icf_count / pof_count if pof_count > 0 else 0.0
        metrics['Qualified to Enrollment %'] = enr_count / pof_count if pof_count > 0 else 0.0
        
        metrics['StS to Appt %'] = appt_count / sts_count if sts_count > 0 else 0.0
        metrics['StS to ICF %'] = icf_count / sts_count if sts_count > 0 else 0.0
        metrics['StS to Enrollment %'] = enr_count / sts_count if sts_count > 0 else 0.0
        metrics['StS to Lost %'] = lost_after_sts_count / sts_count if sts_count > 0 else 0.0
        
        metrics['ICF to Enrollment %'] = enr_count / icf_count if icf_count > 0 else 0.0
        metrics['SF or Lost After ICF %'] = sf_or_lost_after_icf_count / icf_count if icf_count > 0 else 0.0
        
        metrics['Avg time from StS to Appt Sched.'] = calculate_avg_lag_generic(group_df, sts_ts_col, appt_ts_col)
        metrics['Avg time from StS to ICF'] = calculate_avg_lag_generic(group_df, sts_ts_col, icf_ts_col)
        metrics['Avg time from StS to Enrollment'] = calculate_avg_lag_generic(group_df, sts_ts_col, enr_ts_col)
        
        metrics_list.append(metrics)
        
    return pd.DataFrame(metrics_list)

def calculate_enhanced_ad_metrics(_processed_df, ordered_stages, ts_col_map, grouping_col, unclassified_label):
    if _processed_df is None or grouping_col not in _processed_df.columns:
        return pd.DataFrame()
        
    df = _processed_df.copy()
    df[grouping_col] = df[grouping_col].astype(str).str.strip().replace('', unclassified_label).fillna(unclassified_label)

    pof_ts_col = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
    sts_ts_col = ts_col_map.get(STAGE_SENT_TO_SITE)
    appt_ts_col = ts_col_map.get(STAGE_APPOINTMENT_SCHEDULED)
    icf_ts_col = ts_col_map.get(STAGE_SIGNED_ICF)
    enr_ts_col = ts_col_map.get(STAGE_ENROLLED)
    sf_ts_col = ts_col_map.get(STAGE_SCREEN_FAILED)

    metrics_list = []
    for group_name, group_df in df.groupby(grouping_col):
        metrics = {grouping_col: group_name}
        
        pof_count = group_df[pof_ts_col].notna().sum()
        sts_count = group_df[sts_ts_col].notna().sum()
        appt_count = group_df[appt_ts_col].notna().sum()
        icf_count = group_df[icf_ts_col].notna().sum()
        enr_count = group_df[enr_ts_col].notna().sum()
        sf_count = group_df[sf_ts_col].notna().sum()
        
        metrics['Total Qualified'] = pof_count
        metrics['StS Count'] = sts_count
        metrics['Appt Count'] = appt_count
        metrics['ICF Count'] = icf_count
        metrics['Enrollment Count'] = enr_count
        metrics['Screen Fail Count'] = sf_count
        
        metrics['Qualified to StS %'] = sts_count / pof_count if pof_count > 0 else 0.0
        metrics['StS to Appt %'] = appt_count / sts_count if sts_count > 0 else 0.0
        metrics['Qualified to ICF %'] = icf_count / pof_count if pof_count > 0 else 0.0
        metrics['Qualified to Enrollment %'] = enr_count / pof_count if pof_count > 0 else 0.0
        metrics['ICF to Enrollment %'] = enr_count / icf_count if icf_count > 0 else 0.0
        metrics['Screen Fail % (from Qualified)'] = sf_count / pof_count if pof_count > 0 else 0.0

        all_ts_cols_after_sts = [ts_col_map.get(s) for s in ordered_stages if ts_col_map.get(s) and ts_col_map.get(s) != sts_ts_col]
        all_ts_cols_after_sts = [c for c in all_ts_cols_after_sts if c in group_df.columns]
        
        def find_first_action_after_sts(row):
            sts_time = row[sts_ts_col]
            if pd.isna(sts_time): return pd.NaT
            future_events = row[all_ts_cols_after_sts][row[all_ts_cols_after_sts] > sts_time]
            return future_events.min() if not future_events.empty else pd.NaT
            
        first_actions = group_df.apply(find_first_action_after_sts, axis=1)
        metrics['Average time to first site action'] = ((first_actions - group_df[sts_ts_col]).dt.total_seconds() / (60*60*24)).mean()
        metrics['Avg time from StS to Appt Sched.'] = calculate_avg_lag_generic(group_df, sts_ts_col, appt_ts_col)

        metrics_list.append(metrics)
        
    return pd.DataFrame(metrics_list)

def calculate_lost_reasons_after_sts(df, ts_col_map, status_history_col, funnel_def, selected_site="Overall"):
    if df is None or df.empty or not funnel_def:
        return pd.Series(dtype='int64')

    lost_ts_col = ts_col_map.get(STAGE_LOST)
    sts_ts_col = ts_col_map.get(STAGE_SENT_TO_SITE)
    if not all(col in df.columns for col in [lost_ts_col, sts_ts_col, status_history_col]):
        return pd.Series(dtype='int64')

    if selected_site != "Overall":
        if 'Site' not in df.columns: return pd.Series(dtype='int64')
        analysis_df = df[df['Site'] == selected_site].copy()
    else:
        analysis_df = df.copy()

    lost_df = analysis_df.dropna(subset=[lost_ts_col, sts_ts_col])
    lost_df = lost_df[lost_df[lost_ts_col] > lost_df[sts_ts_col]].copy()
    
    if lost_df.empty:
        return pd.Series(dtype='int64')

    valid_lost_statuses = funnel_def.get(STAGE_LOST, [])
    
    def get_final_lost_status(row):
        history = row[status_history_col]
        if not isinstance(history, list) or not history:
            return "Lost - Unspecified"
        
        for event_name, event_dt in reversed(history):
            if event_name in valid_lost_statuses:
                return event_name
        
        return "Lost - Unspecified"

    lost_df['Lost Reason'] = lost_df.apply(get_final_lost_status, axis=1)
    
    return lost_df['Lost Reason'].value_counts()