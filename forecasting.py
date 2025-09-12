# utils/forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

from constants import *
from calculations import calculate_avg_lag_generic

def determine_effective_projection_rates(_processed_df, ordered_stages, ts_col_map,
                                          rate_method_sidebar, rolling_window_sidebar, manual_rates_sidebar,
                                          inter_stage_lags_for_maturity,
                                          sidebar_display_area=None):
    MIN_DENOMINATOR_FOR_RATE_CALC = 5; DEFAULT_MATURITY_DAYS = 45; MATURITY_LAG_MULTIPLIER = 1.5
    MIN_EFFECTIVE_MATURITY_DAYS = 20; MIN_TOTAL_DENOMINATOR_FOR_OVERALL_RATE = 20
    if ts_col_map is None:
        if sidebar_display_area: sidebar_display_area.error("TS Column Map not available for rate calculation.")
        return manual_rates_sidebar, "Manual (TS Col Map Missing)"
    if _processed_df is None or _processed_df.empty:
        if sidebar_display_area: sidebar_display_area.caption("Using manual rates (No historical data for rolling).")
        return manual_rates_sidebar, "Manual (No History)"
    if rate_method_sidebar == 'Manual Input Below':
        if sidebar_display_area: sidebar_display_area.caption("Using manually input conversion rates for projection.")
        return manual_rates_sidebar, "Manual Input"
    calculated_rolling_rates = {}; method_description = "Manual (Error or No History for Rolling)"; substitutions_made_log = []
    MATURITY_PERIODS_DAYS = {}
    if inter_stage_lags_for_maturity:
        for rate_key_for_lag in manual_rates_sidebar.keys():
            avg_lag_for_key = inter_stage_lags_for_maturity.get(rate_key_for_lag)
            if pd.notna(avg_lag_for_key) and avg_lag_for_key > 0:
                calculated_maturity = round(MATURITY_LAG_MULTIPLIER * avg_lag_for_key)
                MATURITY_PERIODS_DAYS[rate_key_for_lag] = max(calculated_maturity, MIN_EFFECTIVE_MATURITY_DAYS)
            else: MATURITY_PERIODS_DAYS[rate_key_for_lag] = DEFAULT_MATURITY_DAYS
    else:
        for rate_key_for_lag in manual_rates_sidebar.keys(): MATURITY_PERIODS_DAYS[rate_key_for_lag] = DEFAULT_MATURITY_DAYS
        substitutions_made_log.append(f"Maturity: Inter-stage lags N/A or empty, used default {DEFAULT_MATURITY_DAYS}d for all.")
    try:
        if "Submission_Month" not in _processed_df.columns or _processed_df["Submission_Month"].dropna().empty:
            if sidebar_display_area: sidebar_display_area.warning("Not enough historical submission month data. Using manual rates.")
            return manual_rates_sidebar, "Manual (No Submission Month History)"
        hist_counts = _processed_df.groupby("Submission_Month").size().to_frame(name="Total_Submissions_Calc")
        reached_stage_cols_map_hist = {}
        for stage_name_iter in ordered_stages:
            ts_col_iter = ts_col_map.get(stage_name_iter)
            if ts_col_iter and ts_col_iter in _processed_df.columns and pd.api.types.is_datetime64_any_dtype(_processed_df[ts_col_iter]):
                cleaned_stage_name_for_col = f"Reached_{stage_name_iter.replace(' ', '_').replace('(', '').replace(')', '')}"
                reached_stage_cols_map_hist[stage_name_iter] = cleaned_stage_name_for_col
                stage_monthly_counts = _processed_df.dropna(subset=[ts_col_iter]).groupby(_processed_df['Submission_Month']).size()
                hist_counts = hist_counts.join(stage_monthly_counts.rename(cleaned_stage_name_for_col), how='left')
        hist_counts = hist_counts.fillna(0)
        pof_hist_col_mapped_name = reached_stage_cols_map_hist.get(STAGE_PASSED_ONLINE_FORM)
        valid_historical_rates_found = False
        for rate_key in manual_rates_sidebar.keys():
            try: stage_from_name, stage_to_name = rate_key.split(" -> ")
            except ValueError:
                calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0)
                substitutions_made_log.append(f"{rate_key}: Error parsing stage names, used manual rate."); continue
            if stage_from_name == STAGE_PASSED_ONLINE_FORM:
                actual_col_from = pof_hist_col_mapped_name if pof_hist_col_mapped_name else "Total_Submissions_Calc"
            else: actual_col_from = reached_stage_cols_map_hist.get(stage_from_name)
            col_to_cleaned_name = reached_stage_cols_map_hist.get(stage_to_name)
            if actual_col_from and col_to_cleaned_name and actual_col_from in hist_counts.columns and col_to_cleaned_name in hist_counts.columns:
                total_numerator = hist_counts[col_to_cleaned_name].sum(); total_denominator = hist_counts[actual_col_from].sum()
                overall_hist_rate_for_key = (total_numerator / total_denominator) if total_denominator > 0 else np.nan
                manual_rate_for_key = manual_rates_sidebar.get(rate_key, 0.0)
                maturity_days_for_this_rate = MATURITY_PERIODS_DAYS.get(rate_key, DEFAULT_MATURITY_DAYS)
                adjusted_monthly_rates_list = []; months_used_for_rate = []
                for month_period_loop in hist_counts.index:
                    if month_period_loop.end_time + pd.Timedelta(days=maturity_days_for_this_rate) < pd.Timestamp(datetime.now()):
                        months_used_for_rate.append(month_period_loop)
                        numerator_val = hist_counts.loc[month_period_loop, col_to_cleaned_name]
                        denominator_val = hist_counts.loc[month_period_loop, actual_col_from]
                        rate_for_this_month_calc = 0.0
                        if denominator_val < MIN_DENOMINATOR_FOR_RATE_CALC:
                            rate_for_this_month_calc = manual_rate_for_key
                            log_reason_detail = f"used manual rate ({manual_rate_for_key*100:.1f}%)"
                            is_manual_rate_placeholder = (manual_rate_for_key >= 0.99 or manual_rate_for_key <= 0.01)
                            if is_manual_rate_placeholder:
                                if pd.notna(overall_hist_rate_for_key) and total_denominator >= MIN_TOTAL_DENOMINATOR_FOR_OVERALL_RATE:
                                    rate_for_this_month_calc = overall_hist_rate_for_key
                                    log_reason_detail = f"manual placeholder, used overall hist. ({overall_hist_rate_for_key*100:.1f}%, total N={total_denominator})"
                            substitutions_made_log.append(f"Mth {month_period_loop.strftime('%Y-%m')} for '{rate_key}': Denom ({denominator_val}) < {MIN_DENOMINATOR_FOR_RATE_CALC}, {log_reason_detail}. Maturity: {maturity_days_for_this_rate}d.")
                        elif denominator_val > 0: rate_for_this_month_calc = numerator_val / denominator_val
                        adjusted_monthly_rates_list.append(rate_for_this_month_calc)
                    else: substitutions_made_log.append(f"Mth {month_period_loop.strftime('%Y-%m')} for '{rate_key}': Excluded (too recent, maturity: {maturity_days_for_this_rate}d).")
                if not adjusted_monthly_rates_list:
                    calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0)
                    substitutions_made_log.append(f"{rate_key}: No mature hist. mths (mat: {maturity_days_for_this_rate}d), used manual rate."); continue
                adjusted_monthly_rates_series = pd.Series(adjusted_monthly_rates_list, index=pd.PeriodIndex(months_used_for_rate, freq='M'))
                actual_window_to_calc = min(rolling_window_sidebar, len(adjusted_monthly_rates_series))
                if actual_window_to_calc > 0 and not adjusted_monthly_rates_series.empty:
                    rolling_avg_rate_series = adjusted_monthly_rates_series.rolling(window=actual_window_to_calc, min_periods=1).mean()
                    if not rolling_avg_rate_series.empty:
                        latest_rolling_rate = rolling_avg_rate_series.iloc[-1]
                        calculated_rolling_rates[rate_key] = latest_rolling_rate if pd.notna(latest_rolling_rate) else 0.0
                        valid_historical_rates_found = True
                    else:
                        calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0)
                        substitutions_made_log.append(f"{rate_key}: Rolling avg empty (mat: {maturity_days_for_this_rate}d), used manual rate.")
                else:
                    if not adjusted_monthly_rates_series.empty:
                        mean_mature_rate_val = adjusted_monthly_rates_series.mean()
                        calculated_rolling_rates[rate_key] = mean_mature_rate_val if pd.notna(mean_mature_rate_val) else manual_rates_sidebar.get(rate_key, 0.0)
                        substitutions_made_log.append(f"{rate_key}: Window {actual_window_to_calc} (mat: {maturity_days_for_this_rate}d) too small or no data; used mean of mature or manual. Valid: {pd.notna(mean_mature_rate_val)}")
                        if pd.notna(mean_mature_rate_val): valid_historical_rates_found = True
                    else:
                        calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0)
                        substitutions_made_log.append(f"{rate_key}: No mature data for rolling (mat: {maturity_days_for_this_rate}d), used manual rate.")
            else:
                calculated_rolling_rates[rate_key] = manual_rates_sidebar.get(rate_key, 0.0)
                substitutions_made_log.append(f"{rate_key}: Stage columns N/A in historical summary, used manual rate.")
        if sidebar_display_area and substitutions_made_log:
            with sidebar_display_area.expander("Rolling Rate Calculation Log (Adjustments & Maturity)", expanded=False):
                sidebar_display_area.caption("Maturity Periods Applied (Days):")
                if MATURITY_PERIODS_DAYS:
                    for r_key_disp_log, mat_days_disp_log in MATURITY_PERIODS_DAYS.items(): sidebar_display_area.caption(f"- {r_key_disp_log}: {mat_days_disp_log} days")
                else: sidebar_display_area.caption("Maturity periods N/A.")
                sidebar_display_area.caption("--- Substitution/Exclusion Log ---")
                for log_entry_disp in substitutions_made_log: sidebar_display_area.caption(log_entry_disp)
        if not valid_historical_rates_found:
            if sidebar_display_area: sidebar_display_area.warning("No valid historical rolling rates could be calculated, using manual inputs provided.")
            return manual_rates_sidebar, "Manual (All Rolling Calcs Failed or Invalid)"
        else:
            if sidebar_display_area:
                sidebar_display_area.markdown("---"); sidebar_display_area.subheader(f"Effective {rolling_window_sidebar}-Mo. Rolling Rates (Adj. & Matured):")
                for key_disp, val_disp in calculated_rolling_rates.items():
                    if key_disp in manual_rates_sidebar: sidebar_display_area.text(f"- {key_disp}: {val_disp*100:.1f}%")
                sidebar_display_area.markdown("---")
            return calculated_rolling_rates, f"Rolling {rolling_window_sidebar}-Month Avg (Adj. & Matured)"
    except Exception as e:
        if sidebar_display_area: sidebar_display_area.error(f"Error calculating rolling rates: {e}"); sidebar_display_area.exception(e)
        return manual_rates_sidebar, "Manual (Error in Rolling Calc)"

@st.cache_data
def calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs):
    default_return_tuple = pd.DataFrame(), np.nan, "N/A", "N/A", pd.DataFrame(), "N/A"
    if _processed_df is None or _processed_df.empty: return default_return_tuple
    required_keys = ['horizon', 'spend_dict', 'cpqr_dict', 'final_conv_rates', 'goal_icf',
                     'site_performance_data', 'inter_stage_lags', 'icf_variation_percentage']
    if not isinstance(projection_inputs, dict) or not all(k in projection_inputs for k in required_keys):
        missing_keys = [k for k in required_keys if k not in projection_inputs]
        st.warning(f"Proj: Missing inputs. Need: {missing_keys}. Got: {list(projection_inputs.keys())}")
        return default_return_tuple

    processed_df = _processed_df.copy(); horizon = projection_inputs['horizon']
    future_spend_dict = projection_inputs['spend_dict']; assumed_cpqr_dict = projection_inputs['cpqr_dict']
    final_projection_conv_rates = projection_inputs['final_conv_rates']; goal_total_icfs = projection_inputs['goal_icf']
    site_performance_data = projection_inputs['site_performance_data']
    inter_stage_lags = projection_inputs.get('inter_stage_lags', {}); icf_variation_percent = projection_inputs.get('icf_variation_percentage', 0)
    variation_factor = icf_variation_percent / 100.0

    avg_actual_lag_days_for_display = np.nan; lag_calculation_method_message = "Lag not calculated."
    projection_segments_for_lag_path = [
        (STAGE_PASSED_ONLINE_FORM, STAGE_PRE_SCREENING_ACTIVITIES),
        (STAGE_PRE_SCREENING_ACTIVITIES, STAGE_SENT_TO_SITE),
        (STAGE_SENT_TO_SITE, STAGE_APPOINTMENT_SCHEDULED),
        (STAGE_APPOINTMENT_SCHEDULED, STAGE_SIGNED_ICF)
    ]
    calculated_sum_of_lags = 0; valid_segments_count = 0; all_segments_found_and_valid = True
    if inter_stage_lags:
        for stage_from, stage_to in projection_segments_for_lag_path:
            lag_key = f"{stage_from} -> {stage_to}"; lag_value = inter_stage_lags.get(lag_key)
            if ts_col_map.get(stage_from) and ts_col_map.get(stage_to):
                if pd.notna(lag_value): calculated_sum_of_lags += lag_value; valid_segments_count += 1
                else: all_segments_found_and_valid = False; break
            # else: all_segments_found_and_valid = False; break
        if all_segments_found_and_valid and valid_segments_count == len(projection_segments_for_lag_path):
            avg_actual_lag_days_for_display = calculated_sum_of_lags
            lag_calculation_method_message = "Using summed inter-stage lags for POF->ICF projection path."
        else:
            all_segments_found_and_valid = False
            lag_calculation_method_message = "Summed inter-stage lag for POF->ICF path failed (missing segments or lags). "
    else: all_segments_found_and_valid = False; lag_calculation_method_message = "Inter-stage lags not available. "

    if not all_segments_found_and_valid or pd.isna(avg_actual_lag_days_for_display):
        start_stage_for_overall_lag = ordered_stages[0] if ordered_stages and len(ordered_stages) > 0 else None
        end_stage_for_overall_lag = STAGE_SIGNED_ICF
        overall_lag_calc_val = np.nan
        if start_stage_for_overall_lag and ts_col_map.get(start_stage_for_overall_lag) and ts_col_map.get(end_stage_for_overall_lag):
            ts_col_start_overall = ts_col_map[start_stage_for_overall_lag]; ts_col_end_overall = ts_col_map[end_stage_for_overall_lag]
            if ts_col_start_overall in processed_df.columns and ts_col_end_overall in processed_df.columns:
                overall_lag_calc_val = calculate_avg_lag_generic(processed_df, ts_col_start_overall, ts_col_end_overall)
        if pd.notna(overall_lag_calc_val):
            avg_actual_lag_days_for_display = overall_lag_calc_val
            lag_calculation_method_message += f"Used historical overall lag ({start_stage_for_overall_lag} -> {end_stage_for_overall_lag})."
        else: avg_actual_lag_days_for_display = 30.0; lag_calculation_method_message += "Used default lag (30 days)."
    if pd.isna(avg_actual_lag_days_for_display): avg_actual_lag_days_for_display = 30.0; lag_calculation_method_message = "Critical Lag Error: All methods failed. Used default 30 days."

    lpi_date_str = "Goal Not Met"; ads_off_date_str = "N/A"; site_level_projections_df_final = pd.DataFrame()
    try:
        last_historical_month = processed_df["Submission_Month"].max() if "Submission_Month" in processed_df and not processed_df["Submission_Month"].empty else pd.Period(datetime.now(), freq='M') - 1
        proj_start_month = last_historical_month + 1
        future_months = pd.period_range(start=proj_start_month, periods=horizon, freq='M')
        projection_cohorts = pd.DataFrame(index=future_months)
        projection_cohorts['Forecasted_Ad_Spend'] = [future_spend_dict.get(m, 0.0) for m in future_months]
        forecasted_psq_list = []
        for month_period in future_months:
            spend = projection_cohorts.loc[month_period, 'Forecasted_Ad_Spend']
            cpqr_for_month = assumed_cpqr_dict.get(month_period, 120.0)
            if cpqr_for_month <= 0: cpqr_for_month = 120.0
            forecasted_psq_list.append(np.round(spend / cpqr_for_month).astype(int) if cpqr_for_month > 0 else 0)
        projection_cohorts['Forecasted_PSQ'] = forecasted_psq_list
        projection_cohorts['Forecasted_PSQ'] = projection_cohorts['Forecasted_PSQ'].fillna(0).astype(int)
        icf_proj_col_name_base = "" ; current_stage_count_col = 'Forecasted_PSQ'
        for stage_from, stage_to in projection_segments_for_lag_path:
            conv_rate_key = f"{stage_from} -> {stage_to}"; conv_rate = final_projection_conv_rates.get(conv_rate_key, 0.0)
            proj_col_to_name = f"Projected_{stage_to.replace(' ', '_').replace('(', '').replace(')', '')}"
            if current_stage_count_col in projection_cohorts.columns:
                proj_counts_for_to_stage = (projection_cohorts[current_stage_count_col] * conv_rate)
                projection_cohorts[proj_col_to_name] = proj_counts_for_to_stage.round(0).fillna(0).astype(int)
                current_stage_count_col = proj_col_to_name
            else: projection_cohorts[proj_col_to_name] = 0; current_stage_count_col = proj_col_to_name
            if stage_to == STAGE_SIGNED_ICF: icf_proj_col_name_base = proj_col_to_name; break

        projection_results = pd.DataFrame(index=future_months); projection_results['Projected_ICF_Landed'] = 0.0
        if not icf_proj_col_name_base or icf_proj_col_name_base not in projection_cohorts.columns:
            st.error(f"Critical Error: Projected ICF column ('{icf_proj_col_name_base}') not found after funnel progression.");
            return default_return_tuple[0], avg_actual_lag_days_for_display, "Error", "Error", pd.DataFrame(), "ICF Proj Col Missing after funnel calc"

        icf_proj_col_low = f"{icf_proj_col_name_base}_low"; icf_proj_col_high = f"{icf_proj_col_name_base}_high"
        projection_cohorts[icf_proj_col_low] = (projection_cohorts[icf_proj_col_name_base] * (1 - variation_factor)).round(0).astype(int).clip(lower=0)
        projection_cohorts[icf_proj_col_high] = (projection_cohorts[icf_proj_col_name_base] * (1 + variation_factor)).round(0).astype(int).clip(lower=0)
        projection_cohorts['Projected_CPICF_Cohort_Mean'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col_name_base].replace(0, np.nan)).round(2)
        projection_cohorts['Projected_CPICF_Cohort_Low'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col_high].replace(0, np.nan)).round(2)
        projection_cohorts['Projected_CPICF_Cohort_High'] = (projection_cohorts['Forecasted_Ad_Spend'] / projection_cohorts[icf_proj_col_low].replace(0, np.nan)).round(2)

        overall_current_lag_days_to_use = avg_actual_lag_days_for_display; days_in_avg_month = 30.4375
        for cohort_start_month_iter, row_data in projection_cohorts.iterrows():
            icfs_from_this_cohort_val = row_data[icf_proj_col_name_base]
            if icfs_from_this_cohort_val == 0: continue
            full_lag_months_val = int(np.floor(overall_current_lag_days_to_use / days_in_avg_month))
            remaining_lag_days_comp_val = overall_current_lag_days_to_use - (full_lag_months_val * days_in_avg_month)
            fraction_for_next_month_val = remaining_lag_days_comp_val / days_in_avg_month; fraction_for_current_offset_month_val = 1.0 - fraction_for_next_month_val
            icfs_month_1_val = icfs_from_this_cohort_val * fraction_for_current_offset_month_val
            icfs_month_2_val = icfs_from_this_cohort_val * fraction_for_next_month_val
            landing_month_1_period_val = cohort_start_month_iter + full_lag_months_val; landing_month_2_period_val = cohort_start_month_iter + full_lag_months_val + 1
            if landing_month_1_period_val in projection_results.index: projection_results.loc[landing_month_1_period_val, 'Projected_ICF_Landed'] += icfs_month_1_val
            if landing_month_2_period_val in projection_results.index: projection_results.loc[landing_month_2_period_val, 'Projected_ICF_Landed'] += icfs_month_2_val
        projection_results['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed'].round(0).fillna(0).astype(int)
        projection_results['Cumulative_ICF_Landed'] = projection_results['Projected_ICF_Landed'].cumsum()

        lpi_month_series_val = projection_results[projection_results['Cumulative_ICF_Landed'] >= goal_total_icfs]
        if not lpi_month_series_val.empty:
            lpi_month_period_val = lpi_month_series_val.index[0]; icfs_in_lpi_month_val = projection_results.loc[lpi_month_period_val, 'Projected_ICF_Landed']
            cumulative_before_lpi_direct_val = projection_results['Cumulative_ICF_Landed'].shift(1).fillna(0).loc[lpi_month_period_val]
            icfs_needed_in_lpi_month_val = goal_total_icfs - cumulative_before_lpi_direct_val
            if icfs_in_lpi_month_val > 0:
                fraction_of_lpi_month_val = max(0,min(1, icfs_needed_in_lpi_month_val / icfs_in_lpi_month_val))
                lpi_day_offset_val = int(np.ceil(fraction_of_lpi_month_val * days_in_avg_month)); lpi_day_offset_val = max(1, lpi_day_offset_val)
                lpi_date_val_calc = lpi_month_period_val.start_time + pd.Timedelta(days=lpi_day_offset_val -1); lpi_date_str = lpi_date_val_calc.strftime('%Y-%m-%d')
            elif icfs_needed_in_lpi_month_val <= 0:
                 lpi_date_str = (lpi_month_period_val.start_time - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            else: lpi_date_str = lpi_month_period_val.start_time.strftime('%Y-%m-%d')

        projection_cohorts['Cumulative_Projected_ICF_Generated'] = projection_cohorts[icf_proj_col_name_base].cumsum()
        ads_off_s_granular = projection_cohorts[projection_cohorts['Cumulative_Projected_ICF_Generated'] >= (goal_total_icfs - 0.5 + 1e-9)]
        if not ads_off_s_granular.empty:
            ads_off_month_period_granular = ads_off_s_granular.index[0]
            icfs_gen_by_ads_off_month_granular = projection_cohorts.loc[ads_off_month_period_granular, icf_proj_col_name_base]
            cum_gen_before_ads_off_month_granular = projection_cohorts.loc[ads_off_month_period_granular, 'Cumulative_Projected_ICF_Generated'] - icfs_gen_by_ads_off_month_granular
            icfs_needed_from_ads_off_month_granular = goal_total_icfs - cum_gen_before_ads_off_month_granular
            if icfs_needed_from_ads_off_month_granular <= 0:
                prev_m_ads_off_granular = ads_off_month_period_granular - 1
                if prev_m_ads_off_granular in projection_cohorts.index and cum_gen_before_ads_off_month_granular >= (goal_total_icfs - 0.5 + 1e-9):
                     ads_off_date_str = prev_m_ads_off_granular.end_time.strftime('%Y-%m-%d')
                else: ads_off_date_str = ads_off_month_period_granular.start_time.strftime('%Y-%m-%d')
            elif icfs_gen_by_ads_off_month_granular > 1e-9:
                fraction_needed_granular = max(0, min(1, icfs_needed_from_ads_off_month_granular / icfs_gen_by_ads_off_month_granular))
                day_offset_granular = int(np.ceil(fraction_needed_granular * days_in_avg_month)); day_offset_granular = max(1,day_offset_granular)
                ads_off_date_str = (ads_off_month_period_granular.start_time + pd.Timedelta(days=day_offset_granular -1)).strftime('%Y-%m-%d')
            else: ads_off_date_str = ads_off_month_period_granular.start_time.strftime('%Y-%m-%d')

        display_df_out = pd.DataFrame(index=future_months)
        display_df_out['Forecasted_Ad_Spend'] = projection_cohorts['Forecasted_Ad_Spend']
        display_df_out['Forecasted_Qual_Referrals'] = projection_cohorts['Forecasted_PSQ']
        display_df_out['Projected_ICF_Landed'] = projection_results['Projected_ICF_Landed']
        cpicf_display_mean_val = pd.Series(index=future_months, dtype=float); cpicf_display_low_val = pd.Series(index=future_months, dtype=float); cpicf_display_high_val = pd.Series(index=future_months, dtype=float)
        lag_for_cpicf_display_val = int(np.round(overall_current_lag_days_to_use / days_in_avg_month))
        for i_cohort_idx, cohort_start_month_cpicf in enumerate(projection_cohorts.index):
            primary_land_m_cpicf = cohort_start_month_cpicf + lag_for_cpicf_display_val
            if primary_land_m_cpicf in cpicf_display_mean_val.index:
                if pd.isna(cpicf_display_mean_val.loc[primary_land_m_cpicf]):
                    cpicf_display_mean_val.loc[primary_land_m_cpicf] = projection_cohorts.iloc[i_cohort_idx]['Projected_CPICF_Cohort_Mean']
                    cpicf_display_low_val.loc[primary_land_m_cpicf] = projection_cohorts.iloc[i_cohort_idx]['Projected_CPICF_Cohort_Low']
                    cpicf_display_high_val.loc[primary_land_m_cpicf] = projection_cohorts.iloc[i_cohort_idx]['Projected_CPICF_Cohort_High']
        display_df_out['Projected_CPICF_Cohort_Source_Mean'] = cpicf_display_mean_val
        display_df_out['Projected_CPICF_Cohort_Source_Low'] = cpicf_display_low_val
        display_df_out['Projected_CPICF_Cohort_Source_High'] = cpicf_display_high_val

        if 'Site' in _processed_df.columns and not _processed_df['Site'].empty and ordered_stages:
            if site_performance_data.empty or 'Site' not in site_performance_data.columns:
                 st.warning("Site performance data is empty or missing 'Site' column for site-level projections.")
                 site_level_projections_df_final = pd.DataFrame()
            else:
                historical_site_pof_proportions = pd.Series(dtype=float)
                pof_ts_col_for_prop = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
                if pof_ts_col_for_prop and pof_ts_col_for_prop in _processed_df.columns:
                    valid_pof_df = _processed_df[_processed_df[pof_ts_col_for_prop].notna()]
                    if not valid_pof_df.empty:
                         historical_site_pof_proportions = valid_pof_df['Site'].value_counts(normalize=True)

                all_sites_proj = site_performance_data['Site'].unique()
                site_data_collector_proj = {}
                for site_name_s_proj in all_sites_proj:
                    site_data_collector_proj[site_name_s_proj] = {}
                    for month_period_s_proj in future_months:
                        month_str_s_proj = month_period_s_proj.strftime('%Y-%m')
                        site_data_collector_proj[site_name_s_proj][(month_str_s_proj, 'Projected Qual. Referrals (POF)')] = 0
                        site_data_collector_proj[site_name_s_proj][(month_str_s_proj, 'Projected ICFs Landed')] = 0.0

                for cohort_start_month_s_proj, cohort_row_s_proj in projection_cohorts.iterrows():
                    total_psq_this_cohort_s_proj = cohort_row_s_proj['Forecasted_PSQ']
                    if total_psq_this_cohort_s_proj <= 0: continue

                    site_pof_allocations_this_cohort = {}
                    if not historical_site_pof_proportions.empty:
                        for site_name_iter_s_proj in all_sites_proj:
                            site_prop_s_proj = historical_site_pof_proportions.get(site_name_iter_s_proj, 0)
                            site_pof_allocations_this_cohort[site_name_iter_s_proj] = total_psq_this_cohort_s_proj * site_prop_s_proj
                    elif len(all_sites_proj) > 0:
                        equal_share = total_psq_this_cohort_s_proj / len(all_sites_proj)
                        for site_name_iter_s_proj in all_sites_proj:
                            site_pof_allocations_this_cohort[site_name_iter_s_proj] = equal_share

                    current_sum_pof_float = sum(site_pof_allocations_this_cohort.values())
                    if abs(current_sum_pof_float - total_psq_this_cohort_s_proj) > 1e-6 and current_sum_pof_float > 0:
                        rescale_factor = total_psq_this_cohort_s_proj / current_sum_pof_float
                        for site_n in site_pof_allocations_this_cohort: site_pof_allocations_this_cohort[site_n] *= rescale_factor

                    rounded_site_pof_allocations = {site: round(val) for site, val in site_pof_allocations_this_cohort.items()}
                    diff_after_round = total_psq_this_cohort_s_proj - sum(rounded_site_pof_allocations.values())

                    if diff_after_round != 0 and site_pof_allocations_this_cohort:
                        site_to_adjust = max(site_pof_allocations_this_cohort, key=lambda s: site_pof_allocations_this_cohort[s] - math.floor(site_pof_allocations_this_cohort[s]))
                        rounded_site_pof_allocations[site_to_adjust] += diff_after_round

                    for site_name_iter_s_proj in all_sites_proj:
                        site_proj_pof_cohort_s_proj = rounded_site_pof_allocations.get(site_name_iter_s_proj, 0)
                        month_str_cohort_start_s_proj = cohort_start_month_s_proj.strftime('%Y-%m')
                        site_data_collector_proj[site_name_iter_s_proj][(month_str_cohort_start_s_proj, 'Projected Qual. Referrals (POF)')] += site_proj_pof_cohort_s_proj

                        site_perf_row_s_proj = site_performance_data[site_performance_data['Site'] == site_name_iter_s_proj]
                        current_site_proj_count_s_proj = float(site_proj_pof_cohort_s_proj)

                        for i_seg_s, (stage_from_seg_s, stage_to_seg_s) in enumerate(projection_segments_for_lag_path):
                            site_rate_key_s = ""
                            if stage_from_seg_s == STAGE_PASSED_ONLINE_FORM and stage_to_seg_s == STAGE_PRE_SCREENING_ACTIVITIES: site_rate_key_s = 'POF -> PSA %'
                            elif stage_from_seg_s == STAGE_PRE_SCREENING_ACTIVITIES and stage_to_seg_s == STAGE_SENT_TO_SITE: site_rate_key_s = 'PSA -> StS %'
                            elif stage_from_seg_s == STAGE_SENT_TO_SITE and stage_to_seg_s == STAGE_APPOINTMENT_SCHEDULED: site_rate_key_s = 'StS -> Appt %'
                            elif stage_from_seg_s == STAGE_APPOINTMENT_SCHEDULED and stage_to_seg_s == STAGE_SIGNED_ICF: site_rate_key_s = 'Appt -> ICF %'

                            overall_rate_key_s = f"{stage_from_seg_s} -> {stage_to_seg_s}"
                            rate_to_use_s = final_projection_conv_rates.get(overall_rate_key_s, 0.0)

                            if not site_perf_row_s_proj.empty and site_rate_key_s and site_rate_key_s in site_perf_row_s_proj.columns:
                                site_specific_rate_val_s = site_perf_row_s_proj[site_rate_key_s].iloc[0]
                                if pd.notna(site_specific_rate_val_s) and site_specific_rate_val_s >= 0:
                                    rate_to_use_s = site_specific_rate_val_s

                            current_site_proj_count_s_proj *= rate_to_use_s
                            if stage_to_seg_s == STAGE_SIGNED_ICF: break

                        site_proj_icfs_generated_this_cohort_s = current_site_proj_count_s_proj

                        lag_to_use_for_site_s = overall_current_lag_days_to_use
                        if not site_perf_row_s_proj.empty and 'Site Projection Lag (Days)' in site_perf_row_s_proj.columns:
                            site_specific_lag_val_s = site_perf_row_s_proj['Site Projection Lag (Days)'].iloc[0]
                            if pd.notna(site_specific_lag_val_s) and site_specific_lag_val_s >=0: lag_to_use_for_site_s = site_specific_lag_val_s

                        if site_proj_icfs_generated_this_cohort_s > 0:
                            full_lag_m_site_s = int(np.floor(lag_to_use_for_site_s / days_in_avg_month))
                            remain_lag_days_comp_site_s = lag_to_use_for_site_s - (full_lag_m_site_s * days_in_avg_month)
                            frac_next_m_site_s = remain_lag_days_comp_site_s / days_in_avg_month; frac_curr_m_site_s = 1.0 - frac_next_m_site_s

                            icfs_m1_site_s = site_proj_icfs_generated_this_cohort_s * frac_curr_m_site_s
                            icfs_m2_site_s = site_proj_icfs_generated_this_cohort_s * frac_next_m_site_s

                            land_m1_p_site_s = cohort_start_month_s_proj + full_lag_m_site_s
                            land_m2_p_site_s = land_m1_p_site_s + 1

                            if land_m1_p_site_s in future_months:
                                site_data_collector_proj[site_name_iter_s_proj][(land_m1_p_site_s.strftime('%Y-%m'), 'Projected ICFs Landed')] += icfs_m1_site_s
                            if land_m2_p_site_s in future_months:
                                site_data_collector_proj[site_name_iter_s_proj][(land_m2_p_site_s.strftime('%Y-%m'), 'Projected ICFs Landed')] += icfs_m2_site_s

                if site_data_collector_proj:
                    site_level_projections_df_final = pd.DataFrame.from_dict(site_data_collector_proj, orient='index')
                    site_level_projections_df_final.columns = pd.MultiIndex.from_tuples(site_level_projections_df_final.columns, names=['Month', 'Metric'])
                    for m_period_fmt_s_proj in future_months.strftime('%Y-%m'):
                        icf_landed_col_tuple = (m_period_fmt_s_proj, 'Projected ICFs Landed')
                        if icf_landed_col_tuple in site_level_projections_df_final.columns:
                            site_level_projections_df_final[icf_landed_col_tuple] = site_level_projections_df_final[icf_landed_col_tuple].round(0).astype(int)

                        pof_col_tuple = (m_period_fmt_s_proj, 'Projected Qual. Referrals (POF)')
                        if pof_col_tuple in site_level_projections_df_final.columns:
                            site_level_projections_df_final[pof_col_tuple] = site_level_projections_df_final[pof_col_tuple].astype(int)

                    site_level_projections_df_final = site_level_projections_df_final.sort_index(axis=1, level=[0,1])
                    if not site_level_projections_df_final.empty:
                        numeric_cols_slp = [col for col in site_level_projections_df_final.columns if pd.api.types.is_numeric_dtype(site_level_projections_df_final[col])]
                        if numeric_cols_slp:
                            total_row_vals_slp = site_level_projections_df_final[numeric_cols_slp].sum(axis=0)
                            total_row_df_slp = pd.DataFrame([total_row_vals_slp], index=["Grand Total"])
                            site_level_projections_df_final = pd.concat([site_level_projections_df_final, total_row_df_slp])

        return display_df_out, avg_actual_lag_days_for_display, lpi_date_str, ads_off_date_str, site_level_projections_df_final, lag_calculation_method_message
    except Exception as e:
        st.error(f"Projection calc error (main or site-level): {e}"); st.exception(e)
        return default_return_tuple[0], avg_actual_lag_days_for_display if pd.notna(avg_actual_lag_days_for_display) else 30.0, "Error", "Error", pd.DataFrame(), f"Error: {e}"


def calculate_ai_forecast_core(
    goal_lpi_date_dt_orig: datetime, goal_icf_number_orig: int, estimated_cpql_user: float,
    icf_variation_percent: float,
    processed_df: pd.DataFrame, ordered_stages: list, ts_col_map: dict,
    effective_projection_conv_rates: dict,
    avg_overall_lag_days: float,
    site_metrics_df: pd.DataFrame, projection_horizon_months: int,
    site_caps_input: dict,
    # NEW: Site Activation/Deactivation Schedule
    site_activity_schedule: dict, # Expected format: {'Site A': {'activation_period': pd.Period, 'deactivation_period': pd.Period}, ...}
    site_scoring_weights_for_ai: dict,
    cpql_inflation_factor_pct: float,
    ql_vol_increase_threshold_pct: float,
    run_mode: str = "primary",
    ai_monthly_ql_capacity_multiplier: float = 3.0,
    ai_lag_method: str = "average",
    ai_lag_p25_days: float = None,
    ai_lag_p50_days: float = None,
    ai_lag_p75_days: float = None,
    baseline_monthly_ql_volume_override: float = None # <<< FIX: ADDED NEW PARAMETER
):
    default_return_ai = pd.DataFrame(), pd.DataFrame(), "N/A", "Not Calculated", True, 0
    days_in_avg_m = 30.4375

    if not all([processed_df is not None, not processed_df.empty, ordered_stages, ts_col_map,
                effective_projection_conv_rates, site_metrics_df is not None]): # site_metrics_df can be empty
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Missing critical base data for Auto Forecast.", True, 0
    if goal_icf_number_orig <= 0: return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Goal ICF number must be positive.", True, 0
    if estimated_cpql_user <= 0: return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Estimated CPQL must be positive.", True, 0
    if cpql_inflation_factor_pct < 0 or ql_vol_increase_threshold_pct < 0:
         return default_return_ai[0], default_return_ai[1], default_return_ai[2], "CPQL inflation parameters cannot be negative.", True, 0
    if ai_monthly_ql_capacity_multiplier <=0: ai_monthly_ql_capacity_multiplier = 1.0

    effective_lag_for_planning_approx = avg_overall_lag_days
    if ai_lag_method == "percentiles":
        if not all(pd.notna(lag_val) and lag_val >= 0 for lag_val in [ai_lag_p25_days, ai_lag_p50_days, ai_lag_p75_days]):
            return default_return_ai[0], default_return_ai[1], default_return_ai[2], "P25/P50/P75 lag days must be valid non-negative numbers.", True, 0
        if not (ai_lag_p25_days <= ai_lag_p50_days <= ai_lag_p75_days):
            return default_return_ai[0], default_return_ai[1], default_return_ai[2], "P25 lag must be <= P50, and P50 <= P75.", True, 0
        effective_lag_for_planning_approx = ai_lag_p50_days
    elif pd.isna(avg_overall_lag_days) or avg_overall_lag_days < 0:
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Average POF to ICF lag is invalid.", True, 0

    ts_pof_col_for_prop = ts_col_map.get(STAGE_PASSED_ONLINE_FORM)
    main_funnel_path_segments = [
        f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}",
        f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}",
        f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}",
        f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"
    ]
    overall_pof_to_icf_rate = 1.0
    for segment in main_funnel_path_segments:
        rate = effective_projection_conv_rates.get(segment)
        if rate is None or rate < 0:
            return default_return_ai[0], default_return_ai[1], default_return_ai[2], f"Conversion rate for segment '{segment}' invalid.", True, 0
        overall_pof_to_icf_rate *= rate
    if overall_pof_to_icf_rate <= 1e-9:
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Overall POF to ICF conversion rate is zero.", True, 0

    last_hist_month = processed_df["Submission_Month"].max() if "Submission_Month" in processed_df and not processed_df["Submission_Month"].empty else pd.Period(datetime.now(), freq='M') - 1
    proj_start_month_period = last_hist_month + 1
    current_goal_icf_number = goal_icf_number_orig
    if run_mode == "primary":
        current_goal_lpi_month_period = pd.Period(goal_lpi_date_dt_orig, freq='M')
    elif run_mode == "best_case_extended_lpi":
        current_goal_lpi_month_period = proj_start_month_period + projection_horizon_months -1 # Max LPI
    else: current_goal_lpi_month_period = pd.Period(goal_lpi_date_dt_orig, freq='M')

    avg_lag_months_approx = int(round(effective_lag_for_planning_approx / days_in_avg_m))
    max_possible_proj_end_month_overall = proj_start_month_period + projection_horizon_months -1
    calc_horizon_end_month_for_display = min(max_possible_proj_end_month_overall, current_goal_lpi_month_period + avg_lag_months_approx + 3)
    calc_horizon_end_month_for_display = max(calc_horizon_end_month_for_display, proj_start_month_period)
    projection_calc_months = pd.period_range(start=proj_start_month_period, end=calc_horizon_end_month_for_display, freq='M')

    if projection_calc_months.empty :
        first_possible_landing_month_check = proj_start_month_period + avg_lag_months_approx
        if current_goal_lpi_month_period < first_possible_landing_month_check:
             feasibility_details_early = f"Goal LPI ({current_goal_lpi_month_period.strftime('%Y-%m')}) is too soon. Minimum landing month: {first_possible_landing_month_check.strftime('%Y-%m')} (Lag: {effective_lag_for_planning_approx:.1f} days)."
             return default_return_ai[0], default_return_ai[1], default_return_ai[2], feasibility_details_early, True, 0
        return default_return_ai[0], default_return_ai[1], default_return_ai[2], "Projection calculation window is invalid.", True, 0

    ai_gen_df = pd.DataFrame(index=projection_calc_months)
    ai_gen_df['Required_QLs_POF_Initial'] = 0.0; ai_gen_df['Required_QLs_POF_Final'] = 0.0
    ai_gen_df['Unallocatable_QLs'] = 0.0; ai_gen_df['Generated_ICF_Mean'] = 0.0
    ai_gen_df['Adjusted_CPQL_For_Month'] = estimated_cpql_user; ai_gen_df['Implied_Ad_Spend'] = 0.0
    icfs_still_to_assign_globally = float(current_goal_icf_number)
    latest_permissible_generation_month = current_goal_lpi_month_period - avg_lag_months_approx
    earliest_permissible_generation_month = proj_start_month_period
    valid_generation_months_for_planning = pd.period_range(
        start=max(earliest_permissible_generation_month, projection_calc_months.min()),
        end=min(latest_permissible_generation_month, projection_calc_months.max())
    )

    # <<< FIX: Use the override value if provided, otherwise default to a safe value.
    # The detailed calculation is now done in the UI to provide a robust value here.
    baseline_monthly_ql_volume = 50.0
    if baseline_monthly_ql_volume_override is not None and baseline_monthly_ql_volume_override > 0:
        baseline_monthly_ql_volume = baseline_monthly_ql_volume_override

    monthly_ql_capacity_target_heuristic = baseline_monthly_ql_volume * ai_monthly_ql_capacity_multiplier

    site_level_monthly_qlof = {}
    all_defined_sites = site_metrics_df['Site'].unique() if not site_metrics_df.empty and 'Site' in site_metrics_df else np.array([])

    hist_site_pof_prop_overall = pd.Series(dtype=float)
    if ts_pof_col_for_prop and ts_pof_col_for_prop in processed_df.columns and 'Site' in processed_df.columns:
        valid_pof_data_for_prop = processed_df[processed_df[ts_pof_col_for_prop].notna()]
        if not valid_pof_data_for_prop.empty and 'Site' in valid_pof_data_for_prop:
             hist_site_pof_prop_overall = valid_pof_data_for_prop['Site'].value_counts(normalize=True)

    site_redistribution_scores_overall = {}
    if not site_metrics_df.empty and 'Site' in site_metrics_df.columns and 'Qual -> ICF %' in site_metrics_df.columns:
        for _, site_row_score in site_metrics_df.iterrows():
            score_val = site_row_score.get('Qual -> ICF %', 0.0)
            score_val = 0.0 if pd.isna(score_val) or score_val < 0 else score_val
            site_redistribution_scores_overall[site_row_score['Site']] = score_val if score_val > 1e-6 else 1e-6

    if not valid_generation_months_for_planning.empty:
        for gen_month in valid_generation_months_for_planning:
            if icfs_still_to_assign_globally <= 1e-9: break

            active_sites_this_gen_month = []
            if all_defined_sites.size > 0:
                for site_name_iter in all_defined_sites:
                    is_active_for_month = True
                    if site_activity_schedule and site_name_iter in site_activity_schedule:
                        activity_info = site_activity_schedule[site_name_iter]
                        activation_pd = activity_info.get('activation_period')
                        deactivation_pd = activity_info.get('deactivation_period')
                        if activation_pd and gen_month < activation_pd: is_active_for_month = False
                        if deactivation_pd and gen_month > deactivation_pd: is_active_for_month = False
                    if is_active_for_month: active_sites_this_gen_month.append(site_name_iter)

            hist_site_pof_prop_active = hist_site_pof_prop_overall[hist_site_pof_prop_overall.index.isin(active_sites_this_gen_month)]
            if not hist_site_pof_prop_active.empty and hist_site_pof_prop_active.sum() > 1e-9 :
                 hist_site_pof_prop_active = hist_site_pof_prop_active / hist_site_pof_prop_active.sum()
            else: hist_site_pof_prop_active = pd.Series(dtype=float)
            site_redist_scores_active = {s: score for s, score in site_redistribution_scores_overall.items() if s in active_sites_this_gen_month}

            qls_theoretically_needed_for_remaining_float = (icfs_still_to_assign_globally / overall_pof_to_icf_rate) if overall_pof_to_icf_rate > 1e-9 else float('inf')
            ql_target_potential_this_month_heuristic_limited = min(qls_theoretically_needed_for_remaining_float, monthly_ql_capacity_target_heuristic)
            if icfs_still_to_assign_globally > 1e-9 and \
               qls_theoretically_needed_for_remaining_float < monthly_ql_capacity_target_heuristic and \
               qls_theoretically_needed_for_remaining_float > 0:
                current_month_initial_ql_target = math.ceil(qls_theoretically_needed_for_remaining_float)
            elif ql_target_potential_this_month_heuristic_limited > 0 and ql_target_potential_this_month_heuristic_limited < 1.0:
                current_month_initial_ql_target = math.ceil(ql_target_potential_this_month_heuristic_limited)
            else: current_month_initial_ql_target = round(max(0, ql_target_potential_this_month_heuristic_limited))

            ai_gen_df.loc[gen_month, 'Required_QLs_POF_Initial'] = current_month_initial_ql_target
            site_ql_allocations_month_specific = {site: 0 for site in active_sites_this_gen_month}
            unallocatable_this_month = 0

            if current_month_initial_ql_target > 0 and active_sites_this_gen_month:
                temp_site_allocations_float = {}
                if not hist_site_pof_prop_active.empty:
                    for site_n_dist, prop_dist in hist_site_pof_prop_active.items():
                        temp_site_allocations_float[site_n_dist] = current_month_initial_ql_target * prop_dist
                else:
                    ql_per_site_fallback_float = current_month_initial_ql_target / len(active_sites_this_gen_month)
                    for site_n_dist in active_sites_this_gen_month: temp_site_allocations_float[site_n_dist] = ql_per_site_fallback_float
                for site_n_round, ql_float_round in temp_site_allocations_float.items():
                    site_ql_allocations_month_specific[site_n_round] = round(ql_float_round)
                current_sum_ql_after_initial_round = sum(site_ql_allocations_month_specific.values())
                diff_ql_rounding_adj = current_month_initial_ql_target - current_sum_ql_after_initial_round
                if diff_ql_rounding_adj != 0 and active_sites_this_gen_month:
                    target_site_for_diff_adj = active_sites_this_gen_month[0]
                    if temp_site_allocations_float:
                         target_site_for_diff_adj = max(temp_site_allocations_float, key=temp_site_allocations_float.get, default=active_sites_this_gen_month[0])
                    elif site_redist_scores_active:
                         best_site_cand_adj = max(site_redist_scores_active, key=site_redist_scores_active.get, default=None)
                         if best_site_cand_adj: target_site_for_diff_adj = best_site_cand_adj
                    site_ql_allocations_month_specific[target_site_for_diff_adj] += diff_ql_rounding_adj
                max_iterations_site_cap_loop = 10
                for iteration_cap_loop in range(max_iterations_site_cap_loop):
                    excess_ql_pool_iter_val = 0; newly_capped_this_iter_val = False
                    for site_n_iter_cap, allocated_qls_iter_cap in list(site_ql_allocations_month_specific.items()):
                        if site_n_iter_cap not in active_sites_this_gen_month: continue
                        site_cap_val_iter = site_caps_input.get(site_n_iter_cap, float('inf'))
                        if allocated_qls_iter_cap > site_cap_val_iter:
                            diff_iter_cap = allocated_qls_iter_cap - site_cap_val_iter
                            excess_ql_pool_iter_val += diff_iter_cap
                            site_ql_allocations_month_specific[site_n_iter_cap] = site_cap_val_iter
                            newly_capped_this_iter_val = True
                    if excess_ql_pool_iter_val < 1: break
                    candidate_sites_for_rd_list = {s: score for s, score in site_redist_scores_active.items() if s in site_ql_allocations_month_specific and site_ql_allocations_month_specific[s] < site_caps_input.get(s, float('inf'))}
                    if not candidate_sites_for_rd_list: unallocatable_this_month += round(excess_ql_pool_iter_val); break
                    total_score_candidates_rd = sum(candidate_sites_for_rd_list.values())
                    if total_score_candidates_rd <= 1e-9: unallocatable_this_month += round(excess_ql_pool_iter_val); break
                    temp_excess_after_rd_iter = excess_ql_pool_iter_val
                    sorted_candidates_for_rd_list = sorted(candidate_sites_for_rd_list.items(), key=lambda item_rd: item_rd[1], reverse=True)
                    for site_rd_val, score_rd_val in sorted_candidates_for_rd_list:
                        if temp_excess_after_rd_iter < 1: break
                        share_of_excess_raw_val = (score_rd_val / total_score_candidates_rd) * excess_ql_pool_iter_val
                        capacity_to_take_val = site_caps_input.get(site_rd_val, float('inf')) - site_ql_allocations_month_specific[site_rd_val]
                        actual_add_to_site_val = min(share_of_excess_raw_val, capacity_to_take_val, temp_excess_after_rd_iter)
                        actual_add_rounded_val = round(actual_add_to_site_val)
                        site_ql_allocations_month_specific[site_rd_val] += actual_add_rounded_val
                        temp_excess_after_rd_iter -= actual_add_rounded_val
                    excess_ql_pool_iter_val = max(0, temp_excess_after_rd_iter)
                    if excess_ql_pool_iter_val < 1 or not newly_capped_this_iter_val:
                        if excess_ql_pool_iter_val >= 1: unallocatable_this_month += round(excess_ql_pool_iter_val)
                        break
                    if iteration_cap_loop == max_iterations_site_cap_loop - 1 and excess_ql_pool_iter_val >=1 :
                        unallocatable_this_month += round(excess_ql_pool_iter_val)
            elif current_month_initial_ql_target > 0 and not active_sites_this_gen_month:
                 unallocatable_this_month = current_month_initial_ql_target

            sum_actually_allocated_qls_final = sum(site_ql_allocations_month_specific.values())
            ai_gen_df.loc[gen_month, 'Required_QLs_POF_Final'] = sum_actually_allocated_qls_final
            ai_gen_df.loc[gen_month, 'Unallocatable_QLs'] = unallocatable_this_month
            site_level_monthly_qlof[gen_month] = site_ql_allocations_month_specific.copy()
            icfs_generated_this_month_float = sum_actually_allocated_qls_final * overall_pof_to_icf_rate
            ai_gen_df.loc[gen_month, 'Generated_ICF_Mean'] = icfs_generated_this_month_float
            icfs_still_to_assign_globally -= icfs_generated_this_month_float
    else:
        if run_mode == "primary" and 'st' in globals() and hasattr(st, 'sidebar'):
            st.sidebar.error("Primary run: No valid generation months for planning.")
        if icfs_still_to_assign_globally > 1e-9:
            first_possible_landing_month_check_no_gen = proj_start_month_period + avg_lag_months_approx
            feasibility_details_no_gen = f"Goal LPI ({current_goal_lpi_month_period.strftime('%Y-%m')}) combined with lag ({effective_lag_for_planning_approx:.1f} days) results in no valid QL generation months. Min. landing: {first_possible_landing_month_check_no_gen.strftime('%Y-%m')}."
            ai_results_df_empty = pd.DataFrame(index=projection_calc_months); ai_results_df_empty['Projected_ICF_Landed'] = 0; ai_results_df_empty['Cumulative_ICF_Landed'] = 0; ai_results_df_empty['Target_QLs_POF'] = 0; ai_results_df_empty['Implied_Ad_Spend'] = 0
            return ai_results_df_empty, pd.DataFrame(), "N/A", feasibility_details_no_gen, True, 0

    total_generated_icfs_float = ai_gen_df['Generated_ICF_Mean'].sum()
    generation_goal_met_or_exceeded_in_float = (total_generated_icfs_float >= (current_goal_icf_number - 0.01))
    total_unallocated_qls_run = 0
    for gen_month_spend_idx_val in ai_gen_df.index:
        final_qls_for_cpql_calc_val = round(ai_gen_df.loc[gen_month_spend_idx_val, 'Required_QLs_POF_Final'])
        current_cpql_for_month_val = estimated_cpql_user
        if ql_vol_increase_threshold_pct > 0 and cpql_inflation_factor_pct > 0 and baseline_monthly_ql_volume > 0:
            if final_qls_for_cpql_calc_val > baseline_monthly_ql_volume:
                ql_increase_pct_val_calc = (final_qls_for_cpql_calc_val - baseline_monthly_ql_volume) / baseline_monthly_ql_volume
                threshold_units_crossed_val_calc = ql_increase_pct_val_calc / (ql_vol_increase_threshold_pct / 100.0)
                if threshold_units_crossed_val_calc > 0:
                    inflation_multiplier_val_calc = 1 + (threshold_units_crossed_val_calc * (cpql_inflation_factor_pct / 100.0))
                    current_cpql_for_month_val = estimated_cpql_user * inflation_multiplier_val_calc
        ai_gen_df.loc[gen_month_spend_idx_val, 'Adjusted_CPQL_For_Month'] = current_cpql_for_month_val
        ai_gen_df.loc[gen_month_spend_idx_val, 'Implied_Ad_Spend'] = final_qls_for_cpql_calc_val * current_cpql_for_month_val
        total_unallocated_qls_run += ai_gen_df.loc[gen_month_spend_idx_val, 'Unallocatable_QLs']

    variation_f_val = icf_variation_percent / 100.0
    ai_gen_df['Generated_ICF_Low'] = (ai_gen_df['Generated_ICF_Mean'] * (1 - variation_f_val)).round(0).astype(int).clip(lower=0)
    ai_gen_df['Generated_ICF_High'] = (ai_gen_df['Generated_ICF_Mean'] * (1 + variation_f_val)).round(0).astype(int).clip(lower=0)
    ai_gen_df['Projected_CPICF_Mean'] = (ai_gen_df['Implied_Ad_Spend'] / ai_gen_df['Generated_ICF_Mean'].replace(0, np.nan)).round(2)
    ai_gen_df['Projected_CPICF_Low'] = (ai_gen_df['Implied_Ad_Spend'] / ai_gen_df['Generated_ICF_High'].replace(0, np.nan)).round(2)
    ai_gen_df['Projected_CPICF_High'] = (ai_gen_df['Implied_Ad_Spend'] / ai_gen_df['Generated_ICF_Low'].replace(0, np.nan)).round(2)

    ai_results_df = pd.DataFrame(index=projection_calc_months); ai_results_df['Projected_ICF_Landed'] = 0.0
    for cohort_g_month_land_idx_val in ai_gen_df.index:
        icfs_gen_this_cohort_land_val = ai_gen_df.loc[cohort_g_month_land_idx_val, 'Generated_ICF_Mean']
        if icfs_gen_this_cohort_land_val <= 0: continue
        if ai_lag_method == "percentiles":
            lags_to_use = [ai_lag_p25_days, ai_lag_p50_days, ai_lag_p75_days]
            proportions = [0.25, 0.50, 0.25]
            for i, lag_days_for_share in enumerate(lags_to_use):
                icf_share = icfs_gen_this_cohort_land_val * proportions[i]
                if icf_share <= 0: continue
                full_lag_mths_share = int(np.floor(lag_days_for_share / days_in_avg_m))
                rem_lag_days_share = lag_days_for_share - (full_lag_mths_share * days_in_avg_m)
                frac_next_share = rem_lag_days_share / days_in_avg_m; frac_curr_share = 1.0 - frac_next_share
                land_m1_share = cohort_g_month_land_idx_val + full_lag_mths_share
                land_m2_share = land_m1_share + 1
                if land_m1_share in ai_results_df.index: ai_results_df.loc[land_m1_share, 'Projected_ICF_Landed'] += icf_share * frac_curr_share
                if land_m2_share in ai_results_df.index: ai_results_df.loc[land_m2_share, 'Projected_ICF_Landed'] += icf_share * frac_next_share
        else:
            full_lag_mths_land_val = int(np.floor(avg_overall_lag_days / days_in_avg_m))
            rem_lag_days_land_val = avg_overall_lag_days - (full_lag_mths_land_val * days_in_avg_m)
            frac_next_land_val = rem_lag_days_land_val / days_in_avg_m; frac_curr_land_val = 1.0 - frac_next_land_val
            land_m1_val = cohort_g_month_land_idx_val + full_lag_mths_land_val
            land_m2_val = land_m1_val + 1
            if land_m1_val in ai_results_df.index: ai_results_df.loc[land_m1_val, 'Projected_ICF_Landed'] += icfs_gen_this_cohort_land_val * frac_curr_land_val
            if land_m2_val in ai_results_df.index: ai_results_df.loc[land_m2_val, 'Projected_ICF_Landed'] += icfs_gen_this_cohort_land_val * frac_next_land_val
    ai_results_df['Projected_ICF_Landed'] = ai_results_df['Projected_ICF_Landed'].round(0).astype(int)
    ai_results_df['Cumulative_ICF_Landed'] = ai_results_df['Projected_ICF_Landed'].cumsum()
    ai_results_df['Target_QLs_POF'] = ai_gen_df['Required_QLs_POF_Initial'].reindex(ai_results_df.index).fillna(0).round(0).astype(int)
    ai_results_df['Implied_Ad_Spend'] = ai_gen_df['Implied_Ad_Spend'].reindex(ai_results_df.index).fillna(0)

    cpicf_m_res = pd.Series(index=ai_results_df.index, dtype=float); cpicf_l_res = pd.Series(index=ai_results_df.index, dtype=float); cpicf_h_res = pd.Series(index=ai_results_df.index, dtype=float)
    for g_m_cpicf_idx_val in ai_gen_df.index:
        if ai_gen_df.loc[g_m_cpicf_idx_val, 'Generated_ICF_Mean'] > 0:
            display_land_m_cpicf_val = g_m_cpicf_idx_val + avg_lag_months_approx
            if display_land_m_cpicf_val in cpicf_m_res.index and pd.isna(cpicf_m_res.loc[display_land_m_cpicf_val]):
                cpicf_m_res.loc[display_land_m_cpicf_val] = ai_gen_df.loc[g_m_cpicf_idx_val, 'Projected_CPICF_Mean']
                cpicf_l_res.loc[display_land_m_cpicf_val] = ai_gen_df.loc[g_m_cpicf_idx_val, 'Projected_CPICF_Low']
                cpicf_h_res.loc[display_land_m_cpicf_val] = ai_gen_df.loc[g_m_cpicf_idx_val, 'Projected_CPICF_High']
    ai_results_df['Projected_CPICF_Cohort_Source_Mean'] = cpicf_m_res
    ai_results_df['Projected_CPICF_Cohort_Source_Low'] = cpicf_l_res
    ai_results_df['Projected_CPICF_Cohort_Source_High'] = cpicf_h_res

    ai_gen_df['Cumulative_Generated_ICF_Final'] = ai_gen_df['Generated_ICF_Mean'].cumsum()
    ads_off_s_val = ai_gen_df[ai_gen_df['Cumulative_Generated_ICF_Final'] >= (current_goal_icf_number - 1e-9)]
    ads_off_date_str_calc_val = "Goal Not Met by End of Projection"
    if not ads_off_s_val.empty:
        ads_off_month_period_val_calc = ads_off_s_val.index[0]
        icfs_generated_by_ads_off_month_cohort_val = ai_gen_df.loc[ads_off_month_period_val_calc, 'Generated_ICF_Mean']
        cumulative_generated_before_ads_off_month_cohort_val = ai_gen_df.loc[ads_off_month_period_val_calc, 'Cumulative_Generated_ICF_Final'] - icfs_generated_by_ads_off_month_cohort_val
        icfs_needed_from_ads_off_month_cohort_val = current_goal_icf_number - cumulative_generated_before_ads_off_month_cohort_val
        if icfs_needed_from_ads_off_month_cohort_val <= 0:
            prev_month_period_ads_off_val = ads_off_month_period_val_calc - 1
            if prev_month_period_ads_off_val in ai_gen_df.index and \
               cumulative_generated_before_ads_off_month_cohort_val >= (current_goal_icf_number - 1e-9) :
                 ads_off_date_str_calc_val = prev_month_period_ads_off_val.end_time.strftime('%Y-%m-%d')
            else: ads_off_date_str_calc_val = ads_off_month_period_val_calc.start_time.strftime('%Y-%m-%d')
        elif icfs_generated_by_ads_off_month_cohort_val > 1e-9:
            fraction_of_ads_off_month_needed_val = max(0, min(1, icfs_needed_from_ads_off_month_cohort_val / icfs_generated_by_ads_off_month_cohort_val))
            ads_off_day_offset_val = int(np.ceil(fraction_of_ads_off_month_needed_val * days_in_avg_m)); ads_off_day_offset_val = max(1, ads_off_day_offset_val)
            ads_off_date_str_calc_val = (ads_off_month_period_val_calc.start_time + pd.Timedelta(days=ads_off_day_offset_val - 1)).strftime('%Y-%m-%d')
        else: ads_off_date_str_calc_val = ads_off_month_period_val_calc.start_time.strftime('%Y-%m-%d')

    ai_site_proj_df = pd.DataFrame()
    if all_defined_sites.size > 0:
        site_data_coll_ai_final_val = {site: {} for site in all_defined_sites}
        for site_n_final_init_val in all_defined_sites:
            for month_p_final_init_val in projection_calc_months:
                month_str_final_init_val = month_p_final_init_val.strftime('%Y-%m')
                site_data_coll_ai_final_val[site_n_final_init_val][(month_str_final_init_val, 'Projected QLs (POF)')] = 0
                site_data_coll_ai_final_val[site_n_final_init_val][(month_str_final_init_val, 'Projected ICFs Landed')] = 0.0
        for gen_month_site_idx_val in ai_gen_df.index:
            qlof_for_month_val = site_level_monthly_qlof.get(gen_month_site_idx_val, {})
            active_sites_for_this_gen_month_for_icf_calc = []
            for site_name_check_icf in all_defined_sites:
                is_active_for_month_icf = True
                if site_activity_schedule and site_name_check_icf in site_activity_schedule:
                    activity_info_icf = site_activity_schedule[site_name_check_icf]
                    act_pd_icf = activity_info_icf.get('activation_period')
                    deact_pd_icf = activity_info_icf.get('deactivation_period')
                    if act_pd_icf and gen_month_site_idx_val < act_pd_icf: is_active_for_month_icf = False
                    if deact_pd_icf and gen_month_site_idx_val > deact_pd_icf: is_active_for_month_icf = False
                if is_active_for_month_icf: active_sites_for_this_gen_month_for_icf_calc.append(site_name_check_icf)

            for site_n_final_val in active_sites_for_this_gen_month_for_icf_calc:
                qls_for_site_this_gen_month_val = round(qlof_for_month_val.get(site_n_final_val, 0))
                site_data_coll_ai_final_val[site_n_final_val][(gen_month_site_idx_val.strftime('%Y-%m'), 'Projected QLs (POF)')] = qls_for_site_this_gen_month_val
                site_perf_r_final_val = site_metrics_df[site_metrics_df['Site'] == site_n_final_val] if not site_metrics_df.empty else pd.DataFrame()
                site_pof_icf_rate_final_val = overall_pof_to_icf_rate
                if not site_perf_r_final_val.empty and 'Qual -> ICF %' in site_perf_r_final_val.columns:
                    rate_val_site = site_perf_r_final_val['Qual -> ICF %'].iloc[0]
                    if pd.notna(rate_val_site) and rate_val_site >= 0: site_pof_icf_rate_final_val = rate_val_site
                site_gen_icfs_this_gen_month_float = qls_for_site_this_gen_month_val * site_pof_icf_rate_final_val
                if site_gen_icfs_this_gen_month_float > 0:
                    lag_days_to_use_for_site_smear = [avg_overall_lag_days]
                    proportions_for_site_smear = [1.0]
                    if ai_lag_method == "percentiles":
                        lag_days_to_use_for_site_smear = [ai_lag_p25_days, ai_lag_p50_days, ai_lag_p75_days]
                        proportions_for_site_smear = [0.25, 0.50, 0.25]
                    for i_site_smear, lag_d_site_smear in enumerate(lag_days_to_use_for_site_smear):
                        icf_s_share_smear = site_gen_icfs_this_gen_month_float * proportions_for_site_smear[i_site_smear]
                        if icf_s_share_smear <=0: continue
                        s_f_lag_m = int(np.floor(lag_d_site_smear / days_in_avg_m))
                        s_r_lag_d = lag_d_site_smear - (s_f_lag_m * days_in_avg_m)
                        s_f_next = s_r_lag_d / days_in_avg_m; s_f_curr = 1.0 - s_f_next
                        s_l_m1 = gen_month_site_idx_val + s_f_lag_m
                        s_l_m2 = s_l_m1 + 1
                        if s_l_m1 in projection_calc_months:
                            k_l_m1_s = (s_l_m1.strftime('%Y-%m'), 'Projected ICFs Landed')
                            site_data_coll_ai_final_val[site_n_final_val][k_l_m1_s] = site_data_coll_ai_final_val[site_n_final_val].get(k_l_m1_s, 0.0) + (icf_s_share_smear * s_f_curr)
                        if s_l_m2 in projection_calc_months:
                            k_l_m2_s = (s_l_m2.strftime('%Y-%m'), 'Projected ICFs Landed')
                            site_data_coll_ai_final_val[site_n_final_val][k_l_m2_s] = site_data_coll_ai_final_val[site_n_final_val].get(k_l_m2_s, 0.0) + (icf_s_share_smear * s_f_next)
        if site_data_coll_ai_final_val:
            ai_site_proj_df = pd.DataFrame.from_dict(site_data_coll_ai_final_val, orient='index')
            if not ai_site_proj_df.empty:
                ai_site_proj_df.columns = pd.MultiIndex.from_tuples(ai_site_proj_df.columns, names=['Month', 'Metric'])
                ai_site_proj_df = ai_site_proj_df.sort_index(axis=1, level=[0,1])
                for m_fmt_site_final_val in projection_calc_months.strftime('%Y-%m'):
                    landed_col_tuple_site = (m_fmt_site_final_val, 'Projected ICFs Landed')
                    if landed_col_tuple_site in ai_site_proj_df.columns:
                         ai_site_proj_df[landed_col_tuple_site] = ai_site_proj_df[landed_col_tuple_site].round(0).astype(int)
                ai_site_proj_df = ai_site_proj_df.fillna(0)
                if not ai_site_proj_df.empty:
                    numeric_cols_site_ai_final_val = [c for c in ai_site_proj_df.columns if pd.api.types.is_numeric_dtype(ai_site_proj_df[c])]
                    if numeric_cols_site_ai_final_val:
                        total_r_ai_final_val = ai_site_proj_df[numeric_cols_site_ai_final_val].sum(axis=0)
                        total_df_ai_final_val = pd.DataFrame(total_r_ai_final_val).T; total_df_ai_final_val.index = ["Grand Total"]
                        ai_site_proj_df = pd.concat([ai_site_proj_df, total_df_ai_final_val])

    final_achieved_icfs_landed_run = ai_results_df['Cumulative_ICF_Landed'].max() if 'Cumulative_ICF_Landed' in ai_results_df and not ai_results_df.empty else 0
    goal_met_on_time_this_run = False
    actual_lpi_month_achieved_this_run = current_goal_lpi_month_period
    if not ai_results_df.empty and 'Cumulative_ICF_Landed' in ai_results_df:
        met_goal_series_val = ai_results_df[ai_results_df['Cumulative_ICF_Landed'] >= (current_goal_icf_number - 1e-9)]
        if not met_goal_series_val.empty:
            actual_lpi_month_achieved_this_run = met_goal_series_val.index.min()
            if actual_lpi_month_achieved_this_run <= current_goal_lpi_month_period :
                 goal_met_on_time_this_run = True

    significant_icfs_still_to_assign_from_gen_planning = icfs_still_to_assign_globally > 0.1
    is_unfeasible_this_run = not goal_met_on_time_this_run or \
                             total_unallocated_qls_run > 0 or \
                             significant_icfs_still_to_assign_from_gen_planning
    feasibility_prefix = "AI Projection: "
    detailed_outcome_message = ""
    effective_lpi_for_run_msg = current_goal_lpi_month_period
    effective_icf_goal_for_run_msg = current_goal_icf_number
    if run_mode == "primary":
        goal_desc_for_msg = f"Original goals ({goal_icf_number_orig} ICFs by {goal_lpi_date_dt_orig.strftime('%Y-%m-%d')})"
    else:
        goal_desc_for_msg = f"Best Case Scenario (LPI extended to {current_goal_lpi_month_period.strftime('%Y-%m')} for original {goal_icf_number_orig} ICFs goal)"
    landed_percentage_of_effective_goal = 0
    if effective_icf_goal_for_run_msg > 0: landed_percentage_of_effective_goal = final_achieved_icfs_landed_run / effective_icf_goal_for_run_msg
    near_miss_threshold_pct = 0.95
    if not is_unfeasible_this_run:
        detailed_outcome_message = f"{goal_desc_for_msg} appear ACHIEVABLE."
        if final_achieved_icfs_landed_run > effective_icf_goal_for_run_msg: detailed_outcome_message += f" (Projected to exceed goal, landing {final_achieved_icfs_landed_run:.0f} ICFs)."
        elif run_mode == "best_case_extended_lpi": detailed_outcome_message = f"{goal_desc_for_msg}: Target of {effective_icf_goal_for_run_msg} ICFs ACHIEVED by {actual_lpi_month_achieved_this_run.strftime('%Y-%m')}."
    else:
        base_unfeasible_msg_for_else = f"{goal_desc_for_msg} "
        constraint_msgs_list = []
        hard_constraints_hit_flag = False
        if total_unallocated_qls_run > 0: constraint_msgs_list.append(f"{total_unallocated_qls_run:.0f} QLs unallocatable (site caps/activity)."); hard_constraints_hit_flag = True
        if significant_icfs_still_to_assign_from_gen_planning: constraint_msgs_list.append(f"{icfs_still_to_assign_globally:.1f} ICFs (target: {effective_icf_goal_for_run_msg}) could not be fully planned in generation phase."); hard_constraints_hit_flag = True
        if hard_constraints_hit_flag: detailed_outcome_message = base_unfeasible_msg_for_else + "appear UNFEASIBLE due to planning constraints: " + " ".join(constraint_msgs_list)
        elif not goal_met_on_time_this_run:
            achieved_by_date_str_msg = actual_lpi_month_achieved_this_run.strftime('%Y-%m') if final_achieved_icfs_landed_run > 0 and pd.notna(actual_lpi_month_achieved_this_run) else 'end of projection'
            target_lpi_str_msg = effective_lpi_for_run_msg.strftime('%Y-%m')
            if landed_percentage_of_effective_goal >= near_miss_threshold_pct and actual_lpi_month_achieved_this_run <= effective_lpi_for_run_msg:
                detailed_outcome_message = base_unfeasible_msg_for_else + f"LANDING GOAL NEAR MISS. Projected {final_achieved_icfs_landed_run:.0f} ICFs ({landed_percentage_of_effective_goal*100:.1f}%) by {achieved_by_date_str_msg} (Target LPI: {target_lpi_str_msg})."
                if generation_goal_met_or_exceeded_in_float and final_achieved_icfs_landed_run < effective_icf_goal_for_run_msg: detailed_outcome_message += " (Note: Goal met in total ICFs generated; landed sum slightly short due to monthly rounding)."
            else:
                if actual_lpi_month_achieved_this_run > effective_lpi_for_run_msg and final_achieved_icfs_landed_run >= effective_icf_goal_for_run_msg : detailed_outcome_message = base_unfeasible_msg_for_else + f"LPI NOT MET. Goal of {effective_icf_goal_for_run_msg} ICFs achieved by {achieved_by_date_str_msg}, which is after target LPI of {target_lpi_str_msg}."
                else: detailed_outcome_message = base_unfeasible_msg_for_else + f"LANDING GOAL SHORTFALL/LPI MISSED. Projected {final_achieved_icfs_landed_run:.0f} ICFs ({landed_percentage_of_effective_goal*100:.1f}%) by {achieved_by_date_str_msg} (Target: {effective_icf_goal_for_run_msg} ICFs by {target_lpi_str_msg})."
        else:
            detailed_outcome_message = base_unfeasible_msg_for_else + "appear UNFEASIBLE for other reasons."
            if total_unallocated_qls_run > 0 and not hard_constraints_hit_flag: detailed_outcome_message += f" Minor QL unallocation: {total_unallocated_qls_run:.0f}."
            if icfs_still_to_assign_globally > 1e-5 and not significant_icfs_still_to_assign_from_gen_planning and not hard_constraints_hit_flag: detailed_outcome_message += f" Small remainder of {icfs_still_to_assign_globally:.2f} ICFs from generation."
    feasibility_msg_final_display = feasibility_prefix + detailed_outcome_message.strip()

    display_end_month_final_val = projection_calc_months[-1] if not projection_calc_months.empty else proj_start_month_period
    if not ai_results_df.empty and 'Cumulative_ICF_Landed' in ai_results_df:
        met_goal_series_trim_val = ai_results_df[ai_results_df['Cumulative_ICF_Landed'] >= (current_goal_icf_number - 1e-9)]
        if not met_goal_series_trim_val.empty:
            lpi_achieved_month_for_trim_val_calc = met_goal_series_trim_val.index.min()
            try:
                candidate_end_month_val_ts_calc = lpi_achieved_month_for_trim_val_calc.to_timestamp() + pd.offsets.MonthEnd(3)
                candidate_end_month_val_calc = candidate_end_month_val_ts_calc.to_period('M')
                display_end_month_final_val = min(projection_calc_months[-1], max(candidate_end_month_val_calc, current_goal_lpi_month_period + 3))
            except Exception: pass
    ai_results_df_final_display_val = pd.DataFrame()
    if not ai_results_df.empty and proj_start_month_period <= display_end_month_final_val:
        try: ai_results_df_final_display_val = ai_results_df.loc[proj_start_month_period:display_end_month_final_val].copy()
        except Exception: ai_results_df_final_display_val = ai_results_df.copy()
    if 'st' in globals() and hasattr(st, 'session_state'):
        if run_mode == "primary": st.session_state.ai_gen_df_debug_primary = ai_gen_df.copy(); st.session_state.ai_results_df_debug_primary = ai_results_df.copy()
        elif run_mode == "best_case_extended_lpi": st.session_state.ai_gen_df_debug_best_case = ai_gen_df.copy(); st.session_state.ai_results_df_debug_best_case = ai_results_df.copy()
    return ai_results_df_final_display_val, ai_site_proj_df, ads_off_date_str_calc_val, feasibility_msg_final_display, is_unfeasible_this_run, final_achieved_icfs_landed_run

@st.cache_data
def calculate_pipeline_projection(
    _processed_df, ordered_stages, ts_col_map, inter_stage_lags,
    conversion_rates, lag_assumption_model
):
    """
    Calculates the future ICFs and Enrollments, and also returns the total theoretical yield.
    """
    default_return = {
        'results_df': pd.DataFrame(),
        'total_icf_yield': 0,
        'total_enroll_yield': 0,
        'in_flight_df_for_narrative': pd.DataFrame()
    }
    if _processed_df is None or _processed_df.empty:
        return default_return

    # --- 1. Define TRUE Terminal Stages & Filter for In-Flight Leads ---
    true_terminal_stages = [s for s in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST] if s in ts_col_map]
    true_terminal_ts_cols = [ts_col_map.get(s) for s in true_terminal_stages]
    in_flight_df = _processed_df.copy()
    for ts_col in true_terminal_ts_cols:
        if ts_col in in_flight_df.columns:
            in_flight_df = in_flight_df[in_flight_df[ts_col].isna()]
    if in_flight_df.empty:
        st.info("Funnel Analysis: No leads are currently in-flight.")
        return default_return

    # --- 2. Determine Current Stage ---
    def get_current_stage(row, ordered_stages, ts_col_map):
        last_stage, last_ts = None, pd.NaT
        for stage in ordered_stages:
            if stage in true_terminal_stages: continue
            ts_col = ts_col_map.get(stage)
            if ts_col and ts_col in row and pd.notna(row[ts_col]):
                if pd.isna(last_ts) or row[ts_col] > last_ts:
                    last_ts, last_stage = row[ts_col], stage
        return last_stage, last_ts
    in_flight_df[['current_stage', 'current_stage_ts']] = in_flight_df.apply(
        lambda row: get_current_stage(row, ordered_stages, ts_col_map), axis=1, result_type='expand'
    )
    in_flight_df.dropna(subset=['current_stage'], inplace=True)

    # --- 3. Create the Master Pool of ALL In-Flight ICFs (Existing and Projected) ---
    all_icfs_to_project = []
    ts_icf_col = ts_col_map.get(STAGE_SIGNED_ICF)
    if ts_icf_col in in_flight_df.columns:
        already_icf_in_flight = in_flight_df[in_flight_df[ts_icf_col].notna()].copy()
        for _, row in already_icf_in_flight.iterrows():
            all_icfs_to_project.append({'prob': 1.0, 'lag_to_icf': 0.0, 'start_date': row[ts_icf_col]})
        leads_before_icf = in_flight_df[in_flight_df[ts_icf_col].isna()].copy()
    else:
        leads_before_icf = in_flight_df.copy()
        
    for _, row in leads_before_icf.iterrows():
        prob_to_icf, lags_to_icf_list, path_found = 1.0, [], False
        start_index = ordered_stages.index(row['current_stage'])
        for i in range(start_index, len(ordered_stages) - 1):
            from_stage, to_stage = ordered_stages[i], ordered_stages[i+1]
            prob_to_icf *= conversion_rates.get(f"{from_stage} -> {to_stage}", 0.0)
            lags_to_icf_list.append(inter_stage_lags.get(f"{from_stage} -> {to_stage}", 0.0))
            if to_stage == STAGE_SIGNED_ICF: path_found = True; break
        if path_found and prob_to_icf > 0:
            total_lag_to_icf = np.nansum(lags_to_icf_list)
            all_icfs_to_project.append({'prob': prob_to_icf, 'lag_to_icf': total_lag_to_icf, 'start_date': row['current_stage_ts']})

    # --- 4. Project Enrollments from the Master Pool of ICFs ---
    projected_enrollments = []
    icf_to_enroll_rate = conversion_rates.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0.0)
    icf_to_enroll_lag = inter_stage_lags.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0.0)
    if pd.isna(icf_to_enroll_lag): icf_to_enroll_lag = 0.0
    for icf in all_icfs_to_project:
        if pd.isna(icf['start_date']) or pd.isna(icf['lag_to_icf']): continue
        icf_landing_date = icf['start_date'] + pd.to_timedelta(icf['lag_to_icf'], unit='D')
        if pd.isna(icf_landing_date): continue
        projected_enrollments.append({'prob': icf['prob'] * icf_to_enroll_rate, 'lag': icf_to_enroll_lag, 'start_date': icf_landing_date})

    # --- 5. Calculate TOTAL THEORETICAL YIELD (the true number) ---
    total_icf_yield = sum(p['prob'] for p in all_icfs_to_project)
    total_enroll_yield = sum(p['prob'] for p in projected_enrollments)

    # --- 6. Dynamically size the results dataframe for FUTURE projections ---
    all_landing_dates = [p['start_date'] + pd.to_timedelta(p['lag_to_icf'], 'D') for p in all_icfs_to_project if pd.notna(p.get('start_date')) and pd.notna(p.get('lag_to_icf'))]
    all_landing_dates.extend([p['start_date'] + pd.to_timedelta(p['lag'], 'D') for p in projected_enrollments if pd.notna(p.get('start_date')) and pd.notna(p.get('lag'))])

    proj_start_month = pd.Period(datetime.now(), 'M')
    valid_dates = [d for d in all_landing_dates if pd.notna(d)]
    if not valid_dates:
        future_months = pd.period_range(start=proj_start_month, periods=6, freq='M')
    else:
        max_date = max(valid_dates)
        proj_end_month = max(pd.Period(max_date, 'M'), proj_start_month)
        future_months = pd.period_range(start=proj_start_month, end=proj_end_month, freq='M')

    results_df = pd.DataFrame(0.0, index=future_months, columns=['Projected_ICF_Landed', 'Projected_Enrollments_Landed'])

    # --- 7. Aggregate and Smear Projections into the FUTURE table ---
    def smear_projection(df, projections_list, lag_col_name, target_col):
        for proj in projections_list:
            if proj['prob'] <= 0 or pd.isna(proj['start_date']) or pd.isna(proj[lag_col_name]): continue
            landing_date = proj['start_date'] + pd.to_timedelta(proj[lag_col_name], unit='D')
            landing_period = pd.Period(landing_date, 'M')
            if landing_period in df.index:
                df.loc[landing_period, target_col] += proj['prob']
        return df

    results_df = smear_projection(results_df, all_icfs_to_project, 'lag_to_icf', 'Projected_ICF_Landed')
    results_df = smear_projection(results_df, projected_enrollments, 'lag', 'Projected_Enrollments_Landed')

    results_df['Projected_ICF_Landed'] = results_df['Projected_ICF_Landed'].round(0).astype(int)
    results_df['Cumulative_ICF_Landed'] = results_df['Projected_ICF_Landed'].cumsum()
    results_df['Projected_Enrollments_Landed'] = results_df['Projected_Enrollments_Landed'].round(0).astype(int)
    results_df['Cumulative_Enrollments_Landed'] = results_df['Cumulative_Enrollments_Landed'].cumsum()

    return {
        'results_df': results_df,
        'total_icf_yield': total_icf_yield,
        'total_enroll_yield': total_enroll_yield,
        'in_flight_df_for_narrative': in_flight_df
    }


def generate_funnel_narrative(in_flight_df, ordered_stages, conversion_rates, inter_stage_lags):
    """
    Generates a list of dictionaries, each representing a step in the funnel narrative.
    NOW includes cumulative downstream projections with cumulative lags.
    """
    if in_flight_df.empty or not inter_stage_lags:
        return []

    narrative_steps = []
    stage_counts = in_flight_df['current_stage'].value_counts().to_dict()

    for i, stage_name in enumerate(ordered_stages):
        if i >= len(ordered_stages) - 1 or stage_name == STAGE_ENROLLED:
            break

        leads_at_this_stage = stage_counts.get(stage_name, 0)

        downstream_projections = []
        cumulative_prob = 1.0
        cumulative_lag = 0.0 # This will track time from the start_stage

        # Project from the *next* stage onwards
        for j in range(i + 1, len(ordered_stages)):
            from_stage = ordered_stages[j-1]
            to_stage = ordered_stages[j]

            step_rate_key = f"{from_stage} -> {to_stage}"
            step_rate = conversion_rates.get(step_rate_key)

            lag_key = f"{from_stage} -> {to_stage}"
            step_lag = inter_stage_lags.get(lag_key)

            if step_rate is None:
                break

            cumulative_prob *= step_rate

            # Add the lag for this step to the cumulative total
            if pd.notna(step_lag):
                cumulative_lag += step_lag

            projected_count = leads_at_this_stage * cumulative_prob

            downstream_projections.append({
                "stage_name": to_stage,
                "projected_count": projected_count,
                # Store the cumulative lag for this projection step
                "cumulative_lag_days": cumulative_lag if pd.notna(step_lag) else None,
            })

        next_stage_name = ordered_stages[i+1]
        rate_key = f"{stage_name} -> {next_stage_name}"
        lag_key = f"{stage_name} -> {next_stage_name}"

        step_data = {
            "current_stage": stage_name,
            "leads_at_stage": leads_at_this_stage,
            "next_stage": next_stage_name,
            "conversion_rate": conversion_rates.get(rate_key),
            "lag_to_next_stage": inter_stage_lags.get(lag_key),
            "downstream_projections": downstream_projections,
        }
        narrative_steps.append(step_data)

    return narrative_steps