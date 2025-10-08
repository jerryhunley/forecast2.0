# pages/2_Site_Performance.py
import streamlit as st
import pandas as pd
from scoring import score_sites
from helpers import format_performance_df, format_days_to_dhm
from calculations import (
    calculate_site_operational_kpis, 
    calculate_site_ttfc_effectiveness, 
    calculate_site_contact_attempt_effectiveness,
    calculate_site_performance_over_time,
    calculate_lost_reasons_after_sts
)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Site Performance", page_icon="🏆", layout="wide")

if 'ranked_sites_df' not in st.session_state:
    st.session_state.ranked_sites_df = pd.DataFrame()

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("🏆 Site Performance Dashboard")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

with st.container(border=True):
    st.subheader("Site Operational Analysis")
    st.markdown("Analyze site-level efficiency. Select a specific site or view the overall average for all metrics in this section.")

    site_list = st.session_state.enhanced_site_metrics_df['Site'].unique().tolist()
    site_list = [site for site in site_list if site != "Unassigned Site"]
    options = ["Overall"] + sorted(site_list)
    
    selected_site = st.selectbox(
        "Select a Site to Analyze (or Overall)", 
        options=options,
        key="site_kpi_selector"
    )

    site_kpis = calculate_site_operational_kpis(
        st.session_state.referral_data_processed,
        st.session_state.ts_col_map,
        "Parsed_Lead_Status_History",
        selected_site
    )
    lost_reasons = calculate_lost_reasons_after_sts(
        st.session_state.referral_data_processed,
        st.session_state.ts_col_map,
        "Parsed_Lead_Status_History",
        st.session_state.funnel_definition,
        selected_site
    )

    kpi_cols = st.columns(3)
    kpi_cols[0].metric(
        label="Avg. Time to First Site Action",
        value=format_days_to_dhm(site_kpis.get('avg_sts_to_first_action')),
        help="Time from when a lead is 'Sent To Site' until the site takes any follow-up action (e.g., status change, appointment scheduled)."
    )
    kpi_cols[1].metric(
        label="Avg. Time Between Site Contacts",
        value=format_days_to_dhm(site_kpis.get('avg_time_between_site_contacts')),
        help="The average time between explicit 'Contact Attempts' made by a site before an appointment is scheduled."
    )
    kpi_cols[2].metric(
        label="Avg. Time StS to Appt. Sched.",
        value=format_days_to_dhm(site_kpis.get('avg_sts_to_appt')),
        help="The average total time from when a lead is 'Sent to Site' until an appointment is successfully scheduled."
    )
    
    st.divider()
    st.subheader(f"Time to First Action Effectiveness: {selected_site}")
    
    site_effectiveness_df = calculate_site_ttfc_effectiveness(st.session_state.referral_data_processed, st.session_state.ts_col_map, selected_site)
    if site_effectiveness_df.empty or site_effectiveness_df['Attempts'].sum() == 0:
        st.info(f"Not enough data for '{selected_site}' to analyze first action effectiveness.")
    else:
        display_df = site_effectiveness_df.copy()
        display_df['Appt. Rate'] = display_df['Appt_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df['ICF Rate'] = display_df['ICF_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df['Enrollment Rate'] = display_df['Enrollment_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df.rename(columns={'Attempts': 'Total Referrals', 'Total_Appts': 'Total Appointments', 'Total_ICF': 'Total ICFs', 'Total_Enrolled': 'Total Enrollments'}, inplace=True)
        final_cols = ['Time to First Site Action', 'Total Referrals', 'Total Appointments', 'Appt. Rate', 'Total ICFs', 'ICF Rate', 'Total Enrollments', 'Enrollment Rate']
        st.dataframe(display_df[final_cols], hide_index=True, use_container_width=True)
    
    st.divider()
    st.subheader(f"Contact Attempt Effectiveness: {selected_site}")
    
    site_contact_effectiveness_df = calculate_site_contact_attempt_effectiveness(st.session_state.referral_data_processed, st.session_state.ts_col_map, "Parsed_Lead_Status_History", selected_site)
    if site_contact_effectiveness_df.empty or site_contact_effectiveness_df['Total Referrals'].sum() == 0:
        st.info(f"Not enough data for '{selected_site}' to analyze contact attempt effectiveness.")
    else:
        display_df_contact = site_contact_effectiveness_df.copy()
        display_df_contact['Appt. Rate'] = display_df_contact['Appt_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df_contact['ICF Rate'] = display_df_contact['ICF_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df_contact['Enrollment Rate'] = display_df_contact['Enrollment_Rate'].map('{:.1%}'.format).replace('nan%', '-')
        display_df_contact.rename(columns={'Total_Appts': 'Total Appointments', 'Total_ICF': 'Total ICFs', 'Total_Enrolled': 'Total Enrollments'}, inplace=True)
        final_cols_contact = ['Number of Site Attempts', 'Total Referrals', 'Total Appointments', 'Appt. Rate', 'Total ICFs', 'ICF Rate', 'Total Enrollments', 'Enrollment Rate']
        st.dataframe(display_df_contact[final_cols_contact], hide_index=True, use_container_width=True)
    
    st.divider()
    st.subheader(f"Lost Reasons (After Sent to Site) for {selected_site}")
    
    chart_col1, _ = st.columns([2, 1]) 
    with chart_col1:
        if not lost_reasons.empty:
            brand_color_sequence = ['#53CA97', '#7991C6', '#F4A261', '#E76F51', '#2A9D8F', '#E9C46A', '#A2D2FF', '#FFB703']
            fig = px.pie(
                values=lost_reasons.values, 
                names=lost_reasons.index,
                title=f"Breakdown of {lost_reasons.sum()} Lost Referrals",
                color_discrete_sequence=brand_color_sequence
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No referrals were marked as 'Lost' after being sent to {selected_site}.")

    st.divider()
    st.subheader(f"Performance Over Time (Weekly): {selected_site}")
    
    over_time_df = calculate_site_performance_over_time(st.session_state.referral_data_processed, st.session_state.ts_col_map, "Parsed_Lead_Status_History", selected_site)
    if over_time_df.empty:
        st.info(f"Not enough data for '{selected_site}' to generate a performance trend graph.")
    else:
        secondary_metric = 'Total Sent to Site per Week'
        primary_metric_options = {
            'Sent to Site -> Appointment %': ('Sent to Site -> Appointment % (Actual)', 'Sent to Site -> Appointment % (Projected)'),
            'Sent to Site -> ICF %': ('Sent to Site -> ICF % (Actual)', 'Sent to Site -> ICF % (Projected)'),
            'Sent to Site -> Enrollment %': ('Sent to Site -> Enrollment % (Actual)', 'Sent to Site -> Enrollment % (Projected)'),
            'Total Appointments per Week': ('Total Appointments per Week', None),
            'Average Time to First Site Action (Days)': ('Average Time to First Site Action (Days)', None),
        }
        available_options = [opt for opt, cols in primary_metric_options.items() if cols[0] in over_time_df.columns]
        
        if available_options:
            primary_metric_display_name = st.selectbox("Select a primary metric to display on the chart:", options=available_options, key=f"site_perf_time_selector_{selected_site.replace(' ', '_')}")
            compare_with_volume = st.toggle(f"Compare with {secondary_metric}", value=True, key=f"site_perf_time_toggle_{selected_site.replace(' ', '_')}")
            actual_col, projected_col = primary_metric_options[primary_metric_display_name]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=over_time_df.index, y=over_time_df[actual_col], name=primary_metric_display_name, mode='lines+markers', line=dict(color='#53CA97')), secondary_y=False)
            if projected_col and projected_col in over_time_df.columns:
                fig.add_trace(go.Scatter(x=over_time_df.index, y=over_time_df[projected_col], name="Projected Trend", mode='lines', line=dict(color='#53CA97', dash='dot'), showlegend=False), secondary_y=False)
            if compare_with_volume and secondary_metric in over_time_df.columns:
                fig.add_trace(go.Scatter(x=over_time_df.index, y=over_time_df[secondary_metric], name=secondary_metric, line=dict(dash='dot', color='gray')), secondary_y=True)
            fig.update_yaxes(title_text=f"<b>{primary_metric_display_name}</b>", secondary_y=False)
            if compare_with_volume and secondary_metric in over_time_df.columns:
                fig.update_yaxes(title_text=f"<b>{secondary_metric}</b>", secondary_y=True, showgrid=False)
            fig.update_layout(title_text=f"Weekly Trend for {selected_site}: {primary_metric_display_name}", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

st.divider()

with st.expander("Adjust Site Performance Scoring Weights"):
    st.markdown("Adjust the importance of different metrics in the overall site score. Changes here do not affect the Ad Performance page.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Conversion Weights")
        st.slider("StS -> Enrollment %", 0, 100, key="w_site_sts_to_enr")
        st.slider("ICF -> Enrollment %", 0, 100, key="w_site_icf_to_enroll")
        st.slider("StS -> ICF %", 0, 100, key="w_site_sts_to_icf")
        st.slider("StS -> Appt %", 0, 100, key="w_site_sts_appt")
        st.slider("StS Contact Rate %", 0, 100, key="w_site_contact_rate")
    with c2:
        st.subheader("Speed / Lag Weights")
        st.markdown("_Lower is better for these metrics._")
        st.slider("Average time to first site action", 0, 100, key="w_site_avg_time_to_first_action")
        st.slider("Avg time from StS to Appt Sched.", 0, 100, key="w_site_lag_sts_appt")
        st.slider("Avg. Time Between Site Contacts", 0, 100, key="w_site_avg_time_between_contacts")
        st.slider("Avg time from StS to ICF", 0, 100, key="w_site_lag_sts_icf")
        st.slider("Total Referrals Awaiting First Site Action", 0, 100, key="w_site_awaiting_action")
    with c3:
        st.subheader("Negative Outcome Weights")
        st.markdown("_Lower is better for these metrics._")
        st.slider("SF or Lost After ICF %", 0, 100, key="w_site_icf_to_lost")
        st.slider("StS to Lost %", 0, 100, key="w_site_sts_to_lost")

    if st.button("Apply & Recalculate Score", type="primary", use_container_width=True, key="apply_site_weights"):
        weights = {
            "StS to Enrollment %": st.session_state.w_site_sts_to_enr,
            "ICF to Enrollment %": st.session_state.w_site_icf_to_enroll,
            "StS to ICF %": st.session_state.w_site_sts_to_icf,
            "StS to Appt %": st.session_state.w_site_sts_appt,
            "StS Contact Rate %": st.session_state.w_site_contact_rate,
            "Average time to first site action": st.session_state.w_site_avg_time_to_first_action,
            "Avg time from StS to Appt Sched.": st.session_state.w_site_lag_sts_appt,
            "Avg. Time Between Site Contacts": st.session_state.w_site_avg_time_between_contacts,
            "Avg time from StS to ICF": st.session_state.w_site_lag_sts_icf,
            "Total Referrals Awaiting First Site Action": st.session_state.w_site_awaiting_action,
            "SF or Lost After ICF %": st.session_state.w_site_icf_to_lost,
            "StS to Lost %": st.session_state.w_site_sts_to_lost,
            'Qualified to Enrollment %': st.session_state.w_site_qual_to_enroll,
            'Qualified to ICF %': st.session_state.w_site_qual_to_icf,
        }
        total_weight = sum(abs(w) for w in weights.values())
        weights_normalized = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {}
        
        # --- FIX: Read pre-calculated DF from state ---
        if not st.session_state.enhanced_site_metrics_df.empty:
            st.session_state.ranked_sites_df = score_sites(st.session_state.enhanced_site_metrics_df, weights_normalized)
        else:
            st.session_state.ranked_sites_df = pd.DataFrame()

with st.container(border=True):
    st.subheader("Site Performance Ranking")
    
    if not st.session_state.ranked_sites_df.empty:
        display_cols = [
            'Site', 'Score', 'Grade', 'Total Qualified', 'Pre-Screening Activities Count', 'StS Count', 'Appt Count', 'ICF Count', 'Enrollment Count', 
            'SF or Lost After ICF Count', 'Lost After StS', 'Total Lost Count', 'Total Referrals Awaiting First Site Action', 
            'Average time to first site action', 'Avg. Time Between Site Contacts', 'Avg number of site contact attempts per referral', 'StS Contact Rate %', 
            'StS to Appt %', 'StS to ICF %', 'StS to Enrollment %', 'StS to Lost %', 'ICF to Enrollment %', 'SF or Lost After ICF %', 
            'Avg time from StS to Appt Sched.', 'Avg time from StS to ICF', 'Avg time from StS to Enrollment',
            'Qualified to StS %', 'Qualified to Appt %', 'Qualified to ICF %', 'Qualified to Enrollment %'
        ]
        
        display_cols_exist = [col for col in display_cols if col in st.session_state.ranked_sites_df.columns]
        
        final_display_df = st.session_state.ranked_sites_df[display_cols_exist]
        rename_map = { 
            'Pre-Screening Activities Count': 'PSA Count',
            'SF or Lost After ICF Count': 'SF/Lost Post-ICF',
            'SF or Lost After ICF %': 'SF/Lost Post-ICF %'
        }
        final_display_df = final_display_df.rename(columns=rename_map)
        
        formatted_df = format_performance_df(final_display_df)
        st.dataframe(formatted_df, hide_index=True, use_container_width=True)
    else:
        st.info("Adjust weights and click 'Apply & Recalculate Score' to generate the ranking table.")