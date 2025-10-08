# pages/8_PC_Performance.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, time
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

from pc_calculations import (
    calculate_heatmap_data, 
    calculate_average_time_metrics, 
    calculate_top_status_flows,
    calculate_ttfc_effectiveness,
    calculate_contact_attempt_effectiveness,
    calculate_performance_over_time,
    analyze_heatmap_efficiency # Import new function
)
from helpers import format_days_to_dhm

st.set_page_config(page_title="PC Performance", page_icon="📞", layout="wide")

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("📞 PC Performance Dashboard")
st.info("This dashboard analyzes the operational efficiency and patterns of the Pre-Screening team's activities.")

if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

processed_data = st.session_state.referral_data_processed
ts_col_map = st.session_state.ts_col_map
parsed_status_history_col = "Parsed_Lead_Status_History" 

st.divider()
submission_date_col = "Submitted On_DT" 
if submission_date_col in processed_data.columns:
    min_date = processed_data[submission_date_col].min().date()
    max_date = processed_data[submission_date_col].max().date()
    with st.container(border=True):
        st.subheader("Filter Data by Submission Date")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
        st.stop()
    start_datetime = datetime.combine(start_date, time.min)
    end_datetime = datetime.combine(end_date, time.max)
    filtered_df = processed_data[
        (processed_data[submission_date_col] >= start_datetime) &
        (processed_data[submission_date_col] <= end_datetime)
    ].copy()
    st.metric(label="Total Referrals in Selected Range", value=f"{len(filtered_df):,}")
else:
    st.warning(f"Date column '{submission_date_col}' not found. Cannot apply date filter.")
    filtered_df = processed_data
st.divider()

with st.spinner("Analyzing status histories for PC activity in selected date range..."):
    if parsed_status_history_col not in filtered_df.columns:
        st.error(f"The required column '{parsed_status_history_col}' was not found in the processed data.")
        st.stop()

    contact_heatmap, sts_heatmap = calculate_heatmap_data(filtered_df, ts_col_map, parsed_status_history_col)
    time_metrics = calculate_average_time_metrics(filtered_df, ts_col_map, parsed_status_history_col)
    top_flows = calculate_top_status_flows(filtered_df, ts_col_map, parsed_status_history_col)
    ttfc_df = calculate_ttfc_effectiveness(filtered_df, ts_col_map)
    attempt_effectiveness_df = calculate_contact_attempt_effectiveness(filtered_df, ts_col_map, parsed_status_history_col)
    over_time_df = calculate_performance_over_time(filtered_df, ts_col_map)
    # --- NEW: Call the analysis function ---
    heatmap_insights = analyze_heatmap_efficiency(contact_heatmap, sts_heatmap)

st.header("Activity Heatmaps")
st.markdown("Visualizing when key activities occur during the week.")
col1, col2 = st.columns(2)
with col1, st.container(border=True):
    st.subheader("Pre-StS Contact Attempts by Time of Day")
    if not contact_heatmap.empty and contact_heatmap.values.sum() > 0:
        fig1 = px.imshow(contact_heatmap, labels=dict(x="Hour of Day", y="Day of Week", color="Contacts"), aspect="auto", color_continuous_scale="Mint")
        fig1.update_layout(xaxis_nticks=12)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No pre-StS contact attempt data found in the selected date range.")
with col2, st.container(border=True):
    st.subheader("Sent To Site Events by Time of Day")
    if not sts_heatmap.empty and sts_heatmap.values.sum() > 0:
        fig2 = px.imshow(sts_heatmap, labels=dict(x="Hour of Day", y="Day of Week", color="Events"), aspect="auto", color_continuous_scale="Mint")
        fig2.update_layout(xaxis_nticks=12)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No 'Sent To Site' data found in the selected date range.")

# --- NEW SECTION: Display Heatmap Insights ---
st.header("Strategic Contact Insights")
st.markdown("Based on an analysis of contact attempts vs. successful 'Sent to Site' outcomes.")

if not heatmap_insights:
    st.info("Not enough data to generate strategic contact insights.")
else:
    insight_cols = st.columns(3)
    with insight_cols[0], st.container(border=True, height=250):
        st.subheader("📈 Best for Volume")
        st.caption("Times with high contact volume that also result in high 'Sent to Site' volume.")
        if heatmap_insights.get("volume_best"):
            for item in heatmap_insights["volume_best"]:
                st.markdown(f"- **{item}**")
        else:
            st.write("No distinct high-volume/high-success time slots found.")
            
    with insight_cols[1], st.container(border=True, height=250):
        st.subheader("🎯 Most Efficient")
        st.caption("Times with the best conversion of contacts to 'Sent to Site'.")
        if heatmap_insights.get("most_efficient"):
            for item in heatmap_insights["most_efficient"]:
                st.markdown(f"- **{item}**")
        else:
            st.write("No distinct high-efficiency time slots found.")

    with insight_cols[2], st.container(border=True, height=250):
        st.subheader("⚠️ Least Efficient")
        st.caption("Times with high contact volume but low 'Sent to Site' outcomes.")
        if heatmap_insights.get("least_efficient"):
            for item in heatmap_insights["least_efficient"]:
                st.markdown(f"- **{item}**")
        else:
            st.write("No distinct low-efficiency time slots found.")

st.divider()

st.header("Key Performance Indicators")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1, st.container(border=True):
    value = time_metrics.get('avg_time_to_first_contact')
    st.metric(label="Average Time to First Contact", value=format_days_to_dhm(value))
with kpi_col2, st.container(border=True):
    value = time_metrics.get('avg_time_between_contacts')
    st.metric(label="Average Time Between Contact Attempts", value=format_days_to_dhm(value))
with kpi_col3, st.container(border=True):
    value = time_metrics.get('avg_time_new_to_sts')
    st.metric(label="Average Time from New to Sent To Site", value=format_days_to_dhm(value))

st.divider()

st.header("Top 5 Common Status Flows to 'Sent to Site'")
if not top_flows:
    st.info("There is not enough data in the selected date range to determine common status flows.")
else:
    with st.container(border=True):
        for i, (path, count) in enumerate(top_flows):
            st.markdown(f"**{i+1}. Most Common Path** ({count} referrals)")
            st.info(f"`{path}`")
            if i < len(top_flows) - 1:
                st.divider()

st.divider()

st.header("Time to First Contact Effectiveness")
st.markdown("Analyzes how the speed of the first contact impacts downstream funnel conversions.")
if ttfc_df.empty or ttfc_df['Attempts'].sum() == 0:
    st.info("Not enough data in the selected date range to analyze the effectiveness of first contact timing.")
else:
    display_df = ttfc_df.copy()
    display_df['StS Rate'] = display_df['StS_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df['ICF Rate'] = display_df['ICF_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df['Enrollment Rate'] = display_df['Enrollment_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df.rename(columns={'Total_StS': 'Total Sent to Site', 'Total_ICF': 'Total ICFs', 'Total_Enrolled': 'Total Enrollments'}, inplace=True)
    final_cols = ['Time to First Contact', 'Attempts', 'Total Sent to Site', 'StS Rate', 'Total ICFs', 'ICF Rate', 'Total Enrollments', 'Enrollment Rate']
    with st.container(border=True):
        st.dataframe(display_df[final_cols], hide_index=True, use_container_width=True)

st.divider()

st.header("Contact Attempt Effectiveness")
st.markdown("Analyzes how the number of pre-site status changes impacts downstream funnel conversions.")
if (attempt_effectiveness_df.empty or 
    'Total Referrals' not in attempt_effectiveness_df.columns or 
    attempt_effectiveness_df['Total Referrals'].sum() == 0):
    st.info("Not enough data in the selected date range to analyze the effectiveness of contact attempts.")
else:
    display_df = attempt_effectiveness_df.copy()
    display_df['StS Rate'] = display_df['StS_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df['ICF Rate'] = display_df['ICF_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df['Enrollment Rate'] = display_df['Enrollment_Rate'].map('{:.1%}'.format).replace('nan%', '-')
    display_df.rename(columns={'Total_StS': 'Total Sent to Site', 'Total_ICF': 'Total ICFs', 'Total_Enrolled': 'Total Enrollments'}, inplace=True)
    final_cols = ['Number of Attempts', 'Total Referrals', 'Total Sent to Site', 'StS Rate', 'Total ICFs', 'ICF Rate', 'Total Enrollments', 'Enrollment Rate']
    with st.container(border=True):
        st.dataframe(display_df[final_cols], hide_index=True, use_container_width=True)

st.divider()

st.header("Performance Over Time (Weekly)")
st.markdown("Track key metrics on a weekly basis to identify trends.")

if over_time_df.empty:
    st.info("Not enough data in the selected date range to generate a performance trend graph.")
else:
    with st.container(border=True):
        secondary_metric = 'Total Qualified per Week'
        
        primary_metric_options = [col for col in over_time_df.columns if col != secondary_metric]
        primary_metric = st.selectbox(
            "Select a primary metric to display on the chart:",
            options=primary_metric_options
        )
        
        compare_with_volume = st.toggle(f"Compare with {secondary_metric}", value=True)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=over_time_df.index, y=over_time_df[primary_metric], name=primary_metric, 
                       line=dict(color='#53CA97')),
            secondary_y=False,
        )

        if compare_with_volume:
            fig.add_trace(
                go.Scatter(x=over_time_df.index, y=over_time_df[secondary_metric], name=secondary_metric, line=dict(dash='dot', color='gray')),
                secondary_y=True,
            )

        fig.update_yaxes(title_text=f"<b>{primary_metric}</b>", secondary_y=False)
        if compare_with_volume:
            fig.update_yaxes(title_text=f"<b>{secondary_metric}</b>", secondary_y=True, showgrid=False)

        fig.update_layout(
            title_text=f"Weekly Trend: {primary_metric}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)