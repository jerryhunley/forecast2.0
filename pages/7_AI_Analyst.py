# pages/7_AI_Analyst.py
import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from io import StringIO
import traceback
import matplotlib.pyplot as plt
import altair as alt
import matplotlib.dates as mdates
import re
import plotly.graph_objects as go
import plotly.express as px
import sys

from constants import *
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

st.title("🤖 Strategic AI Analyst")
st.info("""
This AI Analyst reasons like a human analyst. It has access to the app's pre-calculated reports (like Site Performance) and can write custom code for novel questions. It uses the global scoring weights set on the **Site Performance** page.
""")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data and Config from Session State ---
df = st.session_state.referral_data_processed
ts_col_map = st.session_state.ts_col_map
ordered_stages = st.session_state.ordered_stages
status_history_col = "Parsed_Lead_Status_History"

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

# --- Configure the Gemini API ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
except Exception as e:
    st.error("Error configuring the AI model. Have you set your GEMINI_API_KEY in Streamlit's secrets?")
    st.exception(e)
    st.stop()

# --- System Prompts for the Advanced Agent ---
@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

@st.cache_data
def get_coder_prompt(_df_info, _ts_col_map_str, _site_perf_info, _utm_perf_info):
    prompt_parts = [
        "You are an expert Python data analyst. Your goal is to answer a user's question by generating a 'Thought' process and then the `Code` to execute it.",
        "\n--- RESPONSE FORMAT ---",
        "You MUST respond in two parts:",
        "1.  **Thought:** A step-by-step thought process explaining your plan. You MUST explicitly state the full, exact column and variable names you will use.",
        "2.  **Code:** A single, executable Python code block that implements your plan.",
        
        "\n--- COMPLETE EXAMPLE OF A COMPLEX REQUEST ---",
        "User Request: \"Show me in a line graph the enrollment trend for each site in the study by week\"",
        "Thought:",
        "1.  The user wants a weekly trend of 'enrollments' for each 'site'.",
        "2.  I must use the raw `df` to get weekly granularity.",
        "3.  The Golden Rule says analysis of 'enrollments' MUST use the enrollment timestamp, which is `'TS_Enrolled'`.",
        "4.  I will group by `'Site'` and `pd.Grouper` on the `'TS_Enrolled'` column with a weekly frequency (`freq='W'`).",
        "5.  Finally, I will use `plotly.express` (as `px`) to create a line chart and display it with `st.plotly_chart()`.",
        "```python",
        "import plotly.express as px",
        "enrollment_col = ts_col_map.get('Enrolled')",
        "if enrollment_col and enrollment_col in df.columns:",
        "    weekly_df = df.dropna(subset=[enrollment_col]).copy()",
        "    weekly_by_site = weekly_df.groupby(['Site', pd.Grouper(key=enrollment_col, freq='W')]).size().reset_index(name='Enrollment Count')",
        "    fig = px.line(weekly_by_site, x=enrollment_col, y='Enrollment Count', color='Site', title='Weekly Enrollment Trend by Site')",
        "    st.plotly_chart(fig, use_container_width=True)",
        "else:",
        "    print('Enrollment data is not available.')",
        "```",
        
        "\n--- AVAILABLE VARIABLES & DATAFRAMES ---",
        "1.  `site_performance_df`: Pre-computed DataFrame with aggregate site metrics.",
        "2.  `utm_performance_df`: Pre-computed DataFrame with aggregate UTM metrics.",
        "3.  `df`: The raw master DataFrame.",
        "4.  `ts_col_map`: Dictionary mapping stage names to timestamp columns.",
        
        "\n--- CRITICAL CODING RULE ---",
        "**Your final output MUST be displayed using a Streamlit function.** For a Matplotlib plot, you MUST end your code with `st.pyplot(plt.gcf())`. **DO NOT use `plt.show()`**. For any other output, use `st.dataframe()` or `print()`.",
        
        "\n--- DATAFRAME SCHEMAS ---",
        f"**`site_performance_df` Schema:**\n{_site_perf_info}",
        f"\n**`utm_performance_df` Schema:**\n{_utm_perf_info}",
        "\n--- RAW `df` SCHEMA ---",
        _df_info,
    ]
    return "\n".join(prompt_parts)

@st.cache_data
def get_synthesizer_prompt():
    return """You are an expert business analyst and senior strategist.
Your goal is to provide a single, cohesive, and insightful executive summary based on a series of data analyses.
You will be given the user's question, the AI's thought process, the code executed, and the raw data result.
- Start with a bolded headline that answers the user's core question.
- Weave the results into a narrative.
- Connect the data to business goals like speed, efficiency, or performance.
- Conclude with a clear recommendation or key takeaway.
"""

# --- Main Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Accessing business reports and forming a plan..."):
            # --- FIX: Read the pre-calculated DataFrames directly from session state ---
            site_perf_df = st.session_state.enhanced_site_metrics_df
            utm_perf_df = st.session_state.enhanced_ad_source_metrics_df
            
            site_perf_info = get_df_info(site_perf_df)
            utm_perf_info = get_df_info(utm_perf_df)
            
            try:
                coder_prompt = get_coder_prompt(get_df_info(df), str(ts_col_map), site_perf_info, utm_perf_info)
                full_coder_prompt = coder_prompt + f"\n\nNow, generate a Thought and Code block for this user request:\n{user_prompt}"
                response = model.generate_content(full_coder_prompt)
                
                match = re.search(r"Thought:(.*?)```python(.*?)```", response.text, re.DOTALL)
                if match:
                    thought = match.group(1).strip()
                    code_response = match.group(2).strip()
                else:
                    thought = "The AI did not provide a thought process. It may be a simple request."
                    code_response = response.text.strip().replace("```python", "").replace("```", "").strip()

            except Exception as e:
                st.error(f"An error occurred while generating code: {e}")
                st.stop()
        
        with st.expander("View AI's Thought Process and Code", expanded=True):
            st.markdown("**Thought Process:**")
            st.info(thought)
            st.markdown("**Generated Code:**")
            st.code(code_response, language="python")

        with st.spinner("Executing code..."):
            st.markdown("**Execution Result:**")
            result_display_area = st.container()
            result_output_str = ""
            try:
                execution_globals = {
                    "__builtins__": __builtins__, "st": st, "pd": pd, "np": np, "plt": plt, "alt": alt, "mdates": mdates, "go": go, "px": px,
                    "df": df, "site_performance_df": site_perf_df, "utm_performance_df": utm_perf_df,
                    "ordered_stages": ordered_stages, "ts_col_map": ts_col_map, "weights": weights
                }
                output_buffer = StringIO()
                sys.stdout = output_buffer
                with result_display_area:
                    exec(code_response, execution_globals)
                sys.stdout = sys.__stdout__
                result_output_str = output_buffer.getvalue()

                if result_output_str:
                    st.text(result_output_str)

            except Exception:
                error_traceback = traceback.format_exc()
                st.error("An error occurred during code execution:")
                st.code(error_traceback, language="bash")
                st.stop()

        with st.spinner("Synthesizing final summary..."):
            synthesis_context_parts = [
                "**User's Question:**", user_prompt, "\n\n",
                "**AI's Thought Process:**", thought, "\n\n",
                "**Executed Code:**", f"```python\n{code_response}\n```\n\n",
                "**Raw Result:**", result_output_str if result_output_str else "A plot was successfully generated."
            ]
            synthesis_context = "".join(synthesis_context_parts)
            
            synthesizer_prompt = get_synthesizer_prompt()
            full_synthesizer_prompt = f"{synthesizer_prompt}\n\n--- ANALYSIS DETAILS ---\n{synthesis_context}\n\n--- EXECUTIVE SUMMARY ---"
            
            try:
                summary_response = model.generate_content(full_synthesizer_prompt)
                summary_text = summary_response.text
            except Exception as e:
                summary_text = f"Could not generate summary: {e}"
        
        st.markdown("--- \n ## Executive Summary")
        st.markdown(summary_text)
        st.session_state.messages.append({"role": "assistant", "content": f"**Executive Summary:**\n{summary_text}"})