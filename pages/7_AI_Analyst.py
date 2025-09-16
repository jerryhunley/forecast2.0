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
import sys

# Direct imports from modules in the root directory
from constants import *
from calculations import calculate_grouped_performance_metrics, calculate_avg_lag_generic
from scoring import score_performance_groups
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 Strategic AI Analyst")
st.info("""
This advanced AI Analyst reasons like a human analyst. It has access to the app's pre-calculated reports (like Site Performance) and can write custom code for novel questions. It will always prefer to use the trusted, pre-calculated data when possible.
""")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data and Config from Session State ---
df = st.session_state.referral_data_processed
ts_col_map = st.session_state.ts_col_map
ordered_stages = st.session_state.ordered_stages
weights = st.session_state.weights_normalized

# --- Configure the Gemini API ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
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
    # This is the final, most robust version of the prompt, using a Chain-of-Thought example.
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
        "2.  I have two main tools: the pre-computed `site_performance_df` and the raw `df`.",
        "3.  I will first check the schema of `site_performance_df`. It contains total 'Enrollment Count' per site, but it does NOT have weekly data. Therefore, it is not suitable for this request.",
        "4.  I must fall back to using the raw `df` to get the necessary weekly granularity.",
        "5.  The Golden Rule states that analysis of 'enrollments' MUST use the enrollment timestamp. I will use the `ts_col_map` to find the correct column name, which is `'TS_Enrolled'`.",
        "6.  My plan is to group the raw `df` by 'Site' and then use `pd.Grouper` on the `'TS_Enrolled'` column with a weekly frequency (`freq='W'`) to get the counts.",
        "7.  Finally, I will use `plotly.express` to create a line chart of the results, with each site as a different color.",
        "```python",
        "enrollment_col = ts_col_map.get('Enrolled')",
        "if enrollment_col and enrollment_col in df.columns:",
        "    weekly_enrollments = df.dropna(subset=[enrollment_col]).copy()",
        "    weekly_enrollments_by_site = weekly_enrollments.groupby(['Site', pd.Grouper(key=enrollment_col, freq='W')]).size().reset_index(name='Enrollment Count')",
        "    fig = px.line(weekly_enrollments_by_site, x=enrollment_col, y='Enrollment Count', color='Site', title='Weekly Enrollment Trend by Site')",
        "    st.plotly_chart(fig, use_container_width=True)",
        "else:",
        "    print('Enrollment data is not available.')",
        "```",
        
        "\n--- AVAILABLE VARIABLES & DATAFRAMES ---",
        "1.  `site_performance_df`: Pre-computed DataFrame with aggregate site metrics.",
        "2.  `utm_performance_df`: Pre-computed DataFrame with aggregate UTM metrics.",
        "3.  `df`: The raw master DataFrame.",
        "4.  `ts_col_map`: Dictionary mapping stage names to timestamp columns.",
        
        "\n--- DATAFRAME SCHEMAS ---",
        f"**`site_performance_df` Schema:**\n{_site_perf_info}",
        f"\n**`utm_performance_df` Schema:**\n{_utm_perf_info}",
        "\n--- RAW `df` SCHEMA ---",
        _df_info,
        "-----------------------------",
    ]
    return "\n".join(prompt_parts)

@st.cache_data
def get_synthesizer_prompt():
    return """You are an expert business analyst and senior strategist for a clinical trial company.
Your goal is to provide a single, cohesive, and insightful executive summary based on a series of data analyses.
You will be given the user's original complex question, the AI's thought process, the Python code executed, and the raw data result from that code.
- **Start with a bolded headline** that directly answers the user's core question.
- **Weave the results into a narrative.**
- **Connect the data to business goals** like speed, efficiency, or performance.
- **Conclude with a clear recommendation** or key takeaway.
"""

# --- Main Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages = [{"role": "user", "content": user_prompt}]
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]

    with st.spinner("Pre-calculating standard business reports..."):
        site_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'Site', 'Unassigned Site')
        utm_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'UTM Source', 'Unclassified Source')
        
        site_perf_info = get_df_info(site_perf_df)
        utm_perf_info = get_df_info(utm_perf_df)

    with st.spinner("AI is forming a plan and writing code..."):
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
    
    with st.chat_message("assistant"):
        with st.expander("View AI's Thought Process and Code", expanded=True):
            st.markdown("**Thought Process:**")
            st.info(thought)
            st.markdown("**Generated Code:**")
            st.code(code_response, language="python")

    with st.spinner("Executing code..."):
        with st.chat_message("assistant"):
            st.markdown("**Execution Result:**")
            result_display_area = st.container()
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
        synthesis_context = (
            f"**User's Question:** {user_prompt}\n\n"
            f"**AI's Thought Process:** {thought}\n\n"
            f"**Executed Code:**\n```python\n{code_response}\n```\n\n"
            f"**Raw Result:**\n{result_output_str if result_output_str else 'A plot was generated.'}"
        )
        
        synthesizer_prompt = get_synthesizer_prompt()
        full_synthesizer_prompt = f"{synthesizer_prompt}\n\n--- ANALYSIS DETAILS ---\n{synthesis_context}\n\n--- EXECUTIVE SUMMARY ---"
        
        try:
            summary_response = model.generate_content(full_synthesizer_prompt)
            summary_text = summary_response.text
        except Exception as e:
            summary_text = f"Could not generate summary: {e}"

    with st.chat_message("assistant"):
        st.markdown("--- \n ## Executive Summary")
        st.markdown(summary_text)

    # Append the final, clean summary to the chat history
    st.session_state.messages.append({"role": "assistant", "content": f"**Executive Summary:**\n{summary_text}"})