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

# Direct imports from modules in the root directory
from constants import *
from calculations import calculate_grouped_performance_metrics, calculate_avg_lag_generic
from scoring import score_performance_groups
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

with st.sidebar:
    st.logo("assets/logo.png", link="https://1nhealth.com")

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

# --- THIS IS THE FIX: Create self-sufficient, synced weights on this page ---
with st.expander("Adjust Performance Scoring Weights"):
    st.info("These weights are used by the AI when asked to score or rank performance. They are synced with the weights on the performance pages.")
    
    # Initialize session state for all weight keys if they don't exist
    weight_defaults = {
        'w_qual_to_enroll': 10, 'w_icf_to_enroll': 10, 'w_qual_to_icf': 20, 'w_avg_ttc': 10,
        'w_site_sf': 5, 'w_sts_appt': 15, 'w_appt_icf': 15, 'w_lag_q_icf': 5,
        'w_generic_sf': 5, 'w_proj_lag': 0
    }
    for key, value in weight_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Create sliders that read from and write to the same session state keys
    st.session_state.w_qual_to_enroll = st.slider("Qual (POF) -> Enrollment %", 0, 100, st.session_state.w_qual_to_enroll, key="w_q_enr_ai")
    st.session_state.w_icf_to_enroll = st.slider("ICF -> Enrollment %", 0, 100, st.session_state.w_icf_to_enroll, key="w_icf_enr_ai")
    st.session_state.w_qual_to_icf = st.slider("Qual (POF) -> ICF %", 0, 100, st.session_state.w_qual_to_icf, key="w_q_icf_ai")
    st.session_state.w_avg_ttc = st.slider("Avg Time to Contact (Sites)", 0, 100, st.session_state.w_avg_ttc, help="Lower is better.", key="w_ttc_ai")
    st.session_state.w_site_sf = st.slider("Site Screen Fail %", 0, 100, st.session_state.w_site_sf, help="Lower is better.", key="w_ssf_ai")
    st.session_state.w_sts_appt = st.slider("StS -> Appt Sched %", 0, 100, st.session_state.w_sts_appt, key="w_sts_appt_ai")
    st.session_state.w_appt_icf = st.slider("Appt Sched -> ICF %", 0, 100, st.session_state.w_appt_icf, key="w_appt_icf_ai")
    st.session_state.w_lag_q_icf = st.slider("Lag Qual -> ICF (Days)", 0, 100, st.session_state.w_lag_q_icf, help="Lower is better.", key="w_lag_ai")
    st.session_state.w_generic_sf = st.slider("Generic Screen Fail % (Ads)", 0, 100, st.session_state.w_generic_sf, help="Lower is better.", key="w_gsf_ai")
    st.session_state.w_proj_lag = st.slider("Generic Projection Lag (Ads)", 0, 100, st.session_state.w_proj_lag, help="Lower is better.", key="w_gpl_ai")

weights = {
    "Qual to Enrollment %": st.session_state.w_qual_to_enroll,
    "ICF to Enrollment %": st.session_state.w_icf_to_enroll,
    "Qual -> ICF %": st.session_state.w_qual_to_icf,
    "Avg TTC (Days)": st.session_state.w_avg_ttc,
    "Site Screen Fail %": st.session_state.w_site_sf,
    "StS -> Appt %": st.session_state.w_sts_appt,
    "Appt -> ICF %": st.session_state.w_appt_icf,
    "Lag Qual -> ICF (Days)": st.session_state.w_lag_q_icf,
    "Screen Fail % (from ICF)": st.session_state.w_generic_sf,
    "Projection Lag (Days)": st.session_state.w_proj_lag,
}
# --- END OF FIX ---

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
    prompt_parts = [
        "You are an expert Python data analyst. Your goal is to write a Python code block to solve a user's request.",
        "\n--- RESPONSE FORMAT ---",
        "You MUST respond in two parts:",
        "1.  **Thought:** A brief, step-by-step thought process explaining which tool you will use and why. **You MUST explicitly state the full, exact column and variable names you will use.**",
        "2.  **Code:** A single, executable Python code block that implements your plan.",
        "\n--- AVAILABLE VARIABLES & DATAFRAMES ---",
        "1.  `site_performance_df`: Pre-computed DataFrame with aggregate site metrics.",
        "2.  `utm_performance_df`: Pre-computed DataFrame with aggregate UTM metrics.",
        "3.  `df`: The raw master DataFrame.",
        "4.  `ts_col_map`: Dictionary mapping stage names to timestamp columns.",
        "\n--- CRITICAL CODING RULE ---",
        "For any time-series grouping (e.g., 'by week' or 'by month'), you MUST use the `pd.Grouper` method. Do not use `.resample()` on a grouped object.",
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
        with st.spinner("Pre-calculating business reports and forming a plan..."):
            site_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'Site', 'Unassigned Site')
            utm_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'UTM Source', 'Unclassified Source')
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