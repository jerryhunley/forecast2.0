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

# --- ALL HELPER FUNCTIONS ARE DEFINED HERE, BEFORE THEY ARE CALLED ---

@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

@st.cache_data
def get_coder_prompt(_df_info, _ts_col_map_str, _site_perf_info, _utm_perf_info):
    prompt_parts = [
        "You are a world-class Python data analyst. Your goal is to answer a user's question by first creating a 'Thought' process and then writing the `Code` to execute it.",
        "\n--- RESPONSE FORMAT ---",
        "You MUST respond in two parts:",
        "1.  **Thought:** A brief, step-by-step thought process explaining which tool you will use and why.",
        "2.  **Code:** A single, executable Python code block that implements your plan.",
        "\n--- AVAILABLE TOOLS (HIERARCHY OF PREFERENCE) ---",
        "You have access to several data sources. You MUST use them in this order of preference:",
        "1.  **`site_performance_df` (HIGHEST PRIORITY):** A pre-computed pandas DataFrame containing all key performance metrics for every site. Use this for ANY question about site performance, rankings, or comparisons.",
        "2.  **`utm_performance_df` (HIGH PRIORITY):** A pre-computed pandas DataFrame with performance metrics for each UTM source. Use this for any question about marketing channel performance.",
        "3.  **`df` (LOWEST PRIORITY):** The raw, unprocessed master DataFrame. Only use this for very specific, ad-hoc queries that cannot be answered by the pre-computed DataFrames above, or for creating custom charts.",
        "\n--- CRITICAL REASONING RULES ---",
        "- In your 'Thought' process, you MUST explicitly state which DataFrame you are choosing to use and why.",
        "- Your primary goal is to use the pre-computed DataFrames whenever possible.",
        "\n--- CRITICAL CODING RULES ---",
        "- **Primary Date Column:** For general date filtering on the raw `df`, use `'Submitted On_DT'`.",
        "- **Final Output:** Display your result with `st.dataframe()`, `st.pyplot()`, `st.altair_chart()`, `st.plotly_chart()`, or `print()`.",
        "- **DEFENSIVE CODING:** Handle division by zero and `NaN`/`inf` values using `np.nan` and `np.inf`.",
        "\n--- PRE-COMPUTED DATAFRAME SCHEMAS ---",
        f"**`site_performance_df` Schema:**\n{_site_perf_info}",
        f"\n**`utm_performance_df` Schema:**\n{_utm_perf_info}",
        "\n--- RAW `df` SCHEMA ---",
        _df_info,
        "-----------------------------",
        f"\n- `ts_col_map` dictionary for raw df: `{_ts_col_map_str}`",
        "-----------------------------\n",
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
            # THIS IS WHERE THE NAMEERROR WAS. THE FUNCTION CALL IS NOW AFTER THE DEFINITION.
            coder_prompt = get_coder_prompt(get_df_info(df), str(ts_col_map), site_perf_info, utm_perf_info)
            full_coder_prompt = coder_prompt + f"\n\nUser Question: {user_prompt}"
            response = model.generate_content(full_coder_prompt)
            
            match = re.search(r"Thought:(.*?)```python(.*?)```", response.text, re.DOTALL)
            if match:
                thought = match.group(1).strip()
                code_response = match.group(2).strip()
            else:
                thought = "No thought process was generated. Proceeding with code execution."
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
                    "__builtins__": __builtins__, "st": st, "pd": pd, "np": np, "plt": plt, "alt": alt, "mdates": mdates, "go": go,
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
    st.rerun()