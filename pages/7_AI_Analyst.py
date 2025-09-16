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

# --- System Prompts ---
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

@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

# --- Main Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_run" not in st.session_state:
    st.session_state.agent_run = {"running": False}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a complex question about your data..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.agent_run = {
        "running": True,
        "plan": None,
        "scratchpad": "",
        "summary": None,
    }
    st.rerun()

if st.session_state.agent_run and st.session_state.agent_run["running"]:
    agent = st.session_state.agent_run
    user_prompt = st.session_state.messages[-1]["content"] # Get the last user message
    
    # AGENTIC WORKFLOW
    if not agent["plan"]:
        with st.chat_message("assistant"):
            with st.spinner("Step 1/3: Decomposing question and forming a plan..."):
                try:
                    site_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'Site', 'Unassigned Site')
                    utm_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'UTM Source', 'Unclassified Source')
                    coder_prompt = get_coder_prompt(get_df_info(df), str(ts_col_map), get_df_info(site_perf_df), get_df_info(utm_perf_df))
                    
                    full_planner_prompt = get_planner_prompt() + f"\n\nUser Question: {user_prompt}"
                    plan_response = model.generate_content(full_planner_prompt)
                    initial_plan = plan_response.text
                except Exception as e:
                    st.error(f"An error occurred while creating the initial plan: {e}")
                    st.stop()
            
            with st.spinner("Step 2/3: Reviewing and refining the plan for errors..."):
                try:
                    critique_prompt = get_critique_prompt() + f"\n\nUser Question: {user_prompt}\n\nProposed Plan:\n{initial_plan}"
                    critique_response = model.generate_content(critique_prompt)
                    final_plan_text = critique_response.text
                    agent["plan"] = final_plan_text
                except Exception as e:
                    st.error(f"An error occurred while refining the plan: {e}")
                    st.stop()
        st.rerun()

    if agent["plan"] and not agent["summary"]:
        analysis_steps = re.findall(r'^\s*\d+\.\s*(.*)', agent["plan"], re.MULTILINE)
        
        with st.chat_message("assistant"):
            with st.expander("View AI's Analysis Plan", expanded=True):
                st.markdown(agent["plan"])
                
        site_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'Site', 'Unassigned Site')
        utm_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'UTM Source', 'Unclassified Source')
        coder_prompt_template = get_coder_prompt(get_df_info(df), str(ts_col_map), get_df_info(site_perf_df), get_df_info(utm_perf_df))

        execution_globals = {
            "__builtins__": __builtins__, "st": st, "pd": pd, "np": np, "plt": plt, "alt": alt, "mdates": mdates, "go": go,
            "df": df, "site_performance_df": site_perf_df, "utm_performance_df": utm_perf_df,
            "ordered_stages": ordered_stages, "ts_col_map": ts_col_map, "weights": weights,
        }

        for i, step in enumerate(analysis_steps):
            with st.chat_message("assistant"):
                with st.spinner(f"Step 3 ({i+1}/{len(analysis_steps)}): Executing '{step}'..."):
                    try:
                        full_coder_prompt = coder_prompt_template + f"\n\n--- SCRATCHPAD (PREVIOUS STEPS) ---\n{agent['scratchpad']}\n\n--- CURRENT STEP ---\nYour task is to write Python code for this step: \"{step}\""
                        response = model.generate_content(full_coder_prompt)
                        code_response = response.text.strip().replace("```python", "").replace("```", "").strip()
                        
                        st.markdown(f"**Step {i+1}: {step}**")
                        result_display_area = st.container()
                        
                        output_buffer = StringIO()
                        original_stdout = sys.stdout
                        sys.stdout = output_buffer
                        
                        with result_display_area:
                            exec(code_response, execution_globals)
                        
                        sys.stdout = original_stdout
                        result_output_str = output_buffer.getvalue()

                        if result_output_str:
                            st.text(result_output_str)

                        agent["scratchpad"] += f"\n\n# Step {i+1}: {step}\n"
                        agent["scratchpad"] += f"```python\n{code_response}\n```\n"
                        agent["scratchpad"] += f"# Result:\n# {result_output_str if result_output_str else 'A plot was generated.'}"

                    except Exception:
                        error_traceback = traceback.format_exc()
                        st.error(f"An error occurred during step {i+1}:")
                        st.code(error_traceback, language="bash")
                        st.stop()
        
        with st.chat_message("assistant"):
            with st.spinner("Step 4/4: Synthesizing results into an executive summary..."):
                business_context_appendix = (
                    "\n\n--- BUSINESS CONTEXT APPENDIX ---\n\n"
                    "**Overall Site Performance Dataframe:**\n"
                    f"```\n{site_perf_df.to_string()}\n```\n\n"
                    "**Overall UTM Source Performance Dataframe:**\n"
                    f"```\n{utm_perf_df.to_string()}\n```"
                )
                
                synthesizer_prompt = get_synthesizer_prompt()
                
                full_synthesizer_prompt = (
                    synthesizer_prompt +
                    "\n\n--- ORIGINAL USER QUESTION ---\n" + user_prompt +
                    "\n\n--- FULL ANALYSIS REPORT (PLAN AND RESULTS) ---\n" + agent["scratchpad"] +
                    business_context_appendix +
                    "\n\n--- YOUR EXECUTIVE SUMMARY ---"
                )
                
                try:
                    summary_response = model.generate_content(full_synthesizer_prompt)
                    summary_text = summary_response.text
                except Exception as e:
                    summary_text = f"Could not generate final summary: {e}"

            st.markdown("--- \n ## Executive Summary")
            st.markdown(summary_text)

        agent["summary"] = summary_text
        st.session_state.messages.append({"role": "assistant", "content": f"**Executive Summary:**\n{summary_text}"})
        # --- FIX: Stop the agent run instead of re-running the whole page ---
        st.session_state.agent_run["running"] = False