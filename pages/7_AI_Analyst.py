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

# Direct imports from modules in the root directory
from constants import *
from calculations import calculate_grouped_performance_metrics, calculate_avg_lag_generic
from scoring import score_performance_groups
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 Strategic AI Analyst")
st.info("""
This advanced AI Analyst uses a **Plan-Critique-Execute** model. It forms an initial plan, has another AI review and correct that plan for logical flaws, and then executes the final, approved plan to provide a robust analysis.
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
def get_planner_prompt():
    # Built with a list of strings to be robust
    prompt_parts = [
        "You are a project manager and expert data analyst. A user will ask a complex business question.",
        "Your first and ONLY task is to break this question down into a series of simple, logical, numbered steps for a junior analyst to execute.",
        "Each step should be a clear, self-contained instruction. The plan should build on itself. It should be efficient and directly answer the user's question.",
        "\n**Business Rules for Clarification:**",
        "- The primary date for determining \"last month\" or \"recent\" activity is `'Submitted On_DT'`.",
        "- A \"site to site trend\" is ambiguous. You MUST interpret this as a request for the performance trend of the **'Sent To Site' STAGE** over time.",
        "\n**Example:**",
        "User Question: \"Explain to me the performance from the last month of the campaign, what sites are highest performing in terms of enrollments, how many enrollments we generated, what is the trend in sent to site?\"",
        "\nYour Output:",
        "1. Determine the start and end dates for the most recent full calendar month in the data.",
        "2. Calculate the total number of enrollments that occurred within that calendar month.",
        "3. Calculate a performance summary for all sites using only the data from that month.",
        "4. From the summary in step 3, identify and display the top 3 sites with the highest 'Enrollment Count'.",
        "5. Generate a line chart showing the weekly trend of the 'Sent to Site' rate over the past 3 months to provide broader context on recent performance."
    ]
    return "\n".join(prompt_parts)

@st.cache_data
def get_critique_prompt():
    return """You are a senior data science manager. A junior analyst has proposed the following plan. Your job is to find flaws and improve it.
Review the plan for misinterpretations of business logic, logical errors, or inefficiency.

**Business Rules for Correction:**
- The primary date for determining "last month" or "recent" activity is `'Submitted On_DT'`.
- A "site to site trend" is ambiguous. You MUST correct the plan to interpret this as a request for the performance trend of the **'Sent To Site' STAGE** over time.

After reviewing, output a **final, corrected, and optimized numbered plan**. If the original plan is already perfect, simply output it again. Your output must be ONLY the final numbered list.
"""

@st.cache_data
def get_coder_prompt(_df_info, _ts_col_map_str):
    prompt_parts = [
        "You are an expert Python data analyst. Your goal is to write a Python code block to solve the CURRENT STEP of an analysis plan.",
        "--- CONTEXT ---",
        "You have access to the user's overall goal and a scratchpad of code and results from previous steps. You MUST use this scratchpad to inform your code (e.g., reusing variables).",
        "--- AVAILABLE TOOLS & LIBRARIES ---",
        "- `df`: The master pandas DataFrame with all the raw data.",
        "- Pre-loaded functions: `calculate_grouped_performance_metrics()`, `calculate_avg_lag_generic()`.",
        "- Libraries: `pandas as pd`, `numpy as np`, `streamlit as st`, `matplotlib.pyplot as plt`, `altair as alt`, `plotly.graph_objects as go`.",
        "--- CODING RULES ---",
        "1.  **Primary Date Column:** For general date filtering, use `'Submitted On_DT'`.",
        "2.  **Final Output:** You MUST display your result. Use `st.dataframe()`, `st.pyplot()`, `st.altair_chart()`, `st.plotly_chart()`, or `print()`.",
        "3.  **DEFENSIVE CODING:** Always check for division by zero and handle `NaN`/`inf` values using `np.nan` and `np.inf`.",
        "--- CONTEXT VARIABLES ---",
        "- `df`: The main pandas DataFrame.",
        "- `np`: The NumPy library, imported as `np`.",
        f"- `ts_col_map`: The dictionary mapping stage names to timestamp columns: `{_ts_col_map_str}`",
        "--- DATAFRAME `df` SCHEMA ---",
        _df_info,
        "-----------------------------",
        "Your response MUST be ONLY the Python code block for the current step, starting with ```python and ends with ```."
    ]
    return "\n".join(prompt_parts)

@st.cache_data
def get_synthesizer_prompt():
    return """You are an expert business analyst and strategist. Your goal is to provide a single, cohesive, and insightful executive summary based on a series of data analyses.
You will be given the user's original complex question and a complete report containing the step-by-step plan, the Python code executed for each step, and the raw data result from each step.
- **Start with a clear, bolded headline** that answers the user's core question.
- **Weave the results into a narrative.** Don't just list them.
- **Connect the different pieces of data.** If one step shows high enrollments but another shows a slow trend, point that out.
- **Provide a concluding sentence** with a key takeaway or recommendation.
"""

@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

# --- Main Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_plan" not in st.session_state:
    st.session_state.analysis_plan = None
if "final_summary" not in st.session_state:
    st.session_state.final_summary = None
if "scratchpad" not in st.session_state:
    st.session_state.scratchpad = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a complex question about your data..."):
    # Start a new analysis run
    st.session_state.messages = [{"role": "user", "content": user_prompt}]
    st.session_state.analysis_plan = None
    st.session_state.final_summary = None
    st.session_state.scratchpad = ""
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    # AGENTIC WORKFLOW
    if not st.session_state.analysis_plan:
        with st.chat_message("assistant"):
            with st.spinner("Step 1/4: Decomposing question and forming a plan..."):
                try:
                    planner_prompt = get_planner_prompt()
                    full_planner_prompt = planner_prompt + f"\n\nUser Question: {user_prompt}"
                    plan_response = model.generate_content(full_planner_prompt)
                    initial_plan = plan_response.text
                except Exception as e:
                    st.error(f"An error occurred while creating the initial plan: {e}")
                    st.stop()
            
            with st.spinner("Step 2/4: Reviewing and refining the plan for errors..."):
                try:
                    critique_prompt = get_critique_prompt()
                    full_critique_prompt = critique_prompt + f"\n\nUser Question: {user_prompt}\n\nProposed Plan:\n{initial_plan}"
                    critique_response = model.generate_content(full_critique_prompt)
                    final_plan_text = critique_response.text
                    st.session_state.analysis_plan = final_plan_text
                except Exception as e:
                    st.error(f"An error occurred while refining the plan: {e}")
                    st.stop()
        st.rerun()

    if st.session_state.get("analysis_plan") and not st.session_state.get("final_summary"):
        final_plan_text = st.session_state.analysis_plan
        analysis_steps = re.findall(r'^\s*\d+\.\s*(.*)', final_plan_text, re.MULTILINE)
        
        with st.chat_message("assistant"):
            with st.expander("View AI's Analysis Plan", expanded=True):
                st.markdown(final_plan_text)
                
        coder_prompt_template = get_coder_prompt(get_df_info(df), str(ts_col_map))
        execution_globals = {
            "__builtins__": __builtins__, "st": st, "pd": pd, "np": np, "plt": plt, "alt": alt, "mdates": mdates, "go": go,
            "df": df, "ordered_stages": ordered_stages, "ts_col_map": ts_col_map, "weights": weights,
            "calculate_grouped_performance_metrics": calculate_grouped_performance_metrics,
            "calculate_avg_lag_generic": calculate_avg_lag_generic, "score_performance_groups": score_performance_groups,
            "format_performance_df": format_performance_df
        }

        for i, step in enumerate(analysis_steps):
            with st.chat_message("assistant"):
                with st.spinner(f"Step 3 ({i+1}/{len(analysis_steps)}): Executing '{step}'..."):
                    try:
                        full_coder_prompt = coder_prompt_template + f"\n\n--- SCRATCHPAD (PREVIOUS STEPS) ---\n{st.session_state.scratchpad}\n\n--- CURRENT STEP ---\nYour task is to write Python code for this step: \"{step}\""
                        response = model.generate_content(full_coder_prompt)
                        code_response = response.text.strip().replace("```python", "").replace("```", "").strip()
                        
                        st.markdown(f"**Step {i+1}: {step}**")
                        result_display_area = st.container()
                        
                        output_buffer = StringIO()
                        sys.stdout = output_buffer
                        
                        with result_display_area:
                            exec(code_response, execution_globals)
                        
                        sys.stdout = sys.__stdout__
                        result_output_str = output_buffer.getvalue()

                        if result_output_str:
                            st.text(result_output_str)

                        st.session_state.scratchpad += f"\n\n# Step {i+1}: {step}\n"
                        st.session_state.scratchpad += f"```python\n{code_response}\n```\n"
                        st.session_state.scratchpad += f"# Result:\n# {result_output_str if result_output_str else 'A plot was generated.'}"

                    except Exception:
                        error_traceback = traceback.format_exc()
                        st.error(f"An error occurred during step {i+1}:")
                        st.code(error_traceback, language="bash")
                        st.stop()
        
        with st.chat_message("assistant"):
            with st.spinner("Step 4/4: Synthesizing results into an executive summary..."):
                site_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'Site', 'Unassigned Site')
                utm_perf_df = calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, 'UTM Source', 'Unclassified Source')

                # --- THIS IS THE CORRECTED AND ROBUST STRING CONSTRUCTION ---
                analysis_context_parts = [
                    "\n\n--- BUSINESS CONTEXT APPENDIX ---\n",
                    "**Overall Site Performance Dataframe:**",
                    f"```\n{site_perf_df.to_string()}\n```\n",
                    "**Overall UTM Source Performance Dataframe:**",
                    f"```\n{utm_perf_df.to_string()}\n```"
                ]
                business_context_appendix = "\n".join(analysis_context_parts)
                
                synthesizer_prompt = get_synthesizer_prompt()
                
                full_synthesizer_prompt_parts = [
                    synthesizer_prompt,
                    "\n\n--- ORIGINAL USER QUESTION ---\n", user_prompt,
                    "\n\n--- FULL ANALYSIS REPORT (PLAN AND RESULTS) ---\n", st.session_state.scratchpad,
                    business_context_appendix,
                    "\n\n--- YOUR EXECUTIVE SUMMARY ---"
                ]
                full_synthesizer_prompt = "".join(full_synthesizer_prompt_parts)
                
                try:
                    summary_response = model.generate_content(full_synthesizer_prompt)
                    summary_text = summary_response.text
                except Exception as e:
                    summary_text = f"Could not generate final summary: {e}"

            st.markdown("--- \n ## Executive Summary")
            st.markdown(summary_text)

        st.session_state.final_summary = summary_text
        st.session_state.messages.append({"role": "assistant", "content": f"**Executive Summary:**\n{summary_text}"})
        st.rerun()