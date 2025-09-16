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

# Direct imports from modules in the root directory
from constants import *
from calculations import calculate_grouped_performance_metrics, calculate_avg_lag_generic
from scoring import score_performance_groups
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 Strategic AI Analyst")
st.info("""
This advanced AI Analyst breaks down complex questions into a logical plan, executes each step sequentially while remembering previous results, and then synthesizes everything into a final summary.
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
    return """You are a project manager and expert data analyst. A user will ask a complex business question.
Your first and ONLY task is to break this question down into a series of simple, logical, numbered steps that can be answered one by one.
Each step should be a clear, self-contained instruction for a junior analyst to execute. The plan should build on itself.

Example:
User Question: "Explain to me the performance from the last month of the campaign, what sites are highest performing in terms of enrollments, how many enrollments we generated, what is the trend in sent to site?"

Your Output:
1. Calculate the total number of enrollments generated in the last calendar month.
2. Calculate the performance metrics for all sites based on data from the last calendar month.
3. From the performance report in step 2, identify and display the top 3 sites with the highest 'Enrollment Count'.
4. Generate a line chart showing the weekly trend of the 'Sent to Site' rate over the past 3 months to understand the broader trend.
"""

@st.cache_data
def get_coder_prompt():
    return """You are an expert Python data analyst. Your goal is to write a Python code block to solve the CURRENT STEP of an analysis plan.

--- CONTEXT ---
You will be given the user's overall goal, a complete history of the code that has been executed so far (the 'scratchpad'), and the specific step you need to accomplish now.
You MUST use the information from the scratchpad to inform your code. For example, if a previous step created a DataFrame called `performance_df`, you can use it.

--- AVAILABLE TOOLS & LIBRARIES ---
- `df`: The master pandas DataFrame with all the raw data.
- Pre-loaded functions: `calculate_grouped_performance_metrics()`, `calculate_avg_lag_generic()`, `score_performance_groups()`.
- Libraries: `pandas as pd`, `numpy as np`, `streamlit as st`, `matplotlib.pyplot as plt`, `altair as alt`, `matplotlib.dates as mdates`.

--- CODING RULES ---
1.  **Analyze the Scratchpad:** Look at the previously executed code and its results to decide if you can reuse a variable or if you need to recalculate something from the base `df`.
2.  **Output:** You MUST display the result of your step. Use `st.dataframe()` for DataFrames, `st.pyplot()` for plots, or `print()` for any other data type.
3.  **Variable Naming:** When creating a new DataFrame, give it a descriptive name (e.g., `last_month_performance`, `top_sites_df`) so it can be used in later steps.
4.  **DO NOT redefine functions.**

Your response MUST be ONLY the Python code block for the current step, starting with ```python and ending with ```."""

@st.cache_data
def get_synthesizer_prompt():
    return """You are an expert business analyst and strategist for a clinical trial company.
Your goal is to provide a single, cohesive, and insightful executive summary based on a series of data analyses.

You will be given the user's original complex question and a complete report containing the step-by-step plan, the Python code executed for each step, and the raw data result from each step.

Your task is to synthesize all of this information into a single, high-level summary.
- **Start with a clear, bolded headline** that answers the user's core question.
- **Do not just list the results.** Weave them together into a narrative.
- **Connect the different pieces of data.** For example, if one step shows high enrollments but another shows a slow trend, point out that discrepancy.
- **Provide a concluding sentence** with a key takeaway or recommendation.
- Keep your entire response to 3-5 sentences.
"""

# --- Main Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a complex question about your data..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # --- AGENTIC WORKFLOW ---
    scratchpad = "" # This will be our agent's memory
    
    # Step 1: DECOMPOSITION - Create a plan
    with st.spinner("Step 1/3: Decomposing question into a plan..."):
        try:
            planner_prompt = get_planner_prompt() + f"\n\nUser Question: {user_prompt}"
            plan_response = model.generate_content(planner_prompt)
            plan_text = plan_response.text
            analysis_steps = re.findall(r'^\s*\d+\.\s*(.*)', plan_text, re.MULTILINE)
            if not analysis_steps: analysis_steps = [line.strip() for line in plan_text.split('\n') if line.strip() and re.match(r'^\s*\d+\.', line)]
        except Exception as e:
            st.error(f"An error occurred while creating the plan: {e}")
            st.stop()
    
    with st.chat_message("assistant"):
        st.markdown("**Analysis Plan:**")
        st.markdown("\n".join(f"{i+1}. {step}" for i, step in enumerate(analysis_steps)))
        
    # Step 2: EXECUTION - Loop through the plan
    coder_prompt_template = get_coder_prompt()
    execution_globals = {
        "__builtins__": __builtins__, "st": st, "pd": pd, "np": np, "plt": plt, "alt": alt, "mdates": mdates,
        "df": df, "ordered_stages": ordered_stages, "ts_col_map": ts_col_map, "weights": weights,
        "calculate_grouped_performance_metrics": calculate_grouped_performance_metrics,
        "calculate_avg_lag_generic": calculate_avg_lag_generic, "score_performance_groups": score_performance_groups,
        "format_performance_df": format_performance_df
    }

    for i, step in enumerate(analysis_steps):
        with st.spinner(f"Step 2 ({i+1}/{len(analysis_steps)}): Executing '{step}'..."):
            try:
                # Add the scratchpad context to the coder prompt
                full_coder_prompt = coder_prompt_template + f"\n\n--- SCRATCHPAD (PREVIOUS STEPS) ---\n{scratchpad}\n\n--- CURRENT STEP ---\nYour task is to write Python code for the following step: \"{step}\""
                
                response = model.generate_content(full_coder_prompt)
                code_response = response.text.strip().replace("```python", "").replace("```", "").strip()

                with st.chat_message("assistant"):
                    st.markdown(f"**Step {i+1}: {step}**")
                    result_display_area = st.empty()
                    
                    output_buffer = StringIO()
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = output_buffer
                    
                    with result_display_area:
                        exec(code_response, execution_globals)
                    
                    sys.stdout = original_stdout
                    result_output_str = output_buffer.getvalue()

                    if result_output_str:
                        st.text(result_output_str)

                # UPDATE THE SCRATCHPAD with the code and result of this step
                scratchpad += f"# Step {i+1}: {step}\n"
                scratchpad += f"```python\n{code_response}\n```\n"
                scratchpad += f"# Result:\n# {result_output_str if result_output_str else 'A plot was generated.'}\n\n"

            except Exception:
                error_traceback = traceback.format_exc()
                st.error(f"An error occurred during step {i+1}:")
                st.code(error_traceback, language="bash")
                st.stop()

    # Step 3: SYNTHESIS - Create the final summary
    with st.spinner("Step 3/3: Synthesizing results into an executive summary..."):
        synthesizer_prompt = get_synthesizer_prompt()
        full_synthesizer_prompt = (
            synthesizer_prompt +
            "\n\n--- ORIGINAL USER QUESTION ---\n" + user_prompt +
            "\n\n--- FULL ANALYSIS REPORT ---\n" + scratchpad +
            "\n\n--- YOUR EXECUTIVE SUMMARY ---"
        )
        
        try:
            summary_response = model.generate_content(full_synthesizer_prompt)
            summary_text = summary_response.text
        except Exception as e:
            summary_text = f"Could not generate final summary: {e}"

    with st.chat_message("assistant"):
        st.markdown("--- \n ## Executive Summary")
        st.markdown(summary_text)

    st.session_state.messages.append({"role": "assistant", "content": f"**Executive Summary:**\n{summary_text}"})