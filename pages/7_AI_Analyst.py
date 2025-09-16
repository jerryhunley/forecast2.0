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
This advanced AI Analyst breaks down complex questions into a logical plan, executes each step, and then synthesizes the results into a single, insightful summary.
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

# --- System Prompts for Multi-Step Agent ---

@st.cache_data
def get_planner_prompt():
    return """You are a project manager and expert data analyst. A user will ask a complex business question.
Your first and ONLY task is to break this question down into a series of simple, logical, numbered steps that can be answered one by one.
Each step should be a clear, self-contained instruction for a junior analyst to execute.

Example:
User Question: "Explain to me the performance from the last month of the campaign, what sites are highest performing in terms of enrollments, how many enrollments we generated, what is the trend in sent to site?"

Your Output:
1. Calculate the total number of enrollments generated in the last month.
2. Identify the top 3 highest-performing sites by enrollment count in the last month and show their counts in a table.
3. Generate a line chart showing the weekly trend of the 'Sent to Site' rate over the past 3 months.
"""

@st.cache_data
def get_coder_prompt(_df_info, _ts_col_map_str):
    # This prompt is built with safe string concatenation to avoid formatting errors.
    prompt_part1 = """You are a world-class Python data analyst. Your goal is to answer the user's question about recruitment data by generating a single, executable Python code block.

--- AVAILABLE TOOLS ---
You MUST use the exact function signatures provided below. Do not add or assume any extra arguments.
1.  **`calculate_grouped_performance_metrics()`**: For performance reports/breakdowns.
2.  **`calculate_avg_lag_generic()`**: For average time/lag between stages.
3.  **`pandas` and Visualization Libraries (`matplotlib`, `altair`)**: For custom analysis.

--- CODING RULES ---
1.  **DO NOT redefine functions.** They are pre-loaded.
2.  **Clarification of Terms:** A "Site" is a location where leads are sent. A "Stage" is a step in the recruitment funnel (e.g., 'Sent To Site', 'Enrolled'). Leads transition between STAGES; they do not transition between SITES. If a user asks for a "site to site" trend, you should interpret this as a request for the performance trend of the 'Sent To Site' STAGE over time.
3.  **Time-Period Filtering:** For questions about a specific time (e.g., "in May"), filter the DataFrame on the relevant **event timestamp column**, not 'Submission_Month'.
4.  **Time-Series Analysis (Counting Events):** To count events "by week" or "by month", you must resample the relevant timestamp column.
5.  **Time-Series Analysis (Rate Trends):** To calculate a rate trend over time, you must calculate the monthly totals for the numerator and the denominator separately, then combine them before dividing.
6.  **Final Output Rendering:**
    *   **DataFrame:** Use `st.dataframe(result_df)`.
    *   **Matplotlib plot:** End with `st.pyplot(plt.gcf())`. For monthly trends, format the x-axis with `mdates.DateFormatter('%Y-%m')`.
    *   **Altair chart:** End with `st.altair_chart(chart, use_container_width=True)`.
    *   **Other (number, string, list):** Use `print()`.
7.  **DEFENSIVE CODING:** Always check for division by zero and handle potential `NaN` or `inf` values gracefully.

--- CONTEXT VARIABLES ---
- `df`: The main pandas DataFrame.
- `np`: The NumPy library, imported as `np`.
- `ordered_stages`: A list of the funnel stage names in order.
- `ts_col_map`: A dictionary mapping stage names to timestamp columns. Here is the exact dictionary: """

    prompt_part2 = f"`{_ts_col_map_str}`\n"
    prompt_part3 = """- `weights`: A dictionary for scoring.

--- DATAFRAME `df` SCHEMA ---
"""
    prompt_part4 = _df_info
    prompt_part5 = """
-----------------------------

Your response MUST be ONLY the Python code block, starting with ```python and ends with ```."""

    return prompt_part1 + prompt_part2 + prompt_part3 + prompt_part4 + prompt_part5

@st.cache_data
def get_synthesizer_prompt():
    return """You are an expert business analyst and strategist for a clinical trial company.
Your goal is to provide a single, cohesive, and insightful executive summary based on a series of data analyses.

You will be given the user's original complex question and a complete report containing the results from each step of the analysis.

Your task is to synthesize all of this information into a single, high-level summary.
- **Start with a clear, bolded headline** that answers the user's core question.
- **Do not just list the results.** Weave them together into a narrative.
- **Connect the different pieces of data.** For example, if one step shows high enrollments but another shows a slow trend, point out that discrepancy.
- **Provide a concluding sentence** with a key takeaway or recommendation.
- Keep your entire response to 3-5 sentences.
"""

@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

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
    full_analysis_report = ""
    
    # Step 1: DECOMPOSITION - Create a plan
    with st.spinner("Step 1/3: Decomposing question into a plan..."):
        try:
            planner_prompt = get_planner_prompt() + f"\n\nUser Question: {user_prompt}"
            plan_response = model.generate_content(planner_prompt)
            plan_text = plan_response.text
            # Use regex to find numbered list items
            analysis_steps = re.findall(r'^\s*\d+\.\s*(.*)', plan_text, re.MULTILINE)
            if not analysis_steps: # Fallback if regex fails
                analysis_steps = [line for line in plan_text.split('\n') if line.strip()]
        except Exception as e:
            st.error(f"An error occurred while creating the plan: {e}")
            st.stop()
    
    with st.chat_message("assistant"):
        st.markdown("**Analysis Plan:**")
        st.markdown("\n".join(f"{i+1}. {step}" for i, step in enumerate(analysis_steps)))
        
    full_analysis_report += "**Analysis Plan:**\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(analysis_steps)) + "\n\n"

    # Step 2: EXECUTION - Loop through the plan
    coder_prompt = get_coder_prompt(get_df_info(df), str(ts_col_map))
    execution_globals = {
        "__builtins__": __builtins__, "st": st, "pd": pd, "np": np, "plt": plt, "alt": alt, "mdates": mdates,
        "df": df, "ordered_stages": ordered_stages, "ts_col_map": ts_col_map, "weights": weights,
        "calculate_grouped_performance_metrics": calculate_grouped_performance_metrics,
        "calculate_avg_lag_generic": calculate_avg_lag_generic, "score_performance_groups": score_performance_groups,
        "format_performance_df": format_performance_df
    }

    for i, step in enumerate(analysis_steps):
        with st.spinner(f"Step 2/{len(analysis_steps)}: Executing '{step}'..."):
            try:
                # Turn 2a: Generate code for this specific step
                full_coder_prompt = coder_prompt + f"\n\nUser Question: {step}"
                response = model.generate_content(full_coder_prompt)
                code_response = response.text.strip().replace("```python", "").replace("```", "").strip()

                # Turn 2b: Execute the code
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
                    else:
                        st.success("A plot was generated successfully.")

                # Add the result of this step to our final report
                full_analysis_report += f"--- Result for Step {i+1}: {step} ---\n"
                full_analysis_report += result_output_str if result_output_str else "A plot was generated.\n\n"

            except Exception:
                error_traceback = traceback.format_exc()
                st.error(f"An error occurred during step {i+1}:")
                st.code(error_traceback, language="bash")
                st.stop()

    # Step 3: SYNTHESIS - Create the final summary
    with st.spinner("Step 3/3: Synthesizing results into an executive summary..."):
        synthesizer_prompt = get_summarizer_prompt()
        full_synthesizer_prompt = (
            synthesizer_prompt +
            "\n\n--- ORIGINAL USER QUESTION ---\n" + user_prompt +
            "\n\n--- FULL ANALYSIS REPORT ---\n" + full_analysis_report +
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

    # Add the final, clean summary to the chat history for display
    st.session_state.messages.append({"role": "assistant", "content": f"**Executive Summary:**\n{summary_text}"})