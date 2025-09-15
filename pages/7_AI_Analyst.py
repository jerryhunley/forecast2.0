# pages/7_AI_Analyst.py
import streamlit as st
import pandas as pd
import google.generativeai as genai
from io import StringIO
import traceback
import matplotlib.pyplot as plt
import altair as alt
import matplotlib.dates as mdates

# Direct imports from modules in the root directory
from constants import *
from calculations import calculate_grouped_performance_metrics, calculate_avg_lag_generic
from scoring import score_performance_groups
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 Conversational AI Analyst")
st.info("""
Ask questions about your data. The AI will write and execute code, then provide a natural language summary of the results.

**Example questions:**
- "What's the overall ICF to Enrollment rate?"
- "Show me a bar chart of the top 5 sites by enrollment count and tell me what it means."
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
def get_coder_prompt(_df_info, _ts_col_map_str):
    prompt = """You are a world-class Python data analyst. Your goal is to answer the user's question by generating a single, executable Python code block.

--- AVAILABLE TOOLS ---

You MUST use the exact function signatures provided below. Do not add or assume any extra arguments.
1.  **`calculate_grouped_performance_metrics()`**: For performance reports/breakdowns.
2.  **`calculate_avg_lag_generic()`**: For average time/lag between stages.
3.  **`pandas` and Visualization Libraries (`matplotlib`, `altair`)**: For custom analysis.

--- IMPORTANT CODING RULES ---

1.  **DO NOT redefine functions.** They are available for you to call.
2.  **Time-Period Filtering:** When asked for data "in May" or "last 30 days", filter on the relevant **event timestamp column**, not 'Submission_Month'.
3.  **Time-Series Analysis (Counting Events):** To count events "by week" or "by month", filter for not-null timestamps, set that SAME timestamp column as the index, then resample.
    *   Example (Weekly ICFs): `weekly_icfs = df[df[ts_col_map['Signed ICF']].notna()].set_index(ts_col_map['Signed ICF']).resample('W').size()`

4.  **Time-Series Analysis (Rate Trends):** To calculate a rate trend (e.g., "Sent to Site rate over time"), you must calculate the monthly totals for the numerator and the denominator separately, then combine them before dividing.
    *   **Correct Pattern for 'Sent to Site Rate Over Time':**
        ```python
        # Define the two timestamp columns needed
        denominator_col = ts_col_map['Passed Online Form']
        numerator_col = ts_col_map['Sent To Site']

        # Calculate monthly totals for the denominator (e.g., Qualified Leads)
        denominator_counts = df[df[denominator_col].notna()].set_index(denominator_col).resample('M').size().rename('Denominator')

        # Calculate monthly totals for the numerator (e.g., Sent To Site)
        numerator_counts = df[df[numerator_col].notna()].set_index(numerator_col).resample('M').size().rename('Numerator')

        # Combine the two series into a DataFrame
        rate_df = pd.concat([denominator_counts, numerator_counts], axis=1).fillna(0)

        # Calculate the rate safely
        if not rate_df.empty:
            rate_df['Rate'] = rate_df.apply(lambda row: row['Numerator'] / row['Denominator'] if row['Denominator'] > 0 else 0, axis=1)
            # Now you can plot rate_df['Rate']
        ```

5.  **How to Count Stages:** Count non-null values in the timestamp column.
6.  **Final Output Rendering:**
    *   **DataFrame:** Use `st.dataframe(result_df)`.
    *   **Matplotlib plot:** End with `st.pyplot(plt.gcf())`.
    *   **Altair chart:** End with `st.altair_chart(chart, use_container_width=True)`.
    *   **Other (number, string, list):** Use `print()`.
7.  **DEFENSIVE CODING:** Always check for division by zero and handle potential `NaN` or `inf` values gracefully.

--- CONTEXT VARIABLES ---
- `df`: The main pandas DataFrame.
- `ordered_stages`: A list of the funnel stage names in order.
- `ts_col_map`: A dictionary mapping stage names to timestamp columns: """
    prompt += f"`{_ts_col_map_str}`\n"
    prompt += """
- `weights`: A dictionary for scoring.

--- DATAFRAME `df` SCHEMA ---
"""
    prompt += _df_info
    prompt += "-----------------------------\n\n"
    prompt += "Your response MUST be a Python code block that starts with ```python and ends with ```. Do not provide any explanation."
    return prompt

@st.cache_data
def get_summarizer_prompt():
    # --- NEW: This prompt is for the second API call: generating the summary. ---
    return """You are a helpful and insightful data analyst for a clinical trial recruitment company.
Your goal is to provide a concise, natural language summary of a data analysis result.

You will be given the user's original question and the direct output (a number, table, or confirmation of a plot) that was generated by a code-executing agent.

Based on this information, provide a one or two-sentence summary that explains the key insight to a manager. Do not just repeat the numbers; interpret what they mean.
"""

@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

coder_prompt = get_coder_prompt(get_df_info(df), str(ts_col_map))
summarizer_prompt = get_summarizer_prompt()

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

    # --- Turn 1: Generate Code ---
    with st.spinner("AI is generating code..."):
        try:
            full_coder_prompt = coder_prompt + f"\n\nUser Question: {user_prompt}"
            response = model.generate_content(full_coder_prompt)
            code_response = response.text.strip().replace("```python", "").replace("```", "").strip()
        except Exception as e:
            st.error(f"An error occurred while generating code: {e}")
            st.stop()
    
    # --- Execute Code and Capture Result ---
    with st.spinner("Executing code..."):
        result_output = None
        with st.chat_message("assistant"):
            with st.expander("View Generated Code", expanded=False):
                st.code(code_response, language="python")

            try:
                execution_globals = {
                    "__builtins__": __builtins__, "st": st, "pd": pd, "plt": plt, "alt": alt, "mdates": mdates,
                    "df": df, "ordered_stages": ordered_stages, "ts_col_map": ts_col_map, "weights": weights,
                    "calculate_grouped_performance_metrics": calculate_grouped_performance_metrics,
                    "calculate_avg_lag_generic": calculate_avg_lag_generic,
                    "score_performance_groups": score_performance_groups,
                    "format_performance_df": format_performance_df
                }
                
                output_buffer = StringIO()
                import sys
                original_stdout = sys.stdout
                sys.stdout = output_buffer
                
                exec(code_response, execution_globals)
                
                sys.stdout = original_stdout
                result_output = output_buffer.getvalue()

            except Exception:
                error_traceback = traceback.format_exc()
                st.error("An error occurred during code execution:")
                st.code(error_traceback, language="bash")
                st.session_state.messages.append({"role": "assistant", "content": f"**Execution Error:**\n```\n{error_traceback}\n```"})
                st.stop()

    # --- Turn 2: Generate Summary ---
    with st.spinner("AI is analyzing the results..."):
        if result_output:
            summary_context = f"The code produced the following text output:\n```\n{result_output}\n```"
            with st.chat_message("assistant"):
                st.write("Execution Result:")
                st.text(result_output)
        else:
            summary_context = "The code successfully generated and displayed a plot."
            with st.chat_message("assistant"):
                st.success("A plot was generated successfully.")

        full_summarizer_prompt = f"{summarizer_prompt}\n\n--- USER'S ORIGINAL QUESTION ---\n{user_prompt}\n\n--- DATA ANALYSIS RESULT ---\n{summary_context}\n\n--- YOUR SUMMARY ---"
        
        try:
            summary_response = model.generate_content(full_summarizer_prompt)
            summary_text = summary_response.text
        except Exception as e:
            summary_text = f"Could not generate summary: {e}"

    # --- Display Final Summary ---
    with st.chat_message("assistant"):
        st.markdown(summary_text)
    
    # Add the full exchange to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"**Generated Code:**\n```python\n{code_response}\n```\n**Result:**\n{result_output if result_output else 'A plot was generated.'}\n\n**Summary:**\n{summary_text}"
    })