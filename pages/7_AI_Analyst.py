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

# Direct imports from modules in the root directory
from constants import *
from calculations import calculate_grouped_performance_metrics, calculate_avg_lag_generic
from scoring import score_performance_groups
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 Conversational AI Analyst")
st.info("""
This AI Analyst now performs a two-step analysis. First, it writes code to find an answer. Then, it interprets the results to provide actionable insights.
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
    # This prompt is built with safe string concatenation to avoid formatting errors.
    prompt_parts = [
        "You are a world-class Python data analyst. Your goal is to answer the user's question about recruitment data by generating a single, executable Python code block.",
        "\n--- AVAILABLE TOOLS ---",
        "You MUST use the exact function signatures provided below. Do not add or assume any extra arguments.",
        "1.  **`calculate_grouped_performance_metrics()`**: For performance reports/breakdowns.",
        "2.  **`calculate_avg_lag_generic()`**: For average time/lag between stages.",
        "3.  **`pandas`, `matplotlib`, `altair`, and `numpy`**: For custom analysis and visualizations.",
        "\n--- CODING RULES ---",
        "1.  **DO NOT redefine functions.** They are pre-loaded.",
        "2.  **Time-Period Filtering:** For questions about a specific time (e.g., \"in May\"), filter the DataFrame on the relevant **event timestamp column**, not 'Submission_Month'.",
        "3.  **Time-Series Analysis (Counting Events):** To count events \"by week\" or \"by month\", you must resample the relevant timestamp column.",
        "4.  **Time-Series Analysis (Rate Trends):** To calculate a rate trend over time, you must calculate the monthly totals for the numerator and the denominator separately, then combine them before dividing.",
        "5.  **Final Output Rendering:**",
        "    *   **DataFrame:** Use `st.dataframe(result_df)`.",
        "    *   **Matplotlib plot:** End with `st.pyplot(plt.gcf())`. For monthly trends, format the x-axis with `mdates.DateFormatter('%Y-%m')`.",
        "    *   **Altair chart:** End with `st.altair_chart(chart, use_container_width=True)`.",
        "    *   **Other (number, string, list):** Use `print()`.",
        "6.  **DEFENSIVE CODING:** Always check for division by zero. For missing or infinite values, you MUST use the NumPy library, which is available as `np`. Use `np.nan` for \"Not a Number\" and `np.inf` for infinity. Do NOT use `pd.inf` or `pd.nan`.",
        "\n--- CONTEXT VARIABLES ---",
        "- `df`: The main pandas DataFrame.",
        "- `np`: The NumPy library, imported as `np`.",
        "- `ordered_stages`: A list of the funnel stage names in order.",
        f"- `ts_col_map`: A dictionary mapping stage names to timestamp columns. Here is the exact dictionary: `{_ts_col_map_str}`",
        "- `weights`: A dictionary for scoring.",
        "\n--- DATAFRAME `df` SCHEMA ---",
        _df_info,
        "-----------------------------\n",
        "Your response MUST be ONLY the Python code block, starting with ```python and ends with ```."
    ]
    return "\n".join(prompt_parts)

@st.cache_data
def get_summarizer_prompt():
    return """You are a helpful and insightful data analyst for a clinical trial recruitment company.
Your goal is to provide a concise, natural language summary of a data analysis result.
You will be given the user's original question, the Python code used to answer it, and the raw data result from that code.
Based on this information, provide a one or two-sentence summary that explains the key insight to a manager. Do not just repeat the numbers; interpret what they mean.
"""

@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

coder_prompt = get_coder_prompt(get_df_info(df), str(ts_col_map))
summarizer_prompt = get_summarizer_prompt()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.spinner("AI is generating analysis code..."):
        try:
            full_coder_prompt = coder_prompt + f"\n\nUser Question: {user_prompt}"
            response = model.generate_content(full_coder_prompt)
            code_response = response.text.strip().replace("```python", "").replace("```", "").strip()
        except Exception as e:
            st.error(f"An error occurred while generating code: {e}")
            st.stop()
    
    result_output_str = ""
    with st.spinner("Executing code..."):
        with st.chat_message("assistant"):
            st.markdown("**Analysis Result:**")
            result_display_area = st.empty()

            try:
                execution_globals = {
                    "__builtins__": __builtins__, "st": st, "pd": pd, "np": np, "plt": plt, "alt": alt, "mdates": mdates,
                    "df": df, "ordered_stages": ordered_stages, "ts_col_map": ts_col_map, "weights": weights,
                    "calculate_grouped_performance_metrics": calculate_grouped_performance_metrics,
                    "calculate_avg_lag_generic": calculate_avg_lag_generic, "score_performance_groups": score_performance_groups,
                    "format_performance_df": format_performance_df
                }
                
                with result_display_area:
                    output_buffer = StringIO()
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = output_buffer
                    
                    exec(code_response, execution_globals)
                    
                    sys.stdout = original_stdout
                    result_output_str = output_buffer.getvalue()

            except Exception:
                error_traceback = traceback.format_exc()
                st.error("An error occurred during code execution:")
                st.code(error_traceback, language="bash")
                st.session_state.messages.append({"role": "assistant", "content": f"**Execution Error:**\n```\n{error_traceback}\n```"})
                st.stop()

    with st.spinner("AI is interpreting the results..."):
        if result_output_str:
            with result_display_area:
                st.text(result_output_str)

        # Build strings safely using concatenation to avoid formatting issues
        analysis_context = "-- PYTHON CODE USED --\n"
        analysis_context += "```python\n"
        analysis_context += code_response + "\n"
        analysis_context += "```\n\n"
        analysis_context += "--- RAW DATA RESULT ---\n"
        analysis_context += result_output_str if result_output_str else "A plot was successfully generated."

        full_summarizer_prompt = (
            summarizer_prompt +
            "\n\n--- USER'S QUESTION ---\n" +
            user_prompt +
            "\n\n--- ANALYSIS & RESULT ---\n" +
            analysis_context +
            "\n\n--- YOUR INSIGHTFUL SUMMARY ---"
        )
        
        try:
            summary_response = model.generate_content(full_summarizer_prompt)
            summary_text = summary_response.text
        except Exception as e:
            summary_text = f"Could not generate summary: {e}"

    with st.chat_message("assistant"):
        st.markdown("**Summary & Insights:**")
        st.markdown(summary_text)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"**Analysis Result:**\n```\n{result_output_str if result_output_str else 'A plot was generated.'}\n```\n**Summary & Insights:**\n{summary_text}"
    })