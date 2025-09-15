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
# Prompt 1: For generating Python code
@st.cache_data
def get_coder_prompt(_df_info, _ts_col_map_str):
    # This prompt is highly structured and strict, focused only on code generation.
    return """You are a Python data analysis expert. Your sole purpose is to generate a single, executable Python code block to answer a user's question about a pandas DataFrame named `df`.

--- AVAILABLE TOOLS ---
You have access to pre-built functions and libraries. You MUST use the exact function signatures provided.
1.  `calculate_grouped_performance_metrics()`: For performance reports (e.g., by 'Site'). Returns a DataFrame.
2.  `calculate_avg_lag_generic()`: For average lag time between two stages. Returns a number.
3.  `pandas`, `matplotlib`, `altair`: For custom analysis and visualizations.

--- CODING RULES ---
1.  **DO NOT redefine functions.** They are pre-loaded.
2.  **Time-Period Filtering:** For questions about a specific time (e.g., "in May"), filter on the relevant event timestamp column.
3.  **Time-Series Analysis (Rate Trends):** To calculate a rate trend over time, calculate monthly totals for the numerator and denominator separately, combine them, then divide.
4.  **Final Output:**
    *   **DataFrame:** Use `st.dataframe(result_df)`.
    *   **Matplotlib:** End with `st.pyplot(plt.gcf())`. For monthly trends, format the x-axis with `mdates.DateFormatter('%Y-%m')`.
    *   **Altair:** End with `st.altair_chart(chart, use_container_width=True)`.
    *   **Other (number, string, list):** Use `print()`.
5.  **DEFENSIVE CODING:** Always check for division by zero and handle potential `NaN` values.

--- CONTEXT VARIABLES ---
- `df`: The main DataFrame.
- `ordered_stages`: A list of funnel stages.
- `ts_col_map`: Dictionary mapping stage names to timestamp columns: """ + f"`{_ts_col_map_str}`" + """

--- DATAFRAME `df` SCHEMA ---
""" + _df_info + """
-----------------------------\n
Your response MUST be ONLY the Python code block, starting with ```python and ending with ```."""

# Prompt 2: For interpreting the results
@st.cache_data
def get_summarizer_prompt():
    return """You are an expert data analyst and business strategist for a clinical trial company. Your goal is to provide actionable insights for a manager.

You will be given the user's original question, the Python code used to answer it, and the raw data result from that code.

Your task is to synthesize this information into a concise, insightful summary.
- **Do not just describe the data.** Explain what it MEANS.
- **Identify the key trend, outlier, or most important number.**
- **Connect the data to business goals** like speed, efficiency, or performance.
- Keep your response to 2-3 sentences. Start with a clear headline in bold.
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

    # Turn 1: Generate Code
    with st.spinner("AI is generating analysis code..."):
        try:
            full_coder_prompt = coder_prompt + f"\n\nUser Question: {user_prompt}"
            response = model.generate_content(full_coder_prompt)
            code_response = response.text.strip().replace("```python", "").replace("```", "").strip()
        except Exception as e:
            st.error(f"An error occurred while generating code: {e}")
            st.stop()
    
    # Execute Code and Capture Result
    result_output_str = ""
    with st.spinner("Executing code..."):
        with st.chat_message("assistant"):
            st.markdown("**Analysis Result:**")
            # Display area for the plot or table
            result_display_area = st.empty()

            try:
                execution_globals = {
                    "__builtins__": __builtins__, "st": st, "pd": pd, "plt": plt, "alt": alt, "mdates": mdates,
                    "df": df, "ordered_stages": ordered_stages, "ts_col_map": ts_col_map, "weights": weights,
                    "calculate_grouped_performance_metrics": calculate_grouped_performance_metrics,
                    "calculate_avg_lag_generic": calculate_avg_lag_generic, "score_performance_groups": score_performance_groups,
                    "format_performance_df": format_performance_df
                }
                
                with result_display_area: # Execute code within the context of the display area
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

    # Turn 2: Generate Summary with Full Context
    with st.spinner("AI is interpreting the results..."):
        # If the code printed something, display it
        if result_output_str:
            with result_display_area:
                st.text(result_output_str)

        # Prepare context for the summarizer. This now includes the code and its output.
        analysis_context = f"""--- PYTHON CODE USED ---
```python
{code_response}