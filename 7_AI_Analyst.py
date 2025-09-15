# pages/7_AI_Analyst.py
import streamlit as st
import pandas as pd
import google.generativeai as genai
from io import StringIO
import traceback
import matplotlib.pyplot as plt
import altair as alt

# Direct imports from modules in the root directory
from constants import *
from calculations import calculate_grouped_performance_metrics, calculate_avg_lag_generic
from scoring import score_performance_groups
from helpers import format_performance_df

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 Tool-Using AI Analyst")
st.info("""
This AI Analyst can now use the app's own calculation functions as tools to answer questions. 
This ensures consistency with the other pages. It can still write custom code for unique requests.

**Example questions:**
- "Can you show me the full site performance report?" (Uses `calculate_grouped_performance_metrics`)
- "What's the average time from Signed ICF to Enrolled?" (Uses `calculate_avg_lag_generic`)
- "Create a bar chart of the top 5 sites by enrollment count." (Now works!)
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

# --- NEW: More Explicit System Prompt ---
@st.cache_data
def get_tool_prompt(_df_info, _ts_col_map_str, _ordered_stages_str):
    prompt = """You are a world-class Python data analyst. Your goal is to answer the user's question about recruitment data by generating a single, executable Python code block.

--- AVAILABLE TOOLS ---

You have access to a pandas DataFrame named `df` and a set of pre-built, trusted Python functions. You should ALWAYS prefer to use a pre-built function if it is suitable for the user's request.

1.  **`calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, grouping_col, unclassified_label)`**
    *   **Description:** The primary tool for performance analysis. Calculates all standard performance metrics (conversion rates, lag times, counts, etc.) for a given grouping.
    *   **Use When:** The user asks for a "performance report", "breakdown", "summary", or comparison of a group like 'Site' or 'UTM Source'.
    *   **Returns:** A pandas DataFrame. You MUST print the result.
    *   **Example Call:** `print(calculate_grouped_performance_metrics(df, ordered_stages, ts_col_map, grouping_col='Site', unclassified_label='Unassigned Site'))`

2.  **`calculate_avg_lag_generic(df, col_from, col_to)`**
    *   **Description:** Calculates the average lag time in days between two specific timestamp columns.
    *   **Use When:** The user asks for the "average time", "lag", "duration", or "how long it takes" between two specific stages.
    *   **Returns:** A number. You MUST print the result.
    *   **Example Call:** `print(calculate_avg_lag_generic(df, ts_col_map.get('Signed ICF'), ts_col_map.get('Enrolled')))`

3.  **`pandas` and Visualization Libraries (`matplotlib`, `altair`)**
    *   **Description:** For any custom analysis or visualization where a pre-built tool is not available.
    *   **Use When:** The user asks for a specific plot (bar, line, pie chart) or a custom data slice that the tools above don't provide.

--- IMPORTANT BUSINESS DEFINITIONS & CODING RULES ---

1.  **How to Count Stages:** To count how many leads reached a stage (e.g., "Enrollment count"), you MUST count the non-null values in the corresponding timestamp column. **DO NOT** look for a column with a name like 'Enrollments'.
    *   **Correct Code:** `df[ts_col_map['Enrolled']].notna().sum()`
    *   **Incorrect Code:** `df['Enrollments'].sum()`
2.  **Conversion Rate:** (Count of Stage B) / (Count of Stage A).
3.  **Plotting:**
    *   For Matplotlib, end with `st.pyplot(plt.gcf())`.
    *   For Altair, end with `st.altair_chart(chart, use_container_width=True)`.
4.  **Final Output:** Your final answer MUST be rendered by `print()` for data/text, `st.pyplot()` for matplotlib, or `st.altair_chart()` for altair.

--- CONTEXT VARIABLES ---

The following variables are pre-loaded and available for your code to use:
- `df`: The main pandas DataFrame.
- `ordered_stages`: A list of the funnel stage names in order.
- `ts_col_map`: A dictionary mapping stage names to their timestamp column names. Here is the exact dictionary: """
    prompt += f"`{_ts_col_map_str}`\n"
    prompt += """
- `weights`: A dictionary of weights for performance scoring.

--- DATAFRAME `df` SCHEMA ---
"""
    prompt += _df_info
    prompt += "-----------------------------\n\n"
    prompt += "Your response MUST be a Python code block that starts with ```python and ends with ```. Do not provide any explanation or preamble."
    return prompt

@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

system_prompt = get_tool_prompt(get_df_info(df), str(ts_col_map), str(ordered_stages))

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

    full_prompt = system_prompt + f"\n\nUser Question: {user_prompt}"

    with st.spinner("AI Analyst is selecting a tool and generating code..."):
        try:
            response = model.generate_content(full_prompt)
            code_response = response.text.strip().replace("```python", "").replace("```", "").strip()
            
            with st.chat_message("assistant"):
                st.write("Generated Code:")
                st.code(code_response, language="python")
                st.session_state.messages.append({"role": "assistant", "content": f"**Generated Code:**\n```python\n{code_response}\n```"})
                
        except Exception as e:
            st.error(f"An error occurred while generating code: {e}")
            st.stop()
    
    with st.spinner("Executing code..."):
        try:
            execution_globals = {
                "__builtins__": __builtins__, "st": st, "pd": pd, "plt": plt, "alt": alt,
                "df": df, "ordered_stages": ordered_stages, "ts_col_map": ts_col_map,
                "weights": weights, "calculate_grouped_performance_metrics": calculate_grouped_performance_metrics,
                "calculate_avg_lag_generic": calculate_avg_lag_generic,
                "score_performance_groups": score_performance_groups,
                "format_performance_df": format_performance_df
            }
            
            with st.chat_message("assistant"):
                with st.expander("Execution Result", expanded=True):
                    output_buffer = StringIO()
                    import sys
                    original_stdout = sys.stdout
                    sys.stdout = output_buffer
                    
                    exec(code_response, execution_globals)
                    
                    sys.stdout = original_stdout
                    result_output = output_buffer.getvalue()
                    
                    if result_output:
                        st.text(result_output)
                        st.session_state.messages.append({"role": "assistant", "content": f"**Result:**\n```\n{result_output}\n```"})
                    else:
                        st.success("Code executed successfully and generated a plot.")
                        st.session_state.messages.append({"role": "assistant", "content": "Code executed successfully and generated a plot."})

        except Exception:
            error_traceback = traceback.format_exc()
            with st.chat_message("assistant"):
                st.error("An error occurred while executing the code:")
                st.code(error_traceback, language="bash")
            st.session_state.messages.append({"role": "assistant", "content": f"**Execution Error:**\n```\n{error_traceback}\n```"})