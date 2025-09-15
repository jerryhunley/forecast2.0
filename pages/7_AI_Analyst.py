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
# ... (this section is unchanged) ...
1.  **`calculate_grouped_performance_metrics(...)`**: ...
2.  **`calculate_avg_lag_generic(...)`**: ...
3.  **`pandas` and Visualization Libraries (...)**: ...

--- IMPORTANT BUSINESS DEFINITIONS & CODING RULES ---

1.  **How to Count Stages:** To count how many leads reached a stage (e.g., "Enrollment count"), you MUST count the non-null values in the corresponding timestamp column. Example: `df[ts_col_map['Enrolled']].notna().sum()`
2.  **Conversion Rate:** (Count of Stage B) / (Count of Stage A).
3.  **Final Output Rendering:** How you display the final answer depends on its type:
    *   **DataFrame:** Use `st.dataframe(result_df)`. DO NOT use `print()`.
    *   **Matplotlib plot:** End with `st.pyplot(plt.gcf())`.
    *   **Altair chart:** End with `st.altair_chart(chart, use_container_width=True)`.
    *   **Other (number, string, list):** Use `print()`.
4.  **DEFENSIVE CODING:** Your generated code MUST be robust. Before performing any division, you MUST check if the denominator is zero. If a calculation results in `NaN` or `inf`, you must handle it gracefully and print a user-friendly message instead of letting the code error out.

--- CONTEXT VARIABLES ---
# ... (this section is unchanged) ...
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