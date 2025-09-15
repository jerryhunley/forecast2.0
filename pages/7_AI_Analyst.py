# pages/7_AI_Analyst.py
import streamlit as st
import pandas as pd
import google.generativeai as genai
from io import StringIO
import traceback
import matplotlib.pyplot as plt # Import for st.pyplot
import altair as alt # Import for st.altair_chart

# Direct import from the root directory
from constants import *

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 AI Code-Executing Analyst")
st.info("""
Ask questions about your data. The AI will write and execute Python code to answer you.
**You can now ask for visualizations!**

**Example questions:**
- "How many total qualified referrals are there?"
- "Create a bar chart of the top 5 sites by enrollment count."
- "Show a line chart of qualified leads over time by month."
""")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data from Session State ---
df = st.session_state.referral_data_processed
ts_col_map = st.session_state.ts_col_map

# --- Configure the Gemini API ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    st.error("Error configuring the AI model. Have you set your GEMINI_API_KEY in Streamlit's secrets?")
    st.exception(e)
    st.stop()

# --- NEW: System Prompt with Visualization Instructions ---
@st.cache_data
def get_system_prompt(df_info, ts_col_map):
    prompt = "You are a world-class Python data analyst. You are an expert in the pandas, matplotlib, and altair libraries.\n"
    prompt += "You will be given a question from a user about the data in a pandas DataFrame named `df`.\n"
    prompt += "Your ONLY task is to respond with a single, executable Python code block to answer the user's question. Do not provide any explanation, preamble, or markdown formatting.\n\n"
    
    prompt += "--- IMPORTANT BUSINESS DEFINITIONS ---\n"
    prompt += f"1. A 'Qualified Lead' is any lead with a valid timestamp in the `{ts_col_map.get(STAGE_PASSED_ONLINE_FORM)}` column.\n"
    prompt += f"2. An 'Enrollment' is any lead with a valid timestamp in the `{ts_col_map.get(STAGE_ENROLLED)}` column.\n"
    prompt += f"3. An 'ICF' is any lead with a valid timestamp in the `{ts_col_map.get(STAGE_SIGNED_ICF)}` column.\n"
    prompt += "4. Conversion Rate (Stage A to B) = (count of B) / (count of A).\n"
    prompt += "-------------------------------------\n\n"

    # --- NEW: Visualization Instructions ---
    prompt += "--- VISUALIZATION INSTRUCTIONS ---\n"
    prompt += "1.  When a user asks for a chart, graph, or plot, you MUST generate the code to create it.\n"
    prompt += "2.  You have two options for plotting: Matplotlib or Altair.\n"
    prompt += "3.  **To display a Matplotlib plot:** Generate the plot as usual, but instead of `plt.show()`, you MUST end your code with `st.pyplot(plt.gcf())`.\n"
    prompt += "4.  **To display an Altair chart:** Create the chart object (e.g., `chart = alt.Chart(...)`) and end your code with `st.altair_chart(chart, use_container_width=True)`.\n"
    prompt += "5.  For any other output (like a number, a list, or a DataFrame), use the `print()` function.\n"
    prompt += "-------------------------------------\n\n"
    
    prompt += "--- DATAFRAME `df` SCHEMA ---\n"
    prompt += df_info
    prompt += "-----------------------------\n\n"
    
    prompt += "Your response MUST be a Python code block that starts with ```python and ends with ```."
    return prompt

# Cache the DataFrame info to avoid re-computing it
@st.cache_data
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

system_prompt = get_system_prompt(get_df_info(df), ts_col_map)

# --- Main Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("is_code", False):
            st.code(message["content"], language="python")
        elif message.get("is_plot", False):
            # The result of a plot is the plot object itself
            st.pyplot(message["content"])
        elif message.get("is_altair", False):
            st.altair_chart(message["content"], use_container_width=True)
        else:
            st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    full_prompt = system_prompt + f"\n\nUser Question: {user_prompt}"

    with st.spinner("AI Analyst is generating code..."):
        try:
            response = model.generate_content(full_prompt)
            code_response = response.text.strip().replace("```python", "").replace("```", "").strip()
            
            st.session_state.messages.append({"role": "assistant", "content": code_response, "is_code": True})
            with st.chat_message("assistant"):
                st.code(code_response, language="python")
                
        except Exception as e:
            st.error(f"An error occurred while generating code: {e}")
            st.stop()
    
    with st.spinner("Executing code..."):
        try:
            output_buffer = StringIO()
            import sys
            original_stdout = sys.stdout
            sys.stdout = output_buffer
            
            # The environment where the code will be executed
            # It needs access to all libraries and the DataFrame
            local_vars = {
                "df": df, 
                "pd": pd, 
                "st": st,
                "plt": plt,
                "alt": alt
            }
            
            exec(code_response, {"__builtins__": __builtins__}, local_vars)
            
            sys.stdout = original_stdout
            result = output_buffer.getvalue()

            if result: # If there was any print output
                st.session_state.messages.append({"role": "assistant", "content": f"**Result:**\n\n{result}", "is_code": False})
                with st.chat_message("assistant"):
                    st.markdown(f"**Result:**\n\n{result}")

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**An error occurred while executing the code:**\n\n{traceback.format_exc()}",
                "is_code": False
            })
            with st.chat_message("assistant"):
                st.error(f"An error occurred while executing the code:")
                st.code(traceback.format_exc(), language="bash")