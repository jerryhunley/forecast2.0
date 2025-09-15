# pages/7_AI_Analyst.py
import streamlit as st
import pandas as pd
import google.generativeai as genai
from io import StringIO
import traceback

# Direct import from the root directory
from constants import *

# --- Page Configuration ---
st.set_page_config(page_title="AI Analyst", page_icon="🤖", layout="wide")

st.title("🤖 AI Code-Executing Analyst")
st.info("""
Ask questions about your recruitment data. The AI will write and execute Python (pandas) code to answer your questions.

**Example questions:**
- "How many total qualified referrals did we get?"
- "What is the overall conversion rate from 'Passed Online Form' to 'Signed ICF'?"
- "Which 5 sites have the most enrollments? Show the result as a table."
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
    # Using a more powerful model is better for code generation
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    st.error("Error configuring the AI model. Have you set your GEMINI_API_KEY in Streamlit's secrets?")
    st.exception(e)
    st.stop()

# --- System Prompt Construction with Code-Gen Instructions ---
@st.cache_data
def get_system_prompt(df, ts_col_map):
    prompt = "You are a world-class Python data analyst. You are an expert in the pandas library.\n"
    prompt += "You will be given a question from a user about the data in a pandas DataFrame named `df`.\n"
    prompt += "Your ONLY task is to respond with a single, executable Python code block to answer the user's question. Do not provide any explanation, preamble, or markdown formatting.\n\n"
    
    prompt += "--- IMPORTANT BUSINESS DEFINITIONS ---\n"
    prompt += f"1. A 'Qualified Lead' or 'Total Qualified Referral' is defined as any lead that has a valid (not null) timestamp in the `{ts_col_map.get(STAGE_PASSED_ONLINE_FORM)}` column. To count them, you should count the non-null values in this column.\n"
    prompt += f"2. An 'Enrollment' is defined as any lead that has a valid (not null) timestamp in the `{ts_col_map.get(STAGE_ENROLLED)}` column.\n"
    prompt += f"3. An 'ICF' or 'Signed ICF' is defined as any lead that has a valid (not null) timestamp in the `{ts_col_map.get(STAGE_SIGNED_ICF)}` column.\n"
    prompt += "4. When asked for a conversion rate between Stage A and Stage B, calculate it as `(count of leads reaching Stage B) / (count of leads reaching Stage A)`.\n"
    prompt += "-------------------------------------\n\n"

    prompt += "--- DATAFRAME `df` SCHEMA ---\n"
    buffer = StringIO()
    df.info(buf=buffer)
    prompt += buffer.getvalue()
    prompt += "-----------------------------\n\n"
    
    prompt += "Your response MUST be a Python code block that starts with ```python and ends with ```. The code should use the `print()` function to output its final answer."
    return prompt

system_prompt = get_system_prompt(df, ts_col_map)

# --- Main Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if the content is code and display it accordingly
        if "code" in message and message["code"]:
            st.code(message["content"], language="python")
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
            # Clean the response to get only the code block
            code_response = response.text.strip().replace("```python", "").replace("```", "").strip()
            
            # Display the generated code
            st.session_state.messages.append({"role": "assistant", "content": code_response, "code": True})
            with st.chat_message("assistant"):
                st.code(code_response, language="python")
                
        except Exception as e:
            st.error(f"An error occurred while generating code: {e}")
            st.stop()
    
    with st.spinner("Executing code..."):
        try:
            # Create a string buffer to capture the print output
            output_buffer = StringIO()
            
            # Create a dictionary of local variables for exec
            local_vars = {'df': df, 'pd': pd, 'st': st}
            
            # Redirect stdout to the buffer and execute the code
            import sys
            original_stdout = sys.stdout
            sys.stdout = output_buffer
            
            exec(code_response, {"__builtins__": __builtins__}, local_vars)
            
            # Restore stdout
            sys.stdout = original_stdout
            
            # Get the result from the buffer
            result = output_buffer.getvalue()

            # Display the result
            st.session_state.messages.append({"role": "assistant", "content": f"**Result:**\n\n{result}", "code": False})
            with st.chat_message("assistant"):
                st.markdown(f"**Result:**\n\n{result}")

        except Exception as e:
            st.error(f"An error occurred while executing the code:")
            st.code(traceback.format_exc(), language="bash")