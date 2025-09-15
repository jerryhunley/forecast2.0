# pages/7_AI_Analyst.py
import streamlit as st
import pandas as pd
import google.generativeai as genai

# Direct import from the root directory
from constants import *

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Analyst",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Analyst")
st.info("""
Ask questions about your recruitment data in plain English. 
This AI assistant has been given the business logic from the other pages to provide more accurate answers.

**Example questions:**
- "How many total qualified leads have we gotten?" (Now works!)
- "What is the overall conversion rate from 'Passed Online Form' to 'Signed ICF'?"
- "Which 5 sites have the most enrollments?"
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
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error("Error configuring the AI model. Have you set your GEMINI_API_KEY in Streamlit's secrets?")
    st.exception(e)
    st.stop()

# --- NEW: System Prompt Construction with Business Logic ---
@st.cache_data
def get_system_prompt(df, ts_col_map):
    prompt = "You are a helpful and expert data analyst for a clinical trial recruitment company.\n"
    prompt += "You will be given a question from a user and a pandas DataFrame named `df`.\n"
    prompt += "Your task is to answer the user's question based on the provided DataFrame and the business definitions below.\n\n"

    # --- Add Business Definitions ---
    prompt += "--- IMPORTANT BUSINESS DEFINITIONS ---\n"
    prompt += f"1.  A **'Qualified Lead'** or **'Total Qualified Referral'** is defined as any lead that has a valid (not null) timestamp in the `{ts_col_map.get(STAGE_PASSED_ONLINE_FORM)}` column. To count them, you should count the non-null values in this column.\n"
    prompt += f"2.  An **'Enrollment'** is defined as any lead that has a valid (not null) timestamp in the `{ts_col_map.get(STAGE_ENROLLED)}` column.\n"
    prompt += f"3.  An **'ICF'** or **'Signed ICF'** is defined as any lead that has a valid (not null) timestamp in the `{ts_col_map.get(STAGE_SIGNED_ICF)}` column.\n"
    prompt += "4.  When asked for a **conversion rate** between Stage A and Stage B, you must calculate it as `(count of leads reaching Stage B) / (count of leads reaching Stage A)`.\n"
    prompt += "-------------------------------------\n\n"

    # --- Add DataFrame Schema ---
    prompt += "Here is the schema of the DataFrame `df`:\n"
    prompt += f"Number of rows: {len(df)}\n"
    prompt += "Columns and their data types:\n"
    # Using a string buffer for efficiency
    from io import StringIO
    buffer = StringIO()
    df.info(buf=buffer)
    prompt += buffer.getvalue()
    
    prompt += "\nWhen providing an answer, be concise and clear. If you are asked to provide a list, use bullet points."
    return prompt

system_prompt = get_system_prompt(df, ts_col_map)

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

    full_prompt = system_prompt + f"\n\nHere is the full DataFrame data in CSV format for your reference:\n```csv\n{df.to_csv(index=False)}\n```\n\nUser Question: {user_prompt}"

    with st.spinner("The AI Analyst is thinking..."):
        try:
            # For this basic version, we pass the full data context each time.
            # In a more advanced version, we would have the AI generate and execute code.
            response = model.generate_content(full_prompt)
            ai_response = response.text
        except Exception as e:
            ai_response = f"An error occurred while communicating with the AI model: {e}"

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)