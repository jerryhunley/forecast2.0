# pages/7_AI_Analyst.py
import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Analyst",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Analyst")
st.info("""
Ask questions about your recruitment data in plain English. 
This AI assistant can perform data analysis and answer questions based on the uploaded files.

**Example questions:**
- "How many total qualified referrals did we get?"
- "What is the overall conversion rate from 'Passed Online Form' to 'Signed ICF'?"
- "Which 5 sites have the most enrollments?"
""")

# --- Page Guard ---
if not st.session_state.get('data_processed_successfully', False):
    st.warning("Please upload and process your data on the 'Home & Data Setup' page first.")
    st.stop()

# --- Load Data from Session State ---
# We'll work with the main processed DataFrame
df = st.session_state.referral_data_processed

# --- Configure the Gemini API ---
try:
    # Get the API key from Streamlit secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    # Initialize the model
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error("Error configuring the AI model. Have you set your GEMINI_API_KEY in Streamlit's secrets? Contact support if the issue persists.")
    st.exception(e) # Also show the full error for debugging
    st.stop()


# --- Main Chat Logic ---

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- System Prompt Construction ---
# This is where we give the LLM its instructions and the data.
# We will use a simplified version of the DataFrame's info to save on tokens.
@st.cache_data
def get_dataframe_schema(df):
    prompt = "You are a helpful and expert data analyst for a clinical trial recruitment company.\n"
    prompt += "You will be given a question from a user and a pandas DataFrame named `df`.\n"
    prompt += "Your task is to answer the user's question based on the provided DataFrame.\n\n"
    prompt += "Here is the schema of the DataFrame `df`:\n"
    prompt += f"Number of rows: {len(df)}\n"
    prompt += "Columns and their data types:\n"
    for col, dtype in df.dtypes.items():
        prompt += f"- {col}: {dtype}\n"
    prompt += "\nWhen providing an answer, be concise and clear. If you are asked to provide a list, use bullet points."
    return prompt

system_prompt = get_dataframe_schema(df)

# --- Handle User Input ---
if user_prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Construct the full prompt for the API
    full_prompt = system_prompt + f"\n\nUser Question: {user_prompt}"

    # Display a thinking spinner and get the response
    with st.spinner("The AI Analyst is thinking..."):
        try:
            response = model.generate_content(full_prompt)
            ai_response = response.text
        except Exception as e:
            ai_response = f"An error occurred while communicating with the AI model: {e}"

    # Add AI response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)