import streamlit as st
import requests
import json
import re
from openai import OpenAI  # OpenAI integration
import yaml
from io import BytesIO
import time
from datetime import datetime
import logging
import ollama

# Set custom Ollama host
ollama_host = "http://localhost:11434"
ollama._client._base_url = ollama_host

# Page Configuration
st.set_page_config(page_title="Smart API Orchestrator", layout="wide")

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sidebar Configuration
st.sidebar.header("⚙️ Settings")

# Model Selection
model_choice = st.sidebar.radio("🧠 Choose AI Model", ["Gemini (Google)", "OpenAI (GPT-4)", "LLaMA 3.2 (Local)"])

# API Key Input
api_key = st.sidebar.text_input("🔑 API Key", type="password")

# Output Format Selection
output_format = st.sidebar.selectbox("📄 Output Format", ["JSON", "Python", "YAML"])

# Advanced Settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Advanced Features")
enable_error_handling = st.sidebar.checkbox("Enable Error Handling", value=True)
enable_api_testing = st.sidebar.checkbox("Enable API Testing", value=False)
enable_dark_mode = st.sidebar.checkbox("Enable Dark Mode", value=False)

# API Testing Settings
if enable_api_testing:
    test_endpoint = st.sidebar.text_input("Test API Endpoint", value="https://jsonplaceholder.typicode.com/posts")
    test_method = st.sidebar.selectbox("Test Method", ["GET", "POST", "PUT", "DELETE"])
    test_payload = st.sidebar.text_area("Test Payload (JSON)", value='{"key": "value"}')
    custom_headers = st.sidebar.text_area("Custom Headers (JSON)", value='{"Content-Type": "application/json"}')
    query_params = st.sidebar.text_area("Query Parameters (JSON)", value='{"param1": "value1"}')

st.sidebar.markdown("---")

# Dark Mode
if enable_dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to fetch response from Gemini API
def fetch_gemini_response(user_input):
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    gemini_prompt = {"contents": [{"parts": [{"text": f"Generate a {output_format} workflow and Python code to execute: {user_input}"}]}]}
    
    response = requests.post(GEMINI_API_URL, json=gemini_prompt)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("❌ Error fetching response from Gemini API.")
        logger.error(f"Gemini API Error: {response.status_code} - {response.text}")
        return None

# Function to fetch response from OpenAI API
def fetch_openai_response(user_input):
    openai.api_key = api_key
    prompt_text = f"Generate a {output_format} workflow and corresponding Python code to execute this task: {user_input}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Change to "gpt-4" if needed
            messages=[{"role": "user", "content": prompt_text}]
        )
        # Structure the response to match Gemini's format
        structured_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": response["choices"][0]["message"]["content"]
                            }
                        ]
                    }
                }
            ]
        }
        return structured_response

    except Exception as e:
        st.error(f"⚠️ OpenAI API Error: {e}")
        logger.error(f"OpenAI API Error: {e}")
    return None

# Function to fetch response from local LLaMA 3.2
def fetch_llama_response(user_input):
    try:
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": f"Generate a structured {output_format} output and Python code to execute: {user_input}"}])
        # Structure the response to match Gemini's format
        structured_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": response.get("message", {}).get("content", "")
                            }
                        ]
                    }
                }
            ]
        }
        return structured_response
    except Exception as e:
        st.error(f"⚠️ Error fetching response from LLaMA 3.2: {e}")
        logger.error(f"LLaMA API Error: {e}")
        return None

# Function to extract JSON, Python, YAML, Markdown, JavaScript, Bash
def extract_content(raw_text, format_type):
    patterns = {
        "JSON": r"```json\n(.*?)\n```",
        "Python": r"```python\n(.*?)\n```",
        "YAML": r"```yaml\n(.*?)\n```"
    }
    match = re.search(patterns[format_type], raw_text, re.DOTALL)
    return match.group(1) if match else None

# Function to test API
def test_api(endpoint, method, payload=None, headers=None, params=None):
    try:
        if method == "GET":
            response = requests.get(endpoint, headers=headers, params=params)
        elif method == "POST":
            response = requests.post(endpoint, json=json.loads(payload), headers=headers, params=params)
        elif method == "PUT":
            response = requests.put(endpoint, json=json.loads(payload), headers=headers, params=params)
        elif method == "DELETE":
            response = requests.delete(endpoint, headers=headers, params=params)
        
        if response.status_code == 200:
            st.success("✅ API Test Successful!")
            st.json(response.json())
        else:
            st.error(f"❌ API Test Failed with Status Code: {response.status_code}")
            logger.error(f"API Test Failed: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"⚠️ API Test Error: {e}")
        logger.error(f"API Test Error: {e}")

# UI Header
st.markdown("<h1 style='text-align: center;'>🚀 Smart API Orchestrator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Automate API workflows with AI! 🤖</p>", unsafe_allow_html=True)

# User Input
user_input = st.text_area("✍️ Enter Your API Task", "e.g., 'Send me daily weather updates via WhatsApp'")

if st.button("⚡ Generate API Workflow"):
    if user_input.strip():
        with st.spinner("🔄 Generating your workflow... Please wait!"):
            gemini_output = None
            openai_output = None
            llama_output = None
            if model_choice == "Gemini (Google)":
                gemini_output = fetch_gemini_response(user_input)
            elif model_choice == "OpenAI (GPT-4)":
                openai_output = fetch_openai_response(user_input)
            elif model_choice == "LLaMA 3.2 (Local)":
                llama_output = fetch_llama_response(user_input)

        

            if gemini_output:
                raw_text = gemini_output["candidates"][0]["content"]["parts"][0]["text"]
            elif openai_output:
                raw_text = openai_output["candidates"][0]["content"]["parts"][0]["text"]
            elif llama_output:
                raw_text = llama_output["candidates"][0]["content"]["parts"][0]["text"]

            if raw_text:
                try:
                    # Extract Data based on selected format
                    extracted_content = extract_content(raw_text, output_format)
                    python_code = extract_content(raw_text, "Python")

                    if extracted_content and python_code:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"📄 Generated {output_format} Workflow")
                            if output_format == "JSON":
                                extracted_content = json.loads(re.sub(r'//.*', '', extracted_content))
                                st.json(extracted_content)
                            elif output_format == "YAML":
                                st.code(extracted_content, language="yaml")
                         
                            else:
                                st.code(extracted_content, language="json")

                        with col2:
                            st.subheader("💻 Generated Python Code")
                            st.code(python_code, language="python")

                        # Save to history
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.append((user_input, extracted_content, python_code))

                        st.success("✅ Workflow and Code Generated Successfully!")

                        # Download options
                        st.download_button("📥 Download Workflow", json.dumps(extracted_content, indent=4), "workflow.json", "application/json")
                        st.download_button("📥 Download Python Code", python_code, "script.py", "text/plain")

                        # User Feedback
                        feedback = st.radio("Rate the generated workflow:", ["👍", "👎"])
                        if feedback:
                            logger.info(f"User Feedback: {feedback} for task: {user_input}")
                    else:
                        st.error("⚠️ Error extracting workflow or Python code.")
                        # st.code(raw_text, language="text")
                except json.JSONDecodeError as e:
                    st.error(f"🚨 JSON Parsing Error: {e}")
                    st.code(raw_text, language="text")
    else:
        st.warning("⚠️ Please enter a valid API task!")

# API Testing Section
if enable_api_testing:
    st.markdown("---")
    st.subheader("🔧 API Testing")
    if st.button("🚀 Run API Test"):
        with st.spinner("🔄 Testing API..."):
            headers = json.loads(custom_headers) if custom_headers else None
            params = json.loads(query_params) if query_params else None
            test_api(test_endpoint, test_method, test_payload, headers, params)

# History Section
if "history" in st.session_state and st.session_state.history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 History")
    for idx, (task, workflow, code) in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5
        with st.sidebar.expander(f"Task {idx+1}: {task[:30]}..."):
            st.json(workflow)
            st.code(code, language="python")