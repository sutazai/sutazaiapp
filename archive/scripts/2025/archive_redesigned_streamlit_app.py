
import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="SutazAI - Redesigned",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Toggle ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def apply_theme():
    light_theme = """
    <style>
        .main-header { font-size: 2.5rem; color: #2c3e50; text-align: center; margin-bottom: 2rem; }
        .sidebar .sidebar-content { background-color: #f8f9fa; }
        .stButton>button { background-color: #3498db; color: white; }
    </style>
    """
    dark_theme = """
    <style>
        body { color: #ecf0f1; background-color: #2c3e50; }
        .main-header { color: #ecf0f1; }
        .sidebar .sidebar-content { background-color: #34495e; }
        .stButton>button { background-color: #2980b9; }
        .stTextInput, .stTextArea, .stSelectbox { background-color: #34495e; }
    </style>
    """
    if st.session_state.theme == "dark":
        st.markdown(dark_theme, unsafe_allow_html=True)
    else:
        st.markdown(light_theme, unsafe_allow_html=True)

apply_theme()

# --- API Communication ---
BACKEND_URL = "http://localhost:8000"

def get_data(endpoint):
    try:
        response = requests.get(f"{BACKEND_URL}{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# --- Page Rendering Functions ---

def render_dashboard():
    st.title("📊 Dashboard")
    st.write("Welcome to the redesigned SutazAI dashboard! Here's a real-time overview of your system.")

    # --- Key Metrics ---
    # In a real app, you'd get this from get_data("/api/metrics")
    metrics = {
        "total_requests": 1234,
        "active_agents": 5,
        "avg_response_time": 0.45,
        "success_rate": 98.5
    }

    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Requests", metrics.get("total_requests", 0), "↗️")
        with col2:
            st.metric("Active Agents", metrics.get("active_agents", 0))
        with col3:
            st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s", "↘️")
        with col4:
            st.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")
    else:
        st.info("Metrics data is not available at the moment.")


    # --- Charts ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Response Time Trend")
        # Placeholder data
        times = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='1H')
        response_times = [0.5 + (i * 0.02) + ((-1)**i * 0.1) for i in range(len(times))] # some variation
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=response_times, mode='lines+markers', name='Response Time'))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Agent Utilization")
        # Placeholder data
        agents = ["AutoGPT", "LocalAGI", "TabbyML", "Semgrep", "Others"]
        utilization = [35, 25, 18, 12, 10]
        fig = px.pie(values=utilization, names=agents, hole=0.4, title="Agent Task Distribution")
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

def render_chat_interface():
    st.title("💬 Chat Interface")

    # --- Model Selection ---
    # In a real app, you'd get this from get_data("/api/models/")
    models = ["deepseek-coder:33b", "mistral:7b", "llama3.2:3b", "tinyllama"]
    selected_model = st.selectbox("Select a model", models)


    # --- Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input ---
    if prompt := st.chat_input("What would you like to discuss?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                # In a real app, this would call the backend API
                # For now, we'll simulate a response
                simulated_response = f"This is a simulated response from **{selected_model}** about: *'{prompt}'*."
                st.markdown(simulated_response)
        st.session_state.messages.append({"role": "assistant", "content": simulated_response})
        st.rerun()

def render_agent_management():
    st.title("🤖 Agent Management")
    st.write("Manage your AI agents and create new ones.")

    # --- Agent Listing ---
    st.subheader("Available Agents")

    # In a real app, you'd get this from get_data("/api/agents/")
    agents = [
        {"name": "Code-Genius", "type": "Code Generation", "status": "active", "model": "deepseek-coder:33b"},
        {"name": "Summarizer", "type": "Summarization", "status": "active", "model": "mistral:7b"},
        {"name": "ChatBot", "type": "General Q&A", "status": "inactive", "model": "llama3.2:3b"},
        {"name": "Researcher", "type": "Research", "status": "active", "model": "tinyllama"},
    ]

    if agents:
        for agent in agents:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                with col1:
                    st.subheader(agent['name'])
                    st.caption(f"Type: {agent['type']}")
                with col2:
                    st.text(f"Model: {agent['model']}")
                with col3:
                    st.text(f"Status: {agent['status'].capitalize()}")
                with col4:
                    if st.button("Delete", key=f"delete_{agent['name']}"):
                        st.success(f"Agent '{agent['name']}' deleted.") # Placeholder
                st.divider()

    else:
        st.info("No agents found.")


    # --- Create New Agent ---
    with st.expander("Create a New Agent"):
        with st.form("new_agent_form", clear_on_submit=True):
            st.subheader("New Agent Configuration")
            name = st.text_input("Agent Name")
            agent_type = st.selectbox("Agent Type", ["Code Generation", "Summarization", "General Q&A", "Research"])
            model = st.selectbox("Underlying Model", ["deepseek-coder:33b", "mistral:7b", "llama3.2:3b", "tinyllama"])

            submitted = st.form_submit_button("Create Agent")
            if submitted:
                # In a real app, this would call the backend API
                st.success(f"Agent '{name}' of type '{agent_type}' created successfully!")

def render_system_metrics():
    st.title("📈 System Metrics")
    st.write("Detailed performance and resource usage metrics.")

    # --- Real-time Data Simulation ---
    # In a real app, this data would be fetched periodically from the backend
    cpu_usage = [random.randint(20, 60) for _ in range(30)]
    memory_usage = [random.randint(40, 70) for _ in range(30)]
    gpu_usage = [random.randint(10, 80) for _ in range(30)]
    time_labels = [datetime.now() - timedelta(seconds=i) for i in range(30)][::-1]

    # --- Resource Usage Charts ---
    st.subheader("Live Resource Usage")
    
    # Create a dataframe for the chart
    chart_data = pd.DataFrame({
        'Time': time_labels,
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (%)': memory_usage,
        'GPU Usage (%)': gpu_usage
    })

    # Create the line chart
    st.line_chart(chart_data.rename(columns={'Time':'index'}).set_index('index'))


    # --- API Performance ---
    st.subheader("API Endpoint Performance")
    
    # Placeholder data
    api_perf_data = {
        "Endpoint": ["/api/chat", "/api/code", "/api/docs", "/api/agents", "/api/models"],
        "Requests": [520, 350, 190, 160, 100],
        "Avg Response (ms)": [240, 175, 290, 115, 85],
        "Success Rate (%)": [99.3, 98.9, 99.6, 100.0, 100.0]
    }
    api_perf_df = pd.DataFrame(api_perf_data)
    st.dataframe(api_perf_df, use_container_width=True)


def render_task_automation():
    st.title("⚙️ Task Automation")
    st.write("Automate complex tasks by selecting a task type and providing the necessary inputs.")

    # --- Task Selection ---
    task_type = st.selectbox(
        "Select a task to automate",
        ["Code Generation", "Document Processing", "Web Automation", "Data Analysis"]
    )

    # --- Dynamic Task UI ---
    if task_type == "Code Generation":
        st.subheader("Code Generation Task")
        language = st.selectbox("Language", ["Python", "JavaScript", "Go", "Rust"])
        description = st.text_area("Describe the code you want to generate:", height=150)
        if st.button("Generate Code"):
            with st.spinner("🤖 Generating code..."):
                # Simulate API call
                st.success("Code generated successfully!")
                st.code(f"# {language.capitalize()} code for: {description}", language=language.lower())

    elif task_type == "Document Processing":
        st.subheader("Document Processing Task")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
        action = st.selectbox("Action", ["Summarize", "Extract Keywords", "Translate"])
        if uploaded_file and st.button("Process Document"):
            with st.spinner("📄 Processing document..."):
                # Simulate API call
                st.success(f"Document '{uploaded_file.name}' processed.")
                st.write(f"**Action:** {action}")
                st.info("Results would be displayed here.")

    elif task_type == "Web Automation":
        st.subheader("Web Automation Task")
        url = st.text_input("Target URL", "https://example.com")
        action = st.selectbox("Action", ["Scrape Content", "Take Screenshot", "Fill Form"])
        if st.button("Run Automation"):
            with st.spinner("🌐 Running web automation..."):
                # Simulate API call
                st.success(f"Web automation task completed on {url}.")
                st.write(f"**Action:** {action}")
                st.info("Results would be displayed here.")

    elif task_type == "Data Analysis":
        st.subheader("Data Analysis Task")
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        query = st.text_input("What do you want to analyze in the data?", "Show me the average sales per region.")
        if uploaded_file and st.button("Analyze Data"):
            with st.spinner("📊 Analyzing data..."):
                # Simulate API call
                st.success(f"Analysis of '{uploaded_file.name}' complete.")
                st.write(f"**Query:** {query}")
                # Placeholder for chart
                df = pd.DataFrame({
                    'Region': ['North', 'South', 'East', 'West'],
                    'Average Sales': [150, 120, 180, 135]
                })
                st.dataframe(df)
                st.bar_chart(df.set_index('Region'))



def render_settings():
    st.title("🔧 Settings")
    st.write("Configure the system and model settings.")

    # --- API Configuration ---
    st.subheader("API Configuration")
    api_key = st.text_input("API Key", type="password", value="xxxxxxxxxx")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    if st.button("Save API Settings"):
        st.success("API settings saved.") # Placeholder

    st.divider()

    # --- Model Settings ---
    st.subheader("Model Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 8000, 2048, 100)
    if st.button("Save Model Settings"):
        st.success("Model settings saved.") # Placeholder

    st.divider()

    # --- Security Settings ---
    st.subheader("Security")
    enable_auth = st.checkbox("Enable Authentication", value=True)
    enable_rate_limiting = st.checkbox("Enable Rate Limiting", value=True)
    if st.button("Save Security Settings"):
        st.success("Security settings saved.") # Placeholder


# --- Main Application ---
def main():
    st.markdown('<h1 class="main-header">SutazAI Redesigned</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", [
            "Dashboard",
            "Chat Interface",
            "Agent Management",
            "System Metrics",
            "Task Automation",
            "Settings"
        ])

        st.header("Theme")
        if st.button("Toggle Theme"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()

        if page == "Chat Interface":
            st.header("Chat Controls")
            if st.button("Clear Chat History"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.rerun()


    if page == "Dashboard":
        render_dashboard()
    elif page == "Chat Interface":
        render_chat_interface()
    elif page == "Agent Management":
        render_agent_management()
    elif page == "System Metrics":
        render_system_metrics()
    elif page == "Task Automation":
        render_task_automation()
    elif page == "Settings":
        render_settings()

if __name__ == "__main__":
    main()
