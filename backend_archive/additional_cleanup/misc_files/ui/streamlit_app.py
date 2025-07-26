#!/usr/bin/env python3
"""
SutazAI Streamlit User Interface

This module provides a web-based interface for interacting with the SutazAI
AGI/ASI system using Streamlit.
"""

import os
import json
import logging
import requests
import io
import base64
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.express as px
from PIL import Image

# Configure logging
# Ensure the main logs directory exists
LOGS_DIR = "/opt/sutazaiapp/logs"
os.makedirs(LOGS_DIR, exist_ok=True)
STREAMLIT_LOG_FILE = os.path.join(LOGS_DIR, "streamlit_ui.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(STREAMLIT_LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("StreamlitUI")

# API Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 120  # Seconds

# UI Configuration
st.set_page_config(
    page_title="SutazAI - Advanced AGI System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cache API requests
@st.cache_data(ttl=60)
def fetch_api_data(endpoint, params=None):
    """Fetch data from API with caching"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return {"status": "error", "message": f"API request failed: {str(e)}"}


def post_api_data(endpoint, data):
    """Post data to API"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.post(url, json=data, timeout=API_TIMEOUT)
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return {"status": "error", "message": f"API request failed: {str(e)}"}


def upload_file(endpoint, file):
    """Upload file to API"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        files = {"file": file}
        response = requests.post(url, files=files, timeout=API_TIMEOUT)
        return response.json()
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        return {"status": "error", "message": f"File upload failed: {str(e)}"}


# UI Helper Functions
def render_markdown(markdown_text):
    """Render markdown with custom styling"""
    st.markdown(markdown_text)


def render_code(code, language="python"):
    """Render code block with syntax highlighting"""
    st.code(code, language=language)


def download_button(object_to_download, download_filename, button_text):
    """Generate a download button for any object"""
    if isinstance(object_to_download, (pd.DataFrame, pd.Series)):
        object_to_download = object_to_download.to_csv(index=False)
        file_extension = "csv"
    elif isinstance(object_to_download, dict):
        object_to_download = json.dumps(object_to_download, indent=2)
        file_extension = "json"
    elif isinstance(object_to_download, (list, tuple)):
        object_to_download = "\n".join(map(str, object_to_download))
        file_extension = "txt"
    else:
        object_to_download = str(object_to_download)
        file_extension = "txt"

    # Create download button
    b64 = base64.b64encode(object_to_download.encode()).decode()
    button_uuid = f"download_{download_filename.replace('.', '_')}"
    custom_css = f"""
        <style>
            #{button_uuid} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25rem 0.75rem;
                border-radius: 0.25rem;
                border: 1px solid rgba(49, 51, 63, 0.2);
                font-size: 0.875rem;
                margin: 0px;
                line-height: 1.6;
            }}
            #{button_uuid}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
        </style>
    """
    download_link = (
        custom_css
        + f'<a id="{button_uuid}" href="data:file/{file_extension};base64,{b64}" download="{download_filename}">{button_text}</a>'
    )
    st.markdown(download_link, unsafe_allow_html=True)


# Set up session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_agent" not in st.session_state:
    st.session_state.current_agent = "LangChain Chat Agent"

if "system_info" not in st.session_state:
    st.session_state.system_info = {}


# Functions for different tabs
def chat_tab():
    """Chat interface for communicating with AI agents"""
    st.header("💬 SutazAI Chat Interface")

    # Agent selection sidebar
    with st.sidebar:
        st.subheader("Agent Configuration")

        # Get available agents
        agents = fetch_api_data("agents/list")
        if "agents" in agents:
            agent_options = [a["name"] for a in agents["agents"]]

            # Agent selection
            selected_agent = st.selectbox(
                "Select Agent",
                agent_options,
                index=agent_options.index(st.session_state.current_agent)
                if st.session_state.current_agent in agent_options
                else 0,
            )

            if selected_agent != st.session_state.current_agent:
                st.session_state.current_agent = selected_agent
                st.session_state.chat_history = []

        # Chat options
        st.subheader("Chat Options")
        # temperature = st.slider(
        #     "Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1
        # )
        # max_tokens = st.slider(
        #     "Max Response Length", min_value=50, max_value=4000, value=1000, step=50
        # )

        # Current agent details
        st.subheader("Current Agent Details")
        agent_info = next(
            (
                a
                for a in agents.get("agents", [])
                if a["name"] == st.session_state.current_agent
            ),
            None,
        )
        if agent_info:
            st.write(
                f"**Description:** {agent_info.get('description', 'No description')}"
            )
            st.write(f"**Model:** {agent_info.get('model', 'Unknown')}")
            st.write("**Capabilities:**")
            for capability in agent_info.get("capabilities", []):
                st.write(f"- {capability}")

    # Chat messages display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]

            if role == "user":
                st.markdown(
                    f"<div style='background-color:#e6f7ff;padding:10px;border-radius:5px;margin-bottom:10px;'><strong>You:</strong> {content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;margin-bottom:10px;'><strong>AI:</strong> {content}</div>",
                    unsafe_allow_html=True,
                )

    # User input
    # Use st.chat_input which is designed for this purpose
    prompt = st.chat_input("Type your message...")

    # Handle new input
    if prompt:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Rerun immediately to display the user message
        st.rerun()

    # Check if the last message was from the user and needs a response
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        # user_input_for_api = st.session_state.chat_history[-1]["content"]
        
        # Show loading indicator
        with st.spinner("AI is thinking..."):
            # Prepare API request
            chat_request = {
                "agent": st.session_state.current_agent,
                "messages": st.session_state.chat_history, # Send full history
                "parameters": {
                    # "temperature": st.session_state.get("temperature", 0.7), 
                    # "max_tokens": st.session_state.get("max_tokens", 1000)
                 },
            }

            # Make API request
            response = post_api_data("chat", chat_request)
            ai_response_content = "Error: Failed to get response from AI"
            if response:
                if "response" in response:
                     ai_response_content = response["response"]
                     # Log token usage if available
                     if "usage" in response:
                         logger.info(f"Token usage: {response['usage']}")
                elif "message" in response:
                     ai_response_content = f"Error: {response['message']}"
            
            # Add AI response to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": ai_response_content}
            )
            # Rerun to display AI response
            st.rerun()

    # Clear chat button (outside the main input flow)
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


def document_analysis_tab():
    """Document analysis interface for processing text and visual documents"""
    st.header("📄 Document Analysis")

    # Upload and analysis options
    with st.sidebar:
        st.subheader("Analysis Options")
        analysis_type = st.selectbox(
            "Analysis Type",
            [
                "Text Extraction",
                "Summarization",
                "Information Extraction",
                "Question Answering",
                "Visual Analysis",
            ],
        )

        if analysis_type == "Information Extraction":
            extraction_fields = st.text_area(
                "Information to Extract (comma-separated)",
                "dates, names, locations, key facts",
            )

        if analysis_type == "Question Answering":
            question = st.text_input("Question to Answer")

    # Main document upload and display area
    upload_col, display_col = st.columns([1, 1])

    with upload_col:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file", type=["pdf", "txt", "png", "jpg", "jpeg", "docx"]
        )

        if uploaded_file is not None:
            # Display file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type,
            }
            st.write(file_details)

            # Display file preview if possible
            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            elif uploaded_file.type == "application/pdf":
                st.write("PDF file uploaded. Analysis will process the content.")
            elif uploaded_file.type.startswith("text"):
                st.write("Text file uploaded. Preview of contents:")
                content = uploaded_file.read().decode()
                st.text_area(
                    "Content Preview",
                    content[:500] + ("..." if len(content) > 500 else ""),
                    height=200,
                    disabled=True,
                )
            else:
                st.write("File uploaded. Analysis will process the content.")

            # Analysis button
            analyze_button = st.button("Analyze Document")

            if analyze_button:
                with st.spinner("Analyzing document..."):
                    # Upload file to API
                    file_upload_response = upload_file(
                        "documents/upload", uploaded_file
                    )

                    if file_upload_response and "document_id" in file_upload_response:
                        document_id = file_upload_response["document_id"]

                        # Prepare analysis request based on analysis type
                        analysis_request = {
                            "document_id": document_id,
                            "analysis_type": analysis_type,
                        }

                        if analysis_type == "Information Extraction":
                            analysis_request["extraction_fields"] = [
                                field.strip()
                                for field in extraction_fields.split(",")
                                if field.strip()
                            ]

                        if analysis_type == "Question Answering":
                            analysis_request["question"] = question

                        # Make analysis request
                        analysis_response = post_api_data(
                            "documents/analyze", analysis_request
                        )

                        # Store analysis results in session state
                        st.session_state.analysis_results = analysis_response
                    else:
                        st.error("Failed to upload document")
                        st.session_state.analysis_results = {
                            "status": "error",
                            "message": file_upload_response.get(
                                "message", "Unknown error"
                            ),
                        }

    with display_col:
        st.subheader("Analysis Results")

        if "analysis_results" in st.session_state:
            results = st.session_state.analysis_results

            if results.get("status") == "success":
                # Display results based on analysis type
                if analysis_type == "Text Extraction":
                    st.write("Extracted Text:")
                    st.text_area(
                        "", results.get("text", "No text extracted"), height=400
                    )

                    # Add download button
                    download_button(
                        results.get("text", ""),
                        "extracted_text.txt",
                        "Download Extracted Text",
                    )

                elif analysis_type == "Summarization":
                    st.write("Document Summary:")
                    st.markdown(results.get("summary", "No summary available"))

                    # Show different summary lengths if available
                    if "summaries" in results:
                        summary_length = st.selectbox(
                            "Summary Length", ["short", "medium", "long"]
                        )
                        st.markdown(
                            results["summaries"].get(summary_length, "Not available")
                        )

                elif analysis_type == "Information Extraction":
                    st.write("Extracted Information:")

                    extracted_info = results.get("extracted_information", {})
                    for field, value in extracted_info.items():
                        st.write(f"**{field.title()}:** {value}")

                    # Add download button
                    download_button(
                        extracted_info,
                        "extracted_info.json",
                        "Download Extracted Information",
                    )

                elif analysis_type == "Question Answering":
                    st.write("Answer:")
                    st.markdown(f"**Question:** {question}")
                    st.markdown(
                        f"**Answer:** {results.get('answer', 'No answer available')}"
                    )

                    # Show confidence if available
                    if "confidence" in results:
                        confidence = results["confidence"]
                        st.progress(confidence)
                        st.write(f"Confidence: {confidence:.2f}")

                elif analysis_type == "Visual Analysis":
                    st.write("Visual Analysis:")

                    # Display original image with annotations if available
                    if "annotated_image" in results:
                        try:
                            image_data = base64.b64decode(results["annotated_image"])
                            annotated_image = Image.open(io.BytesIO(image_data))
                            st.image(
                                annotated_image,
                                caption="Annotated Image",
                                use_column_width=True,
                            )
                        except Exception as img_err:
                            logger.error(f"Error displaying annotated image: {img_err}")
                            st.warning("Could not display annotated image.")

                    # Display identified elements in expanders
                    if "elements" in results and results["elements"]:
                        st.write("Identified Elements:")
                        elements_list = results["elements"]
                        # Sort elements perhaps by position or type if available, otherwise by index
                        # Example: elements_list.sort(key=lambda x: (x.get('position', [0,0])[1], x.get('position', [0,0])[0]))

                        for i, element in enumerate(elements_list):
                            # Determine element type/label for expander title
                            element_type = element.get('type', 'Unknown')
                            element_label = element.get('label', f'Element {i+1}')
                            expander_title = f"{element_type.capitalize()}: {element_label}"

                            with st.expander(expander_title):
                                # Display all properties of the element
                                for key, value in element.items():
                                    st.write(f"**{key.replace('_', ' ').capitalize()}:** {value}")

                        # Keep the option to download the raw data
                        elements_df = pd.DataFrame(elements_list)
                        download_button(
                            elements_df, "visual_elements.csv", "Download All Elements Data"
                        )
                    elif "elements" in results:
                         st.info("No visual elements identified.")
            else:
                st.error(f"Analysis failed: {results.get('message', 'Unknown error')}")


def code_generation_tab():
    """Code generation and execution interface"""
    st.header("💻 Code Generation & Execution")

    # Setup sidebar options
    with st.sidebar:
        st.subheader("Generation Options")

        code_language = st.selectbox(
            "Programming Language",
            [
                "Python",
                "JavaScript",
                "TypeScript",
                "Java",
                "C++",
                "Rust",
                "Go",
                "Shell",
                "SQL",
                "Other",
            ],
        )

        generation_mode = st.selectbox(
            "Generation Mode", ["Complete", "Iterative", "Edit Existing"]
        )

        # Advanced options
        st.subheader("Advanced Options")

        # temp = st.slider(
        #     "Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1
        # )

        top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.95, step=0.01)

        use_comments = st.checkbox("Generate with Comments", value=True)

        # Only show test generation for Python
        if code_language == "Python":
            generate_tests = st.checkbox("Generate Tests", value=False)

    # Main code generation area
    input_col, output_col = st.columns([1, 1])

    with input_col:
        st.subheader("Requirements & Context")

        # Input for requirements
        requirements = st.text_area(
            "Describe what you want to build in detail:",
            height=150,
            help="Be specific about functionality, inputs, outputs, and any constraints.",
        )

        # Input for existing code if in edit mode
        if generation_mode == "Edit Existing":
            existing_code = st.text_area(
                "Existing Code to Edit:",
                height=250,
                help="Paste the code you want to edit or improve.",
            )

        # Generate button
        generate_button = st.button("Generate Code")

        if generate_button:
            with st.spinner("Generating code..."):
                # Prepare generation request
                generation_request = {
                    "requirements": requirements,
                    "language": code_language,
                    "mode": generation_mode,
                    "parameters": {
                        # "temperature": temp,
                        "top_p": top_p,
                        "use_comments": use_comments,
                    },
                }

                if generation_mode == "Edit Existing":
                    generation_request["existing_code"] = existing_code

                if (
                    code_language == "Python"
                    and generation_mode == "Complete"
                    and generate_tests
                ):
                    generation_request["generate_tests"] = True

                # Make API request
                response = post_api_data("code/generate", generation_request)

                # Store generation results
                st.session_state.generation_results = response

    with output_col:
        st.subheader("Generated Code")

        if "generation_results" in st.session_state:
            results = st.session_state.generation_results

            if results.get("status") == "success":
                # Display generated code
                generated_code = results.get("code", "")
                st.text_area("", generated_code, height=400)

                # Add download button
                extension_map = {
                    "Python": "py",
                    "JavaScript": "js",
                    "TypeScript": "ts",
                    "Java": "java",
                    "C++": "cpp",
                    "Rust": "rs",
                    "Go": "go",
                    "Shell": "sh",
                    "SQL": "sql",
                }
                extension = extension_map.get(code_language, "txt")
                download_button(
                    generated_code, f"generated_code.{extension}", "Download Code"
                )

                # Display tests if available
                if "tests" in results:
                    st.subheader("Generated Tests")
                    test_code = results["tests"]
                    st.text_area("", test_code, height=200)
                    download_button(test_code, f"tests.{extension}", "Download Tests")

                # Code execution (only for Python)
                if code_language == "Python":
                    st.subheader("Code Execution")
                    execute_code = st.button("Execute Code in Sandbox")

                    if execute_code:
                        with st.spinner("Executing code..."):
                            execution_request = {
                                "code": generated_code,
                                "language": code_language.lower(),
                            }

                            execution_response = post_api_data(
                                "code/execute", execution_request
                            )

                            if execution_response.get("status") == "success":
                                st.write("Execution successful!")
                                st.subheader("Output:")
                                st.code(
                                    execution_response.get("output", "No output"),
                                    language="bash",
                                )
                            else:
                                st.error("Execution failed")
                                st.subheader("Error:")
                                st.code(
                                    execution_response.get("error", "Unknown error"),
                                    language="bash",
                                )
            else:
                st.error(
                    f"Code generation failed: {results.get('message', 'Unknown error')}"
                )


def system_monitoring_tab():
    """System monitoring and diagnostics interface"""
    st.header("🔍 System Monitoring & Diagnostics")

    # Fetch system information
    with st.spinner("Fetching system information..."):
        system_info = fetch_api_data("system/status")
        st.session_state.system_info = system_info

    # Overview row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "System Status",
            system_info.get("status", "Unknown"),
            delta=None,
            delta_color="normal",
        )

    with col2:
        st.metric(
            "CPU Usage",
            f"{system_info.get('cpu_usage', 0):.1f}%",
            delta=f"{system_info.get('cpu_trend', 0):.1f}%"
            if "cpu_trend" in system_info
            else None,
            delta_color="inverse",
        )

    with col3:
        st.metric(
            "Memory Usage",
            f"{system_info.get('memory_usage', 0):.1f}%",
            delta=f"{system_info.get('memory_trend', 0):.1f}%"
            if "memory_trend" in system_info
            else None,
            delta_color="inverse",
        )

    with col4:
        st.metric(
            "Disk Usage",
            f"{system_info.get('disk_usage', 0):.1f}%",
            delta=f"{system_info.get('disk_trend', 0):.1f}%"
            if "disk_trend" in system_info
            else None,
            delta_color="inverse",
        )

    # Tabs for different system aspects
    system_tabs = st.tabs(["Performance", "Services", "Models", "Logs"])

    # Performance tab
    with system_tabs[0]:
        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            # CPU Usage Over Time
            if (
                "resource_history" in system_info
                and "cpu" in system_info["resource_history"]
            ):
                cpu_history = system_info["resource_history"]["cpu"]

                cpu_df = pd.DataFrame(
                    {
                        "Time": [
                            datetime.fromtimestamp(ts)
                            for ts in cpu_history["timestamps"]
                        ],
                        "CPU Usage (%)": cpu_history["values"],
                    }
                )

                cpu_fig = px.line(
                    cpu_df, x="Time", y="CPU Usage (%)", title="CPU Usage Over Time"
                )

                st.plotly_chart(cpu_fig, use_container_width=True)

        with perf_col2:
            # Memory Usage Over Time
            if (
                "resource_history" in system_info
                and "memory" in system_info["resource_history"]
            ):
                memory_history = system_info["resource_history"]["memory"]

                memory_df = pd.DataFrame(
                    {
                        "Time": [
                            datetime.fromtimestamp(ts)
                            for ts in memory_history["timestamps"]
                        ],
                        "Memory Usage (%)": memory_history["values"],
                    }
                )

                memory_fig = px.line(
                    memory_df,
                    x="Time",
                    y="Memory Usage (%)",
                    title="Memory Usage Over Time",
                )

                st.plotly_chart(memory_fig, use_container_width=True)

        # GPU monitoring if available
        if "gpu_info" in system_info and system_info["gpu_info"]:
            st.subheader("GPU Information")

            gpu_cols = st.columns(len(system_info["gpu_info"]))

            for i, gpu in enumerate(system_info["gpu_info"]):
                with gpu_cols[i]:
                    st.write(f"**GPU {i}:** {gpu.get('name', 'Unknown')}")
                    st.progress(gpu.get("usage", 0) / 100)
                    st.write(
                        f"Memory: {gpu.get('memory_used', 0)}/{gpu.get('memory_total', 0)} MB"
                    )
                    st.write(f"Temperature: {gpu.get('temperature', 0)}°C")

    # Services tab
    with system_tabs[1]:
        if "services" in system_info:
            services = system_info["services"]

            # Create dataframe for services
            services_df = pd.DataFrame(services)

            # Add status indicator
            def status_indicator(status):
                if status == "running":
                    return "🟢 Running"
                elif status == "starting":
                    return "🟡 Starting"
                elif status == "stopped":
                    return "🔴 Stopped"
                else:
                    return "⚪ Unknown"

            services_df["status_display"] = services_df["status"].apply(
                status_indicator
            )

            # Display services dataframe
            st.dataframe(
                services_df[
                    [
                        "name",
                        "status_display",
                        "uptime",
                        "pid",
                        "memory_usage",
                        "restarts",
                    ]
                ],
                column_config={
                    "name": "Service Name",
                    "status_display": "Status",
                    "uptime": "Uptime (s)",
                    "pid": "Process ID",
                    "memory_usage": "Memory Usage (MB)",
                    "restarts": "Restart Count",
                },
                use_container_width=True,
            )

            # Service control section
            st.subheader("Service Control")

            service_control_cols = st.columns(3)

            with service_control_cols[0]:
                service_to_control = st.selectbox(
                    "Select Service", options=[service["name"] for service in services]
                )

            with service_control_cols[1]:
                control_action = st.selectbox(
                    "Action", options=["Restart", "Stop", "Start"]
                )

            with service_control_cols[2]:
                execute_action = st.button("Execute")

                if execute_action:
                    action_request = {
                        "service": service_to_control,
                        "action": control_action.lower(),
                    }

                    with st.spinner(f"{control_action}ing {service_to_control}..."):
                        response = post_api_data(
                            "system/service_control", action_request
                        )

                        if response.get("status") == "success":
                            st.success(
                                f"{control_action} action on {service_to_control} was successful"
                            )
                        else:
                            st.error(
                                f"Failed to {control_action.lower()} {service_to_control}: {response.get('message', 'Unknown error')}"
                            )

    # Models tab
    with system_tabs[2]:
        if "models" in system_info:
            models_data = system_info["models"] # Get the raw dict

            # Convert the dictionary of models into a list of dictionaries
            # This is suitable for pd.DataFrame, ensuring model_id is a column
            models_list = []
            for model_id, model_info in models_data.items():
                if isinstance(model_info, dict): # Ensure it's a dict
                     model_info['model_id'] = model_id # Add model_id as a field
                     models_list.append(model_info)
                else:
                     logger.warning(f"Skipping invalid model data for {model_id}: {model_info}")

            if not models_list:
                 st.warning("No valid model data found.")
                 return # Exit this tab if no data

            # Create dataframe for models
            models_df = pd.DataFrame(models_list)

            # --- Convert ALL complex columns to strings for display --- 
            # Iterate through all columns
            for col in models_df.columns:
                 # Check if any cell in the column contains a dict or list
                 if any(isinstance(x, (dict, list)) for x in models_df[col]):
                     logger.info(f"Converting complex column '{col}' to JSON strings for display.")
                     try:
                        # Apply conversion to the entire column
                        models_df[col] = models_df[col].apply(
                            lambda x: json.dumps(x, indent=2) if isinstance(x, (dict, list)) else str(x)
                        )
                     except Exception as e:
                          logger.error(f"Error converting column '{col}' to string: {e}")
                          # Fallback: Convert entire column to string representations
                          models_df[col] = models_df[col].astype(str)
            # --------------------------------------------------------

            # Define columns to display - ensure model_id is included
            # Let's display all columns found after conversion for clarity
            existing_display_columns = models_df.columns.tolist()
            # display_columns = ['model_id', 'name', 'type', 'framework', 'capabilities', 'parameters']
            # existing_display_columns = [col for col in display_columns if col in models_df.columns]

            st.dataframe(
                models_df[existing_display_columns], # Display all processed columns
                use_container_width=True,
            )

            # Model loading/unloading interface (ensure it uses model_id)
            st.subheader("Model Management")

            model_mgmt_cols = st.columns(3)

            with model_mgmt_cols[0]:
                # Use the model_id from the DataFrame for selection
                available_model_ids = models_df['model_id'].tolist()
                # Add type annotation for model_to_manage
                model_to_manage: str = st.selectbox(
                    "Select Model ID", 
                    options=available_model_ids,
                    key="selected_model_monitor"
                )

            with model_mgmt_cols[1]:
                model_action = st.selectbox("Action", options=["Load", "Unload"], key="model_action_select")

            with model_mgmt_cols[2]:
                execute_model_action = st.button("Execute Model Action")

                if execute_model_action:
                    # The API endpoint likely expects the model_id
                    model_request = {
                        "model": model_to_manage, # Send the model_id
                        "action": model_action.lower(),
                    }

                    with st.spinner(f"{model_action}ing {model_to_manage}..."):
                        response = post_api_data("system/model_control", model_request)

                        if response.get("status") == "success":
                            st.success(
                                f"{model_action} action on {model_to_manage} was successful"
                            )
                            # Refresh data after action
                            st.cache_data.clear() 
                            st.rerun()
                        else:
                            st.error(
                                f"Failed to {model_action.lower()} {model_to_manage}: {response.get('message', 'Unknown error')}"
                            )
        else:
             st.warning("Model information not available from system status.")

    # Logs tab
    with system_tabs[3]:
        log_config_cols = st.columns(3)

        with log_config_cols[0]:
            log_service = st.selectbox(
                "Service",
                options=["All"]
                + [service["name"] for service in system_info.get("services", [])],
            )

        with log_config_cols[1]:
            log_level = st.selectbox(
                "Log Level", options=["ALL", "INFO", "WARNING", "ERROR", "DEBUG"]
            )

        with log_config_cols[2]:
            log_lines = st.slider(
                "Number of Lines", min_value=10, max_value=500, value=50, step=10
            )

        refresh_logs = st.button("Refresh Logs")

        # Fetch logs
        if refresh_logs or "logs" not in st.session_state:
            with st.spinner("Fetching logs..."):
                logs_request = {
                    "service": log_service,
                    "level": log_level,
                    "lines": log_lines,
                }

                logs_response = post_api_data("system/logs", logs_request)

                if logs_response.get("status") == "success":
                    st.session_state.logs = logs_response.get("logs", [])
                else:
                    st.error(
                        f"Failed to fetch logs: {logs_response.get('message', 'Unknown error')}"
                    )
                    st.session_state.logs = []

        # Display logs
        st.code("\n".join(st.session_state.logs), language="bash")

        # Add download button for logs
        download_button(
            "\n".join(st.session_state.logs), "sutazai_logs.txt", "Download Logs"
        )


# Main application layout
def main():
    """Main application entry point"""
    # Check API connection
    try:
        health_check = fetch_api_data("health")
        if health_check.get("status") != "ok":
            st.error(
                "Warning: Backend API is not responding correctly. Some features may not work."
            )
    except Exception as e:
        st.error(f"Error connecting to backend API: {e}")

    # Application header and navigation
    with st.sidebar:
        # Use absolute path to static assets, check if file exists
        # logo_path = "/opt/sutazaiapp/static/sutazai_logo.png"
        # if os.path.exists(logo_path):
        #     st.image(logo_path, width=100)
        # else:
        #     st.warning("Logo not found at expected location.")
        #     logger.warning(f"Logo file not found at {logo_path}")
        st.title("SutazAI System")

        # Display API Status
        api_status_msg = "Unknown"
        api_status_icon = "⚪"
        if 'health_check' in locals(): # Check if health_check variable exists
            if health_check.get("status") == "ok":
                api_status_msg = "Online"
                api_status_icon = "🟢"
            else:
                api_status_msg = "Offline"
                api_status_icon = "🔴"
        else: # API check might have failed earlier
            api_status_msg = "Error"
            api_status_icon = "🔴"

        st.markdown(f"**API Status:** {api_status_icon} {api_status_msg}")
        st.markdown("---")

        # Navigation tabs
        page = st.radio(
            "Navigate",
            [
                "💬 Chat",
                "📄 Document Analysis",
                "💻 Code Generation",
                "🔍 System Monitoring",
            ],
        )

        # System information display
        if "system_info" in st.session_state and st.session_state.system_info:
            st.subheader("System Information")

            system_info = st.session_state.system_info
            st.write(f"**Version:** {system_info.get('version', 'Unknown')}")
            st.write(f"**Uptime:** {system_info.get('uptime_hours', 0):.1f} hours")
            # Keep the existing overall system status indicator as well
            status = system_info.get("status", "unknown")
            if status == "healthy":
                st.success("System Status: Healthy")
            elif status == "degraded":
                st.warning("System Status: Degraded")
            elif status == "maintenance":
                st.info("System Status: Maintenance")
            else:
                st.error("System Status: Unknown")

        st.markdown("---")

        # Useful Links
        st.subheader("Useful Links")
        st.markdown(f"[🔗 API Docs]({API_BASE_URL}/docs)", unsafe_allow_html=True)
        # Placeholder for GitHub URL - replace with actual URL if known
        github_url = "https://github.com/yourusername/sutazai" # <-- TODO: Update this URL
        st.markdown(f"[🐙 GitHub Repo]({github_url})", unsafe_allow_html=True)
        st.markdown("---")

        # Theme Information
        st.subheader("Appearance")
        st.markdown(
            "Adjust the theme (Light/Dark) via the **Settings** menu:"
            "\n1. Click the **⚙️** icon (top right)."
            "\n2. Select **Settings**."
            "\n3. Choose your preferred **Theme**."
        )

        # Footer
        st.markdown("---")
        st.markdown("© 2023 SutazAI System")

    # Display selected page
    if page == "💬 Chat":
        chat_tab()
    elif page == "📄 Document Analysis":
        document_analysis_tab()
    elif page == "💻 Code Generation":
        code_generation_tab()
    elif page == "🔍 System Monitoring":
        system_monitoring_tab()


if __name__ == "__main__":
    main()
