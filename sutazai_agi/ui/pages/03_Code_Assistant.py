import streamlit as st
import requests
import logging
import os
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Backend URL (ensure this matches the actual backend host/port)
# BACKEND_URL = "http://localhost:8000/api/v1"
BACKEND_URL = "http://172.31.77.193:8000/api/v1" # Use the host IP from backend logs
CODE_ENDPOINT = f"{BACKEND_URL}/code"

# --- Helper Functions --- 

def generate_code_on_backend(prompt: str, project_name: str) -> Optional[Dict[str, Any]]:
    """Sends a code generation request to the backend."""
    url = f"{CODE_ENDPOINT}/generate"
    payload = {"prompt": prompt, "project_name": project_name}
    logger.info(f"Sending code generation request for project '{project_name}' to {url}")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Code generation response: {result}")
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Error during code generation request: {e}")
        logger.error(f"Failed to send code generation request: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during code generation: {e}")
        logger.error(f"Unexpected error during code generation API call: {e}", exc_info=True)
        return None

def edit_code_on_backend(files: List[str], instruction: str, repo_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Sends a code editing request to the backend."""
    url = f"{CODE_ENDPOINT}/edit"
    payload = {"files": files, "instruction": instruction, "repo_path": repo_path}
    logger.info(f"Sending code edit request for files {files} to {url}")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Code edit response: {result}")
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Error during code edit request: {e}")
        logger.error(f"Failed to send code edit request: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during code editing: {e}")
        logger.error(f"Unexpected error during code edit API call: {e}", exc_info=True)
        return None

# --- Streamlit Page Setup --- 
st.set_page_config(page_title="SutazAI Code Assistant", page_icon="💻")
st.title("💻 Code Assistant")

logger.info("Code Assistant UI page loaded.")

# --- UI Tabs --- 
tab1, tab2 = st.tabs(["Generate Codebase (GPT-Engineer)", "Edit Code Files (Aider)"])

with tab1:
    st.header("Generate New Codebase")
    st.caption("Uses GPT-Engineer to create a new project based on your description.")

    project_name = st.text_input("Project Name (e.g., my_new_app)", key="gen_project_name", help="A unique name for the new project directory.")
    prompt = st.text_area("Project Description / Prompt", key="gen_prompt", height=200, help="Describe the application you want to generate in detail (features, language, frameworks, etc.).")

    if st.button("Generate Codebase", key="gen_button"):
        if project_name and prompt:
            with st.spinner(f"Generating codebase for '{project_name}'..."):
                gen_result = generate_code_on_backend(prompt=prompt, project_name=project_name)
                # Store result for display
                st.session_state.generation_result = gen_result 
        else:
            st.warning("Please provide both a project name and a description.")

    # Display Generation Result
    if 'generation_result' in st.session_state:
        result = st.session_state.generation_result
        if result:
            st.subheader("Generation Result")
            if result.get("status") == "success":
                st.success(result.get("message", "Success!"))
                st.info(f"Output Path: {result.get('output_path', 'N/A')}")
                st.text_area("Log Output", value=result.get("log", "No logs."), height=150, disabled=True)
            else:
                st.error(result.get("message", "Code generation failed."))
                st.text_area("Log Output (stderr/stdout)", value=result.get("log", "") + "\n" + result.get("output", ""), height=150, disabled=True)
        # Clear result after displaying once? Or keep until next run?
        # del st.session_state.generation_result 

with tab2:
    st.header("Edit Existing Code Files")
    st.caption("Uses Aider to modify files based on your instructions. Ensure files are in the workspace.")
    
    # File selection (simplified for now, could use a file browser widget later)
    st.warning("File selection is simplified. Enter paths relative to the agent workspace.")
    file_paths_str = st.text_input("File Paths (comma-separated)", key="edit_files", help="e.g., my_project/main.py, my_project/utils.py")
    # Optional: Specify repo path if Aider should run within a specific git context
    repo_path_str = st.text_input("Git Repo Path (optional, relative to workspace)", key="edit_repo", help="e.g., my_project. Leave blank to run in workspace root.")

    instruction = st.text_area("Editing Instruction", key="edit_instruction", height=150, help="Describe the changes you want to make.")

    if st.button("Edit Files", key="edit_button"):
        if file_paths_str and instruction:
            files_list = [f.strip() for f in file_paths_str.split(',') if f.strip()]
            repo_path = repo_path_str.strip() if repo_path_str else None
            if files_list:
                 with st.spinner(f"Editing files {files_list} with Aider..."):
                    edit_result = edit_code_on_backend(files=files_list, instruction=instruction, repo_path=repo_path)
                    # Store result for display
                    st.session_state.edit_result = edit_result
            else:
                 st.warning("Please enter valid file paths.")
        else:
            st.warning("Please provide file paths and an editing instruction.")

    # Display Edit Result
    if 'edit_result' in st.session_state:
        result = st.session_state.edit_result
        if result:
            st.subheader("Editing Result")
            if result.get("status") == "success":
                st.success(result.get("message", "Success!"))
                # Aider's output often contains the diff or applied changes
                st.text_area("Aider Output (Diff/Log)", value=result.get("output", "No output.") + "\n" + result.get("log", ""), height=200, disabled=True)
            else:
                st.error(result.get("message", "Code editing failed."))
                st.text_area("Aider Output (stdout/stderr)", value=result.get("output", "") + "\n" + result.get("log", ""), height=200, disabled=True)
        # Clear result after displaying once? 
        # del st.session_state.edit_result 