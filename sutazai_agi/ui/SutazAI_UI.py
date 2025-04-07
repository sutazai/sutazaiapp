import streamlit as st
import logging

# Configure logging (ensure it's configured globally, perhaps in core/config_loader)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="SutazAI - AGI/ASI System",
    page_icon="🤖", # You can use an emoji or a path to an icon file
    layout="wide", # Can be "wide" or "centered"
    initial_sidebar_state="expanded", # Can be "auto", "expanded", "collapsed"
    menu_items={
        'Get Help': 'https://github.com/sutazai/sutazai-agi', # Replace with actual repo link
        'Report a bug': "https://github.com/sutazai/sutazai-agi/issues",
        'About': """
        # SutazAI AGI/ASI System
        
        An autonomous, offline AI system based on the provided blueprint.
        
        **Warning:** This is a developmental system.
        """
    }
)

# --- Main Page Content --- 
# This file acts as the landing page when multi-page apps structure is used.
# The actual functionality will be in the files inside the 'pages/' directory.

logger.info("Main SutazAI UI page loaded.")

st.title("Welcome to SutazAI 🤖")

st.sidebar.success("Select a function above.")

st.markdown("""
## SutazAI AGI/ASI System

This is the user interface for the SutazAI system, an autonomous, offline Artificial General Intelligence / Artificial Superintelligence prototype.

Please select a function from the sidebar to begin interacting with the system.

**Available Modules (Planned):**
*   **Chatbot:** Engage in conversation with various AI agents.
*   **Document Analysis:** Upload and analyze documents (PDF, DOCX, TXT).
*   **Code Assistant:** Generate, debug, and edit code.
*   **Financial Analysis:** Perform AI-powered financial tasks.
*   **System Status:** Monitor system performance and loaded models (Admin).
*   **API Console:** Test backend API endpoints (Admin/Developer).

*Note: Ensure the backend server and Ollama are running before using the UI.*
""")

# You can add system status indicators or other general info here if desired.
# For example, try connecting to the backend to show status:
# import requests
# try:
#     response = requests.get("http://localhost:8000/") # Assuming default backend URL
#     if response.status_code == 200:
#         st.sidebar.success("Backend Status: Connected")
#     else:
#         st.sidebar.error(f"Backend Status: Error ({response.status_code})")
# except requests.exceptions.ConnectionError:
#     st.sidebar.error("Backend Status: Disconnected") 