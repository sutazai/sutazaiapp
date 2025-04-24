import streamlit as st
import requests
import logging
import json
from typing import Dict, List, Optional, Any
import uuid # Import uuid module

logger = logging.getLogger(__name__)

# Backend URL (should ideally be configurable)
# BACKEND_URL = "http://localhost:8000/api/v1" # Default, ensure FastAPI is running here
# Use the host IP address identified during backend startup
BACKEND_URL = "http://172.31.77.193:8000/api/v1" 
CHAT_ENDPOINT = f"{BACKEND_URL}/chat/completions" # Corrected endpoint path
AGENTS_ENDPOINT = f"{BACKEND_URL}/agents"

# --- Helper Functions --- 

def get_available_agents() -> List[Dict[str, Any]]:
    """Fetches the list of enabled agents from the backend."""
    try:
        response = requests.get(AGENTS_ENDPOINT) # Use constant
        response.raise_for_status() # Raise exception for bad status codes
        data = response.json()
        return data.get("agents", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend to fetch agents: {e}")
        logger.error(f"Failed to fetch agents from backend: {e}")
        return []
    except json.JSONDecodeError:
        st.error("Error decoding agent list from backend.")
        logger.error("Failed to decode JSON response for agent list.")
        return []

def send_chat_message(messages: List[Dict[str, str]], agent_name: Optional[str], session_id: str) -> Optional[Dict[str, Any]]:
    """Sends the chat message history to the backend API in OpenAI format."""
    if not agent_name:
        st.error("No agent selected.")
        return None
        
    # Construct payload matching ChatCompletionRequest model
    payload = {
        "model": agent_name,       # Use 'model' key for agent name
        "messages": messages,      # Send the entire message history
        "session_id": session_id,  # Keep session_id 
        "stream": False          # Assuming non-streaming for now
    }
    logger.debug(f"Sending payload to {CHAT_ENDPOINT}: {payload}")
    try:
        # Use the corrected endpoint constant
        response = requests.post(CHAT_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending message to backend: {e}")
        logger.error(f"Failed to send chat message to backend: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding response from backend.")
        logger.error("Failed to decode JSON response from chat endpoint.")
        return None

# --- Streamlit Page Setup --- 

st.set_page_config(page_title="SutazAI Chatbot", page_icon="💬")
st.title("💬 SutazAI Chatbot")

logger.info("Chatbot UI page loaded.")

# --- Sidebar for Agent Selection --- 

with st.sidebar:
    st.header("Chat Options")
    agents = get_available_agents()
    agent_names = [agent["name"] for agent in agents if agent.get("enabled")]
    
    if not agent_names:
        st.warning("No agents available from backend. Ensure backend is running and agents are configured.")
        selected_agent = None
    else:
        # Add a "Default" option maybe?
        # agent_names.insert(0, "Default Agent") 
        selected_agent = st.selectbox(
            "Select Agent:", 
            options=agent_names, 
            index=0, # Default to the first available agent
            help="Choose the AI agent to interact with."
        )
        # Display agent description if available
        for agent in agents:
            if agent["name"] == selected_agent:
                st.info(f"**Description:** {agent.get('description', 'N/A')}")
                break

    st.markdown("--- ")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Chat Interface --- 

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.debug("Initialized chat history in session state.")

# Initialize a unique session ID if it doesn't exist
if "session_id" not in st.session_state:
    # Generate a unique ID using UUID
    st.session_state.session_id = f"ui_session_{uuid.uuid4()}"
    logger.debug(f"Initialized session ID: {st.session_state.session_id}")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("What would you like to ask?"):
    logger.info(f"User input received: '{prompt[:50]}...'")
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send message to backend and get response
    if selected_agent:
        with st.spinner(f"Asking {selected_agent}..."): 
            logger.debug(f"Sending query to backend. Agent: {selected_agent}, Session: {st.session_state.session_id}")
            # Pass the entire message history to send_chat_message
            response_data = send_chat_message(
                messages=st.session_state.messages, 
                agent_name=selected_agent, 
                session_id=st.session_state.session_id
            )
            
            if response_data and "choices" in response_data and response_data["choices"]:
                # Initialize ai_response before try block
                ai_response = "*Error processing backend response.*" # Default error message
                try:
                    # Attempt to parse standard OpenAI response structure
                    ai_response = response_data['choices'][0]['message']['content']
                    if not ai_response: # Handle empty content
                        ai_response = "*Received empty response.*"
                    # Correctly indented logger call
                    logger.info(f"Received successful response from agent: '{ai_response[:50]}...'")
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Failed to parse AI response structure: {e}. Response: {response_data}")
                    ai_response = f"*Error parsing backend response: {e}*" # Keep specific error
                    st.error(f"Error parsing response from {selected_agent}. Response: {response_data}")

                # Add AI response (or error message from parsing) to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                # Display AI response
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
            elif response_data:
                # This case might now indicate structured errors *within* a 200 OK response,
                # which shouldn't happen with the current backend logic but is kept for resilience.
                error_msg = response_data.get("message", "Unknown error from backend.")
                # Try extracting detail if it follows FastAPI error format
                if "detail" in response_data:
                    error_msg = response_data["detail"]
                logger.error(f"Backend returned error: {error_msg}")
                st.error(f"Error from {selected_agent}: {error_msg}")
                # Optionally add error message to history?
            else:
                # Error already shown by send_chat_message
                logger.error("No valid response data received from backend.")
                # Optionally add error message to history?
                # st.session_state.messages.append({"role": "assistant", "content": "*Error: Failed to get response from backend.*"})
    else:
        st.warning("Please select an available agent from the sidebar to chat.") 