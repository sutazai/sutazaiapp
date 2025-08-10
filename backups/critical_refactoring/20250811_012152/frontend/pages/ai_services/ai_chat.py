"""
AI Chat Page Module - Extracted from monolith
Advanced AI chat interface with model selection and conversation management
"""

import streamlit as st
import asyncio
from datetime import datetime
import sys
import os

# Import shared components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.api_client import call_api, handle_api_error
from utils.formatters import format_timestamp

def show_ai_chat():
    """AI Chat interface with enhanced features"""
    
    st.header("🤖 AI Chat Assistant", divider='blue')
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "tinyllama"
    
    # Model Selection and Settings
    with st.sidebar:
        st.subheader("🎛️ Chat Settings")
        
        # Model selection
        models = ["tinyllama", "gpt-oss"]  # Available models
        model_descriptions = {
            "tinyllama": "Fast, lightweight model (637MB)",
            "gpt-oss": "Advanced model (if available)"
        }
        
        selected_model = st.selectbox(
            "Select AI Model:",
            models,
            index=models.index(st.session_state.selected_model),
            help="Choose the AI model for conversation"
        )
        
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.rerun()
        
        st.markdown(f"*{model_descriptions.get(selected_model, 'Model description not available')}*")
        
        # Chat settings
        st.subheader("⚙️ Advanced Settings")
        
        use_cache = st.checkbox("Use Response Cache", value=True, help="Cache responses for faster replies")
        
        temperature = st.slider("Response Creativity", min_value=0.1, max_value=2.0, value=0.7, step=0.1,
                               help="Lower = more focused, Higher = more creative")
        
        max_tokens = st.slider("Max Response Length", min_value=50, max_value=2000, value=500, step=50,
                              help="Maximum tokens in AI response")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    
    # Chat Interface
    st.subheader("💬 Conversation")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            is_user = message["role"] == "user"
            
            if is_user:
                with st.chat_message("user"):
                    st.markdown(message["content"])
                    st.caption(f"📅 {format_timestamp(message.get('timestamp', ''))}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    
                    # Show metadata if available
                    metadata = message.get("metadata", {})
                    if metadata:
                        with st.expander("🔍 Response Details", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Model:** {metadata.get('model', 'Unknown')}")
                                st.markdown(f"**Tokens:** {metadata.get('tokens', 'N/A')}")
                            with col2:
                                st.markdown(f"**Response Time:** {metadata.get('response_time', 'N/A')}ms")
                                st.markdown(f"**Cached:** {'Yes' if metadata.get('cached') else 'No'}")
                    
                    st.caption(f"🤖 {format_timestamp(message.get('timestamp', ''))}")
    
    # Chat input
    st.subheader("✏️ Your Message")
    
    # Create a form for better UX
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Type your message:",
            placeholder="Ask me anything about SutazAI, AI, technology, or general topics...",
            height=100,
            help="Press Ctrl+Enter to send"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            send_button = st.form_submit_button("🚀 Send Message", use_container_width=True)
        
        with col2:
            if st.form_submit_button("💡 Suggest", use_container_width=True):
                suggestions = [
                    "Explain how SutazAI works",
                    "What are your capabilities?",
                    "Help me optimize my workflow",
                    "Analyze this data for me",
                    "Generate a code example"
                ]
                user_input = suggestions[len(st.session_state.chat_history) % len(suggestions)]
                send_button = True
        
        with col3:
            if st.form_submit_button("🎯 Focus Mode", use_container_width=True):
                st.session_state.focus_mode = not st.session_state.get("focus_mode", False)
                st.rerun()
    
    # Process chat message
    if send_button and user_input.strip():
        # Add user message to history
        user_message = {
            "role": "user",
            "content": user_input.strip(),
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.chat_history.append(user_message)
        
        # Show loading indicator
        with st.spinner(f"🤖 {selected_model} is thinking..."):
            try:
                # Prepare chat request
                chat_request = {
                    "message": user_input.strip(),
                    "model": selected_model,
                    "use_cache": use_cache,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "conversation_history": st.session_state.chat_history[-10:]  # Last 10 messages for context
                }
                
                # Call AI chat API
                response = asyncio.run(call_api("/api/v1/chat", method="POST", data=chat_request))
                
                if response and handle_api_error(response, "AI chat"):
                    # Add AI response to history
                    ai_message = {
                        "role": "assistant", 
                        "content": response.get("response", "Sorry, I couldn't generate a response."),
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "model": selected_model,
                            "tokens": response.get("tokens_used", "N/A"),
                            "response_time": response.get("response_time_ms", "N/A"),
                            "cached": response.get("from_cache", False)
                        }
                    }
                    st.session_state.chat_history.append(ai_message)
                    
                    # Show success feedback
                    st.success("✅ Response generated successfully!")
                    
                else:
                    # Add error message
                    error_message = {
                        "role": "assistant",
                        "content": "Sorry, I'm having trouble connecting to the AI service. Please try again later.",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {"error": True}
                    }
                    st.session_state.chat_history.append(error_message)
                    
            except Exception as e:
                st.error(f"Chat error: {str(e)}")
                
                # Add error message to history
                error_message = {
                    "role": "assistant", 
                    "content": f"Error: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"error": True}
                }
                st.session_state.chat_history.append(error_message)
        
        # Refresh the interface
        st.rerun()
    
    # Focus Mode (minimal UI)
    if st.session_state.get("focus_mode", False):
        st.markdown("""
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 12px;
                max-width: 600px;
                width: 90%;
            ">
                <h3>🎯 Focus Mode Active</h3>
                <p>Minimal interface for distraction-free chatting</p>
                <button onclick="window.parent.postMessage({type: 'streamlit:toggleFocusMode'}, '*')">
                    Exit Focus Mode
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Add custom CSS for better chat styling  
st.markdown("""
<style>
.chat-message {
    margin: 1rem 0;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1a73e8;
}

.chat-message.user {
    background-color: rgba(26, 115, 232, 0.1);
    border-left-color: #1a73e8;
}

.chat-message.assistant {
    background-color: rgba(52, 168, 83, 0.1); 
    border-left-color: #34a853;
}

.chat-timestamp {
    font-size: 0.8em;
    color: #666;
    margin-top: 0.5rem;
}

div[data-testid="stChatMessage"] {
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)