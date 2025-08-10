"""
AI Chat Page Module - Extracted from monolith
Advanced AI chat interface with model selection and conversation management
"""
import streamlit as st
import asyncio
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.api_client import call_api, handle_api_error
from utils.formatters import format_timestamp

def show_ai_chat():
    """AI Chat interface with enhanced features"""
    st.header('🤖 AI Chat Assistant', divider='blue')
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'tinyllama'
    with st.sidebar:
        st.subheader('🎛️ Chat Settings')
        models = ['tinyllama', 'gpt-oss']
        model_descriptions = {'tinyllama': 'Fast, lightweight model (637MB)', 'gpt-oss': 'Advanced model (if available)'}
        selected_model = st.selectbox('Select AI Model:', models, index=models.index(st.session_state.selected_model), help='Choose the AI model for conversation')
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.rerun()
        st.markdown(f'*{model_descriptions.get(selected_model, 'Model description not available')}*')
        st.subheader('⚙️ Advanced Settings')
        use_cache = st.checkbox('Use Response Cache', value=True, help='Cache responses for faster replies')
        temperature = st.slider('Response Creativity', min_value=0.1, max_value=2.0, value=0.7, step=0.1, help='Lower = more focused, Higher = more creative')
        max_tokens = st.slider('Max Response Length', min_value=50, max_value=2000, value=500, step=50, help='Maximum tokens in AI response')
        if st.button('🗑️ Clear Chat History', use_container_width=True):
            st.session_state.chat_history = []
            st.success('Chat history cleared!')
            st.rerun()
    st.subheader('💬 Conversation')
    chat_container = st.container()
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            is_user = message['role'] == 'user'
            if is_user:
                with st.chat_message('user'):
                    st.markdown(message['content'])
                    st.caption(f'📅 {format_timestamp(message.get('timestamp', ''))}')
            else:
                with st.chat_message('assistant'):
                    st.markdown(message['content'])
                    metadata = message.get('metadata', {})
                    if metadata:
                        with st.expander('🔍 Response Details', expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f'**Model:** {metadata.get('model', 'Unknown')}')
                                st.markdown(f'**Tokens:** {metadata.get('tokens', 'N/A')}')
                            with col2:
                                st.markdown(f'**Response Time:** {metadata.get('response_time', 'N/A')}ms')
                                st.markdown(f'**Cached:** {('Yes' if metadata.get('cached') else 'No')}')
                    st.caption(f'🤖 {format_timestamp(message.get('timestamp', ''))}')
    st.subheader('✏️ Your Message')
    with st.form('chat_form', clear_on_submit=True):
        user_input = st.text_area('Type your message:', placeholder='Ask me anything about SutazAI, AI, technology, or general topics...', height=100, help='Press Ctrl+Enter to send')
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            send_button = st.form_submit_button('🚀 Send Message', use_container_width=True)
        with col2:
            if st.form_submit_button('💡 Suggest', use_container_width=True):
                suggestions = ['Explain how SutazAI works', 'What are your capabilities?', 'Help me optimize my workflow', 'Analyze this data for me', 'Generate a code example']
                user_input = suggestions[len(st.session_state.chat_history) % len(suggestions)]
                send_button = True
        with col3:
            if st.form_submit_button('🎯 Focus Mode', use_container_width=True):
                st.session_state.focus_mode = not st.session_state.get('focus_mode', False)
                st.rerun()
    if send_button and user_input.strip():
        user_message = {'role': 'user', 'content': user_input.strip(), 'timestamp': datetime.now().isoformat()}
        st.session_state.chat_history.append(user_message)
        with st.spinner(f'🤖 {selected_model} is thinking...'):
            try:
                chat_request = {'message': user_input.strip(), 'model': selected_model, 'use_cache': use_cache, 'temperature': temperature, 'max_tokens': max_tokens, 'conversation_history': st.session_state.chat_history[-10:]}
                response = asyncio.run(call_api('/api/v1/chat', method='POST', data=chat_request))
                if response and handle_api_error(response, 'AI chat'):
                    ai_message = {'role': 'assistant', 'content': response.get('response', "Sorry, I couldn't generate a response."), 'timestamp': datetime.now().isoformat(), 'metadata': {'model': selected_model, 'tokens': response.get('tokens_used', 'N/A'), 'response_time': response.get('response_time_ms', 'N/A'), 'cached': response.get('from_cache', False)}}
                    st.session_state.chat_history.append(ai_message)
                    st.success('✅ Response generated successfully!')
                else:
                    error_message = {'role': 'assistant', 'content': "Sorry, I'm having trouble connecting to the AI service. Please try again later.", 'timestamp': datetime.now().isoformat(), 'metadata': {'error': True}}
                    st.session_state.chat_history.append(error_message)
            except Exception as e:
                st.error(f'Chat error: {str(e)}')
                error_message = {'role': 'assistant', 'content': f'Error: {str(e)}', 'timestamp': datetime.now().isoformat(), 'metadata': {'error': True}}
                st.session_state.chat_history.append(error_message)
        st.rerun()
    if st.session_state.get('focus_mode', False):
        st.markdown('\n        <div style="\n            position: fixed;\n            top: 0;\n            left: 0;\n            right: 0;\n            bottom: 0;\n            background: rgba(0, 0, 0, 0.8);\n            z-index: 999;\n            display: flex;\n            align-items: center;\n            justify-content: center;\n        ">\n            <div style="\n                background: white;\n                padding: 2rem;\n                border-radius: 12px;\n                max-width: 600px;\n                width: 90%;\n            ">\n                <h3>🎯 Focus Mode Active</h3>\n                <p>Minimal interface for distraction-free chatting</p>\n                <button onclick="window.parent.postMessage({type: \'streamlit:toggleFocusMode\'}, \'*\')">\n                    Exit Focus Mode\n                </button>\n            </div>\n        </div>\n        ', unsafe_allow_html=True)
st.markdown('\n<style>\n.chat-message {\n    margin: 1rem 0;\n    padding: 1rem;\n    border-radius: 8px;\n    border-left: 4px solid #1a73e8;\n}\n\n.chat-message.user {\n    background-color: rgba(26, 115, 232, 0.1);\n    border-left-color: #1a73e8;\n}\n\n.chat-message.assistant {\n    background-color: rgba(52, 168, 83, 0.1); \n    border-left-color: #34a853;\n}\n\n.chat-timestamp {\n    font-size: 0.8em;\n    color: #666;\n    margin-top: 0.5rem;\n}\n\ndiv[data-testid="stChatMessage"] {\n    margin-bottom: 1rem;\n}\n</style>\n', unsafe_allow_html=True)