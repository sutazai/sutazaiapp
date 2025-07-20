import streamlit as st
import requests

def chat_interface(BACKEND_URL):
    """
    Renders the chat interface.
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Intelligent Chat")
        
        # Chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get AI response
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/api/v1/chat/completions",
                        json={
                            "messages": [{"role": "user", "content": prompt}],
                            "model": st.session_state.selected_model
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        ai_response = response.json()["choices"][0]["message"]["content"]
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    else:
                        st.error(f"Backend error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.rerun()
    
    with col2:
        st.header("🎯 Quick Prompts")
        
        prompts = {
            "📝 Generate Code": "Generate a Python function to ",
            "🔍 Analyze Code": "Analyze this code for improvements: ",
            "🐛 Debug Issue": "Help me debug this error: ",
            "📚 Explain Concept": "Explain the concept of ",
            "🏗️ System Design": "Design a system architecture for ",
            "🔒 Security Check": "Check this code for security issues: "
        }
        
        for label, prompt_start in prompts.items():
            if st.button(label, use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt_start
                })
                st.rerun()