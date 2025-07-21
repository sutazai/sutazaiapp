import streamlit as st
import requests
import json

st.set_page_config(
    page_title="SutazAI Control Panel",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SutazAI Control Panel")

# Backend URL - using localhost since both are on same machine
BACKEND_URL = "http://localhost:8000"

# Check backend connection
try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=2)
    if response.status_code == 200:
        st.success("✅ Backend connected!")
        health_data = response.json()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Get system status
        status_response = requests.get(f"{BACKEND_URL}/api/system/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            
            with col1:
                st.metric("Status", "🟢 Operational")
            with col2:
                st.metric("Agents", status_data.get("agents", 0))
            with col3:
                st.metric("Models", status_data.get("models", 0))
            with col4:
                st.metric("Requests", status_data.get("requests", 0))
        
        # Services status
        st.subheader("Services")
        services = health_data.get("services", {})
        cols = st.columns(len(services))
        for idx, (service, status) in enumerate(services.items()):
            with cols[idx]:
                if status == "connected":
                    st.success(f"✅ {service.capitalize()}")
                else:
                    st.error(f"❌ {service.capitalize()}")
        
        # Chat interface
        st.subheader("Chat Interface")
        user_input = st.text_input("Ask me anything:")
        if user_input:
            chat_response = requests.post(
                f"{BACKEND_URL}/api/v1/chat",
                json={"content": user_input}
            )
            if chat_response.status_code == 200:
                st.write("**Response:**", chat_response.json()["response"])
        
        # Models
        st.subheader("Available Models")
        models_response = requests.get(f"{BACKEND_URL}/api/models")
        if models_response.status_code == 200:
            models = models_response.json()["models"]
            for model in models:
                st.write(f"• {model['name']} ({model['size']})")
        
        # Agents
        st.subheader("AI Agents")
        agents_response = requests.get(f"{BACKEND_URL}/api/agents")
        if agents_response.status_code == 200:
            agents = agents_response.json()["agents"]
            agent_cols = st.columns(4)
            for idx, agent in enumerate(agents[:4]):
                with agent_cols[idx]:
                    st.info(f"**{agent['name']}**\nStatus: {agent['status']}")
        
except requests.exceptions.RequestException as e:
    st.error(f"❌ Backend error: {str(e)}")
    st.info("Backend URL: " + BACKEND_URL)
    st.info("Make sure the backend is running on port 8000")

st.markdown("---")
st.caption("SutazAI v9.0 - AGI/ASI System")
