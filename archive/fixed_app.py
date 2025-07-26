import streamlit as st
import requests
import json
import os

st.set_page_config(
    page_title="SutazAI Control Panel",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SutazAI Control Panel")

# Backend URL - use environment variable or host IP
BACKEND_HOST = os.getenv("BACKEND_HOST", "172.31.77.193")
BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

# Show current backend URL
st.sidebar.info(f"Backend URL: {BACKEND_URL}")

# Check backend connection
try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if response.status_code == 200:
        st.success("✅ Backend connected!")
        health_data = response.json()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Get system status
        try:
            status_response = requests.get(f"{BACKEND_URL}/api/system/status", timeout=5)
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
        except:
            st.warning("Could not fetch system status")
        
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
            try:
                chat_response = requests.post(
                    f"{BACKEND_URL}/api/v1/chat",
                    json={"content": user_input},
                    timeout=5
                )
                if chat_response.status_code == 200:
                    st.write("**Response:**", chat_response.json()["response"])
            except Exception as e:
                st.error(f"Chat error: {str(e)}")
        
        # Models
        st.subheader("Available Models")
        try:
            models_response = requests.get(f"{BACKEND_URL}/api/models", timeout=5)
            if models_response.status_code == 200:
                models = models_response.json()["models"]
                for model in models:
                    st.write(f"• {model['name']} ({model['size']})")
        except:
            st.info("Models information unavailable")
        
        # Agents
        st.subheader("AI Agents")
        try:
            agents_response = requests.get(f"{BACKEND_URL}/api/agents", timeout=5)
            if agents_response.status_code == 200:
                agents = agents_response.json()["agents"]
                agent_cols = st.columns(4)
                for idx, agent in enumerate(agents[:4]):
                    with agent_cols[idx]:
                        st.info(f"**{agent['name']}**\nStatus: {agent['status']}")
        except:
            st.info("Agents information unavailable")
        
except requests.exceptions.RequestException as e:
    st.error(f"❌ Backend error: Cannot connect to backend at {BACKEND_URL}")
    st.error(f"Error details: {str(e)}")
    
    # Troubleshooting info
    with st.expander("Troubleshooting"):
        st.markdown("""
        ### Common Issues:
        
        1. **Backend not running**: Check if backend container is running
           ```bash
           docker ps | grep backend
           ```
        
        2. **Wrong IP/Port**: The backend might be on a different IP
           - Current backend URL: `{}`
           - Try accessing: http://172.31.77.193:8000/health
        
        3. **Network issues**: Containers might not be on same network
           ```bash
           docker network ls
           docker network inspect bridge
           ```
        
        4. **Firewall**: Check if port 8000 is open
           ```bash
           sudo iptables -L -n | grep 8000
           ```
        """.format(BACKEND_URL))

st.markdown("---")
st.caption("SutazAI v9.0 - AGI/ASI System")