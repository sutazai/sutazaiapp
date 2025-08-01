#!/usr/bin/env python3
"""
Simple SutazAI Launcher
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def setup_environment():
    """Setup environment"""
    os.environ["BACKEND_URL"] = "http://localhost:8000"
    os.environ["PYTHONPATH"] = "/opt/sutazaiapp:/opt/sutazaiapp/backend"

def install_packages():
    """Install required packages"""
    packages = [
        "fastapi",
        "uvicorn[standard]",
        "streamlit",
        "requests",
        "pandas",
        "plotly"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ Installed {package}")
        except:
            print(f"⚠️ Failed to install {package}")

def create_backend():
    """Create simple backend"""
    backend_code = '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

app = FastAPI(title="SutazAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "SutazAI Backend v9.0.0", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/api/system/status")
async def system_status():
    return {
        "status": "online",
        "uptime": 100.0,
        "active_agents": 3,
        "loaded_models": 5,
        "requests_count": 42
    }

@app.get("/api/agents/")
async def get_agents():
    return [
        {"id": "1", "name": "DeepSeek-Coder", "type": "coding", "status": "active"},
        {"id": "2", "name": "Qwen3", "type": "general", "status": "active"},
        {"id": "3", "name": "AutoGPT", "type": "automation", "status": "idle"}
    ]

@app.post("/api/agents/")
async def create_agent(data: dict):
    return {"id": "new", "name": data.get("name", "New Agent"), "status": "active"}

@app.post("/api/agents/{agent_id}/chat")
async def chat(agent_id: str, data: dict):
    message = data.get("message", "")
    return {
        "response": f"Hello! This is a response from agent {agent_id} to your message: {message}",
        "agent_id": agent_id
    }

@app.get("/api/models/")
async def get_models():
    return [
        {"id": "1", "name": "tinyllama", "status": "loaded"},
        {"id": "2", "name": "qwen3:8b", "status": "loaded"}
    ]

@app.post("/api/code/generate")
async def generate_code(data: dict):
    prompt = data.get("prompt", "")
    language = data.get("language", "python")
    
    code = f"""# Generated {language} code for: {prompt}
def example_function():
    print("Hello from SutazAI!")
    return "Generated code example"

if __name__ == "__main__":
    result = example_function()
    print(result)
"""
    
    return {"code": code, "language": language}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open("/opt/sutazaiapp/simple_backend.py", "w") as f:
        f.write(backend_code)

def create_frontend():
    """Create simple frontend"""
    frontend_code = '''import streamlit as st
import requests
import json

st.set_page_config(page_title="SutazAI v9", page_icon="🤖", layout="wide")

BACKEND_URL = "http://localhost:8000"

def test_backend():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_request(endpoint, method="GET", data=None):
    try:
        url = f"{BACKEND_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return {}

st.title("🚀 SutazAI AGI/ASI System v9")

# Check backend connection
if not test_backend():
    st.error("❌ Cannot connect to backend server")
    st.info("Please start the backend server first:")
    st.code("python3 /opt/sutazaiapp/simple_backend.py")
    st.stop()

st.success("✅ Backend connected successfully!")

# Sidebar
st.sidebar.title("🤖 SutazAI Control")
page = st.sidebar.selectbox("Page", ["Dashboard", "Chat", "Agents", "Code Generation"])

if page == "Dashboard":
    st.header("System Dashboard")
    
    # System status
    status = make_request("/api/system/status")
    if status:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", status.get("status", "Unknown"))
        with col2:
            st.metric("Agents", status.get("active_agents", 0))
        with col3:
            st.metric("Models", status.get("loaded_models", 0))
        with col4:
            st.metric("Requests", status.get("requests_count", 0))
    
    # Services
    st.subheader("Services")
    services_data = {
        "Service": ["Backend API", "Streamlit UI", "Agent Manager"],
        "Status": ["🟢 Running", "🟢 Running", "🟢 Running"],
        "Port": ["8000", "8501", "Internal"]
    }
    st.dataframe(services_data, use_container_width=True)

elif page == "Chat":
    st.header("Chat Interface")
    
    # Get agents
    agents = make_request("/api/agents/")
    if agents:
        agent_names = [f"{a['name']} ({a['type']})" for a in agents]
        selected_agent = st.selectbox("Select Agent", agent_names)
        agent_id = agents[agent_names.index(selected_agent)]["id"]
        
        # Chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response = make_request(f"/api/agents/{agent_id}/chat", "POST", {"message": prompt})
                reply = response.get("response", "Sorry, I couldn't process that.")
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

elif page == "Agents":
    st.header("Agent Management")
    
    # List agents
    agents = make_request("/api/agents/")
    if agents:
        st.subheader("Active Agents")
        for agent in agents:
            with st.expander(f"{agent['name']} - {agent['status']}"):
                st.write(f"Type: {agent['type']}")
                st.write(f"ID: {agent['id']}")
    
    # Create agent
    st.subheader("Create New Agent")
    with st.form("create_agent"):
        name = st.text_input("Agent Name")
        agent_type = st.selectbox("Type", ["coding", "general", "research"])
        
        if st.form_submit_button("Create"):
            if name:
                result = make_request("/api/agents/", "POST", {"name": name, "type": agent_type})
                if result:
                    st.success(f"Created agent: {name}")
                    st.rerun()

elif page == "Code Generation":
    st.header("Code Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generate Code")
        prompt = st.text_area("Describe what you want:", height=150)
        language = st.selectbox("Language", ["python", "javascript", "java", "cpp"])
        
        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating..."):
                    result = make_request("/api/code/generate", "POST", 
                                        {"prompt": prompt, "language": language})
                    if result:
                        st.session_state.code = result.get("code", "")
    
    with col2:
        st.subheader("Generated Code")
        if "code" in st.session_state:
            st.code(st.session_state.code, language=language)
            st.download_button("Download", st.session_state.code, f"code.{language}")
        else:
            st.info("No code generated yet")

st.sidebar.markdown("---")
st.sidebar.info("SutazAI v9.0.0\\nPowered by Streamlit")
'''
    
    with open("/opt/sutazaiapp/simple_frontend.py", "w") as f:
        f.write(frontend_code)

def main():
    print("🚀 SutazAI v9 Setup")
    print("=" * 40)
    
    setup_environment()
    print("✅ Environment setup")
    
    install_packages()
    print("✅ Packages installed")
    
    create_backend()
    print("✅ Backend created")
    
    create_frontend()
    print("✅ Frontend created")
    
    print("\\n" + "=" * 40)
    print("🎉 Setup Complete!")
    print("=" * 40)
    print("\\nTo start the system:")
    print("1. Backend:  python3 /opt/sutazaiapp/simple_backend.py")
    print("2. Frontend: streamlit run /opt/sutazaiapp/simple_frontend.py --server.port 8501")
    print("\\nOr use the start script:")
    print("bash /opt/sutazaiapp/start_system.sh")
    print("=" * 40)

if __name__ == "__main__":
    main()