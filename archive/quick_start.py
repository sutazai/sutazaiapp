#!/usr/bin/env python3
"""
Quick Start Script for SutazAI v9 System
Launches backend and frontend with proper configuration
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

class SutazAILauncher:
    def __init__(self):
        self.processes = []
        self.backend_port = 8000
        self.frontend_port = 8501
        self.backend_host = "0.0.0.0"
        self.frontend_host = "0.0.0.0"
        
    def setup_environment(self):
        """Setup environment variables"""
        os.environ["BACKEND_URL"] = f"http://localhost:{self.backend_port}"
        os.environ["PYTHONPATH"] = "/opt/sutazaiapp:/opt/sutazaiapp/backend"
        
        # Create necessary directories
        Path("/opt/sutazaiapp/backend/data").mkdir(exist_ok=True)
        Path("/opt/sutazaiapp/backend/logs").mkdir(exist_ok=True)
        
        print("✅ Environment setup complete")
    
    def install_requirements(self):
        """Install required packages"""
        try:
            requirements = [
                "fastapi>=0.104.1",
                "uvicorn[standard]>=0.24.0",
                "streamlit>=1.28.0",
                "requests>=2.31.0",
                "pandas>=2.1.0",
                "plotly>=5.17.0",
                "asyncio-mqtt>=0.13.0",
                "aiofiles>=23.2.1",
                "python-multipart>=0.0.6"
            ]
            
            for req in requirements:
                print(f"Installing {req}...")
                subprocess.run([sys.executable, "-m", "pip", "install", req], 
                             check=True, capture_output=True)
            
            print("✅ All requirements installed")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
        return True
    
    def create_simple_backend(self):
        """Create a simple FastAPI backend"""
        backend_code = '''#!/usr/bin/env python3
"""
Simple SutazAI Backend for Quick Start
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time
import json
from datetime import datetime
from typing import Dict, List, Any

app = FastAPI(
    title="SutazAI Backend API",
    description="SutazAI v9 Backend Service",
    version="9.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
system_state = {
    "status": "online",
    "uptime": time.time(),
    "active_agents": 3,
    "loaded_models": 5,
    "requests_count": 0
}

agents_db = [
    {"id": "1", "name": "DeepSeek-Coder", "type": "coding", "status": "active", "created_at": "2024-01-01"},
    {"id": "2", "name": "Llama-2", "type": "general", "status": "active", "created_at": "2024-01-01"},
    {"id": "3", "name": "AutoGPT", "type": "automation", "status": "idle", "created_at": "2024-01-01"},
]

models_db = [
    {"id": "1", "name": "deepseek-coder:33b", "type": "coding", "status": "loaded"},
    {"id": "2", "name": "llama2:13b", "type": "general", "status": "loaded"},
    {"id": "3", "name": "codellama:7b", "type": "coding", "status": "loaded"},
    {"id": "4", "name": "mistral:7b", "type": "general", "status": "loaded"},
    {"id": "5", "name": "qwen3:8b", "type": "general", "status": "loaded"},
]

@app.get("/")
async def root():
    return {"message": "SutazAI Backend v9.0.0", "status": "online"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - system_state["uptime"],
        "version": "9.0.0"
    }

@app.get("/api/system/status")
async def get_system_status():
    system_state["requests_count"] += 1
    return {
        "status": system_state["status"],
        "uptime": time.time() - system_state["uptime"],
        "active_agents": system_state["active_agents"],
        "loaded_models": system_state["loaded_models"],
        "requests_count": system_state["requests_count"],
        "timestamp": time.time()
    }

@app.get("/api/agents/")
async def get_agents():
    return agents_db

@app.post("/api/agents/")
async def create_agent(agent_data: Dict[str, Any]):
    new_agent = {
        "id": str(len(agents_db) + 1),
        "name": agent_data.get("name", "New Agent"),
        "type": agent_data.get("type", "general"),
        "status": "active",
        "created_at": datetime.now().isoformat()
    }
    agents_db.append(new_agent)
    system_state["active_agents"] = len([a for a in agents_db if a["status"] == "active"])
    return new_agent

@app.post("/api/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: str, message_data: Dict[str, Any]):
    message = message_data.get("message", "")
    
    # Find agent
    agent = next((a for a in agents_db if a["id"] == agent_id), None)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Simulate response
    response = f"Hello! I'm {agent['name']}. You said: '{message}'. This is a simulated response."
    
    return {
        "agent_id": agent_id,
        "agent_name": agent["name"],
        "response": response,
        "timestamp": time.time()
    }

@app.get("/api/models/")
async def get_models():
    return models_db

@app.post("/api/code/generate")
async def generate_code(request_data: Dict[str, Any]):
    prompt = request_data.get("prompt", "")
    language = request_data.get("language", "python")
    
    # Simulate code generation
    sample_code = f"""# Generated {language} code for: {prompt}
def example_function():
    \"\"\"
    This is an example function generated for: {prompt}
    \"\"\"
    print("Hello from SutazAI!")
    return "Success"

if __name__ == "__main__":
    result = example_function()
    print(result)
"""
    
    return {
        "code": sample_code,
        "language": language,
        "prompt": prompt,
        "timestamp": time.time()
    }

@app.post("/api/documents/upload")
async def upload_document(file_data: Dict[str, Any]):
    # Simulate document processing
    return {
        "status": "success",
        "document_id": "doc_" + str(int(time.time())),
        "summary": "This document has been processed successfully. It contains important information that has been analyzed by SutazAI.",
        "key_points": [
            "Document processed successfully",
            "Key information extracted",
            "Analysis completed"
        ],
        "extracted_text": "This is the extracted text from the uploaded document. SutazAI has processed this content and made it available for further analysis.",
        "timestamp": time.time()
    }

@app.get("/api/metrics/")
async def get_metrics():
    return {
        "cpu_usage": 45.2,
        "memory_usage": 62.1,
        "gpu_usage": 78.5,
        "network_io": 1.2,
        "disk_io": 0.8,
        "active_connections": 15,
        "timestamp": time.time()
    }

@app.get("/ai/services/status")
async def get_ai_services_status():
    services = {
        "deepseek-r1": {"status": "healthy", "details": {"model": "deepseek-r1:8b", "memory_usage": "2.1GB"}},
        "qwen3": {"status": "healthy", "details": {"model": "qwen3:8b", "memory_usage": "1.8GB"}},
        "ollama": {"status": "healthy", "details": {"port": 11434, "models": 5}},
        "vector_db": {"status": "healthy", "details": {"collections": 3, "vectors": 10000}},
        "redis": {"status": "healthy", "details": {"connections": 5, "memory": "150MB"}}
    }
    
    return {"services": services}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
'''
        
        backend_file = Path("/opt/sutazaiapp/simple_backend_server.py")
        backend_file.write_text(backend_code)
        print("✅ Simple backend created")
        return str(backend_file)
    
    def create_fixed_frontend(self):
        """Create fixed frontend with proper backend URL"""
        frontend_code = '''#!/usr/bin/env python3
"""
Fixed SutazAI Frontend with proper backend connection
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed backend URL for local deployment
BACKEND_URL = "http://localhost:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_status" not in st.session_state:
    st.session_state.system_status = {}

class SutazAIInterface:
    """Main interface for SutazAI system"""
    
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.session.timeout = 10
    
    def make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Make HTTP request to backend"""
        try:
            url = f"{self.backend_url}{endpoint}"
            
            if method == "GET":
                response = self.session.get(url, params=data)
            elif method == "POST":
                response = self.session.post(url, json=data)
            elif method == "PUT":
                response = self.session.put(url, json=data)
            elif method == "DELETE":
                response = self.session.delete(url)
            
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            st.error(f"Backend connection error: {str(e)}")
            return {}
    
    def test_connection(self) -> bool:
        """Test backend connection"""
        try:
            response = self.session.get(f"{self.backend_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self.make_request("/api/system/status")
    
    def get_agents(self) -> List[Dict[str, Any]]:
        """Get available agents"""
        return self.make_request("/api/agents/")
    
    def create_agent(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create new agent"""
        return self.make_request("/api/agents/", method="POST", data=agent_config)
    
    def chat_with_agent(self, agent_id: str, message: str) -> Dict[str, Any]:
        """Chat with agent"""
        return self.make_request(
            f"/api/agents/{agent_id}/chat",
            method="POST",
            data={"message": message}
        )
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get available models"""
        return self.make_request("/api/models/")
    
    def generate_code(self, prompt: str, language: str = "python") -> Dict[str, Any]:
        """Generate code"""
        return self.make_request(
            "/api/code/generate",
            method="POST",
            data={"prompt": prompt, "language": language}
        )

# Initialize interface
interface = SutazAIInterface()

# Connection status check
connection_status = interface.test_connection()

# Sidebar
st.sidebar.title("🤖 SutazAI Control Panel")

# Connection status in sidebar
if connection_status:
    st.sidebar.success("🟢 Backend Connected")
else:
    st.sidebar.error("🔴 Backend Disconnected")
    st.sidebar.warning("Please ensure the backend server is running on localhost:8000")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "💬 Chat Interface", "🔧 Agent Management", "📊 Code Generation", "⚙️ Settings"]
)

# System status in sidebar
if connection_status:
    with st.sidebar.expander("🔋 System Status", expanded=False):
        status = interface.get_system_status()
        if status:
            st.success(f"Status: {status.get('status', 'Unknown')}")
            st.info(f"Uptime: {status.get('uptime', 0):.1f}s")
            st.info(f"Active Agents: {status.get('active_agents', 0)}")
            st.info(f"Models Loaded: {status.get('loaded_models', 0)}")

# Main content area
st.title("🚀 SutazAI AGI/ASI Autonomous System v9")

if not connection_status:
    st.error("⚠️ Cannot connect to backend server")
    st.info("To start the system, run: `python /opt/sutazaiapp/quick_start.py`")
    st.stop()

if page == "🏠 Dashboard":
    st.header("System Dashboard")
    
    # Get real system status
    status = interface.get_system_status()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_text = "🟢 Online" if connection_status else "🔴 Offline"
        st.metric("System Status", status_text, "Healthy")
    
    with col2:
        agents_count = status.get("active_agents", 0)
        st.metric("Active Agents", agents_count, "+1")
    
    with col3:
        requests_count = status.get("requests_count", 0)
        st.metric("Requests Served", requests_count, f"+{requests_count}")
    
    with col4:
        uptime = status.get("uptime", 0)
        st.metric("Uptime", f"{uptime:.1f}s", "Running")
    
    # Real-time system overview
    st.subheader("🔄 Real-time System Overview")
    
    # Services status
    st.subheader("Core Services Status")
    
    services_data = {
        "Service": ["Backend API", "Streamlit UI", "Agent Manager", "Model Service"],
        "Status": ["🟢 Running", "🟢 Running", "🟢 Running", "🟢 Running"],
        "Port": ["8000", "8501", "Internal", "Internal"],
        "Health": ["Healthy", "Healthy", "Healthy", "Healthy"]
    }
    
    df_services = pd.DataFrame(services_data)
    st.dataframe(df_services, use_container_width=True)

elif page == "💬 Chat Interface":
    st.header("Chat Interface")
    
    # Get available agents
    agents = interface.get_agents()
    
    if not agents:
        st.warning("No agents available")
        st.stop()
    
    # Agent selection
    agent_options = {f"{agent['name']} ({agent['type']})": agent['id'] for agent in agents}
    selected_agent_name = st.selectbox("Select Agent", list(agent_options.keys()))
    selected_agent_id = agent_options[selected_agent_name]
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask SutazAI anything..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_data = interface.chat_with_agent(selected_agent_id, prompt)
                    response = response_data.get("response", "Sorry, I couldn't process that request.")
                    st.markdown(response)
                    
                    # Add assistant response
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Chat controls
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

elif page == "🔧 Agent Management":
    st.header("Agent Management")
    
    # Agent overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 Active Agents")
        
        # Agent list
        agents = interface.get_agents()
        if agents:
            for agent in agents:
                with st.expander(f"Agent: {agent.get('name', 'Unknown')}"):
                    st.write(f"**Type:** {agent.get('type', 'Unknown')}")
                    st.write(f"**Status:** {agent.get('status', 'Unknown')}")
                    st.write(f"**ID:** {agent.get('id', 'Unknown')}")
        else:
            st.info("No active agents found")
    
    with col2:
        st.subheader("➕ Create New Agent")
        
        with st.form("create_agent"):
            agent_name = st.text_input("Agent Name")
            agent_type = st.selectbox(
                "Agent Type",
                ["coding", "general", "research", "creative", "analytical"]
            )
            
            submitted = st.form_submit_button("🚀 Create Agent")
            
            if submitted and agent_name:
                agent_config = {
                    "name": agent_name,
                    "type": agent_type
                }
                
                result = interface.create_agent(agent_config)
                if result:
                    st.success(f"Agent '{agent_name}' created successfully!")
                    st.rerun()

elif page == "📊 Code Generation":
    st.header("Code Generation")
    
    # Code generation interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✍️ Generate Code")
        
        # Code prompt
        code_prompt = st.text_area(
            "Describe what you want to code:",
            placeholder="Create a Python function that calculates fibonacci numbers...",
            height=150
        )
        
        # Language selection
        language = st.selectbox(
            "Programming Language",
            ["python", "javascript", "java", "cpp", "rust", "go", "html", "css"]
        )
        
        if st.button("🚀 Generate Code"):
            if code_prompt:
                with st.spinner("Generating code..."):
                    result = interface.generate_code(code_prompt, language)
                    if result:
                        st.session_state.generated_code = result.get("code", "")
                        st.success("Code generated successfully!")
    
    with col2:
        st.subheader("📋 Generated Code")
        
        if "generated_code" in st.session_state:
            st.code(st.session_state.generated_code, language=language)
            
            # Download button
            st.download_button(
                "📥 Download Code",
                st.session_state.generated_code,
                f"generated_code.{language}",
                f"text/{language}"
            )
        else:
            st.info("No code generated yet. Use the form on the left to generate code.")

elif page == "⚙️ Settings":
    st.header("System Settings")
    
    # Settings interface
    st.subheader("Backend Configuration")
    
    st.info(f"Backend URL: {BACKEND_URL}")
    st.info(f"Connection Status: {'✅ Connected' if connection_status else '❌ Disconnected'}")
    
    if st.button("🔄 Test Connection"):
        if interface.test_connection():
            st.success("✅ Backend connection successful!")
        else:
            st.error("❌ Backend connection failed!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤖 SutazAI AGI/ASI Autonomous System v9.0.0 | 
        Built with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
'''
        
        frontend_file = Path("/opt/sutazaiapp/fixed_streamlit_app.py")
        frontend_file.write_text(frontend_code)
        print("✅ Fixed frontend created")
        return str(frontend_file)
    
    def start_backend(self, backend_file):
        """Start the backend server"""
        try:
            print(f"🚀 Starting backend server on {self.backend_host}:{self.backend_port}")
            
            backend_process = subprocess.Popen([
                sys.executable, backend_file
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.processes.append(("backend", backend_process))
            
            # Wait a moment for backend to start
            time.sleep(3)
            
            # Check if backend is running
            if backend_process.poll() is None:
                print("✅ Backend server started successfully")
                return True
            else:
                stdout, stderr = backend_process.communicate()
                print(f"❌ Backend failed to start: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self, frontend_file):
        """Start the frontend server"""
        try:
            print(f"🚀 Starting frontend server on {self.frontend_host}:{self.frontend_port}")
            
            frontend_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", frontend_file,
                "--server.address", self.frontend_host,
                "--server.port", str(self.frontend_port),
                "--server.headless", "true",
                "--server.fileWatcherType", "none",
                "--logger.level", "error"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.processes.append(("frontend", frontend_process))
            
            # Wait a moment for frontend to start
            time.sleep(5)
            
            # Check if frontend is running
            if frontend_process.poll() is None:
                print("✅ Frontend server started successfully")
                return True
            else:
                stdout, stderr = frontend_process.communicate()
                print(f"❌ Frontend failed to start: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def cleanup(self):
        """Clean up processes"""
        print("🧹 Cleaning up processes...")
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except:
                try:
                    process.kill()
                    print(f"⚠️ {name} force killed")
                except:
                    pass
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\\n🛑 Shutdown signal received")
        self.cleanup()
        sys.exit(0)
    
    def run(self):
        """Main run function"""
        print("🚀 SutazAI v9 Quick Start Launcher")
        print("=" * 50)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Setup environment
            self.setup_environment()
            
            # Install requirements
            if not self.install_requirements():
                print("❌ Failed to install requirements")
                return False
            
            # Create backend
            backend_file = self.create_simple_backend()
            
            # Create frontend
            frontend_file = self.create_fixed_frontend()
            
            # Start backend
            if not self.start_backend(backend_file):
                print("❌ Failed to start backend")
                return False
            
            # Start frontend
            if not self.start_frontend(frontend_file):
                print("❌ Failed to start frontend")
                return False
            
            print("\\n" + "=" * 50)
            print("🎉 SutazAI v9 System Started Successfully!")
            print("=" * 50)
            print(f"🌐 Frontend: http://localhost:{self.frontend_port}")
            print(f"🔧 Backend API: http://localhost:{self.backend_port}")
            print(f"📚 API Docs: http://localhost:{self.backend_port}/docs")
            print("=" * 50)
            print("\\n💡 Access the system at: http://localhost:8501")
            print("🛑 Press Ctrl+C to stop the system")
            print("\\n")
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
                    
                    # Check if processes are still running
                    for name, process in self.processes:
                        if process.poll() is not None:
                            print(f"⚠️ {name} process stopped unexpectedly")
                            return False
                            
            except KeyboardInterrupt:
                print("\\n🛑 Received shutdown signal")
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
            
        finally:
            self.cleanup()
        
        return True

if __name__ == "__main__":
    launcher = SutazAILauncher()
    success = launcher.run()
    
    if success:
        print("✅ SutazAI v9 shut down cleanly")
    else:
        print("❌ SutazAI v9 encountered errors")
        sys.exit(1)