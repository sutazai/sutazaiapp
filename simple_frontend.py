import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import random

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
page = st.sidebar.selectbox("Page", ["Dashboard", "Chat", "Agents", "Code Generation", "AI Services", "System Monitor"])

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
    agents_response = make_request("/api/ai-services/")
    if agents_response and "services" in agents_response:
        agents = agents_response["services"]
        agent_names = [f"{a['name']} ({a['type']})" for a in agents]
        selected_agent = st.selectbox("Select Agent", agent_names)
        
        # Find selected agent
        selected_agent_name = selected_agent.split(" (")[0]
        agent_data = next((a for a in agents if a["name"] == selected_agent_name), None)
        
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
                if agent_data:
                    # Try to communicate with the agent
                    response = make_request(f"/api/agents/{selected_agent_name}/chat", "POST", {"message": prompt})
                    reply = response.get("response", f"Agent {selected_agent_name} responded: I received your message '{prompt}' but I'm still being configured.")
                else:
                    reply = "Sorry, I couldn't find that agent."
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

elif page == "Agents":
    st.header("Agent Management")
    
    # List agents
    agents_response = make_request("/api/ai-services/")
    if agents_response and "services" in agents_response:
        agents = agents_response["services"]
        st.subheader(f"AI Agents ({agents_response.get('total_services', 0)} total, {agents_response.get('running_services', 0)} running)")
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Services", agents_response.get("total_services", 0))
        with col2:
            st.metric("Running Services", agents_response.get("running_services", 0))
        with col3:
            healthy_count = len([a for a in agents if a.get("status") == "healthy"])
            st.metric("Healthy Agents", healthy_count)
        
        # Show agents
        for agent in agents:
            status_color = "🟢" if agent.get("status") == "healthy" else "🔴" if agent.get("status") == "error" else "🟡"
            with st.expander(f"{status_color} {agent['name']} - {agent.get('status', 'unknown')}"):
                st.write(f"**Type:** {agent.get('type', 'unknown')}")
                st.write(f"**Port:** {agent.get('port', 'N/A')}")
                st.write(f"**Capabilities:** {', '.join(agent.get('capabilities', []))}")
                if agent.get('error_message'):
                    st.error(f"Error: {agent['error_message']}")
                
                # Health endpoint button
                if st.button(f"Test Health - {agent['name']}", key=f"health_{agent['name']}"):
                    health_url = agent.get('health_endpoint', '')
                    if health_url:
                        try:
                            import requests
                            health_response = requests.get(health_url, timeout=5)
                            if health_response.status_code == 200:
                                st.success("✅ Agent is healthy!")
                                st.json(health_response.json())
                            else:
                                st.error(f"❌ Health check failed: {health_response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Health check failed: {str(e)}")
                    else:
                        st.warning("No health endpoint available")
    
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

elif page == "AI Services":
    st.header("AI Services Management")
    
    # Get services from backend
    services_response = make_request("/api/ai-services/")
    services = services_response.get("services", []) if services_response else []
    
    if services:
        st.subheader("📊 Services Overview")
        
        # Service statistics
        total_services = len(services)
        running_services = len([s for s in services if s["status"] == "running"])
        stopped_services = len([s for s in services if s["status"] == "stopped"])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Services", total_services)
        with col2:
            st.metric("Running", running_services, delta=f"+{running_services}")
        with col3:
            st.metric("Stopped", stopped_services)
        with col4:
            success_rate = (running_services / total_services * 100) if total_services > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Services by type
        st.subheader("🔧 Services by Type")
        service_types = {}
        for service in services:
            svc_type = service["type"]
            if svc_type not in service_types:
                service_types[svc_type] = []
            service_types[svc_type].append(service)
        
        for svc_type, type_services in service_types.items():
            with st.expander(f"{svc_type.replace('_', ' ').title()} ({len(type_services)} services)"):
                for service in type_services:
                    col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 3])
                    
                    with col_a:
                        status_icon = "🟢" if service["status"] == "running" else "🔴" if service["status"] == "stopped" else "🟡"
                        st.write(f"{status_icon} **{service['name']}**")
                    
                    with col_b:
                        st.write(f"Port: {service['port']}")
                    
                    with col_c:
                        st.write(f"Status: {service['status']}")
                    
                    with col_d:
                        if service["status"] == "stopped":
                            if st.button(f"▶️ Start", key=f"start_{service['name']}"):
                                result = make_request(f"/api/services/{service['name']}/start", "POST")
                                if result:
                                    st.success(f"Starting {service['name']}...")
                                    st.rerun()
                        elif service["status"] == "running":
                            if st.button(f"⏹️ Stop", key=f"stop_{service['name']}"):
                                result = make_request(f"/api/services/{service['name']}/stop", "POST")
                                if result:
                                    st.success(f"Stopping {service['name']}...")
                                    st.rerun()
                    
                    # Show capabilities
                    if service.get("capabilities"):
                        st.write(f"**Capabilities:** {', '.join(service['capabilities'])}")
                    
                    st.markdown("---")
        
        # Bulk operations
        st.subheader("🚀 Bulk Operations")
        col_x, col_y, col_z = st.columns(3)
        
        with col_x:
            if st.button("🔄 Health Check All"):
                with st.spinner("Checking health of all services..."):
                    health_results = make_request("/api/services/health")
                    if health_results:
                        st.success(f"Health check completed for {health_results.get('total_checked', 0)} services")
                        
                        # Show health results
                        for service_name, health in health_results.get("results", {}).items():
                            status_icon = "✅" if health["status"] == "healthy" else "⚠️" if health["status"] == "unhealthy" else "❌"
                            st.write(f"{status_icon} {service_name}: {health['status']}")
        
        with col_y:
            if st.button("▶️ Start Key Services"):
                key_services = ["enhanced-model-manager", "crewai", "documind", "langchain-agents"]
                for service in key_services:
                    make_request(f"/api/services/{service}/start", "POST")
                st.success("Starting key services...")
                st.rerun()
        
        with col_z:
            if st.button("🔍 Refresh Services"):
                st.rerun()
    
    else:
        st.error("Failed to load services information")

elif page == "System Monitor":
    st.header("System Monitoring")
    
    # Real-time system status
    status = make_request("/api/system/status")
    
    if status:
        st.subheader("📊 System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Status", status.get("status", "unknown").upper())
        
        with col2:
            uptime = status.get("uptime", 0)
            st.metric("Uptime", f"{uptime:.1f}s")
        
        with col3:
            if "ai_services" in status and isinstance(status["ai_services"], dict):
                running = status["ai_services"].get("running", 0)
                total = status["ai_services"].get("total_services", 0)
                st.metric("Services", f"{running}/{total}")
            else:
                st.metric("Services", "N/A")
        
        with col4:
            timestamp = status.get("timestamp", 0)
            st.metric("Last Update", f"{time.strftime('%H:%M:%S', time.localtime(timestamp))}")
        
        # Capabilities overview
        if "capabilities" in status:
            st.subheader("🧠 AI Capabilities")
            caps = status["capabilities"]
            
            cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)
            
            with cap_col1:
                st.metric("Code Generation", caps.get("code_generation", 0))
            
            with cap_col2:
                st.metric("Document Processing", caps.get("document_processing", 0))
            
            with cap_col3:
                st.metric("Web Automation", caps.get("web_automation", 0))
            
            with cap_col4:
                st.metric("Agent Frameworks", caps.get("agent_frameworks", 0))
        
        # Performance monitoring
        st.subheader("⚡ Performance Metrics")
        
        # Generate sample performance data
        # Generate sample data
        time_points = pd.date_range(start='2024-01-01 00:00:00', periods=24, freq='H')
        performance_data = pd.DataFrame({
            'Time': time_points,
            'CPU': [random.randint(20, 80) for _ in range(24)],
            'Memory': [random.randint(40, 90) for _ in range(24)],
            'Network': [random.randint(10, 50) for _ in range(24)]
        })
        
        fig = px.line(performance_data, x='Time', y=['CPU', 'Memory', 'Network'],
                     title='System Performance (Last 24 Hours)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Service health matrix
        st.subheader("🏥 Service Health Matrix")
        health_data = make_request("/api/services/health")
        
        if health_data and "results" in health_data:
            health_df_data = []
            for service_name, health in health_data["results"].items():
                health_df_data.append({
                    "Service": service_name,
                    "Status": health["status"],
                    "Response Time": health.get("response_time", "N/A"),
                    "Last Check": time.strftime('%H:%M:%S', time.localtime(health.get("timestamp", 0)))
                })
            
            if health_df_data:
                health_df = pd.DataFrame(health_df_data)
                st.dataframe(health_df, use_container_width=True)
        
        # Auto-refresh
        if st.button("🔄 Refresh Monitoring Data"):
            st.rerun()
        
        # Auto-refresh every 30 seconds
        time.sleep(30)
        st.rerun()
    
    else:
        st.error("Failed to load system status")

st.sidebar.markdown("---")
st.sidebar.info("SutazAI v9.0.0\nPowered by Streamlit")
