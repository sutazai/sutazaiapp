#!/bin/bash
# SutazAI AGI/ASI Integration Script
# Integrates new AGI/ASI components with existing application

set -euo pipefail

# ===============================================
# CONFIGURATION
# ===============================================

export WORKSPACE_ROOT="/workspace"
export PROJECT_ROOT="/opt/sutazaiapp"
export EXISTING_APP_URL="http://172.31.77.193:8501"
export LOCAL_IP=$(hostname -I | awk '{print $1}')

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# ===============================================
# LOGGING FUNCTIONS
# ===============================================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  WARNING: $1${NC}"
}

log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️  INFO: $1${NC}"
}

log_phase() {
    echo -e "${PURPLE}${BOLD}[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 PHASE: $1${NC}"
}

# ===============================================
# SYSTEM ANALYSIS
# ===============================================

analyze_existing_system() {
    log_phase "Analyzing Existing System"
    
    # Check existing containers
    log_info "Checking existing Docker containers..."
    existing_containers=$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "Docker not available")
    echo "$existing_containers"
    
    # Check Streamlit app
    if curl -s -o /dev/null -w "%{http_code}" ${EXISTING_APP_URL} | grep -q "200"; then
        log "✅ Streamlit app is running at ${EXISTING_APP_URL}"
    else
        log_warn "⚠️  Streamlit app not accessible at ${EXISTING_APP_URL}"
    fi
    
    # Check existing models
    if command -v ollama >/dev/null 2>&1; then
        log_info "Ollama is installed, checking models..."
        ollama list 2>/dev/null || log_warn "Cannot list Ollama models"
    fi
}

# ===============================================
# INTEGRATE MODELS
# ===============================================

integrate_model_management() {
    log_phase "Integrating Model Management"
    
    # Create model integration configuration
    cat > ${WORKSPACE_ROOT}/config/agi_models.yaml << EOF
# AGI/ASI Model Configuration
models:
  ollama:
    endpoint: "http://localhost:11434"
    models:
      - deepseek-r1:8b
      - qwen3:8b
      - codellama:7b
      - llama2:7b
  
  litellm:
    endpoint: "http://localhost:4000"
    api_key: "sk-sutazai-local-key"
    
  external_models:
    - name: "gpt-4"
      provider: "openai"
      fallback: "deepseek-r1:8b"
    - name: "claude-3"
      provider: "anthropic" 
      fallback: "qwen3:8b"

routing_strategy:
  default: "deepseek-r1:8b"
  code_tasks: "codellama:7b"
  general_tasks: "llama2:7b"
  complex_tasks: "qwen3:8b"
EOF

    # Pull essential models
    log_info "Pulling essential Ollama models..."
    if command -v ollama >/dev/null 2>&1; then
        for model in "deepseek-r1:8b" "qwen3:8b" "codellama:7b"; do
            log_info "Pulling $model..."
            ollama pull $model || log_warn "Failed to pull $model"
        done
    fi
}

# ===============================================
# INTEGRATE VECTOR DATABASES
# ===============================================

integrate_vector_databases() {
    log_phase "Integrating Vector Databases"
    
    # Create vector DB integration service
    mkdir -p ${WORKSPACE_ROOT}/services/vector_integration
    
    cat > ${WORKSPACE_ROOT}/services/vector_integration/vector_router.py << 'EOF'
"""
Vector Database Router for AGI/ASI System
Routes requests to appropriate vector database based on use case
"""

from typing import List, Dict, Any, Optional
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings

app = FastAPI(title="Vector Database Router")

# Initialize clients
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
faiss_url = "http://localhost:8100"

class VectorRequest(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = {}
    collection: str = "default"

class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    collection: str = "default"
    use_faiss: bool = False

@app.post("/index")
async def index_vector(request: VectorRequest):
    """Index a vector in the appropriate database"""
    try:
        if len(request.vector) > 1024 or request.use_faiss:
            # Use FAISS for high-dimensional vectors
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{faiss_url}/index",
                    json={"id": request.id, "vector": request.vector}
                )
                return {"status": "indexed", "database": "faiss", "id": request.id}
        else:
            # Use ChromaDB for standard vectors
            collection = chroma_client.get_or_create_collection(request.collection)
            collection.add(
                ids=[request.id],
                embeddings=[request.vector],
                metadatas=[request.metadata]
            )
            return {"status": "indexed", "database": "chromadb", "id": request.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_vectors(request: SearchRequest):
    """Search for similar vectors"""
    try:
        if request.use_faiss:
            # Search in FAISS
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{faiss_url}/search",
                    json={"vector": request.vector, "k": request.k}
                )
                return response.json()
        else:
            # Search in ChromaDB
            collection = chroma_client.get_collection(request.collection)
            results = collection.query(
                query_embeddings=[request.vector],
                n_results=request.k
            )
            return {
                "results": [
                    {
                        "id": results["ids"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                    }
                    for i in range(len(results["ids"][0]))
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check health of vector databases"""
    status = {"chromadb": False, "faiss": False}
    
    try:
        # Check ChromaDB
        chroma_client.heartbeat()
        status["chromadb"] = True
    except:
        pass
    
    try:
        # Check FAISS
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{faiss_url}/health")
            status["faiss"] = response.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy" if all(status.values()) else "degraded",
        "databases": status
    }
EOF
}

# ===============================================
# INTEGRATE AI AGENTS
# ===============================================

integrate_ai_agents() {
    log_phase "Integrating AI Agents"
    
    # Create agent integration configuration
    cat > ${WORKSPACE_ROOT}/config/agent_integration.json << EOF
{
  "agents": {
    "letta": {
      "endpoint": "http://localhost:8283",
      "capabilities": ["memory", "conversation", "task_planning"],
      "memory_limit_mb": 512
    },
    "autogpt": {
      "endpoint": "http://localhost:8080",
      "capabilities": ["task_automation", "code_generation", "research"],
      "max_iterations": 10
    },
    "localagi": {
      "endpoint": "http://localhost:8090",
      "capabilities": ["orchestration", "agent_coordination"],
      "timeout_seconds": 300
    },
    "tabbyml": {
      "endpoint": "http://localhost:8085",
      "capabilities": ["code_completion", "code_suggestion"],
      "model": "StarCoder-1B"
    },
    "semgrep": {
      "endpoint": "http://localhost:8087",
      "capabilities": ["code_analysis", "security_scanning"],
      "rules": ["python", "javascript", "security"]
    },
    "langchain": {
      "endpoint": "http://localhost:8095",
      "capabilities": ["chain_orchestration", "tool_usage", "reasoning"],
      "tools": ["web_search", "calculator", "code_interpreter"]
    }
  },
  "routing_rules": {
    "code_generation": ["tabbyml", "autogpt", "langchain"],
    "code_analysis": ["semgrep", "langchain"],
    "task_automation": ["autogpt", "letta", "localagi"],
    "memory_tasks": ["letta"],
    "orchestration": ["localagi", "langchain"]
  }
}
EOF

    log "Agent integration configuration created"
}

# ===============================================
# UPDATE STREAMLIT APP
# ===============================================

update_streamlit_app() {
    log_phase "Updating Streamlit Application"
    
    # Create AGI integration page for Streamlit
    cat > ${WORKSPACE_ROOT}/frontend/pages/8_🧠_AGI_System.py << 'EOF'
"""
AGI/ASI System Integration Page
"""

import streamlit as st
import httpx
import asyncio
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AGI/ASI System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AGI/ASI System Control Center")
st.markdown("---")

# Initialize session state
if 'agi_orchestrator_url' not in st.session_state:
    st.session_state.agi_orchestrator_url = "http://localhost:8200"

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    orchestrator_url = st.text_input(
        "Orchestrator URL",
        value=st.session_state.agi_orchestrator_url
    )
    st.session_state.agi_orchestrator_url = orchestrator_url
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    # Get system stats
    try:
        response = httpx.get(f"{orchestrator_url}/health", timeout=5.0)
        if response.status_code == 200:
            health_data = response.json()
            
            # Show service status
            st.metric("System Status", health_data.get("status", "Unknown").title())
            
            services = health_data.get("services", {})
            online_count = sum(1 for status in services.values() if status)
            st.metric("Services Online", f"{online_count}/{len(services)}")
    except:
        st.error("Cannot connect to orchestrator")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Task Execution",
    "📊 System Status",
    "🤖 Agent Management",
    "💾 Vector Databases",
    "📈 Performance Metrics"
])

# Tab 1: Task Execution
with tab1:
    st.header("Execute AGI Task")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        task_type = st.selectbox(
            "Task Type",
            ["general", "code_generation", "code_analysis", "task_automation", "memory_task"],
            help="Select the type of task to execute"
        )
        
        prompt = st.text_area(
            "Task Description",
            height=150,
            placeholder="Describe the task you want the AGI system to perform..."
        )
        
        with st.expander("🔧 Advanced Options"):
            # Get available agents
            try:
                response = httpx.get(f"{orchestrator_url}/services", timeout=5.0)
                available_agents = response.json().get("services", [])
            except:
                available_agents = []
            
            selected_agents = st.multiselect(
                "Select Specific Agents",
                available_agents,
                help="Leave empty for automatic agent selection"
            )
            
            context = st.text_area(
                "Additional Context (JSON)",
                value="{}",
                height=100
            )
            
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            with col_adv2:
                max_tokens = st.number_input("Max Tokens", 100, 8000, 2000)
    
    with col2:
        st.markdown("### 💡 Task Templates")
        
        if st.button("📝 Code Review"):
            st.session_state.template_prompt = "Review the following code for security vulnerabilities and best practices"
            st.session_state.template_type = "code_analysis"
        
        if st.button("🔧 Debug Code"):
            st.session_state.template_prompt = "Debug and fix the following code"
            st.session_state.template_type = "code_generation"
        
        if st.button("🤖 Automate Task"):
            st.session_state.template_prompt = "Create an automation script for"
            st.session_state.template_type = "task_automation"
        
        if st.button("💭 Remember Context"):
            st.session_state.template_prompt = "Remember and summarize our conversation about"
            st.session_state.template_type = "memory_task"
    
    # Apply template if selected
    if hasattr(st.session_state, 'template_prompt'):
        prompt = st.session_state.template_prompt
        task_type = st.session_state.template_type
        del st.session_state.template_prompt
        del st.session_state.template_type
        st.experimental_rerun()
    
    # Execute button
    if st.button("🚀 Execute Task", type="primary", use_container_width=True):
        if prompt:
            with st.spinner("🔄 Processing task..."):
                try:
                    # Parse context
                    context_dict = json.loads(context)
                    context_dict.update({
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    })
                    
                    # Execute task
                    response = httpx.post(
                        f"{orchestrator_url}/execute",
                        json={
                            "task_type": task_type,
                            "prompt": prompt,
                            "context": context_dict,
                            "agents": selected_agents
                        },
                        timeout=300.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Task completed successfully!")
                        
                        # Display results
                        st.markdown("### 📋 Results")
                        
                        # Main result
                        with st.expander("🎯 Task Output", expanded=True):
                            st.json(result.get("result", {}))
                        
                        # Agent details
                        with st.expander("🤖 Agent Execution Details"):
                            agents_used = result.get("agents_used", [])
                            st.write(f"**Agents Used:** {', '.join(agents_used)}")
                            
                            # Show individual agent results
                            if "details" in result.get("result", {}):
                                for agent, details in result["result"]["details"].items():
                                    st.markdown(f"**{agent}:**")
                                    st.json(details)
                        
                        # Metadata
                        with st.expander("📊 Execution Metadata"):
                            st.json(result.get("metadata", {}))
                    else:
                        st.error(f"Task execution failed: {response.text}")
                
                except json.JSONDecodeError:
                    st.error("Invalid JSON in context field")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a task description")

# Tab 2: System Status
with tab2:
    st.header("System Status Dashboard")
    
    # Refresh button
    if st.button("🔄 Refresh Status"):
        st.experimental_rerun()
    
    try:
        # Get health status
        response = httpx.get(f"{orchestrator_url}/health", timeout=10.0)
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Overall status
            status = health_data.get("status", "unknown")
            status_color = "green" if status == "healthy" else "orange"
            
            st.markdown(f"### Overall Status: <span style='color:{status_color}'>{status.upper()}</span>", unsafe_allow_html=True)
            
            # Service status grid
            services = health_data.get("services", {})
            
            if services:
                st.markdown("### 🔧 Service Status")
                
                # Create columns for service status
                cols = st.columns(3)
                
                for i, (service, is_online) in enumerate(services.items()):
                    with cols[i % 3]:
                        if is_online:
                            st.success(f"✅ {service}")
                        else:
                            st.error(f"❌ {service}")
                
                # Service status chart
                st.markdown("### 📊 Service Availability")
                
                service_df = pd.DataFrame([
                    {"Service": service, "Status": "Online" if status else "Offline", "Value": 1}
                    for service, status in services.items()
                ])
                
                fig = px.bar(
                    service_df,
                    x="Service",
                    y="Value",
                    color="Status",
                    color_discrete_map={"Online": "#00CC00", "Offline": "#CC0000"},
                    height=400
                )
                fig.update_layout(showlegend=True, yaxis_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to fetch system status")
    
    except Exception as e:
        st.error(f"Error fetching status: {str(e)}")

# Tab 3: Agent Management
with tab3:
    st.header("AI Agent Management")
    
    # Load agent configuration
    try:
        with open("/workspace/config/agent_integration.json", "r") as f:
            agent_config = json.load(f)
    except:
        agent_config = {"agents": {}, "routing_rules": {}}
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🤖 Available Agents")
        
        for agent_name, agent_info in agent_config.get("agents", {}).items():
            with st.expander(f"**{agent_name.title()}**"):
                st.write(f"**Endpoint:** {agent_info.get('endpoint', 'N/A')}")
                st.write(f"**Capabilities:**")
                for cap in agent_info.get("capabilities", []):
                    st.write(f"- {cap}")
    
    with col2:
        st.markdown("### 🔄 Routing Rules")
        
        routing_rules = agent_config.get("routing_rules", {})
        
        if routing_rules:
            # Create routing visualization
            routing_data = []
            for task, agents in routing_rules.items():
                for agent in agents:
                    routing_data.append({"Task": task, "Agent": agent, "Priority": agents.index(agent) + 1})
            
            if routing_data:
                routing_df = pd.DataFrame(routing_data)
                
                fig = px.sunburst(
                    routing_df,
                    path=["Task", "Agent"],
                    values="Priority",
                    title="Agent Routing Configuration"
                )
                st.plotly_chart(fig, use_container_width=True)

# Tab 4: Vector Databases
with tab4:
    st.header("Vector Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🗄️ ChromaDB")
        
        try:
            # Check ChromaDB status
            response = httpx.get("http://localhost:8000/api/v1/heartbeat", timeout=5.0)
            if response.status_code == 200:
                st.success("ChromaDB is online")
                
                # Get collections
                response = httpx.get("http://localhost:8000/api/v1/collections", timeout=5.0)
                if response.status_code == 200:
                    collections = response.json()
                    st.metric("Collections", len(collections))
                    
                    if collections:
                        st.write("**Available Collections:**")
                        for col in collections[:5]:  # Show first 5
                            st.write(f"- {col.get('name', 'Unknown')}")
            else:
                st.error("ChromaDB is offline")
        except:
            st.error("Cannot connect to ChromaDB")
    
    with col2:
        st.markdown("### 🔍 FAISS")
        
        try:
            # Check FAISS status
            response = httpx.get("http://localhost:8100/health", timeout=5.0)
            if response.status_code == 200:
                st.success("FAISS is online")
                
                health_data = response.json()
                st.metric("Indexed Vectors", health_data.get("vectors", 0))
            else:
                st.error("FAISS is offline")
        except:
            st.error("Cannot connect to FAISS")
    
    # Vector search interface
    st.markdown("### 🔍 Vector Search Test")
    
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_text = st.text_input("Search Query", placeholder="Enter text to search...")
    
    with search_col2:
        search_k = st.number_input("Results", min_value=1, max_value=50, value=10)
    
    if st.button("🔍 Search Vectors"):
        if search_text:
            st.info("Vector search functionality would be implemented here")

# Tab 5: Performance Metrics
with tab5:
    st.header("Performance Metrics")
    
    # Generate sample metrics (in production, these would come from monitoring)
    metrics_data = {
        "timestamps": pd.date_range(start="2024-01-01", periods=24, freq="H"),
        "requests": np.random.randint(50, 200, 24),
        "response_time": np.random.uniform(0.1, 2.0, 24),
        "errors": np.random.randint(0, 10, 24)
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Request volume chart
    st.markdown("### 📊 Request Volume")
    fig_requests = px.line(
        metrics_df,
        x="timestamps",
        y="requests",
        title="Requests per Hour"
    )
    st.plotly_chart(fig_requests, use_container_width=True)
    
    # Response time chart
    st.markdown("### ⏱️ Response Times")
    fig_response = px.scatter(
        metrics_df,
        x="timestamps",
        y="response_time",
        title="Response Time (seconds)",
        color="response_time",
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig_response, use_container_width=True)
    
    # Error rate
    st.markdown("### ❌ Error Rate")
    error_rate = (metrics_df["errors"].sum() / metrics_df["requests"].sum()) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Requests", metrics_df["requests"].sum())
    with col2:
        st.metric("Total Errors", metrics_df["errors"].sum())
    with col3:
        st.metric("Error Rate", f"{error_rate:.2f}%")

# Footer
st.markdown("---")
st.markdown("### 💡 Tips")
st.info("""
- Use specific agents for specialized tasks to get better results
- Complex tasks may take longer to process - be patient
- Check system status regularly to ensure all services are running
- Use the performance metrics to identify bottlenecks
""")
EOF

    log "Streamlit AGI page created"
}

# ===============================================
# CREATE MONITORING SETUP
# ===============================================

setup_monitoring() {
    log_phase "Setting Up AGI Monitoring"
    
    # Create monitoring configuration
    mkdir -p ${WORKSPACE_ROOT}/monitoring/agi
    
    cat > ${WORKSPACE_ROOT}/monitoring/agi/prometheus-agi.yml << EOF
# AGI/ASI System Prometheus Configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agi-orchestrator'
    static_configs:
      - targets: ['localhost:8200']
    metrics_path: '/metrics'

  - job_name: 'ollama'
    static_configs:
      - targets: ['localhost:11434']
    metrics_path: '/api/metrics'

  - job_name: 'litellm'
    static_configs:
      - targets: ['localhost:4000']
    metrics_path: '/metrics'

  - job_name: 'chromadb'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  - job_name: 'ai-agents'
    static_configs:
      - targets: 
        - 'localhost:8283'  # Letta
        - 'localhost:8080'  # AutoGPT
        - 'localhost:8090'  # LocalAGI
        - 'localhost:8085'  # TabbyML
        - 'localhost:8087'  # Semgrep
        - 'localhost:8095'  # LangChain
EOF

    log "Monitoring configuration created"
}

# ===============================================
# CREATE SYSTEMD SERVICES
# ===============================================

create_systemd_services() {
    log_phase "Creating Systemd Services"
    
    # AGI Orchestrator service
    sudo tee /etc/systemd/system/agi-orchestrator.service > /dev/null << EOF
[Unit]
Description=SutazAI AGI Orchestrator
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
User=$(whoami)
WorkingDirectory=${WORKSPACE_ROOT}
ExecStart=/usr/bin/docker-compose -f docker-compose-agi-asi.yml up agi-orchestrator
ExecStop=/usr/bin/docker-compose -f docker-compose-agi-asi.yml stop agi-orchestrator

[Install]
WantedBy=multi-user.target
EOF

    # Vector Router service
    sudo tee /etc/systemd/system/vector-router.service > /dev/null << EOF
[Unit]
Description=SutazAI Vector Database Router
After=network.target

[Service]
Type=simple
Restart=always
RestartSec=10
User=$(whoami)
WorkingDirectory=${WORKSPACE_ROOT}/services/vector_integration
ExecStart=/usr/bin/python3 -m uvicorn vector_router:app --host 0.0.0.0 --port 8150
Environment="PYTHONPATH=${WORKSPACE_ROOT}"

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    log "Systemd services created"
}

# ===============================================
# MAIN INTEGRATION FUNCTION
# ===============================================

main() {
    log_phase "Starting AGI/ASI System Integration"
    
    # Step 1: Analyze existing system
    analyze_existing_system
    
    # Step 2: Integrate model management
    integrate_model_management
    
    # Step 3: Integrate vector databases
    integrate_vector_databases
    
    # Step 4: Integrate AI agents
    integrate_ai_agents
    
    # Step 5: Update Streamlit app
    update_streamlit_app
    
    # Step 6: Setup monitoring
    setup_monitoring
    
    # Step 7: Create systemd services
    create_systemd_services
    
    # Step 8: Display summary
    log_phase "Integration Complete!"
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}AGI/ASI System Integration Summary${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "\n${CYAN}📍 Access Points:${NC}"
    echo -e "  • Streamlit App: ${EXISTING_APP_URL}"
    echo -e "  • AGI Orchestrator: http://${LOCAL_IP}:8200"
    echo -e "  • Vector Router: http://${LOCAL_IP}:8150"
    echo -e "  • LiteLLM Proxy: http://${LOCAL_IP}:4000"
    echo -e "\n${CYAN}🧠 New Features:${NC}"
    echo -e "  • AGI System page in Streamlit app"
    echo -e "  • Unified model management"
    echo -e "  • Intelligent agent routing"
    echo -e "  • Vector database integration"
    echo -e "\n${CYAN}📊 Monitoring:${NC}"
    echo -e "  • Prometheus config: ${WORKSPACE_ROOT}/monitoring/agi/prometheus-agi.yml"
    echo -e "  • System metrics available at each service's /metrics endpoint"
    echo -e "\n${CYAN}🚀 Next Steps:${NC}"
    echo -e "  1. Run: ./deploy_agi_asi_system.sh"
    echo -e "  2. Access the AGI System page in Streamlit"
    echo -e "  3. Test the integrated components"
    echo -e "\n${GREEN}========================================${NC}\n"
}

# Run main function
main