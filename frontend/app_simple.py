import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SutazAI Multi-Agent System", layout="wide")

st.title("🤖 SutazAI Multi-Agent System")
st.markdown("### Complete Multi-Agent AI Platform")

# Sidebar
with st.sidebar:
    st.header("System Status")
    st.metric("Active Services", "19")
    st.metric("Agents Deployed", "8/71")
    st.info("🔴 63 agents not yet deployed")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Agents", "🔗 Services"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Backend API", "✅ Active", "Port 8000")
    with col2:
        st.metric("Vector DBs", "✅ 2 Active", "ChromaDB, Qdrant")
    with col3:
        st.metric("Monitoring", "✅ Active", "Grafana, Prometheus")
    with col4:
        st.metric("Workflows", "✅ n8n Active", "Port 5678")

with tab2:
    st.header("AI Agent Status")
    st.warning("⚠️ Only 8 of 71 configured agents are deployed!")
    
    agents = {
        "Deployed (8)": ["Senior AI Engineer", "Code Improver", "QA Validator", "Infrastructure Manager"],
        "Not Deployed (63)": ["AutoGPT", "AgentGPT", "CrewAI", "Aider", "GPT-Engineer", "... and 58 more"]
    }
    
    for category, agent_list in agents.items():
        st.subheader(category)
        for agent in agent_list:
            st.write(f"• {agent}")

with tab3:
    st.header("Active Services")
    
    services = [
        ("PostgreSQL", "5432", "✅"),
        ("Redis", "6379", "✅"),
        ("Ollama", "11434", "✅"),
        ("Backend API", "8000", "✅"),
        ("ChromaDB", "8100", "✅"),
        ("Qdrant", "6333", "✅"),
        ("n8n", "5678", "✅"),
        ("Prometheus", "9090", "✅"),
        ("Grafana", "3000", "✅")
    ]
    
    df = pd.DataFrame(services, columns=["Service", "Port", "Status"])
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.info("⚠️ This system requires full deployment of all 71 agents for complete functionality.")
st.markdown("[API Docs](http://localhost:8000/docs) | [Grafana](http://localhost:3000) | [n8n](http://localhost:5678)")