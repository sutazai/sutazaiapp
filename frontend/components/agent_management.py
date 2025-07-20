import streamlit as st
import requests

def agent_management(BACKEND_URL):
    """
    Renders the agent management interface.
    """
    st.header("🤖 AI Agent Management")
    
    # Refresh agent status
    if st.button("🔄 Refresh Agent Status"):
        try:
            response = requests.get(f"{BACKEND_URL}/api/v1/agents", timeout=10)
            if response.status_code == 200:
                st.session_state.agent_status = response.json()
            else:
                st.error(f"Failed to load agents: {response.status_code}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
    
    # Agent status grid
    agent_data = st.session_state.agent_status.get("agents", {})
    
    if agent_data:
        # Create columns for agent cards
        cols = st.columns(3)
        
        for idx, (agent_name, agent_info) in enumerate(agent_data.items()):
            col = cols[idx % 3]
            
            with col:
                status = agent_info.get("status", "unknown")
                status_color = "🟢" if status == "healthy" else "🔴"
                
                st.markdown(f"""
                <div class="agent-card">
                    <h4>{status_color} {agent_info.get('name', agent_name)}</h4>
                    <p><strong>Type:</strong> {agent_info.get('type', 'Unknown')}</p>
                    <p><strong>Status:</strong> {status}</p>
                    <p><strong>Capabilities:</strong> {', '.join(agent_info.get('capabilities', []))}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Agent actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Test", key=f"test_{agent_name}"):
                        st.info(f"Testing {agent_name}...")
                with col2:
                    if st.button(f"Restart", key=f"restart_{agent_name}"):
                        st.warning(f"Restarting {agent_name}...")
    else:
        st.info("No agent data available. Click refresh to load.")
