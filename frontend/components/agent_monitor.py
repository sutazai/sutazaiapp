# frontend/components/agent_monitor.py
import streamlit as st
import pandas as pd

class AgentMonitor:
    def __init__(self, api_client):
        self.api_client = api_client

    def render(self):
        st.write("### Active Agents")
        data = {
            'Agent ID': ['agent_001', 'agent_002'],
            'Task': ['Data Analysis', 'Web Research'],
            'Status': ['Running', 'Idle'],
            'Uptime (H)': [2.5, 10.1]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        st.info("Real-time agent monitoring is not fully implemented yet.")
