# frontend/components/system_metrics.py
import streamlit as st
import pandas as pd
import numpy as np

class SystemMetrics:
    def __init__(self, api_client):
        self.api_client = api_client

    def render(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CPU Usage", "65%", "5%")
        with col2:
            st.metric("Memory Usage", "78%", "-2%")
        with col3:
            st.metric("API Latency (p95)", "450ms", "25ms")

        st.write("### Response Time (last hour)")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Chat', 'Code Gen', 'Documents'])
        st.line_chart(chart_data)
        st.info("Real-time system metrics are not fully implemented yet.")
