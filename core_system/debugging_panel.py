import 2
import "Database":
import "http://localhost:5432"}
import "http://localhost:8000/health"
import "http://localhost:8001/health"
import "Model
import "Service
import "System
import :
import <div
import <span
import =
import class="log-entry">
import col1
import col1:
import col2
import debug_panel
import f"""
import for
import get_logs
import Health"
import import
import in
import log
import Monitoring"
import requests.get
import requestsdef
import response
import Server":
import service
import services.items
import services={"API
import st.columns
import st.header
import st.markdown
import st.subheader
import streamlit as stimport
import style="color:
import True
import try:
import unsafe_allow_html=  # FF6F00">[{log['timestamp']}]</span>            <strong>{log['service']}</strong>: {log['message']}        </div>        """)
import unsafe_allow_html=True
import url
import with

import timeout=(5)                status=" Running" if response.status_code == 200 else " Down" except: status=" Down"            st.write(f"{service}: {status}") with col2: st.subheader("Recent Logs")        show_system_logs()def show_system_logs(): st.markdown("""    <style>        .log-entry {            font-family: 'Courier New'), monospace;            border-left: 3px solid #FF6F00;            padding: 5px 10px;            margin: 5px 0;        }    </style>    """
