# frontend/components/code_editor.py
import streamlit as st

class CodeEditor:
    def __init__(self, api_client):
        self.api_client = api_client

    def render(self):
        st.text_area("Code Prompt", key="code_prompt", height=100, placeholder="Describe the code you want to generate...")
        st.code("# Generated code will appear here", language="python")
