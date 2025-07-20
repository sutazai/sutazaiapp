# frontend/components/document_uploader.py
import streamlit as st

class DocumentUploader:
    def __init__(self, api_client):
        self.api_client = api_client

    def render(self):
        uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'txt', 'md'])
        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' uploaded successfully.")
            st.info("Document processing is not fully implemented yet.")
