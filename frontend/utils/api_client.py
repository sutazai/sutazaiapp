# frontend/utils/api_client.py
import os
import requests
import streamlit as st

class APIClient:
    def __init__(self, base_url=None):
        # Use environment variable if available, otherwise default to localhost
        if base_url is None:
            base_url = os.environ.get('BACKEND_URL', 'http://localhost:8000')
        self.base_url = base_url

    def post(self, endpoint, data=None, json=None):
        try:
            response = requests.post(f"{self.base_url}{endpoint}", data=data, json=json)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return None
