import streamlit as st
import requests
import logging
import os
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Backend URL (ensure this matches the actual backend host/port)
# BACKEND_URL = "http://localhost:8000/api/v1"
BACKEND_URL = "http://172.31.77.193:8000/api/v1" # Use the host IP from backend logs
DOCUMENTS_ENDPOINT = f"{BACKEND_URL}/documents"

# --- Helper Functions --- 

def upload_document_to_backend(uploaded_file) -> Optional[Dict[str, Any]]:
    """Uploads the file to the backend /documents/upload endpoint."""
    if not uploaded_file:
        return None
    
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    logger.info(f"Uploading file: {uploaded_file.name} ({uploaded_file.type}) to {DOCUMENTS_ENDPOINT}/upload")
    try:
        response = requests.post(f"{DOCUMENTS_ENDPOINT}/upload", files=files)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Upload response: {result}")
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading file: {e}")
        logger.error(f"Failed to upload document: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during upload: {e}")
        logger.error(f"Unexpected error during document upload API call: {e}", exc_info=True)
        return None

def analyze_document_on_backend(doc_id: str, analysis_type: str = "summary") -> Optional[Dict[str, Any]]:
    """Requests analysis from the backend /documents/{doc_id}/analyze endpoint."""
    url = f"{DOCUMENTS_ENDPOINT}/{doc_id}/analyze"
    params = {"analysis_type": analysis_type}
    logger.info(f"Requesting analysis for doc_id {doc_id} (type: {analysis_type}) from {url}")
    try:
        response = requests.post(url, params=params)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Analysis response for {doc_id}: {result}")
        return result
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(f"Document '{doc_id}' not found on backend.")
        else:
             st.error(f"Error analyzing document '{doc_id}': {e.response.status_code} - {e.response.text}")
        logger.error(f"HTTP error analyzing document {doc_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend for analysis: {e}")
        logger.error(f"Request error analyzing document {doc_id}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis request: {e}")
        logger.error(f"Unexpected error during document analysis API call: {e}", exc_info=True)
        return None

def query_document_on_backend(doc_id: str, query: str) -> Optional[Dict[str, Any]]:
    """Sends a query about a document to the backend /documents/{doc_id}/query endpoint."""
    url = f"{DOCUMENTS_ENDPOINT}/{doc_id}/query"
    payload = {"query": query}
    logger.info(f"Sending query for doc_id {doc_id} to {url}: '{query[:50]}...'")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Query response for {doc_id}: {result}")
        return result
    except requests.exceptions.HTTPError as e:
        st.error(f"Error querying document '{doc_id}': {e.response.status_code} - {e.response.text}")
        logger.error(f"HTTP error querying document {doc_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend for query: {e}")
        logger.error(f"Request error querying document {doc_id}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during query request: {e}")
        logger.error(f"Unexpected error during document query API call: {e}", exc_info=True)
        return None

# --- Streamlit Page Setup --- 
st.set_page_config(page_title="SutazAI Document Analysis", page_icon="📄")
st.title("📄 Document Analysis")

logger.info("Document Analysis UI page loaded.")

# Initialize session state for tracking uploaded documents
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = {} # Store as {doc_id: filename}

# --- UI Sections --- 

st.header("Upload Document")
# Get allowed extensions from settings (consider fetching from backend later if dynamic)
allowed_extensions = get_setting("document_processing.allowed_upload_extensions", [".pdf", ".docx", ".txt", ".md"])
# Streamlit uses list of extensions without dot for type param
st_allowed_types = [ext.lstrip('.') for ext in allowed_extensions]
st.caption(f"Allowed file types: {', '.join(st_allowed_types)}")

uploaded_file = st.file_uploader("Choose a document to upload and analyze", type=st_allowed_types)

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write("File Details:", file_details)
    
    # Use a button to trigger processing
    if st.button(f"Process '{uploaded_file.name}'"):
        with st.spinner(f"Uploading and indexing '{uploaded_file.name}'..."):
            upload_result = upload_document_to_backend(uploaded_file)
            if upload_result and upload_result.get("status") == "success":
                doc_id = upload_result.get("doc_id")
                if doc_id:
                    st.success(f"Document '{uploaded_file.name}' processed successfully! Document ID: {doc_id}")
                    # Store processed document info
                    st.session_state.processed_docs[doc_id] = uploaded_file.name
                    st.rerun() # Rerun to update the selection box
                else:
                     st.error("Processing succeeded but no Document ID received from backend.")
            else:
                # Error handled in helper function
                 pass 

st.markdown("--- ")

st.header("Analyze or Query Document")

if not st.session_state.processed_docs:
    st.info("Upload a document first to analyze or query it.")
else:
    # Create options for the selection box: "doc_id - filename"
    doc_options = {doc_id: f"{doc_id} - {filename}" 
                   for doc_id, filename in st.session_state.processed_docs.items()}
    
    selected_doc_id = st.selectbox(
        "Select Processed Document:", 
        options=list(doc_options.keys()), 
        format_func=lambda doc_id: doc_options.get(doc_id, doc_id) # Show filename in dropdown
    )

    if selected_doc_id:
        st.subheader(f"Actions for: {doc_options[selected_doc_id]}")
        
        col1, col2 = st.columns([1, 3])

        with col1:
            # Analysis Action (Summary)
            if st.button("Get Summary"):
                 with st.spinner(f"Generating summary for {selected_doc_id}..."):
                     analysis_result = analyze_document_on_backend(selected_doc_id, analysis_type="summary")
                     if analysis_result and analysis_result.get("status") == "success":
                         # Store result in session state for display
                         st.session_state[f"analysis_{selected_doc_id}"] = analysis_result.get("result", "No summary returned.")
                     else:
                         st.session_state[f"analysis_{selected_doc_id}"] = "*Failed to get summary.*"
            
            # Display Analysis Result
            if f"analysis_{selected_doc_id}" in st.session_state:
                st.text_area("Summary:", value=st.session_state[f"analysis_{selected_doc_id}"], height=200, disabled=True)

        with col2:
            # Query Action
            user_query = st.text_input(f"Ask a question about {doc_options[selected_doc_id]}:", key=f"query_input_{selected_doc_id}")
            if st.button("Send Query", key=f"query_button_{selected_doc_id}"):
                if user_query:
                    with st.spinner(f"Querying document {selected_doc_id}..."):
                        query_result = query_document_on_backend(selected_doc_id, user_query)
                        if query_result and query_result.get("status") == "success":
                            st.session_state[f"query_answer_{selected_doc_id}"] = query_result.get("answer", "No answer returned.")
                            st.session_state[f"query_sources_{selected_doc_id}"] = query_result.get("sources", [])
                        else:
                            st.session_state[f"query_answer_{selected_doc_id}"] = "*Failed to get answer.*"
                            st.session_state[f"query_sources_{selected_doc_id}"] = []
                else:
                    st.warning("Please enter a question.")
            
            # Display Query Result
            if f"query_answer_{selected_doc_id}" in st.session_state:
                st.markdown("**Answer:**")
                st.markdown(st.session_state[f"query_answer_{selected_doc_id}"])
                sources = st.session_state.get(f"query_sources_{selected_doc_id}", [])
                if sources:
                     st.caption(f"Sources: {', '.join(sources)}") 