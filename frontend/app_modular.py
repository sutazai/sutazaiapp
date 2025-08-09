"""
SutazAI Frontend - Modernized Modular Architecture
Clean, maintainable frontend using extracted page components
"""

import streamlit as st
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Configure page
st.set_page_config(
    page_title="SutazAI - Autonomous AI System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/sutazai/sutazaiapp',
        'Report a bug': 'https://github.com/sutazai/sutazaiapp/issues',
        'About': "SutazAI - Advanced AI Agent Orchestration System"
    }
)

# Add components to path
sys.path.append(os.path.dirname(__file__))

# Import modular components
from pages import PAGE_REGISTRY, PAGE_CATEGORIES, get_page_function, get_page_icon, get_all_page_names
from utils.api_client import sync_call_api, sync_check_service_health
from components.enhanced_ui import ModernMetrics, NotificationSystem

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {
            "theme": "auto",
            "notifications": True,
            "auto_refresh": False
        }
    
    if "navigation_history" not in st.session_state:
        st.session_state.navigation_history = []

def render_header():
    """Render modern application header"""
    
    # System status check
    backend_healthy = sync_check_service_health("http://127.0.0.1:10010/health")
    status_indicator = "🟢" if backend_healthy else "🔴"
    
    # Header layout
    header_col1, header_col2, header_col3 = st.columns([2, 3, 1])
    
    with header_col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px;">
            <h1 style="margin: 0; color: #1a73e8;">🚀 SutazAI</h1>
            <span style="font-size: 1.2em;">{status_indicator}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with header_col2:
        st.markdown(f"""
        <div style="text-align: center; padding-top: 8px;">
            <h3 style="margin: 0; color: #666;">Autonomous AI Agent Orchestration</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with header_col3:
        # Quick system info
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="text-align: right; padding-top: 12px; font-size: 0.9em; color: #888;">
            <div>System Time</div>
            <div style="font-weight: bold;">{current_time}</div>
        </div>
        """, unsafe_allow_html=True)

def render_navigation():
    """Render modern sidebar navigation"""
    
    with st.sidebar:
        
        # System status widget
        with st.container():
            st.markdown("### 🏥 System Status")
            
            try:
                health_data = sync_call_api("/health", timeout=2.0)
                if health_data:
                    status = health_data.get("status", "unknown")
                    if status == "healthy":
                        st.success("🟢 All Systems Operational")
                    else:
                        st.warning(f"🟡 System Status: {status.title()}")
                else:
                    st.error("🔴 Backend Unreachable")
            except:
                st.error("🔴 Connection Failed")
        
        st.divider()
        
        # Navigation menu
        st.markdown("### 🧭 Navigation")
        
        # Group pages by category
        for category, category_name in PAGE_CATEGORIES.items():
            category_pages = [
                page_name for page_name, info in PAGE_REGISTRY.items()
                if info.get("category") == category
            ]
            
            if category_pages:
                st.markdown(f"**{category_name}**")
                
                for page_name in category_pages:
                    icon = get_page_icon(page_name)
                    button_style = "primary" if st.session_state.current_page == page_name else "secondary"
                    
                    if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True, type=button_style):
                        # Update navigation history
                        if page_name != st.session_state.current_page:
                            st.session_state.navigation_history.append(st.session_state.current_page)
                            if len(st.session_state.navigation_history) > 10:
                                st.session_state.navigation_history = st.session_state.navigation_history[-10:]
                        
                        st.session_state.current_page = page_name
                        st.rerun()
                
                st.write("")  # Add spacing
        
        st.divider()
        
        # Navigation history
        if st.session_state.navigation_history:
            st.markdown("### 📝 Recent Pages")
            recent_pages = list(set(st.session_state.navigation_history[-3:]))
            
            for page in reversed(recent_pages):
                if page != st.session_state.current_page:
                    icon = get_page_icon(page)
                    if st.button(f"{icon} {page}", key=f"history_{page}", use_container_width=True):
                        st.session_state.current_page = page
                        st.rerun()
        
        st.divider()
        
        # User preferences
        with st.expander("⚙️ Settings"):
            
            # Theme selection
            theme = st.selectbox(
                "Theme:",
                ["auto", "light", "dark"],
                index=["auto", "light", "dark"].index(st.session_state.user_preferences["theme"])
            )
            st.session_state.user_preferences["theme"] = theme
            
            # Notifications
            notifications = st.checkbox(
                "Enable Notifications",
                value=st.session_state.user_preferences["notifications"]
            )
            st.session_state.user_preferences["notifications"] = notifications
            
            # Auto-refresh
            auto_refresh = st.checkbox(
                "Auto-refresh Data",
                value=st.session_state.user_preferences["auto_refresh"]
            )
            st.session_state.user_preferences["auto_refresh"] = auto_refresh
            
            if auto_refresh:
                refresh_interval = st.slider("Refresh Interval (s)", 5, 60, 30)
                st.session_state.user_preferences["refresh_interval"] = refresh_interval
        
        # About section
        st.divider()
        st.markdown("""
        ### ℹ️ About
        
        **SutazAI v2.0**  
        Modular Architecture
        
        🏗️ Components: Modular  
        🔧 Backend: FastAPI  
        🗄️ Database: PostgreSQL  
        🧠 LLM: TinyLlama/Ollama  
        
        ---
        
        💡 **Quick Tips:**
        - Use search in Agent Control
        - Check system status regularly  
        - Export reports for analysis
        """)

def render_main_content():
    """Render main content area with current page"""
    
    current_page = st.session_state.current_page
    page_function = get_page_function(current_page)
    
    if page_function:
        try:
            # Add breadcrumb
            st.markdown(f"""
            <div style="margin-bottom: 1rem; padding: 0.5rem 0; border-bottom: 1px solid #eee;">
                <span style="color: #888;">Navigation:</span>
                <span style="color: #1a73e8; font-weight: bold;">{current_page}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Render the page
            page_function()
            
        except Exception as e:
            st.error(f"Error loading page '{current_page}': {str(e)}")
            
            # Show fallback
            st.markdown("### 🔧 Page Loading Error")
            st.info("This page is temporarily unavailable. Please try refreshing or select a different page.")
            
            # Debug info in expander
            with st.expander("🐛 Debug Information"):
                st.code(f"Page: {current_page}")
                st.code(f"Function: {page_function}")
                st.code(f"Error: {str(e)}")
                
            # Return to dashboard button
            if st.button("🏠 Return to Dashboard"):
                st.session_state.current_page = "Dashboard"
                st.rerun()
    else:
        st.error(f"Page '{current_page}' not found!")
        
        # Show available pages
        st.markdown("### Available Pages:")
        for page_name in get_all_page_names():
            icon = get_page_icon(page_name)
            st.markdown(f"- {icon} {page_name}")

def main():
    """Main application entry point"""
    
    # Initialize
    initialize_session_state()
    
    # Render UI
    render_header()
    render_navigation()
    render_main_content()
    
    # Auto-refresh logic
    if st.session_state.user_preferences.get("auto_refresh", False):
        refresh_interval = st.session_state.user_preferences.get("refresh_interval", 30)
        
        # Auto-refresh placeholder
        if st.session_state.get("last_refresh", 0) + refresh_interval < datetime.now().timestamp():
            st.session_state.last_refresh = datetime.now().timestamp()
            st.rerun()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #ddd;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #eee;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .stContainer {
        animation: fadeIn 0.5s ease;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()