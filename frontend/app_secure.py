"""
SutazAI Frontend - ULTRA-SECURE Version with XSS Protection
Clean, maintainable frontend using extracted page components with comprehensive security
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

# Import secure components
from utils.secure_components import secure, get_csp_header, apply_security_headers
from utils.xss_protection import sanitize_user_input, sanitize_api_response, is_safe_url

# Import optimized and resilient components
from pages import PAGE_REGISTRY, PAGE_CATEGORIES, get_page_function, get_page_icon, get_all_page_names
from utils.resilient_api_client import sync_health_check, sync_call_api, get_system_status, with_api_error_handling
from utils.performance_cache import cache, SmartRefresh
from components.enhanced_ui import ModernMetrics, NotificationSystem
from components.resilient_ui import SystemStatusIndicator, LoadingStateManager, ErrorRecoveryUI, OfflineModeUI
from components.lazy_loader import lazy_loader, SmartPreloader
import time

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

@with_api_error_handling(fallback_value=None, show_user_message=False)
def render_header():
    """Render modern application header with secure health checks"""
    
    # Check if we're in offline mode
    if st.session_state.get('offline_mode', False):
        OfflineModeUI.render_offline_banner()
    
    # Resilient system status check with circuit breaker protection
    health_data = sync_health_check(use_cache=True)
    
    # Sanitize health data
    if health_data:
        health_data = sanitize_api_response(health_data)
    
    # Determine status from resilient health check
    if health_data:
        status = health_data.get("status", "unknown")
        if status == "healthy":
            status_indicator = "🟢"
        elif status == "cached":
            status_indicator = "🟡"
        elif status == "circuit_open":
            status_indicator = "🔴"
        else:
            status_indicator = "⚪"
    else:
        status_indicator = "🔴"
    
    # Header layout with SECURE rendering
    header_col1, header_col2, header_col3 = st.columns([2, 3, 1])
    
    with header_col1:
        # Use secure HTML rendering
        header_html = f"""
        <div style="display: flex; align-items: center; gap: 12px;">
            <h1 style="margin: 0; color: #1a73e8;">🚀 SutazAI</h1>
            <span style="font-size: 1.2em;" title="System Status">{status_indicator}</span>
        </div>
        """
        secure.html(header_html)
    
    with header_col2:
        # Static content - safe
        secure.html("""
        <div style="text-align: center; padding-top: 8px;">
            <h3 style="margin: 0; color: #666;">Autonomous AI Agent Orchestration</h3>
        </div>
        """)
    
    with header_col3:
        # Quick system info with status tooltip - SANITIZED
        current_time = datetime.now().strftime("%H:%M:%S")
        status_text = sanitize_user_input(health_data.get("status", "unknown").title() if health_data else "Unknown")
        
        status_html = f"""
        <div style="text-align: right; padding-top: 12px; font-size: 0.9em; color: #888;">
            <div>Status: {status_text}</div>
            <div style="font-weight: bold;">{current_time}</div>
        </div>
        """
        secure.html(status_html)
    
    # Render system-wide status banner
    SystemStatusIndicator.render_status_banner()

def render_navigation():
    """Render modern sidebar navigation with security"""
    
    with st.sidebar:
        
        # Resilient system status widget with circuit breaker protection
        with st.container():
            secure.markdown("### 🏥 System Status")
            
            # Use resilient health check with extended caching during issues
            health_data = sync_health_check(use_cache=True)
            
            # Sanitize health data
            if health_data:
                health_data = sanitize_api_response(health_data)
            
            if health_data:
                status = health_data.get("status", "unknown")
                
                if status == "healthy":
                    st.success("🟢 All Systems Operational")
                    # Show response time if available
                    if "response_time" in health_data:
                        response_time = float(health_data.get('response_time', 0))
                        secure.text(f"⚡ Response: {response_time:.2f}s")
                        
                elif status == "cached":
                    secure.info("🟡 Using Cached Data")
                    st.caption("Backend may be restarting")
                    
                elif status == "circuit_open":
                    secure.warning("🔴 Service Protection Active")
                    retry_time = health_data.get('retry_in', 60)
                    st.caption(f"Recovery in {retry_time}s")
                    
                elif status in ["startup", "warmup"]:
                    secure.info("🚀 System Starting Up")
                    st.caption("This may take 2-3 minutes")
                    
                elif status == "error":
                    secure.error("🔴 System Error")
                    error_message = sanitize_user_input(health_data.get("message", "Unknown error"))
                    if len(error_message) > 50:
                        with st.expander("Error Details"):
                            secure.text(error_message)
                    else:
                        st.caption(error_message)
                        
                else:
                    secure.warning(f"🟡 Status: {sanitize_user_input(status.title())}")
            else:
                secure.error("🔴 Unable to Check Status")
                st.caption("Backend may be offline")
        
        st.divider()
        
        # Navigation menu - SECURE
        secure.markdown("### 🧭 Navigation")
        
        # Group pages by category
        for category, category_name in PAGE_CATEGORIES.items():
            category_pages = [
                page_name for page_name, info in PAGE_REGISTRY.items()
                if info.get("category") == category
            ]
            
            if category_pages:
                # Sanitize category name
                safe_category_name = sanitize_user_input(category_name)
                secure.markdown(f"**{safe_category_name}**")
                
                for page_name in category_pages:
                    icon = get_page_icon(page_name)
                    # Sanitize page name and icon
                    safe_page_name = sanitize_user_input(page_name)
                    safe_icon = sanitize_user_input(icon)
                    
                    button_style = "primary" if st.session_state.current_page == page_name else "secondary"
                    
                    if st.button(f"{safe_icon} {safe_page_name}", key=f"nav_{page_name}", use_container_width=True, type=button_style):
                        # Update navigation history
                        if page_name != st.session_state.current_page:
                            st.session_state.navigation_history.append(st.session_state.current_page)
                            if len(st.session_state.navigation_history) > 10:
                                st.session_state.navigation_history = st.session_state.navigation_history[-10:]
                        
                        st.session_state.current_page = page_name
                        st.rerun()
                
                secure.write("")  # Add spacing
        
        st.divider()
        
        # Navigation history - SECURE
        if st.session_state.navigation_history:
            secure.markdown("### 📝 Recent Pages")
            recent_pages = list(set(st.session_state.navigation_history[-3:]))
            
            for page in reversed(recent_pages):
                if page != st.session_state.current_page:
                    icon = get_page_icon(page)
                    safe_page = sanitize_user_input(page)
                    safe_icon = sanitize_user_input(icon)
                    
                    if st.button(f"{safe_icon} {safe_page}", key=f"history_{page}", use_container_width=True):
                        st.session_state.current_page = page
                        st.rerun()
        
        st.divider()
        
        # User preferences - SECURE
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
            
            # Performance metrics toggle (new feature)
            show_metrics = st.checkbox(
                "Show Performance Metrics",
                value=st.session_state.user_preferences.get("show_performance_metrics", False),
                help="Display API performance and caching statistics"
            )
            st.session_state.user_preferences["show_performance_metrics"] = show_metrics
        
        # About section - SECURE
        st.divider()
        secure.markdown("""
        ### ℹ️ About
        
        **SutazAI v2.0 - SECURE**  
        Modular Architecture with XSS Protection
        
        🏗️ Components: Modular  
        🔧 Backend: FastAPI  
        🗄️ Database: PostgreSQL  
        🧠 LLM: TinyLlama/Ollama  
        🛡️ Security: XSS Protection Active
        
        ---
        
        💡 **Quick Tips:**
        - Use search in Agent Control
        - Check system status regularly  
        - Export reports for analysis
        - All content is automatically sanitized
        """)

@with_api_error_handling(fallback_value=None, show_user_message=False)  
def render_main_content():
    """Render main content area with current page and error recovery - SECURE"""
    
    current_page = sanitize_user_input(st.session_state.current_page)
    page_function = get_page_function(st.session_state.current_page)  # Use original for function lookup
    
    if page_function:
        try:
            # Add breadcrumb with system status context - SANITIZED
            system_status = get_system_status()
            if system_status:
                system_status = sanitize_api_response(system_status)
            
            system_state = sanitize_user_input(system_status.get("system_state", "unknown") if system_status else "unknown")
            
            # Show loading screen during system startup
            if system_state == "startup" and current_page != "Dashboard":
                LoadingStateManager.render_startup_loading()
                return
                
            # Add breadcrumb - SECURE
            breadcrumb_html = f"""
            <div style="margin-bottom: 1rem; padding: 0.5rem 0; border-bottom: 1px solid #eee;">
                <span style="color: #888;">Navigation:</span>
                <span style="color: #1a73e8; font-weight: bold;">{current_page}</span>
                <span style="color: #999; margin-left: 12px; font-size: 0.8em;">({system_state})</span>
            </div>
            """
            secure.html(breadcrumb_html)
            
            # Render the page with error handling
            page_function()
            
        except Exception as e:
            logger.error(f"Page rendering error for '{current_page}': {e}", exc_info=True)
            
            # Determine error type and show appropriate recovery UI
            error_str = str(e).lower()
            if "timeout" in error_str:
                ErrorRecoveryUI.render_error_recovery_panel("Timeout", current_page)
            elif "connection" in error_str:
                ErrorRecoveryUI.render_error_recovery_panel("Connection", current_page)
            elif "circuit" in error_str:
                ErrorRecoveryUI.render_error_recovery_panel("Circuit_Breaker", current_page)
            elif "cors" in error_str:
                ErrorRecoveryUI.render_error_recovery_panel("CORS", current_page)
            else:
                # Generic error with recovery options - SECURE ERROR DISPLAY
                secure.error(f"⚠️ **Error loading '{current_page}'**")
                error_details = sanitize_user_input(str(e))
                secure.write(f"**Details:** {error_details}")
                
                # Recovery options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Retry Page", use_container_width=True):
                        st.cache_data.clear()
                        st.rerun()
                
                with col2:
                    if st.button("🏠 Go to Dashboard", use_container_width=True):
                        st.session_state.current_page = "Dashboard"
                        st.rerun()
                
                with col3:
                    if st.button("📱 Limited Mode", use_container_width=True):
                        st.session_state['offline_mode'] = True
                        st.rerun()
                
                # Show debug info for developers - SANITIZED
                with st.expander("🐛 Developer Information"):
                    secure.code(f"Page: {current_page}")
                    secure.code(f"Function: {page_function}")
                    secure.code(f"Error: {error_details}")
                    secure.code(f"System State: {system_status.get('system_state', 'unknown') if system_status else 'unknown'}")
                    
                    # Show circuit breaker states
                    circuit_breakers = system_status.get("circuit_breakers", {}) if system_status else {}
                    for name, cb in circuit_breakers.items():
                        safe_name = sanitize_user_input(name)
                        safe_state = sanitize_user_input(cb.get('state', 'unknown'))
                        secure.code(f"Circuit {safe_name}: {safe_state}")
    else:
        secure.error(f"Page '{current_page}' not found!")
        
        # Show available pages with navigation - SECURE
        secure.markdown("### 📄 Available Pages:")
        
        available_pages = get_all_page_names()
        for page_name in available_pages:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                icon = get_page_icon(page_name)
                safe_page_name = sanitize_user_input(page_name)
                safe_icon = sanitize_user_input(icon)
                secure.markdown(f"{safe_icon} **{safe_page_name}**")
            
            with col2:
                if st.button("Go", key=f"nav_to_{page_name}"):
                    st.session_state.current_page = page_name
                    st.rerun()

def main():
    """Main application entry point with comprehensive security and performance optimizations"""
    
    # Apply security headers
    security_headers = apply_security_headers()
    
    # Initialize (Rule 2: preserve existing functionality)
    initialize_session_state()
    
    # Smart component preloading based on current page (performance boost)
    SmartPreloader.preload_for_page()
    
    # Periodic cache cleanup to prevent memory bloat
    if st.session_state.get("last_cache_cleanup", 0) + 300 < time.time():
        expired_count = cache.clear_expired()
        if expired_count > 0:
            logger.info(f"Cache optimization: cleared {expired_count} expired entries")
        st.session_state.last_cache_cleanup = time.time()
    
    # Render UI (preserved functionality with security)
    render_header()
    render_navigation()
    render_main_content()
    
    # Performance monitoring sidebar (new feature for professional deployment)
    if st.session_state.user_preferences.get("show_performance_metrics", False):
        with st.sidebar:
            with st.expander("📊 Performance Metrics"):
                # Cache performance statistics
                cache_stats = cache.get_stats()
                col1, col2 = st.columns(2)
                with col1:
                    secure.metric("Cache Entries", cache_stats['total_entries'])
                with col2:
                    secure.metric("Cache Usage", cache_stats['cache_utilization'])
                
                secure.text(f"💾 Memory: ~{cache_stats['estimated_size_bytes']} bytes")
                
                # Clear cache button
                if st.button("🗑️ Clear Cache", help="Clear all cached data"):
                    cache.clear_all()
                    st.success("Cache cleared!")
                    st.rerun()
    
    # Optimized auto-refresh with smart refresh logic
    if st.session_state.user_preferences.get("auto_refresh", False):
        refresh_interval = st.session_state.user_preferences.get("refresh_interval", 30)
        
        # Smart refresh: only refresh when needed, not blanket refresh
        if SmartRefresh.should_refresh("auto_refresh", refresh_interval):
            SmartRefresh.mark_refreshed("auto_refresh")
            st.rerun()
    
    # SECURE Custom CSS - No user input, static content only
    secure.html("""
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
    
    /* Security notice */
    .security-notice {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 0.8em;
        color: #2e7d32;
        margin-bottom: 1rem;
    }
    </style>
    """)

if __name__ == "__main__":
    main()