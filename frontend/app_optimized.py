"""
SutazAI Frontend - ULTRA-OPTIMIZED Architecture
High-performance frontend with lazy loading, caching, and intelligent resource management
"""

import streamlit as st
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page with optimization settings
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

# Import optimized components
from utils.optimized_api_client import optimized_client, sync_check_service_health
from utils.performance_cache import cache
from components.lazy_loader import lazy_loader, SmartPreloader, ConditionalRenderer, LazyLoadMetrics

# Page registry with lazy loading
PAGE_REGISTRY = {
    "Dashboard": {
        "import_path": "pages.dashboard.main_dashboard",
        "function": "show_dashboard",
        "icon": "🏠",
        "category": "main",
        "preload_components": ["plotly_charts"]
    },
    "AI Chat": {
        "import_path": "pages.ai_services.ai_chat",
        "function": "show_ai_chat",
        "icon": "🤖", 
        "category": "ai_services",
        "preload_components": []
    },
    "Agent Control": {
        "import_path": "pages.system.agent_control",
        "function": "show_agent_control",
        "icon": "👥",
        "category": "system",
        "preload_components": ["numpy_compute"]
    },
    "Hardware Optimizer": {
        "import_path": "pages.system.hardware_optimization",
        "function": "show_hardware_optimization",
        "icon": "🔧",
        "category": "system",
        "preload_components": ["plotly_charts", "numpy_compute"]
    },
    "Performance Metrics": {
        "import_path": None,  # Built-in component
        "function": "show_performance_metrics",
        "icon": "📊",
        "category": "system",
        "preload_components": []
    }
}

PAGE_CATEGORIES = {
    "main": "Core Features",
    "ai_services": "AI Services",
    "system": "System Management",
    "analytics": "Analytics & Reports", 
    "integrations": "Integrations"
}

def initialize_optimized_session_state():
    """Initialize session state with performance optimizations"""
    
    # Core navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Performance settings
    if "performance_mode" not in st.session_state:
        st.session_state.performance_mode = "auto"  # auto, fast, quality
    
    # Cache settings
    if "enable_caching" not in st.session_state:
        st.session_state.enable_caching = True
    
    # Auto-refresh settings
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    
    # Lazy loading flags
    if "preloaded_components" not in st.session_state:
        st.session_state.preloaded_components = set()
    
    # Navigation history (limited to prevent memory bloat)
    if "navigation_history" not in st.session_state:
        st.session_state.navigation_history = []

def render_optimized_header():
    """Render application header with cached health status"""
    
    # Use cached health check
    try:
        if st.session_state.get("enable_caching", True):
            health_data = optimized_client.sync_health_check()
            backend_healthy = health_data.get("status") == "healthy"
        else:
            backend_healthy = sync_check_service_health("http://127.0.0.1:10010/health")
    except:
        backend_healthy = False
    
    status_indicator = "🟢" if backend_healthy else "🔴"
    
    # Header layout
    header_col1, header_col2, header_col3 = st.columns([2, 3, 1])
    
    with header_col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px;">
            <h1 style="margin: 0; color: #1a73e8;">🚀 SutazAI</h1>
            <span style="font-size: 1.2em;" title="System Status">{status_indicator}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with header_col2:
        st.markdown(f"""
        <div style="text-align: center; padding-top: 8px;">
            <h3 style="margin: 0; color: #666;">Optimized AI Agent Orchestration</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with header_col3:
        current_time = datetime.now().strftime("%H:%M:%S")
        performance_mode = st.session_state.get("performance_mode", "auto")
        st.markdown(f"""
        <div style="text-align: right; padding-top: 8px; font-size: 0.85em; color: #888;">
            <div>Mode: {performance_mode.title()}</div>
            <div style="font-weight: bold;">{current_time}</div>
        </div>
        """, unsafe_allow_html=True)

def render_optimized_sidebar():
    """Render optimized sidebar with smart loading"""
    
    with st.sidebar:
        # Performance control panel
        with st.expander("⚡ Performance Settings"):
            performance_mode = st.selectbox(
                "Performance Mode:",
                ["auto", "fast", "quality"],
                index=["auto", "fast", "quality"].index(
                    st.session_state.get("performance_mode", "auto")
                ),
                help="Auto: Balance speed and quality, Fast: Prioritize speed, Quality: Best visual experience"
            )
            st.session_state.performance_mode = performance_mode
            
            enable_caching = st.checkbox(
                "Enable Smart Caching",
                value=st.session_state.get("enable_caching", True),
                help="Cache API responses and computed data"
            )
            st.session_state.enable_caching = enable_caching
            
            if enable_caching:
                if st.button("Clear Cache", help="Clear all cached data"):
                    cache.clear_all()
                    st.success("Cache cleared!")
                    st.rerun()
                
                # Show cache stats
                cache_stats = cache.get_stats()
                st.metrics([
                    ("Cache Entries", cache_stats["total_entries"]),
                    ("Cache Usage", cache_stats["cache_utilization"])
                ])
        
        st.divider()
        
        # System status with caching
        with st.container():
            st.markdown("### 🏥 System Status")
            
            try:
                if st.session_state.get("enable_caching", True):
                    dashboard_data = optimized_client.sync_get_dashboard_data()
                    health_data = dashboard_data.get("health", {})
                else:
                    health_data = optimized_client.sync_call_api("/health", timeout=2.0)
                
                if health_data:
                    status = health_data.get("status", "unknown")
                    if status == "healthy":
                        st.success("🟢 All Systems Operational")
                    else:
                        st.warning(f"🟡 Status: {status.title()}")
                else:
                    st.error("🔴 Backend Unreachable")
            except Exception as e:
                logger.error(f"Sidebar health check failed: {e}")
                st.error("🔴 Connection Failed")
        
        st.divider()
        
        # Navigation with smart preloading
        st.markdown("### 🧭 Navigation")
        
        for category, category_name in PAGE_CATEGORIES.items():
            category_pages = [
                page_name for page_name, info in PAGE_REGISTRY.items()
                if info.get("category") == category
            ]
            
            if category_pages:
                st.markdown(f"**{category_name}**")
                
                for page_name in category_pages:
                    page_info = PAGE_REGISTRY[page_name]
                    icon = page_info.get("icon", "📄")
                    button_style = "primary" if st.session_state.current_page == page_name else "secondary"
                    
                    if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", 
                                use_container_width=True, type=button_style):
                        
                        # Preload components for target page
                        preload_components = page_info.get("preload_components", [])
                        if preload_components:
                            for component in preload_components:
                                if component not in st.session_state.preloaded_components:
                                    lazy_loader.preload_components([component])
                                    st.session_state.preloaded_components.add(component)
                        
                        # Update navigation
                        if page_name != st.session_state.current_page:
                            st.session_state.navigation_history.append(st.session_state.current_page)
                            # Limit history to prevent memory bloat
                            if len(st.session_state.navigation_history) > 5:
                                st.session_state.navigation_history = st.session_state.navigation_history[-5:]
                        
                        st.session_state.current_page = page_name
                        st.rerun()
                
                st.write("")
        
        st.divider()
        
        # Performance metrics toggle
        if st.session_state.performance_mode != "fast":
            with st.expander("📊 Performance Metrics"):
                LazyLoadMetrics.render_metrics_widget()
        
        # About section
        st.divider()
        st.markdown(f"""
        ### ℹ️ About
        
        **SutazAI v2.1 OPTIMIZED**  
        Ultra-Performance Architecture
        
        ⚡ **Optimizations:**
        - Smart caching enabled
        - Lazy component loading  
        - Connection pooling
        - Request batching
        
        🏗️ **Stack:**
        - Frontend: Streamlit (Optimized)
        - Backend: FastAPI  
        - Database: PostgreSQL  
        - AI: TinyLlama/Ollama
        
        💡 **Performance Mode:** {st.session_state.get('performance_mode', 'auto').title()}
        """)

async def load_page_component(page_info: Dict[str, Any]):
    """Lazily load page component"""
    import_path = page_info.get("import_path")
    function_name = page_info.get("function")
    
    if not import_path:
        return None
    
    try:
        # Register component if not already done
        component_name = f"page_{import_path.replace('.', '_')}"
        if component_name not in lazy_loader._component_registry:
            lazy_loader.register_component(component_name, import_path)
        
        # Load the module
        module = await lazy_loader.load_component(component_name)
        if module and hasattr(module, function_name):
            return getattr(module, function_name)
        
    except Exception as e:
        logger.error(f"Failed to load page component {import_path}: {e}")
    
    return None

def show_performance_metrics():
    """Built-in performance metrics page"""
    st.header("📊 System Performance Metrics", divider='rainbow')
    
    # Cache statistics
    st.subheader("🧠 Cache Performance")
    cache_stats = cache.get_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cache Entries", cache_stats["total_entries"])
    with col2:
        st.metric("Est. Memory Usage", f"{cache_stats['estimated_size_bytes'] // 1024} KB")
    with col3:
        st.metric("Cache Utilization", cache_stats["cache_utilization"])
    
    # Lazy loading statistics  
    st.subheader("⚡ Lazy Loading Stats")
    loading_stats = LazyLoadMetrics.get_loading_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Components Loaded", f"{loading_stats['total_loaded']}/{loading_stats['total_registered']}")
    with col2:
        st.metric("Loading Efficiency", loading_stats['loading_ratio'])
    
    # Performance controls
    st.subheader("🎛️ Performance Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear All Caches"):
            cache.clear_all()
            st.success("All caches cleared!")
            st.rerun()
    
    with col2:
        expired_count = cache.clear_expired()
        if st.button("🗑️ Clear Expired Data"):
            st.info(f"Cleared {expired_count} expired entries")

def render_optimized_main_content():
    """Render main content with optimized loading"""
    
    current_page = st.session_state.current_page
    page_info = PAGE_REGISTRY.get(current_page)
    
    if not page_info:
        st.error(f"Page '{current_page}' not found!")
        return
    
    # Add breadcrumb
    st.markdown(f"""
    <div style="margin-bottom: 1rem; padding: 0.5rem 0; border-bottom: 1px solid #eee;">
        <span style="color: #888;">Navigation:</span>
        <span style="color: #1a73e8; font-weight: bold;">{current_page}</span>
        <span style="color: #888; font-size: 0.8em; margin-left: 8px;">
            (Mode: {st.session_state.get('performance_mode', 'auto')})
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Handle built-in pages
        if current_page == "Performance Metrics":
            show_performance_metrics()
            return
        
        # Lazy load page component
        if st.session_state.performance_mode == "fast":
            # Fast mode: simpler loading
            import_path = page_info.get("import_path")
            function_name = page_info.get("function")
            
            if import_path and function_name:
                module = __import__(import_path, fromlist=[function_name])
                page_function = getattr(module, function_name)
                page_function()
        else:
            # Quality/Auto mode: optimized loading with progress
            with st.spinner(f"Loading {current_page}..."):
                page_function = asyncio.run(load_page_component(page_info))
                
                if page_function:
                    page_function()
                else:
                    st.error(f"Failed to load page: {current_page}")
                    st.info("Try switching to Fast mode in the sidebar settings.")
        
    except Exception as e:
        logger.error(f"Error loading page {current_page}: {e}")
        st.error(f"Error loading page '{current_page}': {str(e)}")
        
        # Enhanced error handling
        with st.expander("🔧 Troubleshooting"):
            st.markdown(f"""
            **Error Details:**
            - Page: {current_page}
            - Error: {str(e)}
            - Performance Mode: {st.session_state.get('performance_mode')}
            
            **Try these solutions:**
            1. Switch to Fast mode in sidebar settings
            2. Clear cache and refresh
            3. Return to Dashboard
            """)
        
        if st.button("🏠 Return to Dashboard"):
            st.session_state.current_page = "Dashboard"
            st.rerun()

def main():
    """Optimized main application entry point"""
    
    # Initialize optimized session
    initialize_optimized_session_state()
    
    # Auto-preload components for better UX
    if st.session_state.get("enable_caching", True):
        SmartPreloader.preload_for_page()
    
    # Render optimized UI
    render_optimized_header()
    render_optimized_sidebar()
    render_optimized_main_content()
    
    # Performance-aware auto-refresh
    if st.session_state.get("auto_refresh", False):
        refresh_interval = st.session_state.get("refresh_interval", 30)
        if st.session_state.get("last_refresh", 0) + refresh_interval < datetime.now().timestamp():
            st.session_state.last_refresh = datetime.now().timestamp()
            # Only refresh if not in fast mode
            if st.session_state.get("performance_mode") != "fast":
                st.rerun()
    
    # Optimized CSS with performance considerations
    css_complexity = "full" if st.session_state.get("performance_mode") != "fast" else "minimal"
    
    if css_complexity == "full":
        st.markdown("""
        <style>
        .stApp { max-width: 1200px; margin: 0 auto; }
        .stSidebar { background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%); }
        .stButton > button { border-radius: 8px; transition: all 0.2s ease; }
        .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()