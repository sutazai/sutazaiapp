"""
Advanced Navigation Component for SutazAI Frontend
Modern, categorized navigation with search and accessibility features
"""

import streamlit as st
import asyncio
from typing import Dict, List, Any, Optional
import time

class NavigationManager:
    """Enterprise-grade navigation management with smart categorization"""
    
    def __init__(self):
        self.navigation_categories = {
            "🏠 Core System": {
                "icon": "🏠",
                "pages": [
                    {"id": "dashboard", "name": "Enterprise Dashboard", "icon": "📊", "description": "System overview and metrics"},
                    {"id": "chat", "name": "AI Chat Hub", "icon": "💬", "description": "Central chat interface for all AI models"},
                    {"id": "processing", "name": "automation Processing Engine", "icon": "🧠", "description": "Advanced processing processing center"}
                ]
            },
            "🤖 AI Agents": {
                "icon": "🤖", 
                "pages": [
                    {"id": "agent_control", "name": "Agent Control Center", "icon": "🎛️", "description": "Manage and monitor all AI agents"},
                    {"id": "orchestration", "name": "Agent Orchestration", "icon": "🎯", "description": "Coordinate multi-agent workflows"},
                    {"id": "autogpt", "name": "AutoGPT Tasks", "icon": "🤖", "description": "Autonomous task execution"},
                    {"id": "crewai", "name": "CrewAI Teams", "icon": "👥", "description": "Collaborative agent teams"},
                    {"id": "task_management", "name": "Task Management", "icon": "📋", "description": "Track and manage AI tasks"}
                ]
            },
            "👨‍💻 Developer Tools": {
                "icon": "👨‍💻",
                "pages": [
                    {"id": "dev_suite", "name": "Developer Suite", "icon": "🔧", "description": "Complete development environment"},
                    {"id": "aider", "name": "Aider Code Editor", "icon": "✏️", "description": "AI-powered code editing"},
                    {"id": "gpt_engineer", "name": "GPT Engineer", "icon": "🏗️", "description": "Automated code generation"},
                    {"id": "semgrep", "name": "Semgrep Security", "icon": "🔍", "description": "Code security analysis"},
                    {"id": "tabbyml", "name": "TabbyML Autocomplete", "icon": "🐱", "description": "Intelligent code completion"},
                    {"id": "shellgpt", "name": "ShellGPT Commands", "icon": "🐚", "description": "AI-powered shell commands"}
                ]
            },
            "🌊 Workflow & Automation": {
                "icon": "🌊",
                "pages": [
                    {"id": "langflow", "name": "LangFlow Builder", "icon": "🌊", "description": "Visual workflow builder"},
                    {"id": "flowiseai", "name": "FlowiseAI", "icon": "🌸", "description": "No-code AI workflows"},
                    {"id": "n8n", "name": "n8n Automation", "icon": "🔗", "description": "Workflow automation platform"},
                    {"id": "bigagi", "name": "BigAGI Interface", "icon": "💼", "description": "Advanced automation interface"},
                    {"id": "dify", "name": "Dify Workflows", "icon": "⚡", "description": "LLM application workflows"}
                ]
            },
            "🧮 AI & ML Services": {
                "icon": "🧮",
                "pages": [
                    {"id": "ollama", "name": "Ollama Models", "icon": "🦙", "description": "Local model management"},
                    {"id": "vectors", "name": "Vector Databases", "icon": "🧮", "description": "Embedding and vector search"},
                    {"id": "knowledge", "name": "Knowledge Graphs", "icon": "🕸️", "description": "Knowledge representation"},
                    {"id": "jax", "name": "JAX Machine Learning", "icon": "🔢", "description": "High-performance ML framework"},
                    {"id": "llamaindex", "name": "LlamaIndex RAG", "icon": "📚", "description": "Retrieval augmented generation"}
                ]
            },
            "📊 Analytics & Data": {
                "icon": "📊",
                "pages": [
                    {"id": "analytics", "name": "Advanced Analytics", "icon": "📈", "description": "System performance analytics"},
                    {"id": "monitoring", "name": "System Monitoring", "icon": "📡", "description": "Real-time system monitoring"},
                    {"id": "insights", "name": "Performance Insights", "icon": "🔍", "description": "Performance analysis and optimization"},
                    {"id": "database", "name": "Database Manager", "icon": "💾", "description": "Database administration"}
                ]
            },
            "🎤 Audio & Communication": {
                "icon": "🎤",
                "pages": [
                    {"id": "realtime_stt", "name": "RealtimeSTT Audio", "icon": "🎤", "description": "Real-time speech recognition"},
                    {"id": "voice", "name": "Voice Interface", "icon": "🎙️", "description": "Voice-controlled interactions"}
                ]
            },
            "💰 Business & Finance": {
                "icon": "💰",
                "pages": [
                    {"id": "finrobot", "name": "FinRobot Analysis", "icon": "💰", "description": "Financial analysis and automation"},
                    {"id": "documents", "name": "Document Processing", "icon": "📑", "description": "Intelligent document handling"}
                ]
            },
            "🌐 Web & Automation": {
                "icon": "🌐",
                "pages": [
                    {"id": "browser", "name": "Browser Automation", "icon": "🌐", "description": "Automated web interactions"},
                    {"id": "scraping", "name": "Web Scraping", "icon": "🕷️", "description": "Intelligent web data extraction"}
                ]
            },
            "⚙️ System & Security": {
                "icon": "⚙️",
                "pages": [
                    {"id": "config", "name": "System Configuration", "icon": "⚙️", "description": "System settings and preferences"},
                    {"id": "security", "name": "Security Center", "icon": "🛡️", "description": "Security management and monitoring"},
                    {"id": "network_recon", "name": "Network Reconnaissance", "icon": "🔍", "description": "Advanced network security scanning"},
                    {"id": "self_improvement", "name": "Self-Improvement", "icon": "🚀", "description": "Autonomous system improvement"},
                    {"id": "api_gateway", "name": "API Gateway", "icon": "📱", "description": "API management and routing"}
                ]
            }
        }
        
        self.page_routes = self._build_page_routes()
        
    def _build_page_routes(self) -> Dict[str, str]:
        """Build mapping of page IDs to display names"""
        routes = {}
        for category, data in self.navigation_categories.items():
            for page in data["pages"]:
                routes[page["id"]] = f"{page['icon']} {page['name']}"
        return routes
    
    def render_advanced_navigation(self) -> Optional[str]:
        """Render the advanced navigation interface"""
        
        # Navigation header with search
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h3 style="margin: 0; color: #ffffff; font-size: 1.2rem;">🧠 SutazAI Navigation</h3>
            <p style="margin: 5px 0 0 0; color: #888; font-size: 0.9rem;">Choose a service to access</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Search functionality
        search_term = st.text_input(
            "🔍 Search services...",
            placeholder="Type to search AI services...",
            key="nav_search",
            help="Search across all available AI services and tools"
        )
        
        # Filter categories and pages based on search
        filtered_categories = self._filter_navigation(search_term) if search_term else self.navigation_categories
        
        # Show search results or categorized navigation
        if search_term:
            return self._render_search_results(filtered_categories, search_term)
        else:
            return self._render_categorized_navigation()
    
    def _filter_navigation(self, search_term: str) -> Dict:
        """Filter navigation based on search term"""
        filtered = {}
        search_lower = search_term.lower()
        
        for category, data in self.navigation_categories.items():
            filtered_pages = []
            for page in data["pages"]:
                if (search_lower in page["name"].lower() or 
                    search_lower in page["description"].lower() or
                    search_lower in page["icon"]):
                    filtered_pages.append(page)
            
            if filtered_pages:
                filtered[category] = {**data, "pages": filtered_pages}
        
        return filtered
    
    def _render_search_results(self, filtered_categories: Dict, search_term: str) -> Optional[str]:
        """Render search results"""
        total_results = sum(len(data["pages"]) for data in filtered_categories.values())
        
        if total_results == 0:
            st.warning(f"No services found matching '{search_term}'")
            return None
        
        st.info(f"Found {total_results} services matching '{search_term}'")
        
        # Display results grouped by category
        for category, data in filtered_categories.items():
            if data["pages"]:
                st.markdown(f"**{category}**")
                for page in data["pages"]:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button(
                            page["icon"], 
                            key=f"search_{page['id']}", 
                            help=page["name"],
                            use_container_width=True
                        ):
                            return page["id"]
                    with col2:
                        if st.button(
                            f"{page['name']}\n{page['description']}", 
                            key=f"search_text_{page['id']}",
                            use_container_width=True
                        ):
                            return page["id"]
                st.markdown("---")
        
        return None
    
    def _render_categorized_navigation(self) -> Optional[str]:
        """Render categorized navigation with collapsible sections"""
        
        # Initialize expanded categories in session state
        if 'expanded_categories' not in st.session_state:
            st.session_state.expanded_categories = {"🏠 Core System": True}  # Core open by default
        
        selected_page = None
        
        for category, data in self.navigation_categories.items():
            # Category header with expand/collapse
            is_expanded = st.session_state.expanded_categories.get(category, False)
            
            # Create expandable category section
            with st.expander(f"{data['icon']} {category.split(' ', 1)[1]}", expanded=is_expanded):
                # Update expanded state
                st.session_state.expanded_categories[category] = True
                
                # Render pages in grid layout
                cols = st.columns(2)
                for i, page in enumerate(data["pages"]):
                    with cols[i % 2]:
                        # Modern card-style button
                        if st.button(
                            f"{page['icon']}\n{page['name']}", 
                            key=f"nav_{page['id']}",
                            help=page["description"],
                            use_container_width=True
                        ):
                            selected_page = page["id"]
                            
                        # Show description on hover (using help text)
                        st.caption(page["description"])
        
        return selected_page
    
    def render_breadcrumb(self, current_page_id: str):
        """Render breadcrumb navigation"""
        if not current_page_id:
            return
            
        # Find current page info
        current_page = None
        current_category = None
        
        for category, data in self.navigation_categories.items():
            for page in data["pages"]:
                if page["id"] == current_page_id:
                    current_page = page
                    current_category = category
                    break
            if current_page:
                break
        
        if current_page and current_category:
            st.markdown(f"""
            <div style="
                padding: 8px 12px; 
                background: rgba(255,255,255,0.05); 
                border-radius: 6px; 
                margin-bottom: 15px;
                font-size: 0.9rem;
                color: #888;
            ">
                🏠 Home > {current_category.split(' ', 1)[1]} > {current_page['icon']} {current_page['name']}
            </div>
            """, unsafe_allow_html=True)
    
    def get_page_route(self, page_id: str) -> str:
        """Get the display name for a page ID"""
        return self.page_routes.get(page_id, "Unknown Page")

class StatusIndicator:
    """Modern status indicator component"""
    
    @staticmethod
    def render_system_status(status_data: Dict[str, Any]):
        """Render modern system status with health indicators"""
        
        if not status_data:
            st.markdown("""
            <div style="
                padding: 12px; 
                background: linear-gradient(135deg, #dc3545, #c82333); 
                border-radius: 8px; 
                text-align: center;
                margin-bottom: 15px;
            ">
                <div style="font-size: 1.1rem; font-weight: 600;">🔴 System Offline</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Backend service unavailable</div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Main status indicator
        st.markdown("""
        <div style="
            padding: 12px; 
            background: linear-gradient(135deg, #28a745, #20c997); 
            border-radius: 8px; 
            text-align: center;
            margin-bottom: 15px;
            animation: pulse 2s infinite;
        ">
            <div style="font-size: 1.1rem; font-weight: 600;">🟢 System Online</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">All services operational</div>
        </div>
        
        <style>
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Component status details
        components = status_data.get("components", {})
        if components:
            with st.expander("📊 System Components", expanded=False):
                cols = st.columns(2)
                for i, (component, health) in enumerate(components.items()):
                    with cols[i % 2]:
                        status_icon = "✅" if health.get("status") == "healthy" else "❌"
                        color = "#28a745" if health.get("status") == "healthy" else "#dc3545"
                        
                        st.markdown(f"""
                        <div style="
                            padding: 8px; 
                            border-left: 3px solid {color}; 
                            background: rgba(255,255,255,0.02);
                            margin: 2px 0;
                            border-radius: 0 4px 4px 0;
                        ">
                            {status_icon} <strong>{component}</strong>
                        </div>
                        """, unsafe_allow_html=True)

class QuickActions:
    """Quick action buttons component"""
    
    @staticmethod
    def render_quick_actions():
        """Render quick action buttons for common tasks"""
        
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh System", use_container_width=True, key="quick_refresh"):
                # Clear all cached data
                for key in list(st.session_state.keys()):
                    if key.startswith('cached_'):
                        del st.session_state[key]
                st.success("System refreshed!")
                st.rerun()
        
        with col2:
            if st.button("💬 New Chat", use_container_width=True, key="quick_chat"):
                return "chat"
        
        with col3:
            if st.button("🤖 Agent Status", use_container_width=True, key="quick_agents"):
                return "agent_control"
        
        return None

# Initialize navigation manager
nav_manager = NavigationManager()
status_indicator = StatusIndicator()
quick_actions = QuickActions() 