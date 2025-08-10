"""
Pages Module - Modular page components extracted from monolith
Contains all major page functions organized by category
"""

# Import all page functions for easy access
from .dashboard.main_dashboard import show_dashboard
from .ai_services.ai_chat import show_ai_chat  
from .system.agent_control import show_agent_control
from .system.hardware_optimization import show_hardware_optimization

# Page registry for navigation
PAGE_REGISTRY = {
    "Dashboard": {
        "function": show_dashboard,
        "icon": "🏠",
        "category": "main"
    },
    "AI Chat": {
        "function": show_ai_chat,
        "icon": "🤖", 
        "category": "ai_services"
    },
    "Agent Control": {
        "function": show_agent_control,
        "icon": "👥",
        "category": "system"
    },
    "Hardware Optimizer": {
        "function": show_hardware_optimization,
        "icon": "🔧",
        "category": "system"
    }
}

# Categories for organization
PAGE_CATEGORIES = {
    "main": "Core Features",
    "ai_services": "AI Services",
    "system": "System Management",
    "analytics": "Analytics & Reports", 
    "integrations": "Integrations"
}

def get_page_function(page_name: str):
    """Get page function by name"""
    page_info = PAGE_REGISTRY.get(page_name)
    if page_info:
        return page_info["function"]
    return None

def get_page_icon(page_name: str) -> str:
    """Get page icon by name"""
    page_info = PAGE_REGISTRY.get(page_name)
    if page_info:
        return page_info["icon"]
    return "📄"

def get_pages_by_category(category: str) -> list:
    """Get all pages in a specific category"""
    return [
        page_name for page_name, info in PAGE_REGISTRY.items()
        if info.get("category") == category
    ]

def get_all_page_names() -> list:
    """Get list of all available page names"""
    return list(PAGE_REGISTRY.keys())