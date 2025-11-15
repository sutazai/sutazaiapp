"""
Theme System and Advanced Styling
Provides dark/light mode, theme persistence, and JARVIS-specific styling
"""

import streamlit as st
from typing import Dict, Any

class ThemeManager:
    """
    Manage application themes with persistence and smooth transitions
    """
    
    JARVIS_DARK_THEME = {
        "name": "JARVIS Dark",
        "primary_color": "#00D4FF",
        "secondary_color": "#FF6B6B",
        "background_color": "#0A0E27",
        "secondary_background": "#1A1F3A",
        "text_color": "#FFFFFF",
        "secondary_text": "#B0B0B0",
        "accent_color": "#4CAF50",
        "warning_color": "#FF9800",
        "error_color": "#F44336",
        "success_color": "#4CAF50"
    }
    
    JARVIS_LIGHT_THEME = {
        "name": "JARVIS Light",
        "primary_color": "#0099CC",
        "secondary_color": "#FF5252",
        "background_color": "#FFFFFF",
        "secondary_background": "#F5F5F5",
        "text_color": "#000000",
        "secondary_text": "#666666",
        "accent_color": "#4CAF50",
        "warning_color": "#FF9800",
        "error_color": "#F44336",
        "success_color": "#4CAF50"
    }
    
    @staticmethod
    def get_theme_css(theme: Dict[str, str]) -> str:
        """
        Generate CSS for the specified theme
        
        Args:
            theme: Theme dictionary with color definitions
        
        Returns:
            CSS string
        """
        return f"""
        <style>
            /* Root variables */
            :root {{
                --primary-color: {theme['primary_color']};
                --secondary-color: {theme['secondary_color']};
                --background-color: {theme['background_color']};
                --secondary-background: {theme['secondary_background']};
                --text-color: {theme['text_color']};
                --secondary-text: {theme['secondary_text']};
                --accent-color: {theme['accent_color']};
                --warning-color: {theme['warning_color']};
                --error-color: {theme['error_color']};
                --success-color: {theme['success_color']};
            }}
            
            /* JARVIS Theme Overrides */
            .stApp {{
                background-color: var(--background-color);
                color: var(--text-color);
                transition: background-color 0.3s ease, color 0.3s ease;
            }}
            
            /* Sidebar */
            .css-1d391kg {{
                background-color: var(--secondary-background);
            }}
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {{
                color: var(--primary-color) !important;
                font-family: 'Orbitron', 'Rajdhani', sans-serif;
            }}
            
            /* Links */
            a {{
                color: var(--primary-color);
                text-decoration: none;
                transition: color 0.2s ease;
            }}
            
            a:hover {{
                color: var(--accent-color);
            }}
            
            /* Buttons */
            .stButton > button {{
                background-color: var(--primary-color);
                color: var(--background-color);
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: 600;
                transition: all 0.2s ease;
            }}
            
            .stButton > button:hover {{
                background-color: var(--accent-color);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
            }}
            
            /* Input fields */
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea,
            .stSelectbox > div > div > select {{
                background-color: var(--secondary-background);
                color: var(--text-color);
                border: 1px solid var(--primary-color);
                border-radius: 8px;
                transition: all 0.2s ease;
            }}
            
            .stTextInput > div > div > input:focus,
            .stTextArea > div > div > textarea:focus {{
                border-color: var(--accent-color);
                box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
            }}
            
            /* Chat messages */
            .stChatMessage {{
                background-color: var(--secondary-background);
                border-radius: 12px;
                padding: 1rem;
                margin: 0.5rem 0;
                transition: transform 0.2s ease;
            }}
            
            .stChatMessage:hover {{
                transform: translateX(4px);
            }}
            
            /* Metrics */
            .stMetric {{
                background-color: var(--secondary-background);
                border-radius: 8px;
                padding: 1rem;
            }}
            
            .stMetric > div > div > div {{
                color: var(--primary-color);
            }}
            
            /* Animated background */
            @keyframes pulse {{
                0%, 100% {{
                    opacity: 0.05;
                }}
                50% {{
                    opacity: 0.15;
                }}
            }}
            
            .arc-reactor {{
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 300px;
                height: 300px;
                border-radius: 50%;
                background: radial-gradient(circle, var(--primary-color) 0%, transparent 70%);
                animation: pulse 3s ease-in-out infinite;
                pointer-events: none;
                z-index: -1;
            }}
            
            /* Loading spinner */
            .stSpinner > div {{
                border-top-color: var(--primary-color) !important;
            }}
            
            /* Expander */
            .streamlit-expanderHeader {{
                background-color: var(--secondary-background);
                border-radius: 8px;
                color: var(--primary-color);
            }}
            
            /* Progress bar */
            .stProgress > div > div > div > div {{
                background-color: var(--primary-color);
            }}
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background-color: var(--secondary-background);
                border-radius: 8px 8px 0 0;
                padding: 0.75rem 1.5rem;
                color: var(--secondary-text);
                transition: all 0.2s ease;
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: var(--primary-color);
                color: var(--background-color);
            }}
            
            /* Code blocks */
            .stCodeBlock {{
                background-color: var(--secondary-background);
                border-left: 3px solid var(--primary-color);
            }}
            
            /* Tooltips */
            .stTooltipIcon {{
                color: var(--primary-color);
            }}
            
            /* Scrollbar */
            ::-webkit-scrollbar {{
                width: 10px;
                height: 10px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: var(--secondary-background);
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: var(--primary-color);
                border-radius: 5px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: var(--accent-color);
            }}
        </style>
        """
    
    @staticmethod
    def inject_theme(theme_name: str = "dark"):
        """
        Inject theme CSS into the Streamlit app
        
        Args:
            theme_name: "dark" or "light"
        """
        if theme_name == "light":
            theme = ThemeManager.JARVIS_LIGHT_THEME
        else:
            theme = ThemeManager.JARVIS_DARK_THEME
        
        css = ThemeManager.get_theme_css(theme)
        st.markdown(css, unsafe_allow_html=True)
    
    @staticmethod
    def render_theme_toggle():
        """Render theme toggle control"""
        # Get current theme from session state
        if "theme" not in st.session_state:
            st.session_state.theme = "dark"
        
        # Toggle button
        current_theme = st.session_state.theme
        icon = "🌙" if current_theme == "dark" else "☀️"
        label = "Dark Mode" if current_theme == "dark" else "Light Mode"
        
        if st.button(f"{icon} {label}", key="theme_toggle"):
            st.session_state.theme = "light" if current_theme == "dark" else "dark"
            st.rerun()
        
        return st.session_state.theme
    
    @staticmethod
    def get_current_theme() -> Dict[str, str]:
        """Get current theme dictionary"""
        theme_name = st.session_state.get("theme", "dark")
        if theme_name == "light":
            return ThemeManager.JARVIS_LIGHT_THEME
        else:
            return ThemeManager.JARVIS_DARK_THEME


class UIComponents:
    """Reusable UI components with JARVIS styling"""
    
    @staticmethod
    def render_status_indicator(status: str, label: str = ""):
        """
        Render status indicator dot with label
        
        Args:
            status: "connected", "disconnected", "warning", "error"
            label: Optional text label
        """
        colors = {
            "connected": "#4CAF50",
            "disconnected": "#F44336",
            "warning": "#FF9800",
            "error": "#F44336",
            "success": "#4CAF50"
        }
        
        color = colors.get(status, "#999999")
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: {color};
                animation: pulse 2s ease-in-out infinite;
            "></div>
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_card(title: str, content: str, icon: str = "📋"):
        """
        Render styled card component
        
        Args:
            title: Card title
            content: Card content
            icon: Emoji icon
        """
        st.markdown(f"""
        <div style="
            background-color: var(--secondary-background);
            border-left: 4px solid var(--primary-color);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        ">
            <h4 style="margin: 0 0 0.5rem 0; color: var(--primary-color);">
                {icon} {title}
            </h4>
            <p style="margin: 0; color: var(--text-color);">
                {content}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_separator(text: str = ""):
        """Render styled separator with optional text"""
        if text:
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                margin: 1.5rem 0;
            ">
                <div style="flex: 1; height: 1px; background-color: var(--primary-color);"></div>
                <span style="padding: 0 1rem; color: var(--primary-color); font-weight: 600;">
                    {text}
                </span>
                <div style="flex: 1; height: 1px; background-color: var(--primary-color);"></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height: 1px; background-color: var(--primary-color); margin: 1rem 0;"></div>
            """, unsafe_allow_html=True)


def inject_custom_fonts():
    """Inject custom fonts for JARVIS theme"""
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Rajdhani', sans-serif;
        }
        
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)
