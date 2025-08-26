"""
Notification System for SutazAI Frontend
Toast notifications, alerts, and user feedback system
"""

import streamlit as st
import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)

class NotificationSystem:
    """Advanced notification system for user feedback"""
    
    @staticmethod
    def show_toast(message: str, type: Literal["success", "info", "warning", "error"] = "info") -> None:
        """
        Show toast notification (using Streamlit's built-in notifications)
        """
        try:
            if type == "success":
                st.success(message)
            elif type == "warning":
                st.warning(message) 
            elif type == "error":
                st.error(message)
            else:
                st.info(message)
        except Exception as e:
            logger.error(f"Failed to show toast notification: {e}")
    
    @staticmethod
    def render_alert_banner(message: str, type: Literal["success", "info", "warning", "error"] = "info") -> None:
        """
        Render alert banner with styled appearance
        """
        colors = {
            "success": {"bg": "#d4edda", "border": "#c3e6cb", "text": "#155724"},
            "info": {"bg": "#cce7ff", "border": "#b3d9ff", "text": "#004085"}, 
            "warning": {"bg": "#fff3cd", "border": "#ffeaa7", "text": "#856404"},
            "error": {"bg": "#f8d7da", "border": "#f5c6cb", "text": "#721c24"}
        }
        
        color_scheme = colors.get(type, colors["info"])
        
        st.markdown(f"""
        <div style="
            padding: 12px 16px;
            margin: 8px 0;
            background-color: {color_scheme['bg']};
            border: 1px solid {color_scheme['border']};
            border-left: 4px solid {color_scheme['border']};
            border-radius: 4px;
            color: {color_scheme['text']};
            font-weight: 500;
        ">
            {message}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_progress_notification(message: str, progress: float, details: Optional[str] = None) -> None:
        """
        Show progress notification with progress bar
        """
        st.info(message)
        if 0 <= progress <= 1:
            st.progress(progress, text=details or "")
        else:
            st.text(details or "Processing...")

# Export for easy import
__all__ = ["NotificationSystem"]