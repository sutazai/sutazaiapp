"""
Enhanced UI Components for SutazAI Frontend
Modern, accessible, and performant UI components
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import time
import json

class ModernMetrics:
    """Enhanced metric display components with animations and better UX"""
    
    @staticmethod
    def render_hero_metrics(metrics: Dict[str, Any]):
        """Render hero metrics with enhanced styling and animations"""
        
        st.markdown("""
        <style>
        .hero-metric {
            background: linear-gradient(135deg, rgba(26, 115, 232, 0.1), rgba(106, 75, 162, 0.1));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            min-height: 120px;
        }
        
        .hero-metric::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
            transition: left 0.5s;
        }
        
        .hero-metric:hover::before {
            left: 100%;
        }
        
        .hero-metric:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 8px 0;
        }
        
        .metric-label {
            font-size: 1rem;
            color: #b4b4b4;
            font-weight: 500;
            margin-bottom: 4px;
        }
        
        .metric-change {
            font-size: 0.875rem;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 12px;
            margin-top: 8px;
        }
        
        .metric-positive {
            background: rgba(0, 200, 83, 0.2);
            color: #00c853;
        }
        
        .metric-negative {
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
        }
        
        .metric-neutral {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Define default metrics if not provided
        default_metrics = {
            "total_agents": {"value": 42, "change": "+3", "icon": "🤖"},
            "active_tasks": {"value": 127, "change": "+15", "icon": "⚡"},
            "models_loaded": {"value": 18, "change": "+2", "icon": "🧠"},
            "success_rate": {"value": "94.2%", "change": "+1.3%", "icon": "🎯"}
        }
        
        # Use provided metrics or defaults
        display_metrics = metrics if metrics else default_metrics
        
        cols = st.columns(len(display_metrics))
        
        for i, (key, data) in enumerate(display_metrics.items()):
            with cols[i]:
                # Determine change class
                change = data.get("change", "")
                if change.startswith("+"):
                    change_class = "metric-positive"
                elif change.startswith("-"):
                    change_class = "metric-negative"
                else:
                    change_class = "metric-neutral"
                
                # Render metric card
                st.markdown(f"""
                <div class="hero-metric">
                    <div style="font-size: 2rem; margin-bottom: 8px;">{data.get('icon', '📊')}</div>
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{data.get('value', 'N/A')}</div>
                    <div class="metric-change {change_class}">{change}</div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def render_performance_chart(data: Optional[Dict] = None):
        """Render performance metrics with interactive charts"""
        
        # Generate sample data if none provided
        if not data:
            dates = pd.date_range(start='2025-01-20', end='2025-01-26', freq='h')
            data = {
                'timestamps': dates,
                'cpu_usage': np.random.normal(45, 15, len(dates)).clip(0, 100),
                'memory_usage': np.random.normal(60, 20, len(dates)).clip(0, 100),
                'active_agents': np.random.poisson(25, len(dates)),
                'request_rate': np.random.exponential(50, len(dates))
            }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Active Agents', 'Request Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=data['cpu_usage'],
                name='CPU Usage',
                line=dict(color='#1a73e8', width=2),
                fill='tonexty',
                fillcolor='rgba(26, 115, 232, 0.1)'
            ),
            row=1, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=data['memory_usage'],
                name='Memory Usage',
                line=dict(color='#00c853', width=2),
                fill='tonexty',
                fillcolor='rgba(0, 200, 83, 0.1)'
            ),
            row=1, col=2
        )
        
        # Active Agents
        fig.add_trace(
            go.Bar(
                x=data['timestamps'][::24],  # Daily data
                y=data['active_agents'][::24],
                name='Active Agents',
                marker_color='#764ba2',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Request Rate
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=data['request_rate'],
                name='Request Rate',
                line=dict(color='#f5576c', width=2, dash='dot'),
                mode='lines+markers',
                marker=dict(size=3)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Update x and y axes
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

class LoadingComponents:
    """Modern loading states and skeleton components"""
    
    @staticmethod
    def skeleton_loader(lines: int = 3, height: str = "20px"):
        """Render skeleton loading animation"""
        
        st.markdown(f"""
        <style>
        .skeleton {{
            background: linear-gradient(90deg, rgba(255,255,255,0.05) 25%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 75%);
            background-size: 200% 100%;
            animation: skeleton-loading 1.5s infinite ease-in-out;
            border-radius: 4px;
            height: {height};
            margin: 8px 0;
        }}
        
        @keyframes skeleton-loading {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        </style>
        """, unsafe_allow_html=True)
        
        for i in range(lines):
            width = np.random.randint(60, 100) if i < lines - 1 else np.random.randint(30, 70)
            st.markdown(f'<div class="skeleton" style="width: {width}%;"></div>', unsafe_allow_html=True)
    
    @staticmethod
    def progress_indicator(progress: float, label: str = "", show_percentage: bool = True):
        """Enhanced progress indicator with animations"""
        
        percentage = int(progress * 100)
        
        st.markdown(f"""
        <div style="margin: 20px 0;">
            {f'<div style="color: #b4b4b4; margin-bottom: 8px; font-size: 0.9rem;">{label}</div>' if label else ''}
            <div style="
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                height: 8px;
                overflow: hidden;
                position: relative;
            ">
                <div style="
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    height: 100%;
                    width: {percentage}%;
                    border-radius: 10px;
                    transition: width 0.5s ease-in-out;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                        animation: progress-shine 2s infinite;
                    "></div>
                </div>
            </div>
            {f'<div style="text-align: right; color: #888; font-size: 0.8rem; margin-top: 4px;">{percentage}%</div>' if show_percentage else ''}
        </div>
        
        <style>
        @keyframes progress-shine {{
            0% {{ transform: translateX(-100%); }}
            50% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def spinning_loader(size: str = "40px", message: str = "Loading..."):
        """Modern spinning loader"""
        
        st.markdown(f"""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
        ">
            <div style="
                width: {size};
                height: {size};
                border: 3px solid rgba(255,255,255,0.1);
                border-top: 3px solid #1a73e8;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 16px;
            "></div>
            <div style="color: #b4b4b4; font-size: 0.9rem;">{message}</div>
        </div>
        
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """, unsafe_allow_html=True)

class NotificationSystem:
    """Advanced notification and alert system"""
    
    @staticmethod
    def show_toast(message: str, type: str = "info", duration: int = 3000):
        """Show toast notification"""
        
        colors = {
            "success": "#00c853",
            "error": "#dc3545", 
            "warning": "#ffc107",
            "info": "#1a73e8"
        }
        
        icons = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️", 
            "info": "ℹ️"
        }
        
        color = colors.get(type, colors["info"])
        icon = icons.get(type, icons["info"])
        
        st.markdown(f"""
        <div style="
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(17, 25, 40, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid {color};
            border-radius: 12px;
            padding: 16px;
            min-width: 300px;
            max-width: 500px;
            z-index: 9999;
            animation: toast-slide-in 0.3s ease-out, toast-fade-out 0.3s ease-in {duration/1000 - 0.3}s forwards;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <div style="font-size: 1.2rem;">{icon}</div>
                <div style="
                    color: white;
                    font-size: 0.9rem;
                    line-height: 1.4;
                ">{message}</div>
            </div>
        </div>
        
        <style>
        @keyframes toast-slide-in {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        
        @keyframes toast-fade-out {{
            to {{ transform: translateX(100%); opacity: 0; }}
        }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_alert_banner(message: str, type: str = "info", dismissible: bool = True):
        """Render alert banner with modern styling"""
        
        colors = {
            "success": {"bg": "rgba(0, 200, 83, 0.1)", "border": "#00c853", "text": "#00c853"},
            "error": {"bg": "rgba(220, 53, 69, 0.1)", "border": "#dc3545", "text": "#dc3545"},
            "warning": {"bg": "rgba(255, 193, 7, 0.1)", "border": "#ffc107", "text": "#ffc107"},
            "info": {"bg": "rgba(26, 115, 232, 0.1)", "border": "#1a73e8", "text": "#1a73e8"}
        }
        
        icons = {
            "success": "🎉",
            "error": "🚨",
            "warning": "⚠️",
            "info": "📢"
        }
        
        style = colors.get(type, colors["info"])
        icon = icons.get(type, icons["info"])
        
        # Create unique key for dismissible state
        dismiss_key = f"dismiss_{hash(message)}"
        
        if dismissible and st.session_state.get(dismiss_key, False):
            return
        
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.markdown(f"""
            <div style="
                background: {style['bg']};
                border: 1px solid {style['border']};
                border-radius: 8px;
                padding: 16px;
                margin: 16px 0;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <div style="font-size: 1.2rem;">{icon}</div>
                <div style="
                    color: {style['text']};
                    font-size: 0.9rem;
                    line-height: 1.4;
                    font-weight: 500;
                ">{message}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if dismissible:
                if st.button("✕", key=f"dismiss_btn_{dismiss_key}", help="Dismiss"):
                    st.session_state[dismiss_key] = True
                    st.rerun()

class InteractiveComponents:
    """Advanced interactive UI components"""
    
    @staticmethod
    def render_tabbed_interface(tabs: Dict[str, callable], icons: Dict[str, str] = None):
        """Enhanced tabbed interface with better styling"""
        
        # Enhanced tab styling
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px 20px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1a73e8, #764ba2) !important;
            border-color: transparent !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create tab labels with icons
        tab_labels = []
        for tab_name in tabs.keys():
            icon = icons.get(tab_name, "📄") if icons else "📄"
            tab_labels.append(f"{icon} {tab_name}")
        
        # Render tabs
        tab_objects = st.tabs(tab_labels)
        
        # Execute tab content
        for i, (tab_name, tab_function) in enumerate(tabs.items()):
            with tab_objects[i]:
                tab_function()
    
    @staticmethod
    def render_expandable_card(title: str, content_function: callable, 
                             icon: str = "📄", expanded: bool = False,
                             badge: str = None):
        """Render expandable card with enhanced styling"""
        
        # Create unique key for state management
        card_key = f"card_{hash(title)}"
        
        # Initialize state
        if card_key not in st.session_state:
            st.session_state[card_key] = expanded
        
        # Card header with click handling
        col1, col2, col3 = st.columns([1, 8, 1])
        
        with col1:
            if st.button(
                "▼" if st.session_state[card_key] else "▶", 
                key=f"{card_key}_toggle",
                help="Expand/Collapse"
            ):
                st.session_state[card_key] = not st.session_state[card_key]
                st.rerun()
        
        with col2:
            badge_html = f'<span style="background: #1a73e8; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; margin-left: 8px;">{badge}</span>' if badge else ""
            st.markdown(f"""
            <div style="
                padding: 16px;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.1);
                cursor: pointer;
            ">
                <h4 style="margin: 0; color: white;">{icon} {title}{badge_html}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Card content
        if st.session_state[card_key]:
            with st.container():
                st.markdown('<div style="margin-top: 16px;">', unsafe_allow_html=True)
                content_function()
                st.markdown('</div>', unsafe_allow_html=True)

class AccessibilityEnhancer:
    """Accessibility improvements and WCAG compliance"""
    
    @staticmethod
    def add_aria_labels():
        """Add ARIA labels for screen readers"""
        
        st.markdown("""
        <script>
        // Add ARIA labels for better accessibility
        document.addEventListener('DOMContentLoaded', function() {
            // Label main navigation
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.setAttribute('role', 'navigation');
                sidebar.setAttribute('aria-label', 'Main navigation');
            }
            
            // Label buttons
            const buttons = document.querySelectorAll('button');
            buttons.forEach(button => {
                if (!button.getAttribute('aria-label') && button.textContent) {
                    button.setAttribute('aria-label', button.textContent.trim());
                }
            });
            
            // Label form inputs
            const inputs = document.querySelectorAll('input, select, textarea');
            inputs.forEach(input => {
                const label = input.previousElementSibling;
                if (label && label.tagName === 'LABEL') {
                    input.setAttribute('aria-describedby', label.textContent);
                }
            });
        });
        </script>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def add_focus_management():
        """Improve keyboard navigation and focus management"""
        
        st.markdown("""
        <style>
        /* Enhanced focus indicators */
        button:focus,
        input:focus,
        select:focus,
        textarea:focus {
            outline: 2px solid #1a73e8 !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.3) !important;
        }
        
        /* Skip to main content link */
        .skip-link {
            position: absolute;
            top: -40px;
            left: 6px;
            background: #1a73e8;
            color: white;
            padding: 8px;
            text-decoration: none;
            border-radius: 4px;
            z-index: 10000;
        }
        
        .skip-link:focus {
            top: 6px;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            * {
                border-color: #ffffff !important;
            }
            
            .glass-card,
            .metric-card,
            .hero-metric {
                background: #000000 !important;
                border: 2px solid #ffffff !important;
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
        </style>
        
        <a href="#main-content" class="skip-link">Skip to main content</a>
        """, unsafe_allow_html=True)

# Initialize components
modern_metrics = ModernMetrics()
loading_components = LoadingComponents()
notification_system = NotificationSystem()
interactive_components = InteractiveComponents()
accessibility_enhancer = AccessibilityEnhancer() 