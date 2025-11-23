# -*- coding: utf-8 -*-

"""
JARVIS Enterprise - Professional AI Assistant Platform
Enterprise-Grade Frontend with Modern UI/UX
Version: 3.0 Enterprise Edition - Enhanced UI/UX
Updated: 2025-11-18
"""

import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.system_monitor import SystemMonitor
from components.voice_assistant import VoiceAssistant
from config.settings import settings
from services.agent_orchestrator import AgentOrchestrator
from services.backend_client_fixed import BackendClient

# Try to import WebRTC utilities when available
try:
    from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True

    class AudioProcessor(AudioProcessorBase):
        """Collect audio frames from WebRTC stream."""

        def __init__(self):
            self.audio_buffer = []

        def recv(self, frame):
            self.audio_buffer.append(frame.to_ndarray())
            return frame

        def get_audio_bytes(self):
            if self.audio_buffer:
                audio_data = np.concatenate(self.audio_buffer)
                return audio_data.tobytes()
            return None

except ImportError:
    WEBRTC_AVAILABLE = False

# ============================================================================
# ENTERPRISE CONFIGURATION - MUST BE FIRST
# ============================================================================

st.set_page_config(
    page_title="JARVIS Enterprise",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",  # Modern full-width layout
    menu_items={
        "Get Help": "https://docs.sutazai.com",
        "Report a bug": "https://github.com/sutazai/sutazaiapp/issues",
        "About": "# JARVIS Enterprise\nProfessional AI Assistant Platform v3.0",
    },
)

# ============================================================================
# ENTERPRISE DESIGN SYSTEM - ENHANCED
# ============================================================================

ENTERPRISE_CSS = """
/* ============================================================================
   DESIGN SYSTEM - Professional Enterprise Theme v3.0
   Based on Modern SaaS Best Practices & Enterprise Standards
   Updated: 2025-11-18 - Comprehensive UI/UX Enhancements
   ============================================================================ */

/* Web Fonts - Professional Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Color Palette - Modern Professional (Enhanced v3.0) */
:root {
    /* Primary Brand Colors - Deep Blues */
    --primary-950: #030712;
    --primary-900: #0F172A;
    --primary-800: #1E293B;
    --primary-700: #334155;
    --primary-600: #475569;
    --primary-500: #64748B;
    
    /* Accent Colors - Vibrant & Accessible */
    --accent-primary: #3B82F6;      /* Professional Blue */
    --accent-primary-light: #60A5FA;
    --accent-primary-dark: #2563EB;
    --accent-secondary: #10B981;     /* Success Green */
    --accent-secondary-light: #34D399;
    --accent-warning: #F59E0B;       /* Warning Amber */
    --accent-warning-light: #FBBF24;
    --accent-error: #EF4444;         /* Error Red */
    --accent-error-light: #F87171;
    --accent-purple: #8B5CF6;        /* Premium Purple */
    --accent-purple-light: #A78BFA;
    --accent-teal: #14B8A6;          /* Teal Accent */
    --accent-indigo: #6366F1;        /* Indigo Accent */
    
    /* Neutral Colors - Extended Range */
    --neutral-50: #F8FAFC;
    --neutral-100: #F1F5F9;
    --neutral-200: #E2E8F0;
    --neutral-300: #CBD5E1;
    --neutral-400: #94A3B8;
    --neutral-500: #64748B;
    --neutral-600: #475569;
    --neutral-700: #334155;
    --neutral-800: #1E293B;
    --neutral-900: #0F172A;
    --neutral-950: #020617;
    
    /* Semantic Colors - Enhanced */
    --success: #10B981;
    --success-bg: rgba(16, 185, 129, 0.1);
    --success-border: rgba(16, 185, 129, 0.3);
    --warning: #F59E0B;
    --warning-bg: rgba(245, 158, 11, 0.1);
    --warning-border: rgba(245, 158, 11, 0.3);
    --error: #EF4444;
    --error-bg: rgba(239, 68, 68, 0.1);
    --error-border: rgba(239, 68, 68, 0.3);
    --info: #3B82F6;
    --info-bg: rgba(59, 130, 246, 0.1);
    --info-border: rgba(59, 130, 246, 0.3);
    
    /* Shadows - Layered System */
    --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    --shadow-inner: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
    
    /* Spacing System (4px base) - Enhanced */
    --space-0: 0;
    --space-px: 1px;
    --space-0-5: 0.125rem;  /* 2px */
    --space-1: 0.25rem;     /* 4px */
    --space-1-5: 0.375rem;  /* 6px */
    --space-2: 0.5rem;      /* 8px */
    --space-2-5: 0.625rem;  /* 10px */
    --space-3: 0.75rem;     /* 12px */
    --space-3-5: 0.875rem;  /* 14px */
    --space-4: 1rem;        /* 16px */
    --space-5: 1.25rem;     /* 20px */
    --space-6: 1.5rem;      /* 24px */
    --space-7: 1.75rem;     /* 28px */
    --space-8: 2rem;        /* 32px */
    --space-9: 2.25rem;     /* 36px */
    --space-10: 2.5rem;     /* 40px */
    --space-11: 2.75rem;    /* 44px */
    --space-12: 3rem;       /* 48px */
    --space-14: 3.5rem;     /* 56px */
    --space-16: 4rem;       /* 64px */
    --space-20: 5rem;       /* 80px */
    --space-24: 6rem;       /* 96px */
    
    /* Additional Semantic Colors for Advanced UI */
    --gradient-primary: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
    --gradient-success: linear-gradient(135deg, #10B981 0%, #14B8A6 100%);
    --gradient-warning: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
    --gradient-glass: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
    
    /* Typography - Professional Scale */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
    --font-size-xs: 0.75rem;      /* 12px */
    --font-size-sm: 0.875rem;     /* 14px */
    --font-size-base: 1rem;       /* 16px */
    --font-size-lg: 1.125rem;     /* 18px */
    --font-size-xl: 1.25rem;      /* 20px */
    --font-size-2xl: 1.5rem;      /* 24px */
    --font-size-3xl: 1.875rem;    /* 30px */
    --font-size-4xl: 2.25rem;     /* 36px */
    --font-size-5xl: 3rem;        /* 48px */
    --line-height-tight: 1.25;
    --line-height-snug: 1.375;
    --line-height-normal: 1.5;
    --line-height-relaxed: 1.625;
    --line-height-loose: 2;
    
    /* Border Radius - Extended System */
    --radius-none: 0;
    --radius-sm: 0.125rem;    /* 2px */
    --radius-md: 0.375rem;    /* 6px */
    --radius-lg: 0.5rem;      /* 8px */
    --radius-xl: 0.75rem;     /* 12px */
    --radius-2xl: 1rem;       /* 16px */
    --radius-3xl: 1.5rem;     /* 24px */
    --radius-full: 9999px;
    
    /* Transitions - Refined Easing */
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: 300ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slower: 500ms cubic-bezier(0.4, 0, 0.2, 1);
    
    /* Z-Index System */
    --z-dropdown: 1000;
    --z-sticky: 1020;
    --z-fixed: 1030;
    --z-modal-backdrop: 1040;
    --z-modal: 1050;
    --z-popover: 1060;
    --z-tooltip: 1070;
}

/* ============================================================================
   GLOBAL RESETS & BASE STYLES - ENHANCED
   ============================================================================ */

* {
    font-family: var(--font-sans);
    box-sizing: border-box;
}

html, body {
    overflow-x: hidden;
    scroll-behavior: smooth;
}

/* Hide Streamlit Branding & Clean UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* App Container - Premium Gradient Background */
.stApp {
    background: linear-gradient(135deg, var(--primary-950) 0%, var(--primary-900) 50%, var(--primary-800) 100%);
    min-height: 100vh;
}

/* Main Content Area - Optimized Spacing */
.main .block-container {
    padding: var(--space-6) var(--space-8);
    max-width: 1400px;
    margin: 0 auto;
}

/* Professional Scrollbar - Enhanced */
::-webkit-scrollbar {
    width: 12px;
    height: 12px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.03);
    border-radius: var(--radius-full);
    margin: var(--space-2);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%);
    border-radius: var(--radius-full);
    border: 3px solid transparent;
    background-clip: padding-box;
    transition: background var(--transition-base);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.5) 0%, rgba(139, 92, 246, 0.5) 100%);
    background-clip: padding-box;
    box-shadow: 0 0 6px rgba(59, 130, 246, 0.4);
}

::-webkit-scrollbar-thumb:active {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.7) 0%, rgba(139, 92, 246, 0.7) 100%);
    background-clip: padding-box;
}

/* ============================================================================
   ENTERPRISE HEADER - ENHANCED
   ============================================================================ */

.enterprise-header {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.06) 100%);
    backdrop-filter: blur(32px) saturate(200%);
    -webkit-backdrop-filter: blur(32px) saturate(200%);
    border-bottom: 1px solid rgba(59, 130, 246, 0.15);
    padding: var(--space-5) var(--space-8);
    margin: calc(var(--space-6) * -1) calc(var(--space-8) * -1) var(--space-8);
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: var(--z-sticky);
    box-shadow: var(--shadow-lg), 0 4px 24px rgba(59, 130, 246, 0.08);
    transition: all var(--transition-base);
}

.enterprise-logo {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    cursor: pointer;
    transition: transform var(--transition-fast);
}

.enterprise-logo:hover {
    transform: translateX(4px);
}

.logo-icon {
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #06B6D4 100%);
    border-radius: var(--radius-2xl);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.875rem;
    box-shadow: var(--shadow-xl), 0 0 40px rgba(59, 130, 246, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
    border: 2px solid rgba(255, 255, 255, 0.15);
    transition: transform var(--transition-base), box-shadow var(--transition-base);
}

.logo-icon::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transform: rotate(45deg);
    animation: shine 3s infinite;
}

@keyframes shine {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

.logo-icon:hover {
    transform: scale(1.05) rotate(2deg);
    box-shadow: var(--shadow-2xl), 0 0 50px rgba(59, 130, 246, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.logo-text {
    display: flex;
    flex-direction: column;
    gap: var(--space-0-5);
}

.logo-title {
    font-size: var(--font-size-xl);
    font-weight: 700;
    color: var(--neutral-50);
    letter-spacing: -0.02em;
    line-height: var(--line-height-tight);
}

.logo-subtitle {
    font-size: var(--font-size-xs);
    color: var(--neutral-400);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

.header-actions {
    display: flex;
    gap: var(--space-3);
    align-items: center;
}

.status-badge {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2-5) var(--space-5);
    border-radius: var(--radius-full);
    font-size: var(--font-size-sm);
    font-weight: 600;
    transition: all var(--transition-base);
}

.status-badge:hover {
    background: rgba(16, 185, 129, 0.15);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.status-badge--healthy {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.08) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6EE7B7;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
}

.status-badge--warning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.08) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #FCD34D;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.2);
}

.status-badge--critical {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.08) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #FCA5A5;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
}

.status-detail {
    display: block;
    font-size: var(--font-size-xs);
    color: var(--neutral-300);
    margin-top: var(--space-0-5);
}

.status-indicator {
    width: 8px;
    height: 8px;
    background: var(--success);
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
    animation: pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

.status-indicator--warning {
    background: var(--warning);
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.6);
}

.status-indicator--critical {
    background: var(--error);
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
}

@keyframes pulse-glow {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.6;
        transform: scale(1.1);
    }
}

/* ============================================================================
   PROFESSIONAL CHAT INTERFACE - ENHANCED
   ============================================================================ */

.chat-container {
    max-width: 1100px;
    margin: 0 auto;
    padding: var(--space-6);
}

.chat-messages {
    display: flex;
    flex-direction: column;
    gap: var(--space-5);
    margin-bottom: var(--space-8);
    min-height: 500px;
    max-height: calc(100vh - 350px);
    overflow-y: auto;
    overflow-x: hidden;
    padding: var(--space-8);
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.04) 0%, rgba(139, 92, 246, 0.02) 100%);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-radius: var(--radius-3xl);
    border: 1px solid rgba(59, 130, 246, 0.12);
    box-shadow: var(--shadow-inner), var(--shadow-xl), 0 8px 32px rgba(0, 0, 0, 0.12);
    position: relative;
}

/* Chat Message Bubbles - Premium Design */
.message {
    display: flex;
    gap: var(--space-4);
    animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    opacity: 0;
    animation-fill-mode: forwards;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(24px) scale(0.92);
        filter: blur(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
        filter: blur(0);
    }
}

@keyframes slideInFromLeft {
    from {
        opacity: 0;
        transform: translateX(-32px) scale(0.92);
        filter: blur(8px);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
        filter: blur(0);
    }
}

@keyframes slideInFromRight {
    from {
        opacity: 0;
        transform: translateX(32px) scale(0.92);
        filter: blur(8px);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
        filter: blur(0);
    }
}

.message-user {
    flex-direction: row-reverse;
}

.message-avatar {
    width: 42px;
    height: 42px;
    border-radius: var(--radius-xl);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    flex-shrink: 0;
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}

.message-avatar::before {
    content: '';
    position: absolute;
    inset: 0;
    background: inherit;
    filter: blur(10px);
    opacity: 0.5;
}

.avatar-user {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-purple));
}

.avatar-assistant {
    background: linear-gradient(135deg, var(--accent-secondary), var(--accent-teal));
}

.message-content {
    max-width: 75%;
    padding: var(--space-4) var(--space-5);
    border-radius: var(--radius-2xl);
    font-size: var(--font-size-base);
    line-height: var(--line-height-relaxed);
    position: relative;
    word-wrap: break-word;
    overflow-wrap: break-word;
    hyphens: auto;
}

.message-user .message-content {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 50%, #1D4ED8 100%);
    color: white;
    border-bottom-right-radius: var(--radius-md);
    box-shadow: 
        var(--shadow-xl), 
        0 8px 32px rgba(59, 130, 246, 0.4),
        0 0 0 1px rgba(255, 255, 255, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.3),
        inset 0 -1px 0 rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-weight: 500;
    position: relative;
    overflow: hidden;
}

.message-user .message-content::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transition: left 0.5s;
}

.message-user:hover .message-content::before {
    left: 100%;
}

.message-assistant .message-content {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.06) 50%, rgba(255, 255, 255, 0.04) 100%);
    color: var(--neutral-50);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-bottom-left-radius: var(--radius-md);
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    box-shadow: 
        var(--shadow-lg), 
        0 8px 24px rgba(0, 0, 0, 0.15),
        0 0 0 1px rgba(255, 255, 255, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.15);
    font-weight: 400;
    position: relative;
}

.message-assistant .message-content::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
}

.message-content {
    transition: all var(--transition-base);
}

.message-content:hover {
    transform: translateY(-3px) scale(1.005);
    box-shadow: 
        var(--shadow-2xl),
        0 12px 40px rgba(0, 0, 0, 0.2);
}

.message-user .message-content:hover {
    box-shadow: 
        var(--shadow-2xl),
        0 12px 48px rgba(59, 130, 246, 0.5),
        0 0 0 1px rgba(255, 255, 255, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.4);
}

.message-timestamp {
    font-size: var(--font-size-xs);
    color: var(--neutral-400);
    margin-top: var(--space-2);
    display: flex;
    align-items: center;
    gap: var(--space-1);
}

.message-actions {
    display: flex;
    gap: var(--space-2);
    margin-top: var(--space-2);
    opacity: 0;
    transition: opacity var(--transition-fast);
}

.message:hover .message-actions {
    opacity: 1;
}

.message-action-btn {
    padding: var(--space-1) var(--space-2);
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-md);
    color: var(--neutral-300);
    font-size: var(--font-size-xs);
    cursor: pointer;
    transition: all var(--transition-fast);
}

.message-action-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: var(--neutral-100);
    transform: translateY(-1px);
}

/* Typing Indicator - Enhanced Animation */
.typing-indicator {
    display: flex;
    gap: var(--space-2);
    padding: var(--space-4) var(--space-5);
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-2xl);
    border-bottom-left-radius: var(--radius-md);
    width: fit-content;
    backdrop-filter: blur(8px);
}

.typing-dot {
    width: 10px;
    height: 10px;
    background: var(--neutral-400);
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) {
    animation-delay: 0s;
}

.typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 60%, 100% {
        transform: translateY(0);
        opacity: 0.4;
    }
    30% {
        transform: translateY(-12px);
        opacity: 1;
    }
}

/* ============================================================================
   PROFESSIONAL INPUT AREA
   ============================================================================ */

.chat-input-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 var(--space-6);
}

.stChatInput {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(139, 92, 246, 0.04) 100%) !important;
    border: 1.5px solid rgba(59, 130, 246, 0.15) !important;
    border-radius: var(--radius-2xl) !important;
    padding: var(--space-4) var(--space-5) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    box-shadow: var(--shadow-md), 0 4px 16px rgba(0, 0, 0, 0.08) !important;
    transition: all var(--transition-base) !important;
}

.stChatInput:hover {
    border-color: rgba(59, 130, 246, 0.3) !important;
    box-shadow: var(--shadow-lg), 0 6px 24px rgba(59, 130, 246, 0.12) !important;
}

.stChatInput:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15), var(--shadow-xl) !important;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.06) 100%) !important;
    transform: translateY(-2px);
}

/* ============================================================================
   PROFESSIONAL BUTTONS - Enhanced v3.0
   ============================================================================ */

.stButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 50%, #1E40AF 100%);
    color: white;
    border: none;
    border-radius: var(--radius-xl);
    padding: var(--space-3-5) var(--space-7);
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.025em;
    transition: all var(--transition-base);
    box-shadow: 
        var(--shadow-lg),
        0 4px 20px rgba(59, 130, 246, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
    text-transform: none;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.6s;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 
        var(--shadow-2xl), 
        0 12px 48px rgba(59, 130, 246, 0.5),
        0 0 0 1px rgba(255, 255, 255, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
    background: linear-gradient(135deg, #2563EB 0%, #1E40AF 50%, #1E3A8A 100%);
}

.stButton > button:active {
    transform: translateY(-1px) scale(0.98);
    box-shadow: 
        var(--shadow-md),
        0 4px 16px rgba(59, 130, 246, 0.4),
        inset 0 2px 4px rgba(0, 0, 0, 0.1);
}

.stButton > button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    box-shadow: var(--shadow-sm);
    background: linear-gradient(135deg, var(--neutral-600) 0%, var(--neutral-700) 100%);
}

/* Secondary Button Variant */
.button-secondary > button,
.stButton.button-secondary > button {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(16px) !important;
    color: var(--neutral-50) !important;
}

.button-secondary > button:hover,
.stButton.button-secondary > button:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
    box-shadow: var(--shadow-lg), 0 8px 24px rgba(255, 255, 255, 0.1) !important;
}

/* Danger Button Variant */
.button-danger > button,
.stButton.button-danger > button {
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
    box-shadow: var(--shadow-md), 0 4px 20px rgba(239, 68, 68, 0.35) !important;
}

.button-danger > button:hover,
.stButton.button-danger > button:hover {
    background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%) !important;
    box-shadow: var(--shadow-xl), 0 8px 32px rgba(239, 68, 68, 0.5) !important;
}

/* Success Button Variant */
.button-success > button,
.stButton.button-success > button {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    box-shadow: var(--shadow-md), 0 4px 20px rgba(16, 185, 129, 0.35) !important;
}

.button-success > button:hover,
.stButton.button-success > button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: var(--shadow-xl), 0 8px 32px rgba(16, 185, 129, 0.5) !important;
}

/* ============================================================================
   ENTERPRISE SIDEBAR
   ============================================================================ */

.sidebar-content {
    background: rgba(255, 255, 255, 0.02);
    border-radius: var(--radius-xl);
    padding: var(--space-5);
    margin-bottom: var(--space-4);
}

.sidebar-section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--neutral-400);
    margin-bottom: var(--space-3);
}

/* ============================================================================
   PROFESSIONAL METRICS CARDS - Enhanced v3.0
   ============================================================================ */

.hero-section {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-8);
    padding: var(--space-7);
    margin: var(--space-8) 0 var(--space-6);
    background: radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.25) 0%, transparent 55%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(15, 118, 110, 0.55) 100%);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: var(--radius-3xl);
    box-shadow: var(--shadow-xl), 0 20px 60px rgba(0, 0, 0, 0.35);
    position: relative;
    overflow: hidden;
}

.hero-section::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(59, 130, 246, 0.12), rgba(99, 102, 241, 0.1) 45%, rgba(139, 92, 246, 0.15));
    opacity: 0.7;
    pointer-events: none;
}

.hero-text {
    flex: 1 1 320px;
    z-index: 1;
}

.hero-kicker {
    text-transform: uppercase;
    letter-spacing: 0.3em;
    font-size: 0.75rem;
    color: var(--accent-primary-light);
    font-weight: 600;
    margin-bottom: var(--space-3);
}

.hero-title {
    font-size: clamp(2.25rem, 3vw, 3.5rem);
    font-weight: 800;
    color: var(--neutral-50);
    margin-bottom: var(--space-3);
}

.hero-subtitle {
    color: var(--neutral-300);
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: var(--space-4);
}

.hero-status {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-2) var(--space-4);
    border-radius: var(--radius-full);
    font-weight: 600;
    border: 1px solid transparent;
}

.hero-status--healthy {
    color: var(--success);
    background: var(--success-bg);
    border-color: var(--success-border);
}

.hero-status--warning {
    color: var(--warning);
    background: var(--warning-bg);
    border-color: var(--warning-border);
}

.hero-status--critical {
    color: var(--error);
    background: var(--error-bg);
    border-color: var(--error-border);
}

.hero-status .status-dot {
    width: 10px;
    height: 10px;
    border-radius: var(--radius-full);
    background: currentColor;
    box-shadow: 0 0 12px currentColor;
}

.status-detail-inline {
    margin-left: var(--space-1);
    color: var(--neutral-400);
    font-size: 0.85rem;
}

.hero-meta {
    margin-top: var(--space-3);
    color: var(--neutral-400);
    font-size: 0.9rem;
}

.hero-visual {
    flex: 0 0 220px;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1;
}

.hero-visual .arc-reactor {
    width: 180px;
    height: 180px;
}

.arc-reactor {
    position: relative;
    border-radius: var(--radius-full);
    border: 4px solid rgba(59, 130, 246, 0.4);
    background: radial-gradient(circle, rgba(59, 130, 246, 0.65) 0%, rgba(99, 102, 241, 0.35) 45%, rgba(15, 23, 42, 0.2) 70%);
    box-shadow:
        0 0 30px rgba(59, 130, 246, 0.85),
        inset 0 0 35px rgba(59, 130, 246, 0.6),
        inset 0 0 60px rgba(99, 102, 241, 0.35);
    animation: reactor-glow 2.6s ease-in-out infinite;
}

.arc-reactor::after {
    content: '';
    position: absolute;
    inset: 20%;
    border-radius: var(--radius-full);
    border: 2px solid rgba(255, 255, 255, 0.45);
    box-shadow:
        0 0 18px rgba(255, 255, 255, 0.35),
        inset 0 0 20px rgba(99, 102, 241, 0.45);
}

@keyframes reactor-glow {
    0%, 100% {
        box-shadow:
            0 0 30px rgba(59, 130, 246, 0.9),
            inset 0 0 35px rgba(59, 130, 246, 0.6),
            inset 0 0 60px rgba(99, 102, 241, 0.35);
        transform: scale(0.98);
    }
    50% {
        box-shadow:
            0 0 55px rgba(96, 165, 250, 1),
            inset 0 0 70px rgba(129, 140, 248, 0.65),
            inset 0 0 95px rgba(59, 130, 246, 0.55);
        transform: scale(1.03);
    }
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: var(--space-5);
    margin-bottom: var(--space-8);
}

.metric-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.04) 50%, rgba(6, 182, 212, 0.06) 100%);
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border: 1.5px solid rgba(59, 130, 246, 0.15);
    border-radius: var(--radius-2xl);
    padding: var(--space-7);
    transition: all var(--transition-slow);
    position: relative;
    overflow: hidden;
    box-shadow: 
        var(--shadow-md),
        0 8px 24px rgba(0, 0, 0, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
    opacity: 0;
    transition: opacity var(--transition-base);
}

.metric-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
    opacity: 0;
    transition: opacity var(--transition-base);
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.03) rotateX(2deg);
    box-shadow: 
        var(--shadow-2xl), 
        0 20px 60px rgba(59, 130, 246, 0.25),
        0 0 0 1px rgba(59, 130, 246, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    border-color: rgba(59, 130, 246, 0.35);
}

.metric-card:hover::before,
.metric-card:hover::after {
    opacity: 1;
}

.metric-label {
    font-size: 0.875rem;
    color: var(--neutral-400);
    margin-bottom: var(--space-3);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 2.25rem;
    font-weight: 800;
    color: var(--neutral-50);
    line-height: 1;
    margin-bottom: var(--space-2);
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-change {
    font-size: 0.875rem;
    margin-top: var(--space-2);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: var(--space-1);
}

.metric-change-positive {
    color: #10B981;
}

.metric-change-positive::before {
    content: '↗';
    font-size: 1.1em;
}

.metric-change-negative {
    color: #EF4444;
}

.metric-change-negative::before {
    content: '↘';
    font-size: 1.1em;
}

.agent-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.07) 0%, rgba(99, 102, 241, 0.05) 50%, rgba(20, 184, 166, 0.06) 100%);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: var(--radius-xl);
    padding: var(--space-5);
    margin: var(--space-3) 0;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    transition: transform var(--transition-base), box-shadow var(--transition-base);
    box-shadow: var(--shadow-sm);
}

.agent-card:hover {
    transform: translateY(-6px);
    box-shadow: var(--shadow-xl), 0 12px 35px rgba(59, 130, 246, 0.25);
    border-color: rgba(59, 130, 246, 0.35);
}

.ws-status {
    display: inline-flex;
    width: 12px;
    height: 12px;
    border-radius: var(--radius-full);
    margin-right: var(--space-2);
    box-shadow: 0 0 10px currentColor;
}

.ws-connected {
    color: var(--success);
    background: var(--success);
    animation: pulse 2s ease-in-out infinite;
}

.ws-disconnected {
    color: var(--error);
    background: var(--error);
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(139, 92, 246, 0.04) 100%) !important;
    border: 1.5px solid rgba(59, 130, 246, 0.2) !important;
    border-radius: var(--radius-xl) !important;
    padding: var(--space-4) var(--space-5) !important;
    color: var(--neutral-50) !important;
    font-size: var(--font-size-base) !important;
    font-weight: 500 !important;
    transition: all var(--transition-base) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    box-shadow: var(--shadow-sm), inset 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}

.stTextInput > div > div > input:hover,
.stTextArea > div > div > textarea:hover,
.stNumberInput > div > div > input:hover {
    border-color: rgba(59, 130, 246, 0.35) !important;
    box-shadow: 
        0 0 0 3px rgba(59, 130, 246, 0.08),
        var(--shadow-md) !important;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.06) 100%) !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 
        0 0 0 4px rgba(59, 130, 246, 0.15), 
        var(--shadow-xl),
        inset 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.08) 100%) !important;
    transform: translateY(-2px);
    outline: none !important;
}

.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder,
.stNumberInput > div > div > input::placeholder {
    color: var(--neutral-400) !important;
    font-weight: 400 !important;
}

/* Enhanced Select/Dropdown */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(139, 92, 246, 0.04) 100%) !important;
    border: 1.5px solid rgba(59, 130, 246, 0.2) !important;
    border-radius: var(--radius-xl) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    transition: all var(--transition-base) !important;
    box-shadow: var(--shadow-sm) !important;
}

.stSelectbox > div > div:hover,
.stMultiSelect > div > div:hover {
    border-color: rgba(59, 130, 246, 0.35) !important;
    box-shadow: 
        0 0 0 3px rgba(59, 130, 246, 0.08),
        var(--shadow-md) !important;
    transform: translateY(-1px);
}

.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 
        0 0 0 4px rgba(59, 130, 246, 0.15),
        var(--shadow-lg) !important;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.06) 100%) !important;
}

/* Select Options Dropdown */
[data-baseweb="select"] > div > div:last-child {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%) !important;
    backdrop-filter: blur(32px) saturate(200%) !important;
    border: 1.5px solid rgba(59, 130, 246, 0.25) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: var(--shadow-2xl), 0 12px 48px rgba(0, 0, 0, 0.4) !important;
    padding: var(--space-2) !important;
    margin-top: var(--space-2) !important;
}

[data-baseweb="select"] li {
    background: transparent !important;
    border-radius: var(--radius-lg) !important;
    padding: var(--space-3) var(--space-4) !important;
    margin: var(--space-1) 0 !important;
    transition: all var(--transition-fast) !important;
    color: var(--neutral-200) !important;
}

[data-baseweb="select"] li:hover {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%) !important;
    color: var(--neutral-50) !important;
    transform: translateX(4px);
}

[data-baseweb="select"] li[aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-purple) 100%) !important;
    color: white !important;
    font-weight: 600 !important;
}

/* Enhanced Expander */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(139, 92, 246, 0.04) 100%) !important;
    border: 1px solid rgba(59, 130, 246, 0.12) !important;
    border-radius: var(--radius-lg) !important;
    padding: var(--space-4) !important;
    transition: all var(--transition-base) !important;
    backdrop-filter: blur(16px) !important;
}

.streamlit-expanderHeader:hover {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.06) 100%) !important;
    border-color: rgba(59, 130, 246, 0.2) !important;
    transform: translateX(4px);
}

.streamlit-expanderContent {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(59, 130, 246, 0.08) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
    padding: var(--space-5) !important;
    backdrop-filter: blur(12px) !important;
}

/* ============================================================================
   RESPONSIVE DESIGN
   ============================================================================ */

@media (max-width: 768px) {
    .enterprise-header {
        flex-direction: column;
        gap: var(--space-3);
    }
    
    .hero-section {
        flex-direction: column;
        text-align: center;
    }
    
    .hero-visual {
        order: -1;
    }
    
    .hero-visual .arc-reactor {
        width: 140px;
        height: 140px;
    }
    
    .message-content {
        max-width: 85%;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
}

/* ============================================================================
   ACCESSIBILITY
   ============================================================================ */

/* Focus Styles */
*:focus-visible {
    outline: 2px solid var(--accent-primary);
    outline-offset: 2px;
    border-radius: var(--radius-sm);
}

button:focus-visible,
a:focus-visible,
input:focus-visible,
textarea:focus-visible {
    outline: 3px solid var(--accent-primary);
    outline-offset: 3px;
}

/* ============================================================================
   PROFESSIONAL TOAST NOTIFICATIONS
   ============================================================================ */

.toast-container {
    position: fixed;
    top: var(--space-6);
    right: var(--space-6);
    z-index: var(--z-tooltip);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    pointer-events: none;
}

.toast {
    padding: var(--space-4) var(--space-6);
    border-radius: var(--radius-xl);
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    box-shadow: var(--shadow-2xl), 0 8px 32px rgba(0, 0, 0, 0.2);
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 320px;
    pointer-events: all;
    animation: slideInRight 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    transition: all var(--transition-base);
    border: 1px solid;
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(100px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
    }
}

.toast-success {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.95) 0%, rgba(5, 150, 105, 0.95) 100%);
    border-color: rgba(255, 255, 255, 0.2);
    color: white;
}

.toast-error {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.95) 0%, rgba(220, 38, 38, 0.95) 100%);
    border-color: rgba(255, 255, 255, 0.2);
    color: white;
}

.toast-warning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.95) 0%, rgba(217, 119, 6, 0.95) 100%);
    border-color: rgba(255, 255, 255, 0.2);
    color: white;
}

.toast-info {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.95) 0%, rgba(37, 99, 235, 0.95) 100%);
    border-color: rgba(255, 255, 255, 0.2);
    color: white;
}

.toast:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-2xl), 0 12px 40px rgba(0, 0, 0, 0.3);
}

/* ============================================================================
   ENHANCED LOADING STATES
   ============================================================================ */

.loading-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid rgba(59, 130, 246, 0.2);
    border-top-color: var(--accent-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.loading-dots {
    display: flex;
    gap: var(--space-2);
    justify-content: center;
    align-items: center;
}

.loading-dot {
    width: 12px;
    height: 12px;
    background: var(--accent-primary);
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
}

.loading-dot:nth-child(1) { animation-delay: -0.32s; }
.loading-dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
    0%, 80%, 100% { 
        transform: scale(0.8);
        opacity: 0.5;
    }
    40% { 
        transform: scale(1.2);
        opacity: 1;
    }
}

/* ============================================================================
   SMOOTH PAGE TRANSITIONS
   ============================================================================ */

.page-transition {
    animation: fadeInScale 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes fadeInScale {
    from {
        opacity: 0;
        transform: scale(0.96);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* ============================================================================
   ENHANCED SIDEBAR
   ============================================================================ */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
    backdrop-filter: blur(32px) saturate(180%);
    -webkit-backdrop-filter: blur(32px) saturate(180%);
    border-right: 1px solid rgba(59, 130, 246, 0.15);
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.2);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--neutral-50);
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-top: var(--space-6);
    margin-bottom: var(--space-4);
}

[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stButton {
    margin-bottom: var(--space-4);
}

/* ============================================================================
   TABS ENHANCEMENT - Professional v3.0
   ============================================================================ */

.stTabs [data-baseweb="tab-list"] {
    gap: var(--space-3);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%);
    padding: var(--space-3);
    border-radius: var(--radius-2xl);
    border: 1.5px solid rgba(255, 255, 255, 0.08);
    box-shadow: var(--shadow-sm), inset 0 1px 2px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-xl);
    padding: var(--space-4) var(--space-6);
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.025em;
    transition: all var(--transition-base);
    background: transparent;
    color: var(--neutral-400);
    position: relative;
    overflow: hidden;
}

.stTabs [data-baseweb="tab"]::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
    opacity: 0;
    transition: opacity var(--transition-fast);
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(59, 130, 246, 0.12);
    color: var(--neutral-100);
    transform: translateY(-2px);
    box-shadow: var(--shadow-sm);
}

.stTabs [data-baseweb="tab"]:hover::before {
    opacity: 1;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #06B6D4 100%) !important;
    color: white !important;
    box-shadow: 
        var(--shadow-lg), 
        0 4px 24px rgba(59, 130, 246, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-2px);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.stTabs [aria-selected="true"]::before {
    opacity: 0;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: var(--space-6);
}

/* ============================================================================
   ADVANCED ACCESSIBILITY MODES
   ============================================================================ */
/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* High Contrast Mode */
@media (prefers-contrast: high) {
    .message-content {
        border: 2px solid currentColor;
    }
}

/* ============================================================================
   PRINT STYLES
   ============================================================================ */

@media print {
    .enterprise-header,
    .sidebar-content,
    .chat-input-container {
        display: none;
    }
    
    .chat-messages {
        max-height: none;
    }
}
"""

class RateLimiter:
    """Simple token bucket rate limiter for chat requests."""

    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque[float] = deque()

    def is_allowed(self) -> Tuple[bool, str]:
        """Return (allowed, message) enforcing rate bounds."""
        current_time = time.time()

        while self.requests and self.requests[0] < current_time - self.time_window:
            self.requests.popleft()

        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True, ""

        oldest_request = self.requests[0]
        wait_time = int(oldest_request + self.time_window - current_time)
        return False, f"Rate limit exceeded. Please wait {wait_time} seconds."

    def reset(self) -> None:
        """Reset limiter state."""
        self.requests.clear()

st.markdown(f"<style>{ENTERPRISE_CSS}</style>", unsafe_allow_html=True)


def ensure_session_state() -> None:
    """Initialize all session state keys exactly once."""

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = RateLimiter(max_requests=20, time_window=60)
    if "last_message_time" not in st.session_state:
        st.session_state.last_message_time = 0.0
    if "message_queue" not in st.session_state:
        st.session_state.message_queue = []
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    if "available_agents" not in st.session_state:
        st.session_state.available_agents = []
    if "backend_client" not in st.session_state:
        st.session_state.backend_client = BackendClient(base_url=settings.BACKEND_URL)
    if "backend_connected" not in st.session_state:
        st.session_state.backend_connected = False
    if "backend_health" not in st.session_state:
        st.session_state.backend_health = {}
    if "websocket_connected" not in st.session_state:
        st.session_state.websocket_connected = False
    if "websocket_latency" not in st.session_state:
        st.session_state.websocket_latency = 0
    if "current_model" not in st.session_state:
        st.session_state.current_model = "tinyllama:latest"
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = "JARVIS Orchestrator"
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = False
    if "is_listening" not in st.session_state:
        st.session_state.is_listening = False
    if "last_transcription" not in st.session_state:
        st.session_state.last_transcription = None
    if "voice_assistant" not in st.session_state:
        st.session_state.voice_assistant = None
    if "agent_orchestrator" not in st.session_state:
        st.session_state.agent_orchestrator = AgentOrchestrator()
    if "last_backend_response" not in st.session_state:
        st.session_state.last_backend_response = {}
    if "performance_history" not in st.session_state:
        st.session_state.performance_history = {
            "cpu": [],
            "memory": [],
        }


ensure_session_state()


# ============================================================================
# METRIC HISTORY UTILITIES
# ============================================================================

def record_metric_history(metric_name: str, value: float, max_points: int = 120) -> None:
    """Append metric snapshot for charts while enforcing retention."""
    history = st.session_state.performance_history.setdefault(metric_name, [])
    history.append({
        "timestamp": datetime.now().isoformat(),
        "value": float(value),
    })
    if len(history) > max_points:
        del history[: len(history) - max_points]


# ============================================================================
# ENTERPRISE HEADER
# ============================================================================

def render_enterprise_header(status_label: str, status_variant: str = "healthy", status_detail: Optional[str] = None):
    """Render professional enterprise header with dynamic health state."""
    badge_class = f"status-badge status-badge--{status_variant}"
    indicator_class = f"status-indicator status-indicator--{status_variant}"
    detail_markup = f"<span class='status-detail'>{status_detail}</span>" if status_detail else ""

    st.markdown(
        f"""
        <div class="enterprise-header">
            <div class="enterprise-logo">
                <div class="logo-icon">🤖</div>
                <div class="logo-text">
                    <div class="logo-title">JARVIS Enterprise</div>
                    <div class="logo-subtitle">AI Assistant Platform</div>
                </div>
            </div>
            <div class="header-actions">
                <div class="{badge_class}">
                    <div class="{indicator_class}"></div>
                    <span>{status_label}</span>
                    {detail_markup}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_enterprise_overview(
    status_label: str,
    status_variant: str,
    status_detail: Optional[str],
    backend_status: bool,
):
    """Render hero section with live backend metrics."""
    health = st.session_state.backend_health or {}
    services = health.get("services") or {}
    total_services = len(services)
    healthy_services = sum(1 for status in services.values() if status)
    uptime = (
        health.get("uptime_human")
        or health.get("uptime")
        or health.get("system_uptime")
        or "Unknown uptime"
    )
    backend_version = health.get("version") or health.get("app_version") or "N/A"

    latency_value = f"{st.session_state.websocket_latency} ms" if backend_status else "N/A"
    latency_detail_text = status_detail or ("Latency stream active" if backend_status else "Awaiting backend")

    model_display = st.session_state.get("current_model") or "Not configured"

    agent_display = st.session_state.get("current_agent") or "Unassigned"
    for agent in st.session_state.get("available_agents", []):
        if agent.get("id") == st.session_state.get("current_agent"):
            agent_display = agent.get("name", agent_display)
            break

    services_value = (
        f"{healthy_services}/{total_services} online" if total_services else "No telemetry"
    )
    services_positive = total_services and healthy_services == total_services

    metrics = [
        {
            "label": "Latency",
            "value": latency_value,
            "detail": latency_detail_text,
            "positive": backend_status,
        },
        {
            "label": "Active Model",
            "value": model_display,
            "detail": "Streaming responses" if backend_status else "Offline fallback",
            "positive": backend_status,
        },
        {
            "label": "Active Agent",
            "value": agent_display,
            "detail": "Coordinating tasks" if backend_status else "Standing by",
            "positive": backend_status,
        },
        {
            "label": "Services",
            "value": services_value,
            "detail": f"Uptime {uptime}",
            "positive": bool(services_positive),
        },
    ]

    hero_html = f"""
    <div class="hero-section">
        <div class="hero-text">
            <div class="hero-kicker">Enterprise Control Center</div>
            <h1 class="hero-title">J.A.R.V.I.S Enterprise</h1>
            <p class="hero-subtitle">
                Unified chat, voice, monitoring, and multi-agent orchestration powered by SutazAI.
                Stay ahead with live telemetry and immediate remediation controls.
            </p>
            <div class="hero-status hero-status--{status_variant}">
                <span class="status-dot"></span>
                <span>{status_label}</span>
                <span class="status-detail-inline">{status_detail or ''}</span>
            </div>
            <div class="hero-meta">Backend version {backend_version} · {uptime}</div>
        </div>
        <div class="hero-visual">
            <div class="arc-reactor"></div>
        </div>
    </div>
    """

    metric_cards = []
    for metric in metrics:
        change_class = "metric-change-positive" if metric["positive"] else "metric-change-negative"
        metric_cards.append(
            f"""
            <div class="metric-card">
                <div class="metric-label">{metric['label']}</div>
                <div class="metric-value">{metric['value']}</div>
                <div class="metric-change {change_class}">{metric['detail']}</div>
            </div>
            """
        )

    st.markdown(hero_html, unsafe_allow_html=True)
    st.markdown(
        f"<div class='metrics-grid'>{''.join(metric_cards)}</div>",
        unsafe_allow_html=True,
    )

# ============================================================================
# PROFESSIONAL CHAT INTERFACE
# ============================================================================

def render_message(message: Dict):
    """Render a single message with professional styling"""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", datetime.now().isoformat())
    
    avatar_class = "avatar-user" if role == "user" else "avatar-assistant"
    message_class = "message message-user" if role == "user" else "message message-assistant"
    avatar_emoji = "👤" if role == "user" else "🤖"
    
    # Format timestamp
    try:
        dt = datetime.fromisoformat(timestamp)
        time_str = dt.strftime("%I:%M %p")
    except:
        time_str = datetime.now().strftime("%I:%M %p")
    
    # Escape HTML in content
    safe_content = content.replace("<", "&lt;").replace(">", "&gt;")
    
    st.markdown(f"""
    <div class="{message_class}">
        <div class="message-avatar {avatar_class}">{avatar_emoji}</div>
        <div class="message-content">
            {safe_content}
            <span class="message-timestamp">{time_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_typing_indicator():
    """Render professional typing indicator"""
    st.markdown("""
    <div class="message message-assistant">
        <div class="message-avatar avatar-assistant">🤖</div>
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_enterprise_chat():
    """Render the enterprise-styled chat experience once."""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="message message-assistant">
                <div class="message-avatar avatar-assistant">🤖</div>
                <div class="message-content">
                    <strong>Welcome to JARVIS Enterprise</strong><br><br>
                    I'm your professional AI assistant. I can help you with:
                    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                        <li>Complex problem solving and analysis</li>
                        <li>Code generation and debugging</li>
                        <li>Data analysis and visualization</li>
                        <li>Research and information synthesis</li>
                        <li>Task automation and orchestration</li>
                    </ul>
                    How can I assist you today?
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.messages:
            render_message(message)

    if st.session_state.is_processing:
        render_typing_indicator()

    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-messages

    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    user_input = st.chat_input("Type your message here...", key="enterprise_chat_input")

    if user_input:
        process_chat_message(user_input)
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-input-container
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container

# ============================================================================
# MAIN APPLICATION
# ============================================================================

render_enterprise_header(status_label="Connected")


# Main chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages
st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

if not st.session_state.messages:
    # Welcome message
    st.markdown("""
    <div class="message message-assistant">
        <div class="message-avatar avatar-assistant">🤖</div>
        <div class="message-content">
            <strong>Welcome to JARVIS Enterprise</strong><br><br>
            I'm your professional AI assistant. I can help you with:
            <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                <li>Complex problem solving and analysis</li>
                <li>Code generation and debugging</li>
                <li>Data analysis and visualization</li>
                <li>Research and information synthesis</li>
                <li>Task automation and orchestration</li>
            </ul>
            How can I assist you today?
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        render_message(message)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-messages

# Chat input
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
user_input = st.chat_input("Type your message here...", key="chat_input")

if user_input:
    # Add user message
    user_msg = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_msg)
    
    # Show typing indicator (in production, this would be during actual processing)
    with st.spinner("JARVIS is thinking..."):
        time.sleep(0.5)  # Simulate processing
        
        # Add assistant response (integrate with your actual backend)
        assistant_msg = {
            "role": "assistant",
            "content": f"I received your message: '{user_input}'. In production, this would be processed by the AI backend and return an intelligent response.",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(assistant_msg)
    
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-input-container
st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container

# ============================================================================
# SIDEBAR - ENTERPRISE CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <div class="sidebar-section-title">Quick Actions</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat_header", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export", key="export_header", use_container_width=True):
            # Export functionality
            chat_export = "\n\n".join([
                f"[{msg['timestamp']}] {msg['role'].upper()}: {msg['content']}"
                for msg in st.session_state.messages
            ])
            st.download_button(
                "Download Chat",
                chat_export,
                key="download_chat_header",
                file_name=f"jarvis_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    st.markdown("""
    <div class="sidebar-content">
        <div class="sidebar-section-title">Model Configuration</div>
    </div>
    """, unsafe_allow_html=True)
    
    model = st.selectbox(
        "AI Model",
        ["tinyllama:latest", "llama2:latest", "mistral:latest"],
        key="model_select"
    )
    
    agent = st.selectbox(
        "Agent Type",
        ["JARVIS Orchestrator", "Code Assistant", "Research Agent", "Data Analyst"],
        key="agent_select"
    )
    
    st.markdown("""
    <div class="sidebar-content">
        <div class="sidebar-section-title">System Status</div>
    </div>
    """, unsafe_allow_html=True)
    
    # System metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Latency", f"{st.session_state.websocket_latency}ms")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: var(--neutral-400); font-size: 0.875rem;">
    <p>JARVIS Enterprise v2.0 | Powered by SutazAI Platform</p>
    <p style="font-size: 0.75rem; margin-top: 0.5rem;">
        © 2025 SutazAI. All rights reserved. | 
        <a href="#" style="color: var(--accent-primary);">Privacy Policy</a> | 
        <a href="#" style="color: var(--accent-primary);">Terms of Service</a>
    </p>
</div>
""", unsafe_allow_html=True)




# Function to check backend connection
def check_backend_connection():
    """Check if backend is connected and update status"""
    try:
        start_time = time.time()
        health = st.session_state.backend_client.check_health_sync()
        latency_ms = int((time.time() - start_time) * 1000)
        st.session_state.websocket_latency = max(latency_ms, 0)
        st.session_state.backend_connected = health.get("status") not in {"error", "down"}
        st.session_state.backend_health = health
        return st.session_state.backend_connected
    except Exception:
        st.session_state.backend_connected = False
        st.session_state.websocket_latency = 0
        return False

# Function to initialize WebSocket connection
def initialize_websocket():
    """Initialize WebSocket connection for real-time updates"""
    if not st.session_state.websocket_connected:
        def on_ws_message(message):
            """Handle WebSocket messages"""
            if message.get("type") == "chat_update":
                # Update chat in real-time
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "timestamp": datetime.now().isoformat()
                })
            elif message.get("type") == "status_update":
                # Update status
                st.session_state.backend_connected = message.get("connected", False)
        
        def on_ws_error(error):
            """Handle WebSocket errors"""
            st.session_state.websocket_connected = False
            print(f"WebSocket error: {error}")
        
        # Connect WebSocket
        ws_thread = st.session_state.backend_client.connect_websocket(
            on_message=on_ws_message,
            on_error=on_ws_error
        )
        
        if ws_thread and ws_thread.is_alive():
            st.session_state.websocket_connected = True

# Function to load available models and agents
def load_backend_resources():
    """Load available models and agents from backend"""
    if st.session_state.backend_connected:
        try:
            st.session_state.available_models = st.session_state.backend_client.get_models_sync()
            st.session_state.available_agents = st.session_state.backend_client.get_agents_sync()
        except Exception as e:
            print(f"Failed to load resources: {e}")

def get_voice_assistant() -> Optional[VoiceAssistant]:
    """Lazily create the voice assistant when voice features are enabled"""
    if not settings.ENABLE_VOICE_COMMANDS:
        return None
    assistant = st.session_state.get("voice_assistant")
    if assistant is None:
        assistant = VoiceAssistant()
        st.session_state.voice_assistant = assistant
    return assistant

def synthesize_speech(text: str) -> bool:
    """Speak text using the local voice assistant when available"""
    assistant = get_voice_assistant()
    if assistant and assistant.tts_available:
        assistant.speak(text, wait=False)
        return True
    return False

# Function to process chat message
def process_chat_message(user_message: str):
    """Process user chat message with rate limiting and throttling"""
    if not user_message or not user_message.strip():
        return
    
    cleaned_message = user_message.strip()
    
    # Check rate limit
    allowed, message = st.session_state.rate_limiter.is_allowed()
    if not allowed:
        st.error(f"⚠️ {message}")
        return
    
    # Check message throttling (minimum 100ms between messages)
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_message_time
    if time_since_last < 0.1:  # 100ms throttle
        st.warning("⚠️ Please wait a moment before sending another message.")
        return
    
    st.session_state.last_message_time = current_time
    
    # Append the user message immediately so it renders in the chat history
    user_entry = {
        "role": "user",
        "content": cleaned_message,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state.messages.append(user_entry)
    
    # Send request to backend
    st.session_state.is_processing = True
    assistant_text = None
    assistant_metadata: Dict[str, Any] = {}
    backend_payload: Dict[str, Any] = {}
    try:
        with st.spinner("JARVIS is thinking..."):
            backend_payload = st.session_state.backend_client.chat_sync(
                cleaned_message,
                agent=st.session_state.current_agent or "default",
                stream=False,
            )
    except Exception as exc:
        assistant_text = f"⚠️ Unable to reach backend: {exc}"
        assistant_metadata = {"model": "unavailable", "agent": st.session_state.current_agent}
    else:
        assistant_text = (
            backend_payload.get("response")
            or backend_payload.get("content")
            or backend_payload.get("message")
        )

        if not assistant_text and backend_payload.get("choices"):
            first_choice = backend_payload["choices"][0]
            if isinstance(first_choice, dict):
                assistant_text = first_choice.get("message", {}).get("content") or first_choice.get("content")

        if not assistant_text:
            assistant_text = "I wasn't able to parse the backend response."
        
        assistant_metadata = backend_payload.get("metadata", {}) or {}
        if backend_payload.get("model") and not assistant_metadata.get("model"):
            assistant_metadata["model"] = backend_payload["model"]
        if backend_payload.get("usage") and not assistant_metadata.get("usage"):
            assistant_metadata["usage"] = backend_payload["usage"]
        assistant_metadata.setdefault("agent", st.session_state.current_agent)
        assistant_metadata.setdefault("success", backend_payload.get("success", True))
    finally:
        st.session_state.is_processing = False
        st.session_state.last_backend_response = backend_payload
    
    assistant_entry = {
        "role": "assistant",
        "content": assistant_text,
        "timestamp": datetime.now().isoformat(),
        "metadata": assistant_metadata,
    }
    st.session_state.messages.append(assistant_entry)

# Function to process voice input
def process_voice_input(audio_bytes):
    """Process voice input and convert to text"""
    if not settings.ENABLE_VOICE_COMMANDS:
        return None
    try:
        # Send audio to backend for processing
        result = st.session_state.backend_client.send_voice_sync(audio_bytes)
        
        if result and "text" in result:
            return result["text"]
        else:
            # Fallback to local processing
            assistant = get_voice_assistant()
            if assistant:
                return assistant.process_audio_bytes(audio_bytes)
            return None
    except Exception as e:
        print(f"Voice processing error: {e}")
        return None

# Determine backend status for header
backend_status = check_backend_connection()
health_status = st.session_state.backend_health.get("status")

if backend_status and health_status == "degraded":
    header_variant = "warning"
    status_label = "Backend Degraded"
elif backend_status:
    header_variant = "healthy"
    status_label = "All Systems Operational"
else:
    header_variant = "critical"
    status_label = "Backend Offline"

latency_detail = (
    f"Latency {st.session_state.websocket_latency} ms"
    if backend_status
    else "Awaiting backend connection"
)

render_enterprise_header(status_label, header_variant, latency_detail)

render_enterprise_overview(status_label, header_variant, latency_detail, backend_status)

# Initialize resources if connected
if backend_status:
    load_backend_resources()
    initialize_websocket()

# Sidebar with controls
with st.sidebar:
    st.markdown("## 🎮 Control Panel")
    
    # Model selector
    st.markdown("### 🤖 AI Model")
    if st.session_state.available_models:
        selected_model = st.selectbox(
            "Select Model",
            st.session_state.available_models,
            index=st.session_state.available_models.index(st.session_state.current_model) 
                if st.session_state.current_model in st.session_state.available_models else 0,
            key="model_selector"
        )
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            st.success(f"Switched to {selected_model}")
    else:
        st.info("No models available. Using default.")
    
    # Agent selector
    st.markdown("### 🚀 AI Agent")
    if st.session_state.available_agents:
        agent_names = [agent["name"] for agent in st.session_state.available_agents]
        agent_ids = [agent["id"] for agent in st.session_state.available_agents]
        
        current_idx = 0
        if st.session_state.current_agent in agent_ids:
            current_idx = agent_ids.index(st.session_state.current_agent)
        
        selected_agent_idx = st.selectbox(
            "Select Agent",
            range(len(agent_names)),
            format_func=lambda x: agent_names[x],
            index=current_idx,
            key="agent_selector"
        )
        
        selected_agent = agent_ids[selected_agent_idx]
        if selected_agent != st.session_state.current_agent:
            st.session_state.current_agent = selected_agent
            st.success(f"Switched to {agent_names[selected_agent_idx]}")
            
        # Show agent description
        st.caption(st.session_state.available_agents[selected_agent_idx].get("description", ""))
    else:
        st.info("Using default agent")
    
    # Voice settings
    st.markdown("### 🎤 Voice Settings")
    if not settings.ENABLE_VOICE_COMMANDS:
        st.session_state.voice_enabled = True
        st.info("Voice controls are disabled for this environment.")
    else:
        current_voice_state = st.session_state.get("voice_enabled", False)
        st.session_state.voice_enabled = st.toggle("Enable Voice", value=current_voice_state)
        
        if st.session_state.voice_enabled:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎙️ Start Listening", key="start_listening_sidebar", use_container_width=True):
                    assistant = get_voice_assistant()
                    if assistant and assistant.audio_available:
                        st.session_state.is_listening = True
                        assistant.start_listening()
                    else:
                        st.warning("Audio input is not available.")
            with col2:
                if st.button("🛑 Stop Listening", key="stop_listening_sidebar", use_container_width=True):
                    st.session_state.is_listening = False
                    assistant = st.session_state.get("voice_assistant")
                    if assistant:
                        assistant.stop_listening()
    
    # System status
    st.markdown("### 📊 System Status")
    
    # Backend connection status
    if backend_status:
        st.success("✅ Backend Connected")
        
        # WebSocket status with latency
        ws_status = "connected" if st.session_state.websocket_connected else "disconnected"
        ws_class = "ws-connected" if st.session_state.websocket_connected else "ws-disconnected"
        
        # Measure backend latency
        if st.session_state.websocket_connected:
            import time
            ping_start = time.time()
            try:
                health_check = st.session_state.backend_client.check_health_sync()
                latency_ms = int((time.time() - ping_start) * 1000)
                latency_indicator = f" ({latency_ms}ms)"
                if latency_ms < 100:
                    latency_color = "#4CAF50"  # Green
                elif latency_ms < 300:
                    latency_color = "#FF9800"  # Orange
                else:
                    latency_color = "#F44336"  # Red
            except:
                latency_indicator = ""
                latency_color = "#999"
        else:
            latency_indicator = ""
            latency_color = "#999"
        
        st.markdown(
            f'<div><span class="ws-status {ws_class}"></span>WebSocket: {ws_status}<span style="color: {latency_color}; margin-left: 8px; font-size: 0.85em;">{latency_indicator}</span></div>',
            unsafe_allow_html=True
        )
        
        # Get detailed health status
        health = st.session_state.backend_client.check_health_sync()
        if "services" in health:
            with st.expander("Service Status"):
                for service, status in health["services"].items():
                    if status:
                        st.markdown(f"✅ {service.title()}")
                    else:
                        st.markdown(f"❌ {service.title()}")
    else:
        st.error("❌ Backend Disconnected")
        if st.button("🔄 Retry Connection", key="retry_connection_sidebar"):
            if check_backend_connection():
                st.success("Reconnected!")
                st.rerun()
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat_sidebar", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💾 Export Chat", key="export_chat_sidebar", use_container_width=True) and st.session_state.messages:
            # Export chat history as text
            chat_export = "\n\n".join([
                f"[{msg.get('timestamp', 'N/A')}] {msg['role'].upper()}: {msg['content']}"
                for msg in st.session_state.messages
            ])
            st.download_button(
                label="Download",
                data=chat_export,
                key="download_chat_sidebar",
                file_name=f"jarvis_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🎤 Voice", "📊 Monitor", "🚀 Agents"])

with tab1:
    # Chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display connection warning if disconnected
    if not backend_status:
        st.warning("⚠️ Backend is disconnected. Responses will be limited.")
    
    render_enterprise_chat()

with tab2:
    # Voice command center
    st.markdown("### 🎙️ Voice Command Center")
    
    if not settings.ENABLE_VOICE_COMMANDS:
        st.info("Voice functionality is disabled for this deployment.")
    else:
        # Check voice service status
        voice_status = st.session_state.backend_client.check_voice_status_sync()
        if voice_status.get("status") == "ready":
            st.success("🎤 Voice service is ready")
        else:
            st.warning(f"⚠️ Voice service status: {voice_status.get('message', 'Unknown')}")
        
        # Simple audio recording section
        st.markdown("### 🎙️ Voice Recording")
        
        # File upload method for audio
        st.markdown("Upload an audio file to transcribe:")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg", "m4a"],
            key="audio_uploader"
        )
        
        if uploaded_file is not None:
            # Display the uploaded audio
            st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎯 Transcribe Audio", key="transcribe_audio_voice", use_container_width=True):
                    # Read the file bytes
                    audio_bytes = uploaded_file.read()
                    uploaded_file.seek(0)  # Reset file pointer
                    
                    # Process the audio
                    with st.spinner("Transcribing audio..."):
                        text = process_voice_input(audio_bytes)
                        
                        if text:
                            st.success(f"📝 Transcription: **{text}**")
                            st.session_state.last_transcription = text
                        else:
                            st.error("❌ Could not transcribe audio. Please try a different file.")
            
            with col2:
                if st.button("💬 Send to Chat", key="send_transcription_to_chat", use_container_width=True, disabled=not st.session_state.get('last_transcription')):
                    if st.session_state.get('last_transcription'):
                        process_chat_message(st.session_state.last_transcription)
                        st.rerun()
        
        # WebRTC recording if available
        if WEBRTC_AVAILABLE:
            st.markdown("### 🎤 Live Recording (Experimental)")
            st.info("Click Start to begin recording from your microphone")
            
            ctx = webrtc_streamer(
                key="voice-recorder",
                mode=WebRtcMode.SENDONLY,
                audio_processor_factory=AudioProcessor,
                media_stream_constraints={"audio": True, "video": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            if ctx.audio_processor:
                if st.button("📤 Process Recording", key="process_recording_webrtc", use_container_width=True):
                    audio_bytes = ctx.audio_processor.get_audio_bytes()
                    if audio_bytes:
                        # Process the audio
                        with st.spinner("Processing your voice..."):
                            text = process_voice_input(audio_bytes)
                            
                            if text:
                                st.success(f"📝 Transcription: **{text}**")
                                # Process as chat
                                with st.spinner("JARVIS is thinking..."):
                                    process_chat_message(text)
                                    st.rerun()
                            else:
                                st.error("❌ Could not transcribe audio.")
                    else:
                        st.warning("No audio recorded. Please speak into your microphone.")
        
        # Voice commands list
        with st.expander("📝 Available Voice Commands"):
            st.markdown("""
            - **"Hey JARVIS"** - Wake word to activate
            - **"What's the time?"** - Get current time
            - **"What's the weather?"** - Weather information
            - **"Tell me a joke"** - Hear a joke
            - **"Search for [query]"** - Web search
            - **"Analyze [topic]"** - Detailed analysis
            - **"Show system status"** - System metrics
            - **"Switch to [agent]"** - Change AI agent
            - **"Help"** - Show available commands
            """)
        
        # Voice settings and testing
        st.markdown("### ⚙️ Voice Testing")
        
        # Test voice pipeline
        if st.button("🧪 Test Voice Health", key="test_voice_health", use_container_width=True):
            with st.spinner("Checking voice service health..."):
                try:
                    health_status = st.session_state.backend_client.check_voice_status_sync()
                    if health_status.get("status") == "ready":
                        st.success("✅ Voice service is healthy!")
                        with st.expander("Health Details"):
                            if "details" in health_status:
                                st.json(health_status["details"])
                            else:
                                st.json(health_status)
                    elif health_status.get("status") == "degraded":
                        st.warning("⚠️ Voice service is degraded")
                        with st.expander("Health Details"):
                            st.json(health_status.get("details", health_status))
                    else:
                        st.error(f"Voice service error: {health_status.get('message', 'Unknown')}")
                except Exception as e:
                    st.error(f"Voice health check error: {e}")
        
        # Text to Speech Test
        st.markdown("#### 🔊 Text-to-Speech Test")
        tts_text = st.text_input("Enter text to synthesize:", "Hello, I am JARVIS, your AI assistant.")
        if st.button("🎵 Synthesize Speech", key="synthesize_speech_test", use_container_width=True):
            if tts_text:
                with st.spinner("Synthesizing speech..."):
                    success = synthesize_speech(tts_text)
                    if success:
                        st.success("✅ Speech synthesized successfully!")
                    else:
                        st.error("❌ Speech synthesis failed")

with tab3:
    # System monitoring dashboard
    st.markdown("### 📊 System Monitoring Dashboard")
    
    # Capture latest telemetry
    cpu_usage = float(SystemMonitor.get_cpu_usage())
    memory_usage = float(SystemMonitor.get_memory_usage())
    disk_usage = float(SystemMonitor.get_disk_usage())
    network_speed = float(SystemMonitor.get_network_speed())

    # Persist history for charts
    record_metric_history("cpu", cpu_usage)
    record_metric_history("memory", memory_usage)

    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "CPU Usage",
            f"{cpu_usage:.1f}%",
            delta=f"{cpu_usage-50:.1f}%" if cpu_usage else None,
        )
    with col2:
        st.metric(
            "Memory",
            f"{memory_usage:.1f}%",
            delta=f"{memory_usage-50:.1f}%" if memory_usage else None,
        )
    with col3:
        st.metric(
            "Disk",
            f"{disk_usage:.1f}%",
            delta=f"{disk_usage-50:.1f}%" if disk_usage else None,
        )
    with col4:
        st.metric("Network", f"{network_speed:.1f} MB/s")
    
    # Docker container status
    st.markdown("#### 🐳 Docker Containers")
    docker_support = SystemMonitor.get_docker_support_status()
    if not settings.SHOW_DOCKER_STATS:
        st.info("Docker statistics are disabled in the configuration.")
    elif not docker_support.get("available"):
        message = docker_support.get("error") or "Docker daemon is not reachable."
        st.warning(f"Docker stats unavailable: {message}")
    else:
        try:
            containers = SystemMonitor.get_docker_stats()
            
            if containers:
                container_data = []
                for container in containers:
                    container_data.append({
                        "Name": container["name"],
                        "Status": "🟢 Running" if container["status"] == "running" else "🔴 Stopped",
                        "CPU": f"{container.get('cpu', 0)}%",
                        "Memory": f"{container.get('memory', 0)} MB",
                        "Uptime": container.get("uptime", "N/A")
                    })
                
                st.dataframe(container_data, use_container_width=True)
            else:
                st.info("No container data available")
        except Exception as e:
            st.error(f"Failed to get container stats: {e}")
    
    # Real-time performance chart
    st.markdown("#### 📈 Real-time Performance")
    
    history = st.session_state.performance_history
    fig = go.Figure()
    traces_added = False

    cpu_history = history.get("cpu", [])[-60:]
    if cpu_history:
        fig.add_trace(
            go.Scatter(
                x=[point["timestamp"] for point in cpu_history],
                y=[point["value"] for point in cpu_history],
                mode="lines",
                name="CPU",
                line=dict(color="#00D4FF", width=2),
            )
        )
        traces_added = True

    memory_history = history.get("memory", [])[-60:]
    if memory_history:
        fig.add_trace(
            go.Scatter(
                x=[point["timestamp"] for point in memory_history],
                y=[point["value"] for point in memory_history],
                mode="lines",
                name="Memory",
                line=dict(color="#FF6B6B", width=2),
            )
        )
        traces_added = True

    if traces_added:
        fig.update_layout(
            template="plotly_dark",
            title="System Performance (Live)",
            xaxis_title="Timestamp",
            yaxis_title="Usage (%)",
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Collecting performance data... interact with the app to populate the chart.")
    
    # Refresh button
    if st.button("🔄 Refresh Metrics", key="refresh_metrics_monitoring"):
        st.rerun()

with tab4:
    # AI Agents Management
    st.markdown("### 🚀 AI Agents Orchestra")
    
    if st.session_state.available_agents:
        # Display agents in a grid
        agent_cols = st.columns(3)
        
        for i, agent in enumerate(st.session_state.available_agents):
            with agent_cols[i % 3]:
                # Agent card
                st.markdown(f"""
                <div class="agent-card">
                    <h4>{agent.get('name', 'Unknown Agent')}</h4>
                    <p>{agent.get('description', 'No description available')}</p>
                    <p>Status: {'🟢 Active' if agent.get('id') == st.session_state.current_agent else '⚪ Ready'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(
                    f"{'✓ Active' if agent.get('id') == st.session_state.current_agent else 'Activate'}",
                    key=f"activate_{agent.get('id')}",
                    disabled=agent.get('id') == st.session_state.current_agent
                ):
                    st.session_state.current_agent = agent.get('id')
                    st.success(f"{agent.get('name')} activated!")
                    st.rerun()
    else:
        st.info("No agents available. Please check backend connection.")
    
    # Task orchestration
    st.markdown("### 🎯 Multi-Agent Task Orchestration")
    
    task_description = st.text_area(
        "Describe your complex task:",
        height=100,
        placeholder="Example: Analyze this document, summarize key points, and generate a report with visualizations"
    )
    
    # Agent selection for task
    if st.session_state.available_agents:
        selected_agents = st.multiselect(
            "Select agents for this task:",
            [agent["name"] for agent in st.session_state.available_agents],
            default=[st.session_state.available_agents[0]["name"]] if st.session_state.available_agents else []
        )
    
    col1, col2 = st.columns(2)
    with col1:
        task_priority = st.select_slider(
            "Priority",
            options=["Low", "Medium", "High", "Critical"],
            value="Medium"
        )
    
    with col2:
        task_timeout = st.number_input(
            "Timeout (seconds)",
            min_value=10,
            max_value=600,
            value=60
        )
    
    if st.button("🚀 Execute Multi-Agent Task", key="execute_multi_agent_task", use_container_width=True):
        if task_description and backend_status:
            with st.spinner("Orchestrating agents..."):
                try:
                    # Execute task through agent orchestrator
                    result = st.session_state.agent_orchestrator.execute_task(
                        task_description,
                        agents=selected_agents if 'selected_agents' in locals() else None,
                        priority=task_priority,
                        timeout=task_timeout
                    )
                    
                    st.success("Task execution completed!")
                    
                    # Display results
                    with st.expander("Task Results", expanded=True):
                        if isinstance(result, dict):
                            for key, value in result.items():
                                st.write(f"**{key}:** {value}")
                        else:
                            st.write(result)
                            
                except Exception as e:
                    st.error(f"Task execution failed: {e}")
        elif not task_description:
            st.warning("Please describe the task first")
        else:
            st.error("Backend is not connected")

# Footer
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: #666;'>JARVIS v2.0 | "
    f"Powered by SutazAI Platform | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
)

# Auto-refresh for monitoring tab (optional)
# if tab3 and st.session_state.get("auto_refresh", False):
#     time.sleep(5)
#     st.rerun()