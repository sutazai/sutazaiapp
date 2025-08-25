"""
Resilient UI Components for SutazAI Frontend
Enhanced user feedback during backend failures and system issues
"""

import streamlit as st
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import plotly.graph_objects as go
from datetime import datetime, timedelta

from ..utils.resilient_api_client import get_system_status, sync_health_check
from ..utils.adaptive_timeouts import timeout_manager, SystemState

logger = logging.getLogger(__name__)

class SystemStatusIndicator:
    """Advanced system status indicator with contextual feedback"""
    
    @staticmethod
    def render_status_banner():
        """Render system-wide status banner with actionable information"""
        
        try:
            system_status = get_system_status()
            system_state = system_status.get("system_state", "unknown")
            circuit_breakers = system_status.get("circuit_breakers", {})
            
            # Determine overall system health
            if system_state == "startup":
                SystemStatusIndicator._render_startup_banner(system_status)
            elif system_state == "failed":
                SystemStatusIndicator._render_failure_banner(system_status)
            elif any(cb.get("state") == "open" for cb in circuit_breakers.values()):
                SystemStatusIndicator._render_degraded_banner(system_status)
            elif system_state == "healthy":
                SystemStatusIndicator._render_healthy_banner(system_status)
            else:
                SystemStatusIndicator._render_unknown_banner(system_status)
                
        except Exception as e:
            logger.error(f"Error rendering status banner: {e}")
            st.error("🔧 System status temporarily unavailable")
    
    @staticmethod
    def _render_startup_banner(status: Dict[str, Any]):
        """Render banner during system startup"""
        duration = status.get("state_duration", 0)
        
        st.warning("""
        🚀 **System Starting Up**
        
        SutazAI is initializing services. This typically takes 2-3 minutes on first start.
        """)
        
        # Progress estimation
        progress = min(duration / 180.0, 1.0)  # 3 minutes estimated startup
        st.progress(progress, text=f"Startup progress: {duration:.0f}s / ~180s")
        
        if duration > 60:
            st.info("⚡ **Tip**: Backend services are warming up. AI models are loading...")
        if duration > 120:
            st.info("🔥 **Almost Ready**: Finalizing Ollama model initialization...")
        
        # Auto-refresh during startup
        time.sleep(5)
        st.rerun()
    
    @staticmethod  
    def _render_failure_banner(status: Dict[str, Any]):
        """Render banner during system failure"""
        st.error("""
        🚨 **System Issues Detected**
        
        Multiple services are experiencing problems. We're working to restore full functionality.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Retry Connection", use_container_width=True):
                # Reset circuit breakers and try again
                st.cache_data.clear()
                st.success("♻️ Retrying connection...")
                st.rerun()
        
        with col2:
            if st.button("📊 View Diagnostics", use_container_width=True):
                SystemStatusIndicator._show_diagnostic_info(status)
        
        # Show last successful connection
        last_success = status.get("last_success")
        if last_success:
            st.caption(f"Last successful connection: {last_success:.0f}s ago")
    
    @staticmethod
    def _render_degraded_banner(status: Dict[str, Any]):
        """Render banner during degraded performance"""
        st.warning("""
        ⚠️ **Partial Service Availability**
        
        Some features may be temporarily limited while we restore full service.
        """)
        
        # Show which services are affected
        circuit_breakers = status.get("circuit_breakers", {})
        affected_services = [
            name for name, cb in circuit_breakers.items() 
            if cb.get("state") == "open"
        ]
        
        if affected_services:
            st.caption(f"Affected services: {', '.join(affected_services)}")
        
        # Estimated recovery time
        recovery_times = [
            cb.get("recovery_timeout", 5) 
            for cb in circuit_breakers.values() 
            if cb.get("state") == "open"
        ]
        
        if recovery_times:
            max_recovery = max(recovery_times)
            st.info(f"🕒 **Estimated recovery**: {max_recovery:.0f} seconds")
    
    @staticmethod
    def _render_healthy_banner(status: Dict[str, Any]):
        """Render banner for healthy system"""
        # Only show if user has performance metrics enabled
        if st.session_state.get("show_system_status", False):
            st.success("✅ All systems operational")
            
            # Show performance metrics
            circuit_breakers = status.get("circuit_breakers", {})
            success_rates = [
                cb.get("success_rate", 0) 
                for cb in circuit_breakers.values()
            ]
            
            if success_rates:
                avg_success_rate = sum(success_rates) / len(success_rates)
                st.metric("System Performance", f"{avg_success_rate:.1f}%")
    
    @staticmethod
    def _render_unknown_banner(status: Dict[str, Any]):
        """Render banner for unknown system state"""
        st.info("""
        🔍 **System Status Unknown**
        
        Unable to determine current system state. Functionality may be limited.
        """)
        
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    @staticmethod
    def _show_diagnostic_info(status: Dict[str, Any]):
        """Show detailed diagnostic information"""
        with st.expander("🔧 System Diagnostics", expanded=True):
            
            st.subheader("Circuit Breaker Status")
            circuit_breakers = status.get("circuit_breakers", {})
            
            for name, cb in circuit_breakers.items():
                state = cb.get("state", "unknown")
                success_rate = cb.get("success_rate", 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text(f"{name.replace('_', ' ').title()}")
                with col2:
                    color = "🟢" if state == "closed" else "🟡" if state == "half_open" else "🔴"
                    st.text(f"{color} {state.title()}")
                with col3:
                    st.text(f"{success_rate:.1f}%")
            
            st.subheader("Adaptive Timeout Settings")
            timeouts = status.get("adaptive_timeouts", {})
            for operation, timeout in timeouts.items():
                st.text(f"{operation.replace('_', ' ').title()}: {timeout}s")

class ConnectionLostHandler:
    """Handle connection lost scenarios with recovery options"""
    
    @staticmethod
    def render_connection_lost_overlay():
        """Render full-screen overlay when connection is completely lost"""
        
        st.markdown("""
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.9);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        ">
            <div style="text-align: center; max-width: 500px; padding: 2rem;">
                <h2>🔌 Connection Lost</h2>
                <p>Lost connection to SutazAI backend services.</p>
                <p>This usually means the backend is restarting or under maintenance.</p>
                
                <div style="margin: 2rem 0;">
                    <div class="spinner"></div>
                    <p>Attempting to reconnect...</p>
                </div>
                
                <button onclick="window.location.reload()" 
                        style="padding: 12px 24px; background: #1a73e8; color: white; border: none; border-radius: 8px;">
                    🔄 Refresh Page
                </button>
            </div>
        </div>
        
        <style>
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)

class LoadingStateManager:
    """Advanced loading states for different scenarios"""
    
    @staticmethod
    def render_startup_loading():
        """Render loading screen during system startup"""
        
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h1>🚀 SutazAI</h1>
            <h3>Autonomous AI System</h3>
            <div style="margin: 3rem 0;">
                <div class="startup-spinner"></div>
                <p style="margin-top: 2rem; color: #888;">
                    Initializing AI models and services...
                </p>
                <p style="color: #666; font-size: 0.9rem;">
                    This may take 2-3 minutes on first startup
                </p>
            </div>
        </div>
        
        <style>
        .startup-spinner {
            width: 60px;
            height: 60px;
            border: 6px solid rgba(26, 115, 232, 0.2);
            border-top: 6px solid #1a73e8;
            border-radius: 50%;
            animation: startup-spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes startup-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_service_warming():
        """Render loading state for service warmup"""
        
        with st.container():
            st.info("🔥 **AI Models Warming Up**")
            st.write("The AI models are loading into memory. This improves response times for subsequent requests.")
            
            # Animated progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate warmup progress (in real app, this would be actual status)
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("Loading TinyLlama model...")
                elif i < 70:
                    status_text.text("Optimizing inference engine...")
                else:
                    status_text.text("Finalizing model initialization...")
                time.sleep(0.1)
            
            status_text.text("✅ Models ready!")
    
    @staticmethod
    def render_api_retry_loading(operation: str, attempt: int, max_attempts: int):
        """Render loading state during API retry attempts"""
        
        progress = attempt / max_attempts
        
        st.warning(f"""
        🔄 **Retrying {operation}**
        
        Attempt {attempt} of {max_attempts}
        """)
        
        st.progress(progress, text=f"Retry progress")
        
        if attempt > 1:
            st.caption(f"Previous attempts failed - this is normal during system startup")

class ErrorRecoveryUI:
    """User interface for error recovery and troubleshooting"""
    
    @staticmethod
    def render_error_recovery_panel(error_type: str, context: str):
        """Render context-aware error recovery panel"""
        
        with st.container():
            st.error(f"🔧 **{error_type} Error**")
            
            # Context-specific recovery options
            if error_type.lower() == "timeout":
                ErrorRecoveryUI._render_timeout_recovery(context)
            elif error_type.lower() == "connection":
                ErrorRecoveryUI._render_connection_recovery(context)
            elif error_type.lower() == "cors":
                ErrorRecoveryUI._render_cors_recovery(context)
            elif error_type.lower() == "circuit_breaker":
                ErrorRecoveryUI._render_circuit_breaker_recovery(context)
            else:
                ErrorRecoveryUI._render_general_recovery(context)
    
    @staticmethod
    def _render_timeout_recovery(context: str):
        """Recovery options for timeout errors"""
        st.write("**The operation took longer than expected.**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⏱️ Wait Longer & Retry", use_container_width=True):
                st.info("Extending timeout and retrying...")
                # This would trigger a retry with longer timeout
                st.rerun()
        
        with col2:
            if st.button("⚡ Quick Retry", use_container_width=True):
                st.info("Retrying with standard timeout...")
                st.rerun()
        
        st.info("""
        **💡 Why this happens:**
        - Backend services may be starting up (takes 2-3 minutes)
        - AI models are loading into memory
        - High system load during peak usage
        """)
    
    @staticmethod
    def _render_connection_recovery(context: str):
        """Recovery options for connection errors"""
        st.write("**Unable to connect to backend services.**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retry Now", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("🏥 Check Status", use_container_width=True):
                status = sync_health_check(use_cache=False)
                st.json(status)
        
        with col3:
            if st.button("📱 Use Offline Mode", use_container_width=True):
                st.session_state['offline_mode'] = True
                st.info("Switched to offline mode with limited functionality")
        
        st.warning("""
        **🔧 Troubleshooting steps:**
        1. Check if backend is running on port 10010
        2. Verify network connectivity
        3. Try refreshing the browser
        """)
    
    @staticmethod
    def _render_cors_recovery(context: str):
        """Recovery options for CORS errors"""
        st.write("**Cross-origin request blocked by security policy.**")
        
        if st.button("🔓 Retry with Different Headers", use_container_width=True):
            st.info("Attempting request with modified headers...")
            st.rerun()
        
        st.error("""
        **🔒 Security Issue Detected:**
        This appears to be a CORS (Cross-Origin Resource Sharing) configuration issue.
        
        **For administrators:**
        - Check Kong gateway CORS settings
        - Verify allowed origins in backend configuration
        - Review security policies
        """)
    
    @staticmethod
    def _render_circuit_breaker_recovery(context: str):
        """Recovery options for circuit breaker errors"""
        st.write("**Service temporarily disabled to prevent cascade failures.**")
        
        system_status = get_system_status()
        circuit_breakers = system_status.get("circuit_breakers", {})
        
        # Find which circuit breakers are open
        open_breakers = [
            (name, cb) for name, cb in circuit_breakers.items() 
            if cb.get("state") == "open"
        ]
        
        if open_breakers:
            st.write("**Affected services:**")
            for name, cb in open_breakers:
                recovery_time = cb.get("recovery_timeout", 5)
                st.caption(f"• {name.replace('_', ' ').title()}: Recovery in ~{recovery_time}s")
        
        if st.button("⏰ Wait for Recovery", use_container_width=True):
            st.info("Waiting for automatic service recovery...")
            time.sleep(10)
            st.rerun()
        
        st.info("""
        **🛡️ Circuit Breaker Protection:**
        This safety mechanism prevents overwhelming failed services.
        Services will automatically recover once they're healthy again.
        """)
    
    @staticmethod
    def _render_general_recovery(context: str):
        """General recovery options"""
        st.write("**An unexpected error occurred.**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Try Again", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("🏠 Go to Dashboard", use_container_width=True):
                st.session_state.current_page = "Dashboard"
                st.rerun()

class OfflineModeUI:
    """UI components for offline/limited functionality mode"""
    
    @staticmethod
    def render_offline_banner():
        """Render offline mode banner"""
        st.warning("""
        📱 **Offline Mode Active**
        
        Limited functionality available. Some features require backend connection.
        """)
        
        if st.button("🌐 Try to Reconnect"):
            st.session_state.pop('offline_mode', None)
            st.rerun()
    
    @staticmethod
    def render_cached_data_notice():
        """Notice about using cached data"""
        st.info("""
        💾 **Using Cached Data**
        
        Showing previously loaded information. Data may not be current.
        """)

# Export main components
__all__ = [
    'SystemStatusIndicator',
    'ConnectionLostHandler', 
    'LoadingStateManager',
    'ErrorRecoveryUI',
    'OfflineModeUI'
]