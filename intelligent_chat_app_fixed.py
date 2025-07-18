#!/usr/bin/env python3
"""
SutazAI Intelligent Chat Application - Fixed Version
Advanced logging, error handling, and session management
"""

import streamlit as st
import requests
import json
import time
import threading
from datetime import datetime
import traceback
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import logging

# Import our enhanced logging system
from enhanced_logging_system import (
    sutazai_logger, info, debug, warning, error, critical, log_exception,
    log_function_calls, log_api_calls, log_context, display_log_viewer, display_log_stats
)

# Configure Streamlit
st.set_page_config(
    page_title="SutazAI AGI/ASI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BACKEND_URL = "http://localhost:8000"
TIMEOUT_SETTINGS = {
    "connect_timeout": 10,
    "read_timeout": 60,  # Increased from 30 to 60 seconds
    "total_timeout": 90   # Added total timeout
}

class SessionManager:
    """Manages Streamlit session state with proper initialization"""
    
    @staticmethod
    def initialize_session():
        """Initialize all session state variables safely"""
        
        with log_context("Initializing session state", category="ui"):
            defaults = {
                "messages": [],
                "chat_input": "",
                "current_model": "llama3.2:1b",
                "system_status": {},
                "last_update": time.time(),
                "logging_enabled": True,
                "show_logs": False,
                "realtime_stt_enabled": False,
                "realtime_stt": None,
                "recording_active": False,
                "api_errors": [],
                "performance_metrics": {},
                "debug_mode": False,
                "total_api_calls": 0,
                "total_tokens": 0,
                "auto_refresh_performance": True,
                "performance_refresh_rate": 2,
                "last_performance_update": 0,
                "last_performance_cache_update": 0,
                "performance_cache": {},
                "performance_cache_status": "none",
                "needs_performance_refresh": False
            }
            
            for key, default_value in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value
                    debug(f"Initialized session state: {key} = {default_value}", category="ui")

class APIClient:
    """Robust API client with enhanced error handling and logging"""
    
    @staticmethod
    @log_api_calls()
    def make_request(endpoint: str, method: str = "GET", data: Dict = None, timeout: int = None) -> Dict[str, Any]:
        """Make API request with comprehensive error handling"""
        
        url = f"{BACKEND_URL}{endpoint}"
        timeout = timeout or TIMEOUT_SETTINGS["read_timeout"]
        
        try:
            debug(f"Making {method} request to {url}", category="api", endpoint=endpoint)
            
            # Track API calls
            if hasattr(st, 'session_state'):
                st.session_state.total_api_calls = st.session_state.get('total_api_calls', 0) + 1
            
            if method.upper() == "POST":
                response = requests.post(
                    url, 
                    json=data, 
                    timeout=timeout,
                    headers={"Content-Type": "application/json"}
                )
            else:
                response = requests.get(url, timeout=timeout)
            
            # Log response details
            debug(
                f"API response: {response.status_code} in {response.elapsed.total_seconds():.3f}s",
                category="api",
                status_code=response.status_code,
                response_time=response.elapsed.total_seconds()
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                error(error_msg, category="api", status_code=response.status_code)
                return {"success": False, "error": error_msg, "status_code": response.status_code}
                
        except requests.exceptions.Timeout as e:
            error_msg = f"Request timeout after {timeout}s for {endpoint}"
            error(error_msg, category="api", timeout=timeout)
            return {"success": False, "error": error_msg, "timeout": True}
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error to {endpoint}: {str(e)}"
            error(error_msg, category="api")
            return {"success": False, "error": error_msg, "connection_error": True}
            
        except Exception as e:
            log_exception(e, context=f"API request to {endpoint}", category="api")
            return {"success": False, "error": str(e), "exception": True}
    
    @staticmethod
    @log_api_calls()
    def get_system_status() -> Dict[str, Any]:
        """Get comprehensive system status"""
        return APIClient.make_request("/health", timeout=10)
    
    @staticmethod
    @log_api_calls()
    def get_available_models() -> Dict[str, Any]:
        """Get available AI models"""
        return APIClient.make_request("/api/models", timeout=15)
    
    @staticmethod
    @log_api_calls()
    def send_chat_message(message: str, model: str = "llama3.2:1b") -> Dict[str, Any]:
        """Send chat message with extended timeout"""
        data = {
            "message": message,
            "model": model,
            "stream": False,
            "temperature": 0.7
        }
        return APIClient.make_request("/api/chat", method="POST", data=data, timeout=90)  # 90 second timeout for chat

class UIComponents:
    """Reusable UI components with logging"""
    
    @staticmethod
    @log_function_calls(category="ui")
    def display_system_status():
        """Display system status with real-time updates"""
        
        st.subheader("🔧 System Status")
        
        # Create status container
        status_container = st.container()
        
        with status_container:
            col1, col2, col3 = st.columns(3)
            
            # Get system status
            status_result = APIClient.get_system_status()
            
            if status_result["success"]:
                with col1:
                    st.success("✅ Backend Online")
                    st.metric("Response Time", f"{status_result.get('response_time', 0):.3f}s")
                
                with col2:
                    st.info("🧠 AI Models Active")
                    models_result = APIClient.get_available_models()
                    model_count = len(models_result.get("data", {}).get("models", [])) if models_result["success"] else 0
                    st.metric("Available Models", model_count)
                
                with col3:
                    st.success("📊 Monitoring Active")
                    st.metric("Session Logs", len(sutazai_logger.log_history))
                    
                # Store status in session
                st.session_state.system_status = status_result["data"]
                
            else:
                with col1:
                    st.error("❌ Backend Offline")
                    st.write(f"Error: {status_result.get('error', 'Unknown error')}")
                
                with col2:
                    st.warning("⚠️ Limited Functionality")
                
                with col3:
                    st.info("📊 Monitoring Only")
                
                # Store error in session
                st.session_state.api_errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "error": status_result.get("error", "Unknown error"),
                    "endpoint": "/health"
                })
    
    @staticmethod
    @log_function_calls(category="ui")
    def display_chat_interface():
        """Display chat interface with enhanced error handling"""
        
        st.subheader("💬 AI Chat Interface")
        
        # Model selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Chat input with Enter key support
            user_input = st.chat_input(
                placeholder="Type your message and press Enter...",
                key="chat_input_widget"
            )
        
        with col2:
            # Model selection
            available_models = ["llama3.2:1b", "deepseek-r1:8b", "qwen3:8b"]
            selected_model = st.selectbox(
                "Model:",
                available_models,
                index=available_models.index(st.session_state.current_model)
                if st.session_state.current_model in available_models else 0
            )
            st.session_state.current_model = selected_model
            
            # Send button
            send_button = st.button("🚀 Send", type="primary", use_container_width=True)
        
        # Handle message sending (both button click and Enter key)
        if (send_button or user_input) and user_input and user_input.strip():
            UIComponents._process_chat_message(user_input.strip(), selected_model)
        
        # Display conversation history
        UIComponents._display_conversation_history()
    
    @staticmethod
    def _process_chat_message(message: str, model: str):
        """Process chat message with comprehensive logging"""
        
        with log_context(f"Processing chat message with {model}", category="ui"):
            
            # Add user message to history
            user_message = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "model": model
            }
            st.session_state.messages.append(user_message)
            
            # Clear input
            st.session_state.chat_input = ""
            
            # Show processing indicator
            with st.spinner(f"🤖 {model} is thinking..."):
                
                # Send to API
                start_time = time.time()
                response = APIClient.send_chat_message(message, model)
                response_time = time.time() - start_time
                
                if response["success"]:
                    # Success - add response to history
                    ai_message = {
                        "role": "assistant",
                        "content": response["data"].get("response", "No response received"),
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "response_time": response_time
                    }
                    st.session_state.messages.append(ai_message)
                    
                    info(
                        f"Chat response received from {model} in {response_time:.3f}s",
                        category="ui",
                        model=model,
                        response_time=response_time
                    )
                    
                    # Track tokens
                    response_content = response["data"].get("response", "")
                    token_count = len(response_content.split())
                    st.session_state.total_tokens = st.session_state.get('total_tokens', 0) + token_count
                    
                    # Update performance metrics
                    if "chat_metrics" not in st.session_state.performance_metrics:
                        st.session_state.performance_metrics["chat_metrics"] = []
                    
                    st.session_state.performance_metrics["chat_metrics"].append({
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "response_time": response_time,
                        "success": True
                    })
                    
                else:
                    # Error - display error message
                    error_message = {
                        "role": "system",
                        "content": f"❌ Error: {response.get('error', 'Unknown error')}",
                        "timestamp": datetime.now().isoformat(),
                        "model": model,
                        "response_time": response_time,
                        "error": True
                    }
                    st.session_state.messages.append(error_message)
                    
                    error(
                        f"Chat error with {model}: {response.get('error', 'Unknown error')}",
                        category="ui",
                        model=model,
                        error_details=response
                    )
                    
                    # Store error for analysis
                    st.session_state.api_errors.append({
                        "timestamp": datetime.now().isoformat(),
                        "error": response.get("error", "Unknown error"),
                        "endpoint": "/api/chat",
                        "model": model,
                        "response_time": response_time
                    })
            
            # Trigger rerun to update UI
            st.rerun()
    
    @staticmethod
    def _display_conversation_history():
        """Display conversation history with formatting"""
        
        st.subheader("💭 Conversation History")
        
        if not st.session_state.messages:
            st.info("No messages yet. Start a conversation!")
            return
        
        # Display recent messages (last 10)
        recent_messages = st.session_state.messages[-10:]
        
        for message in recent_messages:
            
            timestamp = message.get("timestamp", "")
            model = message.get("model", "")
            response_time = message.get("response_time")
            
            if message["role"] == "user":
                st.markdown(f"**👤 You** ({timestamp[:19]})")
                st.markdown(f"> {message['content']}")
                
            elif message["role"] == "assistant":
                time_info = f" • {response_time:.2f}s" if response_time else ""
                st.markdown(f"**🤖 {model}** ({timestamp[:19]}{time_info})")
                st.markdown(message['content'])
                
            elif message["role"] == "system":
                st.error(message['content'])
            
            st.markdown("---")
    
    @staticmethod
    @log_function_calls(category="ui")
    def display_debug_panel():
        """Display debug information panel"""
        
        if not st.session_state.debug_mode:
            return
        
        st.subheader("🐛 Debug Information")
        
        tab1, tab2, tab3 = st.tabs(["Session State", "API Errors", "Performance"])
        
        with tab1:
            st.json(dict(st.session_state))
        
        with tab2:
            if st.session_state.api_errors:
                for error in st.session_state.api_errors[-5:]:  # Last 5 errors
                    st.error(f"{error['timestamp']}: {error['error']}")
            else:
                st.info("No API errors recorded")
        
        with tab3:
            if st.session_state.performance_metrics:
                st.json(st.session_state.performance_metrics)
            else:
                st.info("No performance metrics available")
    
    @staticmethod
    @log_function_calls(category="ui")
    def display_performance_metrics():
        """Display real-time performance metrics with live updates"""
        
        # Performance controls header
        col_title, col_controls = st.columns([3, 1])
        
        with col_title:
            st.subheader("📊 Real-Time Performance Metrics")
        
        with col_controls:
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 Live", value=st.session_state.get("auto_refresh_performance", True), key="auto_refresh_toggle")
            st.session_state.auto_refresh_performance = auto_refresh
        
        # Initialize placeholders if not exist
        if "performance_placeholder" not in st.session_state:
            st.session_state.performance_placeholder = st.empty()
        
        # Update performance data when needed
        current_time = time.time()
        refresh_rate = st.session_state.get("performance_refresh_rate", 2)
        last_update = st.session_state.get("last_performance_update", 0)
        
        # Check if we need to update
        should_update = (current_time - last_update) >= refresh_rate
        
        if should_update or not st.session_state.get("performance_cache"):
            UIComponents._update_performance_cache()
            st.session_state.last_performance_update = current_time
        
        # Always render the content (no conditional refreshing)
        UIComponents._render_performance_content()
        
        # Auto-refresh with controlled timing to minimize flickering
        if auto_refresh:
            # Only refresh when actually needed and with proper timing
            if should_update:
                # Shorter sleep for more responsive 2-second updates
                time.sleep(1)  # 1 second - good balance for 2-second refresh rate
                st.rerun()
    
    @staticmethod
    def _render_performance_content():
        """Render the actual performance metrics content from cache"""
        try:
            # Use cached data if available, otherwise get fresh data
            cache_status = st.session_state.get("performance_cache_status", "none")
            
            if cache_status == "success" and "performance_cache" in st.session_state:
                # Use cached data
                summary = st.session_state.performance_cache
                summary_result = {"success": True, "data": summary}
            else:
                # Get fresh data if cache is not available
                summary_result = APIClient.make_request("/api/performance/summary", timeout=5)
                
                if not summary_result["success"]:
                    # Fall back to local performance collection
                    summary_result = UIComponents._collect_local_performance_data()
            
            if summary_result["success"]:
                summary = summary_result["data"]
                
                # Overall health status
                health_status = summary.get("overall_health", "unknown")
                health_color = {
                    "healthy": "🟢",
                    "warning": "🟡", 
                    "critical": "🔴",
                    "unknown": "⚪"
                }.get(health_status, "⚪")
                
                # Add live indicator and timestamp
                col_health, col_live = st.columns([3, 1])
                with col_health:
                    st.markdown(f"**System Health:** {health_color} {health_status.title()}")
                with col_live:
                    if st.session_state.get("auto_refresh_performance", False):
                        st.markdown("🔴 **LIVE**")
                    else:
                        st.markdown("⚪ **STATIC**")
                
                # Create metrics columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🖥️ System")
                    system_summary = summary.get("system_summary", {})
                    
                    # Get previous values for delta calculation
                    prev_metrics = st.session_state.get("prev_performance_metrics", {})
                    
                    # CPU with delta
                    cpu_current = system_summary.get('cpu_percent', 0)
                    cpu_prev = prev_metrics.get('cpu_percent', cpu_current)
                    cpu_delta = cpu_current - cpu_prev
                    st.metric("CPU Usage", f"{cpu_current:.1f}%", delta=f"{cpu_delta:+.1f}%" if abs(cpu_delta) > 0.1 else None)
                    
                    # Memory with delta
                    mem_current = system_summary.get('memory_percent', 0)
                    mem_prev = prev_metrics.get('memory_percent', mem_current)
                    mem_delta = mem_current - mem_prev
                    st.metric("Memory Usage", f"{mem_current:.1f}%", delta=f"{mem_delta:+.1f}%" if abs(mem_delta) > 0.1 else None)
                    
                    # Processes with delta
                    proc_current = system_summary.get('process_count', 0)
                    proc_prev = prev_metrics.get('process_count', proc_current)
                    proc_delta = proc_current - proc_prev
                    st.metric("Processes", proc_current, delta=f"{proc_delta:+d}" if proc_delta != 0 else None)
                    
                    # Store current values for next update
                    st.session_state.prev_performance_metrics = {
                        'cpu_percent': cpu_current,
                        'memory_percent': mem_current,
                        'process_count': proc_current
                    }
                
                with col2:
                    st.subheader("🌐 API")
                    api_summary = summary.get("api_summary", {})
                    st.metric("Total Requests", api_summary.get('total_requests', 0))
                    st.metric("Error Rate", f"{api_summary.get('error_rate', 0):.1%}")
                    st.metric("Avg Response", f"{api_summary.get('average_response_time', 0):.2f}s")
                    st.metric("Requests/Min", api_summary.get('requests_per_minute', 0))
                
                with col3:
                    st.subheader("🤖 Agents & Models")
                    model_summary = summary.get("model_summary", {})
                    agent_summary = summary.get("agent_summary", {})
                    st.metric("Active Models", model_summary.get('active_models', 0))
                    st.metric("Tokens Processed", model_summary.get('total_tokens_processed', 0))
                    st.metric("Active Agents", agent_summary.get('active_agents', 0))
                    st.metric("Tasks Completed", agent_summary.get('tasks_completed', 0))
                
                # Performance alerts - generate locally if API not available
                alerts_result = APIClient.make_request("/api/performance/alerts", timeout=5)
                if not alerts_result["success"]:
                    # Generate alerts locally
                    alerts_result = UIComponents._generate_local_alerts(summary)
                
                if alerts_result["success"]:
                    alerts_data = alerts_result["data"]
                    alerts = alerts_data.get("alerts", [])
                    
                    if alerts:
                        st.subheader("⚠️ Performance Alerts")
                        for alert in alerts:
                            alert_type = alert.get("type", "info")
                            metric = alert.get("metric", "Unknown")
                            value = alert.get("value", "")
                            message = alert.get("message", "")
                            
                            if alert_type == "critical":
                                st.error(f"🔴 **{metric}**: {value} - {message}")
                            elif alert_type == "warning":
                                st.warning(f"🟡 **{metric}**: {value} - {message}")
                            else:
                                st.info(f"🔵 **{metric}**: {value} - {message}")
                    else:
                        st.success("✅ No performance alerts")
                
                # Performance history chart (simplified)
                if st.checkbox("Show Performance History"):
                    history_result = APIClient.make_request("/api/performance/history?minutes=30", timeout=10)
                    if history_result["success"]:
                        history_data = history_result["data"].get("history", [])
                        
                        if history_data:
                            import pandas as pd
                            
                            # Extract CPU and memory data for chart
                            chart_data = []
                            for entry in history_data[-20:]:  # Last 20 data points
                                timestamp = entry.get("timestamp", "")
                                system = entry.get("system", {})
                                cpu = system.get("cpu", {}).get("percent", 0)
                                memory = system.get("memory", {}).get("percent", 0)
                                
                                chart_data.append({
                                    "Time": timestamp[-8:-3] if len(timestamp) > 8 else timestamp,  # Just time portion
                                    "CPU %": cpu,
                                    "Memory %": memory
                                })
                            
                            if chart_data:
                                df = pd.DataFrame(chart_data)
                                st.line_chart(df.set_index("Time"))
                        else:
                            st.info("No performance history available yet")
                
                # Last updated with refresh countdown
                last_updated = summary.get("last_updated", "Never")
                current_time = time.time()
                last_update_time = st.session_state.get("last_performance_update", current_time)
                seconds_since_update = current_time - last_update_time
                refresh_rate = st.session_state.get("performance_refresh_rate", 2)
                
                col_update, col_progress = st.columns([2, 1])
                with col_update:
                    # Format timestamp properly
                    if last_updated != "Never":
                        try:
                            if "T" in last_updated:
                                # ISO format timestamp
                                from datetime import datetime
                                dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                                formatted_time = dt.strftime("%H:%M:%S")
                            else:
                                # Assume it's already a time string
                                formatted_time = last_updated[-8:] if len(last_updated) > 8 else last_updated
                        except:
                            # Fallback for any timestamp format issues
                            formatted_time = datetime.now().strftime("%H:%M:%S")
                    else:
                        formatted_time = "Never"
                    
                    st.caption(f"Last updated: {formatted_time}")
                
                with col_progress:
                    if st.session_state.get("auto_refresh_performance", False):
                        progress = min(seconds_since_update / refresh_rate, 1.0)
                        st.progress(progress)
                        if progress >= 1.0:
                            st.caption("Refreshing...")
                        else:
                            next_refresh = refresh_rate - seconds_since_update
                            st.caption(f"Next: {next_refresh:.1f}s")
                
            else:
                # Show cached error or fresh error
                if cache_status == "error":
                    error_msg = st.session_state.get("performance_cache_error", "Unknown error")
                    st.error(f"Failed to get performance data: {error_msg}")
                    st.info("💡 Tip: Check if the SutazAI backend is running on port 8000")
                else:
                    st.error(f"Failed to get performance data: {summary_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Error loading performance metrics: {str(e)}")
            st.info("💡 Tip: Try refreshing the page or checking system status")
            
        # Performance control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh Now"):
                # Clear cache to force fresh data
                st.session_state.last_performance_cache_update = 0
                st.session_state.last_performance_update = 0
                # Use gentler refresh mechanism
                UIComponents._update_performance_cache()
        with col2:
            live_button_text = "⏸️ Pause Live" if st.session_state.get("auto_refresh_performance", False) else "▶️ Start Live"
            if st.button(live_button_text):
                st.session_state.auto_refresh_performance = not st.session_state.get("auto_refresh_performance", False)
                # Don't use st.rerun() for smoother experience
                if st.session_state.auto_refresh_performance:
                    st.success("Live monitoring started")
                else:
                    st.info("Live monitoring paused")
        with col3:
            if st.button("🗑️ Clear History"):
                if "prev_performance_metrics" in st.session_state:
                    del st.session_state.prev_performance_metrics
                if "performance_cache" in st.session_state:
                    del st.session_state.performance_cache
                st.success("Performance history cleared")
    
    @staticmethod
    def _collect_local_performance_data() -> Dict[str, Any]:
        """Collect performance data locally when API is not available"""
        try:
            import psutil
            import requests
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            process_count = len(psutil.pids())
            
            # Determine health status
            health_status = "healthy"
            if cpu_percent > 90 or memory.percent > 95:
                health_status = "critical"
            elif cpu_percent > 70 or memory.percent > 80:
                health_status = "warning"
            
            # Check backend
            try:
                backend_response = requests.get("http://localhost:8000/health", timeout=3)
                backend_healthy = backend_response.status_code == 200
                backend_response_time = backend_response.elapsed.total_seconds()
            except:
                backend_healthy = False
                backend_response_time = 0
            
            # Check Ollama
            try:
                ollama_response = requests.get("http://localhost:11434/api/version", timeout=3)
                ollama_healthy = ollama_response.status_code == 200
            except:
                ollama_healthy = False
            
            return {
                "success": True,
                "data": {
                    "overall_health": health_status,
                    "monitoring_active": True,
                    "last_updated": datetime.now().isoformat(),
                    "system_summary": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "disk_percent": (disk.used / disk.total) * 100,
                        "process_count": process_count
                    },
                    "api_summary": {
                        "total_requests": st.session_state.get('total_api_calls', 0),
                        "error_rate": 0.0,
                        "average_response_time": backend_response_time if backend_healthy else 0,
                        "requests_per_minute": 0
                    },
                    "model_summary": {
                        "active_models": 2 if ollama_healthy else 0,
                        "total_tokens_processed": st.session_state.get('total_tokens', 0)
                    },
                    "agent_summary": {
                        "total_agents": 0,
                        "active_agents": 0,
                        "tasks_completed": len(st.session_state.get('messages', [])),
                        "tasks_failed": 0
                    },
                    "services": {
                        "backend": "healthy" if backend_healthy else "offline",
                        "ollama": "healthy" if ollama_healthy else "offline"
                    }
                }
            }
        except ImportError:
            return {
                "success": False,
                "error": "psutil not available - install with: pip install psutil"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def _generate_local_alerts(summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance alerts locally"""
        alerts = []
        
        system_summary = summary.get("system_summary", {})
        cpu_percent = system_summary.get("cpu_percent", 0)
        memory_percent = system_summary.get("memory_percent", 0)
        disk_percent = system_summary.get("disk_percent", 0)
        
        # CPU alerts
        if cpu_percent > 90:
            alerts.append({"type": "critical", "metric": "CPU", "value": f"{cpu_percent:.1f}%", "message": "CPU usage critically high"})
        elif cpu_percent > 70:
            alerts.append({"type": "warning", "metric": "CPU", "value": f"{cpu_percent:.1f}%", "message": "CPU usage high"})
        
        # Memory alerts
        if memory_percent > 95:
            alerts.append({"type": "critical", "metric": "Memory", "value": f"{memory_percent:.1f}%", "message": "Memory usage critically high"})
        elif memory_percent > 80:
            alerts.append({"type": "warning", "metric": "Memory", "value": f"{memory_percent:.1f}%", "message": "Memory usage high"})
        
        # Disk alerts
        if disk_percent > 90:
            alerts.append({"type": "critical", "metric": "Disk", "value": f"{disk_percent:.1f}%", "message": "Disk usage critically high"})
        elif disk_percent > 75:
            alerts.append({"type": "warning", "metric": "Disk", "value": f"{disk_percent:.1f}%", "message": "Disk usage high"})
        
        # Service alerts
        services = summary.get("services", {})
        for service, status in services.items():
            if status != "healthy":
                alerts.append({"type": "warning", "metric": f"{service.title()} Service", "value": status, "message": f"{service.title()} service is {status}"})
        
        return {
            "success": True,
            "data": {
                "alerts": alerts,
                "alert_count": len(alerts),
                "critical_count": sum(1 for alert in alerts if alert["type"] == "critical"),
                "warning_count": sum(1 for alert in alerts if alert["type"] == "warning"),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def _update_performance_cache():
        """Update performance data cache to avoid constant API calls"""
        current_time = time.time()
        cache_duration = st.session_state.get("performance_refresh_rate", 2)
        last_update = st.session_state.get("last_performance_cache_update", 0)
        
        # Only update cache if enough time has passed
        if current_time - last_update >= cache_duration:
            try:
                # Try API first, fall back to local collection
                summary_result = APIClient.make_request("/api/performance/summary", timeout=5)
                
                if not summary_result["success"]:
                    # Fall back to local performance collection
                    summary_result = UIComponents._collect_local_performance_data()
                
                if summary_result["success"]:
                    st.session_state.performance_cache = summary_result["data"]
                    st.session_state.last_performance_cache_update = current_time
                    st.session_state.performance_cache_status = "success"
                else:
                    st.session_state.performance_cache_status = "error"
                    st.session_state.performance_cache_error = summary_result.get("error", "Unknown error")
                    
            except Exception as e:
                st.session_state.performance_cache_status = "error"
                st.session_state.performance_cache_error = str(e)
    
    @staticmethod
    def _schedule_background_refresh(refresh_rate: int):
        """Schedule background refresh using optimized session state mechanism"""
        current_time = time.time()
        last_refresh = st.session_state.get("last_performance_update", 0)
        
        # Check if it's time to refresh
        if current_time - last_refresh >= refresh_rate:
            st.session_state.last_performance_update = current_time
            
            # Use a more gentle refresh mechanism
            # Instead of st.rerun(), we update the timestamp to trigger a targeted refresh
            if st.session_state.get("auto_refresh_performance", False):
                # Mark that we need a refresh
                st.session_state.needs_performance_refresh = True
                
                # Use JavaScript to smoothly update without flickering
                UIComponents._add_smooth_refresh_script(refresh_rate)
    
    @staticmethod
    def _add_smooth_refresh_script(refresh_rate: int):
        """Add optimized JavaScript for smooth background refresh"""
        # Only add script if auto-refresh is enabled
        if not st.session_state.get("auto_refresh_performance", False):
            return
            
        # Create a non-flickering refresh mechanism
        refresh_script = f"""
        <script>
        // Smooth performance metrics refresh
        if (typeof sutazaiRefreshInterval !== 'undefined') {{
            clearInterval(sutazaiRefreshInterval);
        }}
        
        let refreshCounter = 0;
        const maxRefreshes = 100; // Prevent infinite refreshes
        
        sutazaiRefreshInterval = setInterval(function() {{
            refreshCounter++;
            
            // Stop after max refreshes to prevent infinite loops
            if (refreshCounter > maxRefreshes) {{
                clearInterval(sutazaiRefreshInterval);
                return;
            }}
            
            // Smooth update mechanism - update metrics containers with fade effect
            const metricElements = document.querySelectorAll('[data-testid="metric-container"]');
            
            if (metricElements.length > 0) {{
                // Add fade transition
                metricElements.forEach(function(element) {{
                    element.style.transition = 'opacity 0.3s ease-in-out';
                    element.style.opacity = '0.8';
                    
                    // Restore opacity after brief moment
                    setTimeout(function() {{
                        element.style.opacity = '1.0';
                    }}, 150);
                }});
                
                // Trigger a targeted refresh after a brief delay
                setTimeout(function() {{
                    // Use Streamlit's fragment refresh instead of full page reload
                    const refreshEvent = new CustomEvent('streamlit:componentReady');
                    document.dispatchEvent(refreshEvent);
                }}, 300);
            }}
        }}, {refresh_rate * 1000});
        
        // Auto-cleanup
        setTimeout(function() {{
            if (typeof sutazaiRefreshInterval !== 'undefined') {{
                clearInterval(sutazaiRefreshInterval);
            }}
        }}, 300000); // 5 minutes max
        
        // Cleanup on page navigation
        window.addEventListener('beforeunload', function() {{
            if (typeof sutazaiRefreshInterval !== 'undefined') {{
                clearInterval(sutazaiRefreshInterval);
            }}
        }});
        </script>
        """
        
        st.markdown(refresh_script, unsafe_allow_html=True)

@log_function_calls(category="ui")
def main():
    """Main application function"""
    
    info("Starting SutazAI Intelligent Chat Application", category="app")
    
    try:
        # Initialize session
        SessionManager.initialize_session()
        
        # Sidebar
        with st.sidebar:
            st.title("🤖 SutazAI Control Panel")
            
            # Monitoring controls
            st.subheader("📋 Monitoring Controls")
            st.session_state.show_logs = st.checkbox("Show Real-time Logs", value=st.session_state.show_logs)
            st.session_state.show_performance = st.checkbox("Show Performance Metrics", value=st.session_state.get("show_performance", False))
            st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
            
            # Performance refresh controls
            if st.session_state.get("show_performance", False):
                st.markdown("**🔄 Live Metrics Settings**")
                st.session_state.auto_refresh_performance = st.checkbox("Auto-refresh", value=st.session_state.get("auto_refresh_performance", True))
                if st.session_state.auto_refresh_performance:
                    st.session_state.performance_refresh_rate = st.selectbox(
                        "Refresh Rate", 
                        options=[1, 2, 3, 5, 10], 
                        index=1,  # Default to 2 seconds
                        format_func=lambda x: f"{x} seconds"
                    )
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            if st.button("🔄 Refresh System"):
                st.rerun()
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                info("Chat history cleared", category="ui")
                st.rerun()
            
            if st.button("📊 System Health Check"):
                status = APIClient.get_system_status()
                if status["success"]:
                    st.success("✅ System Healthy")
                else:
                    st.error(f"❌ System Issues: {status.get('error', 'Unknown')}")
        
        # Main content area
        st.title("🤖 SutazAI AGI/ASI Intelligent System")
        st.markdown("*Advanced AI-powered chat interface with comprehensive monitoring*")
        
        # System status
        UIComponents.display_system_status()
        
        # Chat interface
        UIComponents.display_chat_interface()
        
        # Debug panel
        UIComponents.display_debug_panel()
        
        # Performance metrics panel
        if st.session_state.get("show_performance", False):
            st.markdown("---")
            UIComponents.display_performance_metrics()
        
        # Logging panel
        if st.session_state.show_logs:
            st.markdown("---")
            display_log_viewer()
            display_log_stats()
        
        # Auto-refresh for logs every 30 seconds
        if st.session_state.show_logs:
            time.sleep(0.1)  # Small delay to prevent excessive refreshing
        
        info("UI rendered successfully", category="ui")
        
    except Exception as e:
        log_exception(e, context="Main application", category="error")
        st.error(f"Application Error: {str(e)}")
        st.error("Please check the logs for more details.")
        
        # Show error details in debug mode
        if st.session_state.get("debug_mode", False):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()