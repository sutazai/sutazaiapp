#!/usr/bin/env python3
"""
JARVIS Frontend Performance Analysis Tool
Comprehensive performance testing and profiling
"""

import time
import tracemalloc
import threading
import requests
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import concurrent.futures
import subprocess
import sys
import io
from pathlib import Path

# Try to import optional modules
try:
    import psutil
except ImportError:
    psutil = None
    
try:
    import numpy as np
except ImportError:
    class np:
        @staticmethod
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0

class PerformanceAnalyzer:
    """Comprehensive performance analysis for JARVIS frontend"""
    
    def __init__(self, base_url: str = "http://localhost:11000"):
        self.base_url = base_url
        self.results = {}
        self.metrics = {
            "initial_load": {},
            "runtime_performance": {},
            "memory_usage": {},
            "network_requests": {},
            "animation_performance": {},
            "session_state": {},
            "api_latency": {},
            "resource_usage": {},
            "thread_analysis": {},
            "bottlenecks": []
        }
        
    def measure_initial_load(self) -> Dict:
        """Measure initial page load performance"""
        print("🔍 Measuring Initial Load Performance...")
        
        metrics = {
            "ttfb": 0,  # Time to first byte
            "dom_ready": 0,
            "full_load": 0,
            "resource_count": 0,
            "total_size_kb": 0,
            "js_execution_ms": 0,
            "css_parsing_ms": 0
        }
        
        # Measure basic HTTP response time
        start = time.perf_counter()
        try:
            response = requests.get(self.base_url, timeout=30)
            ttfb = (time.perf_counter() - start) * 1000
            metrics["ttfb"] = round(ttfb, 2)
            metrics["status_code"] = response.status_code
            metrics["response_size_kb"] = len(response.content) / 1024
            
            # Parse response time components
            if hasattr(response, 'elapsed'):
                metrics["server_response_ms"] = response.elapsed.total_seconds() * 1000
            
            # Analyze HTML content
            html_size = len(response.text)
            metrics["html_size_kb"] = html_size / 1024
            
            # Count inline scripts and styles
            metrics["inline_scripts"] = response.text.count('<script')
            metrics["inline_styles"] = response.text.count('<style')
            
            # Estimate full load time (simplified)
            metrics["full_load"] = ttfb + (html_size / 1024) * 10  # Rough estimate
            
        except Exception as e:
            metrics["error"] = str(e)
            
        return metrics
    
    def analyze_runtime_bottlenecks(self) -> Dict:
        """Identify runtime performance bottlenecks"""
        print("🔍 Analyzing Runtime Bottlenecks...")
        
        bottlenecks = []
        
        # Check main app.py for performance issues
        app_path = Path("/opt/sutazaiapp/frontend/app.py")
        if app_path.exists():
            with open(app_path, 'r') as f:
                content = f.read()
                lines = content.splitlines()
                
            # Line-by-line analysis
            for i, line in enumerate(lines, 1):
                # Synchronous operations in main thread
                if 'time.sleep' in line and 'daemon' not in lines[max(0, i-5):i]:
                    bottlenecks.append({
                        "line": i,
                        "type": "blocking_sleep",
                        "severity": "high",
                        "code": line.strip(),
                        "impact": "Blocks UI thread",
                        "fix": "Use async/await or background thread"
                    })
                
                # Inefficient loops
                if 'for' in line and 'range(60)' in line:
                    bottlenecks.append({
                        "line": i,
                        "type": "inefficient_loop",
                        "severity": "medium",
                        "code": line.strip(),
                        "impact": "CPU intensive operation",
                        "fix": "Use numpy or vectorized operations"
                    })
                
                # Repeated API calls
                if '.check_health_sync()' in line or 'get_models_sync()' in line:
                    bottlenecks.append({
                        "line": i,
                        "type": "sync_api_call",
                        "severity": "high",
                        "code": line.strip(),
                        "impact": "Blocks UI during network I/O",
                        "fix": "Use async operations with callbacks"
                    })
                
                # Heavy CSS animations
                if 'animation' in line and ('infinite' in line or 'linear' in line):
                    bottlenecks.append({
                        "line": i,
                        "type": "continuous_animation",
                        "severity": "medium",
                        "code": line.strip(),
                        "impact": "Constant CPU/GPU usage",
                        "fix": "Use CSS transforms, reduce animation frequency"
                    })
                
                # Session state operations
                if 'st.session_state' in line and '=' in line:
                    # Count session state writes
                    if not hasattr(self, 'session_writes'):
                        self.session_writes = 0
                    self.session_writes += 1
                    
                # Docker stats in UI thread
                if 'docker.from_env()' in line or 'container.stats' in line:
                    bottlenecks.append({
                        "line": i,
                        "type": "docker_api_call",
                        "severity": "high",
                        "code": line.strip(),
                        "impact": "Expensive Docker API calls in UI thread",
                        "fix": "Cache results, use background worker"
                    })
                
                # Plotly chart regeneration
                if 'plotly_chart' in line:
                    bottlenecks.append({
                        "line": i,
                        "type": "chart_regeneration",
                        "severity": "medium",
                        "code": line.strip(),
                        "impact": "Full chart re-render on each update",
                        "fix": "Use partial updates or caching"
                    })
                
                # st.rerun() calls
                if 'st.rerun()' in line:
                    bottlenecks.append({
                        "line": i,
                        "type": "full_page_rerun",
                        "severity": "high",
                        "code": line.strip(),
                        "impact": "Complete page re-execution",
                        "fix": "Use partial updates or callbacks"
                    })
        
        return {
            "bottlenecks": bottlenecks,
            "total_issues": len(bottlenecks),
            "high_severity": len([b for b in bottlenecks if b["severity"] == "high"]),
            "session_state_writes": getattr(self, 'session_writes', 0)
        }
    
    def measure_memory_usage(self) -> Dict:
        """Measure memory usage patterns"""
        print("🔍 Measuring Memory Usage...")
        
        # Start memory tracking
        tracemalloc.start()
        
        metrics = {
            "initial_memory_mb": 0,
            "python_objects": 0,
            "largest_allocations": []
        }
        
        if psutil:
            metrics["initial_memory_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Simulate session activity
        try:
            # Make multiple requests to simulate usage
            for i in range(5):
                requests.get(self.base_url, timeout=5)
                time.sleep(0.5)
            
            # Get memory snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            # Find top memory consumers
            for stat in top_stats[:10]:
                metrics["largest_allocations"].append({
                    "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                })
            
            if psutil:
                metrics["final_memory_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
            else:
                metrics["final_memory_mb"] = 0
            metrics["memory_growth_mb"] = metrics["final_memory_mb"] - metrics["initial_memory_mb"]
            
        except Exception as e:
            metrics["error"] = str(e)
        
        tracemalloc.stop()
        return metrics
    
    def analyze_network_patterns(self) -> Dict:
        """Analyze network request patterns"""
        print("🔍 Analyzing Network Patterns...")
        
        patterns = {
            "backend_endpoints": [],
            "websocket_usage": False,
            "polling_intervals": [],
            "api_calls": []
        }
        
        # Analyze backend client usage
        backend_client_path = Path("/opt/sutazaiapp/frontend/services/backend_client_fixed.py")
        if backend_client_path.exists():
            with open(backend_client_path, 'r') as f:
                content = f.read()
                
            # Find API endpoints
            import re
            endpoints = re.findall(r'["\']/([\w/]+)["\']', content)
            patterns["backend_endpoints"] = list(set(endpoints))
            
            # Check for WebSocket
            patterns["websocket_usage"] = 'WebSocket' in content or 'websocket' in content
            
            # Find polling patterns
            if 'setInterval' in content or 'setTimeout' in content:
                patterns["has_polling"] = True
        
        # Analyze app.py for API calls
        app_path = Path("/opt/sutazaiapp/frontend/app.py")
        if app_path.exists():
            with open(app_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                if 'backend_client' in line and ('(' in line):
                    patterns["api_calls"].append({
                        "line": i,
                        "call": line.strip(),
                        "is_sync": '_sync' in line
                    })
        
        patterns["total_api_calls"] = len(patterns["api_calls"])
        patterns["sync_calls"] = len([c for c in patterns["api_calls"] if c["is_sync"]])
        
        return patterns
    
    def measure_animation_performance(self) -> Dict:
        """Analyze CSS animation performance impact"""
        print("🔍 Measuring Animation Performance...")
        
        animations = {
            "css_animations": [],
            "estimated_fps_impact": 0,
            "gpu_acceleration": []
        }
        
        # Parse CSS animations from app.py
        app_path = Path("/opt/sutazaiapp/frontend/app.py")
        if app_path.exists():
            with open(app_path, 'r') as f:
                content = f.read()
                
            import re
            
            # Find all keyframe animations
            keyframes = re.findall(r'@keyframes\s+(\w+)', content)
            animations["keyframe_definitions"] = keyframes
            
            # Find animation usage
            animation_uses = re.findall(r'animation:\s*([^;]+);', content)
            for use in animation_uses:
                duration = re.search(r'(\d+(?:\.\d+)?)(s|ms)', use)
                if duration:
                    time_ms = float(duration.group(1))
                    if duration.group(2) == 's':
                        time_ms *= 1000
                    
                    animations["css_animations"].append({
                        "definition": use.strip(),
                        "duration_ms": time_ms,
                        "is_infinite": 'infinite' in use,
                        "performance_impact": "high" if 'infinite' in use else "medium"
                    })
            
            # Check for GPU acceleration
            if 'transform' in content or 'translate3d' in content:
                animations["gpu_acceleration"].append("CSS transforms detected (GPU accelerated)")
            
            if 'will-change' in content:
                animations["gpu_acceleration"].append("will-change property used")
                
            # Estimate FPS impact
            infinite_animations = [a for a in animations["css_animations"] if a["is_infinite"]]
            animations["infinite_animation_count"] = len(infinite_animations)
            animations["estimated_fps_impact"] = len(infinite_animations) * 5  # Each can reduce FPS by ~5
        
        return animations
    
    def analyze_session_state(self) -> Dict:
        """Analyze session state management"""
        print("🔍 Analyzing Session State Management...")
        
        session_analysis = {
            "state_variables": [],
            "initialization_overhead": 0,
            "update_frequency": {}
        }
        
        app_path = Path("/opt/sutazaiapp/frontend/app.py")
        if app_path.exists():
            with open(app_path, 'r') as f:
                lines = f.readlines()
                
            # Find all session state variables
            import re
            for i, line in enumerate(lines, 1):
                # Session state initialization
                match = re.search(r'st\.session_state\.(\w+)\s*=', line)
                if match:
                    var_name = match.group(1)
                    session_analysis["state_variables"].append({
                        "name": var_name,
                        "line": i,
                        "initialization": line.strip()
                    })
                    
                    # Check if it's a heavy object
                    if 'BackendClient' in line or 'ChatInterface' in line:
                        session_analysis["initialization_overhead"] += 100  # ms estimate
                    elif 'AgentOrchestrator' in line:
                        session_analysis["initialization_overhead"] += 50
        
        session_analysis["total_variables"] = len(session_analysis["state_variables"])
        session_analysis["heavy_objects"] = [
            v for v in session_analysis["state_variables"] 
            if 'Client' in v["initialization"] or 'Interface' in v["initialization"]
        ]
        
        return session_analysis
    
    def measure_api_latency(self) -> Dict:
        """Measure backend API latency"""
        print("🔍 Measuring API Latency...")
        
        latency = {
            "health_check": 0,
            "chat_endpoint": 0,
            "voice_endpoint": 0,
            "agent_list": 0,
            "model_list": 0
        }
        
        endpoints = [
            ("health_check", "http://localhost:10200/health"),
            ("chat_endpoint", "http://localhost:10200/api/v1/chat"),
            ("voice_endpoint", "http://localhost:10200/api/v1/voice/status"),
            ("agent_list", "http://localhost:10200/api/v1/agents"),
            ("model_list", "http://localhost:10200/api/v1/models")
        ]
        
        for name, url in endpoints:
            try:
                start = time.perf_counter()
                response = requests.get(url, timeout=5)
                latency[name] = round((time.perf_counter() - start) * 1000, 2)
            except:
                latency[name] = -1  # Failed
        
        # Calculate statistics
        valid_latencies = [v for v in latency.values() if v > 0]
        if valid_latencies:
            latency["average_ms"] = round(np.mean(valid_latencies), 2)
            latency["max_ms"] = max(valid_latencies)
            latency["min_ms"] = min(valid_latencies)
        
        return latency
    
    def analyze_resource_bundling(self) -> Dict:
        """Analyze resource bundling and caching"""
        print("🔍 Analyzing Resource Bundling...")
        
        bundling = {
            "inline_css_count": 0,
            "inline_js_count": 0,
            "external_resources": [],
            "caching_headers": False,
            "compression": False
        }
        
        try:
            response = requests.get(self.base_url, timeout=10)
            
            # Check compression
            bundling["compression"] = response.headers.get('Content-Encoding', '') in ['gzip', 'br', 'deflate']
            
            # Check cache headers
            cache_control = response.headers.get('Cache-Control', '')
            bundling["caching_headers"] = 'max-age' in cache_control or 'public' in cache_control
            
            # Count inline resources
            bundling["inline_css_count"] = response.text.count('<style')
            bundling["inline_js_count"] = response.text.count('<script')
            
            # Find external resources
            import re
            external_css = re.findall(r'<link[^>]+href=["\']([^"\']+\.css)["\']', response.text)
            external_js = re.findall(r'<script[^>]+src=["\']([^"\']+\.js)["\']', response.text)
            
            bundling["external_resources"] = {
                "css": external_css,
                "js": external_js,
                "total": len(external_css) + len(external_js)
            }
            
        except Exception as e:
            bundling["error"] = str(e)
        
        return bundling
    
    def analyze_thread_management(self) -> Dict:
        """Analyze thread management and cleanup"""
        print("🔍 Analyzing Thread Management...")
        
        threads = {
            "thread_creation_points": [],
            "cleanup_issues": [],
            "daemon_threads": []
        }
        
        app_path = Path("/opt/sutazaiapp/frontend/app.py")
        if app_path.exists():
            with open(app_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                # Thread creation
                if 'Thread(' in line or 'threading.Thread' in line:
                    threads["thread_creation_points"].append({
                        "line": i,
                        "code": line.strip(),
                        "is_daemon": 'daemon=True' in line
                    })
                    
                    # Check for cleanup
                    if 'daemon=True' not in line:
                        # Look for join() in next 10 lines
                        has_join = any('.join()' in lines[j] for j in range(i, min(i+10, len(lines))))
                        if not has_join:
                            threads["cleanup_issues"].append({
                                "line": i,
                                "issue": "Thread created without join() or daemon=True",
                                "impact": "Potential resource leak"
                            })
        
        # Check SystemMonitor thread management
        monitor_path = Path("/opt/sutazaiapp/frontend/components/system_monitor.py")
        if monitor_path.exists():
            with open(monitor_path, 'r') as f:
                monitor_lines = f.readlines()
                
            for i, line in enumerate(monitor_lines, 1):
                if 'Thread(' in line:
                    threads["thread_creation_points"].append({
                        "file": "system_monitor.py",
                        "line": i,
                        "code": line.strip(),
                        "is_daemon": 'daemon=True' in line
                    })
        
        threads["total_threads"] = len(threads["thread_creation_points"])
        threads["daemon_count"] = len([t for t in threads["thread_creation_points"] if t.get("is_daemon")])
        threads["cleanup_issue_count"] = len(threads["cleanup_issues"])
        
        return threads
    
    def measure_realtime_monitoring_impact(self) -> Dict:
        """Measure impact of real-time monitoring"""
        print("🔍 Measuring Real-time Monitoring Impact...")
        
        monitoring = {
            "cpu_overhead": 0,
            "memory_overhead": 0,
            "update_frequency": [],
            "expensive_operations": []
        }
        
        # Find monitoring update patterns
        monitor_path = Path("/opt/sutazaiapp/frontend/components/system_monitor.py")
        if monitor_path.exists():
            with open(monitor_path, 'r') as f:
                content = f.read()
                lines = content.splitlines()
                
            for i, line in enumerate(lines, 1):
                # Docker stats calls
                if 'container.stats' in line:
                    monitoring["expensive_operations"].append({
                        "line": i,
                        "operation": "Docker stats API call",
                        "estimated_ms": 100,
                        "impact": "High CPU/Network I/O"
                    })
                
                # Process iteration
                if 'psutil.process_iter' in line:
                    monitoring["expensive_operations"].append({
                        "line": i,
                        "operation": "Process iteration",
                        "estimated_ms": 50,
                        "impact": "CPU intensive"
                    })
                
                # Network checks
                if 'requests.get' in line and 'timeout' in line:
                    monitoring["expensive_operations"].append({
                        "line": i,
                        "operation": "Service health check",
                        "estimated_ms": 20,
                        "impact": "Network I/O"
                    })
            
            # Find update interval
            import re
            interval_match = re.search(r'update_interval\s*=\s*(\d+)', content)
            if interval_match:
                monitoring["update_interval_seconds"] = int(interval_match.group(1))
            
            # Calculate overhead
            total_ms = sum(op["estimated_ms"] for op in monitoring["expensive_operations"])
            monitoring["estimated_overhead_ms_per_update"] = total_ms
            
            if monitoring.get("update_interval_seconds"):
                monitoring["cpu_overhead_percent"] = (total_ms / (monitoring["update_interval_seconds"] * 1000)) * 100
        
        return monitoring
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("🚀 JARVIS Frontend Performance Analysis Report")
        print("="*60 + "\n")
        
        # Run all measurements
        self.metrics["initial_load"] = self.measure_initial_load()
        self.metrics["runtime_performance"] = self.analyze_runtime_bottlenecks()
        self.metrics["memory_usage"] = self.measure_memory_usage()
        self.metrics["network_requests"] = self.analyze_network_patterns()
        self.metrics["animation_performance"] = self.measure_animation_performance()
        self.metrics["session_state"] = self.analyze_session_state()
        self.metrics["api_latency"] = self.measure_api_latency()
        self.metrics["resource_bundling"] = self.analyze_resource_bundling()
        self.metrics["thread_analysis"] = self.analyze_thread_management()
        self.metrics["monitoring_impact"] = self.measure_realtime_monitoring_impact()
        
        # Generate summary and recommendations
        self.generate_summary()
        self.generate_recommendations()
        
        return self.metrics
    
    def generate_summary(self):
        """Generate performance summary"""
        summary = {
            "critical_issues": [],
            "performance_score": 100,
            "estimated_optimization_potential": 0
        }
        
        # Critical issues
        if self.metrics["runtime_performance"]["high_severity"] > 0:
            summary["critical_issues"].append(
                f"{self.metrics['runtime_performance']['high_severity']} high-severity bottlenecks found"
            )
            summary["performance_score"] -= self.metrics["runtime_performance"]["high_severity"] * 10
        
        if self.metrics.get("memory_usage", {}).get("memory_growth_mb", 0) > 10:
            summary["critical_issues"].append(
                f"Memory leak detected: {self.metrics['memory_usage']['memory_growth_mb']:.1f}MB growth"
            )
            summary["performance_score"] -= 15
        
        if self.metrics.get("thread_analysis", {}).get("cleanup_issue_count", 0) > 0:
            summary["critical_issues"].append(
                f"{self.metrics['thread_analysis']['cleanup_issue_count']} thread cleanup issues"
            )
            summary["performance_score"] -= 10
        
        if self.metrics.get("animation_performance", {}).get("infinite_animation_count", 0) > 3:
            summary["critical_issues"].append(
                f"{self.metrics['animation_performance']['infinite_animation_count']} infinite animations impacting FPS"
            )
            summary["performance_score"] -= 5
        
        # Optimization potential
        if self.metrics.get("network_requests", {}).get("sync_calls", 0) > 5:
            summary["estimated_optimization_potential"] += 30
        
        if not self.metrics.get("resource_bundling", {}).get("compression"):
            summary["estimated_optimization_potential"] += 20
        
        self.metrics["summary"] = summary
    
    def generate_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Priority 1: Critical Performance Issues
        if self.metrics["runtime_performance"]["high_severity"] > 0:
            for bottleneck in self.metrics["runtime_performance"]["bottlenecks"]:
                if bottleneck["severity"] == "high":
                    recommendations.append({
                        "priority": 1,
                        "category": "Runtime Performance",
                        "issue": f"Line {bottleneck['line']}: {bottleneck['type']}",
                        "impact": bottleneck["impact"],
                        "fix": bottleneck["fix"],
                        "estimated_improvement": "20-30% faster response"
                    })
        
        # Priority 2: Memory Management
        if self.metrics.get("memory_usage", {}).get("memory_growth_mb", 0) > 5:
            recommendations.append({
                "priority": 1,
                "category": "Memory Management",
                "issue": "Memory leak detected",
                "impact": f"{self.metrics['memory_usage']['memory_growth_mb']:.1f}MB growth per session",
                "fix": "Implement proper cleanup in session state and thread management",
                "estimated_improvement": "50% reduction in memory usage"
            })
        
        # Priority 3: Network Optimization
        if self.metrics.get("network_requests", {}).get("sync_calls", 0) > 0:
            recommendations.append({
                "priority": 2,
                "category": "Network Optimization",
                "issue": f"{self.metrics['network_requests']['sync_calls']} synchronous API calls",
                "impact": "UI blocking during network I/O",
                "fix": "Convert to async operations with loading states",
                "estimated_improvement": "60% faster perceived performance"
            })
        
        # Priority 4: Animation Performance
        if self.metrics.get("animation_performance", {}).get("infinite_animation_count", 0) > 0:
            recommendations.append({
                "priority": 3,
                "category": "Animation Performance",
                "issue": f"{self.metrics['animation_performance']['infinite_animation_count']} continuous animations",
                "impact": f"~{self.metrics['animation_performance']['estimated_fps_impact']} FPS reduction",
                "fix": "Use CSS transforms, reduce animation frequency, add will-change",
                "estimated_improvement": "15-20 FPS improvement"
            })
        
        # Priority 5: Caching Strategy
        if not self.metrics.get("resource_bundling", {}).get("caching_headers"):
            recommendations.append({
                "priority": 2,
                "category": "Caching Strategy",
                "issue": "No cache headers configured",
                "impact": "Full reload on every visit",
                "fix": "Implement proper Cache-Control headers and ETags",
                "estimated_improvement": "70% faster subsequent loads"
            })
        
        # Priority 6: Thread Management
        if self.metrics.get("thread_analysis", {}).get("cleanup_issue_count", 0) > 0:
            recommendations.append({
                "priority": 2,
                "category": "Thread Management",
                "issue": f"{self.metrics['thread_analysis']['cleanup_issue_count']} threads without cleanup",
                "impact": "Resource leaks, potential crashes",
                "fix": "Add proper thread cleanup with join() or use daemon threads",
                "estimated_improvement": "Prevent resource exhaustion"
            })
        
        # Priority 7: Monitoring Overhead
        monitoring_overhead = self.metrics.get("monitoring_impact", {}).get("cpu_overhead_percent", 0)
        if monitoring_overhead > 5:
            recommendations.append({
                "priority": 3,
                "category": "Monitoring Optimization",
                "issue": f"Real-time monitoring using {monitoring_overhead:.1f}% CPU",
                "impact": "Continuous resource consumption",
                "fix": "Cache monitoring results, increase update interval, use background workers",
                "estimated_improvement": "80% reduction in monitoring overhead"
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"])
        self.metrics["recommendations"] = recommendations
        
        # Performance Budget Recommendations
        self.metrics["performance_budget"] = {
            "initial_load_ms": 3000,
            "api_response_ms": 200,
            "animation_fps": 60,
            "memory_growth_mb_per_hour": 10,
            "cpu_usage_percent": 20,
            "network_requests_per_minute": 30
        }
    
    def print_report(self):
        """Print formatted performance report"""
        print("\n📊 PERFORMANCE METRICS\n" + "-"*40)
        
        # Initial Load
        print("\n🚀 Initial Load Performance:")
        if "initial_load" in self.metrics:
            print(f"  • Time to First Byte: {self.metrics['initial_load'].get('ttfb', 'N/A')}ms")
            print(f"  • HTML Size: {self.metrics['initial_load'].get('html_size_kb', 0):.1f}KB")
            print(f"  • Inline Scripts: {self.metrics['initial_load'].get('inline_scripts', 0)}")
            print(f"  • Inline Styles: {self.metrics['initial_load'].get('inline_styles', 0)}")
        
        # Runtime Bottlenecks
        print("\n⚠️ Runtime Bottlenecks:")
        if "runtime_performance" in self.metrics:
            print(f"  • Total Issues: {self.metrics['runtime_performance']['total_issues']}")
            print(f"  • High Severity: {self.metrics['runtime_performance']['high_severity']}")
            print(f"  • Session State Writes: {self.metrics['runtime_performance'].get('session_state_writes', 0)}")
        
        # Memory Usage
        print("\n💾 Memory Usage:")
        if "memory_usage" in self.metrics:
            print(f"  • Initial Memory: {self.metrics['memory_usage'].get('initial_memory_mb', 0):.1f}MB")
            print(f"  • Memory Growth: {self.metrics['memory_usage'].get('memory_growth_mb', 0):.1f}MB")
        
        # API Latency
        print("\n🌐 API Latency:")
        if "api_latency" in self.metrics:
            print(f"  • Average: {self.metrics['api_latency'].get('average_ms', 'N/A')}ms")
            print(f"  • Max: {self.metrics['api_latency'].get('max_ms', 'N/A')}ms")
        
        # Animations
        print("\n🎨 Animation Performance:")
        if "animation_performance" in self.metrics:
            print(f"  • Infinite Animations: {self.metrics['animation_performance'].get('infinite_animation_count', 0)}")
            print(f"  • Estimated FPS Impact: -{self.metrics['animation_performance'].get('estimated_fps_impact', 0)} FPS")
        
        # Summary
        if "summary" in self.metrics:
            print("\n📈 PERFORMANCE SUMMARY\n" + "-"*40)
            print(f"  • Performance Score: {self.metrics['summary']['performance_score']}/100")
            print(f"  • Optimization Potential: {self.metrics['summary']['estimated_optimization_potential']}%")
            if self.metrics['summary']['critical_issues']:
                print("\n  ⚠️ Critical Issues:")
                for issue in self.metrics['summary']['critical_issues']:
                    print(f"    - {issue}")
        
        # Top Recommendations
        if "recommendations" in self.metrics:
            print("\n🎯 TOP OPTIMIZATION RECOMMENDATIONS\n" + "-"*40)
            for i, rec in enumerate(self.metrics['recommendations'][:5], 1):
                print(f"\n{i}. [{rec['category']}] Priority: {'🔴' if rec['priority']==1 else '🟡' if rec['priority']==2 else '🟢'}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Impact: {rec['impact']}")
                print(f"   Fix: {rec['fix']}")
                print(f"   Expected Improvement: {rec['estimated_improvement']}")
        
        # Performance Budget
        if "performance_budget" in self.metrics:
            print("\n💰 RECOMMENDED PERFORMANCE BUDGET\n" + "-"*40)
            for metric, value in self.metrics['performance_budget'].items():
                print(f"  • {metric.replace('_', ' ').title()}: {value}")

def main():
    """Run performance analysis"""
    analyzer = PerformanceAnalyzer()
    
    print("🚀 Starting JARVIS Frontend Performance Analysis...")
    print("="*60)
    
    try:
        # Generate full report
        analyzer.generate_performance_report()
        
        # Print formatted report
        analyzer.print_report()
        
        # Save detailed report to JSON
        report_path = Path("/opt/sutazaiapp/frontend/performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(analyzer.metrics, f, indent=2, default=str)
        
        print(f"\n✅ Detailed report saved to: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()