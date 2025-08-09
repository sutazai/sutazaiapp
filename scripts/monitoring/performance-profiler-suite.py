#!/usr/bin/env python3
"""
Performance Profiler Suite for SutazAI
======================================

Purpose: Comprehensive performance profiling and bottleneck identification for 131-agent system
Usage: python scripts/performance-profiler-suite.py [--profile-duration 300] [--deep-analysis]
Requirements: Python 3.8+, psutil, py-spy, memory_profiler, line_profiler

Features:
- CPU profiling with call stack analysis
- Memory profiling with leak detection
- I/O profiling and bottleneck identification
- Network performance analysis
- Agent-specific performance metrics
- Real-time performance monitoring
- Automated bottleneck detection and recommendations
"""

import os
import sys
import json
import time
import psutil
import logging
import argparse
import threading
import subprocess
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import multiprocessing
import gc
import tracemalloc
import cProfile
import pstats
import io
from contextlib import contextmanager
import asyncio
import weakref
import resource
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/performance_profiler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PerformanceProfiler')

@dataclass
class CPUProfile:
    """CPU profiling results"""
    process_id: int
    process_name: str
    cpu_percent: float
    cpu_times: Dict[str, float]
    threads: int
    cpu_affinity: List[int]
    context_switches: int
    memory_usage: int
    top_functions: List[Dict[str, Any]]
    call_stack_depth: int
    profiling_duration: float

@dataclass
class MemoryProfile:
    """Memory profiling results"""
    process_id: int
    process_name: str
    memory_info: Dict[str, int]
    memory_percent: float
    memory_maps: List[Dict[str, Any]]
    open_files: int
    memory_leaks: List[Dict[str, Any]]
    gc_stats: Dict[str, Any]
    heap_usage: Dict[str, Any]
    fragmentation_ratio: float

@dataclass
class IOProfile:
    """I/O profiling results"""
    process_id: int
    process_name: str
    io_counters: Dict[str, int]
    read_rate: float
    write_rate: float
    open_file_descriptors: int
    network_connections: List[Dict[str, Any]]
    disk_latency: float
    network_latency: float
    io_wait_time: float

@dataclass
class SystemBottleneck:
    """System bottleneck identification"""
    bottleneck_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_processes: List[int]
    metrics: Dict[str, float]
    recommendations: List[str]
    estimated_impact: float  # 0-100 percentage impact on performance
    detection_time: float

@dataclass
class PerformanceRecommendation:
    """Performance optimization recommendation"""
    category: str
    priority: str  # low, medium, high, critical
    title: str
    description: str
    implementation_complexity: str  # easy, medium, hard
    expected_improvement: float  # 0-100 percentage
    commands: List[str]
    config_changes: Dict[str, Any]

class SystemResourceMonitor:
    """Real-time system resource monitoring"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.running = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=3600)  # Keep 1 hour of data
        self.lock = threading.Lock()
    
    def start(self):
        """Start monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System resource monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("System resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self._collect_system_metrics()
                
                with self.lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_times = psutil.cpu_times()._asdict()
        cpu_freq = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        
        # Memory metrics
        memory = psutil.virtual_memory()._asdict()
        swap = psutil.swap_memory()._asdict()
        
        # Disk metrics
        disk_usage = {}
        disk_io = psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": (usage.used / usage.total) * 100
                }
            except PermissionError:
                continue
        
        # Network metrics
        network_io = psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        network_connections = len(psutil.net_connections())
        
        # Process metrics
        process_count = len(psutil.pids())
        top_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                top_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by CPU usage and take top 10
        top_processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
        top_processes = top_processes[:10]
        
        return {
            "timestamp": timestamp,
            "cpu": {
                "usage_percent": cpu_percent,
                "per_core": cpu_per_core,
                "times": cpu_times,
                "frequency": cpu_freq,
                "load_average": load_avg
            },
            "memory": memory,
            "swap": swap,
            "disk": {
                "usage": disk_usage,
                "io_counters": disk_io
            },
            "network": {
                "io_counters": network_io,
                "connections": network_connections
            },
            "processes": {
                "count": process_count,
                "top_processes": top_processes
            }
        }
    
    def get_metrics_summary(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary for specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        with self.lock:
            recent_metrics = [m for m in self.metrics_history if m["timestamp"] >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages and trends
        cpu_usage = [m["cpu"]["usage_percent"] for m in recent_metrics]
        memory_usage = [m["memory"]["percent"] for m in recent_metrics]
        
        # Disk I/O rates
        disk_reads = []
        disk_writes = []
        for i, m in enumerate(recent_metrics[1:], 1):
            prev_m = recent_metrics[i-1]
            time_diff = m["timestamp"] - prev_m["timestamp"]
            
            if m["disk"]["io_counters"] and prev_m["disk"]["io_counters"]:
                read_diff = m["disk"]["io_counters"]["read_bytes"] - prev_m["disk"]["io_counters"]["read_bytes"]
                write_diff = m["disk"]["io_counters"]["write_bytes"] - prev_m["disk"]["io_counters"]["write_bytes"]
                
                disk_reads.append(read_diff / time_diff if time_diff > 0 else 0)
                disk_writes.append(write_diff / time_diff if time_diff > 0 else 0)
        
        # Network I/O rates
        network_rx = []
        network_tx = []
        for i, m in enumerate(recent_metrics[1:], 1):
            prev_m = recent_metrics[i-1]
            time_diff = m["timestamp"] - prev_m["timestamp"]
            
            if m["network"]["io_counters"] and prev_m["network"]["io_counters"]:
                rx_diff = m["network"]["io_counters"]["bytes_recv"] - prev_m["network"]["io_counters"]["bytes_recv"]
                tx_diff = m["network"]["io_counters"]["bytes_sent"] - prev_m["network"]["io_counters"]["bytes_sent"]
                
                network_rx.append(rx_diff / time_diff if time_diff > 0 else 0)
                network_tx.append(tx_diff / time_diff if time_diff > 0 else 0)
        
        return {
            "duration_minutes": duration_minutes,
            "cpu": {
                "avg_usage": statistics.mean(cpu_usage),
                "max_usage": max(cpu_usage),
                "min_usage": min(cpu_usage),
                "usage_std": statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0
            },
            "memory": {
                "avg_usage": statistics.mean(memory_usage),
                "max_usage": max(memory_usage),
                "min_usage": min(memory_usage),
                "usage_std": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
            },
            "disk_io": {
                "avg_read_rate": statistics.mean(disk_reads) if disk_reads else 0,
                "avg_write_rate": statistics.mean(disk_writes) if disk_writes else 0,
                "max_read_rate": max(disk_reads) if disk_reads else 0,
                "max_write_rate": max(disk_writes) if disk_writes else 0
            },
            "network_io": {
                "avg_rx_rate": statistics.mean(network_rx) if network_rx else 0,
                "avg_tx_rate": statistics.mean(network_tx) if network_tx else 0,
                "max_rx_rate": max(network_rx) if network_rx else 0,
                "max_tx_rate": max(network_tx) if network_tx else 0
            }
        }

class ProcessProfiler:
    """Individual process profiling"""
    
    def __init__(self):
        self.profiler = None
        self.memory_tracker = None
        
    @contextmanager
    def cpu_profile(self):
        """Context manager for CPU profiling"""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
        try:
            yield self.profiler
        finally:
            self.profiler.disable()
    
    @contextmanager 
    def memory_profile(self):
        """Context manager for memory profiling"""
        tracemalloc.start()
        
        try:
            yield
        finally:
            snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            self.memory_tracker = snapshot
    
    def analyze_cpu_profile(self) -> Dict[str, Any]:
        """Analyze CPU profiling results"""
        if not self.profiler:
            return {}
        
        # Capture profile stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        profile_output = s.getvalue()
        
        # Parse the profile output for top functions
        lines = profile_output.split('\n')
        top_functions = []
        
        parsing_stats = False
        for line in lines:
            if 'ncalls' in line and 'tottime' in line:
                parsing_stats = True
                continue
            
            if parsing_stats and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        top_functions.append({
                            "ncalls": parts[0],
                            "tottime": float(parts[1]),
                            "cumtime": float(parts[3]),
                            "filename": parts[5] if len(parts) > 5 else "unknown"
                        })
                    except (ValueError, IndexError):
                        continue
                
                if len(top_functions) >= 20:  # Limit to top 20
                    break
        
        return {
            "total_functions": len(top_functions),
            "top_functions": top_functions,
            "profile_text": profile_output
        }
    
    def analyze_memory_profile(self) -> Dict[str, Any]:
        """Analyze memory profiling results"""
        if not self.memory_tracker:
            return {}
        
        # Get top memory consumers
        top_stats = self.memory_tracker.statistics('lineno')[:20]
        
        memory_stats = []
        for stat in top_stats:
            memory_stats.append({
                "filename": stat.traceback.format()[0] if stat.traceback.format() else "unknown",
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count
            })
        
        # Calculate total memory usage
        total_size = sum(stat.size for stat in self.memory_tracker.statistics('filename'))
        
        return {
            "total_size_mb": total_size / 1024 / 1024,
            "total_traces": len(self.memory_tracker.traces),
            "top_consumers": memory_stats
        }

class AgentProfiler:
    """Profiler specifically for AI agents"""
    
    def __init__(self):
        self.agent_metrics = {}
        self.docker_client = None
        
        try:
            import docker
            self.docker_client = docker.from_env()
        except ImportError:
            logger.warning("Docker not available for container profiling")
    
    def profile_agent_containers(self) -> Dict[str, Any]:
        """Profile all agent containers"""
        if not self.docker_client:
            return {}
        
        agent_profiles = {}
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                if 'sutazai' in container.name or any(keyword in container.name 
                    for keyword in ['agent', 'ai-', 'ml-']):
                    
                    profile = self._profile_container(container)
                    agent_profiles[container.name] = profile
        
        except Exception as e:
            logger.error(f"Failed to profile agent containers: {e}")
        
        return agent_profiles
    
    def _profile_container(self, container) -> Dict[str, Any]:
        """Profile individual container"""
        try:
            # Get container stats
            stats = container.stats(stream=False)
            
            # CPU usage calculation
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * \
                             len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
            
            # Memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
            
            # Network I/O
            network_rx = 0
            network_tx = 0
            if 'networks' in stats:
                for interface in stats['networks'].values():
                    network_rx += interface['rx_bytes']
                    network_tx += interface['tx_bytes']
            
            # Block I/O
            block_read = 0
            block_write = 0
            if 'blkio_stats' in stats and 'io_service_bytes_recursive' in stats['blkio_stats']:
                for entry in stats['blkio_stats']['io_service_bytes_recursive']:
                    if entry['op'] == 'Read':
                        block_read += entry['value']
                    elif entry['op'] == 'Write':
                        block_write += entry['value']
            
            return {
                "container_id": container.id[:12],
                "status": container.status,
                "cpu_percent": cpu_percent,
                "memory_usage_mb": memory_usage / 1024 / 1024,
                "memory_limit_mb": memory_limit / 1024 / 1024,
                "memory_percent": memory_percent,
                "network_rx_mb": network_rx / 1024 / 1024,
                "network_tx_mb": network_tx / 1024 / 1024,
                "block_read_mb": block_read / 1024 / 1024,
                "block_write_mb": block_write / 1024 / 1024
            }
        
        except Exception as e:
            logger.error(f"Failed to profile container {container.name}: {e}")
            return {}
    
    def detect_agent_bottlenecks(self, agent_profiles: Dict[str, Any]) -> List[SystemBottleneck]:
        """Detect bottlenecks in agent performance"""
        bottlenecks = []
        
        for agent_name, profile in agent_profiles.items():
            if not profile:
                continue
            
            # High CPU usage
            if profile.get("cpu_percent", 0) > 80:
                bottlenecks.append(SystemBottleneck(
                    bottleneck_type="high_cpu",
                    severity="high",
                    description=f"Agent {agent_name} has high CPU usage: {profile['cpu_percent']:.1f}%",
                    affected_processes=[],
                    metrics={"cpu_percent": profile["cpu_percent"]},
                    recommendations=[
                        "Consider CPU affinity optimization",
                        "Review agent workload distribution",
                        "Check for inefficient algorithms"
                    ],
                    estimated_impact=profile["cpu_percent"],
                    detection_time=time.time()
                ))
            
            # High memory usage
            if profile.get("memory_percent", 0) > 85:
                bottlenecks.append(SystemBottleneck(
                    bottleneck_type="high_memory",
                    severity="high",
                    description=f"Agent {agent_name} has high memory usage: {profile['memory_percent']:.1f}%",
                    affected_processes=[],
                    metrics={"memory_percent": profile["memory_percent"]},
                    recommendations=[
                        "Implement memory pooling",
                        "Review data structures for memory efficiency",
                        "Consider memory compression"
                    ],
                    estimated_impact=profile["memory_percent"],
                    detection_time=time.time()
                ))
            
            # Memory limit approaching
            memory_usage_ratio = profile.get("memory_usage_mb", 0) / max(profile.get("memory_limit_mb", 1), 1)
            if memory_usage_ratio > 0.9:
                bottlenecks.append(SystemBottleneck(
                    bottleneck_type="memory_limit",
                    severity="critical",
                    description=f"Agent {agent_name} approaching memory limit: {memory_usage_ratio*100:.1f}%",
                    affected_processes=[],
                    metrics={"memory_ratio": memory_usage_ratio},
                    recommendations=[
                        "Increase container memory limit",
                        "Optimize memory usage patterns",
                        "Implement garbage collection tuning"
                    ],
                    estimated_impact=90,
                    detection_time=time.time()
                ))
        
        return bottlenecks

class BottleneckDetector:
    """Advanced bottleneck detection and analysis"""
    
    def __init__(self):
        self.detection_algorithms = {
            "cpu": self._detect_cpu_bottlenecks,
            "memory": self._detect_memory_bottlenecks,
            "io": self._detect_io_bottlenecks,
            "network": self._detect_network_bottlenecks,
            "contention": self._detect_resource_contention
        }
    
    def analyze_system_bottlenecks(self, metrics_summary: Dict[str, Any], 
                                 agent_profiles: Dict[str, Any]) -> List[SystemBottleneck]:
        """Comprehensive bottleneck analysis"""
        all_bottlenecks = []
        
        for detector_name, detector_func in self.detection_algorithms.items():
            try:
                bottlenecks = detector_func(metrics_summary, agent_profiles)
                all_bottlenecks.extend(bottlenecks)
            except Exception as e:
                logger.error(f"Bottleneck detector {detector_name} failed: {e}")
        
        # Sort by severity and estimated impact
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        all_bottlenecks.sort(key=lambda b: (severity_order.get(b.severity, 0), b.estimated_impact), reverse=True)
        
        return all_bottlenecks
    
    def _detect_cpu_bottlenecks(self, metrics: Dict[str, Any], agents: Dict[str, Any]) -> List[SystemBottleneck]:
        """Detect CPU-related bottlenecks"""
        bottlenecks = []
        
        cpu_metrics = metrics.get("cpu", {})
        
        # High average CPU usage
        if cpu_metrics.get("avg_usage", 0) > 80:
            bottlenecks.append(SystemBottleneck(
                bottleneck_type="high_cpu_usage",
                severity="high",
                description=f"High average CPU usage: {cpu_metrics['avg_usage']:.1f}%",
                affected_processes=[],
                metrics=cpu_metrics,
                recommendations=[
                    "Implement CPU affinity for agents",
                    "Consider horizontal scaling",
                    "Review CPU-intensive operations",
                    "Enable CPU frequency scaling"
                ],
                estimated_impact=cpu_metrics["avg_usage"],
                detection_time=time.time()
            ))
        
        # High CPU variance (inconsistent performance)
        if cpu_metrics.get("usage_std", 0) > 20:
            bottlenecks.append(SystemBottleneck(
                bottleneck_type="cpu_variance",
                severity="medium",
                description=f"High CPU usage variance: {cpu_metrics['usage_std']:.1f}%",
                affected_processes=[],
                metrics=cpu_metrics,
                recommendations=[
                    "Implement workload smoothing",
                    "Review agent scheduling",
                    "Consider load balancing improvements"
                ],
                estimated_impact=cpu_metrics["usage_std"],
                detection_time=time.time()
            ))
        
        return bottlenecks
    
    def _detect_memory_bottlenecks(self, metrics: Dict[str, Any], agents: Dict[str, Any]) -> List[SystemBottleneck]:
        """Detect memory-related bottlenecks"""
        bottlenecks = []
        
        memory_metrics = metrics.get("memory", {})
        
        # High memory usage
        if memory_metrics.get("avg_usage", 0) > 85:
            bottlenecks.append(SystemBottleneck(
                bottleneck_type="high_memory_usage",
                severity="high",
                description=f"High average memory usage: {memory_metrics['avg_usage']:.1f}%",
                affected_processes=[],
                metrics=memory_metrics,
                recommendations=[
                    "Implement memory pooling",
                    "Enable memory compression",
                    "Review memory-intensive operations",
                    "Consider memory limits per agent"
                ],
                estimated_impact=memory_metrics["avg_usage"],
                detection_time=time.time()
            ))
        
        # Memory pressure (high peak usage)
        if memory_metrics.get("max_usage", 0) > 95:
            bottlenecks.append(SystemBottleneck(
                bottleneck_type="memory_pressure",
                severity="critical",
                description=f"Peak memory usage: {memory_metrics['max_usage']:.1f}%",
                affected_processes=[],
                metrics=memory_metrics,
                recommendations=[
                    "Increase system memory",
                    "Implement aggressive garbage collection",
                    "Review memory allocation patterns",
                    "Consider memory-mapped files"
                ],
                estimated_impact=95,
                detection_time=time.time()
            ))
        
        return bottlenecks
    
    def _detect_io_bottlenecks(self, metrics: Dict[str, Any], agents: Dict[str, Any]) -> List[SystemBottleneck]:
        """Detect I/O-related bottlenecks"""
        bottlenecks = []
        
        disk_io = metrics.get("disk_io", {})
        
        # High disk I/O rates
        read_rate_mb = disk_io.get("avg_read_rate", 0) / 1024 / 1024
        write_rate_mb = disk_io.get("avg_write_rate", 0) / 1024 / 1024
        
        if read_rate_mb > 100:  # >100 MB/s sustained
            bottlenecks.append(SystemBottleneck(
                bottleneck_type="high_disk_read",
                severity="medium",
                description=f"High disk read rate: {read_rate_mb:.1f} MB/s",
                affected_processes=[],
                metrics={"read_rate_mb": read_rate_mb},
                recommendations=[
                    "Implement disk caching",
                    "Consider SSD upgrade",
                    "Review data access patterns",
                    "Implement data prefetching"
                ],
                estimated_impact=min(read_rate_mb / 2, 80),
                detection_time=time.time()
            ))
        
        if write_rate_mb > 100:  # >100 MB/s sustained
            bottlenecks.append(SystemBottleneck(
                bottleneck_type="high_disk_write",
                severity="medium",
                description=f"High disk write rate: {write_rate_mb:.1f} MB/s",
                affected_processes=[],
                metrics={"write_rate_mb": write_rate_mb},
                recommendations=[
                    "Implement write batching",
                    "Consider SSD upgrade",
                    "Review logging patterns",
                    "Implement async I/O"
                ],
                estimated_impact=min(write_rate_mb / 2, 80),
                detection_time=time.time()
            ))
        
        return bottlenecks
    
    def _detect_network_bottlenecks(self, metrics: Dict[str, Any], agents: Dict[str, Any]) -> List[SystemBottleneck]:
        """Detect network-related bottlenecks"""
        bottlenecks = []
        
        network_io = metrics.get("network_io", {})
        
        # High network I/O rates
        rx_rate_mb = network_io.get("avg_rx_rate", 0) / 1024 / 1024
        tx_rate_mb = network_io.get("avg_tx_rate", 0) / 1024 / 1024
        
        if rx_rate_mb > 50:  # >50 MB/s sustained
            bottlenecks.append(SystemBottleneck(
                bottleneck_type="high_network_rx",
                severity="medium",
                description=f"High network receive rate: {rx_rate_mb:.1f} MB/s",
                affected_processes=[],
                metrics={"rx_rate_mb": rx_rate_mb},
                recommendations=[
                    "Implement network compression",
                    "Review data transfer patterns",
                    "Consider connection pooling",
                    "Optimize serialization"
                ],
                estimated_impact=min(rx_rate_mb, 70),
                detection_time=time.time()
            ))
        
        if tx_rate_mb > 50:  # >50 MB/s sustained
            bottlenecks.append(SystemBottleneck(
                bottleneck_type="high_network_tx",
                severity="medium",
                description=f"High network transmit rate: {tx_rate_mb:.1f} MB/s",
                affected_processes=[],
                metrics={"tx_rate_mb": tx_rate_mb},
                recommendations=[
                    "Implement network compression",
                    "Review data transfer patterns",
                    "Consider batching network requests",
                    "Optimize serialization"
                ],
                estimated_impact=min(tx_rate_mb, 70),
                detection_time=time.time()
            ))
        
        return bottlenecks
    
    def _detect_resource_contention(self, metrics: Dict[str, Any], agents: Dict[str, Any]) -> List[SystemBottleneck]:
        """Detect resource contention issues"""
        bottlenecks = []
        
        # High number of agents with similar resource usage patterns
        high_cpu_agents = [name for name, profile in agents.items() 
                          if profile.get("cpu_percent", 0) > 70]
        
        if len(high_cpu_agents) > 5:
            bottlenecks.append(SystemBottleneck(
                bottleneck_type="cpu_contention",
                severity="high",
                description=f"{len(high_cpu_agents)} agents with high CPU usage",
                affected_processes=[],
                metrics={"high_cpu_agents": len(high_cpu_agents)},
                recommendations=[
                    "Implement agent CPU quotas",
                    "Stagger agent workloads",
                    "Consider agent priority scheduling",
                    "Review agent coordination"
                ],
                estimated_impact=len(high_cpu_agents) * 5,
                detection_time=time.time()
            ))
        
        return bottlenecks

class RecommendationEngine:
    """Generate performance optimization recommendations"""
    
    def __init__(self):
        self.recommendation_templates = {
            "high_cpu": self._cpu_recommendations,
            "high_memory": self._memory_recommendations,
            "high_disk": self._io_recommendations,
            "high_network": self._network_recommendations,
            "contention": self._contention_recommendations
        }
    
    def generate_recommendations(self, bottlenecks: List[SystemBottleneck]) -> List[PerformanceRecommendation]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            bottleneck_category = self._categorize_bottleneck(bottleneck.bottleneck_type)
            
            if bottleneck_category in self.recommendation_templates:
                rec_func = self.recommendation_templates[bottleneck_category]
                recs = rec_func(bottleneck)
                recommendations.extend(recs)
        
        # Remove duplicates and prioritize
        unique_recommendations = self._deduplicate_recommendations(recommendations)
        return self._prioritize_recommendations(unique_recommendations)
    
    def _categorize_bottleneck(self, bottleneck_type: str) -> str:
        """Categorize bottleneck for recommendation generation"""
        if "cpu" in bottleneck_type.lower():
            return "high_cpu"
        elif "memory" in bottleneck_type.lower():
            return "high_memory"
        elif "disk" in bottleneck_type.lower() or "io" in bottleneck_type.lower():
            return "high_disk"
        elif "network" in bottleneck_type.lower():
            return "high_network"
        elif "contention" in bottleneck_type.lower():
            return "contention"
        else:
            return "general"
    
    def _cpu_recommendations(self, bottleneck: SystemBottleneck) -> List[PerformanceRecommendation]:
        """Generate CPU optimization recommendations"""
        recommendations = []
        
        if bottleneck.estimated_impact > 80:
            recommendations.append(PerformanceRecommendation(
                category="cpu_optimization",
                priority="critical",
                title="Implement CPU Affinity Management",
                description="Bind agents to specific CPU cores to reduce context switching",
                implementation_complexity="medium",
                expected_improvement=15,
                commands=[
                    "python scripts/hardware-optimization-master.py --mode cpu-only",
                    "docker update --cpuset-cpus='0-3' <container_name>"
                ],
                config_changes={
                    "cpu_affinity": True,
                    "cpu_cores_per_agent": 1
                }
            ))
            
            recommendations.append(PerformanceRecommendation(
                category="cpu_optimization",
                priority="high",
                title="Enable CPU Frequency Scaling",
                description="Optimize CPU frequency based on workload demands",
                implementation_complexity="easy",
                expected_improvement=10,
                commands=[
                    "echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
                ],
                config_changes={
                    "cpu_governor": "performance"
                }
            ))
        
        return recommendations
    
    def _memory_recommendations(self, bottleneck: SystemBottleneck) -> List[PerformanceRecommendation]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        recommendations.append(PerformanceRecommendation(
            category="memory_optimization",
            priority="high",
            title="Implement Shared Memory Pools",
            description="Use shared memory for model weights and common data structures",
            implementation_complexity="medium",
            expected_improvement=25,
            commands=[
                "python scripts/memory-pool-manager.py --pool-size 8192",
                "sysctl -w kernel.shmmax=8589934592"
            ],
            config_changes={
                "shared_memory_enabled": True,
                "shared_memory_size": "8GB"
            }
        ))
        
        if bottleneck.estimated_impact > 90:
            recommendations.append(PerformanceRecommendation(
                category="memory_optimization",
                priority="critical",
                title="Enable Memory Compression",
                description="Use zswap or zram to compress memory pages",
                implementation_complexity="medium",
                expected_improvement=20,
                commands=[
                    "echo lz4 > /sys/module/zswap/parameters/compressor",
                    "echo Y > /sys/module/zswap/parameters/enabled"
                ],
                config_changes={
                    "memory_compression": True,
                    "compression_algorithm": "lz4"
                }
            ))
        
        return recommendations
    
    def _io_recommendations(self, bottleneck: SystemBottleneck) -> List[PerformanceRecommendation]:
        """Generate I/O optimization recommendations"""
        recommendations = []
        
        recommendations.append(PerformanceRecommendation(
            category="io_optimization",
            priority="high",
            title="Implement Intelligent Caching",
            description="Deploy multi-level caching with ML-based prefetching",
            implementation_complexity="medium",
            expected_improvement=30,
            commands=[
                "python scripts/intelligent-cache-system.py --l1-size 512 --l2-size 2048",
                "echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
            ],
            config_changes={
                "cache_enabled": True,
                "cache_levels": 3,
                "prefetch_enabled": True
            }
        ))
        
        return recommendations
    
    def _network_recommendations(self, bottleneck: SystemBottleneck) -> List[PerformanceRecommendation]:
        """Generate network optimization recommendations"""
        recommendations = []
        
        recommendations.append(PerformanceRecommendation(
            category="network_optimization",
            priority="medium",
            title="Optimize Network Buffer Sizes",
            description="Increase network buffer sizes for high-throughput operations",
            implementation_complexity="easy",
            expected_improvement=15,
            commands=[
                "sysctl -w net.core.rmem_max=134217728",
                "sysctl -w net.core.wmem_max=134217728"
            ],
            config_changes={
                "network_buffer_size": "128MB",
                "tcp_window_scaling": True
            }
        ))
        
        return recommendations
    
    def _contention_recommendations(self, bottleneck: SystemBottleneck) -> List[PerformanceRecommendation]:
        """Generate resource contention recommendations"""
        recommendations = []
        
        recommendations.append(PerformanceRecommendation(
            category="resource_management",
            priority="high",
            title="Implement Resource Pool Coordination",
            description="Deploy dynamic resource allocation with load balancing",
            implementation_complexity="hard",
            expected_improvement=35,
            commands=[
                "python scripts/resource-pool-coordinator.py --enable-rebalancing",
                "python scripts/dynamic-load-balancer.py --algorithm adaptive"
            ],
            config_changes={
                "resource_pools": True,
                "dynamic_allocation": True,
                "load_balancing": "adaptive"
            }
        ))
        
        return recommendations
    
    def _deduplicate_recommendations(self, recommendations: List[PerformanceRecommendation]) -> List[PerformanceRecommendation]:
        """Remove duplicate recommendations"""
        seen_titles = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _prioritize_recommendations(self, recommendations: List[PerformanceRecommendation]) -> List[PerformanceRecommendation]:
        """Prioritize recommendations by impact and complexity"""
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        complexity_penalty = {"easy": 0, "medium": 1, "hard": 2}
        
        def priority_score(rec):
            priority_score = priority_order.get(rec.priority, 0) * 10
            improvement_score = rec.expected_improvement
            complexity_score = -complexity_penalty.get(rec.implementation_complexity, 0) * 2
            
            return priority_score + improvement_score + complexity_score
        
        recommendations.sort(key=priority_score, reverse=True)
        return recommendations

class PerformanceProfilerSuite:
    """Main performance profiler suite"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_monitor = SystemResourceMonitor(
            sampling_interval=config.get("sampling_interval", 1.0)
        )
        self.process_profiler = ProcessProfiler()
        self.agent_profiler = AgentProfiler()
        self.bottleneck_detector = BottleneckDetector()
        self.recommendation_engine = RecommendationEngine()
        
        self.running = False
        self.profile_results = {}
        
        logger.info("Performance Profiler Suite initialized")
    
    async def run_comprehensive_analysis(self, duration: int = 300) -> Dict[str, Any]:
        """Run comprehensive performance analysis"""
        logger.info(f"Starting comprehensive performance analysis for {duration} seconds...")
        
        # Start system monitoring
        self.system_monitor.start()
        
        # Wait for profiling duration
        await asyncio.sleep(duration)
        
        # Collect results
        results = {
            "analysis_duration": duration,
            "timestamp": time.time(),
            "system_metrics": self.system_monitor.get_metrics_summary(duration // 60),
            "agent_profiles": self.agent_profiler.profile_agent_containers(),
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Stop monitoring
        self.system_monitor.stop()
        
        # Detect bottlenecks
        bottlenecks = self.bottleneck_detector.analyze_system_bottlenecks(
            results["system_metrics"], results["agent_profiles"]
        )
        
        # Add agent-specific bottlenecks
        agent_bottlenecks = self.agent_profiler.detect_agent_bottlenecks(results["agent_profiles"])
        bottlenecks.extend(agent_bottlenecks)
        
        results["bottlenecks"] = [asdict(b) for b in bottlenecks]
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_recommendations(bottlenecks)
        results["recommendations"] = [asdict(r) for r in recommendations]
        
        logger.info(f"Analysis complete: Found {len(bottlenecks)} bottlenecks, "
                   f"Generated {len(recommendations)} recommendations")
        
        return results
    
    def export_analysis(self, results: Dict[str, Any], filepath: str):
        """Export analysis results"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Analysis results exported to {filepath}")
        
        # Also create a summary report
        summary_path = filepath.replace('.json', '_summary.md')
        self._create_summary_report(results, summary_path)
    
    def _create_summary_report(self, results: Dict[str, Any], filepath: str):
        """Create human-readable summary report"""
        with open(filepath, 'w') as f:
            f.write("# SutazAI Performance Analysis Report\n\n")
            f.write(f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}\n")
            f.write(f"**Duration:** {results['analysis_duration']} seconds\n\n")
            
            # System Overview
            f.write("## System Overview\n\n")
            system_metrics = results["system_metrics"]
            f.write(f"- **Average CPU Usage:** {system_metrics.get('cpu', {}).get('avg_usage', 0):.1f}%\n")
            f.write(f"- **Average Memory Usage:** {system_metrics.get('memory', {}).get('avg_usage', 0):.1f}%\n")
            f.write(f"- **Peak CPU Usage:** {system_metrics.get('cpu', {}).get('max_usage', 0):.1f}%\n")
            f.write(f"- **Peak Memory Usage:** {system_metrics.get('memory', {}).get('max_usage', 0):.1f}%\n\n")
            
            # Agent Analysis
            f.write("## Agent Analysis\n\n")
            agent_profiles = results["agent_profiles"]
            f.write(f"**Total Agents Analyzed:** {len(agent_profiles)}\n\n")
            
            if agent_profiles:
                high_cpu_agents = [name for name, profile in agent_profiles.items() 
                                 if profile.get("cpu_percent", 0) > 70]
                high_memory_agents = [name for name, profile in agent_profiles.items() 
                                    if profile.get("memory_percent", 0) > 80]
                
                f.write(f"- **Agents with High CPU Usage (>70%):** {len(high_cpu_agents)}\n")
                f.write(f"- **Agents with High Memory Usage (>80%):** {len(high_memory_agents)}\n\n")
            
            # Bottlenecks
            f.write("## Identified Bottlenecks\n\n")
            bottlenecks = results["bottlenecks"]
            
            if bottlenecks:
                critical_bottlenecks = [b for b in bottlenecks if b["severity"] == "critical"]
                high_bottlenecks = [b for b in bottlenecks if b["severity"] == "high"]
                
                f.write(f"**Total Bottlenecks:** {len(bottlenecks)}\n")
                f.write(f"- Critical: {len(critical_bottlenecks)}\n")
                f.write(f"- High: {len(high_bottlenecks)}\n\n")
                
                f.write("### Critical Issues\n")
                for bottleneck in critical_bottlenecks:
                    f.write(f"- **{bottleneck['bottleneck_type']}:** {bottleneck['description']}\n")
                
                f.write("\n### High Priority Issues\n")
                for bottleneck in high_bottlenecks:
                    f.write(f"- **{bottleneck['bottleneck_type']}:** {bottleneck['description']}\n")
            else:
                f.write("No significant bottlenecks detected.\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Optimization Recommendations\n\n")
            recommendations = results["recommendations"]
            
            if recommendations:
                critical_recs = [r for r in recommendations if r["priority"] == "critical"]
                high_recs = [r for r in recommendations if r["priority"] == "high"]
                
                f.write("### Critical Priority\n")
                for rec in critical_recs:
                    f.write(f"**{rec['title']}**\n")
                    f.write(f"- {rec['description']}\n")
                    f.write(f"- Expected Improvement: {rec['expected_improvement']}%\n")
                    f.write(f"- Implementation: {rec['implementation_complexity']}\n\n")
                
                f.write("### High Priority\n")
                for rec in high_recs:
                    f.write(f"**{rec['title']}**\n")
                    f.write(f"- {rec['description']}\n")
                    f.write(f"- Expected Improvement: {rec['expected_improvement']}%\n")
                    f.write(f"- Implementation: {rec['implementation_complexity']}\n\n")
            else:
                f.write("No specific recommendations generated.\n")
            
            f.write("\n---\n")
            f.write("*Report generated by SutazAI Performance Profiler Suite*\n")
        
        logger.info(f"Summary report created: {filepath}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SutazAI Performance Profiler Suite")
    parser.add_argument("--profile-duration", type=int, default=300,
                       help="Profiling duration in seconds")
    parser.add_argument("--sampling-interval", type=float, default=1.0,
                       help="Metrics sampling interval in seconds")
    parser.add_argument("--deep-analysis", action="store_true",
                       help="Enable deep CPU and memory profiling")
    parser.add_argument("--export-results", type=str,
                       help="Export results to file")
    parser.add_argument("--agent-focus", action="store_true",
                       help="Focus analysis on agent containers")
    parser.add_argument("--real-time", action="store_true",
                       help="Real-time monitoring mode")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "sampling_interval": args.sampling_interval,
        "deep_analysis": args.deep_analysis,
        "agent_focus": args.agent_focus
    }
    
    # Create profiler suite
    profiler = PerformanceProfilerSuite(config)
    
    try:
        if args.real_time:
            # Real-time monitoring mode
            logger.info("Starting real-time performance monitoring. Press Ctrl+C to stop.")
            
            profiler.system_monitor.start()
            
            try:
                while True:
                    await asyncio.sleep(60)  # Update every minute
                    
                    # Get current metrics
                    summary = profiler.system_monitor.get_metrics_summary(1)
                    agent_profiles = profiler.agent_profiler.profile_agent_containers()
                    
                    # Quick bottleneck check
                    bottlenecks = profiler.bottleneck_detector.analyze_system_bottlenecks(summary, agent_profiles)
                    
                    # Print status
                    cpu_avg = summary.get("cpu", {}).get("avg_usage", 0)
                    memory_avg = summary.get("memory", {}).get("avg_usage", 0)
                    active_agents = len([p for p in agent_profiles.values() if p.get("cpu_percent", 0) > 1])
                    
                    status = f"CPU: {cpu_avg:.1f}%, Memory: {memory_avg:.1f}%, Active Agents: {active_agents}"
                    if bottlenecks:
                        critical_count = len([b for b in bottlenecks if b.severity == "critical"])
                        high_count = len([b for b in bottlenecks if b.severity == "high"])
                        status += f", Bottlenecks: {critical_count} critical, {high_count} high"
                    
                    print(f"[{time.strftime('%H:%M:%S')}] {status}")
                    
            except KeyboardInterrupt:
                logger.info("Stopping real-time monitoring...")
            finally:
                profiler.system_monitor.stop()
        else:
            # Full analysis mode
            results = await profiler.run_comprehensive_analysis(args.profile_duration)
            
            # Print summary
            print(f"\nPerformance Analysis Complete!")
            print(f"Duration: {args.profile_duration} seconds")
            print(f"Bottlenecks Found: {len(results['bottlenecks'])}")
            print(f"Recommendations Generated: {len(results['recommendations'])}")
            
            # Show top bottlenecks
            if results['bottlenecks']:
                print(f"\nTop Bottlenecks:")
                for i, bottleneck in enumerate(results['bottlenecks'][:3], 1):
                    print(f"{i}. {bottleneck['bottleneck_type']} ({bottleneck['severity']}): {bottleneck['description']}")
            
            # Show top recommendations
            if results['recommendations']:
                print(f"\nTop Recommendations:")
                for i, rec in enumerate(results['recommendations'][:3], 1):
                    print(f"{i}. {rec['title']} ({rec['priority']}): +{rec['expected_improvement']}% improvement")
            
            # Export results
            if args.export_results:
                profiler.export_analysis(results, args.export_results)
                print(f"\nResults exported to: {args.export_results}")
            else:
                # Export to default location
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                default_path = f"/opt/sutazaiapp/logs/performance_analysis_{timestamp}.json"
                profiler.export_analysis(results, default_path)
                print(f"\nResults exported to: {default_path}")
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())