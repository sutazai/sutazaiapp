#!/usr/bin/env python3
"""
Enhanced Debug Tracer for Hardware Resource Optimizer Agent

Purpose: Provide detailed debugging and tracing of actual optimization execution
Usage: Used by the main agent to trace real system calls and effects
Requirements: psutil, logging, time measurement utilities
"""

import os
import time
import logging
import subprocess
import psutil
import json
from datetime import datetime
from functools import wraps
import traceback

class DebugTracer:
    """Enhanced debugging and tracing for hardware optimization"""
    
    def __init__(self, log_level=logging.DEBUG):
        self.debug_dir = "/opt/sutazaiapp/agents/hardware-resource-optimizer/debug_logs"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Setup detailed debug logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.debug_dir, f"debug_trace_{timestamp}.log")
        
        self.logger = logging.getLogger("HardwareOptimizerDebug")
        self.logger.setLevel(log_level)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Trace storage
        self.execution_traces = []
        self.performance_metrics = {}
        self.system_snapshots = []
        self.file_operations = []
        self.subprocess_calls = []
        
        self.logger.info(f"Debug tracer initialized - logs: {log_file}")
    
    def take_system_snapshot(self, label: str) -> Dict[str, Any]:
        """Take comprehensive system snapshot for before/after comparison"""
        try:
            snapshot = {
                "label": label,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "memory": dict(psutil.virtual_memory()._asdict()),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "disk_usage": dict(psutil.disk_usage('/')._asdict()),
                "processes": self._get_process_summary(),
                "open_files": self._get_open_files_count(),
                "network_connections": len(psutil.net_connections()),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            # Add Docker info if available
            try:
                import docker
                client = docker.from_env()
                snapshot["docker"] = {
                    "containers_running": len(client.containers.list()),
                    "containers_total": len(client.containers.list(all=True)),
                    "images_count": len(client.images.list()),
                    "volumes_count": len(client.volumes.list())
                }
            except Exception:
                snapshot["docker"] = None
            
            self.system_snapshots.append(snapshot)
            self.logger.debug(f"System snapshot '{label}': {json.dumps(snapshot, indent=2)}")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to take system snapshot '{label}': {e}")
            return {"label": label, "error": str(e), "timestamp": time.time()}
    
    def _get_process_summary(self) -> Dict[str, Any]:
        """Get summary of running processes"""
        try:
            processes = []
            total_memory = 0
            total_cpu = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
                try:
                    info = proc.info
                    if info['cpu_percent'] and info['cpu_percent'] > 1:  # Only significant processes
                        processes.append(info)
                        total_memory += info['memory_percent'] or 0
                        total_cpu += info['cpu_percent'] or 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                "total_processes": len(psutil.pids()),
                "high_cpu_processes": len([p for p in processes if p['cpu_percent'] > 5]),
                "high_memory_processes": len([p for p in processes if p['memory_percent'] > 5]),
                "top_cpu_processes": sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:5],
                "top_memory_processes": sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:5]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_open_files_count(self) -> int:
        """Get total number of open file descriptors"""
        try:
            total = 0
            for proc in psutil.process_iter():
                try:
                    total += proc.num_fds()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return total
        except Exception:
            return 0
    
    def trace_function_execution(self, func_name: str = None):
        """Decorator to trace function execution with detailed timing and effects"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                
                # Take before snapshot
                before_snapshot = self.take_system_snapshot(f"before_{name}")
                
                self.logger.info(f"TRACE START: {name} with args={args[:2] if args else []}, kwargs={kwargs}")
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Take after snapshot
                    after_snapshot = self.take_system_snapshot(f"after_{name}")
                    
                    execution_time = time.time() - start_time
                    
                    # Calculate differences
                    differences = self._calculate_snapshot_differences(before_snapshot, after_snapshot)
                    
                    # Store execution trace
                    trace = {
                        "function": name,
                        "start_time": start_time,
                        "execution_time": execution_time,
                        "success": True,
                        "before_snapshot": before_snapshot,
                        "after_snapshot": after_snapshot,
                        "differences": differences,
                        "result_status": result.get('status') if isinstance(result, dict) else 'unknown',
                        "actions_taken": result.get('actions_taken', []) if isinstance(result, dict) else []
                    }
                    
                    self.execution_traces.append(trace)
                    
                    self.logger.info(f"TRACE END: {name} - SUCCESS in {execution_time:.3f}s - {differences}")
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    after_snapshot = self.take_system_snapshot(f"error_{name}")
                    
                    trace = {
                        "function": name,
                        "start_time": start_time,
                        "execution_time": execution_time,
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "before_snapshot": before_snapshot,
                        "after_snapshot": after_snapshot
                    }
                    
                    self.execution_traces.append(trace)
                    
                    self.logger.error(f"TRACE ERROR: {name} - FAILED in {execution_time:.3f}s - {e}")
                    
                    raise
            
            return wrapper
        
        return decorator
    
    def _calculate_snapshot_differences(self, before: Dict, after: Dict) -> Dict[str, Any]:
        """Calculate meaningful differences between system snapshots"""
        try:
            if before.get('error') or after.get('error'):
                return {"error": "Cannot calculate differences due to snapshot errors"}
            
            differences = {}
            
            # Memory differences
            if 'memory' in before and 'memory' in after:
                mem_before = before['memory']
                mem_after = after['memory']
                differences['memory_change_mb'] = (mem_before['used'] - mem_after['used']) / (1024 * 1024)
                differences['memory_percent_change'] = mem_before['percent'] - mem_after['percent']
            
            # Disk differences
            if 'disk_usage' in before and 'disk_usage' in after:
                disk_before = before['disk_usage']
                disk_after = after['disk_usage']
                differences['disk_freed_mb'] = (disk_before['used'] - disk_after['used']) / (1024 * 1024)
                differences['disk_percent_change'] = ((disk_before['used'] / disk_before['total']) - 
                                                    (disk_after['used'] / disk_after['total'])) * 100
            
            # Process differences
            if 'processes' in before and 'processes' in after:
                differences['process_count_change'] = (before['processes'].get('total_processes', 0) - 
                                                     after['processes'].get('total_processes', 0))
            
            # Docker differences
            if before.get('docker') and after.get('docker'):
                docker_before = before['docker']
                docker_after = after['docker']
                differences['docker_containers_removed'] = (docker_before['containers_total'] - 
                                                          docker_after['containers_total'])
                differences['docker_images_removed'] = (docker_before['images_count'] - 
                                                      docker_after['images_count'])
            
            return differences
            
        except Exception as e:
            return {"calculation_error": str(e)}
    
    def trace_subprocess_call(self, cmd: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Trace subprocess calls with detailed logging"""
        start_time = time.time()
        cmd_str = ' '.join(cmd)
        
        self.logger.info(f"SUBPROCESS START: {cmd_str}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                check=False  # Don't raise on non-zero exit
            )
            
            execution_time = time.time() - start_time
            
            subprocess_trace = {
                "command": cmd_str,
                "start_time": start_time,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout[:1000] if result.stdout else "",  # Limit output
                "stderr": result.stderr[:1000] if result.stderr else "",
                "success": result.returncode == 0
            }
            
            self.subprocess_calls.append(subprocess_trace)
            
            if result.returncode == 0:
                self.logger.info(f"SUBPROCESS SUCCESS: {cmd_str} in {execution_time:.3f}s")
            else:
                self.logger.warning(f"SUBPROCESS FAILED: {cmd_str} - code {result.returncode} - {result.stderr}")
            
            return result
            
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            self.logger.error(f"SUBPROCESS TIMEOUT: {cmd_str} after {execution_time:.3f}s")
            
            subprocess_trace = {
                "command": cmd_str,
                "start_time": start_time,
                "execution_time": execution_time,
                "error": "timeout",
                "success": False
            }
            self.subprocess_calls.append(subprocess_trace)
            
            raise
    
    def trace_file_operation(self, operation: str, path: str, details: Dict = None):
        """Trace file system operations"""
        file_op = {
            "timestamp": time.time(),
            "operation": operation,
            "path": path,
            "details": details or {},
            "exists_before": os.path.exists(path),
            "size_before": os.path.getsize(path) if os.path.exists(path) else 0
        }
        
        self.file_operations.append(file_op)
        self.logger.debug(f"FILE OP: {operation} on {path} - {details}")
    
    def verify_actual_changes(self, expected_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that expected changes actually occurred in the system"""
        verification_results = {
            "timestamp": time.time(),
            "expected": expected_changes,
            "actual": {},
            "verified": {}
        }
        
        # Check memory changes
        if "memory_freed_mb" in expected_changes:
            if len(self.system_snapshots) >= 2:
                latest_diff = self._calculate_snapshot_differences(
                    self.system_snapshots[-2], 
                    self.system_snapshots[-1]
                )
                actual_memory_change = latest_diff.get('memory_change_mb', 0)
                expected_memory_change = expected_changes['memory_freed_mb']
                
                verification_results["actual"]["memory_change_mb"] = actual_memory_change
                verification_results["verified"]["memory_change"] = (
                    abs(actual_memory_change - expected_memory_change) < expected_memory_change * 0.5
                )
        
        # Check disk changes
        if "space_freed_mb" in expected_changes:
            if len(self.system_snapshots) >= 2:
                latest_diff = self._calculate_snapshot_differences(
                    self.system_snapshots[-2], 
                    self.system_snapshots[-1]
                )
                actual_disk_change = latest_diff.get('disk_freed_mb', 0)
                expected_disk_change = expected_changes['space_freed_mb']
                
                verification_results["actual"]["disk_change_mb"] = actual_disk_change
                verification_results["verified"]["disk_change"] = (
                    abs(actual_disk_change - expected_disk_change) < max(expected_disk_change * 0.5, 1.0)
                )
        
        self.logger.info(f"VERIFICATION: {verification_results}")
        return verification_results
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report"""
        report = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "debug_session_duration": time.time() - (self.system_snapshots[0]['timestamp'] if self.system_snapshots else time.time()),
            "system_snapshots_count": len(self.system_snapshots),
            "execution_traces_count": len(self.execution_traces),
            "subprocess_calls_count": len(self.subprocess_calls),
            "file_operations_count": len(self.file_operations),
            "system_snapshots": self.system_snapshots,
            "execution_traces": self.execution_traces,
            "subprocess_calls": self.subprocess_calls,
            "file_operations": self.file_operations,
            "performance_summary": self._generate_performance_summary()
        }
        
        # Save report to file
        report_file = os.path.join(self.debug_dir, f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Debug report saved: {report_file}")
        return report
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from traces"""
        if not self.execution_traces:
            return {"no_traces": True}
        
        successful_traces = [t for t in self.execution_traces if t.get('success', False)]
        failed_traces = [t for t in self.execution_traces if not t.get('success', False)]
        
        execution_times = [t['execution_time'] for t in successful_traces]
        
        return {
            "total_functions_traced": len(self.execution_traces),
            "successful_executions": len(successful_traces),
            "failed_executions": len(failed_traces),
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "longest_execution": max(execution_times) if execution_times else 0,
            "shortest_execution": min(execution_times) if execution_times else 0,
            "total_execution_time": sum(execution_times),
            "functions_by_execution_time": sorted(
                [(t['function'], t['execution_time']) for t in successful_traces],
                key=lambda x: x[1], reverse=True
            )[:10]
        }

# Global debug tracer instance
debug_tracer = DebugTracer()