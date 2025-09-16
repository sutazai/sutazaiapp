"""
System Monitor Module
Real-time monitoring of system resources and service health
"""

import psutil
import docker
import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import subprocess
import json
import requests
from queue import Queue
import asyncio
import concurrent.futures

class SystemMonitor:
    """Advanced system monitoring with real-time metrics"""
    
    # Class-level instance for singleton pattern
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def get_cpu_usage(cls) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception:
            return 0.0
    
    @classmethod
    def get_memory_usage(cls) -> float:
        """Get current memory usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0
    
    @classmethod
    def get_disk_usage(cls) -> float:
        """Get disk usage percentage for root partition"""
        try:
            return psutil.disk_usage('/').percent
        except Exception:
            return 0.0
    
    @classmethod
    def get_network_speed(cls) -> float:
        """Get current network speed in MB/s"""
        try:
            stats = psutil.net_io_counters()
            # Simple approximation - would need to track over time for actual speed
            return round((stats.bytes_sent + stats.bytes_recv) / (1024 * 1024 * 100), 2)
        except Exception:
            return 0.0
    
    @classmethod
    def get_docker_stats(cls) -> List[Dict]:
        """Get Docker container statistics"""
        containers = []
        try:
            docker_client = docker.from_env()
            for container in docker_client.containers.list():
                stats = container.stats(stream=False)
                
                # Calculate CPU percentage
                cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                           stats["precpu_stats"]["cpu_usage"]["total_usage"]
                system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                              stats["precpu_stats"]["system_cpu_usage"]
                cpu_percent = 0.0
                if system_delta > 0:
                    cpu_percent = (cpu_delta / system_delta) * 100.0
                
                # Calculate memory usage
                mem_usage = stats["memory_stats"].get("usage", 0) / (1024 * 1024)  # MB
                
                containers.append({
                    "name": container.name,
                    "status": container.status,
                    "cpu": round(cpu_percent, 2),
                    "memory": round(mem_usage, 2),
                    "uptime": container.attrs["State"].get("StartedAt", "Unknown")
                })
        except Exception as e:
            print(f"Docker stats error: {e}")
        
        return containers
    
    def __init__(self):
        self.metrics_queue = Queue()
        self.monitoring_thread = None
        self.is_monitoring = False
        self.update_interval = 5  # seconds
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            print(f"Docker client initialization warning: {e}")
            self.docker_client = None
        
        # Service endpoints
        self.service_endpoints = {
            "PostgreSQL": "http://localhost:10000",
            "Redis": "http://localhost:10001",
            "Neo4j": "http://localhost:10002",
            "RabbitMQ": "http://localhost:10005/api/health/checks/virtual-hosts",
            "Consul": "http://localhost:10007/v1/health/node/consul",
            "Kong": "http://localhost:10009/status",
            "ChromaDB": "http://localhost:10100/api/v1/heartbeat",
            "Qdrant": "http://localhost:10102/health",
            "FAISS": "http://localhost:10103/health",
            "Backend API": "http://localhost:10200/health",
            "Ollama": "http://localhost:11434/api/tags"
        }
        
        # Historical metrics storage
        self.metrics_history = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
            "gpu": []
        }
        self.max_history_size = 100
        
    def start_monitoring(self, callback=None):
        """Start background monitoring thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitor_loop,
                args=(callback,),
                daemon=True
            )
            self.monitoring_thread.start()
            return True
        return False
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
            self.monitoring_thread = None
    
    def _monitor_loop(self, callback):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self.get_all_metrics()
                self.metrics_queue.put(metrics)
                
                # Store historical data
                self._update_history(metrics)
                
                if callback:
                    callback(metrics)
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
    
    def get_all_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": self.get_cpu_metrics(),
            "memory": self.get_memory_metrics(),
            "disk": self.get_disk_metrics(),
            "network": self.get_network_metrics(),
            "gpu": self.get_gpu_metrics(),
            "docker": self.get_docker_metrics(),
            "services": self.get_service_health(),
            "processes": self.get_top_processes()
        }
        return metrics
    
    def get_cpu_metrics(self) -> Dict:
        """Get CPU usage metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_freq = psutil.cpu_freq()
            cpu_stats = psutil.cpu_stats()
            
            return {
                "usage_percent": sum(cpu_percent) / len(cpu_percent),
                "per_core": cpu_percent,
                "core_count": psutil.cpu_count(logical=False),
                "thread_count": psutil.cpu_count(logical=True),
                "frequency_current": cpu_freq.current if cpu_freq else 0,
                "frequency_max": cpu_freq.max if cpu_freq else 0,
                "ctx_switches": cpu_stats.ctx_switches,
                "interrupts": cpu_stats.interrupts,
                "load_average": psutil.getloadavg()
            }
        except Exception as e:
            print(f"CPU metrics error: {e}")
            return {"error": str(e)}
    
    def get_memory_metrics(self) -> Dict:
        """Get memory usage metrics"""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "total_gb": mem.total / (1024**3),
                "used_gb": mem.used / (1024**3),
                "available_gb": mem.available / (1024**3),
                "percent": mem.percent,
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_gb": swap.used / (1024**3),
                "swap_percent": swap.percent,
                "buffers_gb": mem.buffers / (1024**3) if hasattr(mem, 'buffers') else 0,
                "cached_gb": mem.cached / (1024**3) if hasattr(mem, 'cached') else 0
            }
        except Exception as e:
            print(f"Memory metrics error: {e}")
            return {"error": str(e)}
    
    def get_disk_metrics(self) -> Dict:
        """Get disk usage metrics"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            partitions = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    partitions.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "percent": usage.percent
                    })
                except:
                    continue
            
            return {
                "total_gb": disk_usage.total / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "percent": disk_usage.percent,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0,
                "partitions": partitions
            }
        except Exception as e:
            print(f"Disk metrics error: {e}")
            return {"error": str(e)}
    
    def get_network_metrics(self) -> Dict:
        """Get network usage metrics"""
        try:
            net_io = psutil.net_io_counters()
            connections = psutil.net_connections(kind='inet')
            
            # Count connections by state
            conn_states = {}
            for conn in connections:
                state = conn.status if hasattr(conn, 'status') else 'UNKNOWN'
                conn_states[state] = conn_states.get(state, 0) + 1
            
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout,
                "connections": conn_states,
                "connection_count": len(connections)
            }
        except Exception as e:
            print(f"Network metrics error: {e}")
            return {"error": str(e)}
    
    def get_gpu_metrics(self) -> Dict:
        """Get GPU metrics using nvidia-smi"""
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpus.append({
                            "name": parts[0],
                            "memory_total_mb": float(parts[1]),
                            "memory_used_mb": float(parts[2]),
                            "memory_free_mb": float(parts[3]),
                            "utilization_percent": float(parts[4]),
                            "temperature_c": float(parts[5])
                        })
                
                return {"gpus": gpus, "available": True}
            else:
                return {"available": False, "error": "nvidia-smi failed"}
                
        except FileNotFoundError:
            return {"available": False, "error": "nvidia-smi not found"}
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def get_docker_metrics(self) -> Dict:
        """Get Docker container metrics"""
        if not self.docker_client:
            return {"available": False, "error": "Docker client not initialized"}
        
        try:
            containers = self.docker_client.containers.list(all=True)
            container_info = []
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0
                    
                    # Calculate memory usage
                    mem_usage = stats['memory_stats'].get('usage', 0)
                    mem_limit = stats['memory_stats'].get('limit', 1)
                    mem_percent = (mem_usage / mem_limit) * 100 if mem_limit > 0 else 0
                    
                    container_info.append({
                        "name": container.name,
                        "id": container.short_id,
                        "status": container.status,
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_mb": round(mem_usage / (1024*1024), 2),
                        "memory_percent": round(mem_percent, 2),
                        "ports": container.ports
                    })
                except Exception as e:
                    container_info.append({
                        "name": container.name,
                        "status": container.status,
                        "error": str(e)
                    })
            
            return {
                "available": True,
                "container_count": len(containers),
                "containers": container_info,
                "running_count": len([c for c in containers if c.status == "running"])
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def get_service_health(self) -> Dict:
        """Check health of all services"""
        health_status = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_service = {
                executor.submit(self._check_service_health, name, url): name 
                for name, url in self.service_endpoints.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_service):
                service_name = future_to_service[future]
                try:
                    health_status[service_name] = future.result()
                except Exception as e:
                    health_status[service_name] = {
                        "status": "error",
                        "error": str(e),
                        "response_time_ms": -1
                    }
        
        # Calculate overall health
        total = len(health_status)
        healthy = len([s for s in health_status.values() if s["status"] == "healthy"])
        
        return {
            "services": health_status,
            "summary": {
                "total": total,
                "healthy": healthy,
                "unhealthy": total - healthy,
                "health_percentage": (healthy / total * 100) if total > 0 else 0
            }
        }
    
    def _check_service_health(self, name: str, url: str) -> Dict:
        """Check individual service health"""
        try:
            start_time = time.time()
            
            # Special handling for certain services
            if name == "PostgreSQL":
                # PostgreSQL doesn't have HTTP endpoint, check port
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(("localhost", 10000))
                sock.close()
                if result == 0:
                    return {
                        "status": "healthy",
                        "response_time_ms": round((time.time() - start_time) * 1000, 2)
                    }
                else:
                    return {"status": "unhealthy", "response_time_ms": -1}
            
            elif name == "Redis":
                # Redis doesn't have HTTP endpoint, check port
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(("localhost", 10001))
                sock.close()
                if result == 0:
                    return {
                        "status": "healthy",
                        "response_time_ms": round((time.time() - start_time) * 1000, 2)
                    }
                else:
                    return {"status": "unhealthy", "response_time_ms": -1}
            
            else:
                # HTTP health check
                response = requests.get(url, timeout=2)
                response_time = round((time.time() - start_time) * 1000, 2)
                
                if response.status_code < 400:
                    return {
                        "status": "healthy",
                        "response_time_ms": response_time,
                        "status_code": response.status_code
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "response_time_ms": response_time,
                        "status_code": response.status_code
                    }
                    
        except requests.ConnectionError:
            return {"status": "offline", "response_time_ms": -1}
        except requests.Timeout:
            return {"status": "timeout", "response_time_ms": -1}
        except Exception as e:
            return {"status": "error", "error": str(e), "response_time_ms": -1}
    
    def get_top_processes(self, count: int = 10) -> List[Dict]:
        """Get top processes by CPU and memory usage"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    processes.append({
                        "pid": pinfo['pid'],
                        "name": pinfo['name'],
                        "cpu_percent": pinfo['cpu_percent'] or 0,
                        "memory_percent": pinfo['memory_percent'] or 0
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            top_cpu = processes[:count]
            
            # Sort by memory usage
            processes.sort(key=lambda x: x['memory_percent'], reverse=True)
            top_memory = processes[:count]
            
            return {
                "top_cpu": top_cpu,
                "top_memory": top_memory
            }
        except Exception as e:
            print(f"Process metrics error: {e}")
            return {"error": str(e)}
    
    def _update_history(self, metrics: Dict):
        """Update historical metrics data"""
        try:
            # Add to history
            if "cpu" in metrics:
                self.metrics_history["cpu"].append({
                    "timestamp": metrics["timestamp"],
                    "value": metrics["cpu"].get("usage_percent", 0)
                })
            
            if "memory" in metrics:
                self.metrics_history["memory"].append({
                    "timestamp": metrics["timestamp"],
                    "value": metrics["memory"].get("percent", 0)
                })
            
            if "disk" in metrics:
                self.metrics_history["disk"].append({
                    "timestamp": metrics["timestamp"],
                    "value": metrics["disk"].get("percent", 0)
                })
            
            if "network" in metrics:
                self.metrics_history["network"].append({
                    "timestamp": metrics["timestamp"],
                    "bytes_sent": metrics["network"].get("bytes_sent", 0),
                    "bytes_recv": metrics["network"].get("bytes_recv", 0)
                })
            
            if "gpu" in metrics and metrics["gpu"].get("available"):
                gpus = metrics["gpu"].get("gpus", [])
                if gpus:
                    self.metrics_history["gpu"].append({
                        "timestamp": metrics["timestamp"],
                        "utilization": gpus[0].get("utilization_percent", 0),
                        "memory_used": gpus[0].get("memory_used_mb", 0)
                    })
            
            # Trim history to max size
            for key in self.metrics_history:
                if len(self.metrics_history[key]) > self.max_history_size:
                    self.metrics_history[key] = self.metrics_history[key][-self.max_history_size:]
                    
        except Exception as e:
            print(f"History update error: {e}")
    
    def get_metrics_history(self, metric_type: str, count: int = 20) -> List[Dict]:
        """Get historical metrics data"""
        if metric_type in self.metrics_history:
            return self.metrics_history[metric_type][-count:]
        return []
    
    def get_system_summary(self) -> Dict:
        """Get a summary of system status"""
        metrics = self.get_all_metrics()
        
        return {
            "timestamp": metrics["timestamp"],
            "cpu_usage": metrics["cpu"].get("usage_percent", 0),
            "memory_usage": metrics["memory"].get("percent", 0),
            "disk_usage": metrics["disk"].get("percent", 0),
            "gpu_available": metrics["gpu"].get("available", False),
            "docker_running": metrics["docker"].get("running_count", 0),
            "services_healthy": metrics["services"]["summary"].get("healthy", 0),
            "services_total": metrics["services"]["summary"].get("total", 0),
            "alerts": self._check_alerts(metrics)
        }
    
    def _check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for system alerts"""
        alerts = []
        
        # CPU alert
        cpu_usage = metrics["cpu"].get("usage_percent", 0)
        if cpu_usage > 90:
            alerts.append({
                "type": "critical",
                "category": "cpu",
                "message": f"High CPU usage: {cpu_usage:.1f}%"
            })
        elif cpu_usage > 75:
            alerts.append({
                "type": "warning",
                "category": "cpu",
                "message": f"Elevated CPU usage: {cpu_usage:.1f}%"
            })
        
        # Memory alert
        mem_usage = metrics["memory"].get("percent", 0)
        if mem_usage > 90:
            alerts.append({
                "type": "critical",
                "category": "memory",
                "message": f"High memory usage: {mem_usage:.1f}%"
            })
        elif mem_usage > 80:
            alerts.append({
                "type": "warning",
                "category": "memory",
                "message": f"Elevated memory usage: {mem_usage:.1f}%"
            })
        
        # Disk alert
        disk_usage = metrics["disk"].get("percent", 0)
        if disk_usage > 95:
            alerts.append({
                "type": "critical",
                "category": "disk",
                "message": f"Critical disk usage: {disk_usage:.1f}%"
            })
        elif disk_usage > 85:
            alerts.append({
                "type": "warning",
                "category": "disk",
                "message": f"High disk usage: {disk_usage:.1f}%"
            })
        
        # Service alerts
        services_summary = metrics["services"]["summary"]
        unhealthy = services_summary.get("unhealthy", 0)
        if unhealthy > 0:
            alerts.append({
                "type": "warning",
                "category": "services",
                "message": f"{unhealthy} service(s) unhealthy"
            })
        
        return alerts
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        if self.docker_client:
            self.docker_client.close()