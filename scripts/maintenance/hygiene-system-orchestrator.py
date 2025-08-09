#!/usr/bin/env python3
"""
Purpose: Master orchestrator for the complete hygiene enforcement system
Usage: python hygiene-system-orchestrator.py [--mode MODE] [--config CONFIG]
Requirements: All system components must be available and properly configured
"""

import os
import sys
import json
import time
import signal
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import socket

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/system-orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/sutazaiapp")
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

@dataclass
class ServiceStatus:
    name: str
    status: str  # starting, running, stopped, error, unhealthy
    pid: Optional[int]
    port: Optional[int]
    health_score: float
    last_health_check: datetime
    start_time: Optional[datetime]
    restart_count: int
    error_message: Optional[str]

@dataclass
class SystemHealthReport:
    timestamp: datetime
    overall_status: str
    services: Dict[str, ServiceStatus]
    resource_usage: Dict[str, float]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]

class HealthChecker:
    """Advanced health checking with circuit breaker pattern"""
    
    def __init__(self):
        self.health_history = {}
        self.circuit_breakers = {}
        
    async def check_service_health(self, service_name: str, config: Dict) -> ServiceStatus:
        """Comprehensive health check for a service"""
        status = ServiceStatus(
            name=service_name,
            status="unknown",
            pid=None,
            port=config.get("port"),
            health_score=0.0,
            last_health_check=datetime.now(),
            start_time=None,
            restart_count=0,
            error_message=None
        )
        
        try:
            # Check if process is running
            pid = self._find_service_pid(service_name, config)
            if pid:
                status.pid = pid
                status.status = "running"
                
                # Get process details
                try:
                    process = psutil.Process(pid)
                    status.start_time = datetime.fromtimestamp(process.create_time())
                    
                    # Check resource usage
                    cpu_percent = process.cpu_percent(interval=0.1)
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    # Health score based on resource usage and responsiveness
                    health_score = 100.0
                    
                    # Deduct points for high resource usage
                    if cpu_percent > 80:
                        health_score -= 30
                    elif cpu_percent > 50:
                        health_score -= 10
                        
                    if memory_mb > 2048:  # 2GB
                        health_score -= 20
                    elif memory_mb > 1024:  # 1GB
                        health_score -= 10
                    
                    # Test service responsiveness if it has an endpoint
                    if config.get("health_endpoint"):
                        response_time = await self._test_endpoint_health(config["health_endpoint"])
                        if response_time > 5000:  # 5 seconds
                            health_score -= 30
                            status.error_message = "Slow response time"
                        elif response_time > 2000:  # 2 seconds
                            health_score -= 15
                        elif response_time < 0:  # Failed
                            health_score -= 50
                            status.error_message = "Endpoint unreachable"
                            status.status = "unhealthy"
                    
                    status.health_score = max(0.0, health_score)
                    
                    # Determine final status
                    if status.health_score >= 80:
                        status.status = "healthy"
                    elif status.health_score >= 50:
                        status.status = "degraded"
                    else:
                        status.status = "unhealthy"
                        
                except psutil.NoSuchProcess:
                    status.status = "stopped"
                    status.error_message = "Process not found"
                    
            else:
                status.status = "stopped"
                status.error_message = "Service not running"
                
        except Exception as e:
            status.status = "error"
            status.error_message = str(e)
            logger.error(f"Health check failed for {service_name}: {e}")
            
        return status
    
    def _find_service_pid(self, service_name: str, config: Dict) -> Optional[int]:
        """Find PID of service process"""
        try:
            # Try multiple approaches to find the process
            
            # Method 1: Check by port if available
            if config.get("port"):
                for conn in psutil.net_connections():
                    if conn.laddr.port == config["port"] and conn.status == psutil.CONN_LISTEN:
                        return conn.pid
            
            # Method 2: Search by command line pattern
            search_patterns = [
                config.get("command_pattern", service_name),
                config.get("script_name", f"{service_name}.py"),
                f"python.*{service_name}",
            ]
            
            for process in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(process.info['cmdline'] or [])
                    for pattern in search_patterns:
                        if pattern.lower() in cmdline.lower():
                            return process.info['pid']
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.warning(f"Error finding PID for {service_name}: {e}")
            
        return None
    
    async def _test_endpoint_health(self, endpoint: str) -> float:
        """Test endpoint responsiveness, return response time in ms or -1 if failed"""
        import aiohttp
        
        try:
            start_time = time.time()
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(endpoint) as response:
                    response_time = (time.time() - start_time) * 1000
                    if response.status < 400:
                        return response_time
                    else:
                        return -1
                        
        except Exception as e:
            logger.debug(f"Endpoint health check failed for {endpoint}: {e}")
            return -1

class HygieneSystemOrchestrator:
    """Master orchestrator for the complete hygiene enforcement system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = PROJECT_ROOT
        self.config_file = Path(config_file) if config_file else CONFIG_DIR / "system-orchestration.json"
        self.services = {}
        self.running = False
        self.health_checker = HealthChecker()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Performance monitoring
        self.start_time = datetime.now()
        self.service_stats = {}
        
    def _load_configuration(self) -> Dict:
        """Load system orchestration configuration"""
        default_config = {
            "services": {
                "rule-control-api": {
                    "script": "scripts/agents/rule-control-manager.py",
                    "port": 8100,
                    "health_endpoint": "http://localhost:8100/api/health",
                    "startup_timeout": 30,
                    "critical": True,
                    "auto_restart": True,
                    "max_restarts": 3,
                    "dependencies": []
                },
                "hygiene-enforcement-coordinator": {
                    "script": "scripts/hygiene-enforcement-coordinator.py",
                    "startup_timeout": 60,
                    "critical": True,
                    "auto_restart": True,
                    "max_restarts": 5,
                    "dependencies": ["rule-control-api"],
                    "periodic_run": True,
                    "run_interval": 300  # 5 minutes
                },
                "testing-qa-validator": {
                    "script": "scripts/agents/testing-qa-validator.py",
                    "startup_timeout": 120,
                    "critical": False,
                    "auto_restart": False,
                    "max_restarts": 1,
                    "dependencies": ["rule-control-api"],
                    "on_demand": True
                },
                "dashboard-server": {
                    "script": "dashboard/start-dashboard.py",
                    "port": 8080,
                    "health_endpoint": "http://localhost:8080/health",
                    "startup_timeout": 20,
                    "critical": False,
                    "auto_restart": True,
                    "max_restarts": 3,
                    "dependencies": ["rule-control-api"]
                }
            },
            "system": {
                "health_check_interval": 30,
                "startup_sequence_delay": 5,
                "shutdown_timeout": 30,
                "log_level": "INFO",
                "enable_performance_monitoring": True,
                "resource_limits": {
                    "max_cpu_percent": 80,
                    "max_memory_mb": 8192,
                    "max_disk_usage_percent": 90
                }
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    user_config = json.load(f)
                    # Deep merge configurations
                    self._deep_merge(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def start_system(self):
        """Start the complete hygiene enforcement system"""
        logger.info("🚀 Starting Sutazai Hygiene Enforcement System...")
        
        self.running = True
        startup_tasks = []
        
        try:
            # Create necessary directories
            LOGS_DIR.mkdir(exist_ok=True)
            CONFIG_DIR.mkdir(exist_ok=True)
            
            # Start services in dependency order
            service_order = self._calculate_startup_order()
            
            for service_name in service_order:
                if not self.running:
                    break
                    
                service_config = self.config["services"][service_name]
                logger.info(f"Starting service: {service_name}")
                
                try:
                    await self._start_service(service_name, service_config)
                    
                    # Wait between service starts to avoid resource contention
                    if self.config["system"]["startup_sequence_delay"] > 0:
                        await asyncio.sleep(self.config["system"]["startup_sequence_delay"])
                        
                except Exception as e:
                    logger.error(f"Failed to start {service_name}: {e}")
                    if service_config.get("critical", True):
                        logger.error(f"Critical service {service_name} failed, aborting startup")
                        await self.shutdown_system()
                        return False
            
            # Start health monitoring
            if self.running:
                health_task = asyncio.create_task(self._health_monitoring_loop())
                startup_tasks.append(health_task)
                
                # Start performance monitoring
                if self.config["system"].get("enable_performance_monitoring", True):
                    perf_task = asyncio.create_task(self._performance_monitoring_loop())
                    startup_tasks.append(perf_task)
                
                logger.info("✅ System startup completed successfully")
                
                # Wait for shutdown signal
                while self.running:
                    await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            return False
        finally:
            # Cancel monitoring tasks
            for task in startup_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        return True
    
    def _calculate_startup_order(self) -> List[str]:
        """Calculate the correct service startup order based on dependencies"""
        services = self.config["services"]
        ordered = []
        visited = set()
        temp_visited = set()
        
        def visit(service_name):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_name}")
            if service_name in visited:
                return
                
            temp_visited.add(service_name)
            
            # Visit dependencies first
            for dep in services[service_name].get("dependencies", []):
                if dep in services:
                    visit(dep)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            ordered.append(service_name)
        
        for service_name in services:
            if service_name not in visited:
                visit(service_name)
        
        return ordered
    
    async def _start_service(self, service_name: str, config: Dict):
        """Start a specific service"""
        script_path = self.project_root / config["script"]
        
        if not script_path.exists():
            raise FileNotFoundError(f"Service script not found: {script_path}")
        
        # Prepare command
        cmd = [sys.executable, str(script_path)]
        
        # Add any service-specific arguments
        if config.get("args"):
            cmd.extend(config["args"])
        
        # Start process
        if config.get("periodic_run"):
            # For periodic services, we'll manage them differently
            self.services[service_name] = {
                "type": "periodic",
                "config": config,
                "last_run": None,
                "next_run": datetime.now()
            }
        else:
            # Start as daemon process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            self.services[service_name] = {
                "type": "daemon",
                "process": process,
                "config": config,
                "start_time": datetime.now(),
                "restart_count": 0
            }
            
            # Wait for service to be ready
            await self._wait_for_service_ready(service_name, config)
    
    async def _wait_for_service_ready(self, service_name: str, config: Dict):
        """Wait for service to be ready and healthy"""
        timeout = config.get("startup_timeout", 30)
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if not self.running:
                return
                
            # Check if service is responding
            if config.get("health_endpoint"):
                try:
                    status = await self.health_checker.check_service_health(service_name, config)
                    if status.status in ["healthy", "running"]:
                        logger.info(f"✅ Service {service_name} is ready")
                        return
                except Exception as e:
                    logger.debug(f"Health check pending for {service_name}: {e}")
            
            await asyncio.sleep(1)
        
        logger.warning(f"⚠️ Service {service_name} did not become ready within {timeout} seconds")
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring and auto-recovery"""
        interval = self.config["system"]["health_check_interval"]
        
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _perform_health_checks(self):
        """Perform health checks on all services"""
        health_report = SystemHealthReport(
            timestamp=datetime.now(),
            overall_status="healthy",
            services={},
            resource_usage={},
            performance_metrics={},
            recommendations=[]
        )
        
        # Check each service
        unhealthy_services = []
        
        for service_name, service_info in self.services.items():
            try:
                if service_info["type"] == "daemon":
                    config = service_info["config"]
                    status = await self.health_checker.check_service_health(service_name, config)
                    health_report.services[service_name] = status
                    
                    # Handle unhealthy services
                    if status.status in ["unhealthy", "error", "stopped"]:
                        unhealthy_services.append(service_name)
                        logger.warning(f"🔴 Service {service_name} is {status.status}: {status.error_message}")
                        
                        # Auto-restart if configured
                        if config.get("auto_restart", False) and service_info["restart_count"] < config.get("max_restarts", 3):
                            logger.info(f"🔄 Auto-restarting {service_name}...")
                            await self._restart_service(service_name)
                            service_info["restart_count"] += 1
                    
                elif service_info["type"] == "periodic":
                    # Handle periodic services
                    await self._handle_periodic_service(service_name, service_info)
                    
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                unhealthy_services.append(service_name)
        
        # Check system resources
        health_report.resource_usage = self._get_system_resources()
        
        # Generate recommendations
        health_report.recommendations = self._generate_health_recommendations(health_report)
        
        # Update overall status
        if len(unhealthy_services) == 0:
            health_report.overall_status = "healthy"
        elif len(unhealthy_services) < len(self.services) / 2:
            health_report.overall_status = "degraded"
        else:
            health_report.overall_status = "critical"
        
        # Log health report
        self._log_health_report(health_report)
        
        # Save detailed report
        await self._save_health_report(health_report)
    
    async def _handle_periodic_service(self, service_name: str, service_info: Dict):
        """Handle periodic service execution"""
        config = service_info["config"]
        interval = config.get("run_interval", 300)  # Default 5 minutes
        
        now = datetime.now()
        if now >= service_info["next_run"]:
            logger.info(f"🔄 Running periodic service: {service_name}")
            
            try:
                script_path = self.project_root / config["script"]
                cmd = [sys.executable, str(script_path)]
                
                # Add periodic run arguments
                if config.get("periodic_args"):
                    cmd.extend(config["periodic_args"])
                else:
                    cmd.extend(["--phase", "1"])  # Default for enforcement coordinator
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.project_root
                )
                
                # Wait for completion with timeout
                timeout = config.get("execution_timeout", 300)  # 5 minute default
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout
                    )
                    
                    if process.returncode == 0:
                        logger.info(f"✅ Periodic service {service_name} completed successfully")
                        service_info["last_run"] = now
                    else:
                        logger.error(f"❌ Periodic service {service_name} failed with code {process.returncode}")
                        logger.error(f"STDERR: {stderr.decode()}")
                        
                except asyncio.TimeoutError:
                    process.terminate()
                    await process.wait()
                    logger.error(f"⏰ Periodic service {service_name} timed out after {timeout} seconds")
                    
            except Exception as e:
                logger.error(f"Error running periodic service {service_name}: {e}")
            
            # Schedule next run
            service_info["next_run"] = now + datetime.timedelta(seconds=interval)
    
    def _get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            }
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {}
    
    def _generate_health_recommendations(self, report: SystemHealthReport) -> List[str]:
        """Generate health recommendations based on current system state"""
        recommendations = []
        
        # Resource usage recommendations
        if report.resource_usage.get("cpu_percent", 0) > 80:
            recommendations.append("High CPU usage detected - consider optimizing service configurations")
        
        if report.resource_usage.get("memory_percent", 0) > 85:
            recommendations.append("High memory usage detected - consider increasing system memory or optimizing services")
        
        if report.resource_usage.get("disk_percent", 0) > 90:
            recommendations.append("Disk space is running low - consider cleanup or expanding storage")
        
        # Service-specific recommendations
        unhealthy_count = sum(1 for status in report.services.values() if status.status in ["unhealthy", "error"])
        if unhealthy_count > 0:
            recommendations.append(f"{unhealthy_count} services need attention - check logs and restart if necessary")
        
        # Performance recommendations
        slow_services = [name for name, status in report.services.items() if status.health_score < 70]
        if slow_services:
            recommendations.append(f"Performance issues detected in: {', '.join(slow_services)}")
        
        return recommendations
    
    def _log_health_report(self, report: SystemHealthReport):
        """Log a concise health report"""
        status_counts = {}
        for status in report.services.values():
            status_counts[status.status] = status_counts.get(status.status, 0) + 1
        
        logger.info(f"🏥 Health Report: Overall={report.overall_status}, Services={status_counts}, CPU={report.resource_usage.get('cpu_percent', 0):.1f}%, Memory={report.resource_usage.get('memory_percent', 0):.1f}%")
    
    async def _save_health_report(self, report: SystemHealthReport):
        """Save detailed health report to file"""
        try:
            report_file = LOGS_DIR / f"health_report_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to JSON-serializable format
            report_data = {
                "timestamp": report.timestamp.isoformat(),
                "overall_status": report.overall_status,
                "services": {
                    name: {
                        "name": status.name,
                        "status": status.status,
                        "pid": status.pid,
                        "port": status.port,
                        "health_score": status.health_score,
                        "last_health_check": status.last_health_check.isoformat(),
                        "start_time": status.start_time.isoformat() if status.start_time else None,
                        "restart_count": status.restart_count,
                        "error_message": status.error_message
                    }
                    for name, status in report.services.items()
                },
                "resource_usage": report.resource_usage,
                "performance_metrics": report.performance_metrics,
                "recommendations": report.recommendations
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
    
    async def _performance_monitoring_loop(self):
        """Monitor system performance and optimize as needed"""
        while self.running:
            try:
                await self._optimize_system_performance()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _optimize_system_performance(self):
        """Automatically optimize system performance"""
        try:
            # Get current resource usage
            resources = self._get_system_resources()
            
            # If CPU usage is very high, temporarily reduce service activity
            if resources.get("cpu_percent", 0) > 90:
                logger.warning("🔥 High CPU usage detected, implementing emergency throttling")
                # Could implement service throttling here
            
            # If memory usage is high, trigger garbage collection
            if resources.get("memory_percent", 0) > 90:
                logger.warning("🧠 High memory usage detected, requesting garbage collection")
                # Could implement memory optimization here
                
        except Exception as e:
            logger.error(f"Performance optimization error: {e}")
    
    async def _restart_service(self, service_name: str):
        """Restart a specific service"""
        service_info = self.services.get(service_name)
        if not service_info:
            logger.error(f"Service {service_name} not found")
            return
        
        try:
            # Stop the service
            if service_info["type"] == "daemon" and "process" in service_info:
                process = service_info["process"]
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            
            # Start the service again
            config = service_info["config"]
            await self._start_service(service_name, config)
            
            logger.info(f"✅ Service {service_name} restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
    
    async def shutdown_system(self):
        """Gracefully shutdown the entire system"""
        logger.info("🛑 Initiating system shutdown...")
        
        self.running = False
        shutdown_timeout = self.config["system"]["shutdown_timeout"]
        
        # Stop all services
        for service_name, service_info in self.services.items():
            try:
                if service_info["type"] == "daemon" and "process" in service_info:
                    process = service_info["process"]
                    logger.info(f"Stopping service: {service_name}")
                    
                    process.terminate()
                    
                    try:
                        await asyncio.wait_for(process.wait(), timeout=shutdown_timeout)
                        logger.info(f"✅ Service {service_name} stopped gracefully")
                    except asyncio.TimeoutError:
                        logger.warning(f"⚠️ Service {service_name} did not stop gracefully, killing...")
                        process.kill()
                        await process.wait()
                        
            except Exception as e:
                logger.error(f"Error stopping service {service_name}: {e}")
        
        # Close executor
        self.executor.shutdown(wait=True)
        
        logger.info("✅ System shutdown completed")
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "overall_status": "running" if self.running else "stopped",
            "services": {},
            "resource_usage": self._get_system_resources(),
            "configuration": {
                "total_services": len(self.services),
                "critical_services": sum(1 for s in self.config["services"].values() if s.get("critical", True)),
                "health_check_interval": self.config["system"]["health_check_interval"]
            }
        }
        
        # Get service statuses
        for service_name, service_info in self.services.items():
            try:
                if service_info["type"] == "daemon":
                    config = service_info["config"]
                    service_status = await self.health_checker.check_service_health(service_name, config)
                    status["services"][service_name] = asdict(service_status)
                else:
                    status["services"][service_name] = {
                        "name": service_name,
                        "type": service_info["type"],
                        "status": "scheduled",
                        "next_run": service_info.get("next_run", {}).isoformat() if service_info.get("next_run") else None
                    }
            except Exception as e:
                logger.error(f"Error getting status for {service_name}: {e}")
                status["services"][service_name] = {
                    "name": service_name,
                    "status": "error",
                    "error": str(e)
                }
        
        return status

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hygiene System Orchestrator")
    parser.add_argument("--mode", choices=["start", "stop", "status", "restart"], 
                       default="start", help="Operation mode")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--service", help="Specific service name (for restart/stop)")
    
    args = parser.parse_args()
    
    orchestrator = HygieneSystemOrchestrator(args.config)
    
    try:
        if args.mode == "start":
            success = await orchestrator.start_system()
            return 0 if success else 1
            
        elif args.mode == "status":
            status = await orchestrator.get_system_status()
            print(json.dumps(status, indent=2))
            return 0
            
        elif args.mode == "stop":
            await orchestrator.shutdown_system()
            return 0
            
        elif args.mode == "restart":
            if args.service:
                await orchestrator._restart_service(args.service)
            else:
                await orchestrator.shutdown_system()
                await asyncio.sleep(2)
                success = await orchestrator.start_system()
                return 0 if success else 1
            return 0
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        await orchestrator.shutdown_system()
        return 0
    except Exception as e:
        logger.error(f"System orchestrator failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)