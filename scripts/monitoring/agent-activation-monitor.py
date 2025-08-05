#!/usr/bin/env python3
"""
Purpose: Monitor system health and agent activation progress during orchestration
Usage: python agent-activation-monitor.py [--dashboard] [--alerts]
Requirements: Docker, psutil, requests
"""

import os
import sys
import json
import time
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import docker
import psutil
import requests
import threading
from dataclasses import dataclass

@dataclass
class AgentStatus:
    name: str
    status: str  # running, starting, stopped, unhealthy
    container_id: str
    health: str  # healthy, unhealthy, unknown
    cpu_usage: float
    memory_usage: float
    uptime: timedelta
    restart_count: int

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    network_io: Dict
    active_agents: int
    healthy_agents: int
    total_containers: int

class AgentActivationMonitor:
    """Real-time monitoring of agent activation and system health"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        self.monitoring_active = False
        self.metrics_history = []
        self.alerts_history = []
        self.dashboard_data = {}
        
        # Alert thresholds
        self.thresholds = {
            "cpu_warning": 75,
            "cpu_critical": 85,
            "memory_warning": 80,
            "memory_critical": 90,
            "disk_warning": 80,
            "disk_critical": 95,
            "agent_failure_rate": 0.15  # 15% failure rate
        }
        
        # Logs and data directories
        self.logs_dir = self.project_root / "logs" / "monitoring"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor_log = self.logs_dir / "activation-monitor.log"
        self.metrics_file = self.logs_dir / "metrics.jsonl"
        self.alerts_file = self.logs_dir / "alerts.jsonl"
    
    def log_event(self, message: str, level: str = "INFO"):
        """Log monitoring events"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        with open(self.monitor_log, "a") as f:
            f.write(log_entry + "\n")
        
        print(log_entry)
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        # Agent counts
        agent_statuses = self.get_agent_statuses()
        active_agents = len([a for a in agent_statuses if a.status == "running"])
        healthy_agents = len([a for a in agent_statuses if a.health == "healthy"])
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            network_io=network_io,
            active_agents=active_agents,
            healthy_agents=healthy_agents,
            total_containers=len(agent_statuses)
        )
    
    def get_agent_statuses(self) -> List[AgentStatus]:
        """Get detailed status of all SutazAI agents"""
        agent_statuses = []
        
        try:
            containers = self.docker_client.containers.list(
                all=True, filters={"name": "sutazai-"}
            )
            
            for container in containers:
                # Extract agent name
                agent_name = container.name.replace("sutazai-", "")
                
                # Get container stats (for CPU/memory usage)
                cpu_usage = 0.0
                memory_usage = 0.0
                
                if container.status == "running":
                    try:
                        stats = container.stats(stream=False)
                        # Calculate CPU percentage
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                   stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                      stats['precpu_stats']['system_cpu_usage']
                        if system_delta > 0:
                            cpu_usage = (cpu_delta / system_delta) * 100.0
                        
                        # Calculate memory percentage
                        memory_usage = (stats['memory_stats']['usage'] / 
                                      stats['memory_stats']['limit']) * 100.0
                    except:
                        pass
                
                # Get health status
                health_status = "unknown"
                health_info = container.attrs.get("State", {}).get("Health", {})
                if health_info:
                    health_status = health_info.get("Status", "unknown")
                elif container.status == "running":
                    health_status = "healthy"  # Assume healthy if running and no healthcheck
                
                # Calculate uptime
                created_time = datetime.fromisoformat(
                    container.attrs["Created"].replace("Z", "+00:00")
                )
                uptime = datetime.now(created_time.tzinfo) - created_time
                
                # Get restart count
                restart_count = container.attrs["RestartCount"]
                
                agent_status = AgentStatus(
                    name=agent_name,
                    status=container.status,
                    container_id=container.id[:12],
                    health=health_status,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    uptime=uptime,
                    restart_count=restart_count
                )
                
                agent_statuses.append(agent_status)
                
        except Exception as e:
            self.log_event(f"Error getting agent statuses: {e}", "ERROR")
        
        return agent_statuses
    
    def check_alerts(self, metrics: SystemMetrics, agent_statuses: List[AgentStatus]) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        timestamp = datetime.now().isoformat()
        
        # System resource alerts
        if metrics.cpu_percent >= self.thresholds["cpu_critical"]:
            alerts.append({
                "timestamp": timestamp,
                "level": "critical",
                "type": "cpu_usage",
                "message": f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                "value": metrics.cpu_percent,
                "threshold": self.thresholds["cpu_critical"]
            })
        elif metrics.cpu_percent >= self.thresholds["cpu_warning"]:
            alerts.append({
                "timestamp": timestamp,
                "level": "warning",
                "type": "cpu_usage", 
                "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                "value": metrics.cpu_percent,
                "threshold": self.thresholds["cpu_warning"]
            })
        
        if metrics.memory_percent >= self.thresholds["memory_critical"]:
            alerts.append({
                "timestamp": timestamp,
                "level": "critical",
                "type": "memory_usage",
                "message": f"Critical memory usage: {metrics.memory_percent:.1f}%",
                "value": metrics.memory_percent,
                "threshold": self.thresholds["memory_critical"]
            })
        elif metrics.memory_percent >= self.thresholds["memory_warning"]:
            alerts.append({
                "timestamp": timestamp,
                "level": "warning",
                "type": "memory_usage",
                "message": f"High memory usage: {metrics.memory_percent:.1f}%",
                "value": metrics.memory_percent,
                "threshold": self.thresholds["memory_warning"]
            })
        
        if metrics.disk_percent >= self.thresholds["disk_critical"]:
            alerts.append({
                "timestamp": timestamp,
                "level": "critical",
                "type": "disk_usage",
                "message": f"Critical disk usage: {metrics.disk_percent:.1f}%",
                "value": metrics.disk_percent,
                "threshold": self.thresholds["disk_critical"]
            })
        elif metrics.disk_percent >= self.thresholds["disk_warning"]:
            alerts.append({
                "timestamp": timestamp,
                "level": "warning",
                "type": "disk_usage",
                "message": f"High disk usage: {metrics.disk_percent:.1f}%",
                "value": metrics.disk_percent,
                "threshold": self.thresholds["disk_warning"]
            })
        
        # Agent health alerts
        if metrics.total_containers > 0:
            failure_rate = 1 - (metrics.healthy_agents / metrics.total_containers)
            if failure_rate >= self.thresholds["agent_failure_rate"]:
                alerts.append({
                    "timestamp": timestamp,
                    "level": "warning",
                    "type": "agent_failure_rate",
                    "message": f"High agent failure rate: {failure_rate:.1%}",
                    "value": failure_rate,
                    "threshold": self.thresholds["agent_failure_rate"]
                })
        
        # Individual agent alerts
        for agent in agent_statuses:
            if agent.status == "running" and agent.health == "unhealthy":
                alerts.append({
                    "timestamp": timestamp,
                    "level": "warning",
                    "type": "agent_unhealthy",
                    "message": f"Agent {agent.name} is unhealthy",
                    "agent": agent.name,
                    "container_id": agent.container_id
                })
            
            if agent.restart_count > 3:
                alerts.append({
                    "timestamp": timestamp,
                    "level": "warning",
                    "type": "agent_restart_loop",
                    "message": f"Agent {agent.name} has restarted {agent.restart_count} times",
                    "agent": agent.name,
                    "restart_count": agent.restart_count
                })
        
        return alerts
    
    def save_metrics(self, metrics: SystemMetrics):
        """Save metrics to JSONL file"""
        metrics_data = {
            "timestamp": metrics.timestamp.isoformat(),
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_available_gb": metrics.memory_available_gb,
            "disk_percent": metrics.disk_percent,
            "disk_free_gb": metrics.disk_free_gb,
            "network_io": metrics.network_io,
            "active_agents": metrics.active_agents,
            "healthy_agents": metrics.healthy_agents,
            "total_containers": metrics.total_containers
        }
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics_data) + "\n")
    
    def save_alerts(self, alerts: List[Dict]):
        """Save alerts to JSONL file"""
        for alert in alerts:
            with open(self.alerts_file, "a") as f:
                f.write(json.dumps(alert) + "\n")
    
    def update_dashboard_data(self, metrics: SystemMetrics, agent_statuses: List[AgentStatus]):
        """Update dashboard data structure"""
        self.dashboard_data = {
            "last_update": datetime.now().isoformat(),
            "system": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_available_gb": round(metrics.memory_available_gb, 2),
                "disk_percent": metrics.disk_percent,
                "disk_free_gb": round(metrics.disk_free_gb, 2)
            },
            "agents": {
                "total": metrics.total_containers,
                "active": metrics.active_agents,
                "healthy": metrics.healthy_agents,
                "utilization_rate": (metrics.active_agents / 137) * 100 if metrics.active_agents > 0 else 0
            },
            "agent_details": [
                {
                    "name": agent.name,
                    "status": agent.status,
                    "health": agent.health,
                    "cpu_usage": round(agent.cpu_usage, 2),
                    "memory_usage": round(agent.memory_usage, 2),
                    "uptime_hours": agent.uptime.total_seconds() / 3600,
                    "restart_count": agent.restart_count
                }
                for agent in agent_statuses
            ],
            "phase_progress": self.calculate_phase_progress(agent_statuses)
        }
        
        # Save dashboard data
        dashboard_file = self.project_root / "monitoring" / "dashboard_data.json"
        dashboard_file.parent.mkdir(exist_ok=True)
        
        with open(dashboard_file, 'w') as f:
            json.dump(self.dashboard_data, f, indent=2)
    
    def calculate_phase_progress(self, agent_statuses: List[AgentStatus]) -> Dict:
        """Calculate activation progress for each phase"""
        # Define phase agents (matching the orchestrator)
        phase_agents = {
            1: ["ai-system-architect", "deployment-automation-master", "mega-code-auditor",
                "system-optimizer-reorganizer", "hardware-resource-optimizer", "ollama-integration-specialist",
                "infrastructure-devops-manager", "ai-agent-orchestrator", "ai-senior-backend-developer",
                "ai-senior-frontend-developer", "testing-qa-validator", "document-knowledge-manager",
                "security-pentesting-specialist", "cicd-pipeline-orchestrator", "ai-system-validator"],
            2: ["garbage-collector-coordinator", "distributed-computing-architect", "edge-computing-optimizer",
                "container-orchestrator-k3s", "gpu-hardware-optimizer", "cpu-only-hardware-optimizer"],
            3: ["quantum-ai-researcher", "neuromorphic-computing-expert", "agentzero-coordinator"]
        }
        
        active_agents = {agent.name for agent in agent_statuses if agent.status == "running"}
        healthy_agents = {agent.name for agent in agent_statuses if agent.health == "healthy"}
        
        progress = {}
        for phase, agents in phase_agents.items():
            total = len(agents)
            active = len([a for a in agents if a in active_agents])
            healthy = len([a for a in agents if a in healthy_agents])
            
            progress[f"phase_{phase}"] = {
                "total_agents": total,
                "active_agents": active,
                "healthy_agents": healthy,
                "activation_rate": (active / total) * 100 if total > 0 else 0,
                "health_rate": (healthy / total) * 100 if total > 0 else 0
            }
        
        return progress
    
    def print_status_report(self, metrics: SystemMetrics, agent_statuses: List[AgentStatus]):
        """Print comprehensive status report"""
        print("\n" + "="*80)
        print(f"SUTAZAI AGENT ACTIVATION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # System metrics
        print(f"\n📊 SYSTEM METRICS")
        print(f"CPU Usage:    {metrics.cpu_percent:6.1f}%")
        print(f"Memory Usage: {metrics.memory_percent:6.1f}% ({metrics.memory_available_gb:.1f}GB available)")
        print(f"Disk Usage:   {metrics.disk_percent:6.1f}% ({metrics.disk_free_gb:.1f}GB free)")
        
        # Agent summary
        print(f"\n🤖 AGENT STATUS")
        print(f"Total Agents:   {metrics.total_containers:3d}")
        print(f"Active Agents:  {metrics.active_agents:3d}")
        print(f"Healthy Agents: {metrics.healthy_agents:3d}")
        print(f"Utilization:    {(metrics.active_agents/137)*100:5.1f}%")
        
        # Phase progress
        phase_progress = self.calculate_phase_progress(agent_statuses)
        print(f"\n📈 PHASE PROGRESS")
        for phase_key, progress in phase_progress.items():
            phase_num = phase_key.split('_')[1]
            print(f"Phase {phase_num}: {progress['active_agents']:2d}/{progress['total_agents']:2d} active "
                  f"({progress['activation_rate']:5.1f}%), {progress['healthy_agents']:2d} healthy "
                  f"({progress['health_rate']:5.1f}%)")
        
        # Recent alerts
        recent_alerts = [alert for alert in self.alerts_history[-5:] 
                        if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(minutes=5)]
        
        if recent_alerts:
            print(f"\n⚠️  RECENT ALERTS")
            for alert in recent_alerts:
                level_icon = "🔴" if alert['level'] == 'critical' else "🟡"
                print(f"{level_icon} {alert['message']}")
        
        print("="*80)
    
    async def monitoring_loop(self, interval: int = 15, dashboard: bool = False, alerts: bool = True):
        """Main monitoring loop"""
        self.monitoring_active = True
        self.log_event("Starting agent activation monitoring")
        
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self.get_system_metrics()
                agent_statuses = self.get_agent_statuses()
                
                # Store metrics
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 1000:  # Keep last 1000 metrics
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Save metrics to file
                self.save_metrics(metrics)
                
                # Check for alerts
                if alerts:
                    current_alerts = self.check_alerts(metrics, agent_statuses)
                    if current_alerts:
                        self.alerts_history.extend(current_alerts)
                        self.save_alerts(current_alerts)
                        
                        # Log critical alerts immediately
                        for alert in current_alerts:
                            if alert['level'] == 'critical':
                                self.log_event(alert['message'], "CRITICAL")
                
                # Update dashboard
                if dashboard:
                    self.update_dashboard_data(metrics, agent_statuses)
                    self.print_status_report(metrics, agent_statuses)
                
                # Wait for next cycle
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                self.monitoring_active = False
                break
            except Exception as e:
                self.log_event(f"Error in monitoring loop: {e}", "ERROR")
                await asyncio.sleep(interval)
        
        self.log_event("Agent activation monitoring stopped")
    
    async def run_monitoring(self, interval: int = 15, duration: Optional[int] = None, 
                           dashboard: bool = False, alerts: bool = True):
        """Run monitoring for specified duration or indefinitely"""
        if duration:
            # Run for specified duration
            monitoring_task = asyncio.create_task(
                self.monitoring_loop(interval, dashboard, alerts)
            )
            
            await asyncio.sleep(duration)
            self.monitoring_active = False
            
            try:
                await asyncio.wait_for(monitoring_task, timeout=10)
            except asyncio.TimeoutError:
                monitoring_task.cancel()
        else:
            # Run indefinitely
            await self.monitoring_loop(interval, dashboard, alerts)

async def main():
    parser = argparse.ArgumentParser(description="Monitor AI agent activation progress")
    parser.add_argument("--interval", type=int, default=15,
                       help="Monitoring interval in seconds (default: 15)")
    parser.add_argument("--duration", type=int, default=None,
                       help="Monitoring duration in seconds (default: infinite)")
    parser.add_argument("--dashboard", action="store_true",
                       help="Enable dashboard display")
    parser.add_argument("--no-alerts", action="store_true",
                       help="Disable alert checking")
    parser.add_argument("--project-root", default="/opt/sutazaiapp",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    monitor = AgentActivationMonitor(args.project_root)
    
    try:
        await monitor.run_monitoring(
            interval=args.interval,
            duration=args.duration,
            dashboard=args.dashboard,
            alerts=not args.no_alerts
        )
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Monitoring failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))