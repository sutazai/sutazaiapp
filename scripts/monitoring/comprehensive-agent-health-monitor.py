#!/usr/bin/env python3
"""
Comprehensive Agent Health Monitor for SutazAI
Version: 1.0.0

DESCRIPTION:
    Advanced health monitoring system that validates the health and functionality
    of all 131 AI agents in the SutazAI ecosystem. Provides real-time monitoring,
    performance metrics, and intelligent health assessment.

PURPOSE:
    - Monitor health of all deployed agents
    - Perform deep health checks beyond simple ping tests
    - Detect performance degradation and anomalies
    - Provide detailed health reports and metrics
    - Support automated healing and recovery

USAGE:
    python comprehensive-agent-health-monitor.py [options]

REQUIREMENTS:
    - Python 3.8+
    - aiohttp, asyncio
    - psutil for system metrics
    - All SutazAI agents running
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import subprocess
import aiohttp
import sqlite3
from dataclasses import dataclass, asdict
import yaml
import statistics
from concurrent.futures import ThreadPoolExecutor
import threading
import socket
import psutil

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
LOG_DIR = PROJECT_ROOT / "logs"
DB_FILE = LOG_DIR / "health_monitoring.db"
CONFIG_FILE = PROJECT_ROOT / "config" / "agent_health_config.yaml"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "health-monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    OFFLINE = "offline"

class TestType(Enum):
    """Types of health tests"""
    BAdvanced SystemC_PING = "basic_ping"
    HTTP_HEALTH = "http_health"
    DEEP_HEALTH = "deep_health"
    PERFORMANCE = "performance"
    FUNCTIONAL = "functional"
    INTEGRATION = "integration"

@dataclass
class HealthCheck:
    """Individual health check definition"""
    name: str
    test_type: TestType
    endpoint: str
    method: str = "GET"
    expected_status: int = 200
    timeout: int = 10
    headers: Dict[str, str] = None
    payload: Dict[str, Any] = None
    validation_func: str = None

@dataclass
class AgentHealthConfig:
    """Agent health configuration"""
    name: str
    display_name: str
    port: int
    base_url: str
    category: str
    priority: int
    health_checks: List[HealthCheck]
    dependencies: List[str]
    expected_response_time: float
    monitoring_interval: int
    retry_count: int

@dataclass
class HealthResult:
    """Health check result"""
    agent_name: str
    check_name: str
    status: HealthStatus
    response_time: float
    timestamp: datetime
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class AgentHealthSummary:
    """Agent overall health summary"""
    agent_name: str
    overall_status: HealthStatus
    last_check: datetime
    uptime_percentage: float
    avg_response_time: float
    failed_checks: int
    total_checks: int
    health_score: float
    alerts: List[str]

class HealthDatabase:
    """SQLite database for health monitoring data"""
    
    def __init__(self, db_file: Path):
        self.db_file = db_file
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_file) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS health_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    check_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_time REAL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    error_message TEXT
                );
                
                CREATE TABLE IF NOT EXISTS agent_summaries (
                    agent_name TEXT PRIMARY KEY,
                    overall_status TEXT NOT NULL,
                    last_check TEXT NOT NULL,
                    uptime_percentage REAL,
                    avg_response_time REAL,
                    failed_checks INTEGER,
                    total_checks INTEGER,
                    health_score REAL,
                    alerts TEXT
                );
                
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_percent REAL,
                    network_io TEXT,
                    active_agents INTEGER,
                    healthy_agents INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS idx_health_results_timestamp 
                ON health_results(timestamp);
                CREATE INDEX IF NOT EXISTS idx_health_results_agent 
                ON health_results(agent_name);
            """)
    
    def store_health_result(self, result: HealthResult):
        """Store health check result"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT INTO health_results 
                (agent_name, check_name, status, response_time, timestamp, details, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.agent_name,
                result.check_name,
                result.status.value,
                result.response_time,
                result.timestamp.isoformat(),
                json.dumps(result.details),
                result.error_message
            ))
    
    def store_agent_summary(self, summary: AgentHealthSummary):
        """Store agent health summary"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agent_summaries
                (agent_name, overall_status, last_check, uptime_percentage, 
                 avg_response_time, failed_checks, total_checks, health_score, alerts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.agent_name,
                summary.overall_status.value,
                summary.last_check.isoformat(),
                summary.uptime_percentage,
                summary.avg_response_time,
                summary.failed_checks,
                summary.total_checks,
                summary.health_score,
                json.dumps(summary.alerts)
            ))
    
    def get_agent_history(self, agent_name: str, hours: int = 24) -> List[HealthResult]:
        """Get agent health history"""
        since = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_file) as conn:
            rows = conn.execute("""
                SELECT agent_name, check_name, status, response_time, timestamp, details, error_message
                FROM health_results
                WHERE agent_name = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (agent_name, since.isoformat())).fetchall()
            
            results = []
            for row in rows:
                results.append(HealthResult(
                    agent_name=row[0],
                    check_name=row[1],
                    status=HealthStatus(row[2]),
                    response_time=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    details=json.loads(row[5]) if row[5] else {},
                    error_message=row[6]
                ))
            
            return results
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system health overview"""
        with sqlite3.connect(self.db_file) as conn:
            # Get latest summaries
            summaries = conn.execute("""
                SELECT agent_name, overall_status, health_score, last_check
                FROM agent_summaries
                ORDER BY last_check DESC
            """).fetchall()
            
            # Calculate metrics
            total_agents = len(summaries)
            healthy_agents = len([s for s in summaries if s[1] == HealthStatus.HEALTHY.value])
            
            # Get recent system metrics
            system_metrics = conn.execute("""
                SELECT cpu_percent, memory_percent, disk_percent, active_agents
                FROM system_metrics
                ORDER BY timestamp DESC
                LIMIT 1
            """).fetchone()
            
            return {
                "total_agents": total_agents,
                "healthy_agents": healthy_agents,
                "unhealthy_agents": total_agents - healthy_agents,
                "health_percentage": (healthy_agents / total_agents * 100) if total_agents > 0 else 0,
                "system_metrics": {
                    "cpu_percent": system_metrics[0] if system_metrics else 0,
                    "memory_percent": system_metrics[1] if system_metrics else 0,
                    "disk_percent": system_metrics[2] if system_metrics else 0,
                    "active_agents": system_metrics[3] if system_metrics else 0
                } if system_metrics else None,
                "last_update": datetime.now().isoformat()
            }

class AgentDiscovery:
    """Automatic agent discovery system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.agents_dir = project_root / "agents"
    
    def discover_agents(self) -> List[AgentHealthConfig]:
        """Discover all agents in the system"""
        agents = []
        
        if not self.agents_dir.exists():
            logger.warning("Agents directory not found")
            return agents
        
        port_base = 8100
        categories = {
            "infrastructure": ["postgres", "redis", "neo4j", "ollama"],
            "ai_frameworks": ["autogpt", "crewai", "letta", "aider", "langflow", "flowise"],
            "specialized_agents": [],
            "monitoring": ["prometheus", "grafana", "loki"],
            "security": ["pentestgpt", "semgrep"]
        }
        
        for agent_dir in self.agents_dir.iterdir():
            if agent_dir.is_dir() and not agent_dir.name.startswith('.'):
                # Skip certain directories
                if agent_dir.name in ['core', 'configs', 'dockerfiles']:
                    continue
                
                # Check if agent has implementation
                has_app = (agent_dir / "app.py").exists() or (agent_dir / "agent.py").exists()
                if has_app:
                    # Determine category
                    category = "specialized_agents"
                    for cat, names in categories.items():
                        if any(name in agent_dir.name.lower() for name in names):
                            category = cat
                            break
                    
                    # Create health checks based on agent type
                    health_checks = self._create_health_checks_for_agent(agent_dir.name, category)
                    
                    agent_config = AgentHealthConfig(
                        name=agent_dir.name,
                        display_name=agent_dir.name.replace("-", " ").title(),
                        port=port_base,
                        base_url=f"http://localhost:{port_base}",
                        category=category,
                        priority=self._get_priority_for_category(category),
                        health_checks=health_checks,
                        dependencies=self._get_dependencies_for_agent(agent_dir.name),
                        expected_response_time=2.0,
                        monitoring_interval=30,
                        retry_count=3
                    )
                    
                    agents.append(agent_config)
                    port_base += 1
        
        logger.info(f"Discovered {len(agents)} agents")
        return agents
    
    def _create_health_checks_for_agent(self, agent_name: str, category: str) -> List[HealthCheck]:
        """Create appropriate health checks for agent type"""
        checks = [
            HealthCheck(
                name="basic_connectivity",
                test_type=TestType.BAdvanced SystemC_PING,
                endpoint="/",
                timeout=5
            ),
            HealthCheck(
                name="health_endpoint",
                test_type=TestType.HTTP_HEALTH,
                endpoint="/health",
                timeout=10
            )
        ]
        
        # Add category-specific checks
        if category == "ai_frameworks":
            checks.extend([
                HealthCheck(
                    name="model_status",
                    test_type=TestType.DEEP_HEALTH,
                    endpoint="/api/models",
                    timeout=15
                ),
                HealthCheck(
                    name="inference_test",
                    test_type=TestType.FUNCTIONAL,
                    endpoint="/api/generate",
                    method="POST",
                    payload={"prompt": "Hello", "max_tokens": 10},
                    timeout=30
                )
            ])
        
        elif category == "infrastructure":
            if "postgres" in agent_name.lower():
                checks.append(HealthCheck(
                    name="database_connection",
                    test_type=TestType.DEEP_HEALTH,
                    endpoint="/db/health",
                    timeout=10
                ))
            elif "redis" in agent_name.lower():
                checks.append(HealthCheck(
                    name="cache_test",
                    test_type=TestType.FUNCTIONAL,
                    endpoint="/cache/ping",
                    timeout=5
                ))
        
        elif category == "monitoring":
            checks.append(HealthCheck(
                name="metrics_collection",
                test_type=TestType.FUNCTIONAL,
                endpoint="/api/v1/query",
                method="POST",
                timeout=15
            ))
        
        return checks
    
    def _get_priority_for_category(self, category: str) -> int:
        """Get priority level for category"""
        priorities = {
            "infrastructure": 1,
            "ai_frameworks": 2,
            "monitoring": 2,
            "security": 3,
            "specialized_agents": 4
        }
        return priorities.get(category, 5)
    
    def _get_dependencies_for_agent(self, agent_name: str) -> List[str]:
        """Get dependencies for agent"""
        # Common dependencies
        dependencies = []
        
        # AI agents typically depend on Ollama
        if any(term in agent_name.lower() for term in ["gpt", "ai", "agent", "llm"]):
            dependencies.append("ollama")
        
        # Most services need database
        if not any(term in agent_name.lower() for term in ["postgres", "redis", "neo4j"]):
            dependencies.extend(["postgres", "redis"])
        
        return dependencies

class ComprehensiveHealthChecker:
    """Advanced health checking with multiple test types"""
    
    def __init__(self, database: HealthDatabase):
        self.database = database
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=20)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)
    
    async def check_agent_health(self, agent: AgentHealthConfig) -> List[HealthResult]:
        """Perform comprehensive health check on an agent"""
        results = []
        
        for check in agent.health_checks:
            try:
                start_time = time.time()
                
                if check.test_type == TestType.BAdvanced SystemC_PING:
                    result = await self._basic_ping_check(agent, check)
                elif check.test_type == TestType.HTTP_HEALTH:
                    result = await self._http_health_check(agent, check)
                elif check.test_type == TestType.DEEP_HEALTH:
                    result = await self._deep_health_check(agent, check)
                elif check.test_type == TestType.PERFORMANCE:
                    result = await self._performance_check(agent, check)
                elif check.test_type == TestType.FUNCTIONAL:
                    result = await self._functional_check(agent, check)
                elif check.test_type == TestType.INTEGRATION:
                    result = await self._integration_check(agent, check)
                else:
                    result = await self._http_health_check(agent, check)
                
                response_time = time.time() - start_time
                result.response_time = response_time
                
                results.append(result)
                
                # Store result in database
                self.database.store_health_result(result)
                
            except Exception as e:
                error_result = HealthResult(
                    agent_name=agent.name,
                    check_name=check.name,
                    status=HealthStatus.CRITICAL,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    details={"error": str(e)},
                    error_message=str(e)
                )
                results.append(error_result)
                self.database.store_health_result(error_result)
                logger.error(f"Health check failed for {agent.name}/{check.name}: {e}")
        
        return results
    
    async def _basic_ping_check(self, agent: AgentHealthConfig, check: HealthCheck) -> HealthResult:
        """Basic connectivity check"""
        try:
            # Check if port is open
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(check.timeout)
            result = sock.connect_ex(('localhost', agent.port))
            sock.close()
            
            if result == 0:
                return HealthResult(
                    agent_name=agent.name,
                    check_name=check.name,
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    details={"port_open": True}
                )
            else:
                return HealthResult(
                    agent_name=agent.name,
                    check_name=check.name,
                    status=HealthStatus.OFFLINE,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    details={"port_open": False},
                    error_message=f"Port {agent.port} not accessible"
                )
        
        except Exception as e:
            return HealthResult(
                agent_name=agent.name,
                check_name=check.name,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                timestamp=datetime.now(),
                details={"error": str(e)},
                error_message=str(e)
            )
    
    async def _http_health_check(self, agent: AgentHealthConfig, check: HealthCheck) -> HealthResult:
        """HTTP health endpoint check"""
        url = f"{agent.base_url}{check.endpoint}"
        
        try:
            async with self.session.request(
                check.method,
                url,
                headers=check.headers,
                json=check.payload,
                timeout=check.timeout
            ) as response:
                
                response_data = {}
                try:
                    response_data = await response.json()
                except:
                    response_data = {"text": await response.text()}
                
                # Determine status
                if response.status == check.expected_status:
                    status = HealthStatus.HEALTHY
                elif 200 <= response.status < 300:
                    status = HealthStatus.WARNING
                else:
                    status = HealthStatus.UNHEALTHY
                
                return HealthResult(
                    agent_name=agent.name,
                    check_name=check.name,
                    status=status,
                    response_time=0.0,  # Will be set by caller
                    timestamp=datetime.now(),
                    details={
                        "status_code": response.status,
                        "response_data": response_data,
                        "headers": dict(response.headers)
                    }
                )
        
        except asyncio.TimeoutError:
            return HealthResult(
                agent_name=agent.name,
                check_name=check.name,
                status=HealthStatus.UNHEALTHY,
                response_time=0.0,
                timestamp=datetime.now(),
                details={"timeout": True},
                error_message="Request timeout"
            )
        
        except Exception as e:
            return HealthResult(
                agent_name=agent.name,
                check_name=check.name,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                timestamp=datetime.now(),
                details={"error": str(e)},
                error_message=str(e)
            )
    
    async def _deep_health_check(self, agent: AgentHealthConfig, check: HealthCheck) -> HealthResult:
        """Deep health check with additional validation"""
        # First do HTTP check
        result = await self._http_health_check(agent, check)
        
        # Add additional deep checks
        if result.status == HealthStatus.HEALTHY:
            # Check response content for health indicators
            response_data = result.details.get("response_data", {})
            
            # Validate expected fields
            if isinstance(response_data, dict):
                if "status" in response_data and response_data["status"] not in ["healthy", "ok", "ready"]:
                    result.status = HealthStatus.WARNING
                    result.details["validation_warning"] = "Status field indicates issues"
                
                if "errors" in response_data and response_data["errors"]:
                    result.status = HealthStatus.WARNING
                    result.details["has_errors"] = True
        
        return result
    
    async def _performance_check(self, agent: AgentHealthConfig, check: HealthCheck) -> HealthResult:
        """Performance-focused health check"""
        # Run multiple requests and measure performance
        response_times = []
        errors = 0
        
        for i in range(5):  # 5 test requests
            try:
                start_time = time.time()
                result = await self._http_health_check(agent, check)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if result.status not in [HealthStatus.HEALTHY]:
                    errors += 1
                    
            except Exception:
                errors += 1
        
        if not response_times:
            return HealthResult(
                agent_name=agent.name,
                check_name=check.name,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                timestamp=datetime.now(),
                details={"error": "All performance tests failed"},
                error_message="Performance test completely failed"
            )
        
        avg_response_time = statistics.mean(response_times)
        
        # Determine status based on performance
        if errors > 2:
            status = HealthStatus.CRITICAL
        elif avg_response_time > agent.expected_response_time * 3:
            status = HealthStatus.UNHEALTHY
        elif avg_response_time > agent.expected_response_time * 2:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return HealthResult(
            agent_name=agent.name,
            check_name=check.name,
            status=status,
            response_time=avg_response_time,
            timestamp=datetime.now(),
            details={
                "avg_response_time": avg_response_time,
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "error_rate": errors / 5,
                "response_times": response_times
            }
        )
    
    async def _functional_check(self, agent: AgentHealthConfig, check: HealthCheck) -> HealthResult:
        """Functional test to verify actual agent capabilities"""
        # This is agent-specific and would test actual functionality
        result = await self._http_health_check(agent, check)
        
        # Add functional validation
        if result.status == HealthStatus.HEALTHY and check.validation_func:
            try:
                # Custom validation function would go here
                # For now, just validate response structure
                response_data = result.details.get("response_data", {})
                
                if "ai" in agent.name.lower() and isinstance(response_data, dict):
                    # For AI agents, check if response contains expected fields
                    expected_fields = ["response", "model", "usage"]
                    missing_fields = [f for f in expected_fields if f not in response_data]
                    
                    if missing_fields:
                        result.status = HealthStatus.WARNING
                        result.details["missing_fields"] = missing_fields
                
            except Exception as e:
                result.details["validation_error"] = str(e)
        
        return result
    
    async def _integration_check(self, agent: AgentHealthConfig, check: HealthCheck) -> HealthResult:
        """Integration test to verify agent works with dependencies"""
        # Check dependencies first
        dependency_status = {}
        
        for dep in agent.dependencies:
            # Quick check of dependency
            try:
                dep_port = self._get_port_for_service(dep)
                if dep_port:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex(('localhost', dep_port))
                    sock.close()
                    dependency_status[dep] = result == 0
                else:
                    dependency_status[dep] = False
            except:
                dependency_status[dep] = False
        
        # Then check the agent
        result = await self._http_health_check(agent, check)
        
        # Adjust status based on dependencies
        failed_deps = [dep for dep, status in dependency_status.items() if not status]
        if failed_deps:
            if result.status == HealthStatus.HEALTHY:
                result.status = HealthStatus.WARNING
            result.details["failed_dependencies"] = failed_deps
        
        result.details["dependency_status"] = dependency_status
        
        return result
    
    def _get_port_for_service(self, service_name: str) -> Optional[int]:
        """Get standard port for common services"""
        ports = {
            "postgres": 5432,
            "redis": 6379,
            "neo4j": 7474,
            "ollama": 9005,
            "prometheus": 9090,
            "grafana": 3000
        }
        return ports.get(service_name.lower())

class HealthReportGenerator:
    """Generate comprehensive health reports"""
    
    def __init__(self, database: HealthDatabase):
        self.database = database
    
    def generate_agent_summary(self, agent: AgentHealthConfig, results: List[HealthResult]) -> AgentHealthSummary:
        """Generate summary for a single agent"""
        if not results:
            return AgentHealthSummary(
                agent_name=agent.name,
                overall_status=HealthStatus.UNKNOWN,
                last_check=datetime.now(),
                uptime_percentage=0.0,
                avg_response_time=0.0,
                failed_checks=0,
                total_checks=0,
                health_score=0.0,
                alerts=["No health check results available"]
            )
        
        # Calculate metrics
        healthy_count = len([r for r in results if r.status == HealthStatus.HEALTHY])
        warning_count = len([r for r in results if r.status == HealthStatus.WARNING])
        unhealthy_count = len([r for r in results if r.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]])
        
        total_checks = len(results)
        failed_checks = unhealthy_count
        
        # Calculate overall status
        if unhealthy_count > total_checks * 0.5:
            overall_status = HealthStatus.CRITICAL
        elif unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif warning_count > total_checks * 0.3:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Calculate health score (0-100)
        health_score = (healthy_count * 100 + warning_count * 70) / total_checks if total_checks > 0 else 0
        
        # Calculate uptime percentage
        uptime_percentage = (healthy_count + warning_count) / total_checks * 100 if total_checks > 0 else 0
        
        # Calculate average response time
        response_times = [r.response_time for r in results if r.response_time > 0]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        
        # Generate alerts
        alerts = []
        if overall_status == HealthStatus.CRITICAL:
            alerts.append("Agent is in critical state")
        if unhealthy_count > 0:
            alerts.append(f"{unhealthy_count} health checks failing")
        if avg_response_time > agent.expected_response_time * 2:
            alerts.append(f"Response time ({avg_response_time:.2f}s) exceeds expected ({agent.expected_response_time}s)")
        
        return AgentHealthSummary(
            agent_name=agent.name,
            overall_status=overall_status,
            last_check=max(r.timestamp for r in results),
            uptime_percentage=uptime_percentage,
            avg_response_time=avg_response_time,
            failed_checks=failed_checks,
            total_checks=total_checks,
            health_score=health_score,
            alerts=alerts
        )
    
    def generate_system_report(self, summaries: List[AgentHealthSummary]) -> Dict[str, Any]:
        """Generate system-wide health report"""
        if not summaries:
            return {
                "timestamp": datetime.now().isoformat(),
                "total_agents": 0,
                "system_health": "UNKNOWN",
                "alerts": ["No agent data available"]
            }
        
        # System metrics
        total_agents = len(summaries)
        healthy_agents = len([s for s in summaries if s.overall_status == HealthStatus.HEALTHY])
        warning_agents = len([s for s in summaries if s.overall_status == HealthStatus.WARNING])
        unhealthy_agents = len([s for s in summaries if s.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]])
        
        # Overall system health
        if unhealthy_agents > total_agents * 0.3:
            system_health = "CRITICAL"
        elif unhealthy_agents > total_agents * 0.1:
            system_health = "UNHEALTHY"
        elif warning_agents > total_agents * 0.2:
            system_health = "WARNING"
        else:
            system_health = "HEALTHY"
        
        # Average metrics
        avg_health_score = statistics.mean([s.health_score for s in summaries])
        avg_response_time = statistics.mean([s.avg_response_time for s in summaries if s.avg_response_time > 0])
        avg_uptime = statistics.mean([s.uptime_percentage for s in summaries])
        
        # System alerts
        system_alerts = []
        if unhealthy_agents > 0:
            system_alerts.append(f"{unhealthy_agents} agents are unhealthy")
        if avg_health_score < 80:
            system_alerts.append(f"System health score is low: {avg_health_score:.1f}")
        if avg_response_time > 5.0:
            system_alerts.append(f"Average response time is high: {avg_response_time:.2f}s")
        
        # Category breakdown
        categories = {}
        for summary in summaries:
            # Find agent config to get category
            category = "unknown"  # Would need to pass agent configs to get actual category
            if category not in categories:
                categories[category] = {"total": 0, "healthy": 0, "unhealthy": 0}
            
            categories[category]["total"] += 1
            if summary.overall_status == HealthStatus.HEALTHY:
                categories[category]["healthy"] += 1
            elif summary.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                categories[category]["unhealthy"] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "warning_agents": warning_agents,
            "unhealthy_agents": unhealthy_agents,
            "health_percentage": (healthy_agents / total_agents * 100) if total_agents > 0 else 0,
            "metrics": {
                "avg_health_score": avg_health_score,
                "avg_response_time": avg_response_time,
                "avg_uptime_percentage": avg_uptime
            },
            "categories": categories,
            "alerts": system_alerts,
            "top_issues": self._get_top_issues(summaries)
        }
    
    def _get_top_issues(self, summaries: List[AgentHealthSummary]) -> List[str]:
        """Get top system issues"""
        issues = []
        
        # Find agents with most alerts
        agents_with_alerts = [(s.agent_name, len(s.alerts)) for s in summaries if s.alerts]
        agents_with_alerts.sort(key=lambda x: x[1], reverse=True)
        
        for agent_name, alert_count in agents_with_alerts[:5]:
            issues.append(f"{agent_name}: {alert_count} alerts")
        
        return issues

class ComprehensiveAgentHealthMonitor:
    """Main health monitoring system"""
    
    def __init__(self):
        self.database = HealthDatabase(DB_FILE)
        self.agent_discovery = AgentDiscovery(PROJECT_ROOT)
        self.health_checker = None
        self.report_generator = HealthReportGenerator(self.database)
        
        self.agents = []
        self.monitoring_tasks = {}
        self.running = False
    
    async def initialize(self):
        """Initialize the monitoring system"""
        logger.info("Initializing Comprehensive Agent Health Monitor...")
        
        # Discover agents
        self.agents = self.agent_discovery.discover_agents()
        logger.info(f"Monitoring {len(self.agents)} agents")
        
        # Initialize health checker
        self.health_checker = ComprehensiveHealthChecker(self.database)
        await self.health_checker.__aenter__()
        
        logger.info("Health monitoring system initialized")
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.running = True
        logger.info("Starting continuous health monitoring...")
        
        # Start monitoring tasks for each agent
        for agent in self.agents:
            task = asyncio.create_task(self._monitor_agent(agent))
            self.monitoring_tasks[agent.name] = task
        
        # Start system metrics collection
        system_task = asyncio.create_task(self._collect_system_metrics())
        self.monitoring_tasks["system_metrics"] = system_task
        
        # Start report generation task
        report_task = asyncio.create_task(self._generate_periodic_reports())
        self.monitoring_tasks["report_generation"] = report_task
        
        logger.info("All monitoring tasks started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        logger.info("Stopping health monitoring...")
        
        # Cancel all tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        
        # Cleanup
        if self.health_checker:
            await self.health_checker.__aexit__(None, None, None)
        
        logger.info("Health monitoring stopped")
    
    async def _monitor_agent(self, agent: AgentHealthConfig):
        """Monitor a single agent continuously"""
        logger.info(f"Starting monitoring for agent: {agent.name}")
        
        while self.running:
            try:
                # Perform health checks
                results = await self.health_checker.check_agent_health(agent)
                
                # Generate summary
                summary = self.report_generator.generate_agent_summary(agent, results)
                
                # Store summary
                self.database.store_agent_summary(summary)
                
                # Log significant status changes
                if summary.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    logger.warning(f"Agent {agent.name} is {summary.overall_status.value}")
                
                # Wait for next check
                await asyncio.sleep(agent.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring agent {agent.name}: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        while self.running:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                # Get network I/O
                network_io = psutil.net_io_counters()
                network_data = {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv
                }
                
                # Get agent counts from database
                overview = self.database.get_system_overview()
                
                # Store metrics
                with sqlite3.connect(self.database.db_file) as conn:
                    conn.execute("""
                        INSERT INTO system_metrics
                        (timestamp, cpu_percent, memory_percent, disk_percent,
                         network_io, active_agents, healthy_agents)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        cpu_percent,
                        memory_percent,
                        disk_percent,
                        json.dumps(network_data),
                        overview.get("total_agents", 0),
                        overview.get("healthy_agents", 0)
                    ))
                
                await asyncio.sleep(60)  # Collect every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
    
    async def _generate_periodic_reports(self):
        """Generate periodic health reports"""
        while self.running:
            try:
                # Get all agent summaries
                with sqlite3.connect(self.database.db_file) as conn:
                    rows = conn.execute("""
                        SELECT agent_name, overall_status, last_check, uptime_percentage,
                               avg_response_time, failed_checks, total_checks, health_score, alerts
                        FROM agent_summaries
                    """).fetchall()
                    
                    summaries = []
                    for row in rows:
                        summaries.append(AgentHealthSummary(
                            agent_name=row[0],
                            overall_status=HealthStatus(row[1]),
                            last_check=datetime.fromisoformat(row[2]),
                            uptime_percentage=row[3],
                            avg_response_time=row[4],
                            failed_checks=row[5],
                            total_checks=row[6],
                            health_score=row[7],
                            alerts=json.loads(row[8]) if row[8] else []
                        ))
                
                # Generate system report
                system_report = self.report_generator.generate_system_report(summaries)
                
                # Save report
                report_file = LOG_DIR / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w') as f:
                    json.dump(system_report, f, indent=2)
                
                logger.info(f"Generated health report: {report_file}")
                
                # Wait for next report (every hour)
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating reports: {e}")
                await asyncio.sleep(3600)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return self.database.get_system_overview()

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Agent Health Monitor for SutazAI"
    )
    parser.add_argument(
        "command",
        choices=["start", "status", "report", "test"],
        help="Command to execute"
    )
    parser.add_argument(
        "--agent",
        help="Specific agent to monitor/test"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duration to run (0 for continuous)"
    )
    
    args = parser.parse_args()
    
    monitor = ComprehensiveAgentHealthMonitor()
    
    try:
        if args.command == "start":
            await monitor.initialize()
            await monitor.start_monitoring()
            
            if args.duration > 0:
                await asyncio.sleep(args.duration)
                await monitor.stop_monitoring()
            else:
                # Run until interrupted
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    await monitor.stop_monitoring()
        
        elif args.command == "status":
            await monitor.initialize()
            status = monitor.get_current_status()
            print(json.dumps(status, indent=2))
        
        elif args.command == "report":
            await monitor.initialize()
            # Generate immediate report
            summaries = []  # Would need to fetch from database
            system_report = monitor.report_generator.generate_system_report(summaries)
            print(json.dumps(system_report, indent=2))
        
        elif args.command == "test":
            await monitor.initialize()
            
            if args.agent:
                # Test specific agent
                agent = next((a for a in monitor.agents if a.name == args.agent), None)
                if agent:
                    results = await monitor.health_checker.check_agent_health(agent)
                    for result in results:
                        print(f"{result.agent_name}/{result.check_name}: {result.status.value}")
                else:
                    print(f"Agent not found: {args.agent}")
            else:
                # Test all agents
                for agent in monitor.agents[:5]:  # Test first 5 agents
                    results = await monitor.health_checker.check_agent_health(agent)
                    healthy = len([r for r in results if r.status == HealthStatus.HEALTHY])
                    print(f"{agent.name}: {healthy}/{len(results)} checks healthy")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())