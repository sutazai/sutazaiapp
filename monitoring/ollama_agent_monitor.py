#!/usr/bin/env python3
"""
Ollama Agent Monitoring System
Comprehensive monitoring for 131 agents with Ollama integration to prevent freezes
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp
import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
import sqlite3
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@dataclass
class AgentHealthMetrics:
    """Agent health metrics structure"""
    agent_name: str
    status: str
    uptime_seconds: float
    tasks_processed: int
    tasks_failed: int
    active_tasks: int
    avg_processing_time: float
    ollama_requests: int
    ollama_failures: int
    circuit_breaker_trips: int
    memory_usage_mb: float
    cpu_usage_percent: float
    last_heartbeat: datetime
    ollama_healthy: bool
    backend_healthy: bool
    model: str

@dataclass
class OllamaSystemMetrics:
    """System-wide Ollama metrics"""
    total_requests: int
    total_failures: int
    active_connections: int
    queue_depth: int
    avg_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    model_loading_time: float
    concurrent_limit: int
    timestamp: datetime

class OllamaAgentMonitor:
    """
    Advanced monitoring system for Ollama-integrated agents
    Provides real-time tracking and freeze prevention
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
        self.metrics_port = int(os.getenv('METRICS_PORT', '8091'))
        self.db_path = '/opt/sutazaiapp/monitoring/ollama_metrics.db'
        
        # Monitoring intervals
        self.health_check_interval = 30  # seconds
        self.system_metrics_interval = 10  # seconds
        self.agent_discovery_interval = 60  # seconds
        
        # Alerting thresholds
        self.max_response_time = 30.0  # seconds
        self.max_queue_depth = 50
        self.max_failure_rate = 0.1  # 10%
        self.max_memory_usage = 80.0  # percent
        self.max_cpu_usage = 90.0  # percent
        
        # Agent tracking
        self.known_agents: Dict[str, AgentHealthMetrics] = {}
        self.system_metrics: Optional[OllamaSystemMetrics] = None
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Database
        self._init_database()
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.logger.info("Ollama Agent Monitor initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        
        # Agent metrics
        self.agent_status_gauge = Gauge(
            'sutazai_agent_status',
            'Agent status (1=active, 0=inactive)',
            ['agent_name', 'model'],
            registry=self.registry
        )
        
        self.agent_tasks_processed = Counter(
            'sutazai_agent_tasks_processed_total',
            'Total tasks processed by agent',
            ['agent_name', 'model'],
            registry=self.registry
        )
        
        self.agent_tasks_failed = Counter(
            'sutazai_agent_tasks_failed_total',
            'Total tasks failed by agent',
            ['agent_name', 'model'],
            registry=self.registry
        )
        
        self.agent_processing_time = Histogram(
            'sutazai_agent_processing_time_seconds',
            'Task processing time distribution',
            ['agent_name', 'model'],
            registry=self.registry
        )
        
        self.agent_memory_usage = Gauge(
            'sutazai_agent_memory_usage_mb',
            'Agent memory usage in MB',
            ['agent_name'],
            registry=self.registry
        )
        
        self.agent_cpu_usage = Gauge(
            'sutazai_agent_cpu_usage_percent',
            'Agent CPU usage percentage',
            ['agent_name'],
            registry=self.registry
        )
        
        # Ollama-specific metrics
        self.ollama_requests_total = Counter(
            'sutazai_ollama_requests_total',
            'Total Ollama requests',
            ['agent_name', 'model', 'status'],
            registry=self.registry
        )
        
        self.ollama_response_time = Histogram(
            'sutazai_ollama_response_time_seconds',
            'Ollama response time distribution',
            ['model'],
            registry=self.registry
        )
        
        self.ollama_queue_depth = Gauge(
            'sutazai_ollama_queue_depth',
            'Current Ollama request queue depth',
            registry=self.registry
        )
        
        self.ollama_active_connections = Gauge(
            'sutazai_ollama_active_connections',
            'Active Ollama connections',
            registry=self.registry
        )
        
        self.circuit_breaker_trips = Counter(
            'sutazai_circuit_breaker_trips_total',
            'Circuit breaker trips',
            ['agent_name'],
            registry=self.registry
        )
        
        # System metrics
        self.system_memory_usage = Gauge(
            'sutazai_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'sutazai_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.freeze_risk_score = Gauge(
            'sutazai_freeze_risk_score',
            'System freeze risk score (0-100)',
            registry=self.registry
        )
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    uptime_seconds REAL,
                    tasks_processed INTEGER,
                    tasks_failed INTEGER,
                    active_tasks INTEGER,
                    avg_processing_time REAL,
                    ollama_requests INTEGER,
                    ollama_failures INTEGER,
                    circuit_breaker_trips INTEGER,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    ollama_healthy BOOLEAN,
                    backend_healthy BOOLEAN,
                    model TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_requests INTEGER,
                    total_failures INTEGER,
                    active_connections INTEGER,
                    queue_depth INTEGER,
                    avg_response_time REAL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    model_loading_time REAL,
                    concurrent_limit INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    agent_name TEXT,
                    metric_value REAL,
                    threshold REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_agent_metrics_name_time ON agent_metrics(agent_name, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_time ON system_metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved, timestamp)')
    
    async def start(self):
        """Start the monitoring system"""
        self.logger.info("Starting Ollama Agent Monitor...")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Start Prometheus metrics server
        start_http_server(self.metrics_port, registry=self.registry)
        self.logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._agent_discovery_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._system_monitoring_loop()),
            asyncio.create_task(self._freeze_prevention_loop()),
            asyncio.create_task(self._alert_processing_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Monitoring tasks cancelled")
        except Exception as e:
            self.logger.error(f"Error in monitoring tasks: {e}")
            raise
    
    async def stop(self):
        """Stop the monitoring system"""
        if self.session:
            await self.session.close()
        self.logger.info("Ollama Agent Monitor stopped")
    
    async def _agent_discovery_loop(self):
        """Discover and track active agents"""
        while True:
            try:
                await self._discover_agents()
                await asyncio.sleep(self.agent_discovery_interval)
            except Exception as e:
                self.logger.error(f"Agent discovery error: {e}")
                await asyncio.sleep(30)
    
    async def _discover_agents(self):
        """Discover active agents from the coordinator"""
        try:
            async with self.session.get(f"{self.backend_url}/api/agents/list") as response:
                if response.status == 200:
                    agents = await response.json()
                    
                    # Update known agents
                    for agent_data in agents:
                        agent_name = agent_data.get('agent_name')
                        if agent_name:
                            # Get detailed health info
                            await self._fetch_agent_health(agent_name)
                    
                    self.logger.debug(f"Discovered {len(agents)} agents")
                else:
                    self.logger.warning(f"Failed to fetch agents list: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error discovering agents: {e}")
    
    async def _fetch_agent_health(self, agent_name: str):
        """Fetch health metrics for a specific agent"""
        try:
            # Try to get health directly from agent
            agent_port = 8000 + hash(agent_name) % 1000  # Simple port mapping
            agent_url = f"http://localhost:{agent_port}/health"
            
            async with self.session.get(agent_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    await self._process_agent_health(health_data)
                    
        except Exception as e:
            self.logger.debug(f"Could not fetch health for {agent_name}: {e}")
            # Fallback to coordinator endpoint
            try:
                async with self.session.get(f"{self.backend_url}/api/agents/{agent_name}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        await self._process_agent_health(health_data)
            except Exception as e2:
                self.logger.debug(f"Coordinator health fetch failed for {agent_name}: {e2}")
    
    async def _process_agent_health(self, health_data: Dict[str, Any]):
        """Process agent health data and update metrics"""
        try:
            agent_name = health_data.get('agent_name', 'unknown')
            
            # Create metrics object
            metrics = AgentHealthMetrics(
                agent_name=agent_name,
                status=health_data.get('status', 'unknown'),
                uptime_seconds=health_data.get('uptime_seconds', 0),
                tasks_processed=health_data.get('tasks_processed', 0),
                tasks_failed=health_data.get('tasks_failed', 0),
                active_tasks=health_data.get('active_tasks', 0),
                avg_processing_time=health_data.get('avg_processing_time', 0),
                ollama_requests=health_data.get('ollama_requests', 0),
                ollama_failures=health_data.get('ollama_failures', 0),
                circuit_breaker_trips=health_data.get('circuit_breaker_trips', 0),
                memory_usage_mb=health_data.get('memory_usage_mb', 0),
                cpu_usage_percent=health_data.get('cpu_usage_percent', 0),
                last_heartbeat=datetime.utcnow(),
                ollama_healthy=health_data.get('ollama_healthy', False),
                backend_healthy=health_data.get('backend_healthy', False),
                model=health_data.get('model', 'unknown')
            )
            
            # Update tracking
            old_metrics = self.known_agents.get(agent_name)
            self.known_agents[agent_name] = metrics
            
            # Update Prometheus metrics
            await self._update_prometheus_metrics(metrics, old_metrics)
            
            # Store in database
            await self._store_agent_metrics(metrics)
            
            # Check for alerts
            await self._check_agent_alerts(metrics)
            
        except Exception as e:
            self.logger.error(f"Error processing agent health: {e}")
    
    async def _update_prometheus_metrics(self, metrics: AgentHealthMetrics, old_metrics: Optional[AgentHealthMetrics]):
        """Update Prometheus metrics with latest agent data"""
        labels = {'agent_name': metrics.agent_name, 'model': metrics.model}
        
        # Status (1 for active, 0 for inactive)
        status_value = 1 if metrics.status == 'active' else 0
        self.agent_status_gauge.labels(**labels).set(status_value)
        
        # Task counters (increment by difference)
        if old_metrics:
            tasks_diff = metrics.tasks_processed - old_metrics.tasks_processed
            if tasks_diff > 0:
                self.agent_tasks_processed.labels(**labels).inc(tasks_diff)
            
            failures_diff = metrics.tasks_failed - old_metrics.tasks_failed
            if failures_diff > 0:
                self.agent_tasks_failed.labels(**labels).inc(failures_diff)
            
            ollama_diff = metrics.ollama_requests - old_metrics.ollama_requests
            if ollama_diff > 0:
                self.ollama_requests_total.labels(
                    agent_name=metrics.agent_name,
                    model=metrics.model,
                    status='success'
                ).inc(ollama_diff - (metrics.ollama_failures - old_metrics.ollama_failures))
                
                self.ollama_requests_total.labels(
                    agent_name=metrics.agent_name,
                    model=metrics.model,
                    status='failure'
                ).inc(metrics.ollama_failures - old_metrics.ollama_failures)
            
            cb_trips_diff = metrics.circuit_breaker_trips - old_metrics.circuit_breaker_trips
            if cb_trips_diff > 0:
                self.circuit_breaker_trips.labels(agent_name=metrics.agent_name).inc(cb_trips_diff)
        
        # Gauges
        self.agent_memory_usage.labels(agent_name=metrics.agent_name).set(metrics.memory_usage_mb)
        self.agent_cpu_usage.labels(agent_name=metrics.agent_name).set(metrics.cpu_usage_percent)
        
        # Processing time histogram
        if metrics.avg_processing_time > 0:
            self.agent_processing_time.labels(**labels).observe(metrics.avg_processing_time)
    
    async def _store_agent_metrics(self, metrics: AgentHealthMetrics):
        """Store agent metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO agent_metrics (
                        agent_name, status, uptime_seconds, tasks_processed, tasks_failed,
                        active_tasks, avg_processing_time, ollama_requests, ollama_failures,
                        circuit_breaker_trips, memory_usage_mb, cpu_usage_percent,
                        ollama_healthy, backend_healthy, model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.agent_name, metrics.status, metrics.uptime_seconds,
                    metrics.tasks_processed, metrics.tasks_failed, metrics.active_tasks,
                    metrics.avg_processing_time, metrics.ollama_requests, metrics.ollama_failures,
                    metrics.circuit_breaker_trips, metrics.memory_usage_mb, metrics.cpu_usage_percent,
                    metrics.ollama_healthy, metrics.backend_healthy, metrics.model
                ))
        except Exception as e:
            self.logger.error(f"Error storing agent metrics: {e}")
    
    async def _health_monitoring_loop(self):
        """Monitor agent health continuously"""
        while True:
            try:
                await self._check_agent_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _check_agent_health(self):
        """Check health of all known agents"""
        current_time = datetime.utcnow()
        stale_agents = []
        
        for agent_name, metrics in self.known_agents.items():
            # Check for stale heartbeats
            time_since_heartbeat = (current_time - metrics.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > 120:  # 2 minutes
                stale_agents.append(agent_name)
                await self._create_alert(
                    'stale_heartbeat',
                    'warning',
                    f"Agent {agent_name} has not sent heartbeat for {time_since_heartbeat:.0f} seconds",
                    agent_name,
                    time_since_heartbeat,
                    120
                )
        
        # Remove stale agents
        for agent_name in stale_agents:
            self.logger.warning(f"Removing stale agent: {agent_name}")
            del self.known_agents[agent_name]
    
    async def _system_monitoring_loop(self):
        """Monitor system-wide metrics"""
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.system_metrics_interval)
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # System resource usage
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.system_memory_usage.set(memory_percent)
            self.system_cpu_usage.set(cpu_percent)
            
            # Ollama-specific metrics
            ollama_metrics = await self._collect_ollama_metrics()
            if ollama_metrics:
                self.system_metrics = ollama_metrics
                
                self.ollama_queue_depth.set(ollama_metrics.queue_depth)
                self.ollama_active_connections.set(ollama_metrics.active_connections)
                
                # Store system metrics
                await self._store_system_metrics(ollama_metrics)
                
                # Check system alerts
                await self._check_system_alerts(ollama_metrics, memory_percent, cpu_percent)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_ollama_metrics(self) -> Optional[OllamaSystemMetrics]:
        """Collect Ollama-specific system metrics"""
        try:
            # Try to get Ollama API metrics
            async with self.session.get(f"{self.ollama_url}/api/ps") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Calculate aggregate metrics from all agents
                    total_requests = sum(m.ollama_requests for m in self.known_agents.values())
                    total_failures = sum(m.ollama_failures for m in self.known_agents.values())
                    active_tasks = sum(m.active_tasks for m in self.known_agents.values())
                    
                    # Estimate queue depth from active tasks and concurrent limit
                    concurrent_limit = int(os.getenv('OLLAMA_NUM_PARALLEL', '2'))
                    queue_depth = max(0, active_tasks - concurrent_limit)
                    
                    # Calculate average response time
                    response_times = [m.avg_processing_time for m in self.known_agents.values() if m.avg_processing_time > 0]
                    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                    
                    return OllamaSystemMetrics(
                        total_requests=total_requests,
                        total_failures=total_failures,
                        active_connections=len([m for m in self.known_agents.values() if m.status == 'active']),
                        queue_depth=queue_depth,
                        avg_response_time=avg_response_time,
                        memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
                        cpu_usage_percent=psutil.cpu_percent(),
                        model_loading_time=0.0,  # TODO: Implement model loading time tracking
                        concurrent_limit=concurrent_limit,
                        timestamp=datetime.utcnow()
                    )
                    
        except Exception as e:
            self.logger.debug(f"Could not collect Ollama metrics: {e}")
            return None
    
    async def _store_system_metrics(self, metrics: OllamaSystemMetrics):
        """Store system metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO system_metrics (
                        total_requests, total_failures, active_connections, queue_depth,
                        avg_response_time, memory_usage_mb, cpu_usage_percent,
                        model_loading_time, concurrent_limit
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.total_requests, metrics.total_failures, metrics.active_connections,
                    metrics.queue_depth, metrics.avg_response_time, metrics.memory_usage_mb,
                    metrics.cpu_usage_percent, metrics.model_loading_time, metrics.concurrent_limit
                ))
        except Exception as e:
            self.logger.error(f"Error storing system metrics: {e}")
    
    async def _freeze_prevention_loop(self):
        """Monitor for conditions that could cause system freezes"""
        while True:
            try:
                freeze_risk = await self._calculate_freeze_risk()
                self.freeze_risk_score.set(freeze_risk)
                
                if freeze_risk > 80:
                    await self._create_alert(
                        'high_freeze_risk',
                        'critical',
                        f"System freeze risk is high: {freeze_risk:.1f}%",
                        None,
                        freeze_risk,
                        80
                    )
                    await self._take_preventive_action()
                
                await asyncio.sleep(self.system_metrics_interval)
            except Exception as e:
                self.logger.error(f"Freeze prevention error: {e}")
                await asyncio.sleep(30)
    
    async def _calculate_freeze_risk(self) -> float:
        """Calculate system freeze risk score (0-100)"""
        risk_factors = []
        
        # Memory usage risk
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            risk_factors.append(30)
        elif memory_percent > 80:
            risk_factors.append(20)
        elif memory_percent > 70:
            risk_factors.append(10)
        
        # CPU usage risk
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > 95:
            risk_factors.append(25)
        elif cpu_percent > 85:
            risk_factors.append(15)
        elif cpu_percent > 75:
            risk_factors.append(5)
        
        # Queue depth risk
        if self.system_metrics:
            queue_depth = self.system_metrics.queue_depth
            if queue_depth > 100:
                risk_factors.append(20)
            elif queue_depth > 50:
                risk_factors.append(15)
            elif queue_depth > 25:
                risk_factors.append(10)
        
        # Circuit breaker trips risk
        recent_trips = sum(m.circuit_breaker_trips for m in self.known_agents.values())
        if recent_trips > 10:
            risk_factors.append(20)
        elif recent_trips > 5:
            risk_factors.append(10)
        
        # Stale agents risk
        current_time = datetime.utcnow()
        stale_count = sum(
            1 for m in self.known_agents.values()
            if (current_time - m.last_heartbeat).total_seconds() > 300
        )
        if stale_count > 10:
            risk_factors.append(15)
        elif stale_count > 5:
            risk_factors.append(10)
        
        return min(100, sum(risk_factors))
    
    async def _take_preventive_action(self):
        """Take preventive actions to avoid system freeze"""
        self.logger.warning("Taking preventive actions to avoid system freeze")
        
        # TODO: Implement preventive actions such as:
        # - Throttling new requests
        # - Killing stale agents
        # - Reducing concurrent limits
        # - Clearing queues
        
        # For now, just log the warning
        self.logger.warning("High freeze risk detected - manual intervention may be required")
    
    async def _alert_processing_loop(self):
        """Process and manage alerts"""
        while True:
            try:
                await self._process_alerts()
                await asyncio.sleep(60)  # Check alerts every minute
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)
    
    async def _process_alerts(self):
        """Process pending alerts"""
        # Auto-resolve old alerts
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE alerts 
                SET resolved = TRUE 
                WHERE resolved = FALSE 
                AND timestamp < datetime('now', '-1 hour')
            ''')
    
    async def _check_agent_alerts(self, metrics: AgentHealthMetrics):
        """Check agent metrics for alert conditions"""
        agent_name = metrics.agent_name
        
        # High failure rate
        if metrics.tasks_processed > 0:
            failure_rate = metrics.tasks_failed / (metrics.tasks_processed + metrics.tasks_failed)
            if failure_rate > self.max_failure_rate:
                await self._create_alert(
                    'high_failure_rate',
                    'warning',
                    f"Agent {agent_name} has high failure rate: {failure_rate:.1%}",
                    agent_name,
                    failure_rate,
                    self.max_failure_rate
                )
        
        # High memory usage
        if metrics.memory_usage_mb > 1000:  # > 1GB
            await self._create_alert(
                'high_memory_usage',
                'warning',
                f"Agent {agent_name} using {metrics.memory_usage_mb:.0f}MB memory",
                agent_name,
                metrics.memory_usage_mb,
                1000
            )
        
        # Circuit breaker trips
        if metrics.circuit_breaker_trips > 5:
            await self._create_alert(
                'circuit_breaker_trips',
                'warning',
                f"Agent {agent_name} has {metrics.circuit_breaker_trips} circuit breaker trips",
                agent_name,
                metrics.circuit_breaker_trips,
                5
            )
        
        # Ollama connectivity issues
        if not metrics.ollama_healthy:
            await self._create_alert(
                'ollama_unhealthy',
                'critical',
                f"Agent {agent_name} cannot connect to Ollama",
                agent_name,
                0,
                1
            )
    
    async def _check_system_alerts(self, metrics: OllamaSystemMetrics, memory_percent: float, cpu_percent: float):
        """Check system-wide metrics for alert conditions"""
        
        # High memory usage
        if memory_percent > self.max_memory_usage:
            await self._create_alert(
                'system_high_memory',
                'critical',
                f"System memory usage is {memory_percent:.1f}%",
                None,
                memory_percent,
                self.max_memory_usage
            )
        
        # High CPU usage
        if cpu_percent > self.max_cpu_usage:
            await self._create_alert(
                'system_high_cpu',
                'critical',
                f"System CPU usage is {cpu_percent:.1f}%",
                None,
                cpu_percent,
                self.max_cpu_usage
            )
        
        # High queue depth
        if metrics.queue_depth > self.max_queue_depth:
            await self._create_alert(
                'high_queue_depth',
                'warning',
                f"Ollama queue depth is {metrics.queue_depth}",
                None,
                metrics.queue_depth,
                self.max_queue_depth
            )
        
        # High response time
        if metrics.avg_response_time > self.max_response_time:
            await self._create_alert(
                'high_response_time',
                'warning',
                f"Average Ollama response time is {metrics.avg_response_time:.1f}s",
                None,
                metrics.avg_response_time,
                self.max_response_time
            )
    
    async def _create_alert(self, alert_type: str, severity: str, message: str, 
                           agent_name: Optional[str], metric_value: float, threshold: float):
        """Create a new alert"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if similar alert already exists
                cursor = conn.execute('''
                    SELECT id FROM alerts 
                    WHERE alert_type = ? AND agent_name = ? AND resolved = FALSE
                    AND timestamp > datetime('now', '-10 minutes')
                ''', (alert_type, agent_name))
                
                if not cursor.fetchone():
                    # Create new alert
                    conn.execute('''
                        INSERT INTO alerts (alert_type, severity, message, agent_name, metric_value, threshold)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (alert_type, severity, message, agent_name, metric_value, threshold))
                    
                    self.logger.warning(f"ALERT [{severity.upper()}] {message}")
                    
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for real-time monitoring"""
        try:
            # Current system status
            active_agents = len([m for m in self.known_agents.values() if m.status == 'active'])
            total_agents = len(self.known_agents)
            
            # Recent alerts
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT alert_type, severity, message, agent_name, timestamp
                    FROM alerts
                    WHERE resolved = FALSE
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''')
                recent_alerts = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
            
            # System metrics
            system_metrics = asdict(self.system_metrics) if self.system_metrics else {}
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'system_status': {
                    'active_agents': active_agents,
                    'total_agents': total_agents,
                    'freeze_risk_score': await self._calculate_freeze_risk(),
                    'memory_usage_percent': psutil.virtual_memory().percent,
                    'cpu_usage_percent': psutil.cpu_percent(),
                },
                'ollama_metrics': system_metrics,
                'agent_metrics': {name: asdict(metrics) for name, metrics in self.known_agents.items()},
                'recent_alerts': recent_alerts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}


async def main():
    """Main entry point"""
    monitor = OllamaAgentMonitor()
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())