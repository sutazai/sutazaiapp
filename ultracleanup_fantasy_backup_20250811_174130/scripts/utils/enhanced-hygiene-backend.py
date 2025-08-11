#!/usr/bin/env python3
"""
Enhanced Hygiene Monitor Backend - Production-Ready Containerized Version
Purpose: Real-time monitoring with PostgreSQL persistence and Redis caching
Author: Sutazai Backend API Architect
Version: 3.0.0 - Perfect Containerized Edition
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import asyncio
import json
import os
import psutil
import time
from pathlib import Path
import structlog
import asyncpg
import redis.asyncio as redis
from aiohttp import web, WSMsgType
from aiohttp_cors import setup as cors_setup, ResourceOptions
import uvloop

class SafeJSONEncoder(json.JSONEncoder):
    """Safe JSON encoder that handles datetime objects and prevents circular references"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_objects = set()
        self.depth = 0
        self.max_depth = 20  # Prevent deep recursion
    
    def encode(self, obj):
        """Override encode to reset state"""
        self.seen_objects = set()
        self.depth = 0
        return super().encode(obj)
    
    def iterencode(self, obj, _one_shot=False):
        """Override iterencode to prevent stack overflow"""
        self.seen_objects = set()
        self.depth = 0
        return super().iterencode(obj, _one_shot)
    
    def default(self, obj):
        # Depth protection
        self.depth += 1
        if self.depth > self.max_depth:
            return '[Max Depth Exceeded]'
        
        try:
            # Handle common types first
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat')):
                return obj.isoformat()
            
            # Check for circular references on complex objects
            obj_id = id(obj)
            if obj_id in self.seen_objects:
                return '[Circular Reference]'
            
            # Only track complex objects
            if isinstance(obj, (dict, list, tuple, set)) or hasattr(obj, '__dict__'):
                self.seen_objects.add(obj_id)
            
            # Handle custom objects
            if hasattr(obj, '__dict__'):
                safe_dict = {}
                for k, v in obj.__dict__.items():
                    if not k.startswith('_') and not callable(v):
                        try:
                            # Test if the value is serializable
                            json.dumps(v, default=str, ensure_ascii=False)
                            safe_dict[k] = v
                        except (TypeError, ValueError):
                            safe_dict[k] = str(v)
                return safe_dict
            
            # Fallback to string representation
            return str(obj)
            
        except Exception:
            return f'[Unserializable: {type(obj).__name__}]'
        finally:
            self.depth -= 1

# Maintain backward compatibility
DateTimeEncoder = SafeJSONEncoder

# Enhanced logging setup
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class EnhancedHygieneBackend:
    """Production-grade hygiene monitoring backend with full persistence"""
    
    def __init__(self):
        self.project_root = Path(os.getenv('PROJECT_ROOT', '/app/project'))
        # Do not hardcode credentials; default to URL without inline secrets
        self.database_url = os.getenv('DATABASE_URL', 'postgresql://hygiene-postgres:5432/hygiene_monitoring')
        self.redis_url = os.getenv('REDIS_URL', 'redis://hygiene-redis:6379/0')
        
        # Connection pools
        self.db_pool = None
        self.redis_pool = None
        
        # WebSocket clients
        self.websocket_clients: Set[web.WebSocketResponse] = set()
        
        # Runtime state
        self.running = False
        self.metrics_cache = {}
        self.last_scan_time = None
        
        logger.info("Enhanced Hygiene Backend initialized", 
                   project_root=str(self.project_root))

    async def initialize_database(self):
        """Initialize PostgreSQL connection pool and tables"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Create tables if they don't exist
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        cpu_usage REAL,
                        memory_percentage REAL,
                        memory_used_gb REAL,
                        memory_total_gb REAL,
                        disk_percentage REAL,
                        disk_used_gb REAL,
                        disk_total_gb REAL,
                        network_status TEXT DEFAULT 'HEALTHY'
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                        ON system_metrics(timestamp DESC);
                ''')
                
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS violations (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        rule_id TEXT NOT NULL,
                        rule_name TEXT,
                        file_path TEXT,
                        severity TEXT NOT NULL,
                        description TEXT,
                        status TEXT DEFAULT 'ACTIVE',
                        resolved_at TIMESTAMPTZ NULL
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_violations_timestamp 
                        ON violations(timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_violations_rule_id 
                        ON violations(rule_id);
                    CREATE INDEX IF NOT EXISTS idx_violations_severity 
                        ON violations(severity);
                ''')
                
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS agent_health (
                        id SERIAL PRIMARY KEY,
                        agent_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'ACTIVE',
                        last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        tasks_completed INTEGER DEFAULT 0,
                        tasks_failed INTEGER DEFAULT 0,
                        cpu_usage REAL DEFAULT 0.0,
                        memory_usage REAL DEFAULT 0.0,
                        metadata JSONB DEFAULT '{}'::jsonb
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_agent_health_agent_id 
                        ON agent_health(agent_id);
                ''')
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            raise

    async def initialize_redis(self):
        """Initialize Redis connection pool"""
        try:
            # Add retry logic for Redis connection
            max_retries = 5
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    self.redis_pool = redis.ConnectionPool.from_url(
                        self.redis_url,
                        max_connections=20,
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5
                    )
                    
                    # Test connection
                    redis_client = redis.Redis(connection_pool=self.redis_pool)
                    await redis_client.ping()
                    await redis_client.aclose()
                    
                    logger.info("Redis connection initialized successfully", 
                               url=self.redis_url, 
                               attempt=attempt + 1)
                    return
                    
                except Exception as conn_error:
                    logger.warning("Redis connection attempt failed", 
                                 attempt=attempt + 1,
                                 max_retries=max_retries,
                                 error=str(conn_error),
                                 redis_url=self.redis_url)
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise conn_error
            
        except Exception as e:
            logger.error("Redis initialization failed completely", error=str(e))
            # Don't raise - allow service to continue without Redis
            self.redis_pool = None

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network check (simplified)
            network_status = 'HEALTHY'
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
            except Exception as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                network_status = 'DEGRADED'
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': round(cpu_percent, 2),
                'memory_percentage': round(memory.percent, 2),
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'disk_percentage': round((disk.used / disk.total) * 100, 2),
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_total_gb': round(disk.total / (1024**3), 2),
                'network_status': network_status,
                'load_avg': list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
            # Store in database
            await self._store_metrics(metrics)
            
            # Cache in Redis (if available)
            if self.redis_pool:
                try:
                    redis_client = redis.Redis(connection_pool=self.redis_pool)
                    await redis_client.setex(
                        'latest_metrics', 
                        60,  # 1 minute TTL
                        json.dumps(metrics, cls=DateTimeEncoder)
                    )
                    await redis_client.aclose()
                except Exception as redis_error:
                    logger.warning("Redis caching failed", error=str(redis_error))
            
            self.metrics_cache = metrics
            return metrics
            
        except Exception as e:
            logger.error("Error collecting metrics", error=str(e))
            return self.metrics_cache or {}

    async def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in PostgreSQL"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO system_metrics (
                        timestamp, cpu_usage, memory_percentage, memory_used_gb,
                        memory_total_gb, disk_percentage, disk_used_gb, 
                        disk_total_gb, network_status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''', 
                datetime.fromisoformat(metrics['timestamp']),
                metrics['cpu_usage'],
                metrics['memory_percentage'], 
                metrics['memory_used_gb'],
                metrics['memory_total_gb'],
                metrics['disk_percentage'],
                metrics['disk_used_gb'],
                metrics['disk_total_gb'],
                metrics['network_status'])
                
        except Exception as e:
            logger.error("Error storing metrics", error=str(e))

    async def scan_violations(self) -> List[Dict[str, Any]]:
        """Advanced violation scanning with rule validation"""
        violations = []
        scan_start = time.time()
        
        try:
            # Define rule patterns
            rule_patterns = {
                'no_fantasy_elements': {
                    'patterns': ['process', 'configurator', 'transfer', 'fairy', 'unicorn'],
                    'severity': 'CRITICAL',
                    'description': 'Fantasy elements detected in code'
                },
                'no_hardcoded_secrets': {
                    'patterns': ['password=', 'api_key=', 'secret=', 'token='],
                    'severity': 'HIGH',
                    'description': 'Potential hardcoded secrets detected'
                },
                    'severity': 'MEDIUM',
                }
            }
            
            # Scan Python files
            python_files = list(self.project_root.rglob('*.py'))
            scanned_files = 0
            
            for file_path in python_files[:50]:  # Limit for performance
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                    
                    relative_path = str(file_path.relative_to(self.project_root))
                    scanned_files += 1
                    
                    # Check each rule
                    for rule_id, rule_config in rule_patterns.items():
                        for pattern in rule_config['patterns']:
                            if pattern in content:
                                violation = {
                                    'timestamp': datetime.now().isoformat(),
                                    'rule_id': rule_id,
                                    'rule_name': rule_id.replace('_', ' ').title(),
                                    'file_path': relative_path,
                                    'severity': rule_config['severity'],
                                    'description': rule_config['description'],
                                    'pattern_matched': pattern
                                }
                                violations.append(violation)
                                
                                # Store in database
                                await self._store_violation(violation)
                                break  # One violation per rule per file
                        
                except Exception as e:
                    logger.warning("Could not scan file", file=str(file_path), error=str(e))
            
            scan_duration = time.time() - scan_start
            self.last_scan_time = datetime.now()
            
            logger.info("Violation scan completed", 
                       files_scanned=scanned_files,
                       violations_found=len(violations),
                       duration_seconds=round(scan_duration, 2))
            
            return violations
            
        except Exception as e:
            logger.error("Error during violation scan", error=str(e))
            return []

    async def _store_violation(self, violation: Dict[str, Any]):
        """Store violation in PostgreSQL"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO violations (
                        timestamp, rule_id, rule_name, file_path, 
                        severity, description
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                ''',
                datetime.fromisoformat(violation['timestamp']),
                violation['rule_id'],
                violation['rule_name'],
                violation['file_path'],
                violation['severity'],
                violation['description'])
                
        except Exception as e:
            logger.error("Error storing violation", error=str(e))

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get latest metrics
            metrics = await self.collect_system_metrics()
            
            # Get recent violations from database
            async with self.db_pool.acquire() as conn:
                violation_rows = await conn.fetch('''
                    SELECT rule_id, rule_name, file_path, severity, description, timestamp
                    FROM violations 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC
                    LIMIT 50
                ''')
                
                violations = [dict(row) for row in violation_rows]
                
                # Get violation counts by severity
                severity_counts = await conn.fetch('''
                    SELECT severity, COUNT(*) as count
                    FROM violations 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY severity
                ''')
                
                severity_map = {row['severity']: row['count'] for row in severity_counts}
            
            # Calculate compliance score
            total_violations = sum(severity_map.values())
            critical_violations = severity_map.get('CRITICAL', 0)
            high_violations = severity_map.get('HIGH', 0)
            
            # Weighted compliance score
            compliance_penalty = (critical_violations * 10) + (high_violations * 5) + (total_violations * 1)
            compliance_score = max(0, 100 - compliance_penalty)
            
            # Mock agent health (would be real in production)
            agent_health = [
                {
                    'agent_id': 'hygiene-scanner',
                    'name': 'Hygiene Scanner',
                    'status': 'ACTIVE',
                    'last_heartbeat': datetime.now().isoformat(),
                    'tasks_completed': len(violations),
                    'tasks_failed': 0,
                    'cpu_usage': round(metrics.get('cpu_usage', 0) * 0.1, 1),
                    'memory_usage': round(metrics.get('memory_percentage', 0) * 0.2, 1)
                },
                {
                    'agent_id': 'rule-enforcer',
                    'name': 'Rule Enforcer',
                    'status': 'ACTIVE',
                    'last_heartbeat': datetime.now().isoformat(),
                    'tasks_completed': total_violations,
                    'tasks_failed': 0,
                    'cpu_usage': 1.5,
                    'memory_usage': 25.3
                }
            ]
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'systemStatus': 'MONITORING' if self.running else 'OFFLINE',
                'complianceScore': compliance_score,
                'totalViolations': total_violations,
                'criticalViolations': critical_violations,
                'warningViolations': high_violations + severity_map.get('MEDIUM', 0),
                'activeAgents': len(agent_health),
                'systemMetrics': metrics,
                'recentViolations': violations[:10],
                'agentHealth': agent_health,
                'lastScanTime': self.last_scan_time.isoformat() if self.last_scan_time else None,
                'recentActions': [
                    {
                        'id': str(int(time.time())),
                        'timestamp': datetime.now().isoformat(),
                        'action_type': 'SCAN',
                        'status': 'COMPLETED',
                        'agent_id': 'hygiene-scanner',
                        'details': f'Scanned codebase, found {total_violations} violations',
                        'duration_ms': 1500
                    }
                ]
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error("Error getting dashboard data", error=str(e))
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'systemStatus': 'ERROR'
            }

    def _safe_serialize(self, obj, seen=None, depth=0):
        """Safely serialize objects with circular reference and depth protection"""
        if seen is None:
            seen = set()
        
        if depth > 20:  # Maximum depth limit
            return '[Max Depth Exceeded]'
        
        obj_id = id(obj)
        if obj_id in seen:
            return '[Circular Reference]'
        
        # Handle basic types that don't need tracking
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Track complex objects
        seen.add(obj_id)
        
        try:
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    try:
                        result[str(k)] = self._safe_serialize(v, seen.copy(), depth + 1)
                    except Exception:
                        result[str(k)] = '[Serialization Error]'
                return result
            
            elif isinstance(obj, (list, tuple)):
                result = []
                for item in obj:
                    try:
                        result.append(self._safe_serialize(item, seen.copy(), depth + 1))
                    except Exception:
                        result.append('[Serialization Error]')
                return result
            
            elif hasattr(obj, '__dict__'):
                result = {}
                for k, v in obj.__dict__.items():
                    if not k.startswith('_') and not callable(v):
                        try:
                            result[k] = self._safe_serialize(v, seen.copy(), depth + 1)
                        except Exception:
                            result[k] = '[Serialization Error]'
                return result
            
            else:
                # Fallback to string representation
                return str(obj)
                
        except Exception:
            return f'[Unserializable: {type(obj).__name__}]'
        finally:
            seen.discard(obj_id)

    async def broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket clients with circular reference protection"""
        if not self.websocket_clients:
            return
        
        closed_clients = set()
        
        try:
            # Safely serialize the message to prevent circular references
            safe_message = self._safe_serialize(message)
            message_json = json.dumps(safe_message, ensure_ascii=False)
            
            # Limit message size to prevent memory issues
            if len(message_json) > 100000:  # 100KB limit
                logger.warning("WebSocket message too large, truncating", 
                             size=len(message_json))
                # Send a simplified version
                simplified_message = {
                    'type': message.get('type', 'unknown'),
                    'timestamp': message.get('timestamp', datetime.now().isoformat()),
                    'error': 'Message truncated due to size limit'
                }
                safe_simplified = self._safe_serialize(simplified_message)
                message_json = json.dumps(safe_simplified, ensure_ascii=False)
            
        except Exception as e:
            logger.error("Failed to serialize WebSocket message", error=str(e))
            # Send error message instead
            error_message = {
                'type': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': f'Failed to serialize message: {str(e)}'
            }
            message_json = json.dumps(error_message)
        
        # Broadcast to all clients
        for ws in list(self.websocket_clients):
            try:
                await asyncio.wait_for(ws.send_str(message_json), timeout=5.0)
            except (Exception, asyncio.TimeoutError) as e:
                logger.warning("WebSocket send failed", error=str(e))
                closed_clients.add(ws)
        
        # Remove closed clients
        self.websocket_clients -= closed_clients
        
        if closed_clients:
            logger.info("Cleaned up closed WebSocket connections", count=len(closed_clients))

    async def background_monitor(self):
        """Background monitoring task with stack overflow prevention"""
        self.running = True
        logger.info("Background monitoring started")
        
        scan_counter = 0
        max_consecutive_errors = 5
        consecutive_errors = 0
        
        while self.running:
            try:
                # Prevent deep call stack by limiting async nesting
                async def safe_collect_metrics():
                    try:
                        return await asyncio.wait_for(self.collect_system_metrics(), timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.warning("Metrics collection timed out")
                        return self.metrics_cache or {}
                    except Exception as e:
                        logger.error("Metrics collection failed", error=str(e))
                        return self.metrics_cache or {}
                
                # Collect metrics with timeout protection
                metrics = await safe_collect_metrics()
                
                # Broadcast metrics (non-blocking)
                asyncio.create_task(self.broadcast_to_websockets({
                    'type': 'system_metrics',
                    'data': metrics
                }))
                
                # Scan for violations every 30 seconds
                if scan_counter % 30 == 0:
                    try:
                        violations = await asyncio.wait_for(self.scan_violations(), timeout=15.0)
                        if violations:
                            # Use create_task to prevent deep call chains
                            asyncio.create_task(self.broadcast_to_websockets({
                                'type': 'violations_update',
                                'data': violations
                            }))
                            
                            # Send dashboard update (non-blocking)
                            async def send_dashboard_update():
                                try:
                                    dashboard_data = await self.get_dashboard_data()
                                    await self.broadcast_to_websockets({
                                        'type': 'dashboard_update',
                                        'data': dashboard_data
                                    })
                                except Exception as e:
                                    logger.error("Dashboard update failed", error=str(e))
                            
                            asyncio.create_task(send_dashboard_update())
                    except asyncio.TimeoutError:
                        logger.warning("Violation scan timed out")
                
                scan_counter += 1
                consecutive_errors = 0  # Reset error counter on success
                await asyncio.sleep(1)  # 1-second intervals
                
            except Exception as e:
                consecutive_errors += 1
                logger.error("Background monitoring error", 
                           error=str(e), 
                           consecutive_errors=consecutive_errors)
                
                # Exponential backoff on consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive errors, extending sleep")
                    await asyncio.sleep(min(30, consecutive_errors * 2))
                else:
                    await asyncio.sleep(5)

# HTTP Handlers
async def websocket_handler(request):
    """Enhanced WebSocket handler"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    backend = request.app['backend']
    backend.websocket_clients.add(ws)
    
    client_id = f"client_{len(backend.websocket_clients)}"
    logger.info("WebSocket client connected", client_id=client_id)
    
    try:
        # Send initial dashboard data
        dashboard_data = await backend.get_dashboard_data()
        initial_message = {
            'type': 'initial_data',
            'data': dashboard_data
        }
        await ws.send_str(json.dumps(initial_message, cls=DateTimeEncoder))
        
        # Handle incoming messages
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    logger.info("WebSocket message received", client_id=client_id, message_type=data.get('type'))
                    
                    # Handle different message types
                    if data.get('type') == 'request_scan':
                        violations = await backend.scan_violations()
                        await ws.send_str(json.dumps({
                            'type': 'scan_result',
                            'data': violations
                        }, cls=DateTimeEncoder))
                        
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received", client_id=client_id)
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error('WebSocket error', client_id=client_id, error=ws.exception())
                break
                
    except Exception as e:
        logger.error("WebSocket handler error", client_id=client_id, error=str(e))
    finally:
        backend.websocket_clients.discard(ws)
        logger.info("WebSocket client disconnected", client_id=client_id)
    
    return ws

async def status_handler(request):
    """Enhanced status endpoint"""
    backend = request.app['backend']
    data = await backend.get_dashboard_data()
    
    return web.Response(
        text=json.dumps(data, cls=DateTimeEncoder),
        content_type='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )

async def metrics_handler(request):
    """System metrics endpoint"""
    backend = request.app['backend']
    metrics = await backend.collect_system_metrics()
    
    return web.Response(
        text=json.dumps(metrics, cls=DateTimeEncoder),
        content_type='application/json',
        headers={
            'Cache-Control': 'no-cache, max-age=10',
            'Access-Control-Allow-Origin': '*'
        }
    )

async def scan_handler(request):
    """Manual violation scan trigger"""
    backend = request.app['backend']
    violations = await backend.scan_violations()
    
    response_data = {
        'success': True,
        'violations_found': len(violations),
        'violations': violations,
        'timestamp': datetime.now().isoformat()
    }
    
    return web.Response(
        text=json.dumps(response_data, cls=DateTimeEncoder),
        content_type='application/json',
        headers={
            'Access-Control-Allow-Origin': '*'
        }
    )

async def health_handler(request):
    """Kubernetes health check"""
    backend = request.app['backend']
    
    try:
        # Quick health checks
        if not backend.db_pool:
            raise Exception("Database pool not initialized")
        
        # Redis is optional, log warning if not available
        if not backend.redis_pool:
            logger.warning("Redis pool not available - service running without caching")
        
        # Test database connection
        async with backend.db_pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '3.0.0'
        }
        
        return web.Response(
            text=json.dumps(health_data, cls=DateTimeEncoder),
            content_type='application/json'
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        error_data = {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        return web.Response(
            text=json.dumps(error_data, cls=DateTimeEncoder),
            content_type='application/json',
            status=503
        )

def create_app():
    """Create and configure the application"""
    app = web.Application()
    
    # Initialize backend
    backend = EnhancedHygieneBackend()
    app['backend'] = backend
    
    # Setup routes
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/api/hygiene/status', status_handler)
    app.router.add_get('/api/system/metrics', metrics_handler)
    app.router.add_post('/api/hygiene/scan', scan_handler)
    app.router.add_get('/health', health_handler)
    
    # CORS setup
    cors = cors_setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

async def init_app():
    """Initialize application with database connections"""
    app = create_app()
    backend = app['backend']
    
    # Initialize connections
    await backend.initialize_database()
    await backend.initialize_redis()
    
    # Start background monitoring
    asyncio.create_task(backend.background_monitor())
    
    return app

async def main():
    """Main entry point with uvloop"""
    # Use uvloop for better performance
    uvloop.install()
    
    app = await init_app()
    
    # Start web server
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    
    logger.info("🚀 Enhanced Hygiene Monitor Backend started",
               host="0.0.0.0", 
               port=8080,
               version="3.0.0")
    
    logger.info("📊 Endpoints available:",
               websocket="ws://0.0.0.0:8080/ws",
               status="/api/hygiene/status",
               metrics="/api/system/metrics",
               health="/health")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
        backend.running = False
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
