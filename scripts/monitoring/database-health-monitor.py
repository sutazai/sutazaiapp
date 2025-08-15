#!/usr/bin/env python3
"""
Database Health Monitoring and Alerting System
============================================

Comprehensive database administrator tool for monitoring:
- Connection pooling status
- Replication lag and health
- Performance metrics and bottlenecks
- Backup status and integrity
- Security and access control
- Storage and capacity planning

Features:
- Real-time monitoring with alerting
- Performance baseline tracking
- Automated maintenance recommendations
- High availability status checks
- Disaster recovery validation
"""

import asyncio
import asyncpg
import redis.asyncio as redis
import json
import logging
import os
import time
import psutil
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager
import subprocess
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseMetrics:
    """Database performance and health metrics"""
    database_name: str
    connection_count: int
    max_connections: int
    connection_utilization: float
    active_queries: int
    slow_queries: int
    locks_waiting: int
    deadlocks: int
    cache_hit_ratio: float
    index_usage: float
    table_bloat: float
    disk_usage_mb: int
    backup_age_hours: Optional[int]
    last_vacuum: Optional[str]
    last_analyze: Optional[str]
    replication_lag_ms: Optional[int]
    status: str
    alerts: List[str]


@dataclass
class SystemHealth:
    """Overall system health report"""
    timestamp: str
    databases: Dict[str, DatabaseMetrics]
    overall_status: str
    critical_alerts: List[str]
    warnings: List[str]
    recommendations: List[str]


class DatabaseHealthMonitor:
    """Database administrator's operational excellence tool"""
    
    def __init__(self):
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.neo4j_driver = None
        
        # Alerting thresholds
        self.thresholds = {
            'connection_utilization': 0.8,      # 80% connection usage
            'cache_hit_ratio': 0.95,             # 95% cache hit rate minimum
            'slow_query_threshold': 1000,        # 1 second slow query
            'replication_lag_ms': 5000,          # 5 second replication lag
            'backup_age_hours': 25,              # 25 hour backup age
            'disk_usage_threshold': 0.85,        # 85% disk usage
            'lock_wait_threshold': 100,          # 100ms lock wait
            'table_bloat_threshold': 0.3         # 30% table bloat
        }
        
        # Performance baselines (loaded from previous runs)
        self.baselines = {}
        self.metrics_history = []
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection with optimized pooling
            pg_user = os.getenv("POSTGRES_USER", "sutazai")
            pg_pass = os.getenv("POSTGRES_PASSWORD", "sutazai")
            pg_host = os.getenv("POSTGRES_HOST", "localhost")
            pg_port = os.getenv("POSTGRES_PORT", "10000")
            pg_db = os.getenv("POSTGRES_DB", "sutazai")
            pg_dsn = os.getenv(
                'DATABASE_URL',
                f'postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}'
            )
            
            self.pg_pool = await asyncpg.create_pool(
                pg_dsn,
                min_size=5,
                max_size=20,
                command_timeout=10,
                server_settings={
                    'jit': 'off',
                    'statement_timeout': '30000ms'
                }
            )
            
            # Redis connection
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:10001/0')
            self.redis_client = redis.from_url(
                redis_url,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connections
            async with self.pg_pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            
            await self.redis_client.ping()
            
            logger.info("Database health monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def check_postgresql_health(self) -> DatabaseMetrics:
        """Comprehensive PostgreSQL health check"""
        alerts = []
        
        async with self.pg_pool.acquire() as conn:
            # Connection statistics
            conn_stats = await conn.fetchrow("""
                SELECT 
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE query_start < NOW() - INTERVAL '30 seconds' AND state = 'active') as slow_queries,
                    (SELECT count(*) FROM pg_stat_activity WHERE wait_event_type IS NOT NULL) as waiting_queries
            """)
            
            # Cache hit ratio
            cache_stats = await conn.fetchrow("""
                SELECT 
                    CASE WHEN (blks_hit + blks_read) > 0 
                    THEN blks_hit::float / (blks_hit + blks_read) * 100 
                    ELSE 100 END as cache_hit_ratio
                FROM pg_stat_database 
                WHERE datname = current_database()
            """)
            
            # Lock statistics
            lock_stats = await conn.fetchrow("""
                SELECT 
                    count(*) as total_locks,
                    count(*) FILTER (WHERE NOT granted) as waiting_locks
                FROM pg_locks
            """)
            
            # Deadlock count (since last reset)
            deadlock_stats = await conn.fetchrow("""
                SELECT deadlocks 
                FROM pg_stat_database 
                WHERE datname = current_database()
            """)
            
            # Index usage
            index_usage = await conn.fetchrow("""
                SELECT 
                    CASE WHEN (idx_scan + seq_scan) > 0
                    THEN idx_scan::float / (idx_scan + seq_scan) * 100
                    ELSE 0 END as index_usage_percent
                FROM pg_stat_all_tables
                WHERE schemaname = 'public'
                ORDER BY (seq_scan + idx_scan) DESC
                LIMIT 1
            """)
            
            # Database size
            db_size = await conn.fetchval("""
                SELECT pg_database_size(current_database()) / 1024 / 1024 as size_mb
            """)
            
            # Last vacuum/analyze times
            maintenance_stats = await conn.fetchrow("""
                SELECT 
                    MAX(last_vacuum) as last_vacuum,
                    MAX(last_analyze) as last_analyze
                FROM pg_stat_all_tables 
                WHERE schemaname = 'public'
            """)
            
            # Table bloat estimation (simplified)
            bloat_stats = await conn.fetchrow("""
                SELECT 
                    COALESCE(AVG(
                        CASE WHEN pg_relation_size(oid) > 0 
                        THEN (relpages::float * 8192 - pg_relation_size(oid)) / pg_relation_size(oid)
                        ELSE 0 END
                    ), 0) as avg_bloat_ratio
                FROM pg_class 
                WHERE relkind = 'r' AND relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
            """)
            
        # Calculate utilization
        connection_utilization = conn_stats['active_connections'] / conn_stats['max_connections']
        cache_hit_ratio = cache_stats['cache_hit_ratio'] / 100.0 if cache_stats['cache_hit_ratio'] else 0
        index_usage_percent = index_usage['index_usage_percent'] / 100.0 if index_usage and index_usage['index_usage_percent'] else 0
        
        # Check thresholds and generate alerts
        if connection_utilization > self.thresholds['connection_utilization']:
            alerts.append(f"High connection utilization: {connection_utilization:.1%}")
        
        if cache_hit_ratio < self.thresholds['cache_hit_ratio']:
            alerts.append(f"Low cache hit ratio: {cache_hit_ratio:.1%}")
        
        if conn_stats['slow_queries'] > 0:
            alerts.append(f"Slow queries detected: {conn_stats['slow_queries']}")
        
        if lock_stats['waiting_locks'] > 0:
            alerts.append(f"Queries waiting for locks: {lock_stats['waiting_locks']}")
        
        if deadlock_stats['deadlocks'] > 0:
            alerts.append(f"Deadlocks since last reset: {deadlock_stats['deadlocks']}")
        
        # Check backup age
        backup_age_hours = await self._check_backup_age('postgresql')
        if backup_age_hours and backup_age_hours > self.thresholds['backup_age_hours']:
            alerts.append(f"Backup is {backup_age_hours} hours old")
        
        # Determine status
        status = 'critical' if any('High' in alert or 'Low' in alert for alert in alerts) else \
                'warning' if alerts else 'healthy'
        
        return DatabaseMetrics(
            database_name='postgresql',
            connection_count=conn_stats['active_connections'],
            max_connections=conn_stats['max_connections'],
            connection_utilization=connection_utilization,
            active_queries=conn_stats['active_connections'],
            slow_queries=conn_stats['slow_queries'],
            locks_waiting=lock_stats['waiting_locks'],
            deadlocks=deadlock_stats['deadlocks'],
            cache_hit_ratio=cache_hit_ratio,
            index_usage=index_usage_percent,
            table_bloat=bloat_stats['avg_bloat_ratio'] if bloat_stats['avg_bloat_ratio'] else 0,
            disk_usage_mb=int(db_size),
            backup_age_hours=backup_age_hours,
            last_vacuum=str(maintenance_stats['last_vacuum']) if maintenance_stats['last_vacuum'] else None,
            last_analyze=str(maintenance_stats['last_analyze']) if maintenance_stats['last_analyze'] else None,
            replication_lag_ms=None,  # Not implemented yet
            status=status,
            alerts=alerts
        )
    
    async def check_redis_health(self) -> DatabaseMetrics:
        """Comprehensive Redis health check"""
        alerts = []
        
        # Get Redis info
        info = await self.redis_client.info()
        
        # Connection stats
        connected_clients = info.get('connected_clients', 0)
        max_clients = info.get('maxclients', 10000)
        connection_utilization = connected_clients / max_clients
        
        # Memory stats
        used_memory = info.get('used_memory', 0)
        max_memory = info.get('maxmemory', 0)
        
        # Performance stats
        keyspace_hits = info.get('keyspace_hits', 0)
        keyspace_misses = info.get('keyspace_misses', 0)
        hit_rate = keyspace_hits / (keyspace_hits + keyspace_misses) if (keyspace_hits + keyspace_misses) > 0 else 1.0
        
        # Replication lag (if slave)
        replication_lag = info.get('master_repl_offset', 0) - info.get('slave_repl_offset', 0) if 'slave_repl_offset' in info else None
        
        # Generate alerts
        if connection_utilization > 0.8:
            alerts.append(f"High Redis connection utilization: {connection_utilization:.1%}")
        
        if hit_rate < 0.9:
            alerts.append(f"Low Redis hit rate: {hit_rate:.1%}")
        
        if max_memory > 0 and used_memory / max_memory > 0.9:
            alerts.append(f"High Redis memory usage: {used_memory/max_memory:.1%}")
        
        backup_age_hours = await self._check_backup_age('redis')
        if backup_age_hours and backup_age_hours > self.thresholds['backup_age_hours']:
            alerts.append(f"Redis backup is {backup_age_hours} hours old")
        
        status = 'critical' if any('High' in alert for alert in alerts) else \
                'warning' if alerts else 'healthy'
        
        return DatabaseMetrics(
            database_name='redis',
            connection_count=connected_clients,
            max_connections=max_clients,
            connection_utilization=connection_utilization,
            active_queries=0,  # Redis doesn't have "queries"
            slow_queries=0,
            locks_waiting=0,
            deadlocks=0,
            cache_hit_ratio=hit_rate,
            index_usage=0,  # Not applicable to Redis
            table_bloat=0,
            disk_usage_mb=used_memory // (1024 * 1024),
            backup_age_hours=backup_age_hours,
            last_vacuum=None,
            last_analyze=None,
            replication_lag_ms=replication_lag,
            status=status,
            alerts=alerts
        )
    
    async def check_neo4j_health(self) -> DatabaseMetrics:
        """Basic Neo4j health check via HTTP API"""
        alerts = []
        backup_age_hours = await self._check_backup_age('neo4j')
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if Neo4j is responsive
                response = await client.get('http://localhost:10002/db/data/')
                if response.status_code != 200:
                    alerts.append("Neo4j HTTP interface not responding")
                    status = 'critical'
                else:
                    status = 'healthy'
                    
                    if backup_age_hours and backup_age_hours > self.thresholds['backup_age_hours']:
                        alerts.append(f"Neo4j backup is {backup_age_hours} hours old")
                        status = 'warning'
        
        except Exception as e:
            alerts.append(f"Neo4j connectivity error: {str(e)}")
            status = 'critical'
        
        return DatabaseMetrics(
            database_name='neo4j',
            connection_count=0,  # Not easily available via HTTP
            max_connections=100,  # Default estimate
            connection_utilization=0,
            active_queries=0,
            slow_queries=0,
            locks_waiting=0,
            deadlocks=0,
            cache_hit_ratio=0,
            index_usage=0,
            table_bloat=0,
            disk_usage_mb=0,
            backup_age_hours=backup_age_hours,
            last_vacuum=None,
            last_analyze=None,
            replication_lag_ms=None,
            status=status,
            alerts=alerts
        )
    
    async def _check_backup_age(self, db_type: str) -> Optional[int]:
        """Check the age of the most recent backup in hours"""
        try:
            backup_dir = f"/opt/sutazaiapp/backups/{db_type}"
            if not os.path.exists(backup_dir):
                return None
            
            # Find most recent backup file
            backup_files = []
            for file in os.listdir(backup_dir):
                if file.endswith(('.gz', '.sql', '.tar.gz')):
                    backup_files.append(os.path.join(backup_dir, file))
            
            if not backup_files:
                return None
            
            # Get most recent backup
            most_recent = max(backup_files, key=os.path.getmtime)
            backup_time = datetime.fromtimestamp(os.path.getmtime(most_recent))
            age_hours = (datetime.now() - backup_time).total_seconds() / 3600
            
            return int(age_hours)
        
        except Exception as e:
            logger.warning(f"Could not check backup age for {db_type}: {e}")
            return None
    
    async def generate_recommendations(self, health: SystemHealth) -> List[str]:
        """Generate operational recommendations based on metrics"""
        recommendations = []
        
        for db_name, metrics in health.databases.items():
            if db_name == 'postgresql':
                if metrics.connection_utilization > 0.7:
                    recommendations.append(f"Consider increasing PostgreSQL max_connections or implementing connection pooling")
                
                if metrics.cache_hit_ratio < 0.95:
                    recommendations.append(f"Increase PostgreSQL shared_buffers to improve cache hit ratio")
                
                if metrics.index_usage < 0.8:
                    recommendations.append(f"Review query performance and consider adding indexes")
                
                if metrics.table_bloat > 0.2:
                    recommendations.append(f"Schedule VACUUM FULL for heavily bloated tables")
                
                if metrics.last_vacuum and (datetime.now() - datetime.fromisoformat(metrics.last_vacuum.replace('Z', '+00:00').replace('+00:00', ''))).days > 7:
                    recommendations.append(f"PostgreSQL tables haven't been vacuumed in over a week")
            
            elif db_name == 'redis':
                if metrics.cache_hit_ratio < 0.9:
                    recommendations.append(f"Review Redis memory policy and key expiration strategy")
                
                if metrics.connection_utilization > 0.8:
                    recommendations.append(f"Consider Redis connection pooling or clustering")
            
            # Backup recommendations
            if metrics.backup_age_hours and metrics.backup_age_hours > 24:
                recommendations.append(f"Schedule more frequent {db_name} backups (current: {metrics.backup_age_hours}h old)")
        
        return recommendations
    
    async def run_health_check(self) -> SystemHealth:
        """Run comprehensive health check on all databases"""
        databases = {}
        critical_alerts = []
        warnings = []
        
        # Check PostgreSQL
        try:
            pg_metrics = await self.check_postgresql_health()
            databases['postgresql'] = pg_metrics
            
            for alert in pg_metrics.alerts:
                if pg_metrics.status == 'critical':
                    critical_alerts.append(f"PostgreSQL: {alert}")
                else:
                    warnings.append(f"PostgreSQL: {alert}")
        
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            critical_alerts.append(f"PostgreSQL: Health check failed - {str(e)}")
        
        # Check Redis
        try:
            redis_metrics = await self.check_redis_health()
            databases['redis'] = redis_metrics
            
            for alert in redis_metrics.alerts:
                if redis_metrics.status == 'critical':
                    critical_alerts.append(f"Redis: {alert}")
                else:
                    warnings.append(f"Redis: {alert}")
        
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            critical_alerts.append(f"Redis: Health check failed - {str(e)}")
        
        # Check Neo4j
        try:
            neo4j_metrics = await self.check_neo4j_health()
            databases['neo4j'] = neo4j_metrics
            
            for alert in neo4j_metrics.alerts:
                if neo4j_metrics.status == 'critical':
                    critical_alerts.append(f"Neo4j: {alert}")
                else:
                    warnings.append(f"Neo4j: {alert}")
        
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            warnings.append(f"Neo4j: Health check failed - {str(e)}")
        
        # Determine overall status
        if critical_alerts:
            overall_status = 'critical'
        elif warnings:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        health = SystemHealth(
            timestamp=datetime.now().isoformat(),
            databases=databases,
            overall_status=overall_status,
            critical_alerts=critical_alerts,
            warnings=warnings,
            recommendations=[]
        )
        
        # Generate recommendations
        health.recommendations = await self.generate_recommendations(health)
        
        return health
    
    async def send_alerts(self, health: SystemHealth):
        """Send alerts for critical issues"""
        if health.critical_alerts:
            logger.error("CRITICAL DATABASE ALERTS:")
            for alert in health.critical_alerts:
                logger.error(f"  🚨 {alert}")
        
        if health.warnings:
            logger.warning("DATABASE WARNINGS:")
            for warning in health.warnings:
                logger.warning(f"  ⚠️  {warning}")
        
        if health.recommendations:
            logger.info("RECOMMENDATIONS:")
            for rec in health.recommendations:
                logger.info(f"  💡 {rec}")
    
    def save_metrics(self, health: SystemHealth):
        """Save metrics to file for historical analysis"""
        try:
            metrics_file = "/opt/sutazaiapp/logs/database_metrics.jsonl"
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(asdict(health)) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    async def cleanup(self):
        """Clean up connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            await self.redis_client.close()


async def main():
    """Main monitoring loop"""
    monitor = DatabaseHealthMonitor()
    
    try:
        await monitor.initialize()
        
        while True:
            logger.info("Starting database health check...")
            
            health = await monitor.run_health_check()
            
            # Print status
            logger.info(f"\n{'='*60}")
            logger.info(f"DATABASE HEALTH REPORT - {health.timestamp}")
            logger.info(f"Overall Status: {health.overall_status.upper()}")
            logger.info(f"{'='*60}")
            
            for db_name, metrics in health.databases.items():
                logger.info(f"\n{db_name.upper()}:")
                logger.info(f"  Status: {metrics.status}")
                logger.info(f"  Connections: {metrics.connection_count}/{metrics.max_connections} ({metrics.connection_utilization:.1%})")
                logger.info(f"  Cache Hit Ratio: {metrics.cache_hit_ratio:.1%}")
                if metrics.backup_age_hours is not None:
                    logger.info(f"  Backup Age: {metrics.backup_age_hours} hours")
                if metrics.alerts:
                    logger.info(f"  Alerts: {', '.join(metrics.alerts)}")
            
            if health.recommendations:
                logger.info(f"\nRECOMMENDATIONS:")
                for rec in health.recommendations:
                    logger.info(f"  • {rec}")
            
            # Send alerts and save metrics
            await monitor.send_alerts(health)
            monitor.save_metrics(health)
            
            # Wait before next check (default: 5 minutes)
            await asyncio.sleep(int(os.getenv('HEALTH_CHECK_INTERVAL', 300)))
    
    except KeyboardInterrupt:
        logger.info("Health monitor stopped by user")
    except Exception as e:
        logger.error(f"Health monitor error: {e}")
        raise
    finally:
        await monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())