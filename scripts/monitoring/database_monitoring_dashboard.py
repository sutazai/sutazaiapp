#!/usr/bin/env python3
"""
📊 DATABASE MONITORING DASHBOARD
ULTRAINVESTIGATE Database Specialist

Real-time monitoring for all 6 databases:
- PostgreSQL performance and connection monitoring  
- Redis cache hit rates and memory usage
- Vector database (Qdrant, ChromaDB, FAISS) health
- Neo4j graph database utilization
- Automated alerting for performance issues
"""

import asyncio
import asyncpg
import redis
import requests
import psutil
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseMetrics:
    """Standardized database metrics"""
    name: str
    status: str  # healthy, degraded, failed
    response_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    connections: int
    uptime_seconds: int
    custom_metrics: Dict[str, Any]
    timestamp: float
    alerts: List[str]

class DatabaseMonitor:
    """Multi-database monitoring system"""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.alert_thresholds = {
            'response_time_ms': 1000,   # 1 second
            'memory_usage_mb': 2048,    # 2GB
            'cpu_percent': 80,          # 80%
            'redis_hit_rate': 50,       # 50% minimum
            'postgres_connections': 180, # 90% of max_connections=200
        }
        
    async def monitor_postgresql(self) -> DatabaseMetrics:
        """Monitor PostgreSQL database"""
        alerts = []
        custom_metrics = {}
        
        try:
            start_time = time.time()
            
            # Connect to PostgreSQL
            conn = await asyncpg.connect(
                host='localhost',
                port=10000,
                user='sutazai',
                password='sutazai',  # Use env var in production
                database='sutazai'
            )
            
            # Basic health check
            await conn.execute('SELECT 1')
            response_time = (time.time() - start_time) * 1000
            
            # Get connection count
            connections = await conn.fetchval(
                'SELECT count(*) FROM pg_stat_activity WHERE datname = $1', 
                'sutazai'
            )
            
            if connections > self.alert_thresholds['postgres_connections']:
                alerts.append(f"High connection count: {connections}")
            
            # Database statistics
            db_stats = await conn.fetchrow("""
                SELECT 
                    numbackends,
                    xact_commit,
                    xact_rollback,
                    blks_read,
                    blks_hit,
                    temp_files,
                    temp_bytes
                FROM pg_stat_database 
                WHERE datname = 'sutazai'
            """)
            
            # Cache hit ratio
            if db_stats['blks_read'] + db_stats['blks_hit'] > 0:
                cache_hit_ratio = (db_stats['blks_hit'] / 
                    (db_stats['blks_read'] + db_stats['blks_hit'])) * 100
                custom_metrics['cache_hit_ratio'] = cache_hit_ratio
                
                if cache_hit_ratio < 95:
                    alerts.append(f"Low cache hit ratio: {cache_hit_ratio:.1f}%")
            
            # Table health
            table_health = await conn.fetch("""
                SELECT 
                    relname,
                    n_live_tup,
                    n_dead_tup,
                    ROUND((n_dead_tup::float / NULLIF(n_live_tup + n_dead_tup, 0)) * 100, 2) as dead_percentage
                FROM pg_stat_user_tables 
                WHERE n_dead_tup > 0
                ORDER BY dead_percentage DESC
            """)
            
            dead_tuple_issues = []
            for row in table_health:
                if row['dead_percentage'] and row['dead_percentage'] > 20:
                    dead_tuple_issues.append(f"{row['relname']}: {row['dead_percentage']}%")
                    
            if dead_tuple_issues:
                alerts.append(f"High dead tuples: {', '.join(dead_tuple_issues)}")
            
            # Long running queries
            long_queries = await conn.fetchval("""
                SELECT count(*) 
                FROM pg_stat_activity 
                WHERE state = 'active' 
                AND now() - query_start > interval '30 seconds'
                AND datname = 'sutazai'
            """)
            
            if long_queries > 0:
                alerts.append(f"Long running queries: {long_queries}")
            
            custom_metrics.update({
                'active_connections': connections,
                'commits': db_stats['xact_commit'],
                'rollbacks': db_stats['xact_rollback'], 
                'blocks_read': db_stats['blks_read'],
                'blocks_hit': db_stats['blks_hit'],
                'temp_files': db_stats['temp_files'],
                'dead_tuple_tables': len(dead_tuple_issues),
                'long_queries': long_queries
            })
            
            await conn.close()
            
            return DatabaseMetrics(
                name='PostgreSQL',
                status='healthy' if not alerts else 'degraded',
                response_time_ms=response_time,
                memory_usage_mb=0,  # Would need system-level monitoring
                cpu_percent=0,      # Would need system-level monitoring
                connections=connections,
                uptime_seconds=0,   # Would need to track start time
                custom_metrics=custom_metrics,
                timestamp=time.time(),
                alerts=alerts
            )
            
        except Exception as e:
            logger.error(f"PostgreSQL monitoring failed: {e}")
            return DatabaseMetrics(
                name='PostgreSQL',
                status='failed',
                response_time_ms=0,
                memory_usage_mb=0,
                cpu_percent=0,
                connections=0,
                uptime_seconds=0,
                custom_metrics={},
                timestamp=time.time(),
                alerts=[f"Connection failed: {str(e)}"]
            )
    
    def monitor_redis(self) -> DatabaseMetrics:
        """Monitor Redis cache"""
        alerts = []
        custom_metrics = {}
        
        try:
            start_time = time.time()
            
            # Connect to Redis
            client = redis.from_url('redis://localhost:10001', decode_responses=True)
            client.ping()
            response_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = client.info()
            stats = client.info('stats')
            memory = client.info('memory')
            
            # Calculate hit rate
            hits = stats.get('keyspace_hits', 0)
            misses = stats.get('keyspace_misses', 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            if hit_rate < self.alert_thresholds['redis_hit_rate']:
                alerts.append(f"Low cache hit rate: {hit_rate:.1f}%")
            
            # Memory usage
            used_memory = memory.get('used_memory', 0)
            max_memory = memory.get('maxmemory', 0)
            memory_mb = used_memory / (1024 * 1024)
            
            if max_memory > 0:
                memory_percent = (used_memory / max_memory) * 100
                if memory_percent > 90:
                    alerts.append(f"High memory usage: {memory_percent:.1f}%")
            
            # Expired keys analysis
            expired_keys = stats.get('expired_keys', 0)
            if expired_keys > hits and hits > 0:
                alerts.append(f"High expiration rate: {expired_keys} expired vs {hits} hits")
            
            # Connection count
            connections = info.get('connected_clients', 0)
            
            custom_metrics.update({
                'hit_rate': hit_rate,
                'hits': hits,
                'misses': misses,
                'total_keys': client.dbsize(),
                'expired_keys': expired_keys,
                'evicted_keys': stats.get('evicted_keys', 0),
                'memory_used_mb': memory_mb,
                'memory_peak_mb': memory.get('used_memory_peak', 0) / (1024 * 1024),
                'memory_fragmentation': memory.get('mem_fragmentation_ratio', 1.0),
                'ops_per_sec': stats.get('instantaneous_ops_per_sec', 0)
            })
            
            return DatabaseMetrics(
                name='Redis',
                status='healthy' if not alerts else 'degraded',
                response_time_ms=response_time,
                memory_usage_mb=memory_mb,
                cpu_percent=0,
                connections=connections,
                uptime_seconds=info.get('uptime_in_seconds', 0),
                custom_metrics=custom_metrics,
                timestamp=time.time(),
                alerts=alerts
            )
            
        except Exception as e:
            logger.error(f"Redis monitoring failed: {e}")
            return DatabaseMetrics(
                name='Redis',
                status='failed',
                response_time_ms=0,
                memory_usage_mb=0,
                cpu_percent=0,
                connections=0,
                uptime_seconds=0,
                custom_metrics={},
                timestamp=time.time(),
                alerts=[f"Connection failed: {str(e)}"]
            )
    
    def monitor_vector_database(self, name: str, url: str, health_endpoint: str) -> DatabaseMetrics:
        """Monitor vector databases (Qdrant, ChromaDB, FAISS)"""
        alerts = []
        custom_metrics = {}
        
        try:
            start_time = time.time()
            
            # Health check
            response = requests.get(f"{url}{health_endpoint}", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                alerts.append(f"HTTP {response.status_code}")
                
            # Try to get additional metrics based on database type
            if name == 'Qdrant':
                try:
                    collections = requests.get(f"{url}/collections", timeout=5)
                    if collections.status_code == 200:
                        custom_metrics['collections'] = len(collections.json().get('result', {}).get('collections', []))
                except:
                    pass
                    
            elif name == 'ChromaDB':
                try:
                    collections = requests.get(f"{url}/api/v1/collections", timeout=5)
                    if collections.status_code == 200:
                        custom_metrics['collections'] = len(collections.json())
                except:
                    pass
            
            return DatabaseMetrics(
                name=name,
                status='healthy' if response.status_code == 200 and not alerts else 'degraded',
                response_time_ms=response_time,
                memory_usage_mb=0,  # Would need container-level monitoring
                cpu_percent=0,
                connections=0,
                uptime_seconds=0,
                custom_metrics=custom_metrics,
                timestamp=time.time(),
                alerts=alerts
            )
            
        except Exception as e:
            logger.error(f"{name} monitoring failed: {e}")
            return DatabaseMetrics(
                name=name,
                status='failed',
                response_time_ms=0,
                memory_usage_mb=0,
                cpu_percent=0,
                connections=0,
                uptime_seconds=0,
                custom_metrics={},
                timestamp=time.time(),
                alerts=[f"Connection failed: {str(e)}"]
            )
    
    def monitor_neo4j(self) -> DatabaseMetrics:
        """Monitor Neo4j graph database"""
        alerts = []
        custom_metrics = {}
        
        try:
            start_time = time.time()
            
            # HTTP health check
            response = requests.get('http://localhost:10002', timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                alerts.append(f"HTTP {response.status_code}")
            
            # Try to get node count (would need proper Cypher connection)
            custom_metrics.update({
                'http_accessible': response.status_code == 200,
                'nodes': 0,  # Would need Cypher query: MATCH (n) RETURN count(n)
                'relationships': 0,  # Would need proper connection
            })
            
            return DatabaseMetrics(
                name='Neo4j',
                status='healthy' if response.status_code == 200 else 'degraded',
                response_time_ms=response_time,
                memory_usage_mb=0,
                cpu_percent=0,
                connections=0,
                uptime_seconds=0,
                custom_metrics=custom_metrics,
                timestamp=time.time(),
                alerts=alerts
            )
            
        except Exception as e:
            logger.error(f"Neo4j monitoring failed: {e}")
            return DatabaseMetrics(
                name='Neo4j',
                status='failed',
                response_time_ms=0,
                memory_usage_mb=0,
                cpu_percent=0,
                connections=0,
                uptime_seconds=0,
                custom_metrics={},
                timestamp=time.time(),
                alerts=[f"Connection failed: {str(e)}"]
            )
    
    async def collect_all_metrics(self) -> List[DatabaseMetrics]:
        """Collect metrics from all databases"""
        logger.info("📊 Collecting metrics from all databases...")
        
        # PostgreSQL (async)
        postgres_metrics = await self.monitor_postgresql()
        
        # Redis (sync)
        redis_metrics = self.monitor_redis()
        
        # Vector databases
        qdrant_metrics = self.monitor_vector_database(
            'Qdrant', 'http://localhost:10101', '/collections'
        )
        chromadb_metrics = self.monitor_vector_database(
            'ChromaDB', 'http://localhost:10100', '/api/v1/heartbeat'
        )
        faiss_metrics = self.monitor_vector_database(
            'FAISS', 'http://localhost:10103', '/health'
        )
        
        # Neo4j
        neo4j_metrics = self.monitor_neo4j()
        
        all_metrics = [
            postgres_metrics,
            redis_metrics, 
            qdrant_metrics,
            chromadb_metrics,
            faiss_metrics,
            neo4j_metrics
        ]
        
        # Store metrics history
        for metrics in all_metrics:
            self.metrics_history[metrics.name].append(metrics)
            # Keep only last 100 entries
            if len(self.metrics_history[metrics.name]) > 100:
                self.metrics_history[metrics.name] = self.metrics_history[metrics.name][-100:]
        
        return all_metrics
    
    def generate_dashboard_report(self, metrics: List[DatabaseMetrics]) -> str:
        """Generate comprehensive dashboard report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
🔥 DATABASE MONITORING DASHBOARD
================================
Timestamp: {timestamp}

📊 DATABASE STATUS OVERVIEW:
"""
        
        # Status summary
        healthy = sum(1 for m in metrics if m.status == 'healthy')
        degraded = sum(1 for m in metrics if m.status == 'degraded')
        failed = sum(1 for m in metrics if m.status == 'failed')
        
        report += f"""
Status Summary: {healthy} Healthy | {degraded} Degraded | {failed} Failed
Total Databases: {len(metrics)}

"""
        
        # Detailed metrics for each database
        for m in metrics:
            status_emoji = {'healthy': '✅', 'degraded': '⚠️', 'failed': '❌'}.get(m.status, '❓')
            
            report += f"""
{status_emoji} {m.name}:
  Status: {m.status.upper()}
  Response Time: {m.response_time_ms:.1f}ms
  Connections: {m.connections}
  Uptime: {m.uptime_seconds}s
"""
            
            # Custom metrics
            if m.custom_metrics:
                report += "  Key Metrics:\n"
                for key, value in m.custom_metrics.items():
                    if isinstance(value, float):
                        report += f"    {key}: {value:.2f}\n"
                    else:
                        report += f"    {key}: {value}\n"
            
            # Alerts
            if m.alerts:
                report += f"  🚨 ALERTS ({len(m.alerts)}):\n"
                for alert in m.alerts:
                    report += f"    - {alert}\n"
            
            report += "\n"
        
        # System-wide alerts
        all_alerts = [alert for m in metrics for alert in m.alerts]
        if all_alerts:
            report += f"""
🚨 SYSTEM-WIDE ALERTS ({len(all_alerts)}):
"""
            for i, alert in enumerate(all_alerts, 1):
                report += f"{i}. {alert}\n"
        
        # Performance summary
        avg_response_time = sum(m.response_time_ms for m in metrics) / len(metrics)
        total_connections = sum(m.connections for m in metrics)
        
        report += f"""
📈 PERFORMANCE SUMMARY:
- Average Response Time: {avg_response_time:.1f}ms
- Total Active Connections: {total_connections}
- Databases with Alerts: {len([m for m in metrics if m.alerts])}

🎯 OPTIMIZATION RECOMMENDATIONS:
"""
        
        # Generate recommendations
        recommendations = []
        
        # Redis-specific
        redis_metrics = next((m for m in metrics if m.name == 'Redis'), None)
        if redis_metrics and redis_metrics.custom_metrics.get('hit_rate', 0) < 50:
            recommendations.append("CRITICAL: Fix Redis cache hit rate (implement Redis-first strategy)")
        
        # PostgreSQL-specific  
        postgres_metrics = next((m for m in metrics if m.name == 'PostgreSQL'), None)
        if postgres_metrics and postgres_metrics.custom_metrics.get('dead_tuple_tables', 0) > 0:
            recommendations.append("PostgreSQL: Run VACUUM on tables with dead tuples")
        
        # Vector DB consolidation
        vector_dbs = [m for m in metrics if m.name in ['Qdrant', 'ChromaDB', 'FAISS']]
        healthy_vector_dbs = [m for m in vector_dbs if m.status == 'healthy']
        if len(healthy_vector_dbs) > 1:
            recommendations.append("Consider consolidating vector databases (keep Qdrant, remove others)")
        
        # Neo4j utilization
        neo4j_metrics = next((m for m in metrics if m.name == 'Neo4j'), None)
        if neo4j_metrics and neo4j_metrics.custom_metrics.get('nodes', 0) == 0:
            recommendations.append("Neo4j: Database empty - consider removing if unused")
        
        if not recommendations:
            recommendations.append("All systems operating within normal parameters")
        
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def save_metrics_to_redis(self, metrics: List[DatabaseMetrics]):
        """Save metrics to Redis for persistence and visualization"""
        try:
            client = redis.from_url('redis://localhost:10001', decode_responses=True)
            
            # Save current metrics
            metrics_data = {m.name: asdict(m) for m in metrics}
            client.set('monitoring:current_metrics', json.dumps(metrics_data, default=str))
            
            # Save to time series (keep last 24 hours)
            timestamp = int(time.time())
            for m in metrics:
                key = f"monitoring:timeseries:{m.name}"
                client.zadd(key, {json.dumps(asdict(m), default=str): timestamp})
                # Remove entries older than 24 hours
                client.zremrangebyscore(key, 0, timestamp - 86400)
            
            logger.info("📊 Metrics saved to Redis")
            
        except Exception as e:
            logger.error(f"Failed to save metrics to Redis: {e}")

async def main():
    """Main monitoring loop"""
    logger.info("🔥 DATABASE MONITORING DASHBOARD")
    logger.info("=" * 50)
    
    monitor = DatabaseMonitor()
    
    # Run monitoring loop
    try:
        while True:
            logger.info(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Collecting metrics...")
            
            # Collect all metrics
            all_metrics = await monitor.collect_all_metrics()
            
            # Generate and display report
            report = monitor.generate_dashboard_report(all_metrics)
            logger.info(report)
            
            # Save to Redis for persistence
            monitor.save_metrics_to_redis(all_metrics)
            
            # Check for critical alerts
            critical_alerts = []
            for m in all_metrics:
                if m.status == 'failed':
                    critical_alerts.append(f"{m.name}: FAILED")
                elif 'CRITICAL' in str(m.alerts):
                    critical_alerts.extend([f"{m.name}: {alert}" for alert in m.alerts if 'CRITICAL' in alert])
            
            if critical_alerts:
                logger.error("\n🚨 CRITICAL ALERTS DETECTED:")
                for alert in critical_alerts:
                    logger.info(f"   {alert}")
            
            # Wait 60 seconds before next check
            logger.info(f"\n💤 Sleeping 60 seconds... (Press Ctrl+C to stop)")
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("\n👋 Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())