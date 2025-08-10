#!/usr/bin/env python3
"""
SutazAI Database Health Monitor
Real-time PostgreSQL monitoring with alerting
Author: DBA Administrator
Date: 2025-08-09
"""

import asyncio
import asyncpg
import json
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/database_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseMonitor:
    """PostgreSQL Database Health Monitor"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 10000,
            'user': 'sutazai',
            'password': os.getenv('POSTGRES_PASSWORD', 'sutazai'),
            'database': 'sutazai'
        }
        self.connection: Optional[asyncpg.Connection] = None
        self.monitoring_interval = 60  # seconds
        self.alert_thresholds = {
            'max_connections_percent': 80,
            'slow_query_threshold': 5.0,  # seconds
            'disk_usage_percent': 85,
            'memory_usage_percent': 90,
            'replication_lag_seconds': 30
        }
    
    async def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = await asyncpg.connect(**self.db_config)
            logger.info("Database connection established")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            logger.info("Database connection closed")
    
    async def check_database_connectivity(self) -> Dict:
        """Test basic database connectivity and response time"""
        start_time = time.time()
        try:
            result = await self.connection.fetchval("SELECT 1")
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy' if result == 1 else 'unhealthy',
                'response_time': round(response_time, 3),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_connection_stats(self) -> Dict:
        """Monitor database connections"""
        try:
            query = """
            SELECT 
                setting::int as max_connections,
                count(*) as current_connections,
                count(*) FILTER (WHERE state = 'active') as active_connections,
                count(*) FILTER (WHERE state = 'idle') as idle_connections,
                count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
            FROM pg_settings 
            CROSS JOIN pg_stat_activity 
            WHERE name = 'max_connections'
            GROUP BY setting;
            """
            
            result = await self.connection.fetchrow(query)
            
            if result:
                usage_percent = (result['current_connections'] / result['max_connections']) * 100
                
                return {
                    'max_connections': result['max_connections'],
                    'current_connections': result['current_connections'],
                    'active_connections': result['active_connections'],
                    'idle_connections': result['idle_connections'],
                    'idle_in_transaction': result['idle_in_transaction'],
                    'usage_percent': round(usage_percent, 2),
                    'status': 'warning' if usage_percent > self.alert_thresholds['max_connections_percent'] else 'healthy'
                }
            else:
                return {'status': 'error', 'error': 'No connection data available'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def check_slow_queries(self) -> Dict:
        """Monitor slow running queries"""
        try:
            query = """
            SELECT 
                pid,
                usename,
                application_name,
                client_addr,
                query_start,
                state,
                EXTRACT(EPOCH FROM (now() - query_start)) as duration,
                left(query, 100) as query_preview
            FROM pg_stat_activity 
            WHERE state = 'active' 
            AND query_start IS NOT NULL
            AND EXTRACT(EPOCH FROM (now() - query_start)) > %s
            ORDER BY duration DESC
            LIMIT 10;
            """
            
            slow_queries = await self.connection.fetch(query, self.alert_thresholds['slow_query_threshold'])
            
            return {
                'slow_query_count': len(slow_queries),
                'threshold_seconds': self.alert_thresholds['slow_query_threshold'],
                'slow_queries': [
                    {
                        'pid': q['pid'],
                        'user': q['usename'],
                        'duration': round(q['duration'], 2),
                        'query_preview': q['query_preview']
                    } for q in slow_queries
                ],
                'status': 'warning' if len(slow_queries) > 0 else 'healthy'
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def check_database_size(self) -> Dict:
        """Monitor database and table sizes"""
        try:
            # Database size
            db_size_query = """
            SELECT 
                pg_size_pretty(pg_database_size(current_database())) as database_size,
                pg_database_size(current_database()) as database_size_bytes
            """
            db_size = await self.connection.fetchrow(db_size_query)
            
            # Table sizes
            table_sizes_query = """
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            LIMIT 10;
            """
            table_sizes = await self.connection.fetch(table_sizes_query)
            
            return {
                'database_size': db_size['database_size'],
                'database_size_bytes': db_size['database_size_bytes'],
                'largest_tables': [
                    {
                        'table': f"{t['schemaname']}.{t['tablename']}",
                        'size': t['size'],
                        'size_bytes': t['size_bytes']
                    } for t in table_sizes
                ],
                'status': 'healthy'
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def check_locks(self) -> Dict:
        """Monitor database locks"""
        try:
            locks_query = """
            SELECT 
                mode,
                locktype,
                count(*) as lock_count
            FROM pg_locks 
            WHERE NOT granted
            GROUP BY mode, locktype
            ORDER BY lock_count DESC;
            """
            
            locks = await self.connection.fetch(locks_query)
            
            total_locks = sum(lock['lock_count'] for lock in locks)
            
            return {
                'total_waiting_locks': total_locks,
                'lock_types': [
                    {
                        'mode': lock['mode'],
                        'type': lock['locktype'],
                        'count': lock['lock_count']
                    } for lock in locks
                ],
                'status': 'warning' if total_locks > 10 else 'healthy'
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def check_replication_status(self) -> Dict:
        """Monitor replication lag (if applicable)"""
        try:
            # Check if this is a replica
            is_replica_query = "SELECT pg_is_in_recovery()"
            is_replica = await self.connection.fetchval(is_replica_query)
            
            if is_replica:
                lag_query = """
                SELECT 
                    EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) as lag_seconds
                """
                lag = await self.connection.fetchval(lag_query)
                
                return {
                    'is_replica': True,
                    'replication_lag_seconds': round(lag, 2) if lag else 0,
                    'status': 'warning' if lag and lag > self.alert_thresholds['replication_lag_seconds'] else 'healthy'
                }
            else:
                # Check for connected replicas (if this is a master)
                replicas_query = """
                SELECT 
                    client_addr,
                    state,
                    sent_lsn,
                    write_lsn,
                    flush_lsn,
                    replay_lsn
                FROM pg_stat_replication;
                """
                replicas = await self.connection.fetch(replicas_query)
                
                return {
                    'is_replica': False,
                    'connected_replicas': len(replicas),
                    'replica_details': [
                        {
                            'client_addr': str(r['client_addr']),
                            'state': r['state']
                        } for r in replicas
                    ],
                    'status': 'healthy'
                }
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def record_health_metrics(self, metrics: Dict):
        """Record health metrics in database"""
        try:
            # Insert into agent_health table for database monitoring
            db_agent_query = "SELECT id FROM agents WHERE name = 'database-monitor' LIMIT 1"
            db_agent = await self.connection.fetchval(db_agent_query)
            
            if not db_agent:
                # Create database monitor agent entry
                insert_agent_query = """
                INSERT INTO agents (name, type, description, endpoint, port, capabilities)
                VALUES ('database-monitor', 'monitoring', 'Database health monitoring agent', 'internal', 0, '["db_monitoring", "health_checks"]')
                RETURNING id;
                """
                db_agent = await self.connection.fetchval(insert_agent_query)
            
            # Insert health metrics
            insert_health_query = """
            INSERT INTO agent_health (
                agent_id, status, last_heartbeat, cpu_usage, memory_usage, 
                response_time, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            status = 'healthy'
            if any(m.get('status') == 'error' for m in metrics.values()):
                status = 'error'
            elif any(m.get('status') == 'warning' for m in metrics.values()):
                status = 'warning'
            
            await self.connection.execute(
                insert_health_query,
                db_agent,
                status,
                datetime.now(),
                0.0,  # CPU usage - would need system monitoring
                0.0,  # Memory usage - would need system monitoring  
                metrics.get('connectivity', {}).get('response_time', 0),
                json.dumps(metrics)
            )
            
        except Exception as e:
            logger.error(f"Failed to record health metrics: {e}")
    
    async def generate_alert(self, alert_type: str, severity: str, title: str, description: str, metadata: Dict = None):
        """Generate system alert for database issues"""
        try:
            alert_query = """
            INSERT INTO system_alerts (alert_type, severity, title, description, source, metadata)
            VALUES ($1, $2, $3, $4, 'database_monitor', $5)
            """
            
            await self.connection.execute(
                alert_query,
                alert_type,
                severity,
                title,
                description,
                json.dumps(metadata or {})
            )
            
            logger.warning(f"ALERT Generated: {severity.upper()} - {title}")
            
        except Exception as e:
            logger.error(f"Failed to generate alert: {e}")
    
    async def run_health_check(self) -> Dict:
        """Run comprehensive database health check"""
        logger.info("Starting database health check...")
        
        health_metrics = {}
        
        # Connectivity check
        health_metrics['connectivity'] = await self.check_database_connectivity()
        
        # Connection stats
        health_metrics['connections'] = await self.check_connection_stats()
        
        # Slow queries
        health_metrics['slow_queries'] = await self.check_slow_queries()
        
        # Database size
        health_metrics['database_size'] = await self.check_database_size()
        
        # Locks
        health_metrics['locks'] = await self.check_locks()
        
        # Replication
        health_metrics['replication'] = await self.check_replication_status()
        
        # Record metrics
        await self.record_health_metrics(health_metrics)
        
        # Generate alerts if needed
        await self.check_and_generate_alerts(health_metrics)
        
        logger.info("Database health check completed")
        return health_metrics
    
    async def check_and_generate_alerts(self, metrics: Dict):
        """Check metrics and generate alerts as needed"""
        
        # Connection usage alert
        if metrics['connections'].get('usage_percent', 0) > self.alert_thresholds['max_connections_percent']:
            await self.generate_alert(
                'high_connection_usage',
                'warning',
                'High Database Connection Usage',
                f"Database connection usage at {metrics['connections']['usage_percent']:.1f}%",
                {'current_connections': metrics['connections']['current_connections']}
            )
        
        # Slow queries alert
        if metrics['slow_queries'].get('slow_query_count', 0) > 0:
            await self.generate_alert(
                'slow_queries_detected',
                'warning',
                'Slow Queries Detected',
                f"Found {metrics['slow_queries']['slow_query_count']} slow running queries",
                {'slow_queries': metrics['slow_queries']['slow_queries']}
            )
        
        # Locks alert
        if metrics['locks'].get('total_waiting_locks', 0) > 10:
            await self.generate_alert(
                'high_lock_contention',
                'warning',
                'High Lock Contention',
                f"Database has {metrics['locks']['total_waiting_locks']} waiting locks",
                {'lock_details': metrics['locks']['lock_types']}
            )
    
    async def start_monitoring(self):
        """Start continuous database monitoring"""
        logger.info(f"Starting database monitoring with {self.monitoring_interval}s interval...")
        
        while True:
            try:
                if not self.connection:
                    if not await self.connect():
                        logger.error("Failed to connect to database, retrying in 30 seconds...")
                        await asyncio.sleep(30)
                        continue
                
                await self.run_health_check()
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await self.disconnect()
                
            await asyncio.sleep(self.monitoring_interval)

async def main():
    """Main monitoring loop"""
    monitor = DatabaseMonitor()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
    finally:
        await monitor.disconnect()

if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs('/opt/sutazaiapp/logs', exist_ok=True)
    asyncio.run(main())