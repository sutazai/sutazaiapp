#!/usr/bin/env python3
"""
PostgreSQL Adapter for SutazAI
Provides connection pooling, query monitoring, and performance optimization
"""

import asyncio
import psycopg2
from psycopg2 import pool
import json
from aiohttp import web
from datetime import datetime
import time
from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger()

# Metrics specific to PostgreSQL
query_count = Counter('postgres_queries_total', 'Total PostgreSQL queries', ['database', 'query_type'])
query_duration = Histogram('postgres_query_duration_seconds', 'Query duration', ['database'])
connection_pool_size = Gauge('postgres_connection_pool_size', 'Connection pool size')
active_connections = Gauge('postgres_active_connections', 'Active database connections')

class PostgreSQLAdapter:
    """PostgreSQL-specific adapter with advanced features"""
    
    def __init__(self, config):
        self.config = config
        self.connection_pool = None
        self.app = web.Application()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup PostgreSQL-specific routes"""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.metrics)
        self.app.router.add_get('/databases', self.list_databases)
        self.app.router.add_get('/stats', self.database_stats)
        self.app.router.add_post('/query', self.execute_query)
        self.app.router.add_get('/connections', self.connection_info)
        self.app.router.add_get('/slow_queries', self.slow_queries)
        
    async def start(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                1,  # Min connections
                self.config.get('pool_size', 20),  # Max connections
                host=self.config['target_host'],
                port=self.config['target_port'],
                user=self.config.get('username', 'postgres'),
                password=self.config.get('password', ''),
                database=self.config.get('database', 'postgres')
            )
            
            connection_pool_size.set(self.config.get('pool_size', 20))
            logger.info("PostgreSQL adapter started", 
                       host=self.config['target_host'],
                       port=self.config['target_port'])
            
        except Exception as e:
            logger.error("Failed to create connection pool", error=str(e))
            raise
    
    async def health_check(self, request):
        """PostgreSQL health check"""
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return web.json_response({
                'status': 'healthy',
                'service': 'postgresql',
                'timestamp': datetime.utcnow().isoformat(),
                'pool_status': {
                    'min_size': self.connection_pool.minconn,
                    'max_size': self.connection_pool.maxconn
                }
            })
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return web.json_response({
                'status': 'unhealthy',
                'error': str(e)
            }, status=503)
    
    async def list_databases(self, request):
        """List all databases"""
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT datname, pg_size_pretty(pg_database_size(datname)) as size
                FROM pg_database
                WHERE datistemplate = false
                ORDER BY datname
            """)
            
            databases = [{'name': row[0], 'size': row[1]} for row in cursor.fetchall()]
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return web.json_response({'databases': databases})
            
        except Exception as e:
            logger.error("Failed to list databases", error=str(e))
            return web.json_response({'error': str(e)}, status=500)
    
    async def database_stats(self, request):
        """Get database statistics"""
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # Get general stats
            cursor.execute("""
                SELECT 
                    numbackends as active_connections,
                    xact_commit as transactions_committed,
                    xact_rollback as transactions_rolled_back,
                    blks_read as blocks_read,
                    blks_hit as blocks_hit,
                    tup_returned as tuples_returned,
                    tup_fetched as tuples_fetched,
                    tup_inserted as tuples_inserted,
                    tup_updated as tuples_updated,
                    tup_deleted as tuples_deleted
                FROM pg_stat_database
                WHERE datname = current_database()
            """)
            
            stats = cursor.fetchone()
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return web.json_response({
                'stats': {
                    'active_connections': stats[0],
                    'transactions': {
                        'committed': stats[1],
                        'rolled_back': stats[2]
                    },
                    'blocks': {
                        'read': stats[3],
                        'hit': stats[4],
                        'hit_ratio': round(stats[4] / (stats[3] + stats[4]) * 100, 2) if (stats[3] + stats[4]) > 0 else 0
                    },
                    'tuples': {
                        'returned': stats[5],
                        'fetched': stats[6],
                        'inserted': stats[7],
                        'updated': stats[8],
                        'deleted': stats[9]
                    }
                }
            })
            
        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return web.json_response({'error': str(e)}, status=500)
    
    async def execute_query(self, request):
        """Execute a query (with safety checks)"""
        try:
            data = await request.json()
            query = data.get('query')
            database = data.get('database', 'postgres')
            
            # Basic safety check (in production, use proper query validation)
            if not query or any(keyword in query.upper() for keyword in ['DROP', 'DELETE', 'TRUNCATE']):
                return web.json_response({'error': 'Query not allowed'}, status=400)
            
            start_time = time.time()
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute(query)
            
            # Get results if it's a SELECT query
            if query.strip().upper().startswith('SELECT'):
                columns = [desc[0] for desc in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            else:
                conn.commit()
                results = {'affected_rows': cursor.rowcount}
            
            duration = time.time() - start_time
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            # Record metrics
            query_type = query.strip().split()[0].upper()
            query_count.labels(database=database, query_type=query_type).inc()
            query_duration.labels(database=database).observe(duration)
            
            return web.json_response({
                'results': results,
                'execution_time': duration,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error("Query execution failed", error=str(e))
            return web.json_response({'error': str(e)}, status=500)
    
    async def connection_info(self, request):
        """Get connection pool information"""
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # Get current connections
            cursor.execute("""
                SELECT 
                    pid,
                    usename,
                    application_name,
                    client_addr,
                    state,
                    query_start,
                    state_change
                FROM pg_stat_activity
                WHERE pid <> pg_backend_pid()
                ORDER BY query_start DESC
            """)
            
            connections = []
            for row in cursor.fetchall():
                connections.append({
                    'pid': row[0],
                    'username': row[1],
                    'application': row[2],
                    'client_address': str(row[3]) if row[3] else 'local',
                    'state': row[4],
                    'query_start': row[5].isoformat() if row[5] else None,
                    'state_change': row[6].isoformat() if row[6] else None
                })
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            active_connections.set(len(connections))
            
            return web.json_response({
                'pool_info': {
                    'min_connections': self.connection_pool.minconn,
                    'max_connections': self.connection_pool.maxconn
                },
                'active_connections': connections
            })
            
        except Exception as e:
            logger.error("Failed to get connection info", error=str(e))
            return web.json_response({'error': str(e)}, status=500)
    
    async def slow_queries(self, request):
        """Get slow queries"""
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    max_time,
                    stddev_time
                FROM pg_stat_statements
                WHERE mean_time > 100  -- Queries averaging over 100ms
                ORDER BY mean_time DESC
                LIMIT 20
            """)
            
            slow_queries = []
            for row in cursor.fetchall():
                slow_queries.append({
                    'query': row[0][:200] + '...' if len(row[0]) > 200 else row[0],
                    'calls': row[1],
                    'total_time': row[2],
                    'mean_time': row[3],
                    'max_time': row[4],
                    'stddev_time': row[5]
                })
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return web.json_response({'slow_queries': slow_queries})
            
        except Exception as e:
            # pg_stat_statements might not be enabled
            if "pg_stat_statements" in str(e):
                return web.json_response({
                    'error': 'pg_stat_statements extension not enabled',
                    'hint': 'Enable pg_stat_statements in postgresql.conf'
                }, status=404)
            
            logger.error("Failed to get slow queries", error=str(e))
            return web.json_response({'error': str(e)}, status=500)
    
    async def metrics(self, request):
        """Prometheus metrics endpoint"""
        from prometheus_client import generate_latest
        metrics_data = generate_latest()
        return web.Response(text=metrics_data.decode('utf-8'),
                          content_type='text/plain; version=0.0.4')

# Integration with the main adapter framework
if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from adapter import ServiceAdapter
    
    # PostgreSQL adapter extends the base adapter
    class PostgreSQLServiceAdapter(ServiceAdapter):
        def __init__(self):
            super().__init__()
            self.pg_adapter = PostgreSQLAdapter(self.config)
            
        async def start(self):
            await super().start()
            await self.pg_adapter.start()
            
            # Mount PostgreSQL routes
            self.app.add_subapp('/pg', self.pg_adapter.app)
    
    # Run the adapter
    adapter = PostgreSQLServiceAdapter()
    asyncio.run(adapter.start())