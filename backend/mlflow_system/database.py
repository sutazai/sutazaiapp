"""
Database management for MLflow tracking system
Optimized for high-volume experiment tracking
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

import asyncpg
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine

from .config import MLflowConfig, mlflow_config


logger = logging.getLogger(__name__)


class MLflowDatabase:
    """Database manager for MLflow tracking system"""
    
    def __init__(self, config: MLflowConfig = None):
        self.config = config or mlflow_config
        self.engine: Optional[Engine] = None
        self.async_pool: Optional[asyncpg.Pool] = None
        
        # Parse database URL
        self.db_config = self._parse_database_url(self.config.backend_store_uri)
    
    def _parse_database_url(self, url: str) -> Dict[str, str]:
        """Parse PostgreSQL connection URL"""
        # Format: postgresql://user:password@host:port/database
        if not url.startswith('postgresql://'):
            raise ValueError("Only PostgreSQL is supported for MLflow backend store")
        
        # Extract components
        url_parts = url.replace('postgresql://', '').split('/')
        auth_host = url_parts[0]
        database = url_parts[1] if len(url_parts) > 1 else 'mlflow'
        
        if '@' in auth_host:
            auth, host_port = auth_host.split('@')
            if ':' in auth:
                user, password = auth.split(':', 1)
            else:
                user, password = auth, ''
        else:
            host_port = auth_host
            user, password = 'mlflow', 'mlflow_secure_pwd'
        
        if ':' in host_port:
            host, port = host_port.split(':')
            port = int(port)
        else:
            host, port = host_port, 5432
        
        return {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
    
    async def initialize(self):
        """Initialize database connections and schema"""
        logger.info("Initializing MLflow database...")
        
        try:
            # Create SQLAlchemy engine for MLflow
            self.engine = create_engine(
                self.config.backend_store_uri,
                poolclass=QueuePool,
                pool_size=self.config.db_pool_size,
                max_overflow=self.config.db_max_overflow,
                pool_timeout=self.config.db_pool_timeout,
                pool_recycle=self.config.db_pool_recycle,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Create async connection pool for high-performance operations
            self.async_pool = await asyncpg.create_pool(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                min_size=5,
                max_size=self.config.db_pool_size,
                command_timeout=30
            )
            
            # Initialize MLflow schema
            await self._initialize_mlflow_schema()
            
            # Create custom indexes for performance
            await self._create_performance_indexes()
            
            logger.info("MLflow database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _initialize_mlflow_schema(self):
        """Initialize MLflow database schema"""
        try:
            # MLflow will create its own schema when first used
            # We just need to ensure the database exists and is accessible
            
            async with self.async_pool.acquire() as conn:
                # Check if MLflow tables exist
                result = await conn.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE '%mlflow%'
                """)
                
                if result:
                    logger.info(f"Found {len(result)} MLflow tables in database")
                else:
                    logger.info("MLflow schema will be created on first use")
                    
        except Exception as e:
            logger.error(f"Failed to check MLflow schema: {e}")
            raise
    
    async def _create_performance_indexes(self):
        """Create additional indexes for better performance"""
        indexes = [
            # Experiment queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_experiments_name 
            ON experiments(name) WHERE lifecycle_stage = 'active'
            """,
            
            # Run queries by experiment
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_runs_experiment_status 
            ON runs(experiment_id, status) WHERE status = 'FINISHED'
            """,
            
            # Run queries by timestamp
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_runs_start_time 
            ON runs(start_time DESC)
            """,
            
            # Metric queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_run_key 
            ON metrics(run_uuid, key)
            """,
            
            # Parameter queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_params_run_key 
            ON params(run_uuid, key)
            """,
            
            # Tag queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tags_run_key 
            ON tags(run_uuid, key)
            """,
            
            # Model registry queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_versions_name_version 
            ON model_versions(name, version)
            """,
        ]
        
        try:
            async with self.async_pool.acquire() as conn:
                for index_sql in indexes:
                    try:
                        await conn.execute(index_sql)
                        logger.debug(f"Created index: {index_sql.split()[5]}")
                    except Exception as e:
                        # Index might already exist or table might not exist yet
                        logger.debug(f"Index creation skipped: {e}")
                        
            logger.info("Performance indexes created/verified")
            
        except Exception as e:
            logger.error(f"Failed to create performance indexes: {e}")
    
    async def create_experiment_batch(self, experiments: List[Dict]) -> List[str]:
        """Create multiple experiments in batch for better performance"""
        experiment_ids = []
        
        try:
            async with self.async_pool.acquire() as conn:
                async with conn.transaction():
                    for experiment in experiments:
                        # This would need to be integrated with MLflow's experiment creation
                        # For now, this is a placeholder for batch operations
                        pass
                        
            return experiment_ids
            
        except Exception as e:
            logger.error(f"Failed to create experiment batch: {e}")
            raise
    
    async def get_experiment_metrics_summary(self, experiment_id: str) -> Dict:
        """Get aggregated metrics for an experiment"""
        try:
            async with self.async_pool.acquire() as conn:
                # Get run count
                run_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM runs WHERE experiment_id = $1",
                    experiment_id
                )
                
                # Get average metrics (if runs table exists)
                try:
                    avg_metrics = await conn.fetch("""
                        SELECT 
                            m.key,
                            AVG(m.value) as avg_value,
                            MIN(m.value) as min_value,
                            MAX(m.value) as max_value,
                            COUNT(*) as count
                        FROM metrics m
                        JOIN runs r ON m.run_uuid = r.run_uuid
                        WHERE r.experiment_id = $1
                        GROUP BY m.key
                    """, experiment_id)
                    
                    metrics_summary = {
                        row['key']: {
                            'avg': float(row['avg_value']),
                            'min': float(row['min_value']),
                            'max': float(row['max_value']),
                            'count': row['count']
                        }
                        for row in avg_metrics
                    }
                except Exception:
                    metrics_summary = {}
                
                return {
                    'experiment_id': experiment_id,
                    'run_count': run_count,
                    'metrics_summary': metrics_summary
                }
                
        except Exception as e:
            logger.error(f"Failed to get experiment metrics summary: {e}")
            return {'experiment_id': experiment_id, 'run_count': 0, 'metrics_summary': {}}
    
    async def cleanup_old_runs(self, days_old: int = 90) -> int:
        """Clean up old completed runs"""
        try:
            async with self.async_pool.acquire() as conn:
                # First, get runs to delete
                old_runs = await conn.fetch("""
                    SELECT run_uuid 
                    FROM runs 
                    WHERE status = 'FINISHED' 
                    AND end_time < NOW() - INTERVAL '%s days'
                """, days_old)
                
                if not old_runs:
                    return 0
                
                run_uuids = [row['run_uuid'] for row in old_runs]
                
                # Delete in transaction
                async with conn.transaction():
                    # Delete metrics
                    await conn.execute(
                        "DELETE FROM metrics WHERE run_uuid = ANY($1)",
                        run_uuids
                    )
                    
                    # Delete params
                    await conn.execute(
                        "DELETE FROM params WHERE run_uuid = ANY($1)",
                        run_uuids
                    )
                    
                    # Delete tags
                    await conn.execute(
                        "DELETE FROM tags WHERE run_uuid = ANY($1)",
                        run_uuids
                    )
                    
                    # Delete runs
                    deleted_count = await conn.execute(
                        "DELETE FROM runs WHERE run_uuid = ANY($1)",
                        run_uuids
                    )
                
                logger.info(f"Cleaned up {len(run_uuids)} old runs")
                return len(run_uuids)
                
        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")
            return 0
    
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            async with self.async_pool.acquire() as conn:
                # Get table sizes
                table_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename LIKE '%mlflow%' OR tablename IN ('experiments', 'runs', 'metrics', 'params', 'tags')
                    ORDER BY size_bytes DESC
                """)
                
                # Get connection stats
                connection_stats = await conn.fetchrow("""
                    SELECT 
                        numbackends as active_connections,
                        xact_commit as committed_transactions,
                        xact_rollback as rolled_back_transactions,
                        blks_read as blocks_read,
                        blks_hit as blocks_hit
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                
                return {
                    'table_stats': [dict(row) for row in table_stats],
                    'connection_stats': dict(connection_stats) if connection_stats else {},
                    'pool_size': self.async_pool.get_size(),
                    'pool_available': self.async_pool.get_idle_size()
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def close(self):
        """Close database connections"""
        try:
            if self.async_pool:
                await self.async_pool.close()
            
            if self.engine:
                self.engine.dispose()
            
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Global database instance
mlflow_database = MLflowDatabase()