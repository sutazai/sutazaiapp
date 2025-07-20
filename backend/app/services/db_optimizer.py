# backend/app/services/db_optimizer.py
from typing import List, Dict, Any
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

class DatabaseOptimizer:
    """Database performance optimization service"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.slow_query_threshold = 1000  # milliseconds
        
    async def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """Identify and analyze slow queries"""
        query = text("""
            SELECT 
                query,
                calls,
                total_time,
                mean_time,
                min_time,
                max_time,
                stddev_time
            FROM pg_stat_statements
            WHERE mean_time > :threshold
            ORDER BY mean_time DESC
            LIMIT 20
        """)
        
        result = await self.db.execute(
            query,
            {"threshold": self.slow_query_threshold}
        )
        
        slow_queries = []
        for row in result:
            slow_queries.append({
                "query": row.query,
                "calls": row.calls,
                "total_time": row.total_time,
                "mean_time": row.mean_time,
                "optimization_suggestions": await self._get_optimization_suggestions(row.query)
            })
        
        return slow_queries
    
    async def _get_optimization_suggestions(self, query: str) -> List[str]:
        """Get optimization suggestions for a query (placeholder)."""
        suggestions = []
        if "SELECT" in query.upper() and "WHERE" not in query.upper():
            suggestions.append("Consider adding a WHERE clause to filter results.")
        # In a real implementation, this would use EXPLAIN ANALYZE
        return suggestions

    async def optimize_indexes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze and suggest index optimizations"""
        
        # Find missing indexes
        missing_indexes_query = text("""
            SELECT 
                relname AS tablename,
                seq_scan - idx_scan AS potential_missing_indexes
            FROM pg_stat_user_tables
            WHERE (seq_scan - idx_scan) > 1000 AND schemaname = 'public'
            ORDER BY potential_missing_indexes DESC;
        """)
        
        # Find unused indexes
        unused_indexes_query = text("""
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE idx_scan = 0
            AND schemaname = 'public'
        """)
        
        missing_result = await self.db.execute(missing_indexes_query)
        unused_result = await self.db.execute(unused_indexes_query)
        
        recommendations: Dict[str, List[Dict[str, Any]]] = {
            "create_indexes": [],
            "drop_indexes": []
        }
        
        # Process missing indexes
        for row in missing_result:
            recommendations["create_indexes"].append({
                "table": row.tablename,
                "reason": f"High number of sequential scans ({row.potential_missing_indexes}) suggests a missing index.",
                "sql": f"Consider adding an index on frequently queried columns of {row.tablename}."
            })
        
        # Process unused indexes
        for row in unused_result:
            recommendations["drop_indexes"].append({
                "index": row.indexname,
                "table": f"{row.schemaname}.{row.tablename}",
                "reason": "No index scans recorded",
                "sql": f"DROP INDEX IF EXISTS {row.schemaname}.{row.indexname};"
            })
        
        return recommendations
    
    async def vacuum_analyze(self, tables: List[str] = None):
        """Run VACUUM ANALYZE on specified tables or all tables"""
        if tables:
            for table in tables:
                await self.db.execute(text(f"VACUUM ANALYZE {table}"))
        else:
            await self.db.execute(text("VACUUM ANALYZE"))
    
    async def update_statistics(self):
        """Update table statistics for query planner"""
        query = text("""
            SELECT 
                schemaname,
                tablename
            FROM pg_tables
            WHERE schemaname = 'public'
        """)
        
        result = await self.db.execute(query)
        
        for row in result:
            analyze_query = text(f"ANALYZE {row.schemaname}.{row.tablename}")
            await self.db.execute(analyze_query)
    
    async def optimize_connection_pool(self) -> Dict[str, Any]:
        """Analyze and optimize connection pool settings"""
        
        # Get current connection stats
        stats_query = text("""
            SELECT
                numbackends,
                xact_commit,
                xact_rollback,
                blks_read,
                blks_hit,
                tup_returned,
                tup_fetched,
                tup_inserted,
                tup_updated,
                tup_deleted
            FROM pg_stat_database
            WHERE datname = current_database()
        """)
        
        result = await self.db.execute(stats_query)
        stats = result.first()
        
        # Calculate metrics
        if stats and stats.blks_read + stats.blks_hit > 0:
            cache_hit_ratio = stats.blks_hit / (stats.blks_read + stats.blks_hit)
        else:
            cache_hit_ratio = 0
        
        recommendations = {
            "current_connections": stats.numbackends if stats else 0,
            "cache_hit_ratio": cache_hit_ratio,
            "recommendations": []
        }
        
        # Make recommendations
        if stats and stats.numbackends > 100:
            recommendations["recommendations"].append({
                "issue": "High connection count",
                "suggestion": "Consider connection pooling with pgBouncer"
            })
        
        if cache_hit_ratio < 0.9:
            recommendations["recommendations"].append({
                "issue": "Low cache hit ratio",
                "suggestion": "Increase shared_buffers in PostgreSQL configuration"
            })
        
        return recommendations
