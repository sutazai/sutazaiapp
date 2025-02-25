#!/usr/bin/env python3
"""
SutazAI Database Manager
------------------------
Provides centralized database connection management and query handling
for the SutazAI system.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

try:
    import psycopg2
    from psycopg2.extras import DictCursor, execute_values
    from psycopg2.pool import ThreadedConnectionPool
except ImportError:
    psycopg2 = None
    logging.warning("psycopg2 not installed, PostgreSQL functionality will be limited")

try:
    import sqlite3
except ImportError:
    sqlite3 = None
    logging.warning("sqlite3 not available, SQLite functionality will be disabled")


class DatabaseManager:
    """Centralized database connection manager for SutazAI."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the database manager with configuration.
        
        Args:
            config: Dictionary containing database configuration
                database_type: 'postgresql' or 'sqlite'
                connection_params: Dict of connection parameters
                    For PostgreSQL: host, port, dbname, user, password
                    For SQLite: database path
                pool_min_conn: Minimum connections in pool (PostgreSQL only)
                pool_max_conn: Maximum connections in pool (PostgreSQL only)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.conn_pool = None
        
        db_type = config.get("database_type", "sqlite").lower()
        self.db_type = db_type
        
        if db_type == "postgresql":
            if psycopg2 is None:
                raise ImportError("psycopg2 is required for PostgreSQL connections")
            self._setup_postgres_pool()
        elif db_type == "sqlite":
            if sqlite3 is None:
                raise ImportError("sqlite3 is required for SQLite connections")
            self.sqlite_path = config.get("connection_params", {}).get(
                "database", ":memory:"
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
            
    def _setup_postgres_pool(self) -> None:
        """Set up the PostgreSQL connection pool."""
        conn_params = self.config.get("connection_params", {})
        min_conn = self.config.get("pool_min_conn", 1)
        max_conn = self.config.get("pool_max_conn", 10)
        
        self.conn_pool = ThreadedConnectionPool(
            min_conn,
            max_conn,
            host=conn_params.get("host", "localhost"),
            port=conn_params.get("port", 5432),
            dbname=conn_params.get("dbname", "sutazai"),
            user=conn_params.get("user", "postgres"),
            password=conn_params.get("password", ""),
        )
        
    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """
        Get a database connection from the pool or create a new SQLite connection.
        
        Yields:
            Connection object (psycopg2 or sqlite3)
        """
        connection = None
        cursor = None
        
        try:
            if self.db_type == "postgresql":
                connection = self.conn_pool.getconn()
                cursor = connection.cursor(cursor_factory=DictCursor)
            else:  # sqlite
                connection = sqlite3.connect(self.sqlite_path)
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                
            yield cursor
            connection.commit()
            
        except Exception as e:
            if connection:
                connection.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise
            
        finally:
            if cursor:
                cursor.close()
            if connection:
                if self.db_type == "postgresql":
                    self.conn_pool.putconn(connection)
                else:
                    connection.close()
    
    def execute_query(
        self, query: str, params: Optional[Union[Tuple, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a database query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters (tuple, dict, or None)
            
        Returns:
            List of dictionaries containing the query results
        """
        with self.get_connection() as cursor:
            cursor.execute(query, params or ())
            
            if cursor.description:
                if self.db_type == "postgresql":
                    return [dict(row) for row in cursor.fetchall()]
                else:  # sqlite
                    columns = [col[0] for col in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []
    
    def execute_batch(
        self, query: str, params_list: List[Union[Tuple, Dict[str, Any]]]
    ) -> None:
        """
        Execute a batch operation.
        
        Args:
            query: SQL query template
            params_list: List of parameter sets for batch execution
        """
        with self.get_connection() as cursor:
            if self.db_type == "postgresql":
                execute_values(cursor, query, params_list)
            else:  # sqlite
                cursor.executemany(query, params_list)
    
    def close(self) -> None:
        """Close all database connections."""
        if self.db_type == "postgresql" and self.conn_pool:
            self.conn_pool.closeall()
            self.logger.info("All database connections closed")


def get_database_manager(config_file: Optional[str] = None) -> DatabaseManager:
    """
    Factory function to create a DatabaseManager instance.
    
    Args:
        config_file: Path to config file (optional)
        
    Returns:
        Configured DatabaseManager instance
    """
    # Default config
    config = {
        "database_type": "sqlite",
        "connection_params": {
            "database": os.path.join(os.path.dirname(__file__), "../database/sutazai.db")
        },
    }
    
    # TODO: Load config from file if provided
    
    return DatabaseManager(config)


# Singleton instance
_db_manager = None


def get_db() -> DatabaseManager:
    """
    Get the singleton DatabaseManager instance.
    
    Returns:
        Global DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = get_database_manager()
    return _db_manager
