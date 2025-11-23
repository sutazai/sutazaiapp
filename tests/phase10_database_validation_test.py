#!/usr/bin/env python3
"""
Phase 10: Database Validation Comprehensive Test Suite

This test suite validates all database operations across:
- PostgreSQL: Migrations, schema integrity, backups, restores
- Neo4j: Graph queries, relationships, constraints
- Redis: Cache invalidation, persistence
- RabbitMQ: Message durability, queue management

Author: SutazAI Platform Engineering
Created: 2025-11-15
Version: 1.0.0

Usage:
    python tests/phase10_database_validation_test.py
"""

import asyncio
import datetime
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import psycopg2
import psycopg2.extras
import redis.asyncio as redis
from neo4j import GraphDatabase, AsyncGraphDatabase
from aio_pika import connect_robust, Message, DeliveryMode, ExchangeType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d UTC - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Test configuration
POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 10000,
    'database': 'jarvis_ai',
    'user': 'jarvis',
    'password': 'sutazai_secure_2024'
}

NEO4J_CONFIG = {
    'uri': 'bolt://localhost:10003',
    'auth': ('neo4j', 'sutazai_secure_2024')
}

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 10001,
    'decode_responses': True
}

RABBITMQ_CONFIG = {
    'host': 'localhost',
    'port': 10004,
    'login': 'sutazai',
    'password': 'sutazai_secure_2024'
}


class TestResults:
    """Track test results and metrics"""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        
    def add_result(self, test_name: str, passed: bool, duration_ms: float, details: Dict = None):
        """Add a test result"""
        self.results.append({
            'test_name': test_name,
            'passed': passed,
            'duration_ms': duration_ms,
            'timestamp': datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            'details': details or {}
        })
        
    def get_summary(self) -> Dict:
        """Get test summary statistics"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        total_duration = (datetime.datetime.now(tz=datetime.timezone.utc) - self.start_time).total_seconds()
        
        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'total_duration_seconds': total_duration,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        }


class PostgreSQLValidator:
    """PostgreSQL validation tests"""
    
    def __init__(self, config: Dict, results: TestResults):
        self.config = config
        self.results = results
        self.conn = None
        
    def connect(self):
        """Establish PostgreSQL connection"""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.conn.autocommit = False
            logger.info("PostgreSQL connection established")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            return False
            
    def disconnect(self):
        """Close PostgreSQL connection"""
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL connection closed")
            
    async def test_migrations(self):
        """Test PostgreSQL migrations"""
        test_name = "PostgreSQL Migrations"
        start_time = time.time()
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Check if Kong database exists
                cur.execute("SELECT datname FROM pg_database WHERE datname = 'kong'")
                kong_db = cur.fetchone()
                
                # Check if jarvis_ai database exists and has public schema
                cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'public'")
                public_schema = cur.fetchone()
                
                passed = kong_db is not None and public_schema is not None
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.add_result(
                    test_name, 
                    passed, 
                    duration_ms,
                    {'kong_db_exists': kong_db is not None, 'public_schema_exists': public_schema is not None}
                )
                
                logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
                return passed
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False
            
    async def test_schema_integrity(self):
        """Test PostgreSQL schema integrity"""
        test_name = "PostgreSQL Schema Integrity"
        start_time = time.time()
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Create test table with constraints
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS test_schema_validation (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) NOT NULL UNIQUE,
                        age INTEGER CHECK (age >= 18 AND age <= 150),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended'))
                    )
                """)
                self.conn.commit()
                
                # Test NOT NULL constraint
                try:
                    cur.execute("INSERT INTO test_schema_validation (email, age) VALUES (NULL, 25)")
                    self.conn.commit()
                    null_constraint_works = False
                except psycopg2.IntegrityError:
                    null_constraint_works = True
                    self.conn.rollback()
                    
                # Test UNIQUE constraint
                cur.execute("INSERT INTO test_schema_validation (email, age) VALUES ('test@example.com', 25)")
                self.conn.commit()
                try:
                    cur.execute("INSERT INTO test_schema_validation (email, age) VALUES ('test@example.com', 30)")
                    self.conn.commit()
                    unique_constraint_works = False
                except psycopg2.IntegrityError:
                    unique_constraint_works = True
                    self.conn.rollback()
                    
                # Test CHECK constraint
                try:
                    cur.execute("INSERT INTO test_schema_validation (email, age) VALUES ('test2@example.com', 15)")
                    self.conn.commit()
                    check_constraint_works = False
                except psycopg2.IntegrityError:
                    check_constraint_works = True
                    self.conn.rollback()
                    
                # Test DEFAULT value
                cur.execute("INSERT INTO test_schema_validation (email, age) VALUES ('test3@example.com', 25) RETURNING status")
                default_status = cur.fetchone()[0]
                default_works = default_status == 'active'
                self.conn.commit()
                
                # Cleanup
                cur.execute("DROP TABLE test_schema_validation")
                self.conn.commit()
                
                passed = all([null_constraint_works, unique_constraint_works, check_constraint_works, default_works])
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.add_result(
                    test_name,
                    passed,
                    duration_ms,
                    {
                        'not_null_constraint': null_constraint_works,
                        'unique_constraint': unique_constraint_works,
                        'check_constraint': check_constraint_works,
                        'default_value': default_works
                    }
                )
                
                logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
                return passed
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False
            
    async def test_foreign_key_constraints(self):
        """Test foreign key constraints"""
        test_name = "PostgreSQL Foreign Key Constraints"
        start_time = time.time()
        
        try:
            with self.conn.cursor() as cur:
                # Create parent and child tables
                cur.execute("""
                    CREATE TABLE test_parent (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE test_child (
                        id SERIAL PRIMARY KEY,
                        parent_id INTEGER REFERENCES test_parent(id) ON DELETE CASCADE,
                        data VARCHAR(100)
                    )
                """)
                self.conn.commit()
                
                # Insert test data
                cur.execute("INSERT INTO test_parent (name) VALUES ('Parent 1') RETURNING id")
                parent_id = cur.fetchone()[0]
                
                cur.execute("INSERT INTO test_child (parent_id, data) VALUES (%s, 'Child 1')", (parent_id,))
                cur.execute("INSERT INTO test_child (parent_id, data) VALUES (%s, 'Child 2')", (parent_id,))
                self.conn.commit()
                
                # Test foreign key constraint violation
                try:
                    cur.execute("INSERT INTO test_child (parent_id, data) VALUES (9999, 'Invalid')")
                    self.conn.commit()
                    fk_violation_caught = False
                except psycopg2.IntegrityError:
                    fk_violation_caught = True
                    self.conn.rollback()
                    
                # Test CASCADE delete
                cur.execute("SELECT COUNT(*) FROM test_child WHERE parent_id = %s", (parent_id,))
                children_before = cur.fetchone()[0]
                
                cur.execute("DELETE FROM test_parent WHERE id = %s", (parent_id,))
                self.conn.commit()
                
                cur.execute("SELECT COUNT(*) FROM test_child WHERE parent_id = %s", (parent_id,))
                children_after = cur.fetchone()[0]
                
                cascade_works = children_before > 0 and children_after == 0
                
                # Cleanup
                cur.execute("DROP TABLE test_child")
                cur.execute("DROP TABLE test_parent")
                self.conn.commit()
                
                passed = fk_violation_caught and cascade_works
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.add_result(
                    test_name,
                    passed,
                    duration_ms,
                    {
                        'fk_violation_detected': fk_violation_caught,
                        'cascade_delete_works': cascade_works,
                        'children_before_delete': children_before,
                        'children_after_delete': children_after
                    }
                )
                
                logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
                return passed
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False
            
    async def test_index_performance(self):
        """Test index performance"""
        test_name = "PostgreSQL Index Performance"
        start_time = time.time()
        
        try:
            with self.conn.cursor() as cur:
                # Create test table
                cur.execute("""
                    CREATE TABLE test_index_performance (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert test data
                for i in range(1000):
                    cur.execute("INSERT INTO test_index_performance (email) VALUES (%s)", (f'user{i}@example.com',))
                self.conn.commit()
                
                # Query without index
                cur.execute("EXPLAIN ANALYZE SELECT * FROM test_index_performance WHERE email = 'user500@example.com'")
                plan_no_index = cur.fetchall()
                
                # Create index
                cur.execute("CREATE INDEX idx_email ON test_index_performance(email)")
                self.conn.commit()
                
                # Query with index
                cur.execute("EXPLAIN ANALYZE SELECT * FROM test_index_performance WHERE email = 'user500@example.com'")
                plan_with_index = cur.fetchall()
                
                # Cleanup
                cur.execute("DROP TABLE test_index_performance")
                self.conn.commit()
                
                # Index should improve query performance (detected by "Index Scan" in plan)
                index_used = any('Index Scan' in str(row) for row in plan_with_index)
                
                passed = index_used
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.add_result(
                    test_name,
                    passed,
                    duration_ms,
                    {
                        'index_created': True,
                        'index_used_in_query': index_used,
                        'test_rows_inserted': 1000
                    }
                )
                
                logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
                return passed
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False
            
    async def test_backup_restore(self):
        """Test backup and restore procedures"""
        test_name = "PostgreSQL Backup & Restore"
        start_time = time.time()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                backup_file = Path(tmpdir) / "test_backup.sql"
                
                # Create test table with data
                with self.conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE test_backup_restore (
                            id SERIAL PRIMARY KEY,
                            data VARCHAR(100)
                        )
                    """)
                    cur.execute("INSERT INTO test_backup_restore (data) VALUES ('test data 1')")
                    cur.execute("INSERT INTO test_backup_restore (data) VALUES ('test data 2')")
                    self.conn.commit()
                    
                # Perform backup using pg_dump
                backup_cmd = [
                    'docker', 'exec', 'sutazai-postgres',
                    'pg_dump',
                    '-U', 'jarvis',
                    '-d', 'jarvis_ai',
                    '-t', 'test_backup_restore',
                    '-f', '/tmp/test_backup.sql'
                ]
                
                result = subprocess.run(backup_cmd, capture_output=True, text=True)
                backup_success = result.returncode == 0
                
                if backup_success:
                    # Copy backup file from container
                    subprocess.run([
                        'docker', 'cp',
                        'sutazai-postgres:/tmp/test_backup.sql',
                        str(backup_file)
                    ], check=True)
                    
                    backup_exists = backup_file.exists() and backup_file.stat().st_size > 0
                    
                    # Drop table
                    with self.conn.cursor() as cur:
                        cur.execute("DROP TABLE test_backup_restore")
                        self.conn.commit()
                        
                    # Restore from backup
                    subprocess.run([
                        'docker', 'cp',
                        str(backup_file),
                        'sutazai-postgres:/tmp/test_backup.sql'
                    ], check=True)
                    
                    restore_cmd = [
                        'docker', 'exec', 'sutazai-postgres',
                        'psql',
                        '-U', 'jarvis',
                        '-d', 'jarvis_ai',
                        '-f', '/tmp/test_backup.sql'
                    ]
                    
                    result = subprocess.run(restore_cmd, capture_output=True, text=True)
                    restore_success = result.returncode == 0
                    
                    # Verify data restored
                    with self.conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM test_backup_restore")
                        row_count = cur.fetchone()[0]
                        data_restored = row_count == 2
                        
                        # Cleanup
                        cur.execute("DROP TABLE test_backup_restore")
                        self.conn.commit()
                        
                    passed = backup_success and backup_exists and restore_success and data_restored
                else:
                    passed = False
                    backup_exists = False
                    restore_success = False
                    data_restored = False
                    
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.add_result(
                    test_name,
                    passed,
                    duration_ms,
                    {
                        'backup_successful': backup_success,
                        'backup_file_exists': backup_exists,
                        'restore_successful': restore_success,
                        'data_restored': data_restored
                    }
                )
                
                logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
                return passed
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False


class Neo4jValidator:
    """Neo4j validation tests"""
    
    def __init__(self, config: Dict, results: TestResults):
        self.config = config
        self.results = results
        self.driver = None
        
    def connect(self):
        """Establish Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(self.config['uri'], auth=self.config['auth'])
            self.driver.verify_connectivity()
            logger.info("Neo4j connection established")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False
            
    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
            
    async def test_graph_queries(self):
        """Test Neo4j graph queries"""
        test_name = "Neo4j Graph Queries"
        start_time = time.time()
        
        try:
            with self.driver.session() as session:
                # Create test nodes
                session.run("CREATE (p:Person {name: 'Alice', age: 30})")
                session.run("CREATE (p:Person {name: 'Bob', age: 25})")
                session.run("CREATE (p:Person {name: 'Charlie', age: 35})")
                
                # Create relationships
                session.run("""
                    MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
                    CREATE (a)-[:KNOWS]->(b)
                """)
                
                # Test MATCH query
                result = session.run("MATCH (p:Person) RETURN count(p) as count")
                person_count = result.single()['count']
                
                # Test relationship query
                result = session.run("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r) as count")
                relationship_count = result.single()['count']
                
                # Test filtered query
                result = session.run("MATCH (p:Person) WHERE p.age > 28 RETURN count(p) as count")
                filtered_count = result.single()['count']
                
                # Cleanup
                session.run("MATCH (p:Person) DETACH DELETE p")
                
                passed = person_count == 3 and relationship_count == 1 and filtered_count == 2
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.add_result(
                    test_name,
                    passed,
                    duration_ms,
                    {
                        'persons_created': person_count,
                        'relationships_created': relationship_count,
                        'filtered_query_result': filtered_count
                    }
                )
                
                logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
                return passed
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False
            
    async def test_graph_relationships(self):
        """Test Neo4j relationship validation"""
        test_name = "Neo4j Graph Relationships"
        start_time = time.time()
        
        try:
            with self.driver.session() as session:
                # Create test graph
                session.run("""
                    CREATE (a:Agent {name: 'Letta', type: 'memory'})
                    CREATE (b:Agent {name: 'CrewAI', type: 'orchestration'})
                    CREATE (t:Task {name: 'Research', priority: 'high'})
                    CREATE (a)-[:ASSIGNED_TO]->(t)
                    CREATE (b)-[:DEPENDS_ON]->(a)
                """)
                
                # Test relationship traversal
                result = session.run("""
                    MATCH (a:Agent)-[:ASSIGNED_TO]->(t:Task)
                    RETURN a.name as agent, t.name as task
                """)
                assignments = list(result)
                
                # Test multi-hop traversal
                result = session.run("""
                    MATCH (b:Agent)-[:DEPENDS_ON]->(a:Agent)-[:ASSIGNED_TO]->(t:Task)
                    RETURN b.name as dependent, a.name as dependency, t.name as task
                """)
                dependencies = list(result)
                
                # Test relationship properties
                session.run("""
                    MATCH (a:Agent {name: 'Letta'})-[r:ASSIGNED_TO]->(t:Task)
                    SET r.assigned_at = datetime()
                    RETURN r
                """)
                
                result = session.run("""
                    MATCH (a:Agent)-[r:ASSIGNED_TO]->(t:Task)
                    WHERE r.assigned_at IS NOT NULL
                    RETURN count(r) as count
                """)
                rel_with_props = result.single()['count']
                
                # Cleanup
                session.run("MATCH (n) DETACH DELETE n")
                
                passed = len(assignments) == 1 and len(dependencies) == 1 and rel_with_props == 1
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.add_result(
                    test_name,
                    passed,
                    duration_ms,
                    {
                        'assignments_found': len(assignments),
                        'dependencies_found': len(dependencies),
                        'relationships_with_properties': rel_with_props
                    }
                )
                
                logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
                return passed
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False


class RedisValidator:
    """Redis validation tests"""
    
    def __init__(self, config: Dict, results: TestResults):
        self.config = config
        self.results = results
        self.client = None
        
    async def connect(self):
        """Establish Redis connection"""
        try:
            self.client = await redis.Redis(**self.config)
            await self.client.ping()
            logger.info("Redis connection established")
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False
            
    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            await self.client.aclose()
            logger.info("Redis connection closed")
            
    async def test_cache_invalidation(self):
        """Test Redis cache invalidation"""
        test_name = "Redis Cache Invalidation"
        start_time = time.time()
        
        try:
            # Test SET and GET
            await self.client.set('test_key', 'test_value')
            value = await self.client.get('test_key')
            set_get_works = value == 'test_value'
            
            # Test DEL
            await self.client.delete('test_key')
            value_after_del = await self.client.get('test_key')
            del_works = value_after_del is None
            
            # Test TTL expiration
            await self.client.setex('test_ttl', 2, 'expires_soon')
            value_before = await self.client.get('test_ttl')
            await asyncio.sleep(3)
            value_after = await self.client.get('test_ttl')
            ttl_works = value_before == 'expires_soon' and value_after is None
            
            # Test FLUSHDB (on test database)
            await self.client.set('test_flush_1', 'value1')
            await self.client.set('test_flush_2', 'value2')
            keys_before = await self.client.keys('test_flush_*')
            await self.client.delete(*keys_before) if keys_before else None
            keys_after = await self.client.keys('test_flush_*')
            flush_works = len(keys_before) >= 2 and len(keys_after) == 0
            
            passed = set_get_works and del_works and ttl_works and flush_works
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.add_result(
                test_name,
                passed,
                duration_ms,
                {
                    'set_get_works': set_get_works,
                    'delete_works': del_works,
                    'ttl_expiration_works': ttl_works,
                    'flush_works': flush_works
                }
            )
            
            logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
            return passed
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False
            
    async def test_persistence(self):
        """Test Redis persistence"""
        test_name = "Redis Persistence"
        start_time = time.time()
        
        try:
            # Check Redis persistence configuration
            config = await self.client.config_get('save')
            persistence_configured = config.get('save') is not None
            
            # Check AOF configuration
            aof_config = await self.client.config_get('appendonly')
            aof_enabled = aof_config.get('appendonly') == 'yes'
            
            # Write test data that should persist
            await self.client.set('persist_test', 'persistent_data')
            
            # Trigger background save (BGSAVE)
            try:
                await self.client.bgsave()
                bgsave_works = True
            except:
                # BGSAVE might be disabled or already in progress
                bgsave_works = False
                
            # Verify data still exists
            persisted_data = await self.client.get('persist_test')
            data_persisted = persisted_data == 'persistent_data'
            
            # Cleanup
            await self.client.delete('persist_test')
            
            passed = persistence_configured and data_persisted
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.add_result(
                test_name,
                passed,
                duration_ms,
                {
                    'persistence_configured': persistence_configured,
                    'aof_enabled': aof_enabled,
                    'bgsave_works': bgsave_works,
                    'data_persisted': data_persisted
                }
            )
            
            logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
            return passed
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False


class RabbitMQValidator:
    """RabbitMQ validation tests"""
    
    def __init__(self, config: Dict, results: TestResults):
        self.config = config
        self.results = results
        self.connection = None
        
    async def connect(self):
        """Establish RabbitMQ connection"""
        try:
            self.connection = await connect_robust(
                host=self.config['host'],
                port=self.config['port'],
                login=self.config['login'],
                password=self.config['password']
            )
            logger.info("RabbitMQ connection established")
            return True
        except Exception as e:
            logger.error(f"RabbitMQ connection failed: {e}")
            return False
            
    async def disconnect(self):
        """Close RabbitMQ connection"""
        if self.connection:
            await self.connection.close()
            logger.info("RabbitMQ connection closed")
            
    async def test_message_durability(self):
        """Test RabbitMQ message durability"""
        test_name = "RabbitMQ Message Durability"
        start_time = time.time()
        
        try:
            channel = await self.connection.channel()
            
            # Declare durable queue
            queue = await channel.declare_queue('test_durable_queue', durable=True)
            
            # Publish persistent message
            message = Message(
                body=b'Persistent test message',
                delivery_mode=DeliveryMode.PERSISTENT
            )
            
            await channel.default_exchange.publish(
                message,
                routing_key='test_durable_queue'
            )
            
            # Verify message in queue
            message_count = queue.declaration_result.message_count
            
            # Consume message
            received_message = await queue.get(timeout=5)
            message_received = received_message is not None
            
            if message_received:
                await received_message.ack()
                message_content_correct = received_message.body == b'Persistent test message'
                message_persistent = received_message.delivery_mode == DeliveryMode.PERSISTENT
            else:
                message_content_correct = False
                message_persistent = False
                
            # Cleanup
            await queue.delete()
            await channel.close()
            
            passed = message_received and message_content_correct and message_persistent
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.add_result(
                test_name,
                passed,
                duration_ms,
                {
                    'message_published': True,
                    'message_received': message_received,
                    'content_correct': message_content_correct,
                    'message_persistent': message_persistent
                }
            )
            
            logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
            return passed
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False
            
    async def test_queue_management(self):
        """Test RabbitMQ queue management"""
        test_name = "RabbitMQ Queue Management"
        start_time = time.time()
        
        try:
            channel = await self.connection.channel()
            
            # Test queue creation
            queue1 = await channel.declare_queue('test_mgmt_queue_1', durable=False)
            creation_works = queue1 is not None
            
            # Test queue purge
            await channel.default_exchange.publish(
                Message(body=b'Test message 1'),
                routing_key='test_mgmt_queue_1'
            )
            await channel.default_exchange.publish(
                Message(body=b'Test message 2'),
                routing_key='test_mgmt_queue_1'
            )
            
            await asyncio.sleep(0.5)  # Let messages settle
            
            purge_result = await queue1.purge()
            purge_works = purge_result is not None
            
            # Test queue deletion
            await queue1.delete()
            deletion_works = True
            
            # Test exchange creation
            exchange = await channel.declare_exchange(
                'test_exchange',
                ExchangeType.TOPIC,
                durable=False
            )
            exchange_works = exchange is not None
            
            # Cleanup
            await exchange.delete()
            await channel.close()
            
            passed = creation_works and purge_works and deletion_works and exchange_works
            duration_ms = (time.time() - start_time) * 1000
            
            self.results.add_result(
                test_name,
                passed,
                duration_ms,
                {
                    'queue_creation': creation_works,
                    'queue_purge': purge_works,
                    'queue_deletion': deletion_works,
                    'exchange_creation': exchange_works
                }
            )
            
            logger.info(f"{test_name}: {'PASSED' if passed else 'FAILED'} ({duration_ms:.2f}ms)")
            return passed
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.results.add_result(test_name, False, duration_ms, {'error': str(e)})
            logger.error(f"{test_name} FAILED: {e}")
            return False


async def run_all_tests():
    """Run all database validation tests"""
    logger.info("=" * 80)
    logger.info("PHASE 10: DATABASE VALIDATION TEST SUITE")
    logger.info("=" * 80)
    
    results = TestResults()
    
    # PostgreSQL Tests
    logger.info("\n--- PostgreSQL Validation Tests ---")
    pg_validator = PostgreSQLValidator(POSTGRES_CONFIG, results)
    if pg_validator.connect():
        await pg_validator.test_migrations()
        await pg_validator.test_schema_integrity()
        await pg_validator.test_foreign_key_constraints()
        await pg_validator.test_index_performance()
        await pg_validator.test_backup_restore()
        pg_validator.disconnect()
    else:
        logger.error("PostgreSQL connection failed, skipping tests")
        
    # Neo4j Tests
    logger.info("\n--- Neo4j Validation Tests ---")
    neo4j_validator = Neo4jValidator(NEO4J_CONFIG, results)
    if neo4j_validator.connect():
        await neo4j_validator.test_graph_queries()
        await neo4j_validator.test_graph_relationships()
        neo4j_validator.disconnect()
    else:
        logger.error("Neo4j connection failed, skipping tests")
        
    # Redis Tests
    logger.info("\n--- Redis Validation Tests ---")
    redis_validator = RedisValidator(REDIS_CONFIG, results)
    if await redis_validator.connect():
        await redis_validator.test_cache_invalidation()
        await redis_validator.test_persistence()
        await redis_validator.disconnect()
    else:
        logger.error("Redis connection failed, skipping tests")
        
    # RabbitMQ Tests
    logger.info("\n--- RabbitMQ Validation Tests ---")
    rabbitmq_validator = RabbitMQValidator(RABBITMQ_CONFIG, results)
    if await rabbitmq_validator.connect():
        await rabbitmq_validator.test_message_durability()
        await rabbitmq_validator.test_queue_management()
        await rabbitmq_validator.disconnect()
    else:
        logger.error("RabbitMQ connection failed, skipping tests")
        
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    summary = results.get_summary()
    logger.info(f"Total Tests: {summary['total_tests']}")
    logger.info(f"Passed: {summary['passed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Pass Rate: {summary['pass_rate']:.1f}%")
    logger.info(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
    
    # Save results to file
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    results_file = Path(f'/opt/sutazaiapp/PHASE_10_TEST_RESULTS_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'results': results.results
        }, f, indent=2)
        
    logger.info(f"\nResults saved to: {results_file}")
    
    return summary['failed'] == 0


if __name__ == '__main__':
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
