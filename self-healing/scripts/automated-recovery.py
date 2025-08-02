#!/usr/bin/env python3
"""
Automated Recovery System for SutazAI
Implements self-diagnostic health checks and automatic remediation
"""

import os
import sys
import time
import json
import logging
import asyncio
import psutil
import docker
import redis
import yaml
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import requests
import psycopg2
from psycopg2.pool import SimpleConnectionPool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthCheck:
    """
    Represents a health check for a service
    """
    
    def __init__(self, 
                 name: str,
                 check_func: Callable,
                 threshold: int = 3,
                 interval: int = 10):
        self.name = name
        self.check_func = check_func
        self.threshold = threshold
        self.interval = interval
        self.failure_count = 0
        self.last_check = None
        self.last_status = None
        
    async def execute(self) -> bool:
        """Execute the health check"""
        try:
            if asyncio.iscoroutinefunction(self.check_func):
                result = await self.check_func()
            else:
                result = self.check_func()
            
            self.last_check = datetime.now()
            self.last_status = result
            
            if result:
                self.failure_count = 0
            else:
                self.failure_count += 1
                
            return result
        except Exception as e:
            logger.error(f"Health check {self.name} failed with exception: {e}")
            self.failure_count += 1
            self.last_check = datetime.now()
            self.last_status = False
            return False
    
    def is_healthy(self) -> bool:
        """Check if service is healthy based on threshold"""
        return self.failure_count < self.threshold


class RecoveryAction:
    """
    Represents a recovery action that can be taken
    """
    
    def __init__(self,
                 name: str,
                 action_func: Callable,
                 triggers: List[str],
                 max_attempts: int = 3):
        self.name = name
        self.action_func = action_func
        self.triggers = triggers
        self.max_attempts = max_attempts
        self.attempt_count = 0
        self.last_attempt = None
        
    async def execute(self) -> bool:
        """Execute the recovery action"""
        if self.attempt_count >= self.max_attempts:
            logger.error(f"Recovery action {self.name} exceeded max attempts")
            return False
            
        try:
            logger.info(f"Executing recovery action: {self.name}")
            self.attempt_count += 1
            self.last_attempt = datetime.now()
            
            if asyncio.iscoroutinefunction(self.action_func):
                result = await self.action_func()
            else:
                result = self.action_func()
                
            if result:
                logger.info(f"Recovery action {self.name} succeeded")
                self.attempt_count = 0  # Reset on success
            else:
                logger.warning(f"Recovery action {self.name} failed")
                
            return result
        except Exception as e:
            logger.error(f"Recovery action {self.name} failed with exception: {e}")
            return False


class AutomatedRecoveryManager:
    """
    Manages automated recovery procedures for the system
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/self-healing/config/self-healing-config.yaml"):
        self.config_path = config_path
        self.health_checks: Dict[str, HealthCheck] = {}
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.docker_client = None
        self.redis_client = None
        self.db_pool = None
        self._load_config()
        self._init_clients()
        self._init_health_checks()
        self._init_recovery_actions()
        self.running = False
        
    def _load_config(self):
        """Load configuration from YAML"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.recovery_config = self.config.get('recovery', {})
            
    def _init_clients(self):
        """Initialize Docker, Redis, and database clients"""
        # Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Connected to Docker")
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            
        # Redis client
        try:
            self.redis_client = redis.Redis(
                host='redis',
                port=6379,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            
        # Database connection pool
        try:
            self.db_pool = SimpleConnectionPool(
                1, 20,
                host='postgres',
                port=5432,
                database=os.getenv('POSTGRES_DB', 'sutazai'),
                user=os.getenv('POSTGRES_USER', 'sutazai'),
                password=os.getenv('POSTGRES_PASSWORD', '')
            )
            logger.info("Created database connection pool")
        except Exception as e:
            logger.warning(f"Failed to create database pool: {e}")
    
    def _init_health_checks(self):
        """Initialize health checks"""
        # Docker container health checks
        self.health_checks['docker_containers'] = HealthCheck(
            name='docker_containers',
            check_func=self._check_docker_containers,
            threshold=3,
            interval=10
        )
        
        # Database health check
        self.health_checks['database'] = HealthCheck(
            name='database',
            check_func=self._check_database,
            threshold=2,
            interval=10
        )
        
        # Redis health check
        self.health_checks['redis'] = HealthCheck(
            name='redis',
            check_func=self._check_redis,
            threshold=3,
            interval=10
        )
        
        # Memory usage check
        self.health_checks['memory'] = HealthCheck(
            name='memory',
            check_func=self._check_memory,
            threshold=5,
            interval=30
        )
        
        # Disk space check
        self.health_checks['disk_space'] = HealthCheck(
            name='disk_space',
            check_func=self._check_disk_space,
            threshold=3,
            interval=60
        )
        
        # Service endpoint checks
        self.health_checks['backend'] = HealthCheck(
            name='backend',
            check_func=lambda: self._check_http_endpoint('http://backend:8000/health'),
            threshold=3,
            interval=15
        )
        
        self.health_checks['frontend'] = HealthCheck(
            name='frontend',
            check_func=lambda: self._check_http_endpoint('http://frontend:8501/healthz'),
            threshold=3,
            interval=15
        )
        
        self.health_checks['ollama'] = HealthCheck(
            name='ollama',
            check_func=lambda: self._check_http_endpoint('http://ollama:11434/api/tags'),
            threshold=5,
            interval=30
        )
    
    def _init_recovery_actions(self):
        """Initialize recovery actions"""
        procedures = self.recovery_config.get('procedures', {})
        
        # Service restart recovery
        if 'service_restart' in procedures:
            self.recovery_actions['service_restart'] = RecoveryAction(
                name='service_restart',
                action_func=self._restart_unhealthy_services,
                triggers=procedures['service_restart']['triggers'],
                max_attempts=3
            )
        
        # Database recovery
        if 'database_recovery' in procedures:
            self.recovery_actions['database_recovery'] = RecoveryAction(
                name='database_recovery',
                action_func=self._recover_database,
                triggers=procedures['database_recovery']['triggers'],
                max_attempts=2
            )
        
        # Memory recovery
        if 'memory_leak_recovery' in procedures:
            self.recovery_actions['memory_recovery'] = RecoveryAction(
                name='memory_recovery',
                action_func=self._recover_memory,
                triggers=procedures['memory_leak_recovery']['triggers'],
                max_attempts=3
            )
    
    def _check_docker_containers(self) -> bool:
        """Check if critical Docker containers are running"""
        if not self.docker_client:
            return False
            
        try:
            critical_services = self.config.get('dependencies', {}).get('critical_services', [])
            containers = self.docker_client.containers.list()
            
            for service in critical_services:
                container_name = f"sutazai-{service}"
                found = False
                for container in containers:
                    if container.name == container_name and container.status == 'running':
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Critical container {container_name} not running")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Docker container check failed: {e}")
            return False
    
    def _check_database(self) -> bool:
        """Check database connectivity"""
        if not self.db_pool:
            return False
            
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return False
        finally:
            if conn:
                self.db_pool.putconn(conn)
    
    def _check_redis(self) -> bool:
        """Check Redis connectivity"""
        if not self.redis_client:
            return False
            
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis check failed: {e}")
            return False
    
    def _check_memory(self) -> bool:
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            threshold = self.recovery_config.get('predictive_monitoring', {}).get('metrics', {}).get('resource_usage', {}).get('memory_threshold', 85)
            
            if memory.percent > threshold:
                logger.warning(f"Memory usage high: {memory.percent}%")
                return False
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """Check disk space"""
        try:
            disk = psutil.disk_usage('/')
            threshold = self.recovery_config.get('predictive_monitoring', {}).get('metrics', {}).get('resource_usage', {}).get('disk_threshold', 90)
            
            if disk.percent > threshold:
                logger.warning(f"Disk usage high: {disk.percent}%")
                return False
            return True
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return False
    
    def _check_http_endpoint(self, url: str, timeout: int = 5) -> bool:
        """Check HTTP endpoint health"""
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"HTTP endpoint check failed for {url}: {e}")
            return False
    
    async def _restart_unhealthy_services(self) -> bool:
        """Restart unhealthy Docker containers"""
        if not self.docker_client:
            return False
            
        try:
            restarted = []
            containers = self.docker_client.containers.list(all=True)
            
            for container in containers:
                if container.name.startswith('sutazai-'):
                    # Check container health
                    if container.status != 'running' or (hasattr(container, 'health') and container.health == 'unhealthy'):
                        logger.info(f"Restarting unhealthy container: {container.name}")
                        container.restart()
                        restarted.append(container.name)
                        
            if restarted:
                # Wait for containers to start
                await asyncio.sleep(10)
                logger.info(f"Restarted containers: {', '.join(restarted)}")
                
            return True
        except Exception as e:
            logger.error(f"Service restart failed: {e}")
            return False
    
    async def _recover_database(self) -> bool:
        """Recover database connections"""
        try:
            # Kill idle connections
            if self.db_pool:
                conn = None
                try:
                    conn = self.db_pool.getconn()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT pg_terminate_backend(pid) 
                        FROM pg_stat_activity 
                        WHERE state = 'idle' 
                        AND state_change < current_timestamp - INTERVAL '5 minutes'
                    """)
                    conn.commit()
                    cursor.close()
                    logger.info("Killed idle database connections")
                except Exception as e:
                    logger.error(f"Failed to kill idle connections: {e}")
                finally:
                    if conn:
                        self.db_pool.putconn(conn)
            
            # Reset connection pool
            if self.db_pool:
                self.db_pool.closeall()
                self._init_clients()  # Reinitialize
                logger.info("Reset database connection pool")
                
            return True
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            return False
    
    async def _recover_memory(self) -> bool:
        """Recover from high memory usage"""
        try:
            # Clear Redis caches
            if self.redis_client:
                self.redis_client.flushdb()
                logger.info("Cleared Redis cache")
            
            # Force garbage collection in Python processes
            import gc
            gc.collect()
            logger.info("Forced garbage collection")
            
            # Restart high-memory containers
            if self.docker_client:
                containers = self.docker_client.containers.list()
                for container in containers:
                    if container.name.startswith('sutazai-'):
                        stats = container.stats(stream=False)
                        memory_usage = stats['memory_stats']['usage']
                        memory_limit = stats['memory_stats']['limit']
                        
                        if memory_limit > 0 and (memory_usage / memory_limit) > 0.9:
                            logger.info(f"Restarting high-memory container: {container.name}")
                            container.restart()
                            
            return True
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False
    
    async def monitor_and_recover(self):
        """Main monitoring and recovery loop"""
        self.running = True
        logger.info("Started automated recovery monitoring")
        
        while self.running:
            try:
                # Execute all health checks
                unhealthy_triggers = []
                
                for name, check in self.health_checks.items():
                    healthy = await check.execute()
                    
                    if not check.is_healthy():
                        # Map health check failures to triggers
                        if name == 'docker_containers':
                            unhealthy_triggers.append('health_check_failed')
                        elif name == 'database':
                            unhealthy_triggers.append('connection_pool_exhausted')
                        elif name == 'memory':
                            unhealthy_triggers.append('memory_threshold_exceeded')
                        
                        logger.warning(f"Health check {name} is unhealthy (failures: {check.failure_count})")
                
                # Execute recovery actions based on triggers
                for trigger in unhealthy_triggers:
                    for action_name, action in self.recovery_actions.items():
                        if trigger in action.triggers:
                            await action.execute()
                
                # Wait before next check
                await asyncio.sleep(self.recovery_config.get('health_check_interval', 10))
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(10)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all checks"""
        return {
            name: {
                'healthy': check.is_healthy(),
                'failure_count': check.failure_count,
                'last_check': check.last_check.isoformat() if check.last_check else None,
                'last_status': check.last_status
            }
            for name, check in self.health_checks.items()
        }
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get status of recovery actions"""
        return {
            name: {
                'attempt_count': action.attempt_count,
                'max_attempts': action.max_attempts,
                'last_attempt': action.last_attempt.isoformat() if action.last_attempt else None,
                'triggers': action.triggers
            }
            for name, action in self.recovery_actions.items()
        }


async def main():
    """Main entry point"""
    manager = AutomatedRecoveryManager()
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(manager.monitor_and_recover())
    
    try:
        # Keep running
        while True:
            # Print status every minute
            await asyncio.sleep(60)
            
            print("\n=== Health Status ===")
            for name, status in manager.get_health_status().items():
                print(f"{name}: {'HEALTHY' if status['healthy'] else 'UNHEALTHY'} (failures: {status['failure_count']})")
            
            print("\n=== Recovery Actions ===")
            for name, status in manager.get_recovery_status().items():
                print(f"{name}: attempts {status['attempt_count']}/{status['max_attempts']}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down automated recovery")
        manager.running = False
        await monitor_task


if __name__ == "__main__":
    asyncio.run(main())