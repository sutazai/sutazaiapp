#!/usr/bin/env python3.11
"""
Health Monitor for Supreme AI Orchestrator

This script monitors the health of the orchestrator system and reports issues.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp
import psutil
import toml
from prometheus_client import start_http_server, Gauge, Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/health_monitor.log')
    ]
)

logger = logging.getLogger(__name__)

# Prometheus metrics
ORCHESTRATOR_UP = Gauge('orchestrator_up', 'Whether the orchestrator is up')
AGENT_COUNT = Gauge('agent_count', 'Number of registered agents')
TASK_QUEUE_SIZE = Gauge('task_queue_size', 'Number of tasks in queue')
SYNC_STATUS = Gauge('sync_status', 'Synchronization status')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
ERROR_COUNT = Counter('error_count', 'Number of errors encountered')

class HealthMonitor:
    """Monitor the health of the Supreme AI Orchestrator"""

    def __init__(self, config_path: str):
        """Initialize the health monitor"""
        self.config = self._load_config(config_path)
        self.primary_url = f"http://{self.config['primary_server']['host']}:{self.config['primary_server']['port']}"
        self.secondary_url = f"http://{self.config['secondary_server']['host']}:{self.config['secondary_server']['port']}"
        self.check_interval = self.config['monitoring']['health_check_interval']
        self.alert_threshold = self.config['monitoring'].get('alert_threshold', 3)
        self.error_count = 0
        self.last_check = None
        self._session = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from TOML file"""
        try:
            with open(config_path, 'r') as f:
                return toml.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            sys.exit(1)

    async def start_monitoring(self):
        """Start the health monitoring process"""
        logger.info("Starting health monitoring")
        self._session = aiohttp.ClientSession()

        try:
            while True:
                await self.check_health()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("Health monitoring stopped")
        finally:
            await self._session.close()

    async def check_health(self):
        """Check the health of the orchestrator system"""
        try:
            # Check primary server
            primary_health = await self._check_server_health(self.primary_url)
            if primary_health:
                self._update_metrics(primary_health, is_primary=True)

            # Check secondary server
            secondary_health = await self._check_server_health(self.secondary_url)
            if secondary_health:
                self._update_metrics(secondary_health, is_primary=False)

            # Check system resources
            self._check_system_resources()

            # Reset error count on successful check
            if primary_health and secondary_health:
                self.error_count = 0
                ORCHESTRATOR_UP.set(1)
            else:
                self._handle_health_check_failure()

            self.last_check = datetime.now()

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            ERROR_COUNT.inc()
            self._handle_health_check_failure()

    async def _check_server_health(self, server_url: str) -> Optional[Dict]:
        """Check the health of a server"""
        try:
            async with self._session.get(f"{server_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Health check failed for {server_url}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Failed to connect to {server_url}: {e}")
            return None

    def _update_metrics(self, health_data: Dict, is_primary: bool):
        """Update Prometheus metrics"""
        server_type = "primary" if is_primary else "secondary"

        # Update agent count
        AGENT_COUNT.labels(server=server_type).set(health_data.get('agent_count', 0))

        # Update task queue size
        TASK_QUEUE_SIZE.labels(server=server_type).set(health_data.get('queue_size', 0))

        # Update sync status
        sync_status = 1 if health_data.get('sync_status') == 'SUCCESS' else 0
        SYNC_STATUS.labels(server=server_type).set(sync_status)

    def _check_system_resources(self):
        """Check system resource usage"""
        try:
            # Get process
            process = psutil.Process(os.getpid())

            # Memory usage
            memory_info = process.memory_info()
            MEMORY_USAGE.set(memory_info.rss)

            # CPU usage
            cpu_percent = process.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)

            # Check thresholds
            if memory_info.rss > self.config['monitoring'].get('max_memory_bytes', 1024 * 1024 * 1024):
                logger.warning("Memory usage exceeds threshold")

            if cpu_percent > self.config['monitoring'].get('max_cpu_percent', 80):
                logger.warning("CPU usage exceeds threshold")

        except Exception as e:
            logger.error(f"Failed to check system resources: {e}")

    def _handle_health_check_failure(self):
        """Handle health check failures"""
        self.error_count += 1
        ORCHESTRATOR_UP.set(0)

        if self.error_count >= self.alert_threshold:
            self._send_alert(
                f"Orchestrator health check failed {self.error_count} times"
            )

    def _send_alert(self, message: str):
        """Send an alert about health issues"""
        logger.error(f"ALERT: {message}")
        # TODO: Implement alert sending (email, Slack, etc.)

async def main():
    """Main entry point"""
    try:
        # Start Prometheus metrics server
        start_http_server(9090)

        # Create and start monitor
        monitor = HealthMonitor('config/orchestrator.toml')
        await monitor.start_monitoring()

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
