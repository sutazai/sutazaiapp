#!/usr/bin/env python3
"""
Master System Optimizer for SutazAI
Handles system optimization, monitoring, and maintenance tasks.
"""

import os
import sys
import psutil
import logging
import asyncio
import aiohttp
import redis
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("LOG_FILE_PATH", "system.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class SystemOptimizer:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
        )
        self.db_engine = create_engine(
            f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
            f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )

    async def check_system_health(self) -> Dict:
        """Check overall system health metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "disk_usage": disk.percent,
            "timestamp": datetime.now().isoformat(),
        }

    async def optimize_performance(self):
        """Optimize system performance."""
        try:
            # Clear Redis cache if memory usage is high
            if psutil.virtual_memory().percent > 90:
                self.redis_client.flushdb()
                logger.info("Cleared Redis cache due to high memory usage")

            # Optimize database
            with self.db_engine.connect() as conn:
                conn.execute("VACUUM ANALYZE")
                logger.info("Database optimization completed")

        except Exception as e:
            logger.error(f"Performance optimization failed: {str(e)}")

    async def monitor_services(self) -> List[Dict]:
        """Monitor status of all system services."""
        services = [
            {"name": "web_server", "url": "http://localhost:8000/health"},
            {"name": "celery", "url": "http://localhost:5555/api/workers"},
            {"name": "redis", "port": 6379},
        ]

        results = []
        async with aiohttp.ClientSession() as session:
            for service in services:
                try:
                    if "url" in service:
                        async with session.get(service["url"]) as response:
                            results.append(
                                {
                                    "name": service["name"],
                                    "status": (
                                        "healthy"
                                        if response.status == 200
                                        else "unhealthy"
                                    ),
                                    "response_time": response.elapsed.total_seconds(),
                                }
                            )
                    elif "port" in service:
                        # Check if port is open
                        import socket

                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        result = sock.connect_ex(("localhost", service["port"]))
                        sock.close()
                        results.append(
                            {
                                "name": service["name"],
                                "status": ("healthy" if result == 0 else "unhealthy"),
                            }
                        )
                except Exception as e:
                    results.append(
                        {
                            "name": service["name"],
                            "status": "error",
                            "error": str(e),
                        }
                    )
        return results

    async def cleanup_old_files(self, days_old: int = 7):
        """Clean up old log and temporary files."""
        cleanup_paths = [
            Path("/opt/sutazai/logs"),
            Path("/opt/sutazai/storage/temp"),
        ]

        for path in cleanup_paths:
            if path.exists():
                for file in path.glob("**/*"):
                    if file.is_file():
                        file_age = datetime.now().timestamp() - file.stat().st_mtime
                        if file_age > (days_old * 86400):  # Convert days to seconds
                            try:
                                file.unlink()
                                logger.info(f"Deleted old file: {file}")
                            except Exception as e:
                                logger.error(f"Failed to delete {file}: {str(e)}")

    async def run_optimization_cycle(self):
        """Run a complete optimization cycle."""
        try:
            # Check system health
            health_metrics = await self.check_system_health()
            logger.info(f"System health metrics: {health_metrics}")

            # Monitor services
            service_status = await self.monitor_services()
            logger.info(f"Service status: {service_status}")

            # Optimize performance if needed
            if health_metrics["cpu_usage"] > 80 or health_metrics["memory_usage"] > 80:
                await self.optimize_performance()

            # Cleanup old files weekly
            if datetime.now().weekday() == 0:  # Run on Mondays
                await self.cleanup_old_files()

        except Exception as e:
            logger.error(f"Optimization cycle failed: {str(e)}")


async def main():
    optimizer = SystemOptimizer()
    while True:
        await optimizer.run_optimization_cycle()
        await asyncio.sleep(int(os.getenv("SYSTEM_CHECK_INTERVAL", 300)))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System optimizer stopped by user")
        sys.exit(0)
