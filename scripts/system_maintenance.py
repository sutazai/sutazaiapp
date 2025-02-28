#!/usr/bin/env python3
"""
System Maintenance Script
Handles routine maintenance tasks, cleanup, and optimization.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import aiohttp
import psutil
import redis
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.getenv("LOG_FILE_PATH", "/opt/sutazaiapp/logs/maintenance.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class SystemMaintenance:
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

    async def check_disk_usage(self) -> Dict:
        """Check disk usage and clean up if necessary."""
        disk = psutil.disk_usage("/")
        logger.info(f"Disk usage: {disk.percent}%")

        if disk.percent > 85:
            await self.cleanup_old_files()

        return {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
        }

    async def cleanup_old_files(self, days: int = 30):
        """Clean up old log and temporary files."""
        cleanup_paths = [
            Path("/opt/sutazaiapp/logs"),
            Path("/opt/sutazaiapp/storage/temp"),
            Path("/tmp"),
        ]

        cutoff_time = datetime.now() - timedelta(days=days)

        for path in cleanup_paths:
            if path.exists():
                for file in path.glob("**/*"):
                    if file.is_file():
                        try:
                            stat = file.stat()
                            if datetime.fromtimestamp(stat.st_mtime) < cutoff_time:
                                file.unlink()
                                logger.info(f"Deleted old file: {file}")
                        except Exception as e:
                            logger.exception(f"Failed to process {file}: {str(e)}")

    async def optimize_database(self):
        """Perform database maintenance and optimization."""
        try:
            with self.db_engine.connect() as conn:
                # Analyze tables
                conn.execute(text("ANALYZE VERBOSE"))

                # Vacuum analyze to reclaim space and update statistics
                conn.execute(text("VACUUM ANALYZE"))

                # Reindex to optimize indexes
                conn.execute(text("REINDEX DATABASE current_database()"))

            logger.info("Database optimization completed successfully")
        except Exception as e:
            logger.exception(f"Database optimization failed: {str(e)}")

    async def clear_redis_cache(self):
        """Clear Redis cache if memory usage is high."""
        try:
            info = self.redis_client.info()
            used_memory = info["used_memory"] / 1024 / 1024  # Convert to MB

            if used_memory > 512:  # If using more than 512MB
                self.redis_client.flushdb()
                logger.info("Cleared Redis cache due to high memory usage")
            else:
                logger.info(f"Redis memory usage is normal: {used_memory:.2f}MB")
        except Exception as e:
            logger.exception(f"Redis cache cleanup failed: {str(e)}")

    async def check_service_health(self) -> List[Dict]:
        """Check health of all system services."""
        services = [
            {"name": "web_server", "url": "http://localhost:8000/health"},
            {"name": "redis", "port": 6379},
            {"name": "postgres", "port": 5432},
            {"name": "monitoring", "port": 9090},
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
                                    "status": "healthy" if response.status == 200 else "unhealthy",
                                    "response_time": response.elapsed.total_seconds(),
                                }
                            )
                    elif "port" in service:
                        import socket

                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        result = sock.connect_ex(("localhost", service["port"]))
                        sock.close()
                        results.append(
                            {
                                "name": service["name"],
                                "status": "healthy" if result == 0 else "unhealthy",
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

    async def run_maintenance(self):
        """Run all maintenance tasks."""
        try:
            # Check disk usage
            disk_status = await self.check_disk_usage()
            logger.info(f"Disk status: {disk_status}")

            # Check service health
            service_status = await self.check_service_health()
            logger.info(f"Service status: {service_status}")

            # Optimize database weekly
            if datetime.now().weekday() == 6:  # Sunday
                await self.optimize_database()

            # Clear Redis cache if needed
            await self.clear_redis_cache()

        except Exception as e:
            logger.exception(f"Maintenance cycle failed: {str(e)}")


async def main():
    maintenance = SystemMaintenance()

    while True:
        await maintenance.run_maintenance()
        # Wait for the next maintenance cycle (default: 1 hour)
        await asyncio.sleep(int(os.getenv("MAINTENANCE_INTERVAL", 3600)))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Maintenance script stopped by user")
        sys.exit(0)
                                                    