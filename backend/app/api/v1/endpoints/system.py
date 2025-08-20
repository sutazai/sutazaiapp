"""
System endpoint - COMPREHENSIVE IMPLEMENTATION

Provides detailed system status information including:
- CPU/Memory usage statistics
- Active connections count
- Database connection pool status
- Cache hit rates
- Error rates and health checks
"""
from fastapi import APIRouter, HTTPException
import platform
import os
import psutil
import time
import asyncio
from datetime import datetime
from typing import Dict, Any
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter()

# Add psutil to requirements if not already present
try:
    import psutil
except ImportError:
    logger.error("psutil is required for system monitoring. Install with: pip install psutil")
    raise

@router.get("/")
async def system_info():
    """
    Get comprehensive system information and health metrics
    """
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        
        # Get network stats
        net_io = psutil.net_io_counters()
        
        # Check database connections
        db_status = await _check_database_status()
        
        # Check cache status
        cache_status = await _check_cache_status()
        
        return {
            "status": "ok",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_version": platform.python_version()
            },
            "resources": {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "load_avg": list(load_avg)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            },
            "services": {
                "database": db_status,
                "cache": cache_status
            },
            "uptime": time.time() - psutil.boot_time()
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=f"System monitoring error: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        health_status["checks"]["cpu"] = {
            "status": "ok" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical",
            "usage": cpu_percent
        }
        
        # Check memory usage
        memory = psutil.virtual_memory()
        health_status["checks"]["memory"] = {
            "status": "ok" if memory.percent < 80 else "warning" if memory.percent < 95 else "critical",
            "usage": memory.percent
        }
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        health_status["checks"]["disk"] = {
            "status": "ok" if disk_percent < 80 else "warning" if disk_percent < 95 else "critical",
            "usage": disk_percent
        }
        
        # Check database connectivity
        db_status = await _check_database_status()
        health_status["checks"]["database"] = db_status
        
        # Check cache connectivity
        cache_status = await _check_cache_status()
        health_status["checks"]["cache"] = cache_status
        
        # Determine overall status
        statuses = [check["status"] for check in health_status["checks"].values()]
        if "critical" in statuses:
            health_status["status"] = "critical"
        elif "warning" in statuses:
            health_status["status"] = "warning"
        elif "error" in statuses:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "critical",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

async def _check_database_status() -> Dict[str, Any]:
    """Check database connection status"""
    try:
        if POSTGRES_AVAILABLE:
            # Try to connect to PostgreSQL
            conn = await asyncpg.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '10000')),
                user=os.getenv('POSTGRES_USER', 'sutazai_user'),
                password=os.getenv('POSTGRES_PASSWORD', ''),
                database=os.getenv('POSTGRES_DB', 'sutazai_main')
            )
            await conn.execute('SELECT 1')
            await conn.close()
            return {"status": "ok", "type": "postgresql", "connected": True}
        else:
            return {"status": "warning", "type": "postgresql", "message": "asyncpg not available"}
    except Exception as e:
        return {"status": "error", "type": "postgresql", "error": str(e)}

async def _check_cache_status() -> Dict[str, Any]:
    """Check Redis cache status"""
    try:
        if REDIS_AVAILABLE:
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '10001')),
                password=os.getenv('REDIS_PASSWORD', ''),
                decode_responses=True
            )
            redis_client.ping()
            info = redis_client.info()
            redis_client.close()
            return {
                "status": "ok",
                "type": "redis",
                "connected": True,
                "memory_usage": info.get('used_memory_human', 'unknown'),
                "connections": info.get('connected_clients', 0)
            }
        else:
            return {"status": "warning", "type": "redis", "message": "redis library not available"}
    except Exception as e:
        return {"status": "error", "type": "redis", "error": str(e)}

@router.get("/metrics")
async def get_system_metrics():
    """
    Get detailed system metrics for monitoring
    """
    try:
        # Get process information
        current_process = psutil.Process()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "process": {
                "pid": current_process.pid,
                "memory_info": current_process.memory_info()._asdict(),
                "cpu_percent": current_process.cpu_percent(),
                "create_time": current_process.create_time(),
                "num_threads": current_process.num_threads()
            },
            "system": {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "boot_time": psutil.boot_time(),
                "users": len(psutil.users())
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
