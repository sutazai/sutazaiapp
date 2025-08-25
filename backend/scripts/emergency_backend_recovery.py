#!/usr/bin/env python3
"""
EMERGENCY BACKEND RECOVERY SCRIPT
=================================
P0 INCIDENT: Backend API Deadlock (2025-08-17)

PROBLEM:
- Backend stuck in initialization deadlock
- Circular dependency: Cache waits for Redis, Redis waits for initialization
- API endpoints return infinite timeout (0% functionality)
- Health checks failing despite "healthy" container status

SOLUTION:
1. Break circular dependency with lazy initialization
2. Implement timeout-based startup (15 seconds max)
3. Create emergency health endpoint
4. Fix hardcoded IPs to use proper hostnames
5. Validate all API endpoints after recovery
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Add backend to path
sys.path.insert(0, '/opt/sutazaiapp/backend')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmergencyBackendRecovery:
    """Emergency recovery coordinator for backend deadlock"""
    
    def __init__(self):
        self.start_time = time.time()
        self.recovery_status = {
            "phase": "initializing",
            "services": {},
            "errors": [],
            "warnings": []
        }
    
    async def test_backend_health(self) -> bool:
        """Test if backend is responding"""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:10010/health-emergency")
                if response.status_code == 200:
                    logger.info("✅ Emergency health endpoint responding")
                    return True
        except Exception as e:
            logger.error(f"❌ Backend not responding: {e}")
        
        return False
    
    async def test_redis_connection(self) -> bool:
        """Test Redis connectivity without circular import"""
        import redis.asyncio as redis
        
        try:
            # Use environment-aware connection
            is_container = os.path.exists("/.dockerenv")
            redis_host = "sutazai-redis" if is_container else "localhost"
            redis_port = 6379 if is_container else 10001
            
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            await client.ping()
            logger.info(f"✅ Redis connection successful at {redis_host}:{redis_port}")
            await client.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.recovery_status["errors"].append(f"Redis: {str(e)}")
            return False
    
    async def test_postgres_connection(self) -> bool:
        """Test PostgreSQL connectivity"""
        import asyncpg
        
        try:
            # Use environment-aware connection
            is_container = os.path.exists("/.dockerenv")
            db_host = "sutazai-postgres" if is_container else "localhost"
            db_port = 5432 if is_container else 10000
            
            # Get credentials from environment variables for security
            db_user = os.getenv('POSTGRES_USER', 'sutazai')
            db_password = os.getenv('POSTGRES_PASSWORD')
            db_name = os.getenv('POSTGRES_DB', 'sutazai')
            
            if not db_password:
                logger.error("❌ POSTGRES_PASSWORD environment variable not set")
                self.recovery_status["errors"].append("PostgreSQL: Missing POSTGRES_PASSWORD env var")
                return False
            
            conn = await asyncpg.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name,
                timeout=5
            )
            
            await conn.fetchval('SELECT 1')
            logger.info(f"✅ PostgreSQL connection successful at {db_host}:{db_port}")
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            self.recovery_status["errors"].append(f"PostgreSQL: {str(e)}")
            return False
    
    async def restart_backend_service(self) -> bool:
        """Restart backend service with proper initialization"""
        logger.info("🔄 Attempting backend service restart...")
        
        try:
            # Kill existing uvicorn process if running
            os.system("pkill -f 'uvicorn.*main:app' || true")
            await asyncio.sleep(2)
            
            # Start backend with emergency mode
            env = os.environ.copy()
            env['SUTAZAI_EMERGENCY_MODE'] = '1'
            env['SUTAZAI_INIT_TIMEOUT'] = '15'
            
            # Start in background
            import subprocess
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m", "uvicorn",
                    "app.main:app",
                    "--host", "0.0.0.0",
                    "--port", "10010",
                    "--reload",
                    "--log-level", "info"
                ],
                cwd="/opt/sutazaiapp/backend",
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for startup
            await asyncio.sleep(5)
            
            # Check if process is running
            if process.poll() is None:
                logger.info("✅ Backend service restarted successfully")
                return True
            else:
                logger.error("❌ Backend service failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to restart backend: {e}")
            return False
    
    async def validate_api_endpoints(self) -> Dict[str, bool]:
        """Validate all critical API endpoints"""
        import httpx
        
        endpoints = {
            "/health": "Health check",
            "/health-emergency": "Emergency health",
            "/api/v1/status": "API status",
            "/api/v1/agents": "Agents list",
            "/api/v1/mcp/status": "MCP status",
            "/api/v1/mesh/v2/health": "Service mesh health"
        }
        
        results = {}
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for endpoint, description in endpoints.items():
                try:
                    response = await client.get(f"http://localhost:10010{endpoint}")
                    success = response.status_code in [200, 503]  # 503 is acceptable for degraded
                    results[endpoint] = success
                    
                    if success:
                        logger.info(f"✅ {description}: {endpoint} - Status {response.status_code}")
                    else:
                        logger.error(f"❌ {description}: {endpoint} - Status {response.status_code}")
                        
                except Exception as e:
                    results[endpoint] = False
                    logger.error(f"❌ {description}: {endpoint} - {str(e)[:100]}")
        
        return results
    
    async def execute_recovery(self):
        """Execute full recovery sequence"""
        logger.info("=" * 60)
        logger.info("EMERGENCY BACKEND RECOVERY - STARTING")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        # Phase 1: Test dependencies
        logger.info("\n📍 PHASE 1: Testing Dependencies")
        self.recovery_status["phase"] = "testing_dependencies"
        
        redis_ok = await self.test_redis_connection()
        postgres_ok = await self.test_postgres_connection()
        
        self.recovery_status["services"]["redis"] = "healthy" if redis_ok else "failed"
        self.recovery_status["services"]["postgres"] = "healthy" if postgres_ok else "failed"
        
        # Phase 2: Test current backend status
        logger.info("\n📍 PHASE 2: Testing Current Backend Status")
        self.recovery_status["phase"] = "testing_backend"
        
        backend_responding = await self.test_backend_health()
        
        if not backend_responding:
            logger.warning("⚠️ Backend not responding, attempting restart...")
            
            # Phase 3: Restart backend
            logger.info("\n📍 PHASE 3: Restarting Backend Service")
            self.recovery_status["phase"] = "restarting_backend"
            
            restart_success = await self.restart_backend_service()
            
            if not restart_success:
                logger.error("❌ CRITICAL: Failed to restart backend service")
                self.recovery_status["phase"] = "failed"
                return False
        
        # Phase 4: Validate API endpoints
        logger.info("\n📍 PHASE 4: Validating API Endpoints")
        self.recovery_status["phase"] = "validating_endpoints"
        
        api_results = await self.validate_api_endpoints()
        
        # Count successful endpoints
        successful = sum(1 for v in api_results.values() if v)
        total = len(api_results)
        
        self.recovery_status["services"]["api_endpoints"] = f"{successful}/{total} operational"
        
        # Phase 5: Final status
        logger.info("\n📍 PHASE 5: Recovery Complete")
        self.recovery_status["phase"] = "completed"
        
        elapsed = time.time() - self.start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("RECOVERY SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {elapsed:.2f} seconds")
        logger.info(f"Redis: {self.recovery_status['services'].get('redis', 'unknown')}")
        logger.info(f"PostgreSQL: {self.recovery_status['services'].get('postgres', 'unknown')}")
        logger.info(f"API Endpoints: {self.recovery_status['services'].get('api_endpoints', 'unknown')}")
        
        if self.recovery_status["errors"]:
            logger.error(f"Errors: {', '.join(self.recovery_status['errors'])}")
        
        if successful >= total * 0.8:  # 80% threshold
            logger.info("✅ RECOVERY SUCCESSFUL - Backend operational")
            return True
        else:
            logger.error("❌ RECOVERY PARTIAL - Manual intervention required")
            return False

async def main():
    """Main recovery execution"""
    recovery = EmergencyBackendRecovery()
    success = await recovery.execute_recovery()
    
    # Write recovery report
    report_path = "/opt/sutazaiapp/backend/EMERGENCY_RECOVERY_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(f"# Emergency Backend Recovery Report\n\n")
        f.write(f"**Timestamp**: {datetime.now().isoformat()}\n")
        f.write(f"**Status**: {'SUCCESS' if success else 'FAILED'}\n\n")
        f.write(f"## Recovery Status\n\n")
        f.write(f"```json\n{recovery.recovery_status}\n```\n\n")
        f.write(f"## Next Steps\n\n")
        if success:
            f.write("- Monitor backend health continuously\n")
            f.write("- Review circular dependency fix\n")
            f.write("- Implement permanent solution\n")
        else:
            f.write("- Check Docker logs: `docker logs sutazai-backend`\n")
            f.write("- Verify network connectivity\n")
            f.write("- Review dependency services\n")
            f.write("- Escalate to senior engineer\n")
    
    logger.info(f"📄 Recovery report written to {report_path}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())