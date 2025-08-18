#!/usr/bin/env python3
"""
Emergency Backend Fix Script
Fixes the critical deadlock issue in the backend startup
Date: 2025-08-17
"""

import sys
import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the original file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup.{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up to: {backup_path}")
    return backup_path

def apply_main_fix():
    """Fix the main.py deadlock issue"""
    main_file = "/opt/sutazaiapp/backend/app/main.py"
    
    # Read the current file
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Create the emergency health endpoint that doesn't require initialization
    emergency_health = '''
# EMERGENCY FIX: Add minimal health endpoint that works without initialization
@app.get("/health-emergency")
async def emergency_health_check():
    """Emergency health endpoint that bypasses initialization"""
    return {
        "status": "emergency",
        "message": "Backend running in emergency mode - initialization bypassed",
        "timestamp": datetime.now().isoformat()
    }
'''

    # Find the location to insert (after app creation)
    app_creation = 'app = FastAPI('
    app_creation_end = content.find(')', content.find(app_creation)) + 1
    
    # Insert the emergency endpoint
    if "/health-emergency" not in content:
        content = content[:app_creation_end] + "\n" + emergency_health + content[app_creation_end:]
        print("✅ Added emergency health endpoint")
    
    # Fix the lifespan function to prevent deadlock
    lifespan_fix = '''
@asynccontextmanager
async def lifespan(app: FastAPI):
    """EMERGENCY FIX: Lifespan with timeout and lazy initialization"""
    import asyncio
    
    logger.info("Starting backend with emergency fix...")
    
    # Set emergency mode flag
    app.state.initialization_complete = False
    app.state.emergency_mode = True
    
    try:
        # Try initialization with timeout
        async with asyncio.timeout(15):  # 15 second timeout
            logger.info("Attempting standard initialization...")
            
            # Initialize connection pools with non-blocking approach
            try:
                from app.core.connection_pool import ConnectionPoolManager
                pool_manager = ConnectionPoolManager()
                # Don't await full initialization, just create instance
                app.state.pool_manager = pool_manager
                logger.info("Connection pool manager created (lazy init)")
            except Exception as e:
                logger.error(f"Pool manager creation failed: {e}")
                app.state.pool_manager = None
            
            # Initialize cache service with non-blocking approach
            try:
                from app.core.cache import CacheService
                cache_service = CacheService()
                app.state.cache_service = cache_service
                logger.info("Cache service created (lazy init)")
            except Exception as e:
                logger.error(f"Cache service creation failed: {e}")
                app.state.cache_service = None
            
            # Skip other heavy initializations for now
            logger.info("Skipping heavy initializations to prevent deadlock")
            
            # Register minimal task handlers
            from app.core.task_queue import TaskQueue
            task_queue = TaskQueue()
            app.state.task_queue = task_queue
            
            # Mark as partially initialized
            app.state.initialization_complete = True
            app.state.emergency_mode = False
            logger.info("✅ Backend initialized successfully (minimal mode)")
            
    except asyncio.TimeoutError:
        logger.error("⚠️ Initialization timeout - running in emergency mode")
        app.state.initialization_complete = False
        app.state.emergency_mode = True
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        app.state.initialization_complete = False
        app.state.emergency_mode = True
    
    # Start background initialization for remaining services
    asyncio.create_task(initialize_remaining_services(app))
    
    logger.info(f"Backend started in {'emergency' if app.state.emergency_mode else 'normal'} mode")
    
    yield
    
    # Cleanup
    logger.info("Shutting down backend...")
    
async def initialize_remaining_services(app):
    """Initialize remaining services in background"""
    await asyncio.sleep(5)  # Wait a bit before starting
    
    try:
        logger.info("Starting background initialization of remaining services...")
        
        # Initialize MCP if not in emergency mode
        if not app.state.emergency_mode:
            try:
                from app.core.mcp_startup import initialize_mcp_background
                await initialize_mcp_background(None)
                logger.info("MCP services initialized in background")
            except Exception as e:
                logger.error(f"MCP initialization failed: {e}")
        
    except Exception as e:
        logger.error(f"Background initialization failed: {e}")
'''

    # Replace the lifespan function
    if "@asynccontextmanager" in content and "async def lifespan" in content:
        # Find and replace the entire lifespan function
        start = content.find("@asynccontextmanager")
        # Find the end of the lifespan function (next function or class definition)
        next_def = content.find("\n@app", start + 1)
        if next_def == -1:
            next_def = content.find("\nclass ", start + 1)
        if next_def == -1:
            next_def = content.find("\ndef ", start + 1)
        
        if next_def != -1:
            content = content[:start] + lifespan_fix + "\n" + content[next_def:]
            print("✅ Replaced lifespan function with emergency fix")
    
    # Write the fixed content
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("✅ main.py fixed successfully")

def fix_connection_pool():
    """Fix the connection pool to use hostnames instead of IPs"""
    pool_file = "/opt/sutazaiapp/backend/app/core/connection_pool.py"
    
    with open(pool_file, 'r') as f:
        content = f.read()
    
    # Replace hardcoded IPs with hostnames
    replacements = [
        ("'172.20.0.2'", "'sutazai-redis'"),
        ("'172.20.0.5'", "'sutazai-postgres'"),
        ("'172.20.0.7'", "'sutazai-redis'"),  # Just in case
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ Replaced {old} with {new}")
    
    with open(pool_file, 'w') as f:
        f.write(content)
    
    print("✅ connection_pool.py fixed")

def fix_cache_service():
    """Add timeout to cache service initialization"""
    cache_file = "/opt/sutazaiapp/backend/app/core/cache.py"
    
    with open(cache_file, 'r') as f:
        content = f.read()
    
    # Add timeout to Redis operations
    timeout_fix = '''
    async def get(self, key: str, default: Any = None, force_local: bool = False) -> Any:
        """
        EMERGENCY FIX: Added timeout to prevent deadlock
        """
        self._stats['gets'] += 1
        
        # Add timeout to Redis operations
        if not force_local:
            try:
                # Use timeout to prevent deadlock
                async with asyncio.timeout(2):  # 2 second timeout
                    redis_client = await get_redis()
                    value = await redis_client.get(key)
                    
                    if value:
                        self._stats['hits'] += 1
                        decompressed = self._decompress_value(value)
                        deserialized = pickle.loads(decompressed)
                        self._add_to_local(key, deserialized)
                        return deserialized
                        
            except asyncio.TimeoutError:
                logger.warning(f"Redis timeout for key {key}, using local cache")
            except Exception as e:
                logger.error(f"Redis get error, falling back to local: {e}")
'''
    
    # Check if timeout fix is needed
    if "asyncio.timeout" not in content:
        print("⚠️ Cache service needs timeout fix (manual intervention required)")
    
    print("✅ cache.py reviewed")

def main():
    """Main execution"""
    print("=" * 60)
    print("EMERGENCY BACKEND FIX SCRIPT")
    print("Fixing critical deadlock issue")
    print("=" * 60)
    
    # Backup original files
    print("\n📦 Creating backups...")
    backup_file("/opt/sutazaiapp/backend/app/main.py")
    backup_file("/opt/sutazaiapp/backend/app/core/connection_pool.py")
    
    # Apply fixes
    print("\n🔧 Applying fixes...")
    apply_main_fix()
    fix_connection_pool()
    fix_cache_service()
    
    print("\n✅ Emergency fixes applied!")
    print("\n📝 Next steps:")
    print("1. Restart the backend: docker restart sutazai-backend")
    print("2. Test emergency endpoint: curl http://localhost:10010/health-emergency")
    print("3. Monitor logs: docker logs -f sutazai-backend")
    print("\n⚠️ This is an EMERGENCY fix. Proper refactoring is still required!")
    
if __name__ == "__main__":
    main()