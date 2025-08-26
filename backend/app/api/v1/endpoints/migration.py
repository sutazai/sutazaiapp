"""
Migration API Endpoints
Clean REST API for managing memory service migrations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from app.services.memory_migration import (
    create_migration_service,
    MigrationConfig,
    MigrationResult
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Store migration results for tracking
_migration_results: Dict[str, MigrationResult] = {}
_active_migrations: Dict[str, bool] = {}

@router.post("/extended-memory-to-unified")
async def migrate_extended_memory(
    background_tasks: BackgroundTasks,
    dry_run: bool = False,
    batch_size: int = 100
):
    """Migrate extended-memory data to unified-memory service"""
    migration_id = f"extended-memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if _active_migrations.get("extended-memory", False):
        raise HTTPException(
            status_code=409,
            detail="Extended memory migration already in progress"
        )
    
    try:
        # Mark migration as active
        _active_migrations["extended-memory"] = True
        
        # Create migration service
        migration_service = create_migration_service(
            source_service="extended-memory",
            target_service="unified-memory",
            batch_size=batch_size,
            dry_run=dry_run,
            backup_enabled=True
        )
        
        # Run migration in background
        background_tasks.add_task(
            _run_extended_memory_migration,
            migration_service,
            migration_id
        )
        
        return {
            "success": True,
            "migration_id": migration_id,
            "message": f"Extended memory migration started {'(dry run)' if dry_run else ''}",
            "status": "in_progress"
        }
        
    except Exception as e:
        _active_migrations["extended-memory"] = False
        logger.error(f"Failed to start extended memory migration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/memory-bank-to-unified")
async def migrate_memory_bank(
    background_tasks: BackgroundTasks,
    dry_run: bool = False,
    batch_size: int = 100
):
    """Migrate memory-bank-mcp data to unified-memory service"""
    migration_id = f"memory-bank_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if _active_migrations.get("memory-bank", False):
        raise HTTPException(
            status_code=409,
            detail="Memory bank migration already in progress"
        )
    
    try:
        # Mark migration as active
        _active_migrations["memory-bank"] = True
        
        # Create migration service
        migration_service = create_migration_service(
            source_service="memory-bank-mcp",
            target_service="unified-memory",
            batch_size=batch_size,
            dry_run=dry_run,
            backup_enabled=True
        )
        
        # Run migration in background
        background_tasks.add_task(
            _run_memory_bank_migration,
            migration_service,
            migration_id
        )
        
        return {
            "success": True,
            "migration_id": migration_id,
            "message": f"Memory bank migration started {'(dry run)' if dry_run else ''}",
            "status": "in_progress"
        }
        
    except Exception as e:
        _active_migrations["memory-bank"] = False
        logger.error(f"Failed to start memory bank migration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{migration_id}")
async def get_migration_status(migration_id: str):
    """Get status of a migration operation"""
    if migration_id not in _migration_results:
        # Check if migration is still active
        service_type = migration_id.split('_')[0]
        if _active_migrations.get(service_type, False):
            return {
                "migration_id": migration_id,
                "status": "in_progress",
                "message": f"Migration {migration_id} is still running"
            }
        else:
            raise HTTPException(status_code=404, detail="Migration not found")
    
    result = _migration_results[migration_id]
    
    return {
        "migration_id": migration_id,
        "status": "completed",
        "total_records": result.total_records,
        "migrated_count": result.migrated_count,
        "failed_count": result.failed_count,
        "duration_seconds": result.duration_seconds,
        "success_rate": (result.migrated_count / result.total_records * 100) if result.total_records > 0 else 0,
        "backup_path": result.backup_path,
        "errors": result.errors[:10] if result.errors else []  # Limit error details
    }

@router.get("/status")
async def get_all_migration_status():
    """Get status of all migration operations"""
    return {
        "active_migrations": _active_migrations,
        "completed_migrations": {
            migration_id: {
                "total_records": result.total_records,
                "migrated_count": result.migrated_count,
                "failed_count": result.failed_count,
                "duration_seconds": result.duration_seconds,
                "success_rate": (result.migrated_count / result.total_records * 100) if result.total_records > 0 else 0
            }
            for migration_id, result in _migration_results.items()
        }
    }

@router.post("/validate-unified-memory")
async def validate_unified_memory():
    """Validate unified memory service functionality"""
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Health check
            health_response = await client.get("http://localhost:3009/health")
            if health_response.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail="Unified memory service health check failed"
                )
            
            # Stats check
            stats_response = await client.get("http://localhost:3009/memory/stats")
            if stats_response.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail="Unified memory service stats endpoint failed"
                )
            
            stats_data = stats_response.json()
            health_data = health_response.json()
            
            return {
                "success": True,
                "service_status": health_data.get("status"),
                "total_memories": stats_data.get("data", {}).get("total_memories", 0),
                "namespaces": stats_data.get("data", {}).get("namespaces", []),
                "message": "Unified memory service is ready for migration"
            }
            
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to unified memory service"
        )
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _run_extended_memory_migration(
    migration_service,
    migration_id: str
):
    """Background task for extended memory migration"""
    try:
        logger.info(f"Starting extended memory migration: {migration_id}")
        result = await migration_service.migrate_extended_memory()
        _migration_results[migration_id] = result
        logger.info(f"Extended memory migration completed: {migration_id}")
    except Exception as e:
        logger.error(f"Extended memory migration failed: {str(e)}")
        # Store error result
        from app.services.memory_migration import MigrationResult
        _migration_results[migration_id] = MigrationResult(
            total_records=0,
            migrated_count=0,
            failed_count=1,
            errors=[str(e)],
            duration_seconds=0.0
        )
    finally:
        _active_migrations["extended-memory"] = False

async def _run_memory_bank_migration(
    migration_service,
    migration_id: str
):
    """Background task for memory bank migration"""
    try:
        logger.info(f"Starting memory bank migration: {migration_id}")
        result = await migration_service.migrate_memory_bank()
        _migration_results[migration_id] = result
        logger.info(f"Memory bank migration completed: {migration_id}")
    except Exception as e:
        logger.error(f"Memory bank migration failed: {str(e)}")
        # Store error result
        from app.services.memory_migration import MigrationResult
        _migration_results[migration_id] = MigrationResult(
            total_records=0,
            migrated_count=0,
            failed_count=1,
            errors=[str(e)],
            duration_seconds=0.0
        )
    finally:
        _active_migrations["memory-bank"] = False