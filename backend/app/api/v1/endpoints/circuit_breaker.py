"""
Circuit Breaker API endpoints for monitoring and control
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging

from app.core.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status")
async def get_circuit_breaker_status() -> Dict[str, Any]:
    """
    Get the status of all circuit breakers
    
    Returns:
        Dictionary containing circuit breaker states and metrics
    """
    try:
        pool_manager = await get_pool_manager()
        status = pool_manager.get_circuit_breaker_status()
        
        # Add summary information
        summary = {
            'healthy_circuits': 0,
            'degraded_circuits': 0,
            'failed_circuits': 0
        }
        
        if 'breakers' in status:
            for breaker_data in status['breakers'].values():
                state = breaker_data.get('state', 'unknown')
                if state == 'closed':
                    summary['healthy_circuits'] += 1
                elif state == 'half_open':
                    summary['degraded_circuits'] += 1
                elif state == 'open':
                    summary['failed_circuits'] += 1
        
        return {
            'summary': summary,
            'details': status
        }
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{service}")
async def get_service_circuit_breaker(service: str) -> Dict[str, Any]:
    """
    Get the status of a specific service's circuit breaker
    
    Args:
        service: Service name (ollama, redis, database, agents, external)
    
    Returns:
        Circuit breaker status for the service
    """
    try:
        pool_manager = await get_pool_manager()
        breaker = pool_manager.get_circuit_breaker(service)
        
        if not breaker:
            raise HTTPException(
                status_code=404, 
                detail=f"Circuit breaker for service '{service}' not found"
            )
        
        return breaker.get_metrics()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting circuit breaker for {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset/{service}")
async def reset_service_circuit_breaker(service: str) -> Dict[str, str]:
    """
    Reset a specific service's circuit breaker to closed state
    
    Args:
        service: Service name to reset
    
    Returns:
        Success message
    """
    try:
        pool_manager = await get_pool_manager()
        pool_manager.reset_circuit_breaker(service)
        
        return {
            'status': 'success',
            'message': f"Circuit breaker for '{service}' has been reset"
        }
    except Exception as e:
        logger.error(f"Error resetting circuit breaker for {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-all")
async def reset_all_circuit_breakers() -> Dict[str, str]:
    """
    Reset all circuit breakers to closed state
    
    Returns:
        Success message
    """
    try:
        pool_manager = await get_pool_manager()
        pool_manager.reset_all_circuit_breakers()
        
        return {
            'status': 'success',
            'message': 'All circuit breakers have been reset'
        }
    except Exception as e:
        logger.error(f"Error resetting all circuit breakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_circuit_breaker_metrics() -> Dict[str, Any]:
    """
    Get detailed metrics for all circuit breakers
    
    Returns:
        Comprehensive metrics including success rates, failure counts, etc.
    """
    try:
        pool_manager = await get_pool_manager()
        status = pool_manager.get_circuit_breaker_status()
        stats = pool_manager.get_stats()
        
        # Calculate aggregate metrics
        total_requests = 0
        total_failures = 0
        total_trips = stats.get('circuit_breaker_trips', 0)
        
        if 'breakers' in status:
            for breaker_data in status['breakers'].values():
                metrics = breaker_data.get('metrics', {})
                total_requests += metrics.get('total_calls', 0)
                total_failures += metrics.get('failed_calls', 0)
        
        failure_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'aggregate': {
                'total_requests': total_requests,
                'total_failures': total_failures,
                'total_circuit_trips': total_trips,
                'overall_failure_rate': f"{failure_rate:.2f}%"
            },
            'per_service': status.get('breakers', {}),
            'connection_pool_stats': stats
        }
    except Exception as e:
        logger.error(f"Error getting circuit breaker metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))