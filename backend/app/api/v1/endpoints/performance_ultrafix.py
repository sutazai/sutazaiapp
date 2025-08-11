"""
ULTRAFIX: Performance Monitoring and Optimization API
Real-time performance metrics and automatic optimization endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List
from datetime import datetime

from app.core.performance_ultrafix import (
    get_performance_monitor,
    run_performance_optimization
)
from app.core.cache_ultrafix import get_ultra_cache
from app.core.connection_pool import get_pool_manager

router = APIRouter(prefix="/performance", tags=["Performance"])


@router.get("/ultrafix/status")
async def get_ultrafix_status() -> Dict[str, Any]:
    """
    ULTRAFIX: Get complete performance optimization status
    
    Returns comprehensive metrics including:
    - Cache hit rates (target: 80%+)
    - Database connection pool status
    - Memory optimization status
    - Container performance metrics
    """
    try:
        # Get cache metrics
        ultra_cache = await get_ultra_cache()
        cache_report = ultra_cache.get_optimization_report()
        
        # Get connection pool metrics
        pool_manager = await get_pool_manager()
        pool_stats = pool_manager.get_stats()
        
        # Get performance metrics
        monitor = await get_performance_monitor()
        performance_report = await monitor.generate_performance_report()
        
        # Calculate overall optimization score
        optimization_score = _calculate_optimization_score(
            cache_report,
            pool_stats,
            performance_report
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'optimization_score': optimization_score,
            'status': _get_status_from_score(optimization_score),
            'cache': {
                'hit_rate': cache_report['hit_rate'],
                'target_hit_rate': 80.0,
                'achieved': cache_report['hit_rate'] >= 80,
                'stats': cache_report['stats'],
                'optimizations': cache_report['optimizations']
            },
            'database': {
                'pool_size': pool_stats.get('pool_size', 0),
                'active_connections': pool_stats.get('active_connections', 0),
                'idle_connections': pool_stats.get('idle_connections', 0),
                'queries_executed': pool_stats.get('db_queries', 0),
                'connection_errors': pool_stats.get('connection_errors', 0)
            },
            'memory': {
                'containers_optimized': performance_report['containers']['optimized'],
                'total_containers': performance_report['containers']['total'],
                'memory_efficiency': performance_report['containers']['memory_efficiency'],
                'potential_savings_mb': performance_report['optimization_opportunities']['potential_memory_savings_mb']
            },
            'system': performance_report['system'],
            'recommendations': performance_report['recommendations']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance check failed: {str(e)}")


@router.post("/ultrafix/optimize")
async def run_ultrafix_optimization(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    ULTRAFIX: Run complete performance optimization
    
    Executes:
    - Cache warming and optimization
    - Memory allocation optimization
    - Resource cleanup
    - Connection pool tuning
    """
    try:
        # Run optimization in background for non-blocking response
        background_tasks.add_task(run_performance_optimization)
        
        return {
            'status': 'optimization_started',
            'timestamp': datetime.now().isoformat(),
            'message': 'ULTRAFIX optimization running in background',
            'expected_improvements': {
                'cache_hit_rate': '80%+',
                'memory_savings': '30%+',
                'response_time': '<200ms',
                'connection_pool': 'optimized'
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/ultrafix/containers")
async def get_container_performance() -> Dict[str, Any]:
    """
    ULTRAFIX: Get real-time container performance metrics
    """
    try:
        monitor = await get_performance_monitor()
        stats = monitor.get_container_stats()
        
        # Group by optimization status
        optimized = [s for s in stats if s['optimized']]
        needs_optimization = [s for s in stats if not s['optimized']]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_containers': len(stats),
            'optimized_count': len(optimized),
            'needs_optimization_count': len(needs_optimization),
            'optimization_rate': round(len(optimized) / len(stats) * 100, 2) if stats else 0,
            'optimized_containers': optimized,
            'needs_optimization': needs_optimization
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Container stats failed: {str(e)}")


@router.post("/ultrafix/cleanup")
async def cleanup_unused_resources() -> Dict[str, Any]:
    """
    ULTRAFIX: Clean up unused Docker resources
    """
    try:
        monitor = await get_performance_monitor()
        cleanup_report = await monitor.cleanup_unused_resources()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cleanup_report': cleanup_report,
            'status': 'success' if cleanup_report['space_reclaimed_mb'] > 0 else 'nothing_to_clean'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/ultrafix/memory-optimization")
async def get_memory_optimization() -> Dict[str, Any]:
    """
    ULTRAFIX: Get memory optimization recommendations
    """
    try:
        monitor = await get_performance_monitor()
        optimization = await monitor.optimize_memory_allocations()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'optimization_report': optimization,
            'status': 'optimizations_available' if optimization['optimizations'] else 'already_optimized'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory optimization check failed: {str(e)}")


@router.get("/ultrafix/benchmarks")
async def get_performance_benchmarks() -> Dict[str, Any]:
    """
    ULTRAFIX: Get performance benchmarks and targets
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'targets': {
            'cache_hit_rate': {
                'baseline': 9.08,
                'target': 80.0,
                'ultra': 95.0,
                'unit': 'percent'
            },
            'response_time': {
                'baseline': 500,
                'target': 200,
                'ultra': 100,
                'unit': 'milliseconds'
            },
            'memory_efficiency': {
                'baseline': 40,
                'target': 70,
                'ultra': 85,
                'unit': 'percent'
            },
            'database_pool_utilization': {
                'baseline': 20,
                'target': 60,
                'ultra': 75,
                'unit': 'percent'
            },
            'concurrent_users': {
                'baseline': 100,
                'target': 1000,
                'ultra': 5000,
                'unit': 'users'
            }
        },
        'optimization_levels': {
            'basic': 'Meets baseline requirements',
            'optimized': 'Achieves target performance',
            'ultrafix': 'Exceeds all performance targets'
        }
    }


def _calculate_optimization_score(
    cache_report: Dict,
    pool_stats: Dict,
    performance_report: Dict
) -> float:
    """Calculate overall optimization score (0-100)"""
    
    scores = []
    
    # Cache score (40% weight)
    cache_hit_rate = cache_report.get('hit_rate', 0)
    cache_score = min(cache_hit_rate / 80 * 100, 100) * 0.4
    scores.append(cache_score)
    
    # Memory efficiency score (30% weight)
    memory_efficiency = performance_report['containers'].get('memory_efficiency', 0)
    memory_score = min(memory_efficiency / 70 * 100, 100) * 0.3
    scores.append(memory_score)
    
    # Container optimization score (20% weight)
    optimization_rate = performance_report['containers'].get('optimization_rate', 0)
    container_score = optimization_rate * 0.2
    scores.append(container_score)
    
    # Connection pool score (10% weight)
    pool_errors = pool_stats.get('connection_errors', 0)
    pool_queries = pool_stats.get('db_queries', 1)
    error_rate = pool_errors / pool_queries if pool_queries > 0 else 0
    pool_score = max(0, (1 - error_rate) * 100) * 0.1
    scores.append(pool_score)
    
    return round(sum(scores), 2)


def _get_status_from_score(score: float) -> str:
    """Get status label from optimization score"""
    if score >= 90:
        return "ULTRAFIX_PERFECT"
    elif score >= 80:
        return "ULTRA_OPTIMIZED"
    elif score >= 70:
        return "WELL_OPTIMIZED"
    elif score >= 50:
        return "PARTIALLY_OPTIMIZED"
    else:
        return "NEEDS_OPTIMIZATION"