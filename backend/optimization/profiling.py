"""
Performance Profiling Tools for SutazAI
Advanced profiling and optimization analysis
"""

import asyncio
import cProfile
import pstats
import io
import logging
import time
import tracemalloc
import sys
import gc
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass
from pathlib import Path
import threading
import linecache

logger = logging.getLogger(__name__)

@dataclass
class ProfileResult:
    """Profiling result data"""
    function_name: str
    total_time: float
    calls: int
    time_per_call: float
    cumulative_time: float
    filename: str
    line_number: int

class PerformanceProfiler:
    """Advanced performance profiling system"""
    
    def __init__(self):
        self.profiling_active = False
        self.profiler = None
        self.memory_profiler_active = False
        self.function_times = {}
        self.memory_snapshots = []
        self.profiling_results = {}
    
    async def initialize(self):
        """Initialize performance profiler"""
        logger.info("🔄 Initializing Performance Profiler")
        
        # Start memory tracing
        tracemalloc.start()
        self.memory_profiler_active = True
        
        # Start memory monitoring
        asyncio.create_task(self._memory_monitoring_loop())
        
        logger.info("✅ Performance Profiler initialized")
    
    def start_profiling(self, name: str = "default"):
        """Start CPU profiling"""
        if self.profiling_active:
            logger.warning("Profiling already active")
            return
        
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.profiling_active = True
        
        logger.info(f"Started CPU profiling: {name}")
    
    def stop_profiling(self, name: str = "default") -> Dict[str, Any]:
        """Stop CPU profiling and return results"""
        if not self.profiling_active or not self.profiler:
            logger.warning("No active profiling to stop")
            return {}
        
        self.profiler.disable()
        self.profiling_active = False
        
        # Generate profiling report
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        # Parse results
        results = self._parse_profiling_results(ps)
        
        # Store results
        self.profiling_results[name] = {
            "timestamp": time.time(),
            "results": results,
            "raw_output": s.getvalue()
        }
        
        logger.info(f"Stopped CPU profiling: {name}")
        return results
    
    def _parse_profiling_results(self, stats: pstats.Stats) -> List[ProfileResult]:
        """Parse profiling statistics into structured results"""
        results = []
        
        try:
            for (filename, line_number, function_name), (calls, total_time, cumulative_time) in stats.stats.items():
                if calls > 0:
                    result = ProfileResult(
                        function_name=function_name,
                        total_time=total_time,
                        calls=calls,
                        time_per_call=total_time / calls,
                        cumulative_time=cumulative_time,
                        filename=filename,
                        line_number=line_number
                    )
                    results.append(result)
            
            # Sort by cumulative time
            results.sort(key=lambda x: x.cumulative_time, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to parse profiling results: {e}")
        
        return results[:50]  # Return top 50
    
    async def _memory_monitoring_loop(self):
        """Monitor memory usage"""
        while self.memory_profiler_active:
            try:
                if tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    
                    snapshot = {
                        "timestamp": time.time(),
                        "current_mb": current / 1024 / 1024,
                        "peak_mb": peak / 1024 / 1024,
                        "tracemalloc_active": True
                    }
                    
                    self.memory_snapshots.append(snapshot)
                    
                    # Keep only recent snapshots
                    self.memory_snapshots = self.memory_snapshots[-1000:]
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(5)
    
    def take_memory_snapshot(self, name: str = None) -> Dict[str, Any]:
        """Take detailed memory snapshot"""
        try:
            if not tracemalloc.is_tracing():
                return {"error": "Memory tracing not active"}
            
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            # Get top memory consumers
            top_consumers = []
            for stat in top_stats[:20]:
                top_consumers.append({
                    "filename": stat.traceback.filename,
                    "line_number": stat.traceback.lineno,
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                })
            
            current, peak = tracemalloc.get_traced_memory()
            
            result = {
                "timestamp": time.time(),
                "name": name or f"snapshot_{int(time.time())}",
                "current_memory_mb": current / 1024 / 1024,
                "peak_memory_mb": peak / 1024 / 1024,
                "top_consumers": top_consumers
            }
            
            logger.info(f"Memory snapshot taken: {result['name']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            return {"error": str(e)}
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile a specific function call"""
        start_time = time.time()
        
        # Enable profiling for this function
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            profiler.disable()
            
            # Generate stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)
            
            return {
                "function": func.__name__,
                "execution_time": execution_time,
                "result": result,
                "profiling_output": s.getvalue(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            profiler.disable()
            logger.error(f"Function profiling failed: {e}")
            return {
                "function": func.__name__,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def profile_async_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile an async function call"""
        start_time = time.time()
        
        # Enable profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            profiler.disable()
            
            # Generate stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)
            
            return {
                "function": func.__name__,
                "execution_time": execution_time,
                "result": result,
                "profiling_output": s.getvalue(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            profiler.disable()
            logger.error(f"Async function profiling failed: {e}")
            return {
                "function": func.__name__,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def analyze_memory_growth(self, hours: int = 1) -> Dict[str, Any]:
        """Analyze memory growth over time"""
        cutoff_time = time.time() - (hours * 3600)
        
        relevant_snapshots = [
            s for s in self.memory_snapshots
            if s["timestamp"] > cutoff_time
        ]
        
        if len(relevant_snapshots) < 2:
            return {"error": "Not enough data for analysis"}
        
        # Calculate growth rate
        first_snapshot = relevant_snapshots[0]
        last_snapshot = relevant_snapshots[-1]
        
        time_diff = last_snapshot["timestamp"] - first_snapshot["timestamp"]
        memory_diff = last_snapshot["current_mb"] - first_snapshot["current_mb"]
        
        growth_rate_mb_per_hour = (memory_diff / time_diff) * 3600 if time_diff > 0 else 0
        
        # Calculate average and peak usage
        current_values = [s["current_mb"] for s in relevant_snapshots]
        peak_values = [s["peak_mb"] for s in relevant_snapshots]
        
        return {
            "analysis_period_hours": hours,
            "snapshots_analyzed": len(relevant_snapshots),
            "memory_growth": {
                "initial_mb": first_snapshot["current_mb"],
                "final_mb": last_snapshot["current_mb"],
                "total_growth_mb": memory_diff,
                "growth_rate_mb_per_hour": growth_rate_mb_per_hour
            },
            "statistics": {
                "avg_current_mb": sum(current_values) / len(current_values),
                "max_current_mb": max(current_values),
                "min_current_mb": min(current_values),
                "avg_peak_mb": sum(peak_values) / len(peak_values),
                "max_peak_mb": max(peak_values)
            },
            "trend": "increasing" if growth_rate_mb_per_hour > 1 else "stable" if growth_rate_mb_per_hour > -1 else "decreasing"
        }
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get summary of all profiling data"""
        return {
            "profiling_active": self.profiling_active,
            "memory_profiler_active": self.memory_profiler_active,
            "profiling_sessions": len(self.profiling_results),
            "memory_snapshots": len(self.memory_snapshots),
            "tracemalloc_active": tracemalloc.is_tracing(),
            "recent_profiling_results": list(self.profiling_results.keys())[-5:],
            "memory_analysis": self.analyze_memory_growth(1) if self.memory_snapshots else {}
        }
    
    async def export_profiling_data(self, filepath: str):
        """Export all profiling data"""
        try:
            data = {
                "export_timestamp": time.time(),
                "profiling_summary": self.get_profiling_summary(),
                "profiling_results": self.profiling_results,
                "memory_snapshots": self.memory_snapshots[-100:],  # Last 100 snapshots
                "memory_analysis": self.analyze_memory_growth(24)
            }
            
            with open(filepath, 'w') as f:
                import json
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Profiling data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export profiling data: {e}")
    
    async def shutdown(self):
        """Shutdown profiler"""
        logger.info("🛑 Shutting down Performance Profiler")
        
        # Stop active profiling
        if self.profiling_active:
            self.stop_profiling("shutdown")
        
        # Stop memory monitoring
        self.memory_profiler_active = False
        
        # Stop memory tracing
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        # Export final data
        try:
            export_path = Path("/opt/sutazaiapp/logs/final_profiling_data.json")
            await self.export_profiling_data(str(export_path))
        except Exception as e:
            logger.warning(f"Failed to export final profiling data: {e}")
        
        logger.info("✅ Performance Profiler shutdown complete")

# Profiling decorators
def profile_execution(save_result: bool = True):
    """Decorator to profile function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler = performance_profiler
            result = await profiler.profile_async_function(func, *args, **kwargs)
            
            if save_result:
                # Store in function times
                profiler.function_times[func.__name__] = result
            
            return result["result"] if "result" in result else None
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            profiler = performance_profiler
            result = profiler.profile_function(func, *args, **kwargs)
            
            if save_result:
                profiler.function_times[func.__name__] = result
            
            return result["result"] if "result" in result else None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def memory_snapshot(name: str = None):
    """Decorator to take memory snapshot before and after function"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            snapshot_name = name or f"{func.__name__}_snapshot"
            
            # Take before snapshot
            before = performance_profiler.take_memory_snapshot(f"{snapshot_name}_before")
            
            try:
                result = await func(*args, **kwargs)
                
                # Take after snapshot
                after = performance_profiler.take_memory_snapshot(f"{snapshot_name}_after")
                
                # Log memory usage
                if "current_memory_mb" in before and "current_memory_mb" in after:
                    memory_diff = after["current_memory_mb"] - before["current_memory_mb"]
                    logger.info(f"Function {func.__name__} memory usage: {memory_diff:.2f}MB")
                
                return result
                
            except Exception as e:
                # Take error snapshot
                performance_profiler.take_memory_snapshot(f"{snapshot_name}_error")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            snapshot_name = name or f"{func.__name__}_snapshot"
            
            before = performance_profiler.take_memory_snapshot(f"{snapshot_name}_before")
            
            try:
                result = func(*args, **kwargs)
                
                after = performance_profiler.take_memory_snapshot(f"{snapshot_name}_after")
                
                if "current_memory_mb" in before and "current_memory_mb" in after:
                    memory_diff = after["current_memory_mb"] - before["current_memory_mb"]
                    logger.info(f"Function {func.__name__} memory usage: {memory_diff:.2f}MB")
                
                return result
                
            except Exception as e:
                performance_profiler.take_memory_snapshot(f"{snapshot_name}_error")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Global profiler instance
performance_profiler = PerformanceProfiler()
