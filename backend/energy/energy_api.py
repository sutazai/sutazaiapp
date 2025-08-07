"""
Energy Management API - REST API endpoints for energy monitoring and control
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .energy_profiler import get_global_profiler, start_global_monitoring, stop_global_monitoring
from .power_optimizer import get_global_optimizer, OptimizationStrategy
from .agent_hibernation import get_hibernation_manager
from .workload_scheduler import get_global_scheduler, SchedulingPolicy, Task, TaskPriority
from .resource_allocator import get_global_allocator, AllocationStrategy
from .sustainability_manager import get_global_sustainability_manager, CarbonIntensitySource
from .monitoring_dashboard import create_dashboard

logger = logging.getLogger(__name__)

# Create API router
energy_router = APIRouter(prefix="/api/energy", tags=["energy"])

# Initialize energy management components
profiler = get_global_profiler()
optimizer = get_global_optimizer()
hibernation_manager = get_hibernation_manager()
scheduler = get_global_scheduler()
allocator = get_global_allocator()
sustainability_manager = get_global_sustainability_manager()

@energy_router.get("/status")
async def get_energy_status():
    """Get overall energy management system status"""
    try:
        current_metrics = profiler.get_current_metrics()
        efficiency_metrics = profiler.get_efficiency_metrics()
        optimization_stats = optimizer.get_optimization_stats()
        hibernation_stats = hibernation_manager.get_hibernation_stats()
        scheduling_stats = scheduler.get_scheduling_stats()
        allocation_stats = allocator.get_allocation_stats()
        
        return {
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "efficiency_metrics": efficiency_metrics,
            "optimization": {
                "active": optimization_stats.get("total_optimizations", 0) > 0,
                "strategy": optimization_stats.get("current_strategy", "balanced"),
                "power_saved_w": optimization_stats.get("total_power_saved_w", 0.0)
            },
            "hibernation": {
                "active": hibernation_stats.get("monitoring_active", False),
                "hibernated_agents": hibernation_stats.get("currently_hibernated", 0),
                "total_agents": hibernation_stats.get("total_agents", 0)
            },
            "scheduling": {
                "active": scheduling_stats.get("scheduling_active", False),
                "pending_tasks": scheduling_stats.get("tasks_pending", 0),
                "running_tasks": scheduling_stats.get("tasks_running", 0)
            },
            "resource_allocation": {
                "active": allocation_stats.get("monitoring_active", False),
                "total_allocations": allocation_stats.get("total_allocations", 0),
                "cpu_utilization": allocation_stats.get("resource_utilization", {}).get("cpu_cores", 0.0)
            }
        }
    except Exception as e:
        logger.error(f"Error getting energy status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/monitoring/start")
async def start_energy_monitoring(background_tasks: BackgroundTasks):
    """Start energy monitoring"""
    try:
        background_tasks.add_task(start_global_monitoring)
        return {"message": "Energy monitoring started", "status": "success"}
    except Exception as e:
        logger.error(f"Error starting energy monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/monitoring/stop")
async def stop_energy_monitoring(background_tasks: BackgroundTasks):
    """Stop energy monitoring"""
    try:
        background_tasks.add_task(stop_global_monitoring)
        return {"message": "Energy monitoring stopped", "status": "success"}
    except Exception as e:
        logger.error(f"Error stopping energy monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/metrics/current")
async def get_current_energy_metrics():
    """Get current energy consumption metrics"""
    try:
        current_metrics = profiler.get_current_metrics()
        efficiency_metrics = profiler.get_efficiency_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "efficiency_metrics": efficiency_metrics
        }
    except Exception as e:
        logger.error(f"Error getting current metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/metrics/history")
async def get_energy_history(hours: float = 24.0):
    """Get energy consumption history"""
    try:
        energy_metrics = profiler.calculate_energy_metrics(hours)
        
        return {
            "period_hours": hours,
            "start_time": energy_metrics.start_time.isoformat(),
            "end_time": energy_metrics.end_time.isoformat(),
            "total_energy_wh": energy_metrics.total_energy_wh,
            "avg_power_w": energy_metrics.avg_power_w,
            "peak_power_w": energy_metrics.peak_power_w,
            "cpu_energy_wh": energy_metrics.cpu_energy_wh,
            "memory_energy_wh": energy_metrics.memory_energy_wh,
            "co2_emission_g": energy_metrics.co2_emission_g,
            "measurement_count": len(energy_metrics.measurements)
        }
    except Exception as e:
        logger.error(f"Error getting energy history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/optimization/start")
async def start_power_optimization(strategy: str = "balanced"):
    """Start power optimization with specified strategy"""
    try:
        if strategy not in [s.value for s in OptimizationStrategy]:
            raise HTTPException(status_code=400, detail="Invalid optimization strategy")
        
        optimizer.strategy = OptimizationStrategy(strategy)
        optimizer.start_optimization()
        
        return {
            "message": f"Power optimization started with {strategy} strategy",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error starting power optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/optimization/stop")
async def stop_power_optimization():
    """Stop power optimization"""
    try:
        optimizer.stop_optimization()
        return {"message": "Power optimization stopped", "status": "success"}
    except Exception as e:
        logger.error(f"Error stopping power optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/optimization/stats")
async def get_optimization_statistics():
    """Get power optimization statistics"""
    try:
        stats = optimizer.get_optimization_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error getting optimization stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/hibernation/start")
async def start_agent_hibernation():
    """Start agent hibernation monitoring"""
    try:
        hibernation_manager.start_monitoring()
        return {"message": "Agent hibernation monitoring started", "status": "success"}
    except Exception as e:
        logger.error(f"Error starting hibernation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/hibernation/stop")
async def stop_agent_hibernation():
    """Stop agent hibernation monitoring"""
    try:
        hibernation_manager.stop_monitoring()
        return {"message": "Agent hibernation monitoring stopped", "status": "success"}
    except Exception as e:
        logger.error(f"Error stopping hibernation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/hibernation/stats")
async def get_hibernation_statistics():
    """Get agent hibernation statistics"""
    try:
        stats = hibernation_manager.get_hibernation_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error getting hibernation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/hibernation/wake-all")
async def wake_all_hibernated_agents():
    """Wake all hibernated agents"""
    try:
        woken_count = hibernation_manager.force_wake_all()
        return {
            "message": f"Woke {woken_count} hibernated agents",
            "woken_count": woken_count,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error waking all agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/hibernation/agent/{agent_id}/wake")
async def wake_specific_agent(agent_id: str):
    """Wake a specific hibernated agent"""
    try:
        success = hibernation_manager.wake_agent(agent_id, "manual_wake")
        if success:
            return {"message": f"Agent {agent_id} woken up", "status": "success"}
        else:
            raise HTTPException(status_code=404, detail="Agent not found or not hibernated")
    except Exception as e:
        logger.error(f"Error waking agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/hibernation/agent/{agent_id}")
async def get_agent_hibernation_info(agent_id: str):
    """Get hibernation information for a specific agent"""
    try:
        info = hibernation_manager.get_agent_hibernation_info(agent_id)
        return {"status": "success", "data": info}
    except Exception as e:
        logger.error(f"Error getting hibernation info for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/scheduling/start")
async def start_workload_scheduling(policy: str = "balanced"):
    """Start energy-aware workload scheduling"""
    try:
        if policy not in [p.value for p in SchedulingPolicy]:
            raise HTTPException(status_code=400, detail="Invalid scheduling policy")
        
        scheduler.policy = SchedulingPolicy(policy)
        scheduler.start_scheduling()
        
        return {
            "message": f"Workload scheduling started with {policy} policy",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error starting scheduling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/scheduling/stop")
async def stop_workload_scheduling():
    """Stop energy-aware workload scheduling"""
    try:
        scheduler.stop_scheduling()
        return {"message": "Workload scheduling stopped", "status": "success"}
    except Exception as e:
        logger.error(f"Error stopping scheduling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/scheduling/stats")
async def get_scheduling_statistics():
    """Get workload scheduling statistics"""
    try:
        stats = scheduler.get_scheduling_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error getting scheduling stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/scheduling/task")
async def submit_task(task_data: Dict[str, Any]):
    """Submit a task for energy-aware scheduling"""
    try:
        # Validate required fields
        required_fields = ["task_id", "agent_id", "task_type", "estimated_duration", "estimated_power_w"]
        for field in required_fields:
            if field not in task_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create task object
        task = Task(
            task_id=task_data["task_id"],
            agent_id=task_data["agent_id"],
            task_type=task_data["task_type"],
            priority=TaskPriority(task_data.get("priority", 3)),
            estimated_duration=task_data["estimated_duration"],
            estimated_cpu_usage=task_data.get("estimated_cpu_usage", 50.0),
            estimated_memory_mb=task_data.get("estimated_memory_mb", 1024.0),
            estimated_power_w=task_data["estimated_power_w"],
            deadline=datetime.fromisoformat(task_data["deadline"]) if task_data.get("deadline") else None,
            dependencies=task_data.get("dependencies", []),
            metadata=task_data.get("metadata", {})
        )
        
        success = scheduler.submit_task(task)
        if success:
            return {"message": f"Task {task.task_id} submitted for scheduling", "status": "success"}
        else:
            raise HTTPException(status_code=400, detail="Task submission failed")
            
    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/scheduling/task/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    try:
        status = scheduler.get_task_status(task_id)
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/allocation/start")
async def start_resource_allocation(strategy: str = "energy_proportional"):
    """Start energy-aware resource allocation"""
    try:
        if strategy not in [s.value for s in AllocationStrategy]:
            raise HTTPException(status_code=400, detail="Invalid allocation strategy")
        
        allocator.strategy = AllocationStrategy(strategy)
        allocator.start_monitoring()
        
        return {
            "message": f"Resource allocation started with {strategy} strategy",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error starting resource allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/allocation/stop")
async def stop_resource_allocation():
    """Stop energy-aware resource allocation"""
    try:
        allocator.stop_monitoring()
        return {"message": "Resource allocation stopped", "status": "success"}
    except Exception as e:
        logger.error(f"Error stopping resource allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/allocation/stats")
async def get_allocation_statistics():
    """Get resource allocation statistics"""
    try:
        stats = allocator.get_allocation_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error getting allocation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/allocation/agent/{agent_id}")
async def allocate_resources_for_agent(agent_id: str, requirements: Dict[str, Any]):
    """Allocate resources for a specific agent"""
    try:
        allocation = allocator.allocate_resources(agent_id, requirements)
        if allocation:
            return {
                "message": f"Resources allocated for agent {agent_id}",
                "allocation": {
                    "cpu_cores": allocation.cpu_cores,
                    "cpu_frequency_mhz": allocation.cpu_frequency_mhz,
                    "memory_mb": allocation.memory_mb,
                    "power_budget_w": allocation.power_budget_w
                },
                "status": "success"
            }
        else:
            raise HTTPException(status_code=400, detail="Resource allocation failed")
    except Exception as e:
        logger.error(f"Error allocating resources for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.delete("/allocation/agent/{agent_id}")
async def deallocate_resources_for_agent(agent_id: str):
    """Deallocate resources for a specific agent"""
    try:
        success = allocator.deallocate_resources(agent_id)
        if success:
            return {"message": f"Resources deallocated for agent {agent_id}", "status": "success"}
        else:
            raise HTTPException(status_code=404, detail="Agent allocation not found")
    except Exception as e:
        logger.error(f"Error deallocating resources for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/allocation/agent/{agent_id}")
async def get_agent_allocation(agent_id: str):
    """Get resource allocation for a specific agent"""
    try:
        allocation = allocator.get_agent_allocation(agent_id)
        if allocation:
            return {"status": "success", "data": allocation}
        else:
            raise HTTPException(status_code=404, detail="Agent allocation not found")
    except Exception as e:
        logger.error(f"Error getting allocation for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/sustainability/metrics")
async def get_sustainability_metrics(hours: float = 24.0):
    """Get comprehensive sustainability metrics"""
    try:
        metrics = sustainability_manager.calculate_sustainability_metrics(hours)
        budget_status = sustainability_manager.get_daily_budget_status()
        
        return {
            "status": "success",
            "data": {
                "metrics": {
                    "period_start": metrics.period_start.isoformat(),
                    "period_end": metrics.period_end.isoformat(),
                    "total_energy_kwh": metrics.total_energy_kwh,
                    "total_co2_kg": metrics.total_co2_kg,
                    "avg_carbon_intensity": metrics.avg_carbon_intensity,
                    "renewable_energy_kwh": metrics.renewable_energy_kwh,
                    "efficiency_score": metrics.efficiency_score,
                    "sustainability_grade": metrics.sustainability_grade,
                    "carbon_saved_kg": metrics.carbon_saved_kg,
                    "energy_saved_kwh": metrics.energy_saved_kwh
                },
                "daily_budget": budget_status
            }
        }
    except Exception as e:
        logger.error(f"Error getting sustainability metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/sustainability/recommendations")
async def get_sustainability_recommendations():
    """Get sustainability improvement recommendations"""
    try:
        recommendations = sustainability_manager.get_sustainability_recommendations()
        return {"status": "success", "data": recommendations}
    except Exception as e:
        logger.error(f"Error getting sustainability recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.get("/sustainability/forecast")
async def get_carbon_forecast(hours: int = 24):
    """Get carbon intensity forecast"""
    try:
        forecast = sustainability_manager.get_carbon_forecast(hours)
        return {"status": "success", "data": forecast}
    except Exception as e:
        logger.error(f"Error getting carbon forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/sustainability/budget")
async def set_daily_budgets(budgets: Dict[str, float]):
    """Set daily energy and carbon budgets"""
    try:
        if "energy_kwh" in budgets:
            sustainability_manager._daily_budgets["energy_kwh"] = budgets["energy_kwh"]
        if "carbon_kg" in budgets:
            sustainability_manager._daily_budgets["carbon_kg"] = budgets["carbon_kg"]
        if "cost_usd" in budgets:
            sustainability_manager._daily_budgets["cost_usd"] = budgets["cost_usd"]
        
        return {"message": "Daily budgets updated", "status": "success"}
    except Exception as e:
        logger.error(f"Error setting daily budgets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/export/energy")
async def export_energy_data(hours: float = 24.0, format: str = "json"):
    """Export energy consumption data"""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'csv'")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/tmp/energy_export_{timestamp}.{format}"
        
        if format == "json":
            profiler.export_measurements(filename, hours)
        # CSV export would be implemented here
        
        return {
            "message": "Energy data exported",
            "filename": filename,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error exporting energy data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/export/sustainability")
async def export_sustainability_report(days: int = 30):
    """Export comprehensive sustainability report"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/tmp/sustainability_report_{timestamp}.json"
        
        sustainability_manager.export_sustainability_report(filename, days)
        
        return {
            "message": "Sustainability report exported",
            "filename": filename,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error exporting sustainability report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/system/start-all")
async def start_all_energy_systems():
    """Start all energy management systems"""
    try:
        # Start all components
        start_global_monitoring()
        optimizer.start_optimization()
        hibernation_manager.start_monitoring()
        scheduler.start_scheduling()
        allocator.start_monitoring()
        sustainability_manager.start_monitoring()
        
        return {
            "message": "All energy management systems started",
            "systems": [
                "energy_monitoring",
                "power_optimization", 
                "agent_hibernation",
                "workload_scheduling",
                "resource_allocation",
                "sustainability_tracking"
            ],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error starting all energy systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@energy_router.post("/system/stop-all")
async def stop_all_energy_systems():
    """Stop all energy management systems"""
    try:
        # Stop all components
        stop_global_monitoring()
        optimizer.stop_optimization()
        hibernation_manager.stop_monitoring()
        scheduler.stop_scheduling()
        allocator.stop_monitoring()
        sustainability_manager.stop_monitoring()
        
        return {
            "message": "All energy management systems stopped",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error stopping all energy systems: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add router to main app (this would be done in the main app file)
def setup_energy_routes(app):
    """Setup energy management routes in the main FastAPI app"""
    app.include_router(energy_router)