"""
Energy-Aware Resource Allocator - Intelligent resource allocation for power optimization
"""

import time
import threading
import logging
import psutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)

class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    ENERGY_PROPORTIONAL = "energy_proportional"  # Allocate based on energy efficiency
    WORKLOAD_AWARE = "workload_aware"            # Allocate based on workload requirements
    THERMAL_BALANCED = "thermal_balanced"         # Balance thermal distribution
    CARBON_OPTIMIZED = "carbon_optimized"        # Optimize for carbon footprint
    PERFORMANCE_FIRST = "performance_first"      # Maximize performance

class ResourceType(Enum):
    """Types of resources that can be allocated"""
    CPU_CORES = "cpu_cores"
    CPU_FREQUENCY = "cpu_frequency"
    MEMORY = "memory"
    BANDWIDTH = "bandwidth"
    STORAGE_IO = "storage_io"

@dataclass
class ResourceConstraints:
    """Resource constraints for allocation"""
    min_cpu_cores: int = 1
    max_cpu_cores: int = 12
    min_cpu_frequency_mhz: int = 1000
    max_cpu_frequency_mhz: int = 3000
    min_memory_mb: int = 512
    max_memory_mb: int = 8192
    min_bandwidth_mbps: float = 10.0
    max_bandwidth_mbps: float = 1000.0
    thermal_limit_celsius: float = 80.0
    power_limit_watts: float = 100.0

@dataclass
class ResourceAllocation:
    """Resource allocation for an agent/workload"""
    agent_id: str
    cpu_cores: int
    cpu_frequency_mhz: int
    memory_mb: int
    bandwidth_mbps: float
    storage_io_priority: int  # 0-7, where 7 is highest priority
    power_budget_w: float
    thermal_budget_c: float
    allocation_timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    efficiency_score: float = 0.0
    utilization_history: List[float] = field(default_factory=list)

@dataclass
class SystemResources:
    """Current system resource availability"""
    total_cpu_cores: int
    available_cpu_cores: int
    total_memory_mb: int
    available_memory_mb: int
    total_bandwidth_mbps: float
    available_bandwidth_mbps: float
    current_temperature_c: float
    current_power_w: float
    cpu_frequencies: List[int]
    load_average: float

class EnergyAwareResourceAllocator:
    """Main energy-aware resource allocation system"""
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.ENERGY_PROPORTIONAL):
        """
        Initialize the resource allocator
        
        Args:
            strategy: Allocation strategy to use
        """
        self.strategy = strategy
        self.constraints = ResourceConstraints()
        
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._system_resources = SystemResources(
            total_cpu_cores=psutil.cpu_count(),
            available_cpu_cores=psutil.cpu_count(),
            total_memory_mb=int(psutil.virtual_memory().total / (1024 * 1024)),
            available_memory_mb=int(psutil.virtual_memory().available / (1024 * 1024)),
            total_bandwidth_mbps=1000.0,  # Assumed
            available_bandwidth_mbps=1000.0,
            current_temperature_c=50.0,
            current_power_w=50.0,
            cpu_frequencies=[1000, 1500, 2000, 2500, 3000],
            load_average=0.0
        )
        
        self._allocation_history: List[Dict[str, Any]] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Energy efficiency coefficients
        self._cpu_power_coeffs = {
            'base': 8.0,      # Base power per core (W)
            'freq': 0.002,    # Power per MHz per core
            'util': 15.0      # Additional power per 100% utilization
        }
        
        self._memory_power_coeff = 0.003  # Power per MB (W)
    
    def start_monitoring(self) -> None:
        """Start resource allocation monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource allocation monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource allocation monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Resource allocation monitoring stopped")
    
    def allocate_resources(self, agent_id: str, workload_requirements: Dict[str, Any]) -> Optional[ResourceAllocation]:
        """
        Allocate resources for an agent based on workload requirements
        
        Args:
            agent_id: ID of the agent requesting resources
            workload_requirements: Dictionary containing workload requirements
            
        Returns:
            ResourceAllocation: Allocated resources or None if allocation failed
        """
        try:
            # Extract requirements
            required_cpu_cores = workload_requirements.get('cpu_cores', 2)
            required_memory_mb = workload_requirements.get('memory_mb', 1024)
            required_bandwidth_mbps = workload_requirements.get('bandwidth_mbps', 100.0)
            expected_duration = workload_requirements.get('duration_seconds', 3600)
            priority = workload_requirements.get('priority', 3)  # 1-5 scale
            
            # Update system resources
            self._update_system_resources()
            
            # Check if resources are available
            if not self._check_resource_availability(required_cpu_cores, required_memory_mb, required_bandwidth_mbps):
                logger.warning(f"Insufficient resources for agent {agent_id}")
                return None
            
            # Calculate optimal allocation based on strategy
            allocation = self._calculate_optimal_allocation(
                agent_id, 
                required_cpu_cores, 
                required_memory_mb, 
                required_bandwidth_mbps,
                expected_duration,
                priority
            )
            
            if allocation:
                # Reserve resources
                self._reserve_resources(allocation)
                
                # Store allocation
                with self._lock:
                    self._allocations[agent_id] = allocation
                
                # Record allocation event
                self._record_allocation_event(allocation, "allocated")
                
                logger.info(f"Allocated resources for agent {agent_id}: "
                          f"{allocation.cpu_cores} cores, {allocation.memory_mb}MB, "
                          f"{allocation.cpu_frequency_mhz}MHz")
                
                return allocation
            
        except Exception as e:
            logger.error(f"Error allocating resources for agent {agent_id}: {e}")
        
        return None
    
    def deallocate_resources(self, agent_id: str) -> bool:
        """
        Deallocate resources for an agent
        
        Args:
            agent_id: ID of the agent to deallocate resources for
            
        Returns:
            bool: True if deallocation was successful
        """
        try:
            with self._lock:
                if agent_id not in self._allocations:
                    return False
                
                allocation = self._allocations[agent_id]
                
                # Release resources
                self._release_resources(allocation)
                
                # Remove allocation
                del self._allocations[agent_id]
            
            # Record deallocation event
            self._record_allocation_event(allocation, "deallocated")
            
            logger.info(f"Deallocated resources for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deallocating resources for agent {agent_id}: {e}")
            return False
    
    def update_allocation(self, agent_id: str, new_requirements: Dict[str, Any]) -> bool:
        """
        Update resource allocation for an agent
        
        Args:
            agent_id: ID of the agent
            new_requirements: New resource requirements
            
        Returns:
            bool: True if update was successful
        """
        try:
            with self._lock:
                if agent_id not in self._allocations:
                    return False
                
                old_allocation = self._allocations[agent_id]
            
            # Deallocate old resources
            self._release_resources(old_allocation)
            
            # Allocate new resources
            new_allocation = self.allocate_resources(agent_id, new_requirements)
            
            if new_allocation:
                logger.info(f"Updated allocation for agent {agent_id}")
                return True
            else:
                # Restore old allocation if new allocation failed
                self._reserve_resources(old_allocation)
                with self._lock:
                    self._allocations[agent_id] = old_allocation
                logger.warning(f"Failed to update allocation for agent {agent_id}, restored old allocation")
                return False
                
        except Exception as e:
            logger.error(f"Error updating allocation for agent {agent_id}: {e}")
            return False
    
    def _calculate_optimal_allocation(
        self, 
        agent_id: str, 
        required_cpu_cores: int, 
        required_memory_mb: int,
        required_bandwidth_mbps: float,
        expected_duration: int,
        priority: int
    ) -> Optional[ResourceAllocation]:
        """Calculate optimal resource allocation based on strategy"""
        
        if self.strategy == AllocationStrategy.ENERGY_PROPORTIONAL:
            return self._calculate_energy_proportional_allocation(
                agent_id, required_cpu_cores, required_memory_mb, 
                required_bandwidth_mbps, expected_duration, priority
            )
        elif self.strategy == AllocationStrategy.WORKLOAD_AWARE:
            return self._calculate_workload_aware_allocation(
                agent_id, required_cpu_cores, required_memory_mb,
                required_bandwidth_mbps, expected_duration, priority
            )
        elif self.strategy == AllocationStrategy.THERMAL_BALANCED:
            return self._calculate_thermal_balanced_allocation(
                agent_id, required_cpu_cores, required_memory_mb,
                required_bandwidth_mbps, expected_duration, priority
            )
        elif self.strategy == AllocationStrategy.CARBON_OPTIMIZED:
            return self._calculate_carbon_optimized_allocation(
                agent_id, required_cpu_cores, required_memory_mb,
                required_bandwidth_mbps, expected_duration, priority
            )
        else:  # PERFORMANCE_FIRST
            return self._calculate_performance_first_allocation(
                agent_id, required_cpu_cores, required_memory_mb,
                required_bandwidth_mbps, expected_duration, priority
            )
    
    def _calculate_energy_proportional_allocation(
        self, agent_id: str, cpu_cores: int, memory_mb: int, 
        bandwidth_mbps: float, duration: int, priority: int
    ) -> Optional[ResourceAllocation]:
        """Calculate allocation optimized for energy efficiency"""
        
        # Find the most energy-efficient CPU frequency
        optimal_frequency = self._find_optimal_frequency(cpu_cores, duration)
        
        # Adjust cores based on energy efficiency
        if priority <= 2:  # Low priority tasks
            cpu_cores = max(1, cpu_cores - 1)  # Use fewer cores
            optimal_frequency = min(optimal_frequency, 2000)  # Lower frequency
        elif priority >= 4:  # High priority tasks
            cpu_cores = min(self.constraints.max_cpu_cores, cpu_cores + 1)
        
        # Calculate power budget
        power_budget = self._calculate_power_consumption(cpu_cores, optimal_frequency, memory_mb)
        
        return ResourceAllocation(
            agent_id=agent_id,
            cpu_cores=cpu_cores,
            cpu_frequency_mhz=optimal_frequency,
            memory_mb=memory_mb,
            bandwidth_mbps=bandwidth_mbps,
            storage_io_priority=max(1, 8 - priority),  # Inverse priority for I/O
            power_budget_w=power_budget,
            thermal_budget_c=self._calculate_thermal_budget(cpu_cores, optimal_frequency),
            efficiency_score=self._calculate_efficiency_score(cpu_cores, optimal_frequency, memory_mb, power_budget)
        )
    
    def _calculate_workload_aware_allocation(
        self, agent_id: str, cpu_cores: int, memory_mb: int,
        bandwidth_mbps: float, duration: int, priority: int
    ) -> Optional[ResourceAllocation]:
        """Calculate allocation based on workload characteristics"""
        
        # Adjust resources based on expected workload
        if duration > 3600:  # Long-running tasks
            # Use lower frequency for sustained performance
            frequency = 2000
            # Ensure adequate memory for long tasks
            memory_mb = int(memory_mb * 1.2)
        else:  # Short tasks
            # Use higher frequency for quick completion
            frequency = 2700
        
        # Adjust based on system load
        load_factor = min(1.0, self._system_resources.load_average / self._system_resources.total_cpu_cores)
        if load_factor > 0.8:
            # System is heavily loaded, be conservative
            cpu_cores = max(1, cpu_cores - 1)
            frequency = min(frequency, 2200)
        
        power_budget = self._calculate_power_consumption(cpu_cores, frequency, memory_mb)
        
        return ResourceAllocation(
            agent_id=agent_id,
            cpu_cores=cpu_cores,
            cpu_frequency_mhz=frequency,
            memory_mb=memory_mb,
            bandwidth_mbps=bandwidth_mbps,
            storage_io_priority=priority,
            power_budget_w=power_budget,
            thermal_budget_c=self._calculate_thermal_budget(cpu_cores, frequency),
            efficiency_score=self._calculate_efficiency_score(cpu_cores, frequency, memory_mb, power_budget)
        )
    
    def _calculate_thermal_balanced_allocation(
        self, agent_id: str, cpu_cores: int, memory_mb: int,
        bandwidth_mbps: float, duration: int, priority: int
    ) -> Optional[ResourceAllocation]:
        """Calculate allocation optimized for thermal management"""
        
        # Reduce frequency if temperature is high
        frequency = 2700
        if self._system_resources.current_temperature_c > 70:
            frequency = 2000
        elif self._system_resources.current_temperature_c > 60:
            frequency = 2200
        
        # Distribute load across more cores at lower frequency for better thermal distribution
        if self._system_resources.current_temperature_c > 65:
            cpu_cores = min(self.constraints.max_cpu_cores, cpu_cores + 1)
            frequency = max(1000, frequency - 300)
        
        power_budget = self._calculate_power_consumption(cpu_cores, frequency, memory_mb)
        thermal_budget = min(75.0, self.constraints.thermal_limit_celsius)
        
        return ResourceAllocation(
            agent_id=agent_id,
            cpu_cores=cpu_cores,
            cpu_frequency_mhz=frequency,
            memory_mb=memory_mb,
            bandwidth_mbps=bandwidth_mbps,
            storage_io_priority=priority,
            power_budget_w=power_budget,
            thermal_budget_c=thermal_budget,
            efficiency_score=self._calculate_efficiency_score(cpu_cores, frequency, memory_mb, power_budget)
        )
    
    def _calculate_carbon_optimized_allocation(
        self, agent_id: str, cpu_cores: int, memory_mb: int,
        bandwidth_mbps: float, duration: int, priority: int
    ) -> Optional[ResourceAllocation]:
        """Calculate allocation optimized for carbon footprint"""
        
        # Use the most energy-efficient configuration
        optimal_frequency = self._find_optimal_frequency(cpu_cores, duration)
        
        # Minimize cores and frequency for non-critical tasks
        if priority <= 3:
            cpu_cores = max(1, int(cpu_cores * 0.8))
            optimal_frequency = min(optimal_frequency, 1800)
        
        # Reduce memory allocation to minimize DRAM power
        memory_mb = max(512, int(memory_mb * 0.9))
        
        power_budget = self._calculate_power_consumption(cpu_cores, optimal_frequency, memory_mb)
        
        # Carbon efficiency score (lower power = better carbon efficiency)
        carbon_efficiency = 100.0 / (power_budget + 1.0)
        
        return ResourceAllocation(
            agent_id=agent_id,
            cpu_cores=cpu_cores,
            cpu_frequency_mhz=optimal_frequency,
            memory_mb=memory_mb,
            bandwidth_mbps=bandwidth_mbps,
            storage_io_priority=max(1, 6 - priority),
            power_budget_w=power_budget,
            thermal_budget_c=self._calculate_thermal_budget(cpu_cores, optimal_frequency),
            efficiency_score=carbon_efficiency
        )
    
    def _calculate_performance_first_allocation(
        self, agent_id: str, cpu_cores: int, memory_mb: int,
        bandwidth_mbps: float, duration: int, priority: int
    ) -> Optional[ResourceAllocation]:
        """Calculate allocation optimized for maximum performance"""
        
        # Use maximum available resources for high-priority tasks
        if priority >= 4:
            cpu_cores = min(self.constraints.max_cpu_cores, cpu_cores + 2)
            frequency = self.constraints.max_cpu_frequency_mhz
            memory_mb = int(memory_mb * 1.5)
        else:
            frequency = 2700
        
        power_budget = self._calculate_power_consumption(cpu_cores, frequency, memory_mb)
        
        return ResourceAllocation(
            agent_id=agent_id,
            cpu_cores=cpu_cores,
            cpu_frequency_mhz=frequency,
            memory_mb=memory_mb,
            bandwidth_mbps=bandwidth_mbps,
            storage_io_priority=priority + 2,
            power_budget_w=power_budget,
            thermal_budget_c=self._calculate_thermal_budget(cpu_cores, frequency),
            efficiency_score=self._calculate_efficiency_score(cpu_cores, frequency, memory_mb, power_budget)
        )
    
    def _find_optimal_frequency(self, cpu_cores: int, duration: int) -> int:
        """Find the most energy-efficient CPU frequency"""
        # Energy-optimal frequency considering performance and power trade-off
        # Generally, moderate frequencies provide best energy efficiency
        
        if duration < 300:  # Short tasks (< 5 minutes)
            return 2700  # High frequency for quick completion
        elif duration < 1800:  # Medium tasks (< 30 minutes)
            return 2200  # Balanced frequency
        else:  # Long tasks
            return 1800  # Lower frequency for sustained efficiency
    
    def _calculate_power_consumption(self, cpu_cores: int, frequency_mhz: int, memory_mb: int) -> float:
        """Calculate estimated power consumption"""
        # CPU power
        cpu_base_power = self._cpu_power_coeffs['base'] * cpu_cores
        cpu_freq_power = self._cpu_power_coeffs['freq'] * frequency_mhz * cpu_cores
        
        # Memory power
        memory_power = self._memory_power_coeff * memory_mb
        
        total_power = cpu_base_power + cpu_freq_power + memory_power
        return min(total_power, self.constraints.power_limit_watts)
    
    def _calculate_thermal_budget(self, cpu_cores: int, frequency_mhz: int) -> float:
        """Calculate thermal budget based on allocation"""
        # Higher cores and frequency = higher thermal load
        base_temp = 40.0
        core_temp_increase = cpu_cores * 2.0
        freq_temp_increase = (frequency_mhz - 1000) / 1000 * 10.0
        
        estimated_temp = base_temp + core_temp_increase + freq_temp_increase
        return min(estimated_temp, self.constraints.thermal_limit_celsius)
    
    def _calculate_efficiency_score(self, cpu_cores: int, frequency_mhz: int, memory_mb: int, power_w: float) -> float:
        """Calculate energy efficiency score (0-100)"""
        # Performance proxy
        performance_proxy = cpu_cores * frequency_mhz / 1000 * math.sqrt(memory_mb / 1024)
        
        # Efficiency = performance per watt
        efficiency = performance_proxy / max(power_w, 1.0)
        
        # Normalize to 0-100 scale
        return min(100.0, efficiency * 10.0)
    
    def _check_resource_availability(self, cpu_cores: int, memory_mb: int, bandwidth_mbps: float) -> bool:
        """Check if requested resources are available"""
        return (
            cpu_cores <= self._system_resources.available_cpu_cores and
            memory_mb <= self._system_resources.available_memory_mb and
            bandwidth_mbps <= self._system_resources.available_bandwidth_mbps
        )
    
    def _reserve_resources(self, allocation: ResourceAllocation) -> None:
        """Reserve system resources for an allocation"""
        self._system_resources.available_cpu_cores -= allocation.cpu_cores
        self._system_resources.available_memory_mb -= allocation.memory_mb
        self._system_resources.available_bandwidth_mbps -= allocation.bandwidth_mbps
    
    def _release_resources(self, allocation: ResourceAllocation) -> None:
        """Release system resources from an allocation"""
        self._system_resources.available_cpu_cores += allocation.cpu_cores
        self._system_resources.available_memory_mb += allocation.memory_mb
        self._system_resources.available_bandwidth_mbps += allocation.bandwidth_mbps
    
    def _update_system_resources(self) -> None:
        """Update current system resource information"""
        try:
            # Update memory info
            memory = psutil.virtual_memory()
            self._system_resources.available_memory_mb = int(memory.available / (1024 * 1024))
            
            # Update load average
            self._system_resources.load_average = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            
            # Update temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    cpu_temps = temps.get('cpu_thermal', temps.get('coretemp', []))
                    if cpu_temps:
                        self._system_resources.current_temperature_c = cpu_temps[0].current
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error updating system resources: {e}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for resource allocation"""
        while self._monitoring:
            try:
                # Update system resources
                self._update_system_resources()
                
                # Update allocation utilization
                self._update_allocation_utilization()
                
                # Optimize existing allocations if needed
                self._optimize_allocations()
                
                # Clean up old history
                self._cleanup_history()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in resource allocation monitoring: {e}")
                time.sleep(30)
    
    def _update_allocation_utilization(self) -> None:
        """Update utilization metrics for all allocations"""
        with self._lock:
            for allocation in self._allocations.values():
                try:
                    # In a real implementation, you would measure actual utilization
                    # For now, we'll simulate some utilization data
                    utilization = 50.0 + (hash(allocation.agent_id) % 50)  # Simulated 50-100% utilization
                    
                    allocation.utilization_history.append(utilization)
                    if len(allocation.utilization_history) > 100:
                        allocation.utilization_history = allocation.utilization_history[-50:]
                    
                    allocation.last_updated = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error updating utilization for {allocation.agent_id}: {e}")
    
    def _optimize_allocations(self) -> None:
        """Optimize existing allocations based on utilization patterns"""
        # This would implement dynamic reallocation based on observed utilization
        # For now, it's a placeholder for future optimization logic
        pass
    
    def _cleanup_history(self) -> None:
        """Clean up old allocation history"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        self._allocation_history = [
            event for event in self._allocation_history
            if datetime.fromisoformat(event['timestamp']) > cutoff_time
        ]
    
    def _record_allocation_event(self, allocation: ResourceAllocation, event_type: str) -> None:
        """Record an allocation event for analysis"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "agent_id": allocation.agent_id,
            "cpu_cores": allocation.cpu_cores,
            "cpu_frequency_mhz": allocation.cpu_frequency_mhz,
            "memory_mb": allocation.memory_mb,
            "bandwidth_mbps": allocation.bandwidth_mbps,
            "power_budget_w": allocation.power_budget_w,
            "efficiency_score": allocation.efficiency_score,
            "system_state": {
                "available_cpu_cores": self._system_resources.available_cpu_cores,
                "available_memory_mb": self._system_resources.available_memory_mb,
                "temperature_c": self._system_resources.current_temperature_c,
                "load_average": self._system_resources.load_average
            }
        }
        
        self._allocation_history.append(event)
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get resource allocation statistics"""
        with self._lock:
            allocations = list(self._allocations.values())
        
        if not allocations:
            return {
                "total_allocations": 0,
                "total_allocated_cores": 0,
                "total_allocated_memory_mb": 0,
                "total_power_budget_w": 0.0,
                "avg_efficiency_score": 0.0,
                "resource_utilization": {
                    "cpu_cores": 0.0,
                    "memory": 0.0,
                    "bandwidth": 0.0
                }
            }
        
        total_cores = sum(a.cpu_cores for a in allocations)
        total_memory = sum(a.memory_mb for a in allocations)
        total_bandwidth = sum(a.bandwidth_mbps for a in allocations)
        total_power = sum(a.power_budget_w for a in allocations)
        avg_efficiency = sum(a.efficiency_score for a in allocations) / len(allocations)
        
        return {
            "strategy": self.strategy.value,
            "total_allocations": len(allocations),
            "total_allocated_cores": total_cores,
            "total_allocated_memory_mb": total_memory,
            "total_allocated_bandwidth_mbps": total_bandwidth,
            "total_power_budget_w": total_power,
            "avg_efficiency_score": avg_efficiency,
            "resource_utilization": {
                "cpu_cores": total_cores / self._system_resources.total_cpu_cores,
                "memory": (self._system_resources.total_memory_mb - self._system_resources.available_memory_mb) / self._system_resources.total_memory_mb,
                "bandwidth": total_bandwidth / self._system_resources.total_bandwidth_mbps
            },
            "system_resources": {
                "total_cpu_cores": self._system_resources.total_cpu_cores,
                "available_cpu_cores": self._system_resources.available_cpu_cores,
                "total_memory_mb": self._system_resources.total_memory_mb,
                "available_memory_mb": self._system_resources.available_memory_mb,
                "current_temperature_c": self._system_resources.current_temperature_c,
                "load_average": self._system_resources.load_average
            },
            "monitoring_active": self._monitoring
        }
    
    def get_agent_allocation(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get resource allocation for a specific agent"""
        with self._lock:
            if agent_id not in self._allocations:
                return None
            
            allocation = self._allocations[agent_id]
        
        avg_utilization = 0.0
        if allocation.utilization_history:
            avg_utilization = sum(allocation.utilization_history) / len(allocation.utilization_history)
        
        return {
            "agent_id": allocation.agent_id,
            "cpu_cores": allocation.cpu_cores,
            "cpu_frequency_mhz": allocation.cpu_frequency_mhz,
            "memory_mb": allocation.memory_mb,
            "bandwidth_mbps": allocation.bandwidth_mbps,
            "storage_io_priority": allocation.storage_io_priority,
            "power_budget_w": allocation.power_budget_w,
            "thermal_budget_c": allocation.thermal_budget_c,
            "efficiency_score": allocation.efficiency_score,
            "avg_utilization": avg_utilization,
            "allocation_timestamp": allocation.allocation_timestamp.isoformat(),
            "last_updated": allocation.last_updated.isoformat()
        }
    
    def set_constraints(self, constraints: ResourceConstraints) -> None:
        """Update resource constraints"""
        self.constraints = constraints
        logger.info("Resource constraints updated")
    
    def export_allocation_data(self, filename: str) -> None:
        """Export allocation data to JSON file"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "strategy": self.strategy.value,
            "stats": self.get_allocation_stats(),
            "current_allocations": {
                agent_id: self.get_agent_allocation(agent_id)
                for agent_id in self._allocations.keys()
            },
            "allocation_history": self._allocation_history[-1000:],  # Last 1000 events
            "constraints": {
                "min_cpu_cores": self.constraints.min_cpu_cores,
                "max_cpu_cores": self.constraints.max_cpu_cores,
                "min_memory_mb": self.constraints.min_memory_mb,
                "max_memory_mb": self.constraints.max_memory_mb,
                "thermal_limit_celsius": self.constraints.thermal_limit_celsius,
                "power_limit_watts": self.constraints.power_limit_watts
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Allocation data exported to {filename}")

# Global resource allocator instance
_global_allocator: Optional[EnergyAwareResourceAllocator] = None

def get_global_allocator(strategy: AllocationStrategy = AllocationStrategy.ENERGY_PROPORTIONAL) -> EnergyAwareResourceAllocator:
    """Get or create global energy-aware resource allocator instance"""
    global _global_allocator
    if _global_allocator is None:
        _global_allocator = EnergyAwareResourceAllocator(strategy)
    return _global_allocator