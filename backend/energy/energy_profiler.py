"""
Energy Profiler - CPU-based energy consumption monitoring
Implements power estimation algorithms for Intel i7-12700H processor
"""

import os
import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import numpy as np

logger = logging.getLogger(__name__)

class CPUEnergyModel:
    """Energy model for Intel i7-12700H processor"""
    
    # Power consumption coefficients for i7-12700H
    BASE_POWER = 45.0  # Base TDP in watts
    MAX_POWER = 115.0  # Maximum turbo power in watts
    IDLE_POWER = 8.0   # Idle power consumption
    
    # Per-core power scaling factors
    PERFORMANCE_CORES = 6  # P-cores
    EFFICIENCY_CORES = 8   # E-cores (if available)
    
    # Frequency-power relationship coefficients
    FREQUENCY_POWER_ALPHA = 2.8  # Frequency scaling factor
    VOLTAGE_SCALING = 1.2        # Voltage scaling approximation
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.cpu_freq_base = 2700.0  # Base frequency in MHz
        
    def estimate_cpu_power(self, cpu_percent: float, frequency_mhz: float = None) -> float:
        """
        Estimate CPU power consumption based on utilization and frequency
        
        Args:
            cpu_percent: CPU utilization percentage (0-100)
            frequency_mhz: Current CPU frequency in MHz
            
        Returns:
            float: Estimated power consumption in watts
        """
        if frequency_mhz is None:
            frequency_mhz = self.cpu_freq_base
            
        # Normalize utilization
        utilization = min(cpu_percent / 100.0, 1.0)
        
        # Frequency scaling factor
        freq_ratio = frequency_mhz / self.cpu_freq_base
        freq_power_factor = freq_ratio ** self.FREQUENCY_POWER_ALPHA
        
        # Calculate dynamic power
        dynamic_power = (self.BASE_POWER - self.IDLE_POWER) * utilization * freq_power_factor
        
        # Total power = idle + dynamic power
        total_power = self.IDLE_POWER + dynamic_power
        
        return min(total_power, self.MAX_POWER)
    
    def estimate_memory_power(self, memory_percent: float) -> float:
        """
        Estimate memory subsystem power consumption
        
        Args:
            memory_percent: Memory utilization percentage
            
        Returns:
            float: Estimated memory power in watts
        """
        # DDR4/DDR5 memory power estimation
        BASE_MEMORY_POWER = 3.0  # Base memory power in watts
        MAX_MEMORY_POWER = 8.0   # Maximum memory power
        
        utilization = min(memory_percent / 100.0, 1.0)
        return BASE_MEMORY_POWER + (MAX_MEMORY_POWER - BASE_MEMORY_POWER) * utilization

@dataclass
class PowerMeasurement:
    """Single power measurement point"""
    timestamp: datetime
    cpu_power: float
    memory_power: float
    total_power: float
    cpu_utilization: float
    memory_utilization: float
    cpu_frequency: float
    process_count: int
    active_agents: int = 0

@dataclass
class EnergyMetrics:
    """Energy consumption metrics over time"""
    start_time: datetime
    end_time: datetime
    total_energy_wh: float  # Watt-hours
    avg_power_w: float      # Average watts
    peak_power_w: float     # Peak watts
    cpu_energy_wh: float
    memory_energy_wh: float
    co2_emission_g: float   # CO2 grams (using grid mix)
    measurements: List[PowerMeasurement] = field(default_factory=list)

class EnergyProfiler:
    """Main energy profiling system"""
    
    def __init__(self, measurement_interval: float = 1.0, grid_carbon_intensity: float = 0.4):
        """
        Initialize energy profiler
        
        Args:
            measurement_interval: Measurement interval in seconds
            grid_carbon_intensity: Grid carbon intensity in kg CO2/kWh
        """
        self.measurement_interval = measurement_interval
        self.grid_carbon_intensity = grid_carbon_intensity  # kg CO2/kWh
        self.energy_model = CPUEnergyModel()
        
        self._measurements: List[PowerMeasurement] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Agent tracking
        self._agent_processes: Dict[str, psutil.Process] = {}
        self._last_measurement_time = None
        
    def start_monitoring(self) -> None:
        """Start energy monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Energy monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop energy monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Energy monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring:
            try:
                measurement = self._take_measurement()
                with self._lock:
                    self._measurements.append(measurement)
                    
                # Keep only last 24 hours of measurements
                cutoff_time = datetime.now() - timedelta(hours=24)
                self._measurements = [m for m in self._measurements if m.timestamp > cutoff_time]
                
                time.sleep(self.measurement_interval)
                
            except Exception as e:
                logger.error(f"Error in energy monitoring: {e}")
                time.sleep(self.measurement_interval)
    
    def _take_measurement(self) -> PowerMeasurement:
        """Take a single power measurement"""
        timestamp = datetime.now()
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            frequency = cpu_freq.current if cpu_freq else self.energy_model.cpu_freq_base
        except:
            frequency = self.energy_model.cpu_freq_base
        
        # Process count
        process_count = len(psutil.pids())
        
        # Estimate power consumption
        cpu_power = self.energy_model.estimate_cpu_power(cpu_percent, frequency)
        memory_power = self.energy_model.estimate_memory_power(memory_percent)
        total_power = cpu_power + memory_power
        
        # Count active agents (processes with 'python' and 'agent' in name)
        active_agents = 0
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if 'agent' in cmdline.lower() or 'sutazai' in cmdline.lower():
                            active_agents += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.warning(f"Error counting agent processes: {e}")
        
        return PowerMeasurement(
            timestamp=timestamp,
            cpu_power=cpu_power,
            memory_power=memory_power,
            total_power=total_power,
            cpu_utilization=cpu_percent,
            memory_utilization=memory_percent,
            cpu_frequency=frequency,
            process_count=process_count,
            active_agents=active_agents
        )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current energy metrics"""
        if not self._measurements:
            return {}
        
        latest = self._measurements[-1]
        return {
            "timestamp": latest.timestamp.isoformat(),
            "current_power_w": latest.total_power,
            "cpu_power_w": latest.cpu_power,
            "memory_power_w": latest.memory_power,
            "cpu_utilization": latest.cpu_utilization,
            "memory_utilization": latest.memory_utilization,
            "cpu_frequency_mhz": latest.cpu_frequency,
            "active_agents": latest.active_agents,
            "process_count": latest.process_count
        }
    
    def calculate_energy_metrics(self, hours_back: float = 1.0) -> EnergyMetrics:
        """
        Calculate energy consumption metrics for the specified time period
        
        Args:
            hours_back: How many hours back to calculate metrics for
            
        Returns:
            EnergyMetrics: Calculated energy metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._lock:
            relevant_measurements = [m for m in self._measurements if m.timestamp > cutoff_time]
        
        if not relevant_measurements:
            # Return empty metrics
            now = datetime.now()
            return EnergyMetrics(
                start_time=now,
                end_time=now,
                total_energy_wh=0.0,
                avg_power_w=0.0,
                peak_power_w=0.0,
                cpu_energy_wh=0.0,
                memory_energy_wh=0.0,
                co2_emission_g=0.0,
                measurements=[]
            )
        
        start_time = relevant_measurements[0].timestamp
        end_time = relevant_measurements[-1].timestamp
        
        # Calculate energy consumption (trapezoidal integration)
        total_energy_wh = 0.0
        cpu_energy_wh = 0.0
        memory_energy_wh = 0.0
        
        for i in range(1, len(relevant_measurements)):
            prev_m = relevant_measurements[i-1]
            curr_m = relevant_measurements[i]
            
            time_delta = (curr_m.timestamp - prev_m.timestamp).total_seconds() / 3600  # hours
            
            # Trapezoidal integration
            avg_total_power = (prev_m.total_power + curr_m.total_power) / 2
            avg_cpu_power = (prev_m.cpu_power + curr_m.cpu_power) / 2
            avg_memory_power = (prev_m.memory_power + curr_m.memory_power) / 2
            
            total_energy_wh += avg_total_power * time_delta
            cpu_energy_wh += avg_cpu_power * time_delta
            memory_energy_wh += avg_memory_power * time_delta
        
        # Calculate other metrics
        power_values = [m.total_power for m in relevant_measurements]
        avg_power_w = sum(power_values) / len(power_values)
        peak_power_w = max(power_values)
        
        # Calculate CO2 emissions
        co2_emission_g = total_energy_wh * self.grid_carbon_intensity * 1000  # grams
        
        return EnergyMetrics(
            start_time=start_time,
            end_time=end_time,
            total_energy_wh=total_energy_wh,
            avg_power_w=avg_power_w,
            peak_power_w=peak_power_w,
            cpu_energy_wh=cpu_energy_wh,
            memory_energy_wh=memory_energy_wh,
            co2_emission_g=co2_emission_g,
            measurements=relevant_measurements
        )
    
    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate energy efficiency metrics"""
        if not self._measurements:
            return {}
        
        recent_measurements = self._measurements[-60:]  # Last 60 measurements
        if not recent_measurements:
            return {}
        
        # Calculate efficiency metrics
        total_power = sum(m.total_power for m in recent_measurements) / len(recent_measurements)
        cpu_utilization = sum(m.cpu_utilization for m in recent_measurements) / len(recent_measurements)
        active_agents = sum(m.active_agents for m in recent_measurements) / len(recent_measurements)
        
        # Power per unit of work
        power_per_cpu_percent = total_power / max(cpu_utilization, 0.1)  # Avoid division by zero
        power_per_agent = total_power / max(active_agents, 1)
        
        # Energy efficiency score (0-100, higher is better)
        # Based on power consumption relative to utilization
        utilization_efficiency = min(cpu_utilization / (total_power / 50) * 100, 100)
        
        return {
            "power_per_cpu_percent": power_per_cpu_percent,
            "power_per_agent": power_per_agent,
            "utilization_efficiency_score": utilization_efficiency,
            "average_power_w": total_power,
            "average_cpu_utilization": cpu_utilization,
            "average_active_agents": active_agents
        }
    
    def export_measurements(self, filename: str, hours_back: float = 24.0) -> None:
        """Export measurements to JSON file"""
        metrics = self.calculate_energy_metrics(hours_back)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics": {
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat(),
                "total_energy_wh": metrics.total_energy_wh,
                "avg_power_w": metrics.avg_power_w,
                "peak_power_w": metrics.peak_power_w,
                "cpu_energy_wh": metrics.cpu_energy_wh,
                "memory_energy_wh": metrics.memory_energy_wh,
                "co2_emission_g": metrics.co2_emission_g
            },
            "measurements": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_power": m.cpu_power,
                    "memory_power": m.memory_power,
                    "total_power": m.total_power,
                    "cpu_utilization": m.cpu_utilization,
                    "memory_utilization": m.memory_utilization,
                    "cpu_frequency": m.cpu_frequency,
                    "process_count": m.process_count,
                    "active_agents": m.active_agents
                }
                for m in metrics.measurements
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Energy measurements exported to {filename}")

# Global profiler instance
_global_profiler: Optional[EnergyProfiler] = None

def get_global_profiler() -> EnergyProfiler:
    """Get or create global energy profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = EnergyProfiler()
    return _global_profiler

def start_global_monitoring() -> None:
    """Start global energy monitoring"""
    profiler = get_global_profiler()
    profiler.start_monitoring()

def stop_global_monitoring() -> None:
    """Stop global energy monitoring"""
    global _global_profiler
    if _global_profiler:
        _global_profiler.stop_monitoring()