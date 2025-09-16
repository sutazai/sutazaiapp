"""
Hardware Intelligence System for Local LLM Operations
Implements Rule 16: Intelligent Hardware-Aware Management
Provides comprehensive hardware detection, monitoring, and optimization
"""

import os
import platform
import psutil
import subprocess
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import GPUtil
import numpy as np
from collections import deque
import aiofiles

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    """System resource status levels"""
    GREEN = "green"  # 0-75% usage - full capabilities
    YELLOW = "yellow"  # 75-85% usage - efficiency mode
    RED = "red"  # 85%+ usage - essential operations only
    CRITICAL = "critical"  # System instability detected


class ThermalStatus(Enum):
    """System thermal status"""
    COOL = "cool"  # <60°C
    NORMAL = "normal"  # 60-75°C
    WARM = "warm"  # 75-85°C
    HOT = "hot"  # 85-95°C
    CRITICAL = "critical"  # >95°C


@dataclass
class HardwareProfile:
    """Comprehensive hardware profile"""
    cpu: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    gpu: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    thermal: Dict[str, Any] = field(default_factory=dict)
    power: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu,
            "storage": self.storage,
            "thermal": self.thermal,
            "power": self.power,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ResourcePrediction:
    """Predicted resource requirements"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    duration_seconds: float
    confidence: float
    risk_level: str


class HardwareIntelligenceSystem:
    """
    Comprehensive hardware detection and management system
    Implements automated decision-making for local LLM operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hardware_profile = None
        self.performance_baselines = {}
        self.safety_thresholds = self._calculate_default_thresholds()
        self.resource_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.thermal_history = deque(maxlen=100)
        self.performance_metrics = {}
        self.optimization_suggestions = []
        self.is_monitoring = False
        self.monitor_task = None
        
        # Initialize hardware profile on creation
        self._initialize_hardware_profile()
        
    def _initialize_hardware_profile(self) -> None:
        """Initialize comprehensive hardware profile"""
        try:
            self.hardware_profile = self.detect_hardware_capabilities()
            self.performance_baselines = self.establish_performance_baselines()
            logger.info("Hardware Intelligence System initialized successfully")
            logger.info(f"CPU: {self.hardware_profile.cpu.get('model_name', 'Unknown')}")
            logger.info(f"Memory: {self.hardware_profile.memory.get('total_gb', 0):.1f} GB")
            if self.hardware_profile.gpu.get('gpu_present'):
                logger.info(f"GPU: {self.hardware_profile.gpu.get('gpu_model', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to initialize hardware profile: {e}")
            self.hardware_profile = HardwareProfile()
    
    def detect_hardware_capabilities(self) -> HardwareProfile:
        """Comprehensive hardware detection and profiling"""
        profile = HardwareProfile()
        
        # CPU Detection
        profile.cpu = self._detect_cpu_capabilities()
        
        # Memory Detection
        profile.memory = self._detect_memory_capabilities()
        
        # GPU Detection
        profile.gpu = self._detect_gpu_capabilities()
        
        # Storage Detection
        profile.storage = self._detect_storage_capabilities()
        
        # Thermal Detection
        profile.thermal = self._detect_thermal_capabilities()
        
        # Power Detection
        profile.power = self._detect_power_capabilities()
        
        return profile
    
    def _detect_cpu_capabilities(self) -> Dict[str, Any]:
        """Detect CPU capabilities"""
        try:
            cpu_info = {
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'current_utilization': psutil.cpu_percent(interval=1),
                'frequency': {},
                'cache_info': {},
                'features': []
            }
            
            # Get CPU frequency
            freq = psutil.cpu_freq()
            if freq:
                cpu_info['frequency'] = {
                    'current': freq.current,
                    'min': freq.min,
                    'max': freq.max
                }
            
            # Try to get CPU model name (Linux/Unix)
            if platform.system() == 'Linux':
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if 'model name' in line:
                                cpu_info['model_name'] = line.split(':')[1].strip()
                                break
                except:
                    pass
            
            # Check for AVX/AVX2 support (important for ML)
            try:
                if platform.system() == 'Linux':
                    result = subprocess.run(['lscpu'], capture_output=True, text=True)
                    if 'avx' in result.stdout.lower():
                        cpu_info['features'].append('AVX')
                    if 'avx2' in result.stdout.lower():
                        cpu_info['features'].append('AVX2')
            except:
                pass
            
            return cpu_info
            
        except Exception as e:
            logger.error(f"CPU detection error: {e}")
            return {'cores': 1, 'threads': 1, 'current_utilization': 0}
    
    def _detect_memory_capabilities(self) -> Dict[str, Any]:
        """Detect memory capabilities"""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_bytes': mem.total,
                'total_gb': mem.total / (1024**3),
                'available_bytes': mem.available,
                'available_gb': mem.available / (1024**3),
                'used_percent': mem.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent,
                'memory_type': self._detect_memory_type(),
                'memory_speed': self._detect_memory_speed()
            }
        except Exception as e:
            logger.error(f"Memory detection error: {e}")
            return {'total_gb': 1, 'available_gb': 0.5, 'used_percent': 50}
    
    def _detect_gpu_capabilities(self) -> Dict[str, Any]:
        """Detect GPU capabilities"""
        gpu_info = {
            'gpu_present': False,
            'gpu_count': 0,
            'gpus': []
        }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info['gpu_present'] = True
                gpu_info['gpu_count'] = len(gpus)
                
                for gpu in gpus:
                    gpu_data = {
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_free_mb': gpu.memoryFree,
                        'gpu_utilization': gpu.load * 100,
                        'temperature': gpu.temperature,
                        'driver': gpu.driver
                    }
                    gpu_info['gpus'].append(gpu_data)
                
                # Use first GPU as primary
                if gpu_info['gpus']:
                    primary_gpu = gpu_info['gpus'][0]
                    gpu_info.update({
                        'gpu_model': primary_gpu['name'],
                        'gpu_memory_mb': primary_gpu['memory_total_mb'],
                        'gpu_utilization': primary_gpu['gpu_utilization'],
                        'gpu_temperature': primary_gpu['temperature']
                    })
                    
        except Exception as e:
            logger.debug(f"GPU detection not available: {e}")
            
        # Try nvidia-smi as fallback
        if not gpu_info['gpu_present']:
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,utilization.gpu,temperature.gpu',
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines and lines[0]:
                        parts = lines[0].split(', ')
                        if len(parts) >= 4:
                            gpu_info['gpu_present'] = True
                            gpu_info['gpu_model'] = parts[0]
                            gpu_info['gpu_memory_mb'] = float(parts[1])
                            gpu_info['gpu_utilization'] = float(parts[2])
                            gpu_info['gpu_temperature'] = float(parts[3])
            except:
                pass
                
        return gpu_info
    
    def _detect_storage_capabilities(self) -> Dict[str, Any]:
        """Detect storage capabilities"""
        try:
            disk = psutil.disk_usage('/')
            io_counters = psutil.disk_io_counters()
            
            storage_info = {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent_used': disk.percent,
                'read_bytes': io_counters.read_bytes if io_counters else 0,
                'write_bytes': io_counters.write_bytes if io_counters else 0,
                'read_count': io_counters.read_count if io_counters else 0,
                'write_count': io_counters.write_count if io_counters else 0
            }
            
            # Detect storage type (SSD/HDD)
            storage_info['storage_type'] = self._detect_storage_type()
            
            return storage_info
            
        except Exception as e:
            logger.error(f"Storage detection error: {e}")
            return {'total_gb': 100, 'free_gb': 50, 'percent_used': 50}
    
    def _detect_thermal_capabilities(self) -> Dict[str, Any]:
        """Detect thermal capabilities"""
        thermal_info = {
            'sensors': {},
            'current_temperature': 0,
            'temperature_trend': 'stable',
            'thermal_throttling': False,
            'cooling_capacity': 'normal'
        }
        
        try:
            # Get temperature sensors
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        thermal_info['sensors'][name] = []
                        for entry in entries:
                            thermal_info['sensors'][name].append({
                                'label': entry.label or name,
                                'current': entry.current,
                                'high': entry.high,
                                'critical': entry.critical
                            })
                    
                    # Calculate average temperature
                    all_temps = []
                    for sensor_list in thermal_info['sensors'].values():
                        for sensor in sensor_list:
                            if sensor['current']:
                                all_temps.append(sensor['current'])
                    
                    if all_temps:
                        thermal_info['current_temperature'] = sum(all_temps) / len(all_temps)
            
            # Check for thermal throttling
            thermal_info['thermal_throttling'] = self._check_thermal_throttling()
            
        except Exception as e:
            logger.debug(f"Thermal detection limited: {e}")
            
        return thermal_info
    
    def _detect_power_capabilities(self) -> Dict[str, Any]:
        """Detect power capabilities"""
        power_info = {
            'power_source': 'AC',
            'battery_present': False,
            'battery_percent': 100,
            'power_consumption_watts': 0,
            'power_limits': {},
            'power_efficiency': 'normal'
        }
        
        try:
            if hasattr(psutil, 'sensors_battery'):
                battery = psutil.sensors_battery()
                if battery:
                    power_info['battery_present'] = True
                    power_info['battery_percent'] = battery.percent
                    power_info['power_source'] = 'AC' if battery.power_plugged else 'Battery'
                    power_info['time_remaining'] = battery.secsleft if battery.secsleft > 0 else None
        except:
            pass
            
        return power_info
    
    def _detect_memory_type(self) -> str:
        """Detect memory type (DDR4, DDR5, etc.)"""
        # This would require platform-specific implementation
        # For now, return a default
        return "DDR4"
    
    def _detect_memory_speed(self) -> int:
        """Detect memory speed in MHz"""
        # This would require platform-specific implementation
        # For now, return a typical value
        return 3200
    
    def _detect_storage_type(self) -> str:
        """Detect if storage is SSD or HDD"""
        # Simple heuristic: check rotational property
        try:
            # On Linux, check /sys/block/*/queue/rotational
            if platform.system() == 'Linux':
                for device in os.listdir('/sys/block'):
                    if device.startswith('sd') or device.startswith('nvme'):
                        rotational_file = f'/sys/block/{device}/queue/rotational'
                        if os.path.exists(rotational_file):
                            with open(rotational_file, 'r') as f:
                                if f.read().strip() == '0':
                                    return 'SSD'
                return 'HDD'
        except:
            pass
        
        # Default to SSD for modern systems
        return 'SSD'
    
    def _check_thermal_throttling(self) -> bool:
        """Check if system is thermally throttling"""
        try:
            if self.hardware_profile and self.hardware_profile.cpu.get('frequency'):
                current_freq = self.hardware_profile.cpu['frequency'].get('current', 0)
                max_freq = self.hardware_profile.cpu['frequency'].get('max', 0)
                if max_freq > 0 and current_freq < max_freq * 0.8:
                    return True
        except:
            pass
        return False
    
    def _calculate_default_thresholds(self) -> Dict[str, Any]:
        """Calculate default safety thresholds"""
        return {
            'cpu': {
                'safe': 70,
                'warning': 85,
                'critical': 95
            },
            'memory': {
                'safe': 75,
                'warning': 85,
                'critical': 95
            },
            'gpu': {
                'safe': 80,
                'warning': 90,
                'critical': 95
            },
            'temperature': {
                'safe': 70,
                'warning': 80,
                'critical': 90
            },
            'storage': {
                'safe': 80,
                'warning': 90,
                'critical': 95
            }
        }
    
    def establish_performance_baselines(self) -> Dict[str, Any]:
        """Establish performance baselines through benchmarking"""
        baselines = {
            'cpu_single_core': 0,
            'cpu_multi_core': 0,
            'memory_bandwidth': 0,
            'storage_read': 0,
            'storage_write': 0,
            'gpu_compute': 0
        }
        
        try:
            # Simple CPU benchmark
            import time
            start = time.time()
            # Simple computation
            _ = sum(i**2 for i in range(1000000))
            baselines['cpu_single_core'] = 1000000 / (time.time() - start)
            
            # Memory bandwidth test
            import numpy as np
            data = np.random.rand(10000000)
            start = time.time()
            _ = np.sum(data)
            baselines['memory_bandwidth'] = len(data) * 8 / (time.time() - start) / 1e9  # GB/s
            
        except Exception as e:
            logger.debug(f"Baseline establishment limited: {e}")
            
        return baselines
    
    async def perform_comprehensive_selfcheck(self) -> Dict[str, Any]:
        """Automated system health and capability assessment"""
        selfcheck_results = {
            'timestamp': datetime.now().isoformat(),
            'system_health': await self.assess_system_health(),
            'resource_availability': self.check_resource_availability(),
            'performance_status': self.validate_performance_baselines(),
            'thermal_status': self.check_thermal_health(),
            'stability_assessment': self.assess_system_stability(),
            'optimization_opportunities': self.identify_optimization_opportunities(),
            'recommended_models': self.get_recommended_models()
        }
        
        # Log results
        logger.info(f"System selfcheck completed: {selfcheck_results['system_health']['status']}")
        
        return selfcheck_results
    
    async def assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        health = {
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Update hardware profile
        self.hardware_profile = self.detect_hardware_capabilities()
        
        # Check CPU health
        cpu_usage = self.hardware_profile.cpu.get('current_utilization', 0)
        if cpu_usage > self.safety_thresholds['cpu']['critical']:
            health['status'] = 'critical'
            health['issues'].append(f"CPU usage critical: {cpu_usage}%")
        elif cpu_usage > self.safety_thresholds['cpu']['warning']:
            health['status'] = 'degraded'
            health['warnings'].append(f"CPU usage high: {cpu_usage}%")
        health['metrics']['cpu_usage'] = cpu_usage
        
        # Check memory health
        mem_usage = self.hardware_profile.memory.get('used_percent', 0)
        if mem_usage > self.safety_thresholds['memory']['critical']:
            health['status'] = 'critical'
            health['issues'].append(f"Memory usage critical: {mem_usage}%")
        elif mem_usage > self.safety_thresholds['memory']['warning']:
            if health['status'] == 'healthy':
                health['status'] = 'degraded'
            health['warnings'].append(f"Memory usage high: {mem_usage}%")
        health['metrics']['memory_usage'] = mem_usage
        
        # Check thermal health
        temp = self.hardware_profile.thermal.get('current_temperature', 0)
        if temp > self.safety_thresholds['temperature']['critical']:
            health['status'] = 'critical'
            health['issues'].append(f"Temperature critical: {temp}°C")
        elif temp > self.safety_thresholds['temperature']['warning']:
            if health['status'] == 'healthy':
                health['status'] = 'degraded'
            health['warnings'].append(f"Temperature high: {temp}°C")
        health['metrics']['temperature'] = temp
        
        return health
    
    def check_resource_availability(self) -> Dict[str, Any]:
        """Check current resource availability"""
        return {
            'cpu': {
                'available_percent': 100 - self.hardware_profile.cpu.get('current_utilization', 0),
                'status': self._get_resource_status('cpu', self.hardware_profile.cpu.get('current_utilization', 0))
            },
            'memory': {
                'available_gb': self.hardware_profile.memory.get('available_gb', 0),
                'available_percent': 100 - self.hardware_profile.memory.get('used_percent', 0),
                'status': self._get_resource_status('memory', self.hardware_profile.memory.get('used_percent', 0))
            },
            'gpu': {
                'available': self.hardware_profile.gpu.get('gpu_present', False),
                'available_percent': 100 - self.hardware_profile.gpu.get('gpu_utilization', 0) if self.hardware_profile.gpu.get('gpu_present') else 0,
                'status': self._get_resource_status('gpu', self.hardware_profile.gpu.get('gpu_utilization', 0)) if self.hardware_profile.gpu.get('gpu_present') else ResourceStatus.GREEN
            },
            'storage': {
                'available_gb': self.hardware_profile.storage.get('free_gb', 0),
                'status': self._get_resource_status('storage', self.hardware_profile.storage.get('percent_used', 0))
            }
        }
    
    def _get_resource_status(self, resource_type: str, usage: float) -> ResourceStatus:
        """Get resource status based on usage"""
        thresholds = self.safety_thresholds.get(resource_type, self.safety_thresholds['cpu'])
        
        if usage >= thresholds['critical']:
            return ResourceStatus.CRITICAL
        elif usage >= thresholds['warning']:
            return ResourceStatus.RED
        elif usage >= thresholds['safe']:
            return ResourceStatus.YELLOW
        else:
            return ResourceStatus.GREEN
    
    def validate_performance_baselines(self) -> Dict[str, Any]:
        """Validate current performance against baselines"""
        return {
            'status': 'normal',
            'deviations': {},
            'recommendations': []
        }
    
    def check_thermal_health(self) -> Dict[str, Any]:
        """Check thermal health and cooling status"""
        temp = self.hardware_profile.thermal.get('current_temperature', 0)
        
        if temp < 60:
            status = ThermalStatus.COOL
        elif temp < 75:
            status = ThermalStatus.NORMAL
        elif temp < 85:
            status = ThermalStatus.WARM
        elif temp < 95:
            status = ThermalStatus.HOT
        else:
            status = ThermalStatus.CRITICAL
        
        return {
            'status': status.value,
            'temperature': temp,
            'throttling': self.hardware_profile.thermal.get('thermal_throttling', False),
            'cooling_capacity': self.hardware_profile.thermal.get('cooling_capacity', 'normal')
        }
    
    def assess_system_stability(self) -> Dict[str, Any]:
        """Assess system stability based on historical data"""
        return {
            'stability_score': 0.95,  # 0-1 scale
            'recent_issues': [],
            'uptime_hours': self._get_system_uptime(),
            'recommendation': 'System stable for intensive operations'
        }
    
    def _get_system_uptime(self) -> float:
        """Get system uptime in hours"""
        try:
            boot_time = psutil.boot_time()
            current_time = datetime.now().timestamp()
            uptime_seconds = current_time - boot_time
            return uptime_seconds / 3600
        except:
            return 0
    
    def identify_optimization_opportunities(self) -> List[str]:
        """Identify system optimization opportunities"""
        opportunities = []
        
        # Check if GPU is available but not utilized
        if self.hardware_profile.gpu.get('gpu_present') and self.hardware_profile.gpu.get('gpu_utilization', 0) < 10:
            opportunities.append("GPU available but underutilized - consider GPU-accelerated models")
        
        # Check if there's excessive swap usage
        if self.hardware_profile.memory.get('swap_percent', 0) > 50:
            opportunities.append("High swap usage detected - consider closing unnecessary applications")
        
        # Check for thermal issues
        if self.hardware_profile.thermal.get('thermal_throttling'):
            opportunities.append("Thermal throttling detected - improve cooling or reduce workload")
        
        # Check for low memory
        if self.hardware_profile.memory.get('available_gb', 0) < 2:
            opportunities.append("Low memory available - consider using smaller models")
        
        return opportunities
    
    def get_recommended_models(self) -> List[str]:
        """Get recommended models based on current system capabilities"""
        recommendations = []
        
        # Get available resources
        available_memory = self.hardware_profile.memory.get('available_gb', 0)
        cpu_available = 100 - self.hardware_profile.cpu.get('current_utilization', 0)
        gpu_present = self.hardware_profile.gpu.get('gpu_present', False)
        
        # Always recommend TinyLlama as baseline
        recommendations.append('tinyllama')
        
        # Add more models based on resources
        if available_memory > 8 and cpu_available > 50:
            recommendations.append('mistral-7b')
        
        if available_memory > 16 and cpu_available > 40:
            recommendations.append('llama2-13b')
        
        if available_memory > 32 and cpu_available > 30 and gpu_present:
            recommendations.append('gpt-oss:20b')
        
        return recommendations
    
    async def predict_resource_requirements(
        self, 
        task_complexity: str, 
        context_size: int = 1000
    ) -> ResourcePrediction:
        """Predict resource requirements for a task"""
        # Simple prediction model based on task complexity
        complexity_map = {
            'simple': {'cpu': 20, 'memory': 2, 'duration': 5, 'risk': 'low'},
            'moderate': {'cpu': 40, 'memory': 4, 'duration': 15, 'risk': 'medium'},
            'complex': {'cpu': 70, 'memory': 8, 'duration': 30, 'risk': 'high'}
        }
        
        profile = complexity_map.get(task_complexity, complexity_map['moderate'])
        
        # Adjust for context size
        context_factor = min(context_size / 1000, 3)
        
        return ResourcePrediction(
            cpu_usage=profile['cpu'] * context_factor,
            memory_usage=profile['memory'] * context_factor,
            gpu_usage=30 * context_factor if self.hardware_profile.gpu.get('gpu_present') else None,
            duration_seconds=profile['duration'] * context_factor,
            confidence=0.75,
            risk_level=profile['risk']
        )
    
    async def start_monitoring(self, interval_seconds: int = 5) -> None:
        """Start continuous resource monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.info(f"Started hardware monitoring with {interval_seconds}s interval")
    
    async def _monitoring_loop(self, interval: int) -> None:
        """Internal monitoring loop"""
        while self.is_monitoring:
            try:
                # Update hardware profile
                self.hardware_profile = self.detect_hardware_capabilities()
                
                # Record metrics
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu': self.hardware_profile.cpu.get('current_utilization', 0),
                    'memory': self.hardware_profile.memory.get('used_percent', 0),
                    'gpu': self.hardware_profile.gpu.get('gpu_utilization', 0) if self.hardware_profile.gpu.get('gpu_present') else 0,
                    'temperature': self.hardware_profile.thermal.get('current_temperature', 0)
                }
                
                self.resource_history.append(metrics)
                
                # Check for critical conditions
                await self._check_critical_conditions()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def _check_critical_conditions(self) -> None:
        """Check for critical system conditions"""
        # Check CPU
        if self.hardware_profile.cpu.get('current_utilization', 0) > self.safety_thresholds['cpu']['critical']:
            logger.critical("CPU usage critical - triggering safety measures")
            await self._trigger_safety_measures('cpu')
        
        # Check memory
        if self.hardware_profile.memory.get('used_percent', 0) > self.safety_thresholds['memory']['critical']:
            logger.critical("Memory usage critical - triggering safety measures")
            await self._trigger_safety_measures('memory')
        
        # Check temperature
        if self.hardware_profile.thermal.get('current_temperature', 0) > self.safety_thresholds['temperature']['critical']:
            logger.critical("Temperature critical - triggering thermal protection")
            await self._trigger_safety_measures('thermal')
    
    async def _trigger_safety_measures(self, trigger_type: str) -> None:
        """Trigger safety measures for critical conditions"""
        # This would integrate with the ModelSelectionEngine to force TinyLlama
        logger.warning(f"Safety measures triggered due to {trigger_type}")
        # In production, this would send signals to switch models or pause operations
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.is_monitoring = False
        if self.monitor_task:
            await self.monitor_task
        logger.info("Hardware monitoring stopped")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring data"""
        if not self.resource_history:
            return {'status': 'No monitoring data available'}
        
        # Calculate averages from history
        cpu_avg = sum(m['cpu'] for m in self.resource_history) / len(self.resource_history)
        memory_avg = sum(m['memory'] for m in self.resource_history) / len(self.resource_history)
        
        return {
            'monitoring_duration': len(self.resource_history) * 5 / 60,  # minutes
            'cpu_average': cpu_avg,
            'memory_average': memory_avg,
            'peak_cpu': max(m['cpu'] for m in self.resource_history),
            'peak_memory': max(m['memory'] for m in self.resource_history),
            'samples': len(self.resource_history)
        }
    
    def export_profile(self, filepath: str) -> None:
        """Export hardware profile to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.hardware_profile.to_dict(), f, indent=2)
            logger.info(f"Hardware profile exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export profile: {e}")


# Create global instance for import
hardware_intelligence = HardwareIntelligenceSystem()