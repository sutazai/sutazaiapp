"""
Power Optimizer - Dynamic power management and optimization strategies
"""

import os
import time
import threading
import logging
import psutil
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import subprocess

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Power optimization strategies"""
    AGGRESSIVE = "aggressive"      # Maximum power savings
    BALANCED = "balanced"          # Balance performance and power
    CONSERVATIVE = "conservative"  # Minimal impact on performance
    CUSTOM = "custom"             # User-defined strategy

class PowerSavingAction(Enum):
    """Available power saving actions"""
    CPU_FREQUENCY_SCALING = "cpu_frequency_scaling"
    PROCESS_NICE_ADJUSTMENT = "process_nice_adjustment"
    IDLE_PROCESS_SUSPENSION = "idle_process_suspension"
    MEMORY_COMPRESSION = "memory_compression"
    CORE_ISOLATION = "core_isolation"
    THERMAL_THROTTLING = "thermal_throttling"

@dataclass
class OptimizationRule:
    """Power optimization rule"""
    name: str
    condition: Callable[[], bool]  # Function that returns True if rule should be applied
    action: PowerSavingAction
    parameters: Dict[str, Any]
    priority: int = 1  # Higher number = higher priority
    enabled: bool = True
    cooldown_seconds: int = 60  # Minimum time between applications
    last_applied: Optional[datetime] = None

@dataclass
class OptimizationResult:
    """Result of an optimization action"""
    action: PowerSavingAction
    success: bool
    power_saved_w: float
    performance_impact: float  # 0.0 to 1.0, where 1.0 is severe impact
    details: str
    timestamp: datetime = field(default_factory=datetime.now)

class CPUFrequencyController:
    """Controls CPU frequency scaling for power optimization"""
    
    def __init__(self):
        self.available_governors = self._get_available_governors()
        self.current_governor = self._get_current_governor()
        self.available_frequencies = self._get_available_frequencies()
        
    def _get_available_governors(self) -> List[str]:
        """Get available CPU governors"""
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors', 'r') as f:
                return f.read().strip().split()
        except (FileNotFoundError, PermissionError):
            logger.warning("CPU frequency governors not accessible")
            return []
    
    def _get_current_governor(self) -> Optional[str]:
        """Get current CPU governor"""
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            return None
    
    def _get_available_frequencies(self) -> List[int]:
        """Get available CPU frequencies"""
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies', 'r') as f:
                return [int(freq) for freq in f.read().strip().split()]
        except (FileNotFoundError, PermissionError):
            return []
    
    def set_governor(self, governor: str) -> bool:
        """Set CPU governor for all cores"""
        if governor not in self.available_governors:
            logger.error(f"Governor '{governor}' not available")
            return False
        
        try:
            cpu_count = psutil.cpu_count()
            for cpu in range(cpu_count):
                gov_path = f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor'
                with open(gov_path, 'w') as f:
                    f.write(governor)
            
            self.current_governor = governor
            logger.info(f"CPU governor set to '{governor}'")
            return True
            
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to set CPU governor: {e}")
            return False
    
    def set_frequency_limits(self, min_freq: int, max_freq: int) -> bool:
        """Set CPU frequency limits"""
        try:
            cpu_count = psutil.cpu_count()
            for cpu in range(cpu_count):
                min_path = f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_min_freq'
                max_path = f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq'
                
                with open(min_path, 'w') as f:
                    f.write(str(min_freq))
                with open(max_path, 'w') as f:
                    f.write(str(max_freq))
            
            logger.info(f"CPU frequency limits set: {min_freq}-{max_freq} Hz")
            return True
            
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to set frequency limits: {e}")
            return False

class ProcessManager:
    """Manages process priorities and states for power optimization"""
    
    def __init__(self):
        self._suspended_processes: Dict[int, Dict[str, Any]] = {}
    
    def adjust_process_priority(self, pid: int, nice_value: int) -> bool:
        """Adjust process priority (nice value)"""
        try:
            process = psutil.Process(pid)
            process.nice(nice_value)
            logger.debug(f"Process {pid} nice value set to {nice_value}")
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            logger.warning(f"Failed to adjust process {pid} priority: {e}")
            return False
    
    def suspend_idle_processes(self, idle_threshold_minutes: int = 30) -> List[int]:
        """Suspend processes that have been idle for specified time"""
        suspended_pids = []
        current_time = datetime.now()
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'create_time']):
                try:
                    # Skip system processes
                    if proc.info['pid'] < 1000:
                        continue
                    
                    # Check if process has been idle
                    create_time = datetime.fromtimestamp(proc.info['create_time'])
                    if (current_time - create_time).total_seconds() < idle_threshold_minutes * 60:
                        continue
                    
                    # Get recent CPU usage
                    cpu_percent = proc.cpu_percent(interval=1.0)
                    if cpu_percent < 0.1:  # Less than 0.1% CPU usage
                        process = psutil.Process(proc.info['pid'])
                        
                        # Store process info before suspending
                        self._suspended_processes[proc.info['pid']] = {
                            'name': proc.info['name'],
                            'suspend_time': current_time,
                            'original_nice': process.nice()
                        }
                        
                        # Suspend process
                        process.suspend()
                        suspended_pids.append(proc.info['pid'])
                        logger.debug(f"Suspended idle process {proc.info['pid']} ({proc.info['name']})")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            logger.error(f"Error suspending idle processes: {e}")
        
        return suspended_pids
    
    def resume_suspended_processes(self, max_suspended_time_minutes: int = 60) -> List[int]:
        """Resume processes that have been suspended too long"""
        resumed_pids = []
        current_time = datetime.now()
        
        for pid, info in list(self._suspended_processes.items()):
            try:
                suspend_duration = (current_time - info['suspend_time']).total_seconds() / 60
                if suspend_duration > max_suspended_time_minutes:
                    process = psutil.Process(pid)
                    process.resume()
                    resumed_pids.append(pid)
                    del self._suspended_processes[pid]
                    logger.debug(f"Resumed process {pid} ({info['name']})")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process no longer exists
                if pid in self._suspended_processes:
                    del self._suspended_processes[pid]
        
        return resumed_pids

class PowerOptimizer:
    """Main power optimization system"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.cpu_controller = CPUFrequencyController()
        self.process_manager = ProcessManager()
        
        self._optimization_rules: List[OptimizationRule] = []
        self._optimization_history: List[OptimizationResult] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default optimization rules based on strategy"""
        
        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            # Aggressive power saving rules
            self._optimization_rules = [
                OptimizationRule(
                    name="Low CPU Utilization Power Scaling",
                    condition=lambda: self._get_cpu_utilization() < 10,
                    action=PowerSavingAction.CPU_FREQUENCY_SCALING,
                    parameters={"governor": "powersave", "max_freq_ratio": 0.5},
                    priority=3,
                    cooldown_seconds=30
                ),
                OptimizationRule(
                    name="Suspend Very Idle Processes", 
                    condition=lambda: self._get_cpu_utilization() < 5,
                    action=PowerSavingAction.IDLE_PROCESS_SUSPENSION,
                    parameters={"idle_threshold_minutes": 10},
                    priority=2,
                    cooldown_seconds=120
                ),
                OptimizationRule(
                    name="Aggressive Process Priority",
                    condition=lambda: self._get_memory_utilization() > 70,
                    action=PowerSavingAction.PROCESS_NICE_ADJUSTMENT,
                    parameters={"nice_increment": 5},
                    priority=1,
                    cooldown_seconds=60
                )
            ]
            
        elif self.strategy == OptimizationStrategy.BALANCED:
            # Balanced optimization rules
            self._optimization_rules = [
                OptimizationRule(
                    name="Moderate CPU Scaling",
                    condition=lambda: self._get_cpu_utilization() < 20,
                    action=PowerSavingAction.CPU_FREQUENCY_SCALING,
                    parameters={"governor": "ondemand", "max_freq_ratio": 0.75},
                    priority=2,
                    cooldown_seconds=60
                ),
                OptimizationRule(
                    name="Suspend Long Idle Processes",
                    condition=lambda: self._get_cpu_utilization() < 10,
                    action=PowerSavingAction.IDLE_PROCESS_SUSPENSION,
                    parameters={"idle_threshold_minutes": 30},
                    priority=1,
                    cooldown_seconds=300
                )
            ]
            
        elif self.strategy == OptimizationStrategy.CONSERVATIVE:
            # Conservative optimization rules
            self._optimization_rules = [
                OptimizationRule(
                    name="Conservative CPU Scaling",
                    condition=lambda: self._get_cpu_utilization() < 5,
                    action=PowerSavingAction.CPU_FREQUENCY_SCALING,
                    parameters={"governor": "conservative"},
                    priority=1,
                    cooldown_seconds=120
                )
            ]
    
    def start_optimization(self) -> None:
        """Start automated power optimization"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Power optimization started with {self.strategy.value} strategy")
    
    def stop_optimization(self) -> None:
        """Stop automated power optimization"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Power optimization stopped")
    
    def _optimization_loop(self) -> None:
        """Main optimization loop"""
        while self._monitoring:
            try:
                self._apply_optimization_rules()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(60)
    
    def _apply_optimization_rules(self) -> None:
        """Apply applicable optimization rules"""
        current_time = datetime.now()
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            [r for r in self._optimization_rules if r.enabled],
            key=lambda x: x.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            try:
                # Check cooldown
                if (rule.last_applied and 
                    (current_time - rule.last_applied).total_seconds() < rule.cooldown_seconds):
                    continue
                
                # Check condition
                if rule.condition():
                    result = self._apply_optimization_action(rule)
                    if result.success:
                        rule.last_applied = current_time
                        
                        with self._lock:
                            self._optimization_history.append(result)
                            # Keep only last 100 results
                            if len(self._optimization_history) > 100:
                                self._optimization_history = self._optimization_history[-100:]
                                
                        logger.info(f"Applied optimization: {rule.name} - {result.details}")
                    
            except Exception as e:
                logger.error(f"Error applying rule '{rule.name}': {e}")
    
    def _apply_optimization_action(self, rule: OptimizationRule) -> OptimizationResult:
        """Apply a specific optimization action"""
        action = rule.action
        params = rule.parameters
        
        if action == PowerSavingAction.CPU_FREQUENCY_SCALING:
            return self._apply_cpu_frequency_scaling(params)
        elif action == PowerSavingAction.IDLE_PROCESS_SUSPENSION:
            return self._apply_idle_process_suspension(params)
        elif action == PowerSavingAction.PROCESS_NICE_ADJUSTMENT:
            return self._apply_process_nice_adjustment(params)
        else:
            return OptimizationResult(
                action=action,
                success=False,
                power_saved_w=0.0,
                performance_impact=0.0,
                details=f"Action {action.value} not implemented"
            )
    
    def _apply_cpu_frequency_scaling(self, params: Dict[str, Any]) -> OptimizationResult:
        """Apply CPU frequency scaling optimization"""
        governor = params.get("governor", "ondemand")
        max_freq_ratio = params.get("max_freq_ratio", 1.0)
        
        success = self.cpu_controller.set_governor(governor)
        
        # Estimate power savings (rough approximation)
        power_saved = 0.0
        if success:
            if governor == "powersave":
                power_saved = 15.0  # Estimated 15W savings
            elif governor == "ondemand":
                power_saved = 8.0   # Estimated 8W savings
            elif governor == "conservative":
                power_saved = 5.0   # Estimated 5W savings
        
        performance_impact = 0.0
        if governor == "powersave":
            performance_impact = 0.3  # 30% performance impact
        elif governor == "conservative":
            performance_impact = 0.1  # 10% performance impact
        
        return OptimizationResult(
            action=PowerSavingAction.CPU_FREQUENCY_SCALING,
            success=success,
            power_saved_w=power_saved,
            performance_impact=performance_impact,
            details=f"Set CPU governor to {governor}"
        )
    
    def _apply_idle_process_suspension(self, params: Dict[str, Any]) -> OptimizationResult:
        """Apply idle process suspension optimization"""
        idle_threshold = params.get("idle_threshold_minutes", 30)
        
        suspended_pids = self.process_manager.suspend_idle_processes(idle_threshold)
        
        # Estimate power savings (roughly 0.5W per suspended process)
        power_saved = len(suspended_pids) * 0.5
        
        return OptimizationResult(
            action=PowerSavingAction.IDLE_PROCESS_SUSPENSION,
            success=len(suspended_pids) > 0,
            power_saved_w=power_saved,
            performance_impact=0.0,  # No performance impact for truly idle processes
            details=f"Suspended {len(suspended_pids)} idle processes"
        )
    
    def _apply_process_nice_adjustment(self, params: Dict[str, Any]) -> OptimizationResult:
        """Apply process nice value adjustment"""
        nice_increment = params.get("nice_increment", 2)
        
        adjusted_count = 0
        try:
            for proc in psutil.process_iter(['pid', 'name', 'nice']):
                try:
                    if proc.info['pid'] > 1000:  # Skip system processes
                        current_nice = proc.info['nice']
                        if current_nice < 10:  # Don't over-nice processes
                            new_nice = min(current_nice + nice_increment, 19)
                            if self.process_manager.adjust_process_priority(proc.info['pid'], new_nice):
                                adjusted_count += 1
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"Error adjusting process priorities: {e}")
        
        # Estimate power savings (roughly 0.2W per adjusted process)
        power_saved = adjusted_count * 0.2
        
        return OptimizationResult(
            action=PowerSavingAction.PROCESS_NICE_ADJUSTMENT,
            success=adjusted_count > 0,
            power_saved_w=power_saved,
            performance_impact=0.05,  # Minimal performance impact
            details=f"Adjusted priority for {adjusted_count} processes"
        )
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization"""
        return psutil.cpu_percent(interval=1.0)
    
    def _get_memory_utilization(self) -> float:
        """Get current memory utilization"""
        return psutil.virtual_memory().percent
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        with self._lock:
            history = self._optimization_history.copy()
        
        if not history:
            return {
                "total_optimizations": 0,
                "total_power_saved_w": 0.0,
                "avg_power_saved_w": 0.0,
                "success_rate": 0.0,
                "actions_applied": {}
            }
        
        successful = [r for r in history if r.success]
        total_power_saved = sum(r.power_saved_w for r in successful)
        
        # Count actions by type
        actions_count = {}
        for result in history:
            action_name = result.action.value
            if action_name not in actions_count:
                actions_count[action_name] = {"total": 0, "successful": 0, "power_saved": 0.0}
            
            actions_count[action_name]["total"] += 1
            if result.success:
                actions_count[action_name]["successful"] += 1
                actions_count[action_name]["power_saved"] += result.power_saved_w
        
        return {
            "total_optimizations": len(history),
            "successful_optimizations": len(successful),
            "total_power_saved_w": total_power_saved,
            "avg_power_saved_w": total_power_saved / len(successful) if successful else 0.0,
            "success_rate": len(successful) / len(history) if history else 0.0,
            "actions_applied": actions_count,
            "current_strategy": self.strategy.value
        }
    
    def add_custom_rule(self, rule: OptimizationRule) -> None:
        """Add a custom optimization rule"""
        self._optimization_rules.append(rule)
        logger.info(f"Added custom optimization rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an optimization rule by name"""
        for i, rule in enumerate(self._optimization_rules):
            if rule.name == rule_name:
                del self._optimization_rules[i]
                logger.info(f"Removed optimization rule: {rule_name}")
                return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable an optimization rule"""
        for rule in self._optimization_rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Enabled optimization rule: {rule_name}")
                return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable an optimization rule"""
        for rule in self._optimization_rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled optimization rule: {rule_name}")
                return True
        return False

# Global optimizer instance
_global_optimizer: Optional[PowerOptimizer] = None

def get_global_optimizer(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> PowerOptimizer:
    """Get or create global power optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PowerOptimizer(strategy)
    return _global_optimizer