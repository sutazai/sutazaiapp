import logging
import time
import threading
from typing import Dict, Any, Optional

class AutoScaler:
    """
    Advanced auto-scaling system for dynamic resource management
    
    Provides intelligent scaling of computational resources based on 
    system metrics, workload, and predefined scaling policies.
    """
    
    def __init__(
        self, 
        log_path: str = 'logs/auto_scaler.log',
        max_scale_up_threshold: float = 80.0,
        max_scale_down_threshold: float = 20.0
    ):
        """
        Initialize the AutoScaler with configuration parameters
        
        :param log_path: Path to log file for auto-scaling events
        :param max_scale_up_threshold: CPU usage percentage to trigger scale-up
        :param max_scale_down_threshold: CPU usage percentage to trigger scale-down
        """
        logging.basicConfig(
            filename=log_path, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.max_scale_up_threshold = max_scale_up_threshold
        self.max_scale_down_threshold = max_scale_down_threshold
        
        self.current_resources = {
            'cpu_cores': 1,
            'memory_gb': 4,
            'max_cpu_cores': 16,
            'max_memory_gb': 64
        }
        
        self.scaling_history = []
    
    def _log_scaling_event(
        self, 
        event_type: str, 
        old_resources: Dict[str, Any], 
        new_resources: Dict[str, Any]
    ) -> None:
        """
        Log scaling events for audit and analysis
        
        :param event_type: Type of scaling event (scale_up/scale_down)
        :param old_resources: Resources before scaling
        :param new_resources: Resources after scaling
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'old_resources': old_resources,
            'new_resources': new_resources
        }
        self.scaling_history.append(event)
        self.logger.info(f"Scaling Event: {event}")
    
    def adjust_resources(self, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dynamically adjust system resources based on current metrics
        
        :param metrics: Optional system metrics dictionary
        :return: Updated resource configuration
        """
        try:
            if metrics is None:
                from core_system.monitoring.system_monitor import SystemMonitor
                metrics = SystemMonitor().get_system_metrics()
            
            cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
            
            old_resources = self.current_resources.copy()
            
            # Scale-up logic
            if cpu_usage > self.max_scale_up_threshold:
                if self.current_resources['cpu_cores'] < self.current_resources['max_cpu_cores']:
                    self.current_resources['cpu_cores'] *= 2
                    self.current_resources['memory_gb'] *= 2
                    self._log_scaling_event('scale_up', old_resources, self.current_resources)
            
            # Scale-down logic
            elif cpu_usage < self.max_scale_down_threshold:
                if self.current_resources['cpu_cores'] > 1:
                    self.current_resources['cpu_cores'] //= 2
                    self.current_resources['memory_gb'] //= 2
                    self._log_scaling_event('scale_down', old_resources, self.current_resources)
            
            return self.current_resources
        
        except Exception as e:
            self.logger.error(f"Resource adjustment failed: {e}")
            return self.current_resources
    
    def monitor_and_scale(self, interval: int = 300) -> None:
        """
        Continuously monitor and adjust system resources
        
        :param interval: Monitoring interval in seconds
        """
        def _scaling_thread():
            try:
                while True:
                    self.adjust_resources()
                    time.sleep(interval)
            except Exception as e:
                self.logger.critical(f"Scaling monitor thread failed: {e}")
        
        scaling_thread = threading.Thread(target=_scaling_thread, daemon=True)
        scaling_thread.start()
        self.logger.info("Auto-scaling monitor started")
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """
        Generate scaling recommendations based on historical data
        
        :return: Dictionary of scaling recommendations
        """
        if not self.scaling_history:
            return {"status": "insufficient_data"}
        
        recommendations = {
            "total_scaling_events": len(self.scaling_history),
            "last_scale_up": None,
            "last_scale_down": None,
            "average_scale_frequency": None
        }
        
        scale_up_events = [event for event in self.scaling_history if event['type'] == 'scale_up']
        scale_down_events = [event for event in self.scaling_history if event['type'] == 'scale_down']
        
        if scale_up_events:
            recommendations['last_scale_up'] = scale_up_events[-1]['timestamp']
        
        if scale_down_events:
            recommendations['last_scale_down'] = scale_down_events[-1]['timestamp']
        
        if len(self.scaling_history) > 1:
            time_between_events = [
                self.scaling_history[i+1]['timestamp'] - self.scaling_history[i]['timestamp']
                for i in range(len(self.scaling_history) - 1)
            ]
            recommendations['average_scale_frequency'] = sum(time_between_events) / len(time_between_events)
        
        return recommendations

def main():
    """
    Standalone execution for auto-scaling system
    """
    scaler = AutoScaler()
    scaler.monitor_and_scale()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Auto-scaling monitor stopped.")

if __name__ == "__main__":
    main()
