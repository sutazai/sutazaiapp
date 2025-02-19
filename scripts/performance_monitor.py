#!/usr/bin/env python3
"""
SutazAI Performance Monitoring and Optimization Script

Provides comprehensive system performance tracking, 
analysis, and autonomous optimization capabilities.
"""

import os
import sys
import time
import logging
import yaml
import threading
import psutil
import ray
import json
from typing import Dict, Any, List, Optional

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import internal performance optimization modules
from core_system.performance_optimizer import AdvancedPerformanceOptimizer

class PerformanceMonitoringSystem:
    """
    Ultra-Comprehensive Performance Monitoring and Optimization Framework
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        config_path: Optional[str] = None
    ):
        """
        Initialize Performance Monitoring System
        
        Args:
            base_dir (str): Base project directory
            config_path (Optional[str]): Path to performance monitoring configuration
        """
        # Core configuration
        self.base_dir = base_dir
        self.config_path = config_path or os.path.join(base_dir, 'config', 'performance_monitoring_config.yml')
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Logging setup
        log_dir = os.path.join(base_dir, 'logs', 'performance_monitoring')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.getLevelName(self.config['logging']['level']),
            format=self.config['logging']['format'],
            filename=os.path.join(log_dir, 'performance_monitor.log')
        )
        self.logger = logging.getLogger('SutazAI.PerformanceMonitor')
        
        # Initialize performance optimizer
        self.performance_optimizer = AdvancedPerformanceOptimizer(base_dir)
        
        # Synchronization primitives
        self._stop_monitoring = threading.Event()
        self._monitoring_thread = None
    
    def start_performance_monitoring(self):
        """
        Start comprehensive performance monitoring
        """
        # Initialize Ray for distributed computing
        ray.init(ignore_reinit_error=True)
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._continuous_performance_monitoring,
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def _continuous_performance_monitoring(self):
        """
        Perform continuous performance monitoring and optimization
        """
        interval = self.config['global']['monitoring_interval']
        
        while not self._stop_monitoring.is_set():
            try:
                # Monitor system resources
                system_metrics = self.performance_optimizer.monitor_system_resources()
                
                # Check resource thresholds
                self._check_resource_thresholds(system_metrics)
                
                # Optimize system performance
                optimization_results = self.performance_optimizer.optimize_system_performance()
                
                # Log optimization insights
                self._log_performance_insights(system_metrics, optimization_results)
                
                # Persist performance history
                self.performance_optimizer.persist_performance_history()
                
                # Wait for next monitoring cycle
                time.sleep(interval)
            
            except Exception as e:
                self.logger.error(f"Performance monitoring cycle failed: {e}")
                time.sleep(interval)  # Backoff on continuous errors
    
    def _check_resource_thresholds(self, system_metrics: Dict[str, Any]):
        """
        Check system resource usage against configured thresholds
        
        Args:
            system_metrics (Dict): Comprehensive system resource metrics
        """
        thresholds = self.config['resource_thresholds']
        alerts = []
        
        # CPU Threshold Check
        if system_metrics['cpu']['usage_percent'] > thresholds['cpu']['critical_percent']:
            alerts.append({
                'type': 'CRITICAL_CPU_USAGE',
                'current_value': system_metrics['cpu']['usage_percent'],
                'threshold': thresholds['cpu']['critical_percent']
            })
        elif system_metrics['cpu']['usage_percent'] > thresholds['cpu']['warning_percent']:
            alerts.append({
                'type': 'WARNING_CPU_USAGE',
                'current_value': system_metrics['cpu']['usage_percent'],
                'threshold': thresholds['cpu']['warning_percent']
            })
        
        # Memory Threshold Check
        if system_metrics['memory']['used_percent'] > thresholds['memory']['critical_percent']:
            alerts.append({
                'type': 'CRITICAL_MEMORY_USAGE',
                'current_value': system_metrics['memory']['used_percent'],
                'threshold': thresholds['memory']['critical_percent']
            })
        elif system_metrics['memory']['used_percent'] > thresholds['memory']['warning_percent']:
            alerts.append({
                'type': 'WARNING_MEMORY_USAGE',
                'current_value': system_metrics['memory']['used_percent'],
                'threshold': thresholds['memory']['warning_percent']
            })
        
        # Disk Threshold Check
        if system_metrics['disk']['used_percent'] > thresholds['disk']['critical_percent']:
            alerts.append({
                'type': 'CRITICAL_DISK_USAGE',
                'current_value': system_metrics['disk']['used_percent'],
                'threshold': thresholds['disk']['critical_percent']
            })
        elif system_metrics['disk']['used_percent'] > thresholds['disk']['warning_percent']:
            alerts.append({
                'type': 'WARNING_DISK_USAGE',
                'current_value': system_metrics['disk']['used_percent'],
                'threshold': thresholds['disk']['warning_percent']
            })
        
        # Handle alerts
        if alerts:
            self._handle_performance_alerts(alerts)
    
    def _handle_performance_alerts(self, alerts: List[Dict[str, Any]]):
        """
        Handle performance alerts with intelligent response
        
        Args:
            alerts (List): Performance alerts
        """
        for alert in alerts:
            # Log alert
            self.logger.warning(f"Performance Alert: {alert['type']} - Current: {alert['current_value']}%")
            
            # Trigger alert channels based on configuration
            if self.config['alerting']['enabled']:
                self._send_performance_alerts(alert)
    
    def _send_performance_alerts(self, alert: Dict[str, Any]):
        """
        Send performance alerts through configured channels
        
        Args:
            alert (Dict): Performance alert details
        """
        channels = self.config['alerting']['channels']
        
        # Placeholder for actual alert implementation
        # In a real-world scenario, integrate with email, Slack, SMS services
        for channel in channels:
            if channel == 'email':
                self._send_email_alert(alert)
            elif channel == 'slack':
                self._send_slack_alert(alert)
            elif channel == 'sms':
                self._send_sms_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """
        Send performance alert via email
        
        Args:
            alert (Dict): Performance alert details
        """
        # Placeholder for email sending logic
        pass
    
    def _send_slack_alert(self, alert: Dict[str, Any]):
        """
        Send performance alert via Slack
        
        Args:
            alert (Dict): Performance alert details
        """
        # Placeholder for Slack notification logic
        pass
    
    def _send_sms_alert(self, alert: Dict[str, Any]):
        """
        Send performance alert via SMS
        
        Args:
            alert (Dict): Performance alert details
        """
        # Placeholder for SMS sending logic
        pass
    
    def _log_performance_insights(
        self, 
        system_metrics: Dict[str, Any], 
        optimization_results: Dict[str, Any]
    ):
        """
        Log comprehensive performance insights
        
        Args:
            system_metrics (Dict): System resource metrics
            optimization_results (Dict): Performance optimization results
        """
        insights_log = {
            'timestamp': time.time(),
            'system_metrics': system_metrics,
            'optimization_recommendations': optimization_results.get('recommendations', [])
        }
        
        # Persist insights
        insights_file = os.path.join(
            self.base_dir, 
            'logs', 
            'performance_insights', 
            f'insights_{time.strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        os.makedirs(os.path.dirname(insights_file), exist_ok=True)
        
        with open(insights_file, 'w') as f:
            json.dump(insights_log, f, indent=2)
    
    def stop_performance_monitoring(self):
        """
        Gracefully stop performance monitoring
        """
        self._stop_monitoring.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join()
        
        # Shutdown Ray
        ray.shutdown()
        
        self.logger.info("Performance monitoring stopped")

def main():
    """
    Main execution for performance monitoring system
    """
    try:
        # Initialize performance monitoring system
        performance_monitor = PerformanceMonitoringSystem()
        
        # Start performance monitoring
        performance_monitor.start_performance_monitoring()
        
        # Keep main thread alive
        while True:
            time.sleep(3600)  # Sleep for an hour
    
    except KeyboardInterrupt:
        performance_monitor.stop_performance_monitoring()
    except Exception as e:
        logging.critical(f"Performance monitoring failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 