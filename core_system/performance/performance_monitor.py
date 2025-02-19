#!/usr/bin/env python3
"""
SutazAI Performance Monitoring System

Comprehensive performance tracking and optimization module
"""

import os
import time
import psutil
import logging
import threading
import multiprocessing
from typing import Dict, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/opt/sutazai_project/SutazAI/logs/performance_monitor.log'
)
logger = logging.getLogger('SutazAI_Performance')

@dataclass
class PerformanceMetrics:
    timestamp: float
    cpu_usage: float
    memory_usage: Dict[str, float]
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    process_count: int
    thread_count: int

class PerformanceMonitor:
    def __init__(self, 
                 sample_interval: int = 5, 
                 log_directory: str = '/opt/sutazai_project/SutazAI/logs/performance'):
        self.sample_interval = sample_interval
        self.log_directory = log_directory
        os.makedirs(log_directory, exist_ok=True)
        self._stop_event = threading.Event()

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system performance metrics"""
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=psutil.cpu_percent(interval=1),
            memory_usage={
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            disk_io=dict(psutil.disk_io_counters()._asdict()),
            network_io=dict(psutil.net_io_counters()._asdict()),
            process_count=len(psutil.process_iter()),
            thread_count=threading.active_count()
        )

    def start_monitoring(self):
        """Start continuous performance monitoring"""
        logger.info("Starting SutazAI Performance Monitoring")
        monitoring_thread = threading.Thread(target=self._monitor_loop)
        monitoring_thread.start()

    def _monitor_loop(self):
        """Continuous monitoring loop with intelligent sampling"""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self._log_metrics(metrics)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics to file"""
        log_file = os.path.join(
            self.log_directory, 
            f'performance_{int(metrics.timestamp)}.json'
        )
        with open(log_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

    def stop_monitoring(self):
        """Gracefully stop performance monitoring"""
        self._stop_event.set()
        logger.info("Stopped SutazAI Performance Monitoring")

def main():
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        while True:
            time.sleep(60)  # Keep main thread alive
    except KeyboardInterrupt:
        monitor.stop_monitoring()

if __name__ == '__main__':
    main() 