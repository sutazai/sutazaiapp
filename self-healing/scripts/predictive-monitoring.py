#!/usr/bin/env python3
"""
Predictive Monitoring System for SutazAI
Detects anomalies and predicts failures before they occur
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis
import psutil
import docker
import requests
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import pickle
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('self_healing_predictions_total', 'Total predictions made', ['type', 'service'])
anomaly_gauge = Gauge('self_healing_anomaly_score', 'Current anomaly score', ['service'])
resource_prediction_gauge = Gauge('self_healing_resource_prediction', 'Predicted resource usage', ['resource', 'service'])
failure_prediction_gauge = Gauge('self_healing_failure_probability', 'Predicted failure probability', ['service'])


class MetricsCollector:
    """
    Collects system and service metrics for analysis
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.docker_client = None
        self._init_docker()
        
    def _init_docker(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': disk.percent,
                'disk_free': disk.free,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'process_count': process_count,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    def collect_container_metrics(self) -> Dict[str, Dict[str, float]]:
        """Collect Docker container metrics"""
        if not self.docker_client:
            return {}
            
        metrics = {}
        try:
            containers = self.docker_client.containers.list()
            for container in containers:
                if container.name.startswith('sutazai-'):
                    try:
                        stats = container.stats(stream=False)
                        
                        # CPU usage
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                   stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                      stats['precpu_stats']['system_cpu_usage']
                        cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0
                        
                        # Memory usage
                        memory_usage = stats['memory_stats'].get('usage', 0)
                        memory_limit = stats['memory_stats'].get('limit', 1)
                        memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
                        
                        # Network I/O
                        rx_bytes = sum(net['rx_bytes'] for net in stats['networks'].values())
                        tx_bytes = sum(net['tx_bytes'] for net in stats['networks'].values())
                        
                        metrics[container.name] = {
                            'cpu_percent': cpu_percent,
                            'memory_percent': memory_percent,
                            'memory_usage': memory_usage,
                            'network_rx_bytes': rx_bytes,
                            'network_tx_bytes': tx_bytes,
                            'status': container.status,
                            'restart_count': container.attrs['RestartCount'],
                            'timestamp': time.time()
                        }
                    except Exception as e:
                        logger.debug(f"Failed to get stats for {container.name}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to collect container metrics: {e}")
            
        return metrics
    
    def collect_service_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Collect application-level service metrics"""
        metrics = {}
        
        # Backend metrics
        try:
            response = requests.get('http://backend:8000/metrics', timeout=5)
            if response.status_code == 200:
                metrics['backend'] = response.json()
        except Exception:
            pass
            
        # Database metrics
        if self.redis_client:
            try:
                info = self.redis_client.info()
                metrics['redis'] = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory', 0),
                    'ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            except Exception:
                pass
                
        return metrics
    
    def store_metrics(self, metrics: Dict[str, Any], metric_type: str):
        """Store metrics in Redis for historical analysis"""
        if not self.redis_client:
            return
            
        try:
            key = f"metrics:{metric_type}:{int(time.time())}"
            self.redis_client.setex(
                key,
                86400,  # 24 hour TTL
                json.dumps(metrics)
            )
            
            # Also store in a time series list for quick access
            list_key = f"metrics:{metric_type}:list"
            self.redis_client.lpush(list_key, json.dumps(metrics))
            self.redis_client.ltrim(list_key, 0, 2880)  # Keep last 2 days at 1-minute intervals
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")


class AnomalyDetector:
    """
    Detects anomalies in system and service metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.threshold = config.get('sensitivity', 0.8)
        
    def train_model(self, service_name: str, historical_data: pd.DataFrame):
        """Train anomaly detection model for a service"""
        try:
            # Select relevant features
            feature_cols = [col for col in historical_data.columns 
                          if col not in ['timestamp', 'status']]
            self.feature_columns[service_name] = feature_cols
            
            # Prepare data
            X = historical_data[feature_cols].fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[service_name] = scaler
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=0.1,  # Expected anomaly rate
                random_state=42,
                n_estimators=100
            )
            model.fit(X_scaled)
            self.models[service_name] = model
            
            logger.info(f"Trained anomaly detection model for {service_name}")
            
        except Exception as e:
            logger.error(f"Failed to train model for {service_name}: {e}")
    
    def detect_anomaly(self, service_name: str, current_metrics: Dict[str, float]) -> Tuple[bool, float]:
        """Detect if current metrics are anomalous"""
        if service_name not in self.models:
            return False, 0.0
            
        try:
            # Prepare features
            features = [current_metrics.get(col, 0) for col in self.feature_columns[service_name]]
            X = np.array(features).reshape(1, -1)
            
            # Scale
            X_scaled = self.scalers[service_name].transform(X)
            
            # Predict
            anomaly_score = self.models[service_name].decision_function(X)[0]
            is_anomaly = self.models[service_name].predict(X_scaled)[0] == -1
            
            # Normalize score to 0-1 range
            normalized_score = 1 / (1 + np.exp(anomaly_score))
            
            # Update Prometheus metric
            anomaly_gauge.labels(service=service_name).set(normalized_score)
            
            return is_anomaly and normalized_score > self.threshold, normalized_score
            
        except Exception as e:
            logger.error(f"Failed to detect anomaly for {service_name}: {e}")
            return False, 0.0


class FailurePredictor:
    """
    Predicts potential failures based on trends and patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prediction_window = config.get('prediction_window', '30m')
        self.redis_client = None
        self._init_redis()
        
    def _init_redis(self):
        """Initialize Redis client"""
        try:
            self.redis_client = redis.Redis(
                host='redis',
                port=6379,
                decode_responses=True
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
    
    def predict_resource_exhaustion(self, 
                                  resource_type: str, 
                                  current_usage: float,
                                  historical_data: List[float]) -> Tuple[bool, float, float]:
        """
        Predict if a resource will be exhausted
        Returns: (will_exhaust, time_to_exhaustion_minutes, predicted_usage)
        """
        if len(historical_data) < 10:
            return False, float('inf'), current_usage
            
        try:
            # Simple linear regression for trend
            x = np.arange(len(historical_data))
            y = np.array(historical_data)
            
            # Calculate trend
            slope, intercept = np.polyfit(x, y, 1)
            
            # Predict future usage
            future_points = 30  # 30 minutes ahead
            predicted_usage = slope * (len(historical_data) + future_points) + intercept
            
            # Check if will exceed threshold
            threshold = self.config.get('metrics', {}).get('resource_usage', {}).get(f'{resource_type}_threshold', 90)
            
            if predicted_usage > threshold and slope > 0:
                # Calculate time to exhaustion
                time_to_threshold = (threshold - current_usage) / slope if slope > 0 else float('inf')
                
                # Update Prometheus metric
                resource_prediction_gauge.labels(
                    resource=resource_type, 
                    service='system'
                ).set(predicted_usage)
                
                return True, time_to_threshold, predicted_usage
                
            return False, float('inf'), predicted_usage
            
        except Exception as e:
            logger.error(f"Failed to predict resource exhaustion: {e}")
            return False, float('inf'), current_usage
    
    def predict_service_failure(self, 
                               service_name: str,
                               metrics_history: pd.DataFrame) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Predict if a service is likely to fail
        Returns: (likely_to_fail, failure_probability, contributing_factors)
        """
        if len(metrics_history) < 5:
            return False, 0.0, {}
            
        try:
            # Calculate risk factors
            risk_factors = {}
            failure_probability = 0.0
            
            # Check restart frequency
            if 'restart_count' in metrics_history.columns:
                recent_restarts = metrics_history['restart_count'].iloc[-5:].diff().sum()
                if recent_restarts > 2:
                    risk_factors['frequent_restarts'] = recent_restarts
                    failure_probability += 0.3
            
            # Check resource usage trends
            if 'cpu_percent' in metrics_history.columns:
                cpu_trend = metrics_history['cpu_percent'].iloc[-10:].mean()
                if cpu_trend > 80:
                    risk_factors['high_cpu'] = cpu_trend
                    failure_probability += 0.2
            
            if 'memory_percent' in metrics_history.columns:
                memory_trend = metrics_history['memory_percent'].iloc[-10:].mean()
                if memory_trend > 85:
                    risk_factors['high_memory'] = memory_trend
                    failure_probability += 0.3
            
            # Check for increasing error rates (if available)
            if 'error_rate' in metrics_history.columns:
                error_trend = metrics_history['error_rate'].iloc[-5:].mean()
                if error_trend > 0.05:
                    risk_factors['high_error_rate'] = error_trend
                    failure_probability += 0.2
            
            # Normalize probability
            failure_probability = min(failure_probability, 1.0)
            
            # Update Prometheus metric
            failure_prediction_gauge.labels(service=service_name).set(failure_probability)
            
            # Record prediction
            prediction_counter.labels(type='service_failure', service=service_name).inc()
            
            return failure_probability > 0.5, failure_probability, risk_factors
            
        except Exception as e:
            logger.error(f"Failed to predict service failure: {e}")
            return False, 0.0, {}


class PredictiveMonitor:
    """
    Main predictive monitoring system
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/self-healing/config/self-healing-config.yaml"):
        self.config_path = config_path
        self._load_config()
        self.collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector(
            self.config.get('anomaly_detection', {})
        )
        self.failure_predictor = FailurePredictor(
            self.config.get('predictive_monitoring', {})
        )
        self.redis_client = None
        self._init_redis()
        
    def _load_config(self):
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.monitoring_config = self.config.get('predictive_monitoring', {})
            
    def _init_redis(self):
        """Initialize Redis client"""
        try:
            self.redis_client = redis.Redis(
                host='redis',
                port=6379,
                decode_responses=True
            )
            self.collector.redis_client = self.redis_client
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        try:
            # Collect metrics
            system_metrics = self.collector.collect_system_metrics()
            container_metrics = self.collector.collect_container_metrics()
            service_metrics = self.collector.collect_service_metrics()
            
            # Store metrics
            self.collector.store_metrics(system_metrics, 'system')
            self.collector.store_metrics(container_metrics, 'containers')
            self.collector.store_metrics(service_metrics, 'services')
            
            # Analyze for anomalies
            alerts = []
            
            # System-level analysis
            if system_metrics:
                # Check CPU
                will_exhaust, time_to_exhaust, predicted = self.failure_predictor.predict_resource_exhaustion(
                    'cpu',
                    system_metrics['cpu_percent'],
                    self._get_historical_values('system', 'cpu_percent')
                )
                if will_exhaust:
                    alerts.append({
                        'type': 'resource_exhaustion',
                        'resource': 'cpu',
                        'time_to_exhaustion': time_to_exhaust,
                        'predicted_usage': predicted,
                        'current_usage': system_metrics['cpu_percent']
                    })
                
                # Check memory
                will_exhaust, time_to_exhaust, predicted = self.failure_predictor.predict_resource_exhaustion(
                    'memory',
                    system_metrics['memory_percent'],
                    self._get_historical_values('system', 'memory_percent')
                )
                if will_exhaust:
                    alerts.append({
                        'type': 'resource_exhaustion',
                        'resource': 'memory',
                        'time_to_exhaustion': time_to_exhaust,
                        'predicted_usage': predicted,
                        'current_usage': system_metrics['memory_percent']
                    })
            
            # Container-level analysis
            for container_name, metrics in container_metrics.items():
                # Detect anomalies
                is_anomaly, score = self.anomaly_detector.detect_anomaly(container_name, metrics)
                if is_anomaly:
                    alerts.append({
                        'type': 'anomaly_detected',
                        'service': container_name,
                        'anomaly_score': score,
                        'metrics': metrics
                    })
                
                # Predict failures
                history = self._get_container_history(container_name)
                if not history.empty:
                    likely_to_fail, probability, factors = self.failure_predictor.predict_service_failure(
                        container_name, 
                        history
                    )
                    if likely_to_fail:
                        alerts.append({
                            'type': 'failure_prediction',
                            'service': container_name,
                            'probability': probability,
                            'factors': factors
                        })
            
            # Process alerts
            if alerts:
                self._process_alerts(alerts)
                
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
    
    def _get_historical_values(self, metric_type: str, field: str, count: int = 30) -> List[float]:
        """Get historical values for a metric"""
        if not self.redis_client:
            return []
            
        try:
            values = []
            list_key = f"metrics:{metric_type}:list"
            
            raw_metrics = self.redis_client.lrange(list_key, 0, count)
            for raw in raw_metrics:
                try:
                    metrics = json.loads(raw)
                    if field in metrics:
                        values.append(float(metrics[field]))
                except Exception:
                    pass
                    
            return values[::-1]  # Reverse to get chronological order
            
        except Exception as e:
            logger.error(f"Failed to get historical values: {e}")
            return []
    
    def _get_container_history(self, container_name: str) -> pd.DataFrame:
        """Get historical metrics for a container"""
        if not self.redis_client:
            return pd.DataFrame()
            
        try:
            data = []
            list_key = os.getenv("METRICS_LIST_KEY", "metrics:containers:list")
            
            raw_metrics = self.redis_client.lrange(list_key, 0, 100)
            for raw in raw_metrics:
                try:
                    all_metrics = json.loads(raw)
                    if container_name in all_metrics:
                        data.append(all_metrics[container_name])
                except Exception:
                    pass
                    
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get container history: {e}")
            return pd.DataFrame()
    
    def _process_alerts(self, alerts: List[Dict[str, Any]]):
        """Process and send alerts"""
        for alert in alerts:
            logger.warning(f"PREDICTIVE ALERT: {alert}")
            
            # Store alert in Redis
            if self.redis_client:
                alert_key = f"alerts:predictive:{int(time.time())}"
                self.redis_client.setex(
                    alert_key,
                    3600,  # 1 hour TTL
                    json.dumps(alert)
                )
            
    
    def train_models(self):
        """Train anomaly detection models on historical data"""
        try:
            # Get list of containers
            containers = self.collector.docker_client.containers.list()
            
            for container in containers:
                if container.name.startswith('sutazai-'):
                    history = self._get_container_history(container.name)
                    if len(history) > 100:
                        self.anomaly_detector.train_model(container.name, history)
                        
        except Exception as e:
            logger.error(f"Failed to train models: {e}")


def main():
    """Main entry point"""
    monitor = PredictiveMonitor()
    
    # Train models on startup
    logger.info("Training anomaly detection models...")
    monitor.train_models()
    
    # Run monitoring loop
    logger.info("Starting predictive monitoring...")
    while True:
        monitor.run_monitoring_cycle()
        time.sleep(60)  # Run every minute


if __name__ == "__main__":
    main()