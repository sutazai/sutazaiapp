#!/usr/bin/env python3
"""
Neural Health Monitor - Deep Learning-based Container Health Monitoring and Prediction
Implements predictive health monitoring, anomaly detection, and intelligent repair strategies
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import docker
import redis
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContainerHealthPredictor(nn.Module):
    """
    Neural network for predicting container health states
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class DependencyGraphNN(nn.Module):
    """
    Graph neural network for modeling container dependencies
    """
    
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_features, adjacency_matrix):
        # Encode nodes
        node_embeddings = self.node_encoder(node_features)
        
        # Apply attention based on adjacency
        attended_features, _ = self.attention(
            node_embeddings, node_embeddings, node_embeddings,
            attn_mask=~adjacency_matrix
        )
        
        # Combine node and edge information
        combined = torch.cat([attended_features, self.edge_encoder(edge_features)], dim=-1)
        
        # Predict health impact
        return self.decoder(combined)


class MetaLearningOptimizer(nn.Module):
    """
    Meta-learning model for optimizing repair strategies
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_encoder = nn.LSTM(state_dim, hidden_dim, num_layers=2, batch_first=True)
        
        self.action_value = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.success_predictor = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, states, actions):
        # Encode state sequence
        _, (h_n, _) = self.state_encoder(states)
        state_encoding = h_n[-1]
        
        # Concatenate with action
        combined = torch.cat([state_encoding, actions], dim=-1)
        
        # Predict action value and success probability
        value = self.action_value(combined)
        success_prob = self.success_predictor(combined)
        
        return value, success_prob


class NeuralHealthMonitor:
    """
    Advanced neural health monitoring system with predictive capabilities
    """
    
    def __init__(self, 
                 model_dir: str = "/opt/sutazaiapp/models/health_monitor",
                 history_window: int = 100):
        
        self.model_dir = model_dir
        self.history_window = history_window
        
        # Initialize Docker and Redis clients
        self.docker_client = docker.from_env()
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Health metrics history
        self.metrics_history = deque(maxlen=history_window)
        self.failure_patterns = {}
        self.repair_success_history = {}
        
        # Feature extractors
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Neural models
        self.health_predictor = None
        self.dependency_graph = None
        self.meta_optimizer = None
        
        # Load or initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize or load neural models"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Health predictor
        self.health_predictor = ContainerHealthPredictor(
            input_dim=50,  # Container metrics features
            hidden_dims=[128, 64, 32],
            output_dim=4  # healthy, warning, critical, failed
        )
        
        # Dependency graph model
        self.dependency_graph = DependencyGraphNN(
            node_features=30,
            edge_features=10,
            hidden_dim=64
        )
        
        # Meta-learning optimizer
        self.meta_optimizer = MetaLearningOptimizer(
            state_dim=40,
            action_dim=10,
            hidden_dim=128
        )
        
        # Load pre-trained weights if available
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained model weights"""
        try:
            if os.path.exists(f"{self.model_dir}/health_predictor.pth"):
                self.health_predictor.load_state_dict(
                    torch.load(f"{self.model_dir}/health_predictor.pth")
                )
                logger.info("Loaded health predictor model")
                
            if os.path.exists(f"{self.model_dir}/dependency_graph.pth"):
                self.dependency_graph.load_state_dict(
                    torch.load(f"{self.model_dir}/dependency_graph.pth")
                )
                logger.info("Loaded dependency graph model")
                
            if os.path.exists(f"{self.model_dir}/meta_optimizer.pth"):
                self.meta_optimizer.load_state_dict(
                    torch.load(f"{self.model_dir}/meta_optimizer.pth")
                )
                logger.info("Loaded meta optimizer model")
                
            if os.path.exists(f"{self.model_dir}/scaler.pkl"):
                self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
                logger.info("Loaded feature scaler")
                
            if os.path.exists(f"{self.model_dir}/anomaly_detector.pkl"):
                self.anomaly_detector = joblib.load(f"{self.model_dir}/anomaly_detector.pkl")
                logger.info("Loaded anomaly detector")
                
        except Exception as e:
            logger.warning(f"Could not load some models: {e}")
    
    def save_models(self):
        """Save trained models"""
        torch.save(self.health_predictor.state_dict(), 
                   f"{self.model_dir}/health_predictor.pth")
        torch.save(self.dependency_graph.state_dict(), 
                   f"{self.model_dir}/dependency_graph.pth")
        torch.save(self.meta_optimizer.state_dict(), 
                   f"{self.model_dir}/meta_optimizer.pth")
        joblib.dump(self.scaler, f"{self.model_dir}/scaler.pkl")
        joblib.dump(self.anomaly_detector, f"{self.model_dir}/anomaly_detector.pkl")
        logger.info("Saved all models")
    
    def extract_container_features(self, container) -> np.ndarray:
        """Extract features from container stats"""
        try:
            stats = container.stats(stream=False)
            
            # CPU features
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0
            
            # Memory features
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
            
            # Network features
            rx_bytes = sum(net['rx_bytes'] for net in stats['networks'].values())
            tx_bytes = sum(net['tx_bytes'] for net in stats['networks'].values())
            
            # Container metadata
            created_time = datetime.fromisoformat(container.attrs['Created'].replace('Z', '+00:00'))
            uptime_hours = (datetime.now(created_time.tzinfo) - created_time).total_seconds() / 3600
            
            # Restart count
            restart_count = container.attrs['RestartCount']
            
            # Health check status
            health_status = 0  # default healthy
            if 'Health' in container.attrs['State']:
                health = container.attrs['State']['Health']['Status']
                health_status = {'healthy': 0, 'unhealthy': 1, 'starting': 2}.get(health, 3)
            
            # Process count (approximation)
            process_count = stats['pids_stats']['current'] if 'pids_stats' in stats else 1
            
            # Build feature vector
            features = [
                cpu_percent,
                memory_percent,
                memory_usage / 1e9,  # GB
                rx_bytes / 1e6,  # MB
                tx_bytes / 1e6,  # MB
                uptime_hours,
                restart_count,
                health_status,
                process_count,
                # Add time-based features
                datetime.now().hour,  # Hour of day
                datetime.now().weekday(),  # Day of week
            ]
            
            # Add rolling statistics if history exists
            if len(self.metrics_history) > 0:
                recent_metrics = [m.get(container.name, {}) for m in self.metrics_history[-10:]]
                if recent_metrics:
                    cpu_history = [m.get('cpu_percent', 0) for m in recent_metrics]
                    memory_history = [m.get('memory_percent', 0) for m in recent_metrics]
                    
                    features.extend([
                        np.mean(cpu_history),
                        np.std(cpu_history),
                        np.max(cpu_history),
                        np.mean(memory_history),
                        np.std(memory_history),
                        np.max(memory_history),
                    ])
                else:
                    features.extend([0] * 6)
            else:
                features.extend([0] * 6)
            
            # Pad to expected dimension
            while len(features) < 50:
                features.append(0)
                
            return np.array(features[:50])
            
        except Exception as e:
            logger.error(f"Error extracting features from {container.name}: {e}")
            return np.zeros(50)
    
    def build_dependency_graph(self, containers: List[docker.models.containers.Container]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build container dependency graph"""
        n = len(containers)
        adjacency = torch.zeros((n, n))
        
        # Extract network connections
        for i, container_i in enumerate(containers):
            for j, container_j in enumerate(containers):
                if i != j:
                    # Check if containers share networks
                    networks_i = set(container_i.attrs['NetworkSettings']['Networks'].keys())
                    networks_j = set(container_j.attrs['NetworkSettings']['Networks'].keys())
                    
                    if networks_i & networks_j:  # Shared networks
                        adjacency[i, j] = 1.0
                        
                    # Check for explicit links or depends_on
                    if 'Links' in container_i.attrs['HostConfig']:
                        links = container_i.attrs['HostConfig']['Links'] or []
                        if any(container_j.name in link for link in links):
                            adjacency[i, j] = 2.0  # Stronger dependency
        
        return adjacency
    
    def predict_container_health(self, container) -> Dict[str, float]:
        """Predict container health state"""
        features = self.extract_container_features(container)
        
        # Detect anomalies
        features_2d = features.reshape(1, -1)
        if hasattr(self.scaler, 'mean_'):  # Check if scaler is fitted
            features_scaled = self.scaler.transform(features_2d)
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
        else:
            is_anomaly = False
            features_scaled = features_2d
        
        # Predict health state
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled)
            health_probs = self.health_predictor(features_tensor).numpy()[0]
        
        health_states = ['healthy', 'warning', 'critical', 'failed']
        predictions = {state: float(prob) for state, prob in zip(health_states, health_probs)}
        predictions['anomaly'] = float(is_anomaly)
        
        return predictions
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze historical failure patterns"""
        if len(self.metrics_history) < 10:
            return {}
        
        patterns = {
            'recurring_failures': {},
            'cascade_patterns': [],
            'time_patterns': {},
            'resource_correlations': {}
        }
        
        # Analyze recurring failures
        for container_name, failures in self.failure_patterns.items():
            if len(failures) > 2:
                # Calculate failure intervals
                intervals = []
                for i in range(1, len(failures)):
                    interval = (failures[i] - failures[i-1]).total_seconds() / 3600
                    intervals.append(interval)
                
                if intervals:
                    patterns['recurring_failures'][container_name] = {
                        'count': len(failures),
                        'mean_interval_hours': np.mean(intervals),
                        'std_interval_hours': np.std(intervals)
                    }
        
        # Detect cascade patterns
        failure_sequences = []
        for i in range(len(self.metrics_history) - 1):
            curr_failed = set(self.metrics_history[i].get('failed_containers', []))
            next_failed = set(self.metrics_history[i + 1].get('failed_containers', []))
            
            new_failures = next_failed - curr_failed
            if len(curr_failed) > 0 and len(new_failures) > 0:
                failure_sequences.append({
                    'trigger': list(curr_failed),
                    'cascade': list(new_failures)
                })
        
        # Group similar cascades
        from collections import Counter
        cascade_counter = Counter(
            tuple(sorted(seq['trigger']) + ['->'] + sorted(seq['cascade']))
            for seq in failure_sequences
        )
        
        patterns['cascade_patterns'] = [
            {'pattern': pattern, 'count': count}
            for pattern, count in cascade_counter.most_common(5)
        ]
        
        return patterns
    
    def recommend_repair_action(self, container_name: str, health_state: Dict[str, float]) -> Dict[str, Any]:
        """Recommend optimal repair action using meta-learning"""
        # Define possible actions
        actions = {
            'restart': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'scale_up': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            'scale_down': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            'update_limits': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            'clear_cache': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            'reconnect_deps': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            'rollback': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            'health_check': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            'wait': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            'custom': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        }
        
        # Get container state history
        state_history = []
        for metrics in list(self.metrics_history)[-10:]:
            if container_name in metrics:
                state_history.append(metrics[container_name].get('features', np.zeros(40)))
        
        if not state_history:
            state_history = [np.zeros(40)]
        
        # Pad state history
        while len(state_history) < 10:
            state_history.insert(0, np.zeros(40))
        
        state_tensor = torch.FloatTensor([state_history])
        
        # Evaluate each action
        action_scores = {}
        with torch.no_grad():
            for action_name, action_vector in actions.items():
                action_tensor = torch.FloatTensor([action_vector])
                value, success_prob = self.meta_optimizer(state_tensor, action_tensor)
                
                # Adjust score based on historical success
                historical_success = self.repair_success_history.get(
                    f"{container_name}_{action_name}", 0.5
                )
                
                adjusted_score = float(value[0]) * float(success_prob[0]) * historical_success
                action_scores[action_name] = {
                    'score': adjusted_score,
                    'value': float(value[0]),
                    'success_probability': float(success_prob[0]),
                    'historical_success': historical_success
                }
        
        # Sort actions by score
        sorted_actions = sorted(action_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Build recommendation
        recommendation = {
            'primary_action': sorted_actions[0][0],
            'confidence': sorted_actions[0][1]['success_probability'],
            'alternatives': [
                {
                    'action': action,
                    'score': details['score'],
                    'confidence': details['success_probability']
                }
                for action, details in sorted_actions[1:4]
            ],
            'reasoning': self._generate_reasoning(container_name, health_state, sorted_actions[0])
        }
        
        return recommendation
    
    def _generate_reasoning(self, container_name: str, health_state: Dict[str, float], 
                           chosen_action: Tuple[str, Dict]) -> str:
        """Generate explanation for recommended action"""
        action_name, action_details = chosen_action
        
        reasons = []
        
        # Health state reasoning
        if health_state['critical'] > 0.5:
            reasons.append(f"Container is in critical state (probability: {health_state['critical']:.2f})")
        elif health_state['warning'] > 0.5:
            reasons.append(f"Container showing warning signs (probability: {health_state['warning']:.2f})")
        
        if health_state.get('anomaly', False):
            reasons.append("Anomalous behavior detected in container metrics")
        
        # Historical reasoning
        if action_details['historical_success'] > 0.7:
            reasons.append(f"This action has {action_details['historical_success']:.0%} historical success rate")
        elif action_details['historical_success'] < 0.3:
            reasons.append("Note: This action has low historical success rate but may be necessary")
        
        # Action-specific reasoning
        action_reasons = {
            'restart': "Quick recovery through container restart",
            'scale_up': "Increase resources to handle load",
            'scale_down': "Reduce resource consumption",
            'update_limits': "Adjust resource limits for optimal performance",
            'clear_cache': "Free up memory and resolve cache-related issues",
            'reconnect_deps': "Restore broken connections with dependent services",
            'rollback': "Revert to previous stable version",
            'health_check': "Perform detailed diagnostics",
            'wait': "Allow temporary issues to self-resolve",
            'custom': "Requires specialized intervention"
        }
        
        if action_name in action_reasons:
            reasons.append(action_reasons[action_name])
        
        return " | ".join(reasons)
    
    def train_models(self, training_data: Optional[Dict] = None):
        """Train neural models on collected data"""
        if training_data is None:
            # Use historical data
            if len(self.metrics_history) < 100:
                logger.warning("Insufficient data for training")
                return
            
            training_data = self._prepare_training_data()
        
        # Train health predictor
        self._train_health_predictor(training_data['health_data'])
        
        # Train dependency graph model
        self._train_dependency_model(training_data['dependency_data'])
        
        # Train meta-optimizer
        self._train_meta_optimizer(training_data['repair_data'])
        
        # Save models
        self.save_models()
        
    def _prepare_training_data(self) -> Dict:
        """Prepare training data from historical metrics"""
        health_data = []
        dependency_data = []
        repair_data = []
        
        # Extract features and labels from history
        for i, metrics in enumerate(self.metrics_history):
            for container_name, container_metrics in metrics.items():
                if 'features' in container_metrics and 'health_state' in container_metrics:
                    health_data.append({
                        'features': container_metrics['features'],
                        'label': container_metrics['health_state']
                    })
        
        return {
            'health_data': health_data,
            'dependency_data': dependency_data,
            'repair_data': repair_data
        }
    
    def _train_health_predictor(self, health_data: List[Dict]):
        """Train the health prediction model"""
        if not health_data:
            return
        
        # Prepare data
        X = np.array([d['features'] for d in health_data])
        y = np.array([d['label'] for d in health_data])
        
        # Fit scaler and anomaly detector
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.anomaly_detector.fit(X_scaled[y == 0])  # Train on healthy samples
        
        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.LongTensor(y)
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train model
        optimizer = optim.Adam(self.health_predictor.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.health_predictor.train()
        for epoch in range(50):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.health_predictor(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Health predictor epoch {epoch}, loss: {total_loss:.4f}")
        
        self.health_predictor.eval()
    
    def _train_dependency_model(self, dependency_data: List[Dict]):
        """Train the dependency graph model"""
        # Implementation depends on collected dependency data
        pass
    
    def _train_meta_optimizer(self, repair_data: List[Dict]):
        """Train the meta-learning optimizer"""
        # Implementation depends on repair action outcomes
        pass
    
    async def continuous_learning_loop(self):
        """Continuous learning from system behavior"""
        while True:
            try:
                # Collect current metrics
                current_metrics = await self.collect_all_metrics()
                
                # Store in history
                self.metrics_history.append(current_metrics)
                
                # Store in Redis for persistence
                self.redis_client.lpush(
                    'health_metrics_history',
                    json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'metrics': current_metrics
                    })
                )
                self.redis_client.ltrim('health_metrics_history', 0, 1000)
                
                # Periodic model retraining
                if len(self.metrics_history) % 100 == 0:
                    logger.info("Starting periodic model retraining")
                    self.train_models()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(60)
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics from all containers"""
        metrics = {}
        
        try:
            containers = self.docker_client.containers.list(all=True)
            
            for container in containers:
                if container.name.startswith('sutazai-'):
                    # Extract features
                    features = self.extract_container_features(container)
                    
                    # Predict health
                    health_prediction = self.predict_container_health(container)
                    
                    # Store metrics
                    metrics[container.name] = {
                        'features': features.tolist(),
                        'health_prediction': health_prediction,
                        'status': container.status,
                        'health_state': np.argmax([
                            health_prediction['healthy'],
                            health_prediction['warning'],
                            health_prediction['critical'],
                            health_prediction['failed']
                        ])
                    }
                    
                    # Track failures
                    if container.status != 'running' or health_prediction['failed'] > 0.5:
                        if container.name not in self.failure_patterns:
                            self.failure_patterns[container.name] = []
                        self.failure_patterns[container.name].append(datetime.now())
            
            # Add system-wide metrics
            metrics['system'] = {
                'total_containers': len(containers),
                'running_containers': len([c for c in containers if c.status == 'running']),
                'failed_containers': [c.name for c in containers if c.status != 'running'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive insights from neural analysis"""
        patterns = self.analyze_failure_patterns()
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'health_summary': {},
            'failure_patterns': patterns,
            'recommendations': [],
            'predicted_issues': [],
            'optimization_opportunities': []
        }
        
        # Current health summary
        current_metrics = self.metrics_history[-1] if self.metrics_history else {}
        for container_name, metrics in current_metrics.items():
            if container_name != 'system' and 'health_prediction' in metrics:
                insights['health_summary'][container_name] = metrics['health_prediction']
        
        # Generate recommendations
        for container_name, health in insights['health_summary'].items():
            if health['critical'] > 0.3 or health['warning'] > 0.5:
                recommendation = self.recommend_repair_action(container_name, health)
                insights['recommendations'].append({
                    'container': container_name,
                    'recommendation': recommendation
                })
        
        # Predict future issues
        if len(self.metrics_history) > 20:
            insights['predicted_issues'] = self._predict_future_issues()
        
        # Identify optimization opportunities
        insights['optimization_opportunities'] = self._identify_optimizations()
        
        return insights
    
    def _predict_future_issues(self) -> List[Dict[str, Any]]:
        """Predict potential future issues based on trends"""
        predictions = []
        
        # Analyze trends in metrics history
        for container_name in set().union(*[set(m.keys()) for m in self.metrics_history]):
            if container_name == 'system':
                continue
            
            # Extract time series
            cpu_series = []
            memory_series = []
            
            for metrics in self.metrics_history:
                if container_name in metrics and 'features' in metrics[container_name]:
                    features = metrics[container_name]['features']
                    cpu_series.append(features[0])  # CPU percent
                    memory_series.append(features[1])  # Memory percent
            
            if len(cpu_series) > 10:
                # Simple trend analysis
                cpu_trend = np.polyfit(range(len(cpu_series)), cpu_series, 1)[0]
                memory_trend = np.polyfit(range(len(memory_series)), memory_series, 1)[0]
                
                # Predict issues
                if cpu_trend > 0.5:  # Increasing CPU usage
                    predictions.append({
                        'container': container_name,
                        'issue': 'CPU exhaustion',
                        'timeframe': 'next 2-4 hours',
                        'confidence': min(0.9, cpu_trend / 2),
                        'current_cpu': cpu_series[-1],
                        'trend': f"+{cpu_trend:.2f}% per measurement"
                    })
                
                if memory_trend > 0.3:  # Increasing memory usage
                    predictions.append({
                        'container': container_name,
                        'issue': 'Memory exhaustion',
                        'timeframe': 'next 4-6 hours',
                        'confidence': min(0.9, memory_trend / 1.5),
                        'current_memory': memory_series[-1],
                        'trend': f"+{memory_trend:.2f}% per measurement"
                    })
        
        return predictions
    
    def _identify_optimizations(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities"""
        optimizations = []
        
        current_metrics = self.metrics_history[-1] if self.metrics_history else {}
        
        for container_name, metrics in current_metrics.items():
            if container_name == 'system' or 'features' not in metrics:
                continue
            
            features = metrics['features']
            cpu_usage = features[0]
            memory_usage = features[1]
            
            # Under-utilized resources
            if cpu_usage < 5 and memory_usage < 10:
                optimizations.append({
                    'container': container_name,
                    'type': 'resource_reduction',
                    'reason': 'Consistently low resource usage',
                    'suggestion': 'Consider reducing allocated resources',
                    'potential_savings': '50-70% of current allocation'
                })
            
            # Over-provisioned based on patterns
            if container_name in self.failure_patterns:
                failure_times = self.failure_patterns[container_name]
                if len(failure_times) > 3:
                    # Frequent failures suggest wrong resource allocation
                    optimizations.append({
                        'container': container_name,
                        'type': 'stability_improvement',
                        'reason': f'Frequent failures ({len(failure_times)} incidents)',
                        'suggestion': 'Review resource limits and health checks',
                        'priority': 'high'
                    })
        
        return optimizations


async def main():
    """Main entry point for neural health monitoring"""
    monitor = NeuralHealthMonitor()
    
    # Start continuous learning
    learning_task = asyncio.create_task(monitor.continuous_learning_loop())
    
    try:
        while True:
            # Generate and display insights every 5 minutes
            await asyncio.sleep(300)
            
            insights = monitor.generate_insights_report()
            
            print("\n" + "="*80)
            print(f"Neural Health Monitor Insights - {insights['timestamp']}")
            print("="*80)
            
            # Health Summary
            print("\n📊 Current Health Status:")
            for container, health in insights['health_summary'].items():
                status = 'healthy' if health['healthy'] > 0.5 else \
                        'warning' if health['warning'] > 0.5 else \
                        'critical' if health['critical'] > 0.5 else 'failed'
                anomaly = " [ANOMALY]" if health.get('anomaly') else ""
                print(f"  {container}: {status.upper()}{anomaly}")
            
            # Recommendations
            if insights['recommendations']:
                print("\n💡 Recommended Actions:")
                for rec in insights['recommendations']:
                    print(f"\n  Container: {rec['container']}")
                    print(f"  Action: {rec['recommendation']['primary_action']}")
                    print(f"  Confidence: {rec['recommendation']['confidence']:.2%}")
                    print(f"  Reasoning: {rec['recommendation']['reasoning']}")
            
            # Predicted Issues
            if insights['predicted_issues']:
                print("\n⚠️  Predicted Issues:")
                for issue in insights['predicted_issues']:
                    print(f"\n  Container: {issue['container']}")
                    print(f"  Issue: {issue['issue']}")
                    print(f"  Timeframe: {issue['timeframe']}")
                    print(f"  Confidence: {issue['confidence']:.2%}")
                    print(f"  Trend: {issue['trend']}")
            
            # Optimization Opportunities
            if insights['optimization_opportunities']:
                print("\n🔧 Optimization Opportunities:")
                for opt in insights['optimization_opportunities']:
                    print(f"\n  Container: {opt['container']}")
                    print(f"  Type: {opt['type']}")
                    print(f"  Suggestion: {opt['suggestion']}")
            
            print("\n" + "="*80)
            
    except KeyboardInterrupt:
        logger.info("Shutting down neural health monitor")
        learning_task.cancel()
        await learning_task


if __name__ == "__main__":
    asyncio.run(main())