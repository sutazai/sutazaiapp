"""
Continuous Learning Pipeline for SutazAI
Implements incremental learning and model adaptation for AI agents
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import os
from pathlib import Path
import asyncio
import aiohttp
import sqlite3
import time
import hashlib
from collections import defaultdict, deque
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """Types of continuous learning strategies"""
    INCREMENTAL = "incremental"
    REPLAY = "replay"
    ELASTIC_WEIGHT_CONSOLIDATION = "ewc"
    PROGRESSIVE_NEURAL_NETWORKS = "progressive"
    ADAPTATION = "adaptation"
    META_LEARNING = "meta_learning"

class LearningMode(Enum):
    """Learning modes"""
    ONLINE = "online"
    BATCH = "batch"
    STREAMING = "streaming"
    TRIGGERED = "triggered"

@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning"""
    strategy: LearningStrategy = LearningStrategy.INCREMENTAL
    mode: LearningMode = LearningMode.ONLINE
    
    # Learning parameters
    learning_rate: float = 1e-4
    adaptation_rate: float = 0.1
    memory_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 100
    
    # Quality control
    quality_threshold: float = 0.7
    feedback_weight: float = 0.3
    confidence_threshold: float = 0.8
    
    # Memory management
    memory_retention_ratio: float = 0.8
    forgetting_rate: float = 0.01
    importance_sampling: bool = True
    
    # Meta-learning parameters
    meta_learning_rate: float = 1e-3
    adaptation_steps: int = 5
    inner_loop_steps: int = 3
    
    # Performance monitoring
    evaluation_frequency: int = 1000
    performance_window: int = 100
    drift_detection_threshold: float = 0.1
    
    # Storage settings
    database_path: str = "continuous_learning.db"
    checkpoint_frequency: int = 1000
    max_checkpoints: int = 10

class ExperienceBuffer:
    """Buffer for storing and managing learning experiences"""
    
    def __init__(self, max_size: int = 10000, importance_sampling: bool = True):
        self.max_size = max_size
        self.importance_sampling = importance_sampling
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.total_priority = 0.0
        self.lock = threading.Lock()
    
    def add_experience(self, prompt: str, response: str, feedback: float, 
                      context: Dict[str, Any] = None, importance: float = 1.0):
        """Add a learning experience to the buffer"""
        with self.lock:
            experience = {
                'prompt': prompt,
                'response': response,
                'feedback': feedback,
                'context': context or {},
                'timestamp': time.time(),
                'importance': importance,
                'id': hashlib.md5(f"{prompt}{response}{time.time()}".encode()).hexdigest()
            }
            
            self.buffer.append(experience)
            
            if self.importance_sampling:
                priority = max(abs(feedback), 0.1)  # Minimum priority
                self.priorities.append(priority)
                self.total_priority += priority
                
                # Remove excess priority if buffer is full
                if len(self.priorities) > len(self.buffer):
                    old_priority = self.priorities.popleft()
                    self.total_priority -= old_priority
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of experiences"""
        with self.lock:
            if len(self.buffer) == 0:
                return []
            
            batch_size = min(batch_size, len(self.buffer))
            
            if self.importance_sampling and len(self.priorities) > 0:
                # Importance sampling
                probabilities = [p / self.total_priority for p in self.priorities]
                indices = np.random.choice(
                    len(self.buffer), 
                    size=batch_size, 
                    p=probabilities, 
                    replace=False
                )
                return [self.buffer[i] for i in indices]
            else:
                # Uniform sampling
                indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
                return [self.buffer[i] for i in indices]
    
    def get_recent_experiences(self, count: int) -> List[Dict[str, Any]]:
        """Get most recent experiences"""
        with self.lock:
            return list(self.buffer)[-count:]
    
    def clear_old_experiences(self, max_age: float = 86400):  # 24 hours
        """Clear experiences older than max_age seconds"""
        with self.lock:
            current_time = time.time()
            new_buffer = deque(maxlen=self.max_size)
            new_priorities = deque(maxlen=self.max_size)
            new_total_priority = 0.0
            
            for i, exp in enumerate(self.buffer):
                if current_time - exp['timestamp'] <= max_age:
                    new_buffer.append(exp)
                    if i < len(self.priorities):
                        priority = self.priorities[i]
                        new_priorities.append(priority)
                        new_total_priority += priority
            
            self.buffer = new_buffer
            self.priorities = new_priorities
            self.total_priority = new_total_priority
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)

class PerformanceTracker:
    """Tracks model performance over time"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.feedback_history = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def record_performance(self, confidence: float, feedback: float, 
                          additional_metrics: Dict[str, float] = None):
        """Record performance metrics"""
        with self.lock:
            timestamp = time.time()
            
            performance_score = 0.7 * confidence + 0.3 * max(feedback, 0)
            
            self.performance_history.append({
                'timestamp': timestamp,
                'performance': performance_score,
                'confidence': confidence,
                'feedback': feedback,
                'additional': additional_metrics or {}
            })
            
            self.confidence_history.append(confidence)
            self.feedback_history.append(feedback)
    
    def get_current_performance(self) -> float:
        """Get current average performance"""
        with self.lock:
            if not self.performance_history:
                return 0.5
            
            recent_scores = [p['performance'] for p in list(self.performance_history)[-10:]]
            return np.mean(recent_scores)
    
    def detect_performance_drift(self, threshold: float = 0.1) -> Tuple[bool, float]:
        """Detect if performance has drifted significantly"""
        with self.lock:
            if len(self.performance_history) < 20:
                return False, 0.0
            
            # Compare recent performance to baseline
            recent_performance = np.mean([
                p['performance'] for p in list(self.performance_history)[-10:]
            ])
            
            baseline_performance = np.mean([
                p['performance'] for p in list(self.performance_history)[:10]
            ])
            
            drift = abs(recent_performance - baseline_performance)
            
            return drift > threshold, drift
    
    def get_performance_trend(self) -> str:
        """Get performance trend (improving, declining, stable)"""
        with self.lock:
            if len(self.performance_history) < 10:
                return "insufficient_data"
            
            recent_avg = np.mean([
                p['performance'] for p in list(self.performance_history)[-5:]
            ])
            
            earlier_avg = np.mean([
                p['performance'] for p in list(self.performance_history)[-10:-5]
            ])
            
            if recent_avg > earlier_avg + 0.05:
                return "improving"
            elif recent_avg < earlier_avg - 0.05:
                return "declining"
            else:
                return "stable"

class ContinuousLearner:
    """Main continuous learning orchestrator"""
    
    def __init__(self, config: ContinuousLearningConfig = None, 
                 ollama_host: str = "http://localhost:11434"):
        self.config = config or ContinuousLearningConfig()
        self.ollama_host = ollama_host
        self.session = None
        
        # Core components
        self.experience_buffer = ExperienceBuffer(
            self.config.memory_size, 
            self.config.importance_sampling
        )
        self.performance_tracker = PerformanceTracker(self.config.performance_window)
        
        # Learning state
        self.learning_step = 0
        self.last_update = time.time()
        self.adaptation_weights = {}
        self.meta_parameters = {}
        
        # Database for persistence
        self.db_path = self.config.database_path
        self._init_database()
        
        # Background learning thread
        self.learning_thread = None
        self.learning_active = False
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                response TEXT,
                feedback REAL,
                context TEXT,
                timestamp REAL,
                importance REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_log (
                timestamp REAL,
                performance REAL,
                confidence REAL,
                feedback REAL,
                additional_metrics TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def initialize(self):
        """Initialize the continuous learner"""
        self.session = aiohttp.ClientSession()
        
        # Load previous learning state
        self._load_learning_state()
        
        # Start background learning if in online mode
        if self.config.mode == LearningMode.ONLINE:
            self._start_background_learning()
        
        logger.info("Continuous learner initialized")
    
    def _load_learning_state(self):
        """Load previous learning state from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load adaptation weights
            cursor.execute("SELECT value FROM learning_state WHERE key = 'adaptation_weights'")
            result = cursor.fetchone()
            if result:
                self.adaptation_weights = json.loads(result[0])
            
            # Load meta parameters
            cursor.execute("SELECT value FROM learning_state WHERE key = 'meta_parameters'")
            result = cursor.fetchone()
            if result:
                self.meta_parameters = json.loads(result[0])
            
            # Load learning step
            cursor.execute("SELECT value FROM learning_state WHERE key = 'learning_step'")
            result = cursor.fetchone()
            if result:
                self.learning_step = int(result[0])
            
            conn.close()
            logger.info("Loaded previous learning state")
            
        except Exception as e:
            logger.warning(f"Could not load learning state: {e}")
    
    def _save_learning_state(self):
        """Save current learning state to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save adaptation weights
            cursor.execute(
                "INSERT OR REPLACE INTO learning_state (key, value) VALUES (?, ?)",
                ('adaptation_weights', json.dumps(self.adaptation_weights))
            )
            
            # Save meta parameters
            cursor.execute(
                "INSERT OR REPLACE INTO learning_state (key, value) VALUES (?, ?)",
                ('meta_parameters', json.dumps(self.meta_parameters))
            )
            
            # Save learning step
            cursor.execute(
                "INSERT OR REPLACE INTO learning_state (key, value) VALUES (?, ?)",
                ('learning_step', str(self.learning_step))
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Could not save learning state: {e}")
    
    async def add_experience(self, prompt: str, response: str, feedback: float,
                           context: Dict[str, Any] = None, model_name: str = None):
        """Add a new learning experience"""
        # Calculate importance based on feedback and novelty
        importance = self._calculate_importance(prompt, response, feedback)
        
        # Add to experience buffer
        self.experience_buffer.add_experience(
            prompt, response, feedback, context, importance
        )
        
        # Record performance
        confidence = context.get('confidence', 0.5) if context else 0.5
        self.performance_tracker.record_performance(confidence, feedback)
        
        # Store in database for persistence
        await self._store_experience_db(prompt, response, feedback, context, importance)
        
        # Trigger learning if conditions are met
        if self._should_trigger_learning():
            await self._trigger_learning_update()
        
        logger.debug(f"Added learning experience with feedback {feedback}")
    
    def _calculate_importance(self, prompt: str, response: str, feedback: float) -> float:
        """Calculate importance score for an experience"""
        # Base importance on feedback magnitude
        feedback_importance = abs(feedback)
        
        # Add novelty factor (simplified)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        novelty_factor = 1.0  # Could implement proper novelty detection
        
        # Combine factors
        importance = 0.7 * feedback_importance + 0.3 * novelty_factor
        
        return max(0.1, min(2.0, importance))  # Clamp to reasonable range
    
    async def _store_experience_db(self, prompt: str, response: str, feedback: float,
                                 context: Dict[str, Any], importance: float):
        """Store experience in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            experience_id = hashlib.md5(f"{prompt}{response}{time.time()}".encode()).hexdigest()
            
            cursor.execute('''
                INSERT INTO experiences 
                (id, prompt, response, feedback, context, timestamp, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                experience_id, prompt, response, feedback,
                json.dumps(context or {}), time.time(), importance
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Could not store experience in DB: {e}")
    
    def _should_trigger_learning(self) -> bool:
        """Determine if learning should be triggered"""
        if self.config.mode == LearningMode.ONLINE:
            return self.learning_step % self.config.update_frequency == 0
        elif self.config.mode == LearningMode.TRIGGERED:
            # Check for performance drift
            drift_detected, _ = self.performance_tracker.detect_performance_drift()
            return drift_detected
        elif self.config.mode == LearningMode.BATCH:
            return self.experience_buffer.size() >= self.config.batch_size
        else:
            return False
    
    async def _trigger_learning_update(self):
        """Trigger a learning update"""
        if self.config.strategy == LearningStrategy.INCREMENTAL:
            await self._incremental_learning()
        elif self.config.strategy == LearningStrategy.ADAPTATION:
            await self._adaptation_learning()
        elif self.config.strategy == LearningStrategy.META_LEARNING:
            await self._meta_learning()
        else:
            await self._incremental_learning()  # Default
        
        self.learning_step += 1
        self._save_learning_state()
    
    async def _incremental_learning(self):
        """Perform incremental learning update"""
        # Sample experiences for learning
        batch = self.experience_buffer.sample_batch(self.config.batch_size)
        
        if not batch:
            return
        
        # Process batch for learning
        learning_data = []
        for exp in batch:
            if exp['feedback'] > self.config.quality_threshold:
                learning_data.append({
                    'prompt': exp['prompt'],
                    'target_response': exp['response'],
                    'feedback': exp['feedback']
                })
        
        if learning_data:
            await self._apply_learning_update(learning_data)
            logger.info(f"Applied incremental learning with {len(learning_data)} examples")
    
    async def _adaptation_learning(self):
        """Perform adaptation-based learning"""
        # Get recent poor-performing experiences
        recent_experiences = self.experience_buffer.get_recent_experiences(50)
        
        adaptation_data = [
            exp for exp in recent_experiences 
            if exp['feedback'] < 0  # Negative feedback indicates need for adaptation
        ]
        
        if adaptation_data:
            # Update adaptation weights based on feedback
            for exp in adaptation_data:
                domain = self._classify_domain(exp['prompt'])
                
                if domain not in self.adaptation_weights:
                    self.adaptation_weights[domain] = 1.0
                
                # Adjust weight based on feedback
                adjustment = self.config.adaptation_rate * abs(exp['feedback'])
                self.adaptation_weights[domain] = max(0.1, 
                    self.adaptation_weights[domain] - adjustment
                )
            
            logger.info(f"Updated adaptation weights for {len(adaptation_data)} experiences")
    
    async def _meta_learning(self):
        """Perform meta-learning update"""
        # Sample diverse experiences for meta-learning
        batch = self.experience_buffer.sample_batch(self.config.batch_size)
        
        if len(batch) < self.config.adaptation_steps:
            return
        
        # Group experiences by domain/task
        domain_groups = defaultdict(list)
        for exp in batch:
            domain = self._classify_domain(exp['prompt'])
            domain_groups[domain].append(exp)
        
        # Meta-learning update for each domain
        for domain, experiences in domain_groups.items():
            if len(experiences) >= self.config.inner_loop_steps:
                await self._meta_update_domain(domain, experiences)
        
        logger.info(f"Applied meta-learning updates for {len(domain_groups)} domains")
    
    async def _meta_update_domain(self, domain: str, experiences: List[Dict[str, Any]]):
        """Apply meta-learning update for a specific domain"""
        # Simplified meta-learning: adjust domain-specific parameters
        if domain not in self.meta_parameters:
            self.meta_parameters[domain] = {
                'temperature': 0.7,
                'response_length_preference': 1.0,
                'confidence_bias': 0.0
            }
        
        # Calculate performance for this domain
        domain_feedback = [exp['feedback'] for exp in experiences]
        avg_feedback = np.mean(domain_feedback)
        
        # Adjust parameters based on feedback
        params = self.meta_parameters[domain]
        
        if avg_feedback < 0:  # Poor performance
            params['temperature'] = max(0.1, params['temperature'] - 0.1)
            params['confidence_bias'] = max(-0.5, params['confidence_bias'] - 0.1)
        else:  # Good performance
            params['temperature'] = min(1.0, params['temperature'] + 0.05)
            params['confidence_bias'] = min(0.5, params['confidence_bias'] + 0.05)
    
    def _classify_domain(self, prompt: str) -> str:
        """Classify prompt into domain categories"""
        prompt_lower = prompt.lower()
        
        domain_keywords = {
            'code': ['python', 'javascript', 'function', 'class', 'programming', 'debug'],
            'math': ['calculate', 'equation', 'formula', 'mathematics', 'solve'],
            'analysis': ['analyze', 'explain', 'describe', 'compare', 'evaluate'],
            'creative': ['write', 'story', 'creative', 'generate', 'imagine'],
            'technical': ['system', 'architecture', 'infrastructure', 'deployment']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    async def _apply_learning_update(self, learning_data: List[Dict[str, Any]]):
        """Apply learning update to models (simplified for Ollama)"""
        # For Ollama models, we can't directly update weights
        # Instead, we can adjust generation parameters and prompt strategies
        
        # Analyze learning data to extract insights
        insights = self._extract_learning_insights(learning_data)
        
        # Update internal parameters based on insights
        for domain, insight in insights.items():
            if domain not in self.adaptation_weights:
                self.adaptation_weights[domain] = 1.0
            
            # Adjust based on success patterns
            if insight['success_rate'] > 0.8:
                self.adaptation_weights[domain] *= 1.1  # Increase confidence
            elif insight['success_rate'] < 0.3:
                self.adaptation_weights[domain] *= 0.9  # Decrease confidence
        
        logger.debug(f"Updated learning parameters based on {len(learning_data)} examples")
    
    def _extract_learning_insights(self, learning_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Extract insights from learning data"""
        domain_insights = defaultdict(lambda: {'examples': [], 'success_rate': 0.5})
        
        for data in learning_data:
            domain = self._classify_domain(data['prompt'])
            domain_insights[domain]['examples'].append(data)
        
        # Calculate success rates and patterns
        for domain, insight in domain_insights.items():
            examples = insight['examples']
            if examples:
                success_rate = sum(1 for ex in examples if ex['feedback'] > 0) / len(examples)
                insight['success_rate'] = success_rate
                insight['avg_feedback'] = np.mean([ex['feedback'] for ex in examples])
        
        return dict(domain_insights)
    
    def _start_background_learning(self):
        """Start background learning thread"""
        self.learning_active = True
        self.learning_thread = threading.Thread(target=self._background_learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        logger.info("Started background learning thread")
    
    def _background_learning_loop(self):
        """Background learning loop"""
        while self.learning_active:
            try:
                # Check if learning update is needed
                if self._should_trigger_learning():
                    # Run learning update in thread pool
                    future = self.executor.submit(
                        asyncio.run, self._trigger_learning_update()
                    )
                    future.result(timeout=30)  # Wait up to 30 seconds
                
                # Clean up old experiences periodically
                if self.learning_step % 1000 == 0:
                    self.experience_buffer.clear_old_experiences()
                
                time.sleep(1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in background learning: {e}")
                time.sleep(5)  # Wait longer after error
    
    def get_adaptation_parameters(self, prompt: str) -> Dict[str, Any]:
        """Get adaptation parameters for a given prompt"""
        domain = self._classify_domain(prompt)
        
        # Base parameters
        params = {
            'temperature': 0.7,
            'confidence_adjustment': 0.0,
            'response_style': 'default'
        }
        
        # Apply domain-specific adaptations
        if domain in self.adaptation_weights:
            weight = self.adaptation_weights[domain]
            params['confidence_adjustment'] = (weight - 1.0) * 0.1
        
        # Apply meta-learning parameters
        if domain in self.meta_parameters:
            meta_params = self.meta_parameters[domain]
            params['temperature'] = meta_params.get('temperature', params['temperature'])
            params['confidence_adjustment'] += meta_params.get('confidence_bias', 0.0)
        
        return params
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        stats = {
            'learning_step': self.learning_step,
            'experience_buffer_size': self.experience_buffer.size(),
            'current_performance': self.performance_tracker.get_current_performance(),
            'performance_trend': self.performance_tracker.get_performance_trend(),
            'adaptation_weights': dict(self.adaptation_weights),
            'meta_parameters': dict(self.meta_parameters),
            'learning_active': self.learning_active
        }
        
        # Check for performance drift
        drift_detected, drift_magnitude = self.performance_tracker.detect_performance_drift()
        stats['drift_detected'] = drift_detected
        stats['drift_magnitude'] = drift_magnitude
        
        return stats
    
    async def manual_learning_update(self):
        """Manually trigger a learning update"""
        await self._trigger_learning_update()
        logger.info("Manual learning update completed")
    
    def reset_learning_state(self):
        """Reset learning state (useful for testing)"""
        self.adaptation_weights.clear()
        self.meta_parameters.clear()
        self.learning_step = 0
        self._save_learning_state()
        logger.info("Learning state reset")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.learning_active = False
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        
        if self.session:
            await self.session.close()
        
        self._save_learning_state()
        logger.info("Continuous learner cleaned up")

class LearningAnalyzer:
    """Analyzes continuous learning performance and provides insights"""
    
    def __init__(self, db_path: str = "continuous_learning.db"):
        self.db_path = db_path
    
    def analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze learning progress from stored data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get performance over time
            cursor.execute('''
                SELECT timestamp, performance, confidence, feedback 
                FROM performance_log 
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''')
            
            performance_data = cursor.fetchall()
            
            if not performance_data:
                conn.close()
                return {'status': 'no_data'}
            
            # Analyze trends
            timestamps = [row[0] for row in performance_data]
            performances = [row[1] for row in performance_data]
            confidences = [row[2] for row in performance_data]
            feedbacks = [row[3] for row in performance_data]
            
            analysis = {
                'total_interactions': len(performance_data),
                'avg_performance': np.mean(performances),
                'avg_confidence': np.mean(confidences),
                'avg_feedback': np.mean(feedbacks),
                'performance_std': np.std(performances),
                'learning_trend': self._calculate_trend(performances),
                'stability_score': self._calculate_stability(performances),
                'improvement_rate': self._calculate_improvement_rate(performances, timestamps)
            }
            
            # Domain-specific analysis
            cursor.execute('''
                SELECT context FROM experiences 
                WHERE context IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT 500
            ''')
            
            context_data = cursor.fetchall()
            domain_analysis = self._analyze_domain_performance(context_data)
            analysis['domain_performance'] = domain_analysis
            
            conn.close()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing learning progress: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_trend(self, performances: List[float]) -> str:
        """Calculate overall performance trend"""
        if len(performances) < 10:
            return 'insufficient_data'
        
        # Compare first and last quartiles
        first_quartile = np.mean(performances[-len(performances)//4:])
        last_quartile = np.mean(performances[:len(performances)//4])
        
        if first_quartile > last_quartile + 0.05:
            return 'improving'
        elif first_quartile < last_quartile - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_stability(self, performances: List[float]) -> float:
        """Calculate stability score (lower variance = more stable)"""
        if len(performances) < 2:
            return 0.0
        
        variance = np.var(performances)
        # Convert to stability score (0-1, higher is more stable)
        stability = max(0.0, 1.0 - variance)
        return stability
    
    def _calculate_improvement_rate(self, performances: List[float], 
                                  timestamps: List[float]) -> float:
        """Calculate rate of improvement over time"""
        if len(performances) < 10:
            return 0.0
        
        # Linear regression to find improvement slope
        x = np.array(timestamps)
        y = np.array(performances)
        
        # Normalize timestamps
        x = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x
        
        # Calculate slope
        coef = np.polyfit(x, y, 1)[0]
        return float(coef)
    
    def _analyze_domain_performance(self, context_data: List[Tuple]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by domain"""
        domain_stats = defaultdict(list)
        
        for row in context_data:
            try:
                context = json.loads(row[0]) if row[0] else {}
                domain = context.get('domain', 'general')
                confidence = context.get('confidence', 0.5)
                domain_stats[domain].append(confidence)
            except:
                continue
        
        domain_analysis = {}
        for domain, confidences in domain_stats.items():
            if confidences:
                domain_analysis[domain] = {
                    'avg_confidence': np.mean(confidences),
                    'stability': 1.0 - np.std(confidences),
                    'sample_count': len(confidences)
                }
        
        return domain_analysis
    
    def generate_learning_report(self) -> str:
        """Generate a comprehensive learning report"""
        analysis = self.analyze_learning_progress()
        
        if analysis.get('status') == 'no_data':
            return "No learning data available for analysis."
        
        if analysis.get('status') == 'error':
            return f"Error generating report: {analysis.get('message')}"
        
        report = []
        report.append("# Continuous Learning Analysis Report")
        report.append("=" * 50)
        
        # Overall performance
        report.append(f"\n## Overall Performance")
        report.append(f"Total Interactions: {analysis['total_interactions']}")
        report.append(f"Average Performance: {analysis['avg_performance']:.3f}")
        report.append(f"Average Confidence: {analysis['avg_confidence']:.3f}")
        report.append(f"Average Feedback: {analysis['avg_feedback']:.3f}")
        report.append(f"Learning Trend: {analysis['learning_trend']}")
        report.append(f"Stability Score: {analysis['stability_score']:.3f}")
        report.append(f"Improvement Rate: {analysis['improvement_rate']:.6f}")
        
        # Domain performance
        if 'domain_performance' in analysis and analysis['domain_performance']:
            report.append(f"\n## Domain Performance")
            for domain, stats in analysis['domain_performance'].items():
                report.append(f"### {domain.title()}")
                report.append(f"  Average Confidence: {stats['avg_confidence']:.3f}")
                report.append(f"  Stability: {stats['stability']:.3f}")
                report.append(f"  Sample Count: {stats['sample_count']}")
        
        return "\n".join(report)

# Factory function for easy integration
async def create_continuous_learner(config: ContinuousLearningConfig = None) -> ContinuousLearner:
    """Create and initialize a continuous learner"""
    if config is None:
        config = ContinuousLearningConfig()
    
    learner = ContinuousLearner(config)
    await learner.initialize()
    
    return learner

# Example usage
async def example_continuous_learning():
    """Example usage of continuous learning"""
    # Create learner
    config = ContinuousLearningConfig(
        strategy=LearningStrategy.ADAPTATION,
        mode=LearningMode.ONLINE,
        update_frequency=10
    )
    
    learner = await create_continuous_learner(config)
    
    # Simulate learning experiences
    experiences = [
        ("What is machine learning?", "ML is a subset of AI...", 0.8),
        ("Write a Python function", "def example():\n    pass", 0.6),
        ("Explain quantum computing", "Quantum computing uses qubits...", -0.2),  # Negative feedback
        ("Debug this code", "The error is in line 5...", 0.9),
    ]
    
    for prompt, response, feedback in experiences:
        await learner.add_experience(
            prompt, response, feedback,
            context={'confidence': abs(feedback)}
        )
        
        # Small delay to simulate real usage
        await asyncio.sleep(0.1)
    
    # Get statistics
    stats = learner.get_learning_statistics()
    print("Learning Statistics:", json.dumps(stats, indent=2))
    
    # Generate learning report
    analyzer = LearningAnalyzer(learner.config.database_path)
    report = analyzer.generate_learning_report()
    print("\nLearning Report:\n", report)
    
    # Cleanup
    await learner.cleanup()
    
    return stats

if __name__ == "__main__":
    # Run example
    async def main():
        stats = await example_continuous_learning()
        return stats
    
    # asyncio.run(main())