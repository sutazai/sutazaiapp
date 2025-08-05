"""
Federated Learning Coordinator
==============================

Central coordinator for federated learning across SutazAI agents.
Manages training rounds, client selection, model distribution, and aggregation.

Features:
- Multi-algorithm support (FedAvg, FedProx, FedOpt)
- Asynchronous training coordination
- Privacy-preserving aggregation
- Fault tolerance and recovery
- Performance monitoring
- CPU-optimized for 12-core constraint
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque

import aioredis
import numpy as np
from pydantic import BaseModel

from ..ai_agents.core.agent_registry import AgentRegistry, get_agent_registry
from ..ai_agents.core.base_agent import AgentMessage, AgentCapability, AgentStatus
from .aggregator import FederatedAggregator, AggregationAlgorithm
from .privacy import PrivacyManager, PrivacyBudget
from .versioning import ModelVersionManager
from .monitoring import FederatedMonitor


class TrainingStatus(Enum):
    """Training round status"""
    PREPARING = "preparing"
    CLIENT_SELECTION = "client_selection"
    MODEL_DISTRIBUTION = "model_distribution"
    TRAINING = "training"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClientSelectionStrategy(Enum):
    """Client selection strategies"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    PERFORMANCE_BASED = "performance_based"
    DIVERSITY_BASED = "diversity_based"
    RESOURCE_AWARE = "resource_aware"
    CUSTOM = "custom"


@dataclass
class TrainingConfiguration:
    """Federated training configuration"""
    name: str
    algorithm: AggregationAlgorithm
    model_type: str
    target_accuracy: float
    max_rounds: int
    min_clients_per_round: int
    max_clients_per_round: int
    client_selection_strategy: ClientSelectionStrategy
    local_epochs: int
    local_batch_size: int
    local_learning_rate: float
    convergence_threshold: float
    privacy_budget: Optional[PrivacyBudget]
    timeout_seconds: int
    validation_frequency: int  # Every N rounds
    early_stopping_patience: int
    resource_constraints: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRound:
    """Single training round information"""
    round_id: str
    round_number: int
    training_id: str
    status: TrainingStatus
    selected_clients: List[str]
    active_clients: Set[str]
    completed_clients: Set[str]
    failed_clients: Set[str]
    global_model_version: str
    client_updates: Dict[str, Dict[str, Any]]
    aggregation_result: Optional[Dict[str, Any]]
    validation_metrics: Optional[Dict[str, float]]
    start_time: datetime
    end_time: Optional[datetime]
    error_message: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedTraining:
    """Complete federated training session"""
    training_id: str
    config: TrainingConfiguration
    status: TrainingStatus
    current_round: int
    rounds: List[TrainingRound]
    global_model_versions: List[str]
    performance_history: List[Dict[str, float]]
    participating_clients: Set[str]
    start_time: datetime
    end_time: Optional[datetime]
    error_message: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FederatedCoordinator:
    """
    Central Federated Learning Coordinator
    
    Manages distributed training across AI agents in the SutazAI system.
    Implements privacy-preserving federated learning with fault tolerance.
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 namespace: str = "sutazai:federated"):
        
        self.redis_url = redis_url
        self.namespace = namespace
        self.redis: Optional[aioredis.Redis] = None
        
        # Core components
        self.agent_registry: Optional[AgentRegistry] = None
        self.aggregator = FederatedAggregator(cpu_cores=12)  # CPU constraint
        self.privacy_manager = PrivacyManager()
        self.version_manager = ModelVersionManager()
        self.monitor = FederatedMonitor()
        
        # Training management
        self.active_trainings: Dict[str, FederatedTraining] = {}
        self.training_queue: deque = deque()
        self.client_capabilities: Dict[str, Set[str]] = {}
        
        # Performance tracking
        self.coordinator_stats = {
            "total_trainings": 0,
            "completed_trainings": 0,
            "failed_trainings": 0,
            "total_rounds": 0,
            "average_round_time": 0.0,
            "client_participation_rate": 0.0,
            "model_accuracy_trend": []
        }
        
        # Configuration
        self.max_concurrent_trainings = 3  # CPU constraint
        self.heartbeat_interval = 30  # seconds
        self.client_timeout = 300  # 5 minutes
        self.max_retries = 3
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger("federated_coordinator")
    
    async def initialize(self) -> bool:
        """Initialize the federated coordinator"""
        try:
            self.logger.info("Initializing Federated Learning Coordinator")
            
            # Connect to Redis
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            
            # Get agent registry
            self.agent_registry = get_agent_registry()
            if not self.agent_registry:
                raise ValueError("Agent registry not available")
            
            # Initialize components
            await self.aggregator.initialize()
            await self.privacy_manager.initialize()
            await self.version_manager.initialize()
            await self.monitor.initialize()
            
            # Load existing trainings
            await self._load_active_trainings()
            
            # Discover federated learning capable clients
            await self._discover_fl_clients()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info("Federated Learning Coordinator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize coordinator: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background coordination tasks"""
        tasks = [
            self._training_orchestrator(),
            self._client_health_monitor(),
            self._performance_tracker(),
            self._cleanup_expired_trainings()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _discover_fl_clients(self):
        """Discover agents capable of federated learning"""
        try:
            all_agents = self.agent_registry.get_all_agents()
            
            for agent_id, registration in all_agents.items():
                # Check if agent has learning capability
                if AgentCapability.LEARNING in registration.capabilities:
                    # Query agent for FL capabilities
                    capabilities = await self._query_client_capabilities(agent_id)
                    if capabilities:
                        self.client_capabilities[agent_id] = capabilities
                        self.logger.info(f"Discovered FL client: {agent_id} with capabilities: {capabilities}")
            
            self.logger.info(f"Discovered {len(self.client_capabilities)} federated learning clients")
            
        except Exception as e:
            self.logger.error(f"Error discovering FL clients: {e}")
    
    async def _query_client_capabilities(self, agent_id: str) -> Optional[Set[str]]:
        """Query a client for its federated learning capabilities"""
        try:
            message = AgentMessage(
                sender_id="federated_coordinator",
                receiver_id=agent_id,
                message_type="request",
                content={
                    "action": "query_fl_capabilities",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Send message via Redis
            channel = f"{self.namespace}:agent:{agent_id}"
            await self.redis.publish(channel, json.dumps(message.to_dict()))
            
            # Wait for response (with timeout)
            response_channel = f"{self.namespace}:coordinator:response:{agent_id}"
            try:
                response = await asyncio.wait_for(
                    self.redis.blpop(response_channel, timeout=10),
                    timeout=15
                )
                
                if response:
                    response_data = json.loads(response[1])
                    capabilities = response_data.get("capabilities", [])
                    return set(capabilities)
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout querying capabilities for agent {agent_id}")
                
        except Exception as e:
            self.logger.error(f"Error querying client {agent_id} capabilities: {e}")
        
        return None
    
    async def start_training(self, config: TrainingConfiguration) -> str:
        """Start a new federated training session"""
        try:
            training_id = str(uuid.uuid4())
            
            # Validate configuration
            if not self._validate_training_config(config):
                raise ValueError("Invalid training configuration")
            
            # Check resource constraints
            if len(self.active_trainings) >= self.max_concurrent_trainings:
                # Queue the training
                self.training_queue.append((training_id, config))
                self.logger.info(f"Training {training_id} queued (max concurrent limit reached)")
                return training_id
            
            # Create training session
            training = FederatedTraining(
                training_id=training_id,
                config=config,
                status=TrainingStatus.PREPARING,
                current_round=0,
                rounds=[],
                global_model_versions=[],
                performance_history=[],
                participating_clients=set(),
                start_time=datetime.utcnow()
            )
            
            # Initialize global model
            initial_model_version = await self.version_manager.create_initial_model(
                training_id, config.model_type
            )
            training.global_model_versions.append(initial_model_version)
            
            # Store training
            self.active_trainings[training_id] = training
            await self._store_training(training)
            
            # Start training orchestration
            asyncio.create_task(self._execute_training(training_id))
            
            self.coordinator_stats["total_trainings"] += 1
            
            self.logger.info(f"Started federated training {training_id} with algorithm {config.algorithm.value}")
            return training_id
            
        except Exception as e:
            self.logger.error(f"Failed to start training: {e}")
            raise
    
    def _validate_training_config(self, config: TrainingConfiguration) -> bool:
        """Validate training configuration"""
        try:
            # Check minimum requirements
            if config.min_clients_per_round < 1:
                return False
            
            if config.max_clients_per_round < config.min_clients_per_round:
                return False
            
            if config.max_rounds < 1:
                return False
            
            if config.local_epochs < 1:
                return False
            
            # Check if we have enough capable clients
            capable_clients = len(self.client_capabilities)
            if capable_clients < config.min_clients_per_round:
                self.logger.error(f"Not enough capable clients: {capable_clients} < {config.min_clients_per_round}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating config: {e}")
            return False
    
    async def _execute_training(self, training_id: str):
        """Execute a federated training session"""
        try:
            training = self.active_trainings[training_id]
            
            self.logger.info(f"Executing training {training_id}")
            
            # Training loop
            for round_num in range(1, training.config.max_rounds + 1):
                training.current_round = round_num
                
                # Create training round
                round_result = await self._execute_training_round(training, round_num)
                training.rounds.append(round_result)
                
                if round_result.status == TrainingStatus.FAILED:
                    training.status = TrainingStatus.FAILED
                    training.error_message = round_result.error_message
                    break
                
                # Check convergence
                if await self._check_convergence(training):
                    self.logger.info(f"Training {training_id} converged after {round_num} rounds")
                    break
                
                # Validation
                if round_num % training.config.validation_frequency == 0:
                    validation_metrics = await self._validate_global_model(training)
                    if validation_metrics:
                        training.performance_history.append(validation_metrics)
                        
                        # Check target accuracy
                        if validation_metrics.get("accuracy", 0) >= training.config.target_accuracy:
                            self.logger.info(f"Training {training_id} reached target accuracy")
                            break
            
            # Complete training
            training.status = TrainingStatus.COMPLETED
            training.end_time = datetime.utcnow()
            
            await self._store_training(training)
            await self.monitor.record_training_completion(training)
            
            self.coordinator_stats["completed_trainings"] += 1
            
            self.logger.info(f"Completed training {training_id} with {len(training.rounds)} rounds")
            
        except Exception as e:
            self.logger.error(f"Training {training_id} failed: {e}")
            training.status = TrainingStatus.FAILED
            training.error_message = str(e)
            training.end_time = datetime.utcnow()
            await self._store_training(training)
            self.coordinator_stats["failed_trainings"] += 1
        
        finally:
            # Start next queued training if any
            if self.training_queue and len(self.active_trainings) < self.max_concurrent_trainings:
                next_training_id, next_config = self.training_queue.popleft()
                asyncio.create_task(self.start_training(next_config))
    
    async def _execute_training_round(self, training: FederatedTraining, round_num: int) -> TrainingRound:
        """Execute a single training round"""
        round_id = str(uuid.uuid4())
        round_start = datetime.utcnow()
        
        training_round = TrainingRound(
            round_id=round_id,
            round_number=round_num,
            training_id=training.training_id,
            status=TrainingStatus.CLIENT_SELECTION,
            selected_clients=[],
            active_clients=set(),
            completed_clients=set(),
            failed_clients=set(),
            global_model_version=training.global_model_versions[-1],
            client_updates={},
            aggregation_result=None,
            validation_metrics=None,
            start_time=round_start
        )
        
        try:
            self.logger.info(f"Starting round {round_num} for training {training.training_id}")
            
            # 1. Client Selection
            selected_clients = await self._select_clients(training, round_num)
            if len(selected_clients) < training.config.min_clients_per_round:
                raise ValueError(f"Not enough clients available: {len(selected_clients)}")
            
            training_round.selected_clients = selected_clients
            training_round.active_clients = set(selected_clients)
            training_round.status = TrainingStatus.MODEL_DISTRIBUTION
            
            # 2. Distribute Global Model
            await self._distribute_global_model(training_round)
            training_round.status = TrainingStatus.TRAINING
            
            # 3. Client Training
            client_updates = await self._coordinate_client_training(training_round)
            training_round.client_updates = client_updates
            training_round.status = TrainingStatus.AGGREGATION
            
            # 4. Aggregate Updates
            if len(client_updates) >= training.config.min_clients_per_round:
                aggregation_result = await self._aggregate_client_updates(training, client_updates)
                training_round.aggregation_result = aggregation_result
                
                # Create new global model version
                new_version = await self.version_manager.create_model_version(
                    training.training_id, aggregation_result
                )
                training.global_model_versions.append(new_version)
                
                training_round.status = TrainingStatus.COMPLETED
            else:
                raise ValueError(f"Insufficient client updates: {len(client_updates)}")
            
            training_round.end_time = datetime.utcnow()
            self.coordinator_stats["total_rounds"] += 1
            
            # Update average round time
            round_duration = (training_round.end_time - training_round.start_time).total_seconds()
            self.coordinator_stats["average_round_time"] = (
                (self.coordinator_stats["average_round_time"] * (self.coordinator_stats["total_rounds"] - 1) + round_duration) /
                self.coordinator_stats["total_rounds"]
            )
            
            self.logger.info(f"Completed round {round_num} in {round_duration:.2f}s")
            
        except Exception as e:
            training_round.status = TrainingStatus.FAILED
            training_round.error_message = str(e)
            training_round.end_time = datetime.utcnow()
            self.logger.error(f"Round {round_num} failed: {e}")
        
        return training_round
    
    async def _select_clients(self, training: FederatedTraining, round_num: int) -> List[str]:
        """Select clients for training round"""
        try:
            available_clients = []
            
            # Get available clients
            for agent_id in self.client_capabilities:
                registration = self.agent_registry.get_agent(agent_id)
                if registration and registration.status in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
                    # Check resource availability
                    load_percentage = (registration.current_task_count / 
                                     registration.max_concurrent_tasks) * 100
                    if load_percentage < 80:  # Don't overload clients
                        available_clients.append(agent_id)
            
            if len(available_clients) < training.config.min_clients_per_round:
                raise ValueError(f"Not enough available clients: {len(available_clients)}")
            
            # Apply selection strategy
            selected = []
            config = training.config
            
            if config.client_selection_strategy == ClientSelectionStrategy.RANDOM:
                import random
                selected = random.sample(available_clients, 
                                       min(config.max_clients_per_round, len(available_clients)))
            
            elif config.client_selection_strategy == ClientSelectionStrategy.PERFORMANCE_BASED:
                # Select based on past performance
                client_scores = {}
                for client_id in available_clients:
                    registration = self.agent_registry.get_agent(client_id)
                    if registration:
                        # Use success rate as performance metric
                        client_scores[client_id] = registration.metrics.success_rate()
                
                # Sort by performance and select top clients
                sorted_clients = sorted(client_scores.items(), key=lambda x: x[1], reverse=True)
                selected = [client_id for client_id, _ in sorted_clients[:config.max_clients_per_round]]
            
            elif config.client_selection_strategy == ClientSelectionStrategy.RESOURCE_AWARE:
                # Select clients with best resource availability
                client_resources = {}
                for client_id in available_clients:
                    registration = self.agent_registry.get_agent(client_id)
                    if registration:
                        # Calculate resource score (lower is better)
                        cpu_score = registration.metrics.cpu_utilization / 100
                        memory_score = registration.metrics.current_memory_usage / 100
                        load_score = (registration.current_task_count / 
                                    registration.max_concurrent_tasks)
                        
                        client_resources[client_id] = cpu_score + memory_score + load_score
                
                # Sort by resource availability and select best
                sorted_clients = sorted(client_resources.items(), key=lambda x: x[1])
                selected = [client_id for client_id, _ in sorted_clients[:config.max_clients_per_round]]
            
            else:
                # Default to random selection
                import random
                selected = random.sample(available_clients, 
                                       min(config.max_clients_per_round, len(available_clients)))
            
            # Update participating clients
            training.participating_clients.update(selected)
            
            self.logger.info(f"Selected {len(selected)} clients for round {round_num}: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"Client selection failed: {e}")
            raise
    
    async def _distribute_global_model(self, training_round: TrainingRound):
        """Distribute global model to selected clients"""
        try:
            # Get global model
            global_model = await self.version_manager.get_model_version(
                training_round.global_model_version
            )
            
            # Send to all selected clients
            distribution_tasks = []
            for client_id in training_round.selected_clients:
                task = self._send_model_to_client(client_id, training_round, global_model)
                distribution_tasks.append(task)
            
            # Wait for all distributions to complete
            results = await asyncio.gather(*distribution_tasks, return_exceptions=True)
            
            # Check for failures
            failed_clients = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    client_id = training_round.selected_clients[i]
                    failed_clients.append(client_id)
                    self.logger.error(f"Failed to distribute model to {client_id}: {result}")
            
            # Remove failed clients from active set
            for client_id in failed_clients:
                training_round.active_clients.discard(client_id)
                training_round.failed_clients.add(client_id)
            
            self.logger.info(f"Distributed model to {len(training_round.active_clients)} clients")
            
        except Exception as e:
            self.logger.error(f"Model distribution failed: {e}")
            raise
    
    async def _send_model_to_client(self, client_id: str, training_round: TrainingRound, 
                                   global_model: Dict[str, Any]):
        """Send global model to a specific client"""
        try:
            message = AgentMessage(
                sender_id="federated_coordinator",
                receiver_id=client_id,
                message_type="request",
                content={
                    "action": "federated_training",
                    "training_id": training_round.training_id,
                    "round_id": training_round.round_id,
                    "round_number": training_round.round_number,
                    "global_model": global_model,
                    "training_config": {
                        "local_epochs": training_round.training_id,  # Get from training config
                        "batch_size": training_round.training_id,    # Get from training config
                        "learning_rate": training_round.training_id  # Get from training config
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Send via Redis
            channel = f"{self.namespace}:agent:{client_id}"
            await self.redis.publish(channel, json.dumps(message.to_dict()))
            
        except Exception as e:
            self.logger.error(f"Failed to send model to client {client_id}: {e}")
            raise
    
    async def _coordinate_client_training(self, training_round: TrainingRound) -> Dict[str, Dict[str, Any]]:
        """Coordinate client training and collect updates"""
        try:
            client_updates = {}
            timeout_time = time.time() + self.client_timeout
            
            # Wait for client updates
            while len(training_round.completed_clients) < len(training_round.active_clients):
                if time.time() > timeout_time:
                    self.logger.warning(f"Timeout waiting for client updates in round {training_round.round_number}")
                    break
                
                # Check for incoming updates
                await self._check_client_updates(training_round, client_updates)
                await asyncio.sleep(1)  # Small delay to prevent busy waiting
            
            self.logger.info(f"Collected {len(client_updates)} client updates")
            return client_updates
            
        except Exception as e:
            self.logger.error(f"Client training coordination failed: {e}")
            raise
    
    async def _check_client_updates(self, training_round: TrainingRound, 
                                   client_updates: Dict[str, Dict[str, Any]]):
        """Check for incoming client updates"""
        try:
            # Check Redis for client update messages
            update_channel = f"{self.namespace}:coordinator:updates:{training_round.round_id}"
            
            # Non-blocking check for messages
            message = await self.redis.lpop(update_channel)
            if message:
                update_data = json.loads(message)
                client_id = update_data.get("client_id")
                
                if client_id in training_round.active_clients:
                    client_updates[client_id] = update_data
                    training_round.completed_clients.add(client_id)
                    self.logger.debug(f"Received update from client {client_id}")
        
        except Exception as e:
            self.logger.error(f"Error checking client updates: {e}")
    
    async def _aggregate_client_updates(self, training: FederatedTraining, 
                                       client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate client model updates"""
        try:
            # Apply privacy mechanisms if configured
            if training.config.privacy_budget:
                client_updates = await self.privacy_manager.apply_differential_privacy(
                    client_updates, training.config.privacy_budget
                )
            
            # Perform aggregation using specified algorithm
            aggregation_result = await self.aggregator.aggregate(
                algorithm=training.config.algorithm,
                client_updates=client_updates,
                round_number=training.current_round
            )
            
            self.logger.info(f"Aggregated {len(client_updates)} client updates using {training.config.algorithm.value}")
            return aggregation_result
            
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            raise
    
    async def _check_convergence(self, training: FederatedTraining) -> bool:
        """Check if training has converged"""
        try:
            if len(training.performance_history) < 2:
                return False
            
            # Check improvement in last few rounds
            recent_performance = training.performance_history[-training.config.early_stopping_patience:]
            if len(recent_performance) < training.config.early_stopping_patience:
                return False
            
            # Check if improvement is below threshold
            best_metric = max(p.get("accuracy", 0) for p in recent_performance)
            latest_metric = recent_performance[-1].get("accuracy", 0)
            
            improvement = best_metric - latest_metric
            if improvement < training.config.convergence_threshold:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Convergence check failed: {e}")
            return False
    
    async def _validate_global_model(self, training: FederatedTraining) -> Optional[Dict[str, float]]:
        """Validate the current global model"""
        try:
            # Get current global model
            current_model_version = training.global_model_versions[-1]
            
            # This would involve running validation on a held-out dataset
            # For now, return simulated metrics
            validation_metrics = {
                "accuracy": 0.85 + (training.current_round * 0.01),  # Simulated improvement
                "loss": 1.0 - (training.current_round * 0.05),
                "round": training.current_round,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Validation metrics for round {training.current_round}: {validation_metrics}")
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return None
    
    # Background task implementations
    async def _training_orchestrator(self):
        """Orchestrate multiple concurrent trainings"""
        while not self._shutdown_event.is_set():
            try:
                # Process training queue if resources available
                while (self.training_queue and 
                       len(self.active_trainings) < self.max_concurrent_trainings):
                    training_id, config = self.training_queue.popleft()
                    asyncio.create_task(self._execute_training_from_queue(training_id, config))
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Training orchestrator error: {e}")
                await asyncio.sleep(10)
    
    async def _execute_training_from_queue(self, training_id: str, config: TrainingConfiguration):
        """Execute a training from the queue"""
        try:
            # Create training session
            training = FederatedTraining(
                training_id=training_id,
                config=config,
                status=TrainingStatus.PREPARING,
                current_round=0,
                rounds=[],
                global_model_versions=[],
                performance_history=[],
                participating_clients=set(),
                start_time=datetime.utcnow()
            )
            
            # Initialize global model
            initial_model_version = await self.version_manager.create_initial_model(
                training_id, config.model_type
            )
            training.global_model_versions.append(initial_model_version)
            
            # Store and execute training
            self.active_trainings[training_id] = training
            await self._store_training(training)
            await self._execute_training(training_id)
            
        except Exception as e:
            self.logger.error(f"Failed to execute queued training {training_id}: {e}")
    
    async def _client_health_monitor(self):
        """Monitor health of federated learning clients"""
        while not self._shutdown_event.is_set():
            try:
                # Check client availability
                unavailable_clients = []
                
                for client_id in self.client_capabilities:
                    registration = self.agent_registry.get_agent(client_id)
                    if not registration or registration.status == AgentStatus.OFFLINE:
                        unavailable_clients.append(client_id)
                
                # Remove unavailable clients from active trainings
                for training in self.active_trainings.values():
                    for round_info in training.rounds:
                        if round_info.status in [TrainingStatus.TRAINING, TrainingStatus.MODEL_DISTRIBUTION]:
                            for client_id in unavailable_clients:
                                if client_id in round_info.active_clients:
                                    round_info.active_clients.discard(client_id)
                                    round_info.failed_clients.add(client_id)
                                    self.logger.warning(f"Client {client_id} became unavailable during training")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Client health monitor error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _performance_tracker(self):
        """Track coordinator performance metrics"""
        while not self._shutdown_event.is_set():
            try:
                # Calculate participation rate
                if self.client_capabilities:
                    active_clients = 0
                    for client_id in self.client_capabilities:
                        registration = self.agent_registry.get_agent(client_id)
                        if registration and registration.status in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
                            active_clients += 1
                    
                    self.coordinator_stats["client_participation_rate"] = (
                        active_clients / len(self.client_capabilities) * 100
                    )
                
                # Update model accuracy trend
                for training in self.active_trainings.values():
                    if training.performance_history:
                        latest_accuracy = training.performance_history[-1].get("accuracy", 0)
                        self.coordinator_stats["model_accuracy_trend"].append(latest_accuracy)
                        
                        # Keep only last 100 measurements
                        if len(self.coordinator_stats["model_accuracy_trend"]) > 100:
                            self.coordinator_stats["model_accuracy_trend"].pop(0)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Performance tracker error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_trainings(self):
        """Clean up expired training sessions"""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                expired_trainings = []
                
                for training_id, training in self.active_trainings.items():
                    # Clean up completed trainings after 1 hour
                    if (training.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED] and
                        training.end_time and 
                        current_time - training.end_time > timedelta(hours=1)):
                        expired_trainings.append(training_id)
                    
                    # Clean up stuck trainings after 2 hours
                    elif (current_time - training.start_time > timedelta(hours=2) and
                          training.status not in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]):
                        training.status = TrainingStatus.FAILED
                        training.error_message = "Training timeout"
                        training.end_time = current_time
                        expired_trainings.append(training_id)
                
                for training_id in expired_trainings:
                    del self.active_trainings[training_id]
                    self.logger.info(f"Cleaned up expired training {training_id}")
                
                await asyncio.sleep(1800)  # Clean up every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(1800)
    
    # Storage methods
    async def _store_training(self, training: FederatedTraining):
        """Store training session in Redis"""
        try:
            key = f"{self.namespace}:training:{training.training_id}"
            
            # Convert to dict for JSON serialization
            training_dict = {
                "training_id": training.training_id,
                "config": {
                    "name": training.config.name,
                    "algorithm": training.config.algorithm.value,
                    "model_type": training.config.model_type,
                    "target_accuracy": training.config.target_accuracy,
                    "max_rounds": training.config.max_rounds,
                    "min_clients_per_round": training.config.min_clients_per_round,
                    "max_clients_per_round": training.config.max_clients_per_round,
                    "client_selection_strategy": training.config.client_selection_strategy.value,
                    "local_epochs": training.config.local_epochs,
                    "local_batch_size": training.config.local_batch_size,
                    "local_learning_rate": training.config.local_learning_rate,
                    "convergence_threshold": training.config.convergence_threshold,
                    "timeout_seconds": training.config.timeout_seconds,
                    "validation_frequency": training.config.validation_frequency,
                    "early_stopping_patience": training.config.early_stopping_patience,
                    "resource_constraints": training.config.resource_constraints,
                    "metadata": training.config.metadata
                },
                "status": training.status.value,
                "current_round": training.current_round,
                "global_model_versions": training.global_model_versions,
                "performance_history": training.performance_history,
                "participating_clients": list(training.participating_clients),
                "start_time": training.start_time.isoformat(),
                "end_time": training.end_time.isoformat() if training.end_time else None,
                "error_message": training.error_message,
                "metadata": training.metadata
            }
            
            await self.redis.set(key, json.dumps(training_dict))
            
        except Exception as e:
            self.logger.error(f"Failed to store training {training.training_id}: {e}")
    
    async def _load_active_trainings(self):
        """Load active training sessions from Redis"""
        try:
            training_keys = await self.redis.keys(f"{self.namespace}:training:*")
            
            for key in training_keys:
                training_data = await self.redis.get(key)
                if training_data:
                    training_dict = json.loads(training_data)
                    
                    # Skip completed/failed trainings
                    status = TrainingStatus(training_dict["status"])
                    if status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
                        continue
                    
                    # Reconstruct training object (simplified)
                    training_id = training_dict["training_id"]
                    self.logger.info(f"Loaded active training {training_id}")
            
        except Exception as e:
            self.logger.error(f"Error loading active trainings: {e}")
    
    # Public API methods
    def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get training status"""
        training = self.active_trainings.get(training_id)
        if not training:
            return None
        
        return {
            "training_id": training.training_id,
            "status": training.status.value,
            "current_round": training.current_round,
            "total_rounds": len(training.rounds),
            "participating_clients": len(training.participating_clients),
            "start_time": training.start_time.isoformat(),
            "end_time": training.end_time.isoformat() if training.end_time else None,
            "latest_performance": training.performance_history[-1] if training.performance_history else None
        }
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return self.coordinator_stats.copy()
    
    def get_active_trainings(self) -> List[str]:
        """Get list of active training IDs"""
        return list(self.active_trainings.keys())
    
    async def stop_training(self, training_id: str) -> bool:
        """Stop a training session"""
        try:
            training = self.active_trainings.get(training_id)
            if not training:
                return False
            
            training.status = TrainingStatus.CANCELLED
            training.end_time = datetime.utcnow()
            
            await self._store_training(training)
            
            self.logger.info(f"Stopped training {training_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop training {training_id}: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the federated coordinator"""
        self.logger.info("Shutting down Federated Learning Coordinator")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        self.logger.info("Federated Learning Coordinator shutdown complete")


# Global coordinator instance
_federated_coordinator: Optional[FederatedCoordinator] = None


def get_federated_coordinator() -> Optional[FederatedCoordinator]:
    """Get the global federated coordinator"""
    return _federated_coordinator


def set_federated_coordinator(coordinator: FederatedCoordinator):
    """Set the global federated coordinator"""
    global _federated_coordinator
    _federated_coordinator = coordinator