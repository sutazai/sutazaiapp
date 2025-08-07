"""
Federated Learning System Integration
====================================

Complete integration script for deploying federated learning capabilities
across the SutazAI distributed agent system.

This module provides:
- System initialization and configuration
- Component orchestration and coordination
- Agent capability enhancement
- Performance optimization for CPU-only environments
- Integration with existing SutazAI infrastructure
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from .coordinator import FederatedCoordinator, set_federated_coordinator
from .aggregator import FederatedAggregator, AggregationAlgorithm
from .client import FederatedClient, FederatedLearningCapability
from .privacy import PrivacyManager, PrivacyLevel
from .versioning import ModelVersionManager, VersioningConfig
from .monitoring import FederatedMonitor, MetricType, TrainingMetric
from .dashboard import FederatedDashboard, run_dashboard

from ..ai_agents.core.agent_registry import get_agent_registry, set_agent_registry, AgentRegistry
from ..ai_agents.core.base_agent import AgentCapability


class FederatedSystemConfig:
    """Configuration for federated learning system"""
    
    def __init__(self):
        # System configuration
        self.redis_url = "redis://localhost:6379"
        self.namespace = "sutazai:federated"
        
        # Resource constraints (CPU-only, 12 cores)
        self.cpu_cores = 12
        self.max_concurrent_trainings = 3
        self.max_memory_mb = 8192
        
        # Default training parameters
        self.default_privacy_level = PrivacyLevel.MEDIUM
        self.default_aggregation_algorithm = AggregationAlgorithm.FEDAVG
        self.checkpoint_interval = 5
        
        # Storage configuration
        self.model_storage_path = "/opt/sutazaiapp/backend/federated_models"
        self.enable_model_compression = True
        self.enable_auto_cleanup = True
        
        # Dashboard configuration
        self.dashboard_host = "0.0.0.0"
        self.dashboard_port = 8080
        
        # Client selection
        self.min_clients_per_round = 3
        self.max_clients_per_round = 20
        self.client_timeout_seconds = 300
        
        # Privacy settings
        self.enable_differential_privacy = True
        self.enable_secure_aggregation = True
        self.default_privacy_budget = {
            "epsilon": 1.0,
            "delta": 1e-5
        }


class FederatedSystemIntegrator:
    """
    Main integration controller for federated learning system
    
    Orchestrates initialization, configuration, and deployment of all
    federated learning components within the SutazAI ecosystem.
    """
    
    def __init__(self, config: FederatedSystemConfig = None):
        self.config = config or FederatedSystemConfig()
        
        # Core components
        self.coordinator: Optional[FederatedCoordinator] = None
        self.monitor: Optional[FederatedMonitor] = None
        self.version_manager: Optional[ModelVersionManager] = None
        self.dashboard: Optional[FederatedDashboard] = None
        
        # Agent registry
        self.agent_registry: Optional[AgentRegistry] = None
        
        # Enhanced agents with FL capabilities
        self.fl_enhanced_agents: Dict[str, FederatedLearningCapability] = {}
        
        # System state
        self.system_initialized = False
        self.components_started = False
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger("federated_integrator")
    
    async def initialize_system(self) -> bool:
        """Initialize the complete federated learning system"""
        try:
            self.logger.info("Initializing SutazAI Federated Learning System")
            
            # Step 1: Initialize core components
            success = await self._initialize_core_components()
            if not success:
                raise Exception("Failed to initialize core components")
            
            # Step 2: Setup agent registry integration
            await self._setup_agent_registry_integration()
            
            # Step 3: Enhance existing agents with FL capabilities
            await self._enhance_agents_with_federated_learning()
            
            # Step 4: Configure system-wide settings
            await self._configure_system_settings()
            
            # Step 5: Start background services
            await self._start_background_services()
            
            # Step 6: Validate system integrity
            await self._validate_system_integrity()
            
            self.system_initialized = True
            self.logger.info("✅ Federated Learning System initialized successfully")
            
            # Generate deployment report
            await self._generate_deployment_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize federated learning system: {e}")
            await self._cleanup_partial_initialization()
            return False
    
    async def _initialize_core_components(self) -> bool:
        """Initialize core federated learning components"""
        try:
            self.logger.info("Initializing core components...")
            
            # Initialize coordinator
            self.coordinator = FederatedCoordinator(
                redis_url=self.config.redis_url,
                namespace=self.config.namespace
            )
            
            success = await self.coordinator.initialize()
            if not success:
                raise Exception("Failed to initialize coordinator")
            
            set_federated_coordinator(self.coordinator)
            
            # Initialize monitor
            self.monitor = FederatedMonitor(
                redis_url=self.config.redis_url,
                namespace=f"{self.config.namespace}:monitoring"
            )
            
            await self.monitor.initialize()
            
            # Initialize version manager
            versioning_config = VersioningConfig(
                storage_path=self.config.model_storage_path,
                compression_enabled=self.config.enable_model_compression,
                auto_cleanup_enabled=self.config.enable_auto_cleanup,
                checkpoint_interval=self.config.checkpoint_interval
            )
            
            self.version_manager = ModelVersionManager(versioning_config)
            await self.version_manager.initialize()
            
            # Initialize dashboard
            self.dashboard = FederatedDashboard(
                coordinator=self.coordinator,
                monitor=self.monitor,
                version_manager=self.version_manager,
                host=self.config.dashboard_host,
                port=self.config.dashboard_port
            )
            
            self.logger.info("✅ Core components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize core components: {e}")
            return False
    
    async def _setup_agent_registry_integration(self):
        """Setup integration with existing agent registry"""
        try:
            self.logger.info("Setting up agent registry integration...")
            
            # Get existing agent registry
            self.agent_registry = get_agent_registry()
            
            if not self.agent_registry:
                self.logger.warning("No existing agent registry found, creating new one")
                self.agent_registry = AgentRegistry(
                    redis_url=self.config.redis_url,
                    namespace="sutazai"
                )
                await self.agent_registry.initialize()
                set_agent_registry(self.agent_registry)
            
            # Register event handlers for agent lifecycle
            self.agent_registry.register_event_handler(
                "agent_registered", self._on_agent_registered
            )
            
            self.agent_registry.register_event_handler(
                "agent_unregistered", self._on_agent_unregistered
            )
            
            self.logger.info("✅ Agent registry integration complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup agent registry integration: {e}")
            raise
    
    async def _enhance_agents_with_federated_learning(self):
        """Enhance existing agents with federated learning capabilities"""
        try:
            self.logger.info("Enhancing agents with federated learning capabilities...")
            
            # Get all registered agents
            all_agents = self.agent_registry.get_all_agents()
            
            enhanced_count = 0
            for agent_id, registration in all_agents.items():
                # Check if agent has learning capability or can be enhanced
                if self._should_enhance_agent(registration):
                    success = await self._enhance_single_agent(agent_id, registration)
                    if success:
                        enhanced_count += 1
            
            self.logger.info(f"✅ Enhanced {enhanced_count} agents with federated learning")
            
        except Exception as e:
            self.logger.error(f"Failed to enhance agents: {e}")
            raise
    
    def _should_enhance_agent(self, registration) -> bool:
        """Determine if an agent should be enhanced with FL capabilities"""
        
        # Check if agent already has learning capability
        if AgentCapability.LEARNING in registration.capabilities:
            return True
        
        # Check if agent has capabilities that could benefit from FL
        beneficial_capabilities = {
            AgentCapability.CODE_GENERATION,
            AgentCapability.CODE_ANALYSIS,
            AgentCapability.REASONING,
            AgentCapability.DATA_PROCESSING
        }
        
        if any(cap in registration.capabilities for cap in beneficial_capabilities):
            return True
        
        # Check agent health and resource availability
        if registration.health.value in ["healthy", "warning"]:
            load_percentage = (registration.current_task_count / 
                             registration.max_concurrent_tasks) * 100
            if load_percentage < 70:  # Don't overload agents
                return True
        
        return False
    
    async def _enhance_single_agent(self, agent_id: str, registration) -> bool:
        """Enhance a single agent with federated learning capability"""
        try:
            # Create federated learning capability for the agent
            # Note: In a real implementation, this would integrate with the actual agent instance
            # For now, we'll create a mock FL capability
            
            fl_client = FederatedClient(
                agent_id=agent_id,
                redis_url=self.config.redis_url,
                namespace=self.config.namespace
            )
            
            success = await fl_client.initialize()
            if success:
                # Store FL capability
                self.fl_enhanced_agents[agent_id] = fl_client
                
                # Add sample training data (in practice, agents would have their own data)
                await self._add_sample_data_to_agent(fl_client)
                
                self.logger.debug(f"Enhanced agent {agent_id} with federated learning")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to enhance agent {agent_id}: {e}")
        
        return False
    
    async def _add_sample_data_to_agent(self, fl_client: FederatedClient):
        """Add sample training data to an agent (for demonstration)"""
        try:
            import numpy as np
            
            # Generate synthetic training data
            n_samples = np.random.randint(100, 1000)
            n_features = 784  # MNIST-like
            n_classes = 10
            
            x_data = np.random.randn(n_samples, n_features)
            y_data = np.random.randint(0, n_classes, n_samples)
            
            fl_client.add_dataset(f"synthetic_data_{fl_client.agent_id}", x_data, y_data)
            
        except Exception as e:
            self.logger.error(f"Failed to add sample data: {e}")
    
    async def _configure_system_settings(self):
        """Configure system-wide federated learning settings"""
        try:
            self.logger.info("Configuring system settings...")
            
            # Configure coordinator settings
            self.coordinator.max_concurrent_trainings = self.config.max_concurrent_trainings
            self.coordinator.client_timeout = self.config.client_timeout_seconds
            
            # Configure privacy settings
            privacy_manager = self.coordinator.privacy_manager
            privacy_manager.global_privacy_level = self.config.default_privacy_level
            
            # Configure monitoring thresholds
            self.monitor.alert_thresholds.update({
                "accuracy_drop": 0.05,  # 5% accuracy drop
                "client_failure_rate": 0.25,  # 25% client failure rate
                "communication_timeout": self.config.client_timeout_seconds,
                "resource_utilization": 0.85  # 85% resource usage
            })
            
            self.logger.info("✅ System settings configured")
            
        except Exception as e:
            self.logger.error(f"Failed to configure system settings: {e}")
            raise
    
    async def _start_background_services(self):
        """Start background services and monitoring"""
        try:
            self.logger.info("Starting background services...")
            
            # Start system monitoring
            monitoring_task = asyncio.create_task(self._system_health_monitor())
            self._background_tasks.add(monitoring_task)
            monitoring_task.add_done_callback(self._background_tasks.discard)
            
            # Start performance optimization
            optimization_task = asyncio.create_task(self._performance_optimizer())
            self._background_tasks.add(optimization_task)
            optimization_task.add_done_callback(self._background_tasks.discard)
            
            # Start agent health monitoring
            agent_monitoring_task = asyncio.create_task(self._agent_health_monitor())
            self._background_tasks.add(agent_monitoring_task)
            agent_monitoring_task.add_done_callback(self._background_tasks.discard)
            
            self.components_started = True
            self.logger.info("✅ Background services started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background services: {e}")
            raise
    
    async def _validate_system_integrity(self):
        """Validate system integrity and readiness"""
        try:
            self.logger.info("Validating system integrity...")
            
            validation_results = {
                "coordinator_ready": False,
                "monitor_ready": False,
                "version_manager_ready": False,
                "enhanced_agents": 0,
                "system_health": "unknown"
            }
            
            # Check coordinator
            if self.coordinator and hasattr(self.coordinator, 'status'):
                validation_results["coordinator_ready"] = True
            
            # Check monitor
            if self.monitor:
                health = self.monitor.get_system_health()
                validation_results["monitor_ready"] = True
                validation_results["system_health"] = health.get("health_status", "unknown")
            
            # Check version manager
            if self.version_manager:
                stats = self.version_manager.get_version_stats()
                validation_results["version_manager_ready"] = True
            
            # Check enhanced agents
            validation_results["enhanced_agents"] = len(self.fl_enhanced_agents)
            
            # Validate minimum requirements
            if validation_results["enhanced_agents"] < 3:
                raise Exception(f"Insufficient FL-capable agents: {validation_results['enhanced_agents']} < 3")
            
            if not all([
                validation_results["coordinator_ready"],
                validation_results["monitor_ready"],
                validation_results["version_manager_ready"]
            ]):
                raise Exception("Core components not ready")
            
            self.logger.info("✅ System integrity validation passed")
            self.logger.info(f"   - FL-capable agents: {validation_results['enhanced_agents']}")
            self.logger.info(f"   - System health: {validation_results['system_health']}")
            
        except Exception as e:
            self.logger.error(f"System integrity validation failed: {e}")
            raise
    
    async def _generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        try:
            report = {
                "deployment_timestamp": datetime.utcnow().isoformat(),
                "system_config": {
                    "cpu_cores": self.config.cpu_cores,
                    "max_concurrent_trainings": self.config.max_concurrent_trainings,
                    "privacy_level": self.config.default_privacy_level.value,
                    "aggregation_algorithm": self.config.default_aggregation_algorithm.value
                },
                "components": {
                    "coordinator": {
                        "status": "active",
                        "redis_url": self.config.redis_url,
                        "namespace": self.config.namespace
                    },
                    "monitor": {
                        "status": "active",
                        "system_health": self.monitor.get_system_health() if self.monitor else None
                    },
                    "version_manager": {
                        "status": "active",
                        "storage_path": self.config.model_storage_path,
                        "compression_enabled": self.config.enable_model_compression
                    },
                    "dashboard": {
                        "status": "ready",
                        "host": self.config.dashboard_host,
                        "port": self.config.dashboard_port
                    }
                },
                "agents": {
                    "total_agents": len(self.agent_registry.get_all_agents()) if self.agent_registry else 0,
                    "fl_enhanced_agents": len(self.fl_enhanced_agents),
                    "enhanced_agent_ids": list(self.fl_enhanced_agents.keys())
                },
                "capabilities": {
                    "supported_algorithms": ["FedAvg", "FedProx", "FedOpt"],
                    "privacy_mechanisms": ["Differential Privacy", "Secure Aggregation"],
                    "model_versioning": True,
                    "real_time_monitoring": True,
                    "web_dashboard": True
                }
            }
            
            # Save report
            report_path = Path(self.config.model_storage_path) / "deployment_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"📊 Deployment report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate deployment report: {e}")
    
    # Background service implementations
    async def _system_health_monitor(self):
        """Monitor overall system health"""
        while not self._shutdown_event.is_set():
            try:
                if self.monitor:
                    health = self.monitor.get_system_health()
                    
                    # Log health summary
                    if health["overall_health_score"] < 0.7:
                        self.logger.warning(f"System health degraded: {health['health_status']} "
                                          f"(score: {health['overall_health_score']:.2f})")
                    
                    # Check for critical alerts
                    alerts = self.monitor.get_alerts(severity="critical")
                    if alerts:
                        self.logger.error(f"Critical alerts detected: {len(alerts)}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"System health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_optimizer(self):
        """Optimize system performance based on usage patterns"""
        while not self._shutdown_event.is_set():
            try:
                if self.coordinator:
                    stats = self.coordinator.get_coordinator_stats()
                    
                    # Adjust concurrent training limit based on performance
                    participation_rate = stats.get("client_participation_rate", 0)
                    
                    if participation_rate > 80 and stats.get("average_round_time", 0) < 60:
                        # High participation, fast rounds - can handle more trainings
                        new_limit = min(self.config.max_concurrent_trainings + 1, 5)
                        if new_limit != self.coordinator.max_concurrent_trainings:
                            self.coordinator.max_concurrent_trainings = new_limit
                            self.logger.info(f"Increased concurrent training limit to {new_limit}")
                    
                    elif participation_rate < 50 or stats.get("average_round_time", 0) > 120:
                        # Low participation or slow rounds - reduce load
                        new_limit = max(self.config.max_concurrent_trainings - 1, 1)
                        if new_limit != self.coordinator.max_concurrent_trainings:
                            self.coordinator.max_concurrent_trainings = new_limit
                            self.logger.info(f"Reduced concurrent training limit to {new_limit}")
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance optimizer error: {e}")
                await asyncio.sleep(300)
    
    async def _agent_health_monitor(self):
        """Monitor health of FL-enhanced agents"""
        while not self._shutdown_event.is_set():
            try:
                unhealthy_agents = []
                
                for agent_id, fl_client in self.fl_enhanced_agents.items():
                    # Check agent registration status
                    registration = self.agent_registry.get_agent(agent_id) if self.agent_registry else None
                    
                    if not registration or registration.status.value == "offline":
                        unhealthy_agents.append(agent_id)
                    
                    # Check FL client stats
                    try:
                        stats = fl_client.get_client_stats()
                        if stats and stats.get("status") == "error":
                            unhealthy_agents.append(agent_id)
                    except:
                        pass  # FL client might not be accessible
                
                # Remove unhealthy agents
                for agent_id in unhealthy_agents:
                    if agent_id in self.fl_enhanced_agents:
                        try:
                            await self.fl_enhanced_agents[agent_id].shutdown()
                            del self.fl_enhanced_agents[agent_id]
                            self.logger.warning(f"Removed unhealthy FL agent: {agent_id}")
                        except:
                            pass
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Agent health monitor error: {e}")
                await asyncio.sleep(120)
    
    # Event handlers
    async def _on_agent_registered(self, event_data: Dict[str, Any]):
        """Handle new agent registration"""
        try:
            agent_id = event_data.get("agent_id")
            if agent_id and agent_id not in self.fl_enhanced_agents:
                # Get agent registration
                registration = self.agent_registry.get_agent(agent_id)
                if registration and self._should_enhance_agent(registration):
                    success = await self._enhance_single_agent(agent_id, registration)
                    if success:
                        self.logger.info(f"Enhanced new agent {agent_id} with federated learning")
        
        except Exception as e:
            self.logger.error(f"Error handling agent registration: {e}")
    
    async def _on_agent_unregistered(self, event_data: Dict[str, Any]):
        """Handle agent unregistration"""
        try:
            agent_id = event_data.get("agent_id")
            if agent_id in self.fl_enhanced_agents:
                await self.fl_enhanced_agents[agent_id].shutdown()
                del self.fl_enhanced_agents[agent_id]
                self.logger.info(f"Removed FL capability from unregistered agent {agent_id}")
        
        except Exception as e:
            self.logger.error(f"Error handling agent unregistration: {e}")
    
    # Public API methods
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "initialized": self.system_initialized,
            "components_started": self.components_started,
            "enhanced_agents": len(self.fl_enhanced_agents),
            "coordinator_status": "active" if self.coordinator else "inactive",
            "monitor_status": "active" if self.monitor else "inactive",
            "version_manager_status": "active" if self.version_manager else "inactive",
            "dashboard_status": "ready" if self.dashboard else "inactive",
            "system_health": self.monitor.get_system_health() if self.monitor else None
        }
    
    def get_enhanced_agents(self) -> List[str]:
        """Get list of FL-enhanced agent IDs"""
        return list(self.fl_enhanced_agents.keys())
    
    async def start_dashboard(self):
        """Start the web dashboard"""
        if self.dashboard:
            await self.dashboard.start_server()
        else:
            raise Exception("Dashboard not initialized")
    
    async def _cleanup_partial_initialization(self):
        """Clean up partially initialized components"""
        try:
            if self.coordinator:
                await self.coordinator.shutdown()
            
            if self.monitor:
                await self.monitor.shutdown()
            
            if self.version_manager:
                await self.version_manager.shutdown()
            
            for fl_client in self.fl_enhanced_agents.values():
                try:
                    await fl_client.shutdown()
                except:
                    pass
            
            self.fl_enhanced_agents.clear()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def shutdown(self):
        """Shutdown the entire federated learning system"""
        self.logger.info("Shutting down Federated Learning System")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown components
        if self.dashboard:
            await self.dashboard.shutdown()
        
        if self.version_manager:
            await self.version_manager.shutdown()
        
        if self.monitor:
            await self.monitor.shutdown()
        
        if self.coordinator:
            await self.coordinator.shutdown()
        
        # Shutdown FL clients
        for fl_client in self.fl_enhanced_agents.values():
            try:
                await fl_client.shutdown()
            except:
                pass
        
        self.fl_enhanced_agents.clear()
        
        self.logger.info("✅ Federated Learning System shutdown complete")


# Main deployment function
async def deploy_federated_learning_system(config: FederatedSystemConfig = None) -> FederatedSystemIntegrator:
    """
    Deploy the complete federated learning system for SutazAI
    
    Args:
        config: System configuration (uses defaults if None)
    
    Returns:
        Initialized FederatedSystemIntegrator instance
    """
    integrator = FederatedSystemIntegrator(config)
    
    success = await integrator.initialize_system()
    if not success:
        raise Exception("Failed to deploy federated learning system")
    
    return integrator


# Standalone deployment script
async def main():
    """Main deployment script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("federated_deployment")
    
    try:
        logger.info("🚀 Starting SutazAI Federated Learning System Deployment")
        
        # Create system configuration
        config = FederatedSystemConfig()
        
        # Deploy system
        integrator = await deploy_federated_learning_system(config)
        
        logger.info("✅ Federated Learning System deployed successfully!")
        logger.info(f"   - Dashboard available at: http://{config.dashboard_host}:{config.dashboard_port}")
        logger.info(f"   - Enhanced agents: {len(integrator.get_enhanced_agents())}")
        
        # Start dashboard
        logger.info("🌐 Starting dashboard server...")
        await integrator.start_dashboard()
        
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())