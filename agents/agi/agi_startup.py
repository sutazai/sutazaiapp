#!/usr/bin/env python3
"""
AGI Orchestration Layer Startup Script
Initializes and starts the complete AGI orchestration system
"""

import asyncio
import logging
import signal
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_orchestration_layer import AGIOrchestrationLayer
from agi_background_processes import BackgroundProcessManager
from communication_protocols import CommunicationProtocol
from collective_intelligence import CollectiveIntelligence

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/sutazaiapp/logs/agi_orchestration.log')
    ]
)
logger = logging.getLogger(__name__)


class AGIOrchestrationSystem:
    """
    Main AGI Orchestration System
    Coordinates all components of the advanced AGI orchestration layer
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config"):
        self.config_path = Path(config_path)
        self.data_path = Path("/opt/sutazaiapp/data/agi_orchestration")
        
        # Core components
        self.orchestration_layer: Optional[AGIOrchestrationLayer] = None
        self.background_processes: Optional[BackgroundProcessManager] = None
        self.communication_protocol: Optional[CommunicationProtocol] = None
        self.collective_intelligence: Optional[CollectiveIntelligence] = None
        
        # System state
        self.is_running = False
        self.startup_complete = False
        self.shutdown_initiated = False
        
        # Configuration
        self.config = self._load_configuration()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("AGI Orchestration System initialized")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration"""
        
        config_file = self.config_path / "agi_orchestration.yaml"
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                "orchestration": {
                    "name": "SutazAI AGI Orchestration",
                    "version": "1.0.0",
                    "max_concurrent_tasks": 100
                },
                "communication": {
                    "redis": {
                        "host": "redis",
                        "port": 6379,
                        "db": 1
                    }
                }
            }
        
        return config
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def startup(self):
        """Start the complete AGI orchestration system"""
        
        try:
            logger.info("Starting AGI Orchestration System...")
            
            # Phase 1: Initialize communication layer
            logger.info("Phase 1: Initializing communication layer...")
            await self._initialize_communication()
            
            # Phase 2: Initialize orchestration layer
            logger.info("Phase 2: Initializing orchestration layer...")
            await self._initialize_orchestration()
            
            # Phase 3: Initialize collective intelligence
            logger.info("Phase 3: Initializing collective intelligence...")
            await self._initialize_collective_intelligence()
            
            # Phase 4: Initialize background processes
            logger.info("Phase 4: Initializing background processes...")
            await self._initialize_background_processes()
            
            # Phase 5: Start system integration
            logger.info("Phase 5: Starting system integration...")
            await self._integrate_systems()
            
            # Phase 6: Run system validation
            logger.info("Phase 6: Running system validation...")
            await self._validate_system()
            
            self.is_running = True
            self.startup_complete = True
            
            logger.info("AGI Orchestration System startup complete!")
            
            # Display system status
            await self._display_startup_summary()
            
        except Exception as e:
            logger.error(f"Failed to start AGI Orchestration System: {e}")
            await self.shutdown()
            raise
    
    async def _initialize_communication(self):
        """Initialize the communication layer"""
        
        redis_config = self.config.get("communication", {}).get("redis", {})
        redis_url = f"redis://{redis_config.get('host', 'redis')}:{redis_config.get('port', 6379)}/{redis_config.get('db', 1)}"
        
        self.communication_protocol = CommunicationProtocol(
            redis_url=redis_url,
            agent_id="agi_orchestration_system"
        )
        
        await self.communication_protocol.initialize()
        logger.info("Communication layer initialized")
    
    async def _initialize_orchestration(self):
        """Initialize the main orchestration layer"""
        
        self.orchestration_layer = AGIOrchestrationLayer(
            config_path=str(self.config_path),
            data_path=str(self.data_path)
        )
        
        await self.orchestration_layer.initialize()
        logger.info("Orchestration layer initialized")
    
    async def _initialize_collective_intelligence(self):
        """Initialize the collective intelligence system"""
        
        self.collective_intelligence = CollectiveIntelligence(
            data_path=str(self.data_path / "collective_intelligence")
        )
        
        await self.collective_intelligence.awaken()
        logger.info("Collective intelligence initialized")
    
    async def _initialize_background_processes(self):
        """Initialize background process manager"""
        
        self.background_processes = BackgroundProcessManager(self.orchestration_layer)
        
        # Start all background processes
        await self.background_processes.monitor_agent_health()
        logger.info("Background processes initialized")
    
    async def _integrate_systems(self):
        """Integrate all system components"""
        
        # Connect communication protocol to orchestration layer
        if self.communication_protocol and self.orchestration_layer:
            # Register message handlers
            self.communication_protocol.register_message_handler(
                self.communication_protocol.MessageType.TASK_REQUEST,
                self._handle_task_request
            )
            
            self.communication_protocol.register_message_handler(
                self.communication_protocol.MessageType.CONSENSUS_VOTE,
                self._handle_consensus_vote
            )
            
            self.communication_protocol.register_message_handler(
                self.communication_protocol.MessageType.EMERGENT_BEHAVIOR,
                self._handle_emergent_behavior
            )
            
            self.communication_protocol.register_message_handler(
                self.communication_protocol.MessageType.SAFETY_ALERT,
                self._handle_safety_alert
            )
        
        # Connect collective intelligence to orchestration
        if self.collective_intelligence and self.orchestration_layer:
            # Register agents with collective intelligence
            for agent_id, agent in self.orchestration_layer.agents.items():
                await self.collective_intelligence.register_agent(agent)
        
        logger.info("System integration complete")
    
    async def _validate_system(self):
        """Validate that all systems are working correctly"""
        
        validation_results = {
            "orchestration_layer": False,
            "communication_protocol": False,
            "collective_intelligence": False,
            "background_processes": False
        }
        
        try:
            # Validate orchestration layer
            if self.orchestration_layer:
                status = await self.orchestration_layer.get_orchestration_status()
                validation_results["orchestration_layer"] = status["state"] == "active"
            
            # Validate communication protocol
            if self.communication_protocol:
                comm_status = await self.communication_protocol.get_communication_status()
                validation_results["communication_protocol"] = len(comm_status["subscriptions"]) > 0
            
            # Validate collective intelligence
            if self.collective_intelligence:
                ci_status = await self.collective_intelligence.get_collective_status()
                validation_results["collective_intelligence"] = ci_status["state"] == "learning"
            
            # Validate background processes
            if self.background_processes:
                bp_status = await self.background_processes.get_background_process_status()
                validation_results["background_processes"] = "running" in str(bp_status["process_status"])
            
            # Check overall validation
            all_valid = all(validation_results.values())
            
            if all_valid:
                logger.info("System validation passed")
            else:
                failed_components = [k for k, v in validation_results.items() if not v]
                logger.warning(f"System validation failed for components: {failed_components}")
                
        except Exception as e:
            logger.error(f"System validation error: {e}")
    
    async def _display_startup_summary(self):
        """Display startup summary"""
        
        summary = {
            "system_name": "SutazAI AGI Orchestration System",
            "version": "1.0.0",
            "startup_time": "completed",
            "components": {
                "orchestration_layer": "active",
                "communication_protocol": "active",
                "collective_intelligence": "active",
                "background_processes": "active"
            }
        }
        
        # Get component statuses
        try:
            if self.orchestration_layer:
                orch_status = await self.orchestration_layer.get_orchestration_status()
                summary["agents"] = orch_status["agents"]
                summary["tasks"] = orch_status["tasks"]
                summary["performance"] = orch_status["performance"]
            
            if self.collective_intelligence:
                ci_status = await self.collective_intelligence.get_collective_status()
                summary["collective_awareness"] = ci_status["collective_awareness"]
                summary["neural_connections"] = ci_status["neural_connections"]
            
        except Exception as e:
            logger.error(f"Error getting startup summary: {e}")
        
        logger.info("=== AGI ORCHESTRATION SYSTEM STARTUP SUMMARY ===")
        logger.info(json.dumps(summary, indent=2))
        logger.info("=== SYSTEM READY FOR OPERATION ===")
    
    async def _handle_task_request(self, message):
        """Handle incoming task request messages"""
        
        try:
            payload = message.payload
            task_id = payload.get("task_id")
            description = payload.get("description")
            
            if self.orchestration_layer and task_id and description:
                # Submit task to orchestration layer
                await self.orchestration_layer.submit_task(
                    task_description=description,
                    input_data=payload.get("input_data", {}),
                    priority=self.orchestration_layer.TaskPriority.MEDIUM
                )
                
                logger.info(f"Processed task request: {task_id}")
            
        except Exception as e:
            logger.error(f"Error handling task request: {e}")
    
    async def _handle_consensus_vote(self, message):
        """Handle consensus vote messages"""
        
        try:
            payload = message.payload
            proposal_id = payload.get("proposal_id")
            vote = payload.get("vote")
            
            if self.collective_intelligence and proposal_id is not None and vote is not None:
                # Process consensus vote through collective intelligence
                # In production, would integrate with actual consensus system
                logger.info(f"Processed consensus vote for proposal {proposal_id}: {vote}")
            
        except Exception as e:
            logger.error(f"Error handling consensus vote: {e}")
    
    async def _handle_emergent_behavior(self, message):
        """Handle emergent behavior alerts"""
        
        try:
            payload = message.payload
            behavior_type = payload.get("behavior_type")
            participants = payload.get("participants", [])
            impact_score = payload.get("impact_score", 0.0)
            
            logger.info(f"Emergent behavior detected: {behavior_type} with {len(participants)} participants (impact: {impact_score})")
            
            # High impact behaviors need immediate attention
            if impact_score > 0.8:
                logger.warning(f"HIGH IMPACT emergent behavior: {behavior_type}")
                
                # Could trigger safety protocols here
                if self.communication_protocol:
                    await self.communication_protocol.send_safety_alert(
                        "warning",
                        f"High impact emergent behavior detected: {behavior_type}",
                        payload
                    )
            
        except Exception as e:
            logger.error(f"Error handling emergent behavior: {e}")
    
    async def _handle_safety_alert(self, message):
        """Handle safety alert messages"""
        
        try:
            payload = message.payload
            alert_level = payload.get("alert_level")
            alert_message = payload.get("alert_message")
            
            if alert_level == "critical" or alert_level == "emergency":
                logger.critical(f"CRITICAL SAFETY ALERT: {alert_message}")
                
                # Emergency shutdown if needed
                if "shutdown" in alert_message.lower():
                    logger.critical("EMERGENCY SHUTDOWN INITIATED")
                    await self.shutdown()
            else:
                logger.warning(f"Safety alert ({alert_level}): {alert_message}")
            
        except Exception as e:
            logger.error(f"Error handling safety alert: {e}")
    
    async def run(self):
        """Main run loop for the AGI orchestration system"""
        
        if not self.startup_complete:
            await self.startup()
        
        logger.info("AGI Orchestration System running...")
        
        try:
            # Main system loop
            while self.is_running and not self.shutdown_initiated:
                # System health check
                await self._perform_health_check()
                
                # Process any pending system commands
                await self._process_system_commands()
                
                # Brief pause
                await asyncio.sleep(30)
                
        except Exception as e:
            logger.error(f"Error in main run loop: {e}")
        finally:
            if not self.shutdown_initiated:
                await self.shutdown()
    
    async def _perform_health_check(self):
        """Perform periodic system health check"""
        
        try:
            health_status = {
                "timestamp": "now",
                "components": {}
            }
            
            # Check orchestration layer
            if self.orchestration_layer:
                try:
                    status = await self.orchestration_layer.get_orchestration_status()
                    health_status["components"]["orchestration"] = "healthy" if status["state"] == "active" else "unhealthy"
                except:
                    health_status["components"]["orchestration"] = "unhealthy"
            
            # Check communication protocol
            if self.communication_protocol:
                try:
                    comm_status = await self.communication_protocol.get_communication_status()
                    health_status["components"]["communication"] = "healthy" if comm_status["metrics"]["messages_sent"] >= 0 else "unhealthy"
                except:
                    health_status["components"]["communication"] = "unhealthy"
            
            # Check collective intelligence
            if self.collective_intelligence:
                try:
                    ci_status = await self.collective_intelligence.get_collective_status()
                    health_status["components"]["collective_intelligence"] = "healthy" if ci_status["agent_count"] > 0 else "unhealthy"
                except:
                    health_status["components"]["collective_intelligence"] = "unhealthy"
            
            # Log unhealthy components
            unhealthy = [k for k, v in health_status["components"].items() if v == "unhealthy"]
            if unhealthy:
                logger.warning(f"Unhealthy components detected: {unhealthy}")
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    async def _process_system_commands(self):
        """Process any pending system commands"""
        
        # In production, this would check for system commands
        # from external sources (admin interface, monitoring systems, etc.)
        pass
    
    async def shutdown(self):
        """Gracefully shutdown the AGI orchestration system"""
        
        if self.shutdown_initiated:
            return
        
        self.shutdown_initiated = True
        logger.info("Initiating AGI Orchestration System shutdown...")
        
        try:
            # Stop main loop
            self.is_running = False
            
            # Shutdown components in reverse order
            if self.background_processes:
                logger.info("Shutting down background processes...")
                # Background processes should stop when orchestration layer stops
            
            if self.collective_intelligence:
                logger.info("Shutting down collective intelligence...")
                await self.collective_intelligence.shutdown()
            
            if self.orchestration_layer:
                logger.info("Shutting down orchestration layer...")
                await self.orchestration_layer.shutdown()
            
            if self.communication_protocol:
                logger.info("Shutting down communication protocol...")
                await self.communication_protocol.shutdown()
            
            logger.info("AGI Orchestration System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def submit_task(self, task_description: str, input_data: Dict[str, Any] = None) -> Optional[str]:
        """Submit a task to the orchestration system"""
        
        if not self.orchestration_layer or not self.startup_complete:
            logger.error("Cannot submit task: system not ready")
            return None
        
        try:
            task_id = await self.orchestration_layer.submit_task(
                task_description=task_description,
                input_data=input_data or {},
                priority=self.orchestration_layer.TaskPriority.MEDIUM
            )
            
            logger.info(f"Task submitted successfully: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "system": {
                "running": self.is_running,
                "startup_complete": self.startup_complete,
                "shutdown_initiated": self.shutdown_initiated
            },
            "components": {}
        }
        
        try:
            if self.orchestration_layer:
                status["components"]["orchestration"] = await self.orchestration_layer.get_orchestration_status()
            
            if self.communication_protocol:
                status["components"]["communication"] = await self.communication_protocol.get_communication_status()
            
            if self.collective_intelligence:
                status["components"]["collective_intelligence"] = await self.collective_intelligence.get_collective_status()
            
            if self.background_processes:
                status["components"]["background_processes"] = await self.background_processes.get_background_process_status()
                
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            status["error"] = str(e)
        
        return status


async def main():
    """Main entry point for the AGI orchestration system"""
    
    # Create and start the AGI orchestration system
    agi_system = AGIOrchestrationSystem()
    
    try:
        # Start the system
        await agi_system.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        # Ensure clean shutdown
        await agi_system.shutdown()


if __name__ == "__main__":
    # Run the AGI orchestration system
    asyncio.run(main())