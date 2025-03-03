#!/usr/bin/env python3.11
"""
Supreme AI Orchestrator

This module implements the Supreme AI Orchestrator that manages AI agents
and handles synchronization between servers.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from core_system.orchestrator.agent_manager import AgentManager
from core_system.orchestrator.sync_manager import SyncManager
from core_system.orchestrator.task_queue import TaskQueue
from core_system.orchestrator.models import Task, AgentStatus, SyncStatus, OrchestratorConfig
from core_system.orchestrator.exceptions import OrchestratorError

logger = logging.getLogger(__name__)

@dataclass
class OrchestratorConfig:
    """Configuration for the Supreme AI Orchestrator"""
    primary_server: str
    secondary_server: str
    sync_interval: int = 60  # seconds
    max_agents: int = 10
    task_timeout: int = 3600  # seconds


class SupremeAIOrchestrator:
    """
    Supreme AI Orchestrator class that manages AI agents and server synchronization
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.task_queue = TaskQueue(config)
        self.agent_manager = AgentManager(config)
        self.sync_manager = SyncManager(config)
        self.is_running = False
        self.logger = logging.getLogger('SupremeAIOrchestrator')

    async def start(self):
        """Start the orchestrator and all its components."""
        try:
            self.is_running = True
            await self.task_queue.start()
            await self.agent_manager.start()
            await self.sync_manager.start()
            self.logger.info("Orchestrator started successfully")
        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start orchestrator: {str(e)}")
            raise OrchestratorError("Failed to start orchestrator") from e

    async def stop(self):
        """Stop the orchestrator and all its components."""
        try:
            self.is_running = False
            await self.task_queue.stop()
            await self.agent_manager.stop()
            await self.sync_manager.stop()
            self.logger.info("Orchestrator stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping orchestrator: {str(e)}")
            raise OrchestratorError("Failed to stop orchestrator") from e

    async def submit_task(self, task: Dict):
        """Submit a task to the task queue."""
        if not self.is_running:
            raise OrchestratorError("Orchestrator is not running")
            
        try:
            await self.task_queue.submit(task)
            self.logger.info(f"Task submitted successfully: {task.get('id', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Failed to submit task: {str(e)}")
            raise OrchestratorError("Failed to submit task") from e

    async def register_agent(self, agent: Dict):
        """Register a new agent with the orchestrator."""
        if not self.is_running:
            raise OrchestratorError("Orchestrator is not running")
            
        try:
            await self.agent_manager.register_agent(agent)
            self.logger.info(f"Agent registered successfully: {agent.get('id', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Failed to register agent: {str(e)}")
            raise OrchestratorError("Failed to register agent") from e

    async def get_agent_status(self, agent_id: str) -> Dict:
        """Get the status of a specific agent."""
        try:
            return await self.agent_manager.get_agent_status(agent_id)
        except Exception as e:
            self.logger.error(f"Failed to get agent status: {str(e)}")
            raise OrchestratorError("Failed to get agent status") from e

    async def list_agents(self) -> List[Dict]:
        """List all registered agents."""
        try:
            return await self.agent_manager.list_agents()
        except Exception as e:
            self.logger.error(f"Failed to list agents: {str(e)}")
            raise OrchestratorError("Failed to list agents") from e

    async def start_agent(self, agent_id: str):
        """Start a specific agent."""
        try:
            await self.agent_manager.start_agent(agent_id)
            self.logger.info(f"Agent started successfully: {agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to start agent: {str(e)}")
            raise OrchestratorError("Failed to start agent") from e

    async def stop_agent(self, agent_id: str):
        """Stop a specific agent."""
        try:
            await self.agent_manager.stop_agent(agent_id)
            self.logger.info(f"Agent stopped successfully: {agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to stop agent: {str(e)}")
            raise OrchestratorError("Failed to stop agent") from e

    async def start_sync(self):
        """Start synchronization with other servers."""
        try:
            await self.sync_manager.start()
            self.logger.info("Synchronization started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start synchronization: {str(e)}")
            raise OrchestratorError("Failed to start synchronization") from e

    async def stop_sync(self):
        """Stop synchronization with other servers."""
        try:
            await self.sync_manager.stop()
            self.logger.info("Synchronization stopped successfully")
        except Exception as e:
            self.logger.error(f"Failed to stop synchronization: {str(e)}")
            raise OrchestratorError("Failed to stop synchronization") from e

    async def deploy(self, target_server: str):
        """Deploy changes to a target server."""
        try:
            await self.sync_manager.deploy(target_server)
            self.logger.info(f"Deployment to {target_server} completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to deploy to {target_server}: {str(e)}")
            raise OrchestratorError("Failed to deploy") from e

    async def rollback(self, target_server: str):
        """Rollback changes on a target server."""
        try:
            await self.sync_manager.rollback(target_server)
            self.logger.info(f"Rollback on {target_server} completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to rollback on {target_server}: {str(e)}")
            raise OrchestratorError("Failed to rollback") from e

    def get_status(self) -> Dict:
        """Get the current status of the orchestrator"""
        return {
            "is_running": self.is_running,
            "agent_count": self.agent_manager.get_agent_count(),
            "queue_size": self.task_queue.size(),
            "sync_status": self.sync_manager.get_status(),
            "last_sync": self.sync_manager.last_sync_time
        }

    def process_tasks(self) -> None:
        """Process tasks."""
        self.task_queue.process()

    def sync(self) -> None:
        """Synchronize with other servers."""
        self.sync_manager.sync()

    def update_agent_heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat."""
        self.agent_manager.update_heartbeat(agent_id) 