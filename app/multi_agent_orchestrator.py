import asyncio
import logging
import time
from typing import Dict, Any

import docker
import requests
from concurrent.futures import ThreadPoolExecutor

from app.agent_manager import AgentManager
from app.models.agent import Agent, AgentStatus
from app.models.task import Task
from app.task_manager import TaskManager

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    def __init__(self):
        self.agent_manager = AgentManager()
        self.task_manager = TaskManager()
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.running = False
        self.coordination_loops = []
        self.docker_client = docker.from_env()

        self.services = {
            "ollama": "http://ollama:11434",
            "backend": "http://backend:8000",
            "qdrant": "http://qdrant:6333",
            "chromadb": "http://chromadb:8001",
            "aider": "sutazai-aider",
            "gpt-engineer": "sutazai-gpt-engineer",
            "semgrep": "sutazai-semgrep"
        }

    async def start_orchestration(self):
        self.running = True
        logger.info("Starting multi-agent orchestration system...")
        self.coordination_loops = [
            asyncio.create_task(self.task_distribution_loop()),
            asyncio.create_task(self.agent_monitoring_loop())
        ]
        logger.info(f"Started orchestration with {len(self.agent_manager.agents)} agents")

    async def stop_orchestration(self):
        self.running = False
        for loop in self.coordination_loops:
            loop.cancel()
        await asyncio.gather(*self.coordination_loops, return_exceptions=True)
        logger.info("Stopped multi-agent orchestration system")

    async def task_distribution_loop(self):
        while self.running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                best_agent = self.agent_manager.find_best_agent_for_task(task.type)
                if best_agent:
                    await self.assign_task_to_agent(task, best_agent)
                else:
                    await self.task_queue.put(task)
                    await asyncio.sleep(5)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in task distribution loop: {e}")
                await asyncio.sleep(1)

    async def agent_monitoring_loop(self):
        while self.running:
            try:
                current_time = time.time()
                for agent in self.agent_manager.get_all_agents():
                    if current_time - agent.last_heartbeat > 60:
                        if agent.status != AgentStatus.OFFLINE:
                            logger.warning(f"Agent {agent.name} appears to be offline")
                            agent.status = AgentStatus.OFFLINE
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error in agent monitoring loop: {e}")
                await asyncio.sleep(30)

    async def assign_task_to_agent(self, task: Task, agent: Agent):
        task.assigned_agent = agent.id
        task.status = "assigned"
        agent.current_task = task.id
        agent.status = AgentStatus.BUSY
        logger.info(f"Assigned task {task.id} to agent {agent.name}")
        asyncio.create_task(self.execute_task(task, agent))

    async def execute_task(self, task: Task, agent: Agent):
        try:
            task.status = "executing"

            if agent.type.value == "aider":
                result = await self.execute_aider_task(task)
            elif agent.type.value == "gpt_engineer":
                result = await self.execute_gpt_engineer_task(task)
            elif agent.type.value == "semgrep":
                result = await self.execute_semgrep_task(task)
            else:
                result = await self.execute_general_task(task, agent)

            task.status = "completed"
            task.completed_at = time.time()
            task.result = result
            agent.current_task = None
            agent.status = AgentStatus.IDLE
            agent.completed_tasks += 1
            agent.last_heartbeat = time.time()
            logger.info(f"Task {task.id} completed by agent {agent.name}")

        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            task.status = "error"
            task.result = {"error": str(e)}
            agent.current_task = None
            agent.status = AgentStatus.ERROR

    async def execute_aider_task(self, task: Task) -> Dict[str, Any]:
        return await self.run_docker_command(self.services["aider"], f"aider --yes --file {task.metadata['file_path']} --message \"{task.description}\"")

    async def execute_gpt_engineer_task(self, task: Task) -> Dict[str, Any]:
        return await self.run_docker_command(self.services["gpt-engineer"], f"gpt-engineer --project-path {task.metadata['project_path']} --prompt \"{task.description}\"")

    async def execute_semgrep_task(self, task: Task) -> Dict[str, Any]:
        return await self.run_docker_command(self.services["semgrep"], f"semgrep scan --config auto /src/{task.metadata['file_path']}")

    async def run_docker_command(self, container_name: str, command: str) -> Dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_name)
            exit_code, output = container.exec_run(command)
            if exit_code == 0:
                return {"output": output.decode('utf-8')}
            else:
                return {"error": output.decode('utf-8')}
        except Exception as e:
            return {"error": str(e)}

    async def execute_general_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.services['ollama']}/api/generate",
                json={
                    "model": agent.config.get("model", "llama3"),
                    "prompt": task.description,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return {"response": response.json().get("response", "")}
        except Exception as e:
            return {"error": str(e)}

    async def submit_task(self, description: str, task_type: str, priority: int = 5, metadata: Dict = None) -> str:
        task = self.task_manager.create_task(description, task_type, priority, metadata)
        await self.task_queue.put(task)
        logger.info(f"Submitted task: {task.id} - {description}")
        return task.id

    def get_system_metrics(self) -> Dict[str, Any]:
        return {
            "agents": [agent.__dict__ for agent in self.agent_manager.get_all_agents()],
            "tasks": [task.__dict__ for task in self.task_manager.get_all_tasks()]
        }