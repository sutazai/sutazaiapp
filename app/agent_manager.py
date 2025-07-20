import time
import logging
from typing import Dict, Optional, List

from app.models.agent import Agent, AgentStatus, AgentType

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.initialize_default_agents()

    def initialize_default_agents(self):
        """Initialize the default set of AI agents"""
        default_agents = [
            {
                "name": "CodeMaster",
                "type": AgentType.CODE_GENERATOR,
                "capabilities": ["python", "javascript", "java", "cpp", "rust", "go", "code_review", "debugging"],
                "config": {"model": "deepseek-coder:33b", "temperature": 0.3}
            },
            {
                "name": "SecurityGuard",
                "type": AgentType.SECURITY_ANALYST,
                "capabilities": ["vulnerability_scan", "security_audit", "penetration_testing", "compliance_check"],
                "config": {"model": "llama3", "temperature": 0.2}
            },
            {
                "name": "DocProcessor",
                "type": AgentType.DOCUMENT_PROCESSOR,
                "capabilities": ["pdf_processing", "text_extraction", "summarization", "translation"],
                "config": {"model": "llama3", "temperature": 0.5}
            },
            {
                "name": "AiderCodeEditor",
                "type": AgentType.AIDER,
                "capabilities": ["edit_code", "apply_diffs"],
                "config": {}
            },
            {
                "name": "GPTEngineerBuilder",
                "type": AgentType.GPT_ENGINEER,
                "capabilities": ["build_project", "scaffold_code"],
                "config": {}
            },
            {
                "name": "SemgrepScanner",
                "type": AgentType.SEMGREP,
                "capabilities": ["scan_code", "find_vulnerabilities"],
                "config": {}
            }
        ]

        for agent_config in default_agents:
            agent_id = f"agent_{len(self.agents) + 1}"
            agent = Agent(
                id=agent_id,
                name=agent_config["name"],
                type=agent_config["type"],
                capabilities=agent_config["capabilities"],
                config=agent_config["config"],
                status=AgentStatus.IDLE
            )
            agent.last_heartbeat = time.time()
            self.agents[agent_id] = agent
            logger.info(f"Initialized agent: {agent.name} ({agent.type.value})")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List[Agent]:
        return list(self.agents.values())

    def get_idle_agents(self) -> List[Agent]:
        return [agent for agent in self.agents.values() if agent.status == AgentStatus.IDLE]

    def find_best_agent_for_task(self, task_type: str) -> Optional[Agent]:
        """Find the best agent to handle a specific task"""
        suitable_agents = []
        for agent in self.agents.values():
            if agent.status == AgentStatus.IDLE and task_type in agent.capabilities:
                suitable_agents.append(agent)
        if not suitable_agents:
            return None
        suitable_agents.sort(key=lambda a: a.completed_tasks)
        return suitable_agents[0]
