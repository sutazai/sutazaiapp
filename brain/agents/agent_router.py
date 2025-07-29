#!/usr/bin/env python3
"""
Agent Router for the Brain
Manages and orchestrates 30+ specialized agents
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx
import docker

from ..core.brain_state import AgentType, AgentCapability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRouter:
    """Routes tasks to appropriate agents and manages execution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = docker.from_env()
        
        # Agent registry with capabilities
        self.agent_registry = self._build_agent_registry()
        
        # HTTP client for agent communication
        self.http_client = httpx.AsyncClient(timeout=300.0)
        
        # Resource tracking
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_metrics: Dict[str, Dict[str, float]] = {}
        
    def _build_agent_registry(self) -> Dict[str, AgentCapability]:
        """Build the registry of all available agents"""
        return {
            # Core AI Orchestration Agents
            AgentType.AUTOGEN: {
                'name': 'AutoGen',
                'type': AgentType.AUTOGEN,
                'capabilities': ['multi-agent-coordination', 'task-decomposition', 'conversation'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'average_latency': 5.0,
                'success_rate': 0.95,
                'specializations': ['complex-reasoning', 'agent-orchestration'],
                'container': 'sutazai/autogen-agent:latest',
                'port': 8001
            },
            
            AgentType.CREWAI: {
                'name': 'CrewAI',
                'type': AgentType.CREWAI,
                'capabilities': ['team-coordination', 'role-based-tasks', 'workflow'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'average_latency': 4.0,
                'success_rate': 0.93,
                'specializations': ['project-management', 'team-tasks'],
                'container': 'sutazai/crewai-agent:latest',
                'port': 8002
            },
            
            AgentType.LANGCHAIN: {
                'name': 'LangChain',
                'type': AgentType.LANGCHAIN,
                'capabilities': ['chain-reasoning', 'tool-use', 'retrieval'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'gpu': 0.0},
                'average_latency': 2.0,
                'success_rate': 0.97,
                'specializations': ['rag', 'api-integration'],
                'container': 'sutazai/langchain-agent:latest',
                'port': 8003
            },
            
            AgentType.AUTOGPT: {
                'name': 'AutoGPT',
                'type': AgentType.AUTOGPT,
                'capabilities': ['autonomous-execution', 'goal-oriented', 'self-improvement'],
                'resource_requirements': {'cpu': 2.0, 'memory': 4.0, 'gpu': 0.0},
                'average_latency': 10.0,
                'success_rate': 0.85,
                'specializations': ['autonomous-tasks', 'research'],
                'container': 'sutazai/autogpt-agent:latest',
                'port': 8004
            },
            
            AgentType.GPT_ENGINEER: {
                'name': 'GPT-Engineer',
                'type': AgentType.GPT_ENGINEER,
                'capabilities': ['code-generation', 'project-creation', 'architecture'],
                'resource_requirements': {'cpu': 1.5, 'memory': 3.0, 'gpu': 0.0},
                'average_latency': 8.0,
                'success_rate': 0.91,
                'specializations': ['full-stack', 'system-design'],
                'container': 'sutazai/gpt-engineer-agent:latest',
                'port': 8005
            },
            
            # Specialized Tool Agents
            AgentType.BROWSER_USE: {
                'name': 'Browser-Use',
                'type': AgentType.BROWSER_USE,
                'capabilities': ['web-browsing', 'form-filling', 'scraping'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'average_latency': 6.0,
                'success_rate': 0.88,
                'specializations': ['web-automation', 'data-extraction'],
                'container': 'sutazai/browser-use-agent:latest',
                'port': 8006
            },
            
            AgentType.SEMGREP: {
                'name': 'Semgrep',
                'type': AgentType.SEMGREP,
                'capabilities': ['code-analysis', 'security-scanning', 'pattern-matching'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'gpu': 0.0},
                'average_latency': 3.0,
                'success_rate': 0.99,
                'specializations': ['security', 'code-quality'],
                'container': 'sutazai/semgrep-agent:latest',
                'port': 8007
            },
            
            AgentType.DOCUMIND: {
                'name': 'Documind',
                'type': AgentType.DOCUMIND,
                'capabilities': ['pdf-parsing', 'document-analysis', 'ocr'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.5},
                'average_latency': 5.0,
                'success_rate': 0.94,
                'specializations': ['document-processing', 'data-extraction'],
                'container': 'sutazai/documind-agent:latest',
                'port': 8008
            },
            
            # Advanced AI Agents
            AgentType.METAGPT: {
                'name': 'MetaGPT',
                'type': AgentType.METAGPT,
                'capabilities': ['software-company-simulation', 'role-play', 'product-development'],
                'resource_requirements': {'cpu': 2.0, 'memory': 4.0, 'gpu': 0.0},
                'average_latency': 12.0,
                'success_rate': 0.87,
                'specializations': ['startup-simulation', 'product-design'],
                'container': 'sutazai/metagpt-agent:latest',
                'port': 8009
            },
            
            AgentType.CAMEL: {
                'name': 'CAMEL',
                'type': AgentType.CAMEL,
                'capabilities': ['role-playing', 'debate', 'collaborative-problem-solving'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'average_latency': 6.0,
                'success_rate': 0.90,
                'specializations': ['negotiation', 'brainstorming'],
                'container': 'sutazai/camel-agent:latest',
                'port': 8010
            },
            
            # Code Development Agents
            AgentType.AIDER: {
                'name': 'Aider',
                'type': AgentType.AIDER,
                'capabilities': ['code-editing', 'refactoring', 'git-integration'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'gpu': 0.0},
                'average_latency': 3.0,
                'success_rate': 0.96,
                'specializations': ['code-improvement', 'bug-fixing'],
                'container': 'sutazai/aider-agent:latest',
                'port': 8011
            },
            
            AgentType.GPTPILOT: {
                'name': 'GPT-Pilot',
                'type': AgentType.GPTPILOT,
                'capabilities': ['app-development', 'debugging', 'testing'],
                'resource_requirements': {'cpu': 1.5, 'memory': 3.0, 'gpu': 0.0},
                'average_latency': 10.0,
                'success_rate': 0.89,
                'specializations': ['full-app-development', 'debugging'],
                'container': 'sutazai/gptpilot-agent:latest',
                'port': 8012
            },
            
            AgentType.DEVIKA: {
                'name': 'Devika',
                'type': AgentType.DEVIKA,
                'capabilities': ['software-engineering', 'planning', 'implementation'],
                'resource_requirements': {'cpu': 1.5, 'memory': 3.0, 'gpu': 0.0},
                'average_latency': 8.0,
                'success_rate': 0.88,
                'specializations': ['engineering-tasks', 'architecture'],
                'container': 'sutazai/devika-agent:latest',
                'port': 8013
            },
            
            AgentType.OPENDEVIN: {
                'name': 'OpenDevin',
                'type': AgentType.OPENDEVIN,
                'capabilities': ['autonomous-coding', 'environment-interaction', 'testing'],
                'resource_requirements': {'cpu': 2.0, 'memory': 4.0, 'gpu': 0.0},
                'average_latency': 12.0,
                'success_rate': 0.86,
                'specializations': ['end-to-end-development', 'testing'],
                'container': 'sutazai/opendevin-agent:latest',
                'port': 8014
            },
            
            # Research Agents
            AgentType.GPTRESEARCHER: {
                'name': 'GPT-Researcher',
                'type': AgentType.GPTRESEARCHER,
                'capabilities': ['research', 'fact-checking', 'report-generation'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'average_latency': 15.0,
                'success_rate': 0.92,
                'specializations': ['academic-research', 'market-analysis'],
                'container': 'sutazai/gpt-researcher-agent:latest',
                'port': 8015
            },
            
            AgentType.STORM: {
                'name': 'STORM',
                'type': AgentType.STORM,
                'capabilities': ['wikipedia-style-writing', 'comprehensive-research', 'citation'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'average_latency': 20.0,
                'success_rate': 0.90,
                'specializations': ['encyclopedic-writing', 'research-synthesis'],
                'container': 'sutazai/storm-agent:latest',
                'port': 8016
            },
            
            AgentType.KHOJ: {
                'name': 'Khoj',
                'type': AgentType.KHOJ,
                'capabilities': ['personal-ai', 'knowledge-management', 'search'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'gpu': 0.0},
                'average_latency': 2.0,
                'success_rate': 0.95,
                'specializations': ['personal-assistant', 'knowledge-base'],
                'container': 'sutazai/khoj-agent:latest',
                'port': 8017
            },
            
            # Analysis Agents
            AgentType.PANDAS_AI: {
                'name': 'Pandas-AI',
                'type': AgentType.PANDAS_AI,
                'capabilities': ['data-analysis', 'visualization', 'statistics'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'average_latency': 4.0,
                'success_rate': 0.94,
                'specializations': ['data-science', 'analytics'],
                'container': 'sutazai/pandas-ai-agent:latest',
                'port': 8018
            },
            
            AgentType.INTERPRETER: {
                'name': 'Open-Interpreter',
                'type': AgentType.INTERPRETER,
                'capabilities': ['code-execution', 'system-control', 'file-manipulation'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'average_latency': 3.0,
                'success_rate': 0.91,
                'specializations': ['automation', 'system-tasks'],
                'container': 'sutazai/open-interpreter-agent:latest',
                'port': 8019
            },
            
            # Creative Agents
            AgentType.FABRIC: {
                'name': 'Fabric',
                'type': AgentType.FABRIC,
                'capabilities': ['pattern-extraction', 'content-generation', 'summarization'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'gpu': 0.0},
                'average_latency': 2.0,
                'success_rate': 0.93,
                'specializations': ['content-creation', 'pattern-analysis'],
                'container': 'sutazai/fabric-agent:latest',
                'port': 8020
            },
            
            AgentType.TXTAI: {
                'name': 'TxtAI',
                'type': AgentType.TXTAI,
                'capabilities': ['semantic-search', 'workflow-automation', 'nlp-pipelines'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.5},
                'average_latency': 3.0,
                'success_rate': 0.96,
                'specializations': ['search', 'nlp-workflows'],
                'container': 'sutazai/txtai-agent:latest',
                'port': 8021
            },
            
            # Security Agents
            AgentType.PANDASEC: {
                'name': 'PandaSec',
                'type': AgentType.PANDASEC,
                'capabilities': ['security-analysis', 'vulnerability-scanning', 'threat-detection'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'average_latency': 5.0,
                'success_rate': 0.97,
                'specializations': ['cybersecurity', 'risk-assessment'],
                'container': 'sutazai/pandasec-agent:latest',
                'port': 8022
            },
            
            AgentType.NUCLEI: {
                'name': 'Nuclei',
                'type': AgentType.NUCLEI,
                'capabilities': ['vulnerability-scanning', 'security-testing', 'compliance'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'gpu': 0.0},
                'average_latency': 4.0,
                'success_rate': 0.98,
                'specializations': ['penetration-testing', 'security-audit'],
                'container': 'sutazai/nuclei-agent:latest',
                'port': 8023
            }
        }
    
    async def select_agents(
        self,
        task_plan: List[Dict[str, Any]],
        available_resources: Dict[str, float],
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select optimal agents for the given task"""
        selected_agents = []
        resource_allocation = {}
        
        # Analyze task requirements
        required_capabilities = set()
        for task in task_plan:
            required_capabilities.update(task.get('required_capabilities', []))
        
        # Score agents based on capability match and past performance
        agent_scores = {}
        for agent_type, agent_info in self.agent_registry.items():
            score = self._calculate_agent_score(
                agent_info,
                required_capabilities,
                memories
            )
            agent_scores[agent_type] = score
        
        # Select top agents within resource constraints
        sorted_agents = sorted(
            agent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        total_cpu = 0
        total_memory = 0
        total_gpu = 0
        
        for agent_type, score in sorted_agents:
            agent_info = self.agent_registry[agent_type]
            req = agent_info['resource_requirements']
            
            # Check if we can fit this agent
            if (total_cpu + req['cpu'] <= available_resources['cpu_percent'] / 100 * self.config['cpu_cores'] and
                total_memory + req['memory'] <= available_resources['memory_available_gb'] and
                (not req['gpu'] or total_gpu + req['gpu'] <= self.config['gpu_memory_gb'])):
                
                selected_agents.append(agent_type)
                resource_allocation[agent_type.value] = req
                
                total_cpu += req['cpu']
                total_memory += req['memory']
                total_gpu += req['gpu']
                
                # Limit number of agents
                if len(selected_agents) >= self.config['max_concurrent_agents']:
                    break
        
        logger.info(f"🎯 Selected {len(selected_agents)} agents: {[a.value for a in selected_agents]}")
        
        return {
            'agents': selected_agents,
            'resources': resource_allocation,
            'total_resources': {
                'cpu': total_cpu,
                'memory': total_memory,
                'gpu': total_gpu
            }
        }
    
    def _calculate_agent_score(
        self,
        agent_info: Dict[str, Any],
        required_capabilities: set,
        memories: List[Dict[str, Any]]
    ) -> float:
        """Calculate agent selection score"""
        # Capability match score
        agent_caps = set(agent_info['capabilities'])
        capability_score = len(agent_caps.intersection(required_capabilities)) / max(len(required_capabilities), 1)
        
        # Historical performance score from memories
        historical_score = agent_info['success_rate']
        for memory in memories:
            if memory.get('metadata', {}).get('agent') == agent_info['name']:
                historical_score = (historical_score + memory.get('metadata', {}).get('score', 0.5)) / 2
        
        # Efficiency score
        efficiency_score = 1 / (1 + agent_info['average_latency'] / 10)
        
        # Combined score
        return (capability_score * 0.5 + historical_score * 0.3 + efficiency_score * 0.2)
    
    async def execute_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent task"""
        agent_type = task['agent_type']
        agent_info = self.agent_registry[agent_type]
        
        start_time = time.time()
        
        try:
            # Ensure agent container is running
            container = await self._ensure_agent_running(agent_type)
            
            # Prepare agent request
            agent_request = {
                'input': task['input'],
                'task_plan': task['task_plan'],
                'context': task['context']
            }
            
            # Call agent API
            response = await self.http_client.post(
                f"http://{container.name}:{agent_info['port']}/execute",
                json=agent_request
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Track metrics
            execution_time = time.time() - start_time
            self._update_agent_metrics(agent_type, execution_time, True)
            
            return {
                'agent': agent_type.value,
                'task_id': task.get('task_id', 'unknown'),
                'output': result.get('output'),
                'success': True,
                'error': None,
                'execution_time': execution_time,
                'resources_used': await self._get_container_resources(container),
                'quality_score': result.get('quality_score')
            }
            
        except Exception as e:
            logger.error(f"❌ Agent {agent_type.value} execution failed: {e}")
            
            execution_time = time.time() - start_time
            self._update_agent_metrics(agent_type, execution_time, False)
            
            return {
                'agent': agent_type.value,
                'task_id': task.get('task_id', 'unknown'),
                'output': None,
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'resources_used': {},
                'quality_score': 0.0
            }
    
    async def _ensure_agent_running(self, agent_type: AgentType) -> docker.models.containers.Container:
        """Ensure agent container is running"""
        agent_info = self.agent_registry[agent_type]
        container_name = f"sutazai-agent-{agent_type.value}"
        
        try:
            container = self.docker_client.containers.get(container_name)
            if container.status != 'running':
                container.start()
                await asyncio.sleep(5)  # Wait for startup
        except docker.errors.NotFound:
            # Create and start container
            container = self.docker_client.containers.run(
                agent_info['container'],
                name=container_name,
                detach=True,
                network='sutazai-network',
                environment={
                    'AGENT_TYPE': agent_type.value,
                    'OLLAMA_HOST': 'sutazai-ollama:11434'
                },
                restart_policy={'Name': 'unless-stopped'}
            )
            await asyncio.sleep(10)  # Wait for initialization
        
        return container
    
    async def _get_container_resources(self, container) -> Dict[str, float]:
        """Get container resource usage"""
        stats = container.stats(stream=False)
        
        # Calculate CPU percentage
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
        cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
        
        # Calculate memory usage
        memory_usage = stats['memory_stats']['usage'] / (1024 * 1024 * 1024)  # Convert to GB
        
        return {
            'cpu_percent': cpu_percent,
            'memory_gb': memory_usage
        }
    
    def _update_agent_metrics(self, agent_type: AgentType, execution_time: float, success: bool):
        """Update agent performance metrics"""
        if agent_type not in self.agent_metrics:
            self.agent_metrics[agent_type] = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'success_rate': 0.0
            }
        
        metrics = self.agent_metrics[agent_type]
        metrics['total_executions'] += 1
        metrics['total_time'] += execution_time
        
        if success:
            metrics['successful_executions'] += 1
        
        metrics['average_time'] = metrics['total_time'] / metrics['total_executions']
        metrics['success_rate'] = metrics['successful_executions'] / metrics['total_executions']
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.http_client.aclose()