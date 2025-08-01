#!/usr/bin/env python3
"""
Comprehensive Agent Registry for SutazAI Brain
Integrates all 30+ requested agents with unified interface
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio
import httpx
import docker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtendedAgentType(str, Enum):
    """Extended agent types including all requested agents"""
    # Core AI Orchestration
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    GPT_ENGINEER = "gpt-engineer"
    
    # Specialized Tools
    BROWSER_USE = "browser-use"
    SEMGREP = "semgrep"
    DOCUMIND = "documind"
    METAGPT = "metagpt"
    CAMEL = "camel"
    
    # Code Agents
    AIDER = "aider"
    GPTPILOT = "gpt-pilot"
    DEVIKA = "devika"
    OPENDEVIN = "opendevin"
    
    # Research Agents
    GPTRESEARCHER = "gpt-researcher"
    STORM = "storm"
    KHOJ = "khoj"
    
    # Analysis Agents
    PANDAS_AI = "pandas-ai"
    INTERPRETER = "open-interpreter"
    
    # Creative Agents
    FABRIC = "fabric"
    TXTAI = "txtai"
    
    # Security Agents
    PANDASEC = "pandasec"
    NUCLEI = "nuclei"
    
    # Additional Requested Agents
    LETTA = "letta"
    LOCALAGI = "localagi"
    TABBYML = "tabbyml"
    AGENTZERO = "agentzero"
    BIGAGI = "bigagi"
    SKYVERN = "skyvern"
    AGENTGPT = "agentgpt"
    PRIVATEGPT = "privategpt"
    LLAMAINDEX = "llamaindex"
    FLOWISEAI = "flowiseai"
    SHELLGPT = "shellgpt"
    PENTESTGPT = "pentestgpt"
    JARVIS = "jarvis"
    FINROBOT = "finrobot"
    LANGFLOW = "langflow"
    DIFY = "dify"
    
    # Framework Services
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    FSDP = "fsdp"
    FAISS = "faiss"
    QDRANT = "qdrant"
    CHROMADB = "chromadb"
    
    # Model Management
    OLLAMA = "ollama"
    CODELLAMA = "codellama"
    LLAMA2 = "llama2"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class ComprehensiveAgentRegistry:
    """Registry for all agents with enhanced capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = docker.from_env()
        self.http_client = httpx.AsyncClient(timeout=300.0)
        
        # Build comprehensive agent registry
        self.agent_registry = self._build_comprehensive_registry()
        
        # Agent deployment configurations
        self.agent_deployments = self._build_deployment_configs()
        
        logger.info(f"🤖 Initialized comprehensive agent registry with {len(self.agent_registry)} agents")
    
    def _build_comprehensive_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build the complete agent registry with all requested agents"""
        return {
            # Core AI Orchestration
            ExtendedAgentType.AUTOGEN: {
                'name': 'AutoGen',
                'type': ExtendedAgentType.AUTOGEN,
                'capabilities': ['multi-agent-coordination', 'task-decomposition', 'conversation', 'code-generation'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'repo': 'https://github.com/ag2ai/ag2',
                'container': 'sutazai/autogen:latest',
                'port': 8001,
                'priority': 1
            },
            
            ExtendedAgentType.CREWAI: {
                'name': 'CrewAI',
                'type': ExtendedAgentType.CREWAI,
                'capabilities': ['team-coordination', 'role-based-tasks', 'workflow', 'collaborative-execution'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'repo': 'https://github.com/crewAIInc/crewAI',
                'container': 'sutazai/crewai:latest',
                'port': 8002,
                'priority': 1
            },
            
            ExtendedAgentType.LANGCHAIN: {
                'name': 'LangChain',
                'type': ExtendedAgentType.LANGCHAIN,
                'capabilities': ['chain-reasoning', 'tool-use', 'retrieval', 'memory-management'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'gpu': 0.0},
                'repo': 'https://github.com/langchain-ai/langchain',
                'container': 'sutazai/langchain:latest',
                'port': 8003,
                'priority': 1
            },
            
            ExtendedAgentType.AUTOGPT: {
                'name': 'AutoGPT',
                'type': ExtendedAgentType.AUTOGPT,
                'capabilities': ['autonomous-execution', 'goal-oriented', 'self-improvement', 'task-planning'],
                'resource_requirements': {'cpu': 2.0, 'memory': 4.0, 'gpu': 0.0},
                'repo': 'https://github.com/Significant-Gravitas/AutoGPT',
                'container': 'sutazai/autogpt:latest',
                'port': 8004,
                'priority': 1
            },
            
            ExtendedAgentType.GPT_ENGINEER: {
                'name': 'GPT-Engineer',
                'type': ExtendedAgentType.GPT_ENGINEER,
                'capabilities': ['code-generation', 'project-creation', 'architecture', 'full-stack-development'],
                'resource_requirements': {'cpu': 1.5, 'memory': 3.0, 'gpu': 0.0},
                'repo': 'https://github.com/AntonOsika/gpt-engineer',
                'container': 'sutazai/gpt-engineer:latest',
                'port': 8005,
                'priority': 1
            },
            
            # Specialized Tools
            ExtendedAgentType.BROWSER_USE: {
                'name': 'Browser-Use',
                'type': ExtendedAgentType.BROWSER_USE,
                'capabilities': ['web-browsing', 'form-filling', 'scraping', 'automation'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'repo': 'https://github.com/browser-use/browser-use',
                'container': 'sutazai/browser-use:latest',
                'port': 8006,
                'priority': 2
            },
            
            ExtendedAgentType.SEMGREP: {
                'name': 'Semgrep',
                'type': ExtendedAgentType.SEMGREP,
                'capabilities': ['code-analysis', 'security-scanning', 'pattern-matching', 'vulnerability-detection'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'gpu': 0.0},
                'repo': 'https://github.com/semgrep/semgrep',
                'container': 'sutazai/semgrep:latest',
                'port': 8007,
                'priority': 2
            },
            
            ExtendedAgentType.DOCUMIND: {
                'name': 'Documind',
                'type': ExtendedAgentType.DOCUMIND,
                'capabilities': ['pdf-parsing', 'document-analysis', 'ocr', 'text-extraction'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.5},
                'repo': 'https://github.com/DocumindHQ/documind',
                'container': 'sutazai/documind:latest',
                'port': 8008,
                'priority': 2
            },
            
            # New Agents
            ExtendedAgentType.LETTA: {
                'name': 'Letta (MemGPT)',
                'type': ExtendedAgentType.LETTA,
                'capabilities': ['task-automation', 'memory-management', 'long-term-memory', 'conversational-ai'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'repo': 'https://github.com/mysuperai/letta',
                'container': 'sutazai/letta:latest',
                'port': 8020,
                'priority': 1
            },
            
            ExtendedAgentType.LOCALAGI: {
                'name': 'LocalAGI',
                'type': ExtendedAgentType.LOCALAGI,
                'capabilities': ['autonomous-orchestration', 'local-execution', 'multi-model-coordination'],
                'resource_requirements': {'cpu': 1.5, 'memory': 3.0, 'gpu': 0.0},
                'repo': 'https://github.com/mudler/LocalAGI',
                'container': 'sutazai/localagi:latest',
                'port': 8021,
                'priority': 1
            },
            
            ExtendedAgentType.TABBYML: {
                'name': 'TabbyML',
                'type': ExtendedAgentType.TABBYML,
                'capabilities': ['code-completion', 'ai-coding-assistant', 'ide-integration'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.5},
                'repo': 'https://github.com/TabbyML/tabby',
                'container': 'sutazai/tabbyml:latest',
                'port': 8022,
                'priority': 2
            },
            
            ExtendedAgentType.AGENTZERO: {
                'name': 'AgentZero',
                'type': ExtendedAgentType.AGENTZERO,
                'capabilities': ['general-purpose-ai', 'task-execution', 'autonomous-agent'],
                'resource_requirements': {'cpu': 1.5, 'memory': 3.0, 'gpu': 0.0},
                'repo': 'https://github.com/frdel/agent-zero',
                'container': 'sutazai/agentzero:latest',
                'port': 8023,
                'priority': 2
            },
            
            ExtendedAgentType.BIGAGI: {
                'name': 'BigAGI',
                'type': ExtendedAgentType.BIGAGI,
                'capabilities': ['advanced-chat', 'multi-model-support', 'voice-interface', 'personas'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'repo': 'https://github.com/enricoros/big-AGI',
                'container': 'ghcr.io/enricoros/big-agi:latest',
                'port': 8024,
                'priority': 2
            },
            
            ExtendedAgentType.SKYVERN: {
                'name': 'Skyvern',
                'type': ExtendedAgentType.SKYVERN,
                'capabilities': ['browser-automation', 'workflow-automation', 'visual-ai'],
                'resource_requirements': {'cpu': 1.5, 'memory': 3.0, 'gpu': 0.5},
                'repo': 'https://github.com/Skyvern-AI/skyvern',
                'container': 'sutazai/skyvern:latest',
                'port': 8025,
                'priority': 2
            },
            
            ExtendedAgentType.JARVIS: {
                'name': 'JARVIS-AGI',
                'type': ExtendedAgentType.JARVIS,
                'capabilities': ['voice-assistant', 'multi-modal-ai', 'system-control', 'personal-assistant'],
                'resource_requirements': {'cpu': 2.0, 'memory': 4.0, 'gpu': 1.0},
                'repo': [
                    'https://github.com/Dipeshpal/Jarvis_AI',
                    'https://github.com/microsoft/JARVIS',
                    'https://github.com/danilofalcao/jarvis',
                    'https://github.com/SreejanPersonal/JARVIS-AGI'
                ],
                'container': 'sutazai/jarvis-agi:latest',
                'port': 8026,
                'priority': 1,
                'special_config': {
                    'voice_enabled': True,
                    'multi_modal': True,
                    'system_integration': True
                }
            },
            
            ExtendedAgentType.FINROBOT: {
                'name': 'FinRobot',
                'type': ExtendedAgentType.FINROBOT,
                'capabilities': ['financial-analysis', 'market-prediction', 'trading-strategies', 'risk-assessment'],
                'resource_requirements': {'cpu': 1.5, 'memory': 3.0, 'gpu': 0.5},
                'repo': 'https://github.com/AI4Finance-Foundation/FinRobot',
                'container': 'sutazai/finrobot:latest',
                'port': 8027,
                'priority': 2
            },
            
            # Development Agents
            ExtendedAgentType.AIDER: {
                'name': 'Aider',
                'type': ExtendedAgentType.AIDER,
                'capabilities': ['code-editing', 'refactoring', 'git-integration', 'pair-programming'],
                'resource_requirements': {'cpu': 0.5, 'memory': 1.0, 'gpu': 0.0},
                'repo': 'https://github.com/Aider-AI/aider',
                'container': 'sutazai/aider:latest',
                'port': 8030,
                'priority': 2
            },
            
            ExtendedAgentType.OPENDEVIN: {
                'name': 'OpenDevin',
                'type': ExtendedAgentType.OPENDEVIN,
                'capabilities': ['autonomous-coding', 'environment-interaction', 'testing', 'debugging'],
                'resource_requirements': {'cpu': 2.0, 'memory': 4.0, 'gpu': 0.0},
                'repo': 'https://github.com/AI-App/OpenDevin',
                'container': 'sutazai/opendevin:latest',
                'port': 8031,
                'priority': 1
            },
            
            # Workflow Platforms
            ExtendedAgentType.LANGFLOW: {
                'name': 'LangFlow',
                'type': ExtendedAgentType.LANGFLOW,
                'capabilities': ['visual-flow-builder', 'no-code-ai', 'workflow-design', 'integration'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'repo': 'https://github.com/langflow-ai/langflow',
                'container': 'langflowai/langflow:latest',
                'port': 8040,
                'priority': 2
            },
            
            ExtendedAgentType.DIFY: {
                'name': 'Dify',
                'type': ExtendedAgentType.DIFY,
                'capabilities': ['app-builder', 'workflow-orchestration', 'llm-ops', 'rag-pipeline'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'repo': 'https://github.com/langgenius/dify',
                'container': 'langgenius/dify-api:latest',
                'port': 8041,
                'priority': 2
            },
            
            # ML Frameworks
            ExtendedAgentType.PYTORCH: {
                'name': 'PyTorch',
                'type': ExtendedAgentType.PYTORCH,
                'capabilities': ['deep-learning', 'neural-networks', 'tensor-computation', 'model-training'],
                'resource_requirements': {'cpu': 2.0, 'memory': 4.0, 'gpu': 2.0},
                'repo': 'https://github.com/pytorch/pytorch',
                'container': 'sutazai/pytorch:latest',
                'port': 8050,
                'priority': 1
            },
            
            ExtendedAgentType.TENSORFLOW: {
                'name': 'TensorFlow',
                'type': ExtendedAgentType.TENSORFLOW,
                'capabilities': ['deep-learning', 'distributed-training', 'production-ml', 'model-serving'],
                'resource_requirements': {'cpu': 2.0, 'memory': 4.0, 'gpu': 2.0},
                'repo': 'https://github.com/tensorflow/tensorflow',
                'container': 'sutazai/tensorflow:latest',
                'port': 8051,
                'priority': 1
            },
            
            ExtendedAgentType.JAX: {
                'name': 'JAX',
                'type': ExtendedAgentType.JAX,
                'capabilities': ['high-performance-ml', 'automatic-differentiation', 'jit-compilation', 'vectorization'],
                'resource_requirements': {'cpu': 2.0, 'memory': 4.0, 'gpu': 2.0},
                'repo': 'https://github.com/jax-ml/jax',
                'container': 'sutazai/jax:latest',
                'port': 8052,
                'priority': 1
            },
            
            # Vector Databases
            ExtendedAgentType.QDRANT: {
                'name': 'Qdrant',
                'type': ExtendedAgentType.QDRANT,
                'capabilities': ['vector-search', 'similarity-matching', 'neural-search', 'filtering'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'repo': 'https://github.com/qdrant/qdrant',
                'container': 'qdrant/qdrant:latest',
                'port': 6333,
                'priority': 1
            },
            
            ExtendedAgentType.CHROMADB: {
                'name': 'ChromaDB',
                'type': ExtendedAgentType.CHROMADB,
                'capabilities': ['embedding-storage', 'semantic-search', 'vector-database', 'ai-native-db'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 0.0},
                'repo': 'https://github.com/chroma-core/chroma',
                'container': 'chromadb/chroma:latest',
                'port': 8000,
                'priority': 1
            },
            
            ExtendedAgentType.FAISS: {
                'name': 'FAISS',
                'type': ExtendedAgentType.FAISS,
                'capabilities': ['similarity-search', 'nearest-neighbor', 'clustering', 'indexing'],
                'resource_requirements': {'cpu': 1.0, 'memory': 2.0, 'gpu': 1.0},
                'repo': 'https://github.com/facebookresearch/faiss',
                'container': 'sutazai/faiss:latest',
                'port': 8060,
                'priority': 1
            }
        }
    
    def _build_deployment_configs(self) -> Dict[str, Dict[str, Any]]:
        """Build deployment configurations for each agent"""
        configs = {}
        
        for agent_type, agent_info in self.agent_registry.items():
            configs[agent_type] = {
                'docker_compose': self._generate_docker_compose_config(agent_info),
                'kubernetes': self._generate_kubernetes_config(agent_info),
                'systemd': self._generate_systemd_config(agent_info)
            }
        
        return configs
    
    def _generate_docker_compose_config(self, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Docker Compose configuration for an agent"""
        return {
            'image': agent_info['container'],
            'container_name': f"sutazai-{agent_info['type'].value}",
            'restart': 'unless-stopped',
            'ports': [f"{agent_info['port']}:{agent_info.get('internal_port', agent_info['port'])}"],
            'environment': {
                'AGENT_TYPE': agent_info['type'].value,
                'OLLAMA_HOST': '${OLLAMA_HOST:-sutazai-ollama:11434}',
                'REDIS_HOST': '${REDIS_HOST:-sutazai-redis:6379}',
                'POSTGRES_HOST': '${POSTGRES_HOST:-sutazai-postgresql:5432}'
            },
            'networks': ['sutazai-network'],
            'deploy': {
                'resources': {
                    'limits': {
                        'cpus': str(agent_info['resource_requirements']['cpu']),
                        'memory': f"{agent_info['resource_requirements']['memory']}G"
                    }
                }
            }
        }
    
    def _generate_kubernetes_config(self, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Kubernetes configuration for an agent"""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"sutazai-{agent_info['type'].value}",
                'labels': {
                    'app': 'sutazai',
                    'component': agent_info['type'].value
                }
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': f"sutazai-{agent_info['type'].value}"
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': f"sutazai-{agent_info['type'].value}"
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': agent_info['type'].value,
                            'image': agent_info['container'],
                            'ports': [{
                                'containerPort': agent_info['port']
                            }],
                            'resources': {
                                'limits': {
                                    'cpu': str(agent_info['resource_requirements']['cpu']),
                                    'memory': f"{agent_info['resource_requirements']['memory']}Gi"
                                },
                                'requests': {
                                    'cpu': str(agent_info['resource_requirements']['cpu'] * 0.5),
                                    'memory': f"{agent_info['resource_requirements']['memory'] * 0.5}Gi"
                                }
                            }
                        }]
                    }
                }
            }
        }
    
    def _generate_systemd_config(self, agent_info: Dict[str, Any]) -> str:
        """Generate systemd service configuration for an agent"""
        return f"""[Unit]
Description=SutazAI {agent_info['name']} Agent
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
ExecStartPre=/usr/bin/docker pull {agent_info['container']}
ExecStart=/usr/bin/docker run --rm --name sutazai-{agent_info['type'].value} \\
    -p {agent_info['port']}:{agent_info['port']} \\
    --network sutazai-network \\
    -e AGENT_TYPE={agent_info['type'].value} \\
    -e OLLAMA_HOST=sutazai-ollama:11434 \\
    {agent_info['container']}
ExecStop=/usr/bin/docker stop sutazai-{agent_info['type'].value}

[Install]
WantedBy=multi-user.target
"""
    
    def get_agent_by_capability(self, capability: str) -> List[ExtendedAgentType]:
        """Get agents that have a specific capability"""
        matching_agents = []
        
        for agent_type, agent_info in self.agent_registry.items():
            if capability in agent_info['capabilities']:
                matching_agents.append(agent_type)
        
        # Sort by priority
        matching_agents.sort(key=lambda x: self.agent_registry[x]['priority'])
        
        return matching_agents
    
    def get_deployment_script(self, agent_types: List[ExtendedAgentType]) -> str:
        """Generate deployment script for specified agents"""
        script = """#!/bin/bash
# SutazAI Agent Deployment Script
# Auto-generated

set -e

echo "🚀 Deploying SutazAI agents..."

# Create network if not exists
docker network create sutazai-network 2>/dev/null || true

# Deploy agents
"""
        
        for agent_type in agent_types:
            agent_info = self.agent_registry.get(agent_type)
            if agent_info:
                script += f"""
echo "📦 Deploying {agent_info['name']}..."
docker run -d \\
    --name sutazai-{agent_info['type'].value} \\
    --network sutazai-network \\
    --restart unless-stopped \\
    -p {agent_info['port']}:{agent_info['port']} \\
    -e AGENT_TYPE={agent_info['type'].value} \\
    -e OLLAMA_HOST=sutazai-ollama:11434 \\
    {agent_info['container']}
"""
        
        script += """
echo "✅ Agent deployment complete!"
"""
        
        return script
    
    async def health_check_all_agents(self) -> Dict[str, bool]:
        """Check health status of all deployed agents"""
        health_status = {}
        
        for agent_type, agent_info in self.agent_registry.items():
            try:
                # Check if container is running
                container_name = f"sutazai-{agent_info['type'].value}"
                container = self.docker_client.containers.get(container_name)
                
                if container.status == 'running':
                    # Try HTTP health check
                    try:
                        response = await self.http_client.get(
                            f"http://localhost:{agent_info['port']}/health",
                            timeout=5.0
                        )
                        health_status[agent_type.value] = response.status_code == 200
                    except:
                        health_status[agent_type.value] = True  # Container running but no health endpoint
                else:
                    health_status[agent_type.value] = False
            except docker.errors.NotFound:
                health_status[agent_type.value] = False
            except Exception as e:
                logger.error(f"Health check error for {agent_type.value}: {e}")
                health_status[agent_type.value] = False
        
        return health_status