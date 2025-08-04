#!/usr/bin/env python3
"""
Agent Coordinator - Manages interaction with 131 AI agents
"""

import asyncio
import logging
import aiohttp
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AgentCoordinator:
    """Coordinates execution across multiple AI agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_url = config.get('backend_url', 'http://localhost:8000')
        self.max_concurrent = config.get('max_concurrent', 5)
        self.timeout = config.get('timeout', 60)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.session = None
        self.agent_registry = {}
        # Load comprehensive agent specialties from registry
        self.agent_specialties = self._load_agent_specialties()
        
        # Voice command patterns for intelligent routing
        self.voice_patterns = {
            # Development commands
            r'(code|develop|build|create|implement).*': ['senior-backend-developer', 'senior-frontend-developer', 'senior-full-stack-developer'],
            r'(backend|api|server|database).*': ['senior-backend-developer', 'ai-senior-backend-developer'],
            r'(frontend|ui|interface|react|vue|angular).*': ['senior-frontend-developer', 'ai-senior-frontend-developer'],
            r'(test|testing|qa|quality).*': ['testing-qa-validator', 'ai-testing-qa-validator'],
            
            # AI/ML commands
            r'(ai|artificial intelligence|machine learning|ml|deep learning).*': ['senior-ai-engineer', 'ai-senior-engineer'],
            r'(train|model|neural|algorithm).*': ['senior-ai-engineer', 'deep-learning-brain-architect'],
            r'(data analysis|statistics|analytics).*': ['private-data-analyst', 'data-analysis-engineer'],
            
            # Infrastructure commands
            r'(deploy|deployment|infrastructure|devops).*': ['infrastructure-devops-manager', 'deployment-automation-master'],
            r'(docker|container|kubernetes|k8s).*': ['container-orchestrator-k3s', 'container-vulnerability-scanner-trivy'],
            r'(monitor|monitoring|metrics|logs).*': ['observability-dashboard-manager-grafana', 'metrics-collector-prometheus'],
            
            # Security commands
            r'(security|secure|hack|pentest|vulnerability).*': ['security-pentesting-specialist', 'kali-security-specialist'],
            r'(scan|analyze code|security audit).*': ['semgrep-security-analyzer', 'adversarial-attack-detector'],
            
            # Specialized commands
            r'(ollama|llm|local model).*': ['ollama-integration-specialist'],
            r'(documentation|docs|knowledge|wiki).*': ['document-knowledge-manager'],
            r'(workflow|automation|process).*': ['langflow-workflow-designer', 'dify-automation-specialist'],
            r'(hardware|resource|optimize|performance).*': ['hardware-resource-optimizer', 'cpu-only-hardware-optimizer'],
            
            # Management commands
            r'(manage|coordinate|orchestrate|plan).*': ['agent-orchestrator', 'task-assignment-coordinator'],
            r'(product|project|scrum|agile).*': ['ai-product-manager', 'ai-scrum-master'],
        }
        
    async def initialize(self):
        """Initialize agent coordinator"""
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            
            # Discover available agents
            await self._discover_agents()
            
            logger.info(f"Agent Coordinator initialized with {len(self.agent_registry)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent coordinator: {e}")
            raise
            
    async def shutdown(self):
        """Shutdown agent coordinator"""
        if self.session:
            await self.session.close()
            
    async def _discover_agents(self):
        """Discover available agents from backend"""
        try:
            async with self.session.get(f"{self.backend_url}/api/agents") as response:
                if response.status == 200:
                    agents = await response.json()
                    for agent in agents:
                        self.agent_registry[agent['name']] = {
                            'url': agent.get('url', f"{self.backend_url}/api/agents/{agent['name']}"),
                            'status': agent.get('status', 'available'),
                            'capabilities': agent.get('capabilities', [])
                        }
                        
        except Exception as e:
            logger.warning(f"Could not discover agents from backend: {e}")
            # Use default agent list
            self._load_default_agents()
            
    def _load_default_agents(self):
        """Load default agent configuration"""
        for agent_name, specialties in self.agent_specialties.items():
            self.agent_registry[agent_name] = {
                'url': f"{self.backend_url}/api/agents/{agent_name}",
                'status': 'available',
                'capabilities': specialties
            }
            
    async def execute_plan(self, plan: Dict[str, Any], 
                          plugin_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a plan using appropriate agents"""
        try:
            steps = plan.get('steps', [])
            results = []
            agents_used = []
            
            # Execute steps with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def execute_step(step):
                async with semaphore:
                    return await self._execute_single_step(step, plugin_results)
                    
            # Execute all steps
            step_results = await asyncio.gather(
                *[execute_step(step) for step in steps],
                return_exceptions=True
            )
            
            # Process results
            success = True
            for i, result in enumerate(step_results):
                if isinstance(result, Exception):
                    logger.error(f"Step {i} failed: {result}")
                    results.append({
                        'step': i,
                        'error': str(result),
                        'success': False
                    })
                    success = False
                else:
                    results.append(result)
                    if result.get('agent'):
                        agents_used.append(result['agent'])
                        
            # Synthesize final result
            final_result = await self._synthesize_results(results, plan)
            
            return {
                'success': success,
                'result': final_result,
                'agents_used': list(set(agents_used)),
                'step_results': results
            }
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agents_used': []
            }
            
    async def _execute_single_step(self, step: Dict[str, Any], 
                                  plugin_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single step of the plan"""
        try:
            # Select appropriate agent
            agent_name = await self._select_agent(step)
            if not agent_name:
                return {
                    'step': step.get('id'),
                    'error': 'No suitable agent found',
                    'success': False
                }
                
            # Prepare task data
            task_data = {
                'type': step.get('type', 'general'),
                'description': step.get('description'),
                'input': step.get('input', {}),
                'context': {
                    'plugin_results': plugin_results,
                    'previous_steps': step.get('dependencies', [])
                }
            }
            
            # Execute task with retry
            for attempt in range(self.retry_attempts):
                try:
                    result = await self._call_agent(agent_name, task_data)
                    return {
                        'step': step.get('id'),
                        'agent': agent_name,
                        'result': result,
                        'success': True
                    }
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {
                'step': step.get('id'),
                'error': str(e),
                'success': False
            }
            
    async def _select_agent(self, step: Dict[str, Any]) -> Optional[str]:
        """Select the most appropriate agent for a step"""
        step_type = step.get('type', '').lower()
        required_capabilities = step.get('required_capabilities', [])
        
        # Find agents with matching capabilities
        candidates = []
        for agent_name, agent_info in self.agent_registry.items():
            if agent_info['status'] != 'available':
                continue
                
            # Check if agent has required capabilities
            agent_caps = agent_info.get('capabilities', [])
            agent_specs = self.agent_specialties.get(agent_name, [])
            
            # Score based on capability match
            score = 0
            for cap in required_capabilities:
                if cap in agent_caps or cap in agent_specs:
                    score += 2
                elif any(cap in spec for spec in agent_specs):
                    score += 1
                    
            # Check step type match
            if step_type in agent_specs:
                score += 3
                
            if score > 0:
                candidates.append((score, agent_name))
                
        # Select highest scoring agent
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
            
        # Default to general agent
        return 'agent-coordinator'
        
    async def _call_agent(self, agent_name: str, task_data: Dict[str, Any]) -> Any:
        """Call an agent to execute a task"""
        agent_info = self.agent_registry.get(agent_name)
        if not agent_info:
            raise ValueError(f"Agent {agent_name} not found")
            
        url = agent_info['url']
        
        try:
            async with self.session.post(url, json=task_data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    raise Exception(f"Agent returned {response.status}: {text}")
                    
        except Exception as e:
            logger.error(f"Failed to call agent {agent_name}: {e}")
            raise
            
    async def _synthesize_results(self, results: List[Dict[str, Any]], 
                                 plan: Dict[str, Any]) -> Any:
        """Synthesize results from multiple steps"""
        # Extract successful results
        successful_results = [
            r['result'] for r in results 
            if r.get('success') and 'result' in r
        ]
        
        if not successful_results:
            return "No successful results to synthesize"
            
        # If single result, return it
        if len(successful_results) == 1:
            return successful_results[0]
            
        # For multiple results, create summary
        synthesis = {
            'summary': plan.get('goal', 'Task completed'),
            'details': successful_results,
            'total_steps': len(results),
            'successful_steps': len(successful_results)
        }
        
        return synthesis
        
    async def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return [
            name for name, info in self.agent_registry.items()
            if info['status'] == 'available'
        ]
        
    async def get_agent_info(self) -> List[Dict[str, Any]]:
        """Get detailed agent information"""
        agent_list = []
        for name, info in self.agent_registry.items():
            agent_list.append({
                'name': name,
                'status': info['status'],
                'capabilities': info.get('capabilities', []),
                'specialties': self.agent_specialties.get(name, [])
            })
        return agent_list
        
    def _load_agent_specialties(self) -> Dict[str, List[str]]:
        """Load comprehensive agent specialties from the SutazAI registry"""
        try:
            # Load from agent registry file
            registry_path = Path('/opt/sutazaiapp/agents/agent_registry.json')
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                    
                specialties = {}
                for agent_name, agent_info in registry.get('agents', {}).items():
                    capabilities = agent_info.get('capabilities', [])
                    description = agent_info.get('description', '').lower()
                    
                    # Extract keywords from description
                    keywords = self._extract_keywords_from_description(description)
                    specialties[agent_name] = list(set(capabilities + keywords))
                    
                return specialties
        except Exception as e:
            logger.warning(f"Could not load agent registry: {e}")
            
        # Fallback to comprehensive manual mapping
        return {
            # Core system agents
            'agent-orchestrator': ['orchestration', 'coordination', 'planning', 'management'],
            'agent-creator': ['agent', 'creation', 'generation', 'development'],
            'agent-debugger': ['debugging', 'troubleshooting', 'analysis', 'fixing'],
            'task-assignment-coordinator': ['task', 'assignment', 'coordination', 'scheduling'],
            
            # Development agents
            'senior-backend-developer': ['backend', 'api', 'server', 'database', 'microservices'],
            'senior-frontend-developer': ['frontend', 'ui', 'react', 'javascript', 'css', 'html'],
            'senior-full-stack-developer': ['fullstack', 'web', 'application', 'development'],
            'senior-ai-engineer': ['ai', 'ml', 'machine learning', 'deep learning', 'neural networks'],
            'code-generation-improver': ['code', 'improvement', 'refactoring', 'optimization'],
            'code-improver': ['code', 'quality', 'cleanup', 'refactoring'],
            'opendevin-code-generator': ['code', 'generation', 'automation', 'development'],
            
            # Testing & QA agents
            'testing-qa-validator': ['testing', 'qa', 'quality', 'validation', 'verification'],
            'ai-testing-qa-validator': ['testing', 'qa', 'ai testing', 'automated testing'],
            'ai-qa-team-lead': ['qa', 'testing', 'leadership', 'quality assurance'],
            'comprehensive-testing-validator': ['testing', 'comprehensive', 'validation'],
            
            # Infrastructure & DevOps agents
            'infrastructure-devops-manager': ['infrastructure', 'devops', 'deployment', 'operations'],
            'deployment-automation-master': ['deployment', 'automation', 'ci/cd', 'release'],
            'container-orchestrator-k3s': ['kubernetes', 'k8s', 'containers', 'orchestration'],
            'container-vulnerability-scanner-trivy': ['security', 'vulnerability', 'scanning', 'containers'],
            'cicd-pipeline-orchestrator': ['ci/cd', 'pipeline', 'automation', 'integration'],
            
            # Security agents
            'security-pentesting-specialist': ['security', 'pentesting', 'penetration testing', 'hacking'],
            'kali-security-specialist': ['kali', 'security', 'hacking', 'penetration testing'],
            'semgrep-security-analyzer': ['security', 'code analysis', 'vulnerability', 'static analysis'],
            'adversarial-attack-detector': ['security', 'adversarial', 'attack', 'detection'],
            'bias-and-fairness-auditor': ['bias', 'fairness', 'ethics', 'auditing'],
            'prompt-injection-guard': ['security', 'prompt injection', 'ai security'],
            
            # AI/ML specialists
            'deep-learning-brain-architect': ['deep learning', 'neural networks', 'architecture', 'ai'],
            'deep-learning-brain-manager': ['deep learning', 'brain', 'management', 'ai'],
            'deep-local-brain-builder': ['local ai', 'brain', 'building', 'neural networks'],
            'neural-architecture-search': ['neural', 'architecture', 'search', 'optimization'],
            'evolution-strategy-trainer': ['evolution', 'strategy', 'training', 'optimization'],
            'genetic-algorithm-tuner': ['genetic', 'algorithm', 'tuning', 'optimization'],
            'quantum-ai-researcher': ['quantum', 'ai', 'research', 'quantum computing'],
            
            # Data & Analytics agents
            'private-data-analyst': ['data', 'analysis', 'analytics', 'private data'],
            'data-analysis-engineer': ['data', 'analysis', 'engineering', 'statistics'],
            'data-drift-detector': ['data', 'drift', 'detection', 'monitoring'],
            'data-lifecycle-manager': ['data', 'lifecycle', 'management', 'governance'],
            'data-version-controller-dvc': ['data', 'version control', 'dvc', 'versioning'],
            
            # Monitoring & Observability agents
            'observability-dashboard-manager-grafana': ['monitoring', 'grafana', 'dashboard', 'observability'],
            'metrics-collector-prometheus': ['metrics', 'prometheus', 'monitoring', 'collection'],
            'log-aggregator-loki': ['logs', 'loki', 'aggregation', 'monitoring'],
            'distributed-tracing-analyzer-jaeger': ['tracing', 'jaeger', 'distributed', 'analysis'],
            'runtime-behavior-anomaly-detector': ['runtime', 'anomaly', 'behavior', 'detection'],
            
            # Hardware & Resource agents
            'hardware-resource-optimizer': ['hardware', 'resource', 'optimization', 'performance'],
            'cpu-only-hardware-optimizer': ['cpu', 'hardware', 'optimization', 'performance'],
            'gpu-hardware-optimizer': ['gpu', 'hardware', 'optimization', 'performance'],
            'ram-hardware-optimizer': ['ram', 'memory', 'optimization', 'performance'],
            'compute-scheduler-and-optimizer': ['compute', 'scheduling', 'optimization', 'resources'],
            'energy-consumption-optimize': ['energy', 'consumption', 'optimization', 'efficiency'],
            
            # Specialized AI agents
            'ollama-integration-specialist': ['ollama', 'llm', 'local models', 'integration'],
            'document-knowledge-manager': ['documentation', 'knowledge', 'management', 'rag'],
            'langflow-workflow-designer': ['langflow', 'workflow', 'design', 'automation'],
            'dify-automation-specialist': ['dify', 'automation', 'workflow', 'ai apps'],
            'flowiseai-flow-manager': ['flowise', 'flow', 'management', 'ai workflows'],
            
            # Management & Coordination agents
            'ai-product-manager': ['product', 'management', 'planning', 'strategy'],
            'ai-scrum-master': ['scrum', 'agile', 'project management', 'coordination'],
            'product-manager': ['product', 'management', 'strategy', 'planning'],
            'scrum-master': ['scrum', 'agile', 'project management'],
            'codebase-team-lead': ['team', 'leadership', 'codebase', 'management'],
            
            # Automation agents
            'autonomous-task-executor': ['autonomous', 'task', 'execution', 'automation'],
            'automated-incident-responder': ['incident', 'response', 'automation', 'ops'],
            'garbage-collector': ['cleanup', 'garbage collection', 'maintenance', 'optimization'],
            'emergency-shutdown-coordinator': ['emergency', 'shutdown', 'coordination', 'safety'],
            
            # Voice interface (self-reference)
            'jarvis-voice-interface': ['voice', 'speech', 'interface', 'natural language', 'assistant']
        }
        
    def _extract_keywords_from_description(self, description: str) -> List[str]:
        """Extract relevant keywords from agent description"""
        keywords = []
        
        # Common technical terms to extract
        tech_terms = [
            'frontend', 'backend', 'fullstack', 'database', 'api', 'microservices',
            'docker', 'kubernetes', 'ci/cd', 'deployment', 'monitoring', 'security',
            'machine learning', 'deep learning', 'ai', 'neural networks', 'data',
            'testing', 'qa', 'automation', 'workflow', 'optimization', 'performance'
        ]
        
        for term in tech_terms:
            if term in description:
                keywords.append(term)
                
        return keywords
        
    async def route_voice_command(self, command: str) -> List[str]:
        """Route voice command to appropriate agents using pattern matching"""
        command_lower = command.lower()
        candidate_agents = []
        
        # Check voice patterns
        for pattern, agents in self.voice_patterns.items():
            if re.match(pattern, command_lower):
                candidate_agents.extend(agents)
                
        # If no patterns match, use keyword-based routing
        if not candidate_agents:
            candidate_agents = await self._keyword_based_routing(command_lower)
            
        # Filter to available agents
        available_candidates = [
            agent for agent in candidate_agents 
            if agent in self.agent_registry and 
            self.agent_registry[agent]['status'] == 'available'
        ]
        
        # Return top 3 most suitable agents
        return available_candidates[:3] if available_candidates else ['agent-orchestrator']
        
    async def _keyword_based_routing(self, command: str) -> List[str]:
        """Route command based on keyword matching with agent specialties"""
        candidates = []
        
        for agent_name, specialties in self.agent_specialties.items():
            score = 0
            for specialty in specialties:
                if specialty.lower() in command:
                    score += len(specialty.split())  # Prefer longer, more specific matches
                    
            if score > 0:
                candidates.append((score, agent_name))
                
        # Sort by score and return agent names
        candidates.sort(reverse=True)
        return [agent for _, agent in candidates[:5]]
        
    async def process_voice_command(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a voice command with intelligent agent routing"""
        try:
            # Route command to appropriate agents
            suggested_agents = await self.route_voice_command(command)
            
            logger.info(f"Voice command '{command}' routed to agents: {suggested_agents}")
            
            # Create execution plan
            plan = {
                'goal': command,
                'steps': [{
                    'id': 'voice_command_execution',
                    'type': 'voice_command',
                    'description': command,
                    'input': {'command': command, 'context': context or {}},
                    'suggested_agents': suggested_agents
                }]
            }
            
            # Execute using the first available agent
            result = await self.execute_plan(plan)
            
            return {
                'command': command,
                'suggested_agents': suggested_agents,
                'execution_result': result,
                'success': result.get('success', False)
            }
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            return {
                'command': command,
                'error': str(e),
                'success': False
            }