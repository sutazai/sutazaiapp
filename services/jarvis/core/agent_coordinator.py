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
        self.agent_specialties = {
            # Core agents
            'agent-coordinator': ['orchestration', 'planning'],
            'task-manager': ['task', 'scheduling'],
            'resource-monitor': ['monitoring', 'resources'],
            
            # AI/ML agents
            'senior-ai-engineer': ['ai', 'ml', 'deep-learning'],
            'ml-engineer': ['machine-learning', 'training'],
            'data-scientist': ['analysis', 'statistics'],
            
            # Development agents
            'senior-backend-developer': ['backend', 'api', 'database'],
            'senior-frontend-developer': ['frontend', 'ui', 'react'],
            'senior-full-stack-developer': ['fullstack', 'web'],
            
            # Specialized agents
            'ollama-integration-specialist': ['ollama', 'llm'],
            'container-orchestrator-k3s': ['kubernetes', 'containers'],
            'security-pentesting-specialist': ['security', 'pentesting'],
            
            # Add more agent specialties as needed
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