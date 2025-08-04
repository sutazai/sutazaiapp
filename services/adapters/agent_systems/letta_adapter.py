"""
Letta (formerly MemGPT) adapter for autonomous agent operations
"""
from typing import Dict, Any, List, Optional
import aiohttp
import json
from ..base_adapter import ServiceAdapter
import logging

logger = logging.getLogger(__name__)


class LettaAdapter(ServiceAdapter):
    """Adapter for Letta (MemGPT) agent system"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Letta", config)
        self.api_endpoint = config.get('api_endpoint', 'http://letta:8283')
        self.default_persona = config.get('default_persona', 'assistant')
        self.default_human = config.get('default_human', 'user')
        self.memory_path = config.get('memory_path', '/data/letta/memories')
        self.agents = {}
        
    async def initialize(self):
        """Initialize Letta connection"""
        try:
            await self.connect()
            
            # Verify API is accessible
            response = await self._make_request('GET', '/api/config')
            if response:
                logger.info(f"Letta initialized successfully")
            else:
                raise Exception("Failed to connect to Letta API")
                
        except Exception as e:
            logger.error(f"Failed to initialize Letta: {str(e)}")
            raise
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get Letta capabilities"""
        return {
            'service': 'Letta',
            'type': 'agent_system',
            'features': [
                'autonomous_agents',
                'long_term_memory',
                'contextual_conversation',
                'task_execution',
                'persona_management',
                'memory_persistence',
                'multi_agent_coordination'
            ],
            'api_endpoint': self.api_endpoint,
            'active_agents': len(self.agents),
            'memory_backend': 'persistent'
        }
        
    async def create_agent(self,
                          agent_id: str,
                          persona: Optional[str] = None,
                          human: Optional[str] = None,
                          memory_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new Letta agent"""
        try:
            agent_config = {
                'agent_id': agent_id,
                'persona': persona or self.default_persona,
                'human': human or self.default_human,
                'model': 'gpt-4',
                'context_window': 8192,
                'memory': memory_config or {
                    'type': 'core_memory',
                    'limit': 2000
                }
            }
            
            response = await self._make_request(
                'POST',
                '/api/agents',
                json=agent_config
            )
            
            if response:
                self.agents[agent_id] = response
                return {
                    'success': True,
                    'agent_id': agent_id,
                    'config': agent_config
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create agent'
                }
                
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def send_message(self,
                          agent_id: str,
                          message: str,
                          include_memory: bool = True) -> Dict[str, Any]:
        """Send a message to an agent"""
        try:
            if agent_id not in self.agents:
                return {
                    'success': False,
                    'error': f"Agent {agent_id} not found"
                }
                
            payload = {
                'message': message,
                'include_memory': include_memory,
                'stream': False
            }
            
            response = await self._make_request(
                'POST',
                f'/api/agents/{agent_id}/messages',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'response': response.get('message', ''),
                    'memory_used': response.get('memory_pressure', 0),
                    'function_calls': response.get('function_calls', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get response'
                }
                
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        """Get agent's memory state"""
        try:
            response = await self._make_request(
                'GET',
                f'/api/agents/{agent_id}/memory'
            )
            
            if response:
                return {
                    'success': True,
                    'core_memory': response.get('core_memory', {}),
                    'recall_memory': response.get('recall_memory', []),
                    'archival_memory': response.get('archival_memory', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get memory'
                }
                
        except Exception as e:
            logger.error(f"Failed to get memory: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def update_agent_memory(self,
                                 agent_id: str,
                                 memory_type: str,
                                 content: str) -> Dict[str, Any]:
        """Update agent's memory"""
        try:
            payload = {
                'memory_type': memory_type,
                'content': content
            }
            
            response = await self._make_request(
                'PUT',
                f'/api/agents/{agent_id}/memory',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'updated': True
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to update memory'
                }
                
        except Exception as e:
            logger.error(f"Failed to update memory: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def execute_task(self,
                          agent_id: str,
                          task: str,
                          context: Optional[Dict] = None) -> Dict[str, Any]:
        """Have an agent execute a specific task"""
        try:
            payload = {
                'task': task,
                'context': context or {},
                'max_steps': 10
            }
            
            response = await self._make_request(
                'POST',
                f'/api/agents/{agent_id}/tasks',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'result': response.get('result', ''),
                    'steps_taken': response.get('steps', []),
                    'success': response.get('task_success', False)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to execute task'
                }
                
        except Exception as e:
            logger.error(f"Failed to execute task: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def list_agents(self) -> Dict[str, Any]:
        """List all active agents"""
        try:
            response = await self._make_request('GET', '/api/agents')
            
            if response:
                return {
                    'success': True,
                    'agents': response.get('agents', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to list agents'
                }
                
        except Exception as e:
            logger.error(f"Failed to list agents: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """Delete an agent"""
        try:
            response = await self._make_request(
                'DELETE',
                f'/api/agents/{agent_id}'
            )
            
            if response:
                if agent_id in self.agents:
                    del self.agents[agent_id]
                return {
                    'success': True,
                    'deleted': True
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to delete agent'
                }
                
        except Exception as e:
            logger.error(f"Failed to delete agent: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def save_conversation(self, 
                               agent_id: str,
                               filename: str) -> Dict[str, Any]:
        """Save agent conversation history"""
        try:
            response = await self._make_request(
                'POST',
                f'/api/agents/{agent_id}/save',
                json={'filename': filename}
            )
            
            if response:
                return {
                    'success': True,
                    'saved_to': response.get('path', filename)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to save conversation'
                }
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }