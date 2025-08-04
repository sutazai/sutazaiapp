"""
AutoGPT adapter for autonomous task execution
"""
from typing import Dict, Any, List, Optional
import asyncio
from ..base_adapter import ServiceAdapter
import logging
import json

logger = logging.getLogger(__name__)


class AutoGPTAdapter(ServiceAdapter):
    """Adapter for AutoGPT autonomous agent system"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AutoGPT", config)
        self.workspace_path = config.get('workspace_path', '/data/autogpt/workspaces')
        self.plugins_path = config.get('plugins_path', '/data/autogpt/plugins')
        self.memory_backend = config.get('memory_backend', 'redis')
        self.agents = {}
        
    async def initialize(self):
        """Initialize AutoGPT environment"""
        try:
            await self.connect()
            
            # Verify AutoGPT API is accessible
            response = await self._make_request('GET', '/api/v1/agents')
            if response:
                logger.info("AutoGPT initialized successfully")
            else:
                raise Exception("Failed to connect to AutoGPT API")
                
        except Exception as e:
            logger.error(f"Failed to initialize AutoGPT: {str(e)}")
            raise
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get AutoGPT capabilities"""
        plugins = await self._get_available_plugins()
        
        return {
            'service': 'AutoGPT',
            'type': 'agent_system',
            'features': [
                'autonomous_task_execution',
                'goal_oriented_planning',
                'web_browsing',
                'file_operations',
                'code_execution',
                'plugin_system',
                'memory_management',
                'continuous_mode'
            ],
            'workspace_path': self.workspace_path,
            'active_agents': len(self.agents),
            'available_plugins': plugins,
            'memory_backend': self.memory_backend
        }
        
    async def create_agent(self,
                          name: str,
                          role: str,
                          goals: List[str],
                          constraints: Optional[List[str]] = None,
                          resources: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new AutoGPT agent"""
        try:
            agent_config = {
                'name': name,
                'role': role,
                'goals': goals,
                'constraints': constraints or [
                    "Do not perform actions that could be harmful",
                    "Stay within the defined workspace",
                    "Respect rate limits and API quotas"
                ],
                'resources': resources or [
                    "Internet access for searches and information gathering",
                    "File system access within workspace",
                    "Access to approved plugins"
                ],
                'settings': {
                    'continuous_mode': False,
                    'speak_mode': False,
                    'fast_llm_model': 'gpt-3.5-turbo',
                    'smart_llm_model': 'gpt-4',
                    'memory_backend': self.memory_backend
                }
            }
            
            response = await self._make_request(
                'POST',
                '/api/v1/agents',
                json=agent_config
            )
            
            if response and 'agent_id' in response:
                agent_id = response['agent_id']
                self.agents[agent_id] = {
                    'name': name,
                    'config': agent_config,
                    'status': 'created'
                }
                
                return {
                    'success': True,
                    'agent_id': agent_id,
                    'workspace': f"{self.workspace_path}/{agent_id}"
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
            
    async def start_agent(self, agent_id: str) -> Dict[str, Any]:
        """Start an AutoGPT agent"""
        try:
            response = await self._make_request(
                'POST',
                f'/api/v1/agents/{agent_id}/start'
            )
            
            if response:
                self.agents[agent_id]['status'] = 'running'
                return {
                    'success': True,
                    'status': 'running'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to start agent'
                }
                
        except Exception as e:
            logger.error(f"Failed to start agent: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def stop_agent(self, agent_id: str) -> Dict[str, Any]:
        """Stop an AutoGPT agent"""
        try:
            response = await self._make_request(
                'POST',
                f'/api/v1/agents/{agent_id}/stop'
            )
            
            if response:
                self.agents[agent_id]['status'] = 'stopped'
                return {
                    'success': True,
                    'status': 'stopped'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to stop agent'
                }
                
        except Exception as e:
            logger.error(f"Failed to stop agent: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def send_message(self, 
                          agent_id: str,
                          message: str,
                          wait_for_response: bool = True) -> Dict[str, Any]:
        """Send a message to an agent"""
        try:
            payload = {
                'message': message,
                'wait_for_response': wait_for_response
            }
            
            response = await self._make_request(
                'POST',
                f'/api/v1/agents/{agent_id}/messages',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'response': response.get('response', ''),
                    'thoughts': response.get('thoughts', {}),
                    'criticism': response.get('criticism', ''),
                    'next_action': response.get('next_action', {})
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to send message'
                }
                
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status and progress"""
        try:
            response = await self._make_request(
                'GET',
                f'/api/v1/agents/{agent_id}/status'
            )
            
            if response:
                return {
                    'success': True,
                    'status': response.get('status', 'unknown'),
                    'current_task': response.get('current_task', ''),
                    'completed_tasks': response.get('completed_tasks', []),
                    'memory_usage': response.get('memory_usage', {}),
                    'tokens_used': response.get('tokens_used', 0)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get status'
                }
                
        except Exception as e:
            logger.error(f"Failed to get status: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_agent_logs(self, 
                           agent_id: str,
                           limit: int = 100) -> Dict[str, Any]:
        """Get agent execution logs"""
        try:
            response = await self._make_request(
                'GET',
                f'/api/v1/agents/{agent_id}/logs',
                params={'limit': limit}
            )
            
            if response:
                return {
                    'success': True,
                    'logs': response.get('logs', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get logs'
                }
                
        except Exception as e:
            logger.error(f"Failed to get logs: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def install_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """Install a plugin for AutoGPT"""
        try:
            response = await self._make_request(
                'POST',
                '/api/v1/plugins/install',
                json={'plugin_name': plugin_name}
            )
            
            if response:
                return {
                    'success': True,
                    'installed': True,
                    'plugin': plugin_name
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to install plugin'
                }
                
        except Exception as e:
            logger.error(f"Failed to install plugin: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _get_available_plugins(self) -> List[str]:
        """Get list of available plugins"""
        try:
            response = await self._make_request('GET', '/api/v1/plugins')
            if response:
                return response.get('plugins', [])
            return []
        except:
            return []
            
    async def execute_command(self,
                            agent_id: str,
                            command: str,
                            arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific command through an agent"""
        try:
            payload = {
                'command': command,
                'arguments': arguments
            }
            
            response = await self._make_request(
                'POST',
                f'/api/v1/agents/{agent_id}/execute',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'result': response.get('result', ''),
                    'output': response.get('output', '')
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to execute command'
                }
                
        except Exception as e:
            logger.error(f"Failed to execute command: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_workspace_files(self, agent_id: str) -> Dict[str, Any]:
        """Get files in agent workspace"""
        try:
            response = await self._make_request(
                'GET',
                f'/api/v1/agents/{agent_id}/workspace'
            )
            
            if response:
                return {
                    'success': True,
                    'files': response.get('files', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get workspace files'
                }
                
        except Exception as e:
            logger.error(f"Failed to get workspace files: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }