"""
Dify adapter for AI workflow orchestration
"""
from typing import Dict, Any, List, Optional
from ..base_adapter import ServiceAdapter
import logging

logger = logging.getLogger(__name__)


class DifyAdapter(ServiceAdapter):
    """Adapter for Dify AI workflow platform"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Dify", config)
        self.web_url = config.get('web_url', 'http://dify-web:3000')
        self.apps = {}
        self.workflows = {}
        
    async def initialize(self):
        """Initialize Dify connection"""
        try:
            await self.connect()
            
            # Verify API is accessible
            response = await self._make_request('GET', '/v1/apps')
            if response is not None:
                logger.info("Dify initialized successfully")
            else:
                raise Exception("Failed to connect to Dify API")
                
        except Exception as e:
            logger.error(f"Failed to initialize Dify: {str(e)}")
            raise
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get Dify capabilities"""
        return {
            'service': 'Dify',
            'type': 'workflow_tool',
            'features': [
                'ai_app_builder',
                'workflow_orchestration',
                'prompt_engineering',
                'dataset_management',
                'model_integration',
                'api_generation',
                'team_collaboration'
            ],
            'app_types': [
                'chatbot',
                'completion',
                'workflow'
            ],
            'model_providers': [
                'openai',
                'anthropic',
                'cohere',
                'huggingface',
                'local_models'
            ],
            'active_apps': len(self.apps)
        }
        
    async def create_app(self,
                        name: str,
                        mode: str,  # chat, completion, workflow
                        model_config: Dict[str, Any],
                        prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """Create a new Dify app"""
        try:
            app_data = {
                'name': name,
                'mode': mode,
                'model_config': model_config,
                'prompt_template': prompt_template or "",
                'opening_statement': f"Welcome to {name}",
                'suggested_questions': []
            }
            
            response = await self._make_request(
                'POST',
                '/v1/apps',
                json=app_data
            )
            
            if response and 'id' in response:
                app_id = response['id']
                self.apps[app_id] = {
                    'name': name,
                    'mode': mode,
                    'api_key': response.get('api_key', '')
                }
                
                return {
                    'success': True,
                    'app_id': app_id,
                    'api_key': response.get('api_key', ''),
                    'web_url': f"{self.web_url}/app/{app_id}"
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create app'
                }
                
        except Exception as e:
            logger.error(f"Failed to create app: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def chat_completion(self,
                            app_id: str,
                            query: str,
                            conversation_id: Optional[str] = None,
                            user: str = "default") -> Dict[str, Any]:
        """Send chat message to app"""
        try:
            payload = {
                'query': query,
                'inputs': {},
                'response_mode': 'blocking',
                'user': user
            }
            
            if conversation_id:
                payload['conversation_id'] = conversation_id
                
            response = await self._make_request(
                'POST',
                f'/v1/chat-messages',
                json=payload,
                headers={'Authorization': f'Bearer {self.apps[app_id]["api_key"]}'}
            )
            
            if response:
                return {
                    'success': True,
                    'answer': response.get('answer', ''),
                    'conversation_id': response.get('conversation_id', ''),
                    'message_id': response.get('id', ''),
                    'tokens_used': response.get('metadata', {}).get('usage', {})
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get response'
                }
                
        except Exception as e:
            logger.error(f"Failed to send chat message: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def run_workflow(self,
                          workflow_id: str,
                          inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow"""
        try:
            payload = {
                'inputs': inputs,
                'response_mode': 'blocking'
            }
            
            response = await self._make_request(
                'POST',
                f'/v1/workflows/{workflow_id}/run',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'outputs': response.get('outputs', {}),
                    'run_id': response.get('run_id', ''),
                    'status': response.get('status', ''),
                    'elapsed_time': response.get('elapsed_time', 0)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to run workflow'
                }
                
        except Exception as e:
            logger.error(f"Failed to run workflow: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def create_dataset(self,
                           name: str,
                           description: str,
                           indexing_technique: str = "high_quality") -> Dict[str, Any]:
        """Create a knowledge dataset"""
        try:
            dataset_data = {
                'name': name,
                'description': description,
                'indexing_technique': indexing_technique,
                'permission': 'all_team_members'
            }
            
            response = await self._make_request(
                'POST',
                '/v1/datasets',
                json=dataset_data
            )
            
            if response and 'id' in response:
                return {
                    'success': True,
                    'dataset_id': response['id'],
                    'document_count': 0
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create dataset'
                }
                
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def add_document(self,
                          dataset_id: str,
                          document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add document to dataset"""
        try:
            response = await self._make_request(
                'POST',
                f'/v1/datasets/{dataset_id}/documents',
                json=document_data
            )
            
            if response:
                return {
                    'success': True,
                    'document_id': response.get('id', ''),
                    'indexing_status': response.get('indexing_status', '')
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to add document'
                }
                
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def list_apps(self) -> Dict[str, Any]:
        """List all apps"""
        try:
            response = await self._make_request('GET', '/v1/apps')
            
            if response:
                return {
                    'success': True,
                    'apps': response.get('data', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to list apps'
                }
                
        except Exception as e:
            logger.error(f"Failed to list apps: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_app_statistics(self, app_id: str) -> Dict[str, Any]:
        """Get app usage statistics"""
        try:
            response = await self._make_request(
                'GET',
                f'/v1/apps/{app_id}/statistics'
            )
            
            if response:
                return {
                    'success': True,
                    'statistics': response
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get statistics'
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def export_app(self, app_id: str) -> Dict[str, Any]:
        """Export app configuration"""
        try:
            response = await self._make_request(
                'GET',
                f'/v1/apps/{app_id}/export'
            )
            
            if response:
                return {
                    'success': True,
                    'app_config': response
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to export app'
                }
                
        except Exception as e:
            logger.error(f"Failed to export app: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def update_app_config(self,
                               app_id: str,
                               config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update app configuration"""
        try:
            response = await self._make_request(
                'PUT',
                f'/v1/apps/{app_id}/model-config',
                json=config_updates
            )
            
            if response:
                return {
                    'success': True,
                    'updated': True
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to update config'
                }
                
        except Exception as e:
            logger.error(f"Failed to update config: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }