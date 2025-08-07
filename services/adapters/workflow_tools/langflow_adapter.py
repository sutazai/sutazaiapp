"""
LangFlow adapter for visual workflow creation and execution
"""
from typing import Dict, Any, List, Optional
import uuid
from ..base_adapter import ServiceAdapter
import logging

logger = logging.getLogger(__name__)


class LangFlowAdapter(ServiceAdapter):
    """Adapter for LangFlow workflow automation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("LangFlow", config)
        self.flows_path = config.get('flows_path', '/data/langflow/flows')
        self.database_url = config.get('database_url')
        self.active_flows = {}
        
    async def initialize(self):
        """Initialize LangFlow connection"""
        try:
            await self.connect()
            
            # Verify API is accessible
            response = await self._make_request('GET', '/api/v1/flows')
            if response is not None:
                logger.info("LangFlow initialized successfully")
            else:
                raise Exception("Failed to connect to LangFlow API")
                
        except Exception as e:
            logger.error(f"Failed to initialize LangFlow: {str(e)}")
            raise
            
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get LangFlow capabilities"""
        return {
            'service': 'LangFlow',
            'type': 'workflow_tool',
            'features': [
                'visual_flow_builder',
                'drag_drop_interface',
                'component_library',
                'flow_execution',
                'api_integration',
                'export_import',
                'version_control'
            ],
            'component_types': [
                'llm_chains',
                'prompts',
                'document_loaders',
                'vector_stores',
                'agents',
                'tools',
                'memory'
            ],
            'active_flows': len(self.active_flows)
        }
        
    async def create_flow(self,
                         name: str,
                         description: str,
                         components: List[Dict[str, Any]],
                         connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a new flow"""
        try:
            flow_id = str(uuid.uuid4())
            
            flow_data = {
                'id': flow_id,
                'name': name,
                'description': description,
                'data': {
                    'nodes': components,
                    'edges': connections
                }
            }
            
            response = await self._make_request(
                'POST',
                '/api/v1/flows',
                json=flow_data
            )
            
            if response:
                self.active_flows[flow_id] = {
                    'name': name,
                    'status': 'created'
                }
                
                return {
                    'success': True,
                    'flow_id': flow_id,
                    'api_endpoint': f"/api/v1/flows/{flow_id}/run"
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create flow'
                }
                
        except Exception as e:
            logger.error(f"Failed to create flow: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def execute_flow(self,
                          flow_id: str,
                          inputs: Dict[str, Any],
                          timeout: int = 300) -> Dict[str, Any]:
        """Execute a flow with given inputs"""
        try:
            payload = {
                'inputs': inputs,
                'timeout': timeout
            }
            
            response = await self._make_request(
                'POST',
                f'/api/v1/flows/{flow_id}/run',
                json=payload
            )
            
            if response:
                return {
                    'success': True,
                    'outputs': response.get('outputs', {}),
                    'execution_time': response.get('execution_time', 0),
                    'logs': response.get('logs', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to execute flow'
                }
                
        except Exception as e:
            logger.error(f"Failed to execute flow: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_flow(self, flow_id: str) -> Dict[str, Any]:
        """Get flow details"""
        try:
            response = await self._make_request(
                'GET',
                f'/api/v1/flows/{flow_id}'
            )
            
            if response:
                return {
                    'success': True,
                    'flow': response
                }
            else:
                return {
                    'success': False,
                    'error': 'Flow not found'
                }
                
        except Exception as e:
            logger.error(f"Failed to get flow: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def update_flow(self,
                         flow_id: str,
                         components: Optional[List[Dict]] = None,
                         connections: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Update an existing flow"""
        try:
            update_data = {}
            if components is not None:
                update_data['nodes'] = components
            if connections is not None:
                update_data['edges'] = connections
                
            response = await self._make_request(
                'PUT',
                f'/api/v1/flows/{flow_id}',
                json={'data': update_data}
            )
            
            if response:
                return {
                    'success': True,
                    'updated': True
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to update flow'
                }
                
        except Exception as e:
            logger.error(f"Failed to update flow: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def list_flows(self) -> Dict[str, Any]:
        """List all flows"""
        try:
            response = await self._make_request('GET', '/api/v1/flows')
            
            if response:
                return {
                    'success': True,
                    'flows': response.get('flows', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to list flows'
                }
                
        except Exception as e:
            logger.error(f"Failed to list flows: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def delete_flow(self, flow_id: str) -> Dict[str, Any]:
        """Delete a flow"""
        try:
            response = await self._make_request(
                'DELETE',
                f'/api/v1/flows/{flow_id}'
            )
            
            if response:
                if flow_id in self.active_flows:
                    del self.active_flows[flow_id]
                    
                return {
                    'success': True,
                    'deleted': True
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to delete flow'
                }
                
        except Exception as e:
            logger.error(f"Failed to delete flow: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def export_flow(self, flow_id: str) -> Dict[str, Any]:
        """Export flow as JSON"""
        try:
            response = await self._make_request(
                'GET',
                f'/api/v1/flows/{flow_id}/export'
            )
            
            if response:
                return {
                    'success': True,
                    'flow_data': response
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to export flow'
                }
                
        except Exception as e:
            logger.error(f"Failed to export flow: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def import_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Import flow from JSON"""
        try:
            response = await self._make_request(
                'POST',
                '/api/v1/flows/import',
                json=flow_data
            )
            
            if response:
                flow_id = response.get('flow_id')
                if flow_id:
                    self.active_flows[flow_id] = {
                        'name': flow_data.get('name', 'Imported Flow'),
                        'status': 'imported'
                    }
                    
                return {
                    'success': True,
                    'flow_id': flow_id
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to import flow'
                }
                
        except Exception as e:
            logger.error(f"Failed to import flow: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_components(self) -> Dict[str, Any]:
        """Get available components"""
        try:
            response = await self._make_request(
                'GET',
                '/api/v1/components'
            )
            
            if response:
                return {
                    'success': True,
                    'components': response.get('components', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to get components'
                }
                
        except Exception as e:
            logger.error(f"Failed to get components: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }