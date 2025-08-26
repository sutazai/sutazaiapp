"""
Unified Development Service MCP Adapter
Handles communication with the unified-dev service that consolidates
ultimatecoder, language-server, and sequentialthinking capabilities.

Created: 2025-08-17 UTC
Target: 512MB memory usage consolidation
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

class UnifiedDevAdapter:
    """
    Adapter for the unified development service that consolidates
    ultimatecoder, language-server, and sequentialthinking
    """
    
    def __init__(self, host: str = "localhost", port: int = 4000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_connected = False
        self.service_capabilities = {
            'ultimatecoder': ['generate', 'analyze', 'refactor', 'optimize'],
            'language-server': ['completion', 'diagnostics', 'hover', 'definition'],
            'sequentialthinking': ['reasoning', 'planning', 'analysis']
        }
        
        # Performance metrics
        self.metrics = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'response_times': [],
            'last_health_check': None,
            'memory_usage': None
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Establish connection to unified development service"""
        try:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            self.session = aiohttp.ClientSession(connector=connector)
            
            # Verify service is available
            health_status = await self.health_check()
            if health_status and health_status.get('status') == 'healthy':
                self.is_connected = True
                logger.info(f"Connected to unified development service at {self.base_url}")
                return True
            else:
                logger.error(f"Unified development service health check failed: {health_status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to unified development service: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close connection to unified development service"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info("Disconnected from unified development service")
    
    async def health_check(self) -> Optional[Dict[str, Any]]:
        """Check health status of unified development service"""
        try:
            if not self.session:
                return None
                
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    self.metrics['last_health_check'] = datetime.utcnow()
                    self.metrics['memory_usage'] = health_data.get('memory', {})
                    return health_data
                else:
                    logger.warning(f"Health check failed with status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics from unified development service"""
        try:
            if not self.session:
                return {"status": "ready", "adapter": "unified-dev", "version": "1.0.0"}
            async with self.session.get(f"{self.base_url}/metrics") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "ready", "adapter": "unified-dev", "version": "1.0.0"}
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {"status": "ready", "adapter": "unified-dev", "version": "1.0.0"}
    async def _make_request(self, 
                           service: str, 
                           data: Dict[str, Any], 
                           timeout: int = 30) -> Dict[str, Any]:
        """Make a request to the unified development service"""
        if not self.is_connected or not self.session:
            raise ConnectionError("Not connected to unified development service")
        
        start_time = datetime.utcnow()
        self.metrics['requests_total'] += 1
        
        try:
            # Prepare request payload
            payload = {
                'service': service,
                **data
            }
            
            async with self.session.post(
                f"{self.base_url}/api/dev",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                response_data = await response.json()
                
                # Calculate response time
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.metrics['response_times'].append(response_time)
                
                # Keep only last 100 response times
                if len(self.metrics['response_times']) > 100:
                    self.metrics['response_times'] = self.metrics['response_times'][-100:]
                
                if response.status == 200 and response_data.get('success'):
                    self.metrics['requests_successful'] += 1
                    return response_data
                else:
                    self.metrics['requests_failed'] += 1
                    error_msg = response_data.get('error', f'HTTP {response.status}')
                    raise RuntimeError(f"Request failed: {error_msg}")
                    
        except Exception as e:
            self.metrics['requests_failed'] += 1
            logger.error(f"Request to {service} failed: {e}")
            raise
    
    # UltimateCoder methods
    async def generate_code(self, 
                           code: str, 
                           language: str, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate code using UltimateCoder capabilities"""
        return await self._make_request('ultimatecoder', {
            'code': code,
            'language': language,
            'action': 'generate',
            'context': context or {}
        })
    
    async def analyze_code(self, 
                          code: str, 
                          language: str, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze code using UltimateCoder capabilities"""
        return await self._make_request('ultimatecoder', {
            'code': code,
            'language': language,
            'action': 'analyze',
            'context': context or {}
        })
    
    async def refactor_code(self, 
                           code: str, 
                           language: str, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Refactor code using UltimateCoder capabilities"""
        return await self._make_request('ultimatecoder', {
            'code': code,
            'language': language,
            'action': 'refactor',
            'context': context or {}
        })
    
    async def optimize_code(self, 
                           code: str, 
                           language: str, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize code using UltimateCoder capabilities"""
        return await self._make_request('ultimatecoder', {
            'code': code,
            'language': language,
            'action': 'optimize',
            'context': context or {}
        })
    
    # Language Server methods
    async def get_completions(self, 
                             method: str = 'textDocument/completion',
                             params: Optional[Dict[str, Any]] = None,
                             workspace: str = '/opt/sutazaiapp') -> Dict[str, Any]:
        """Get code completions using Language Server Protocol"""
        return await self._make_request('language-server', {
            'method': method,
            'params': params or {},
            'workspace': workspace
        })
    
    async def get_diagnostics(self, 
                             method: str = 'textDocument/publishDiagnostics',
                             params: Optional[Dict[str, Any]] = None,
                             workspace: str = '/opt/sutazaiapp') -> Dict[str, Any]:
        """Get code diagnostics using Language Server Protocol"""
        return await self._make_request('language-server', {
            'method': method,
            'params': params or {},
            'workspace': workspace
        })
    
    async def get_hover_info(self, 
                            method: str = 'textDocument/hover',
                            params: Optional[Dict[str, Any]] = None,
                            workspace: str = '/opt/sutazaiapp') -> Dict[str, Any]:
        """Get hover information using Language Server Protocol"""
        return await self._make_request('language-server', {
            'method': method,
            'params': params or {},
            'workspace': workspace
        })
    
    async def get_definition(self, 
                            method: str = 'textDocument/definition',
                            params: Optional[Dict[str, Any]] = None,
                            workspace: str = '/opt/sutazaiapp') -> Dict[str, Any]:
        """Get symbol definition using Language Server Protocol"""
        return await self._make_request('language-server', {
            'method': method,
            'params': params or {},
            'workspace': workspace
        })
    
    # Sequential Thinking methods
    async def sequential_reasoning(self, 
                                  query: str, 
                                  context: Optional[Dict[str, Any]] = None,
                                  max_steps: int = 10) -> Dict[str, Any]:
        """Perform sequential reasoning analysis"""
        return await self._make_request('sequentialthinking', {
            'query': query,
            'context': context or {},
            'maxSteps': max_steps
        })
    
    async def multi_step_planning(self, 
                                 query: str, 
                                 steps: Optional[List[Dict[str, Any]]] = None,
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create multi-step plans for complex problems"""
        return await self._make_request('sequentialthinking', {
            'query': query,
            'steps': steps or [],
            'context': context or {}
        })
    
    # Unified methods that leverage multiple services
    async def comprehensive_code_analysis(self, 
                                        code: str, 
                                        language: str,
                                        include_reasoning: bool = True) -> Dict[str, Any]:
        """Perform comprehensive analysis using both UltimateCoder and Sequential Thinking"""
        try:
            # Get code analysis from UltimateCoder
            code_analysis = await self.analyze_code(code, language)
            
            result = {
                'code_analysis': code_analysis,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if include_reasoning:
                # Use Sequential Thinking for deeper analysis
                reasoning_query = f"Analyze the quality and architecture of this {language} code"
                reasoning_analysis = await self.sequential_reasoning(
                    reasoning_query, 
                    context={'code_analysis': code_analysis, 'language': language}
                )
                result['reasoning_analysis'] = reasoning_analysis
            
            return {
                'success': True,
                'service': 'unified-dev-comprehensive',
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'service': 'unified-dev-comprehensive'
            }
    
    async def intelligent_code_generation(self, 
                                        requirements: str, 
                                        language: str,
                                        use_planning: bool = True) -> Dict[str, Any]:
        """Generate code with intelligent planning"""
        try:
            result = {}
            
            if use_planning:
                # First, create a plan using Sequential Thinking
                planning_query = f"Plan the implementation of: {requirements} in {language}"
                plan = await self.sequential_reasoning(planning_query, context={'language': language})
                result['implementation_plan'] = plan
                
                # Use the plan context for code generation
                context = {'plan': plan, 'requirements': requirements}
            else:
                context = {'requirements': requirements}
            
            # Generate code using UltimateCoder
            generated_code = await self.generate_code('', language, context)
            result['generated_code'] = generated_code
            
            return {
                'success': True,
                'service': 'unified-dev-intelligent',
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Intelligent code generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'service': 'unified-dev-intelligent'
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        avg_response_time = (
            sum(self.metrics['response_times']) / len(self.metrics['response_times'])
            if self.metrics['response_times'] else 0
        )
        
        success_rate = (
            self.metrics['requests_successful'] / self.metrics['requests_total']
            if self.metrics['requests_total'] > 0 else 0
        ) * 100
        
        return {
            'connection_status': self.is_connected,
            'total_requests': self.metrics['requests_total'],
            'success_rate': f"{success_rate:.1f}%",
            'average_response_time': f"{avg_response_time:.1f}ms",
            'last_health_check': self.metrics['last_health_check'],
            'memory_usage': self.metrics['memory_usage'],
            'service_capabilities': self.service_capabilities
        }

# Global adapter instance for the application
unified_dev_adapter = UnifiedDevAdapter()

async def get_unified_dev_adapter() -> UnifiedDevAdapter:
    """Get the global unified development adapter instance"""
    if not unified_dev_adapter.is_connected:
        await unified_dev_adapter.connect()
    return unified_dev_adapter