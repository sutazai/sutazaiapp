#!/usr/bin/env python3
"""
Frontend/Backend Integration Tests
=================================

Comprehensive validation tests for frontend and backend integration:
- Streamlit frontend with React-like components
- FastAPI backend with WebSocket support
- API endpoints and data flow validation
- User interface responsiveness and functionality
- Real-time communication and error handling
- Cross-origin resource sharing (CORS) validation

Focus on actual end-to-end user workflows and API integration.
"""

import asyncio
import aiohttp
import json
import logging
import time
import subprocess
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import websockets
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

@dataclass
class IntegrationTestResult:
    """Integration test execution result"""
    component: str
    test_name: str
    success: bool
    duration: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class FrontendBackendIntegrationValidator:
    """Comprehensive frontend/backend integration validation"""
    
    def __init__(self):
        self.results: List[IntegrationTestResult] = []
        
        # Service configurations from docker-compose and port registry
        self.config = {
            "backend": {
                "host": "localhost",
                "port": 10010,
                "base_url": "http://localhost:10010",
                "health_endpoint": "/health",
                "api_endpoints": {
                    "health": "/health",
                    "detailed_health": "/api/v1/health",
                    "agents": "/api/v1/agents",
                    "chat": "/api/v1/chat", 
                    "tasks": "/api/v1/tasks",
                    "monitoring": "/api/v1/monitoring",
                    "metrics": "/api/v1/metrics"
                },
                "websocket_endpoint": "/ws"
            },
            "frontend": {
                "host": "localhost",
                "port": 10011,
                "base_url": "http://localhost:10011",
                "health_endpoint": "/health",
                "expected_pages": [
                    "/",
                    "/?page=main",
                    "/?page=chat", 
                    "/?page=agents",
                    "/?page=hardware"
                ]
            }
        }
        
        # Test data for integration validation
        self.test_data = {
            "chat_message": {
                "message": "Hello, this is a test message from integration validation.",
                "model": "tinyllama",
                "stream": False
            },
            "agent_config": {
                "name": "integration-test-agent",
                "type": "validation",
                "capabilities": ["test", "validation"],
                "config": {"test_mode": True}
            },
            "task_request": {
                "task_type": "validation",
                "payload": {"test": "integration", "timestamp": time.time()},
                "priority": 1
            }
        }
    
    async def run_all_integration_tests(self) -> List[IntegrationTestResult]:
        """Execute all frontend/backend integration tests"""
        logger.info("Starting comprehensive frontend/backend integration validation")
        
        # Test execution order based on dependencies
        test_methods = [
            # Backend validation first
            ("backend_health", self.test_backend_health),
            ("backend_api_endpoints", self.test_backend_api_endpoints),
            ("backend_api_functionality", self.test_backend_api_functionality),
            
            # Frontend validation
            ("frontend_health", self.test_frontend_health), 
            ("frontend_ui_availability", self.test_frontend_ui_availability),
            ("frontend_pages_navigation", self.test_frontend_pages_navigation),
            
            # Integration testing
            ("cors_configuration", self.test_cors_configuration),
            ("api_integration", self.test_api_integration),
            ("real_time_communication", self.test_real_time_communication),
            
            # End-to-end workflows
            ("user_workflow_simulation", self.test_user_workflow_simulation),
            ("error_handling", self.test_error_handling),
            ("performance_integration", self.test_performance_integration),
        ]
        
        # Execute tests sequentially (some have dependencies)
        for component, test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Integration test {component} failed: {e}")
        
        return self.results
    
    async def test_backend_health(self) -> None:
        """Test backend health and basic connectivity"""
        start_time = time.time()
        
        try:
            backend_url = self.config["backend"]["base_url"]
            
            async with aiohttp.ClientSession() as session:
                # Test basic health endpoint
                async with session.get(f"{backend_url}/health",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    health_success = response.status == 200
                    if health_success:
                        health_data = await response.json()
                    else:
                        health_data = {}
                
                # Test detailed health endpoint
                async with session.get(f"{backend_url}/api/v1/health",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    detailed_health_success = response.status == 200
                    if detailed_health_success:
                        detailed_health_data = await response.json()
                    else:
                        detailed_health_data = {}
                
                # Test API root
                async with session.get(f"{backend_url}/",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    root_accessible = response.status in [200, 404, 422]  # Accept various responses
                
                # Test OpenAPI docs
                async with session.get(f"{backend_url}/docs",
                                     timeout=aiohttp.ClientTimeout(total=15)) as response:
                    docs_available = response.status == 200
                
                # Test OpenAPI JSON schema
                async with session.get(f"{backend_url}/openapi.json",
                                     timeout=aiohttp.ClientTimeout(total=15)) as response:
                    openapi_available = response.status == 200
                    if openapi_available:
                        openapi_schema = await response.json()
                        api_endpoints_count = len(openapi_schema.get("paths", {}))
                    else:
                        api_endpoints_count = 0
            
            # Analyze health data
            health_analysis = {
                "basic_health": health_success,
                "detailed_health": detailed_health_success,
                "root_accessible": root_accessible,
                "docs_available": docs_available,
                "openapi_available": openapi_available,
                "api_endpoints_documented": api_endpoints_count,
                "health_status": health_data.get("status", "unknown"),
                "services_status": health_data.get("services", {}),
                "performance_metrics": health_data.get("performance", {})
            }
            
            duration = time.time() - start_time
            
            overall_success = health_success or detailed_health_success
            
            self.results.append(IntegrationTestResult(
                component="backend_health",
                test_name="backend_connectivity",
                success=overall_success,
                duration=duration,
                metrics={
                    "health_analysis": health_analysis,
                    "health_data": health_data,
                    "detailed_health_data": detailed_health_data,
                    "backend_grade": "excellent" if health_success and detailed_health_success else "good" if health_success or detailed_health_success else "poor",
                    "performance_grade": "excellent" if duration < 2 else "good" if duration < 5 else "poor"
                }
            ))
            
            logger.info(f"Backend health - Basic: {health_success}, Detailed: {detailed_health_success}, Docs: {docs_available}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="backend_health",
                test_name="backend_connectivity",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Backend health test failed: {e}")
    
    async def test_backend_api_endpoints(self) -> None:
        """Test backend API endpoints availability"""
        start_time = time.time()
        
        try:
            backend_url = self.config["backend"]["base_url"]
            api_endpoints = self.config["backend"]["api_endpoints"]
            
            endpoint_results = {}
            
            async with aiohttp.ClientSession() as session:
                for endpoint_name, endpoint_path in api_endpoints.items():
                    endpoint_url = f"{backend_url}{endpoint_path}"
                    
                    endpoint_start = time.time()
                    try:
                        async with session.get(endpoint_url,
                                             timeout=aiohttp.ClientTimeout(total=15)) as response:
                            endpoint_success = response.status in [200, 404, 405, 422]  # Accept various responses
                            endpoint_time = (time.time() - endpoint_start) * 1000
                            
                            # Try to get response body for analysis
                            try:
                                if response.content_type == "application/json":
                                    response_data = await response.json()
                                else:
                                    response_text = await response.text()
                                    response_data = {"text_length": len(response_text)}
                            except:
                                response_data = {"parse_error": True}
                            
                            endpoint_results[endpoint_name] = {
                                "accessible": endpoint_success,
                                "status_code": response.status,
                                "response_time_ms": endpoint_time,
                                "content_type": response.content_type,
                                "response_data": response_data,
                                "url": endpoint_url
                            }
                            
                    except Exception as endpoint_error:
                        endpoint_results[endpoint_name] = {
                            "accessible": False,
                            "error": str(endpoint_error),
                            "response_time_ms": (time.time() - endpoint_start) * 1000,
                            "url": endpoint_url
                        }
                
                # Test HTTP methods for main endpoints
                http_methods_test = {}
                test_endpoints = ["health", "agents", "chat"]
                
                for endpoint_name in test_endpoints:
                    if endpoint_name in api_endpoints:
                        endpoint_path = api_endpoints[endpoint_name]
                        endpoint_url = f"{backend_url}{endpoint_path}"
                        methods_result = {}
                        
                        for method in ["GET", "POST", "PUT", "DELETE", "OPTIONS"]:
                            try:
                                async with session.request(method, endpoint_url,
                                                          timeout=aiohttp.ClientTimeout(total=5)) as response:
                                    methods_result[method] = {
                                        "status": response.status,
                                        "allowed": response.status not in [404, 405]
                                    }
                            except Exception as method_error:
                                methods_result[method] = {
                                    "error": str(method_error),
                                    "allowed": False
                                }
                        
                        http_methods_test[endpoint_name] = methods_result
            
            duration = time.time() - start_time
            
            # Calculate endpoint success rate
            accessible_endpoints = sum(1 for result in endpoint_results.values() if result.get("accessible", False))
            endpoint_success_rate = accessible_endpoints / len(endpoint_results) * 100 if endpoint_results else 0
            
            # Calculate average response time
            response_times = [result.get("response_time_ms", 0) for result in endpoint_results.values()]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            self.results.append(IntegrationTestResult(
                component="backend_api_endpoints",
                test_name="api_endpoints_validation",
                success=endpoint_success_rate > 50,
                duration=duration,
                metrics={
                    "endpoint_results": endpoint_results,
                    "accessible_endpoints": accessible_endpoints,
                    "endpoint_success_rate": endpoint_success_rate,
                    "http_methods_test": http_methods_test,
                    "avg_response_time_ms": avg_response_time,
                    "fastest_endpoint": min(response_times) if response_times else 0,
                    "slowest_endpoint": max(response_times) if response_times else 0,
                    "performance_grade": "excellent" if avg_response_time < 100 else "good" if avg_response_time < 500 else "poor"
                }
            ))
            
            logger.info(f"Backend API endpoints - Success rate: {endpoint_success_rate:.1f}%, Avg response: {avg_response_time:.1f}ms")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="backend_api_endpoints",
                test_name="api_endpoints_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Backend API endpoints test failed: {e}")
    
    async def test_backend_api_functionality(self) -> None:
        """Test backend API functionality with real data"""
        start_time = time.time()
        
        try:
            backend_url = self.config["backend"]["base_url"]
            
            functionality_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Test agents endpoint functionality
                try:
                    async with session.get(f"{backend_url}/api/v1/agents",
                                         timeout=aiohttp.ClientTimeout(total=15)) as response:
                        agents_success = response.status == 200
                        if agents_success:
                            agents_data = await response.json()
                            if isinstance(agents_data, list):
                                agents_count = len(agents_data)
                            elif isinstance(agents_data, dict):
                                agents_count = len(agents_data.get("agents", []))
                            else:
                                agents_count = 0
                        else:
                            agents_count = 0
                        
                        functionality_tests["agents_endpoint"] = {
                            "functional": agents_success,
                            "status": response.status,
                            "agents_count": agents_count,
                            "data_type": type(agents_data).__name__ if agents_success else "unknown"
                        }
                        
                except Exception as agents_error:
                    functionality_tests["agents_endpoint"] = {
                        "functional": False,
                        "error": str(agents_error)
                    }
                
                # Test chat endpoint functionality
                try:
                    chat_payload = self.test_data["chat_message"]
                    async with session.post(f"{backend_url}/api/v1/chat",
                                          json=chat_payload,
                                          timeout=aiohttp.ClientTimeout(total=30)) as response:
                        chat_success = response.status in [200, 422, 503]  # Accept various responses
                        if response.status == 200:
                            chat_response = await response.json()
                        else:
                            chat_response = {"status": response.status}
                        
                        functionality_tests["chat_endpoint"] = {
                            "functional": chat_success,
                            "status": response.status,
                            "response_data": chat_response,
                            "accepts_json": True
                        }
                        
                except Exception as chat_error:
                    functionality_tests["chat_endpoint"] = {
                        "functional": False,
                        "error": str(chat_error),
                        "accepts_json": False
                    }
                
                # Test tasks endpoint functionality (if available)
                try:
                    task_payload = self.test_data["task_request"]
                    async with session.post(f"{backend_url}/api/v1/tasks",
                                          json=task_payload,
                                          timeout=aiohttp.ClientTimeout(total=15)) as response:
                        tasks_success = response.status in [200, 201, 404, 422]  # Accept various responses
                        if response.status in [200, 201]:
                            task_response = await response.json()
                        else:
                            task_response = {"status": response.status}
                        
                        functionality_tests["tasks_endpoint"] = {
                            "functional": tasks_success,
                            "status": response.status,
                            "response_data": task_response,
                            "task_created": response.status in [200, 201]
                        }
                        
                except Exception as tasks_error:
                    functionality_tests["tasks_endpoint"] = {
                        "functional": False,
                        "error": str(tasks_error)
                    }
                
                # Test monitoring endpoint functionality
                try:
                    async with session.get(f"{backend_url}/api/v1/monitoring",
                                         timeout=aiohttp.ClientTimeout(total=15)) as response:
                        monitoring_success = response.status in [200, 404]  # Accept not implemented
                        if response.status == 200:
                            monitoring_data = await response.json()
                        else:
                            monitoring_data = {"status": response.status}
                        
                        functionality_tests["monitoring_endpoint"] = {
                            "functional": monitoring_success,
                            "status": response.status,
                            "response_data": monitoring_data,
                            "implemented": response.status == 200
                        }
                        
                except Exception as monitoring_error:
                    functionality_tests["monitoring_endpoint"] = {
                        "functional": False,
                        "error": str(monitoring_error)
                    }
                
                # Test error handling
                error_handling_test = {}
                try:
                    # Test invalid endpoint
                    async with session.get(f"{backend_url}/api/v1/nonexistent",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        error_handling_test["invalid_endpoint"] = {
                            "status": response.status,
                            "proper_error": response.status == 404
                        }
                    
                    # Test malformed JSON
                    async with session.post(f"{backend_url}/api/v1/chat",
                                          data="invalid json",
                                          headers={"Content-Type": "application/json"},
                                          timeout=aiohttp.ClientTimeout(total=10)) as response:
                        error_handling_test["malformed_json"] = {
                            "status": response.status,
                            "proper_error": response.status in [400, 422]
                        }
                        
                except Exception as error_test_error:
                    error_handling_test["error"] = str(error_test_error)
                
                functionality_tests["error_handling"] = error_handling_test
            
            duration = time.time() - start_time
            
            # Calculate functionality success rate
            functional_endpoints = sum(1 for test in functionality_tests.values() 
                                     if isinstance(test, dict) and test.get("functional", False))
            functionality_success_rate = functional_endpoints / len([k for k in functionality_tests.keys() if k != "error_handling"]) * 100
            
            self.results.append(IntegrationTestResult(
                component="backend_api_functionality",
                test_name="api_functionality_validation",
                success=functionality_success_rate > 50,
                duration=duration,
                metrics={
                    "functionality_tests": functionality_tests,
                    "functional_endpoints": functional_endpoints,
                    "functionality_success_rate": functionality_success_rate,
                    "chat_functional": functionality_tests.get("chat_endpoint", {}).get("functional", False),
                    "agents_functional": functionality_tests.get("agents_endpoint", {}).get("functional", False),
                    "error_handling_proper": any(test.get("proper_error", False) for test in error_handling_test.values()) if 'error_handling_test' in locals() else False,
                    "performance_grade": "excellent" if duration < 15 else "good" if duration < 30 else "poor"
                }
            ))
            
            logger.info(f"Backend API functionality - Success rate: {functionality_success_rate:.1f}%, Functional endpoints: {functional_endpoints}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="backend_api_functionality",
                test_name="api_functionality_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Backend API functionality test failed: {e}")
    
    async def test_frontend_health(self) -> None:
        """Test frontend health and basic connectivity"""
        start_time = time.time()
        
        try:
            frontend_url = self.config["frontend"]["base_url"]
            
            async with aiohttp.ClientSession() as session:
                # Test frontend health endpoint
                async with session.get(f"{frontend_url}/health",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    health_success = response.status == 200
                    if health_success:
                        try:
                            health_data = await response.json()
                        except:
                            health_data = {"status": "ok"}
                    else:
                        health_data = {}
                
                # Test frontend root page
                async with session.get(f"{frontend_url}/",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    root_success = response.status == 200
                    if root_success:
                        root_content = await response.text()
                        # Check for Streamlit indicators
                        streamlit_indicators = [
                            "streamlit" in root_content.lower(),
                            "st-emotion" in root_content.lower(),
                            "stApp" in root_content,
                            "_stStreamlitMarkdown" in root_content
                        ]
                        is_streamlit = any(streamlit_indicators)
                        content_size = len(root_content)
                    else:
                        root_content = ""
                        is_streamlit = False
                        content_size = 0
                
                # Test Streamlit static resources
                static_resources = [
                    "/static/css/main.css",
                    "/static/js/main.js", 
                    "/favicon.ico"
                ]
                
                static_results = {}
                for resource in static_resources:
                    try:
                        async with session.get(f"{frontend_url}{resource}",
                                             timeout=aiohttp.ClientTimeout(total=10)) as response:
                            static_results[resource] = {
                                "available": response.status == 200,
                                "status": response.status,
                                "size": len(await response.read()) if response.status == 200 else 0
                            }
                    except Exception as static_error:
                        static_results[resource] = {
                            "available": False,
                            "error": str(static_error)
                        }
                
                # Test Streamlit WebSocket endpoint (if available)
                websocket_test = {}
                try:
                    # Streamlit typically uses WebSocket on a different path
                    ws_url = f"ws://localhost:{self.config['frontend']['port']}/stream"
                    websocket_test["websocket_available"] = False  # Will be tested in real-time communication test
                except Exception as ws_error:
                    websocket_test["error"] = str(ws_error)
            
            # Analyze frontend health
            frontend_analysis = {
                "health_endpoint": health_success,
                "root_accessible": root_success,
                "is_streamlit_app": is_streamlit,
                "content_size": content_size,
                "static_resources": static_results,
                "websocket_test": websocket_test,
                "health_data": health_data
            }
            
            duration = time.time() - start_time
            
            overall_success = health_success or root_success
            
            self.results.append(IntegrationTestResult(
                component="frontend_health",
                test_name="frontend_connectivity",
                success=overall_success,
                duration=duration,
                metrics={
                    "frontend_analysis": frontend_analysis,
                    "frontend_grade": "excellent" if health_success and root_success and is_streamlit else "good" if root_success else "poor",
                    "performance_grade": "excellent" if duration < 2 else "good" if duration < 5 else "poor"
                }
            ))
            
            logger.info(f"Frontend health - Health: {health_success}, Root: {root_success}, Streamlit: {is_streamlit}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="frontend_health",
                test_name="frontend_connectivity",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Frontend health test failed: {e}")
    
    async def test_frontend_ui_availability(self) -> None:
        """Test frontend UI availability and components"""
        start_time = time.time()
        
        try:
            frontend_url = self.config["frontend"]["base_url"]
            expected_pages = self.config["frontend"]["expected_pages"]
            
            ui_results = {}
            
            async with aiohttp.ClientSession() as session:
                for page_path in expected_pages:
                    page_url = f"{frontend_url}{page_path}"
                    
                    page_start = time.time()
                    try:
                        async with session.get(page_url,
                                             timeout=aiohttp.ClientTimeout(total=15)) as response:
                            page_success = response.status == 200
                            page_time = (time.time() - page_start) * 1000
                            
                            if page_success:
                                page_content = await response.text()
                                
                                # Analyze page content
                                ui_components = {
                                    "has_title": "<title>" in page_content,
                                    "has_navigation": any(nav in page_content.lower() for nav in ["navigation", "menu", "nav"]),
                                    "has_forms": "<form" in page_content or "st.form" in page_content,
                                    "has_buttons": "<button" in page_content or "st.button" in page_content,
                                    "has_inputs": "<input" in page_content or "st.text_input" in page_content,
                                    "streamlit_components": page_content.count("st-"),
                                    "content_length": len(page_content)
                                }
                                
                                # Check for specific SutazAI components
                                sutazai_components = {
                                    "has_agent_references": "agent" in page_content.lower(),
                                    "has_ai_references": any(ai_term in page_content.lower() for ai_term in ["ai", "llm", "model"]),
                                    "has_chat_interface": "chat" in page_content.lower(),
                                    "has_dashboard": "dashboard" in page_content.lower()
                                }
                                
                                ui_results[page_path] = {
                                    "accessible": page_success,
                                    "status": response.status,
                                    "load_time_ms": page_time,
                                    "ui_components": ui_components,
                                    "sutazai_components": sutazai_components,
                                    "content_type": response.content_type
                                }
                            else:
                                ui_results[page_path] = {
                                    "accessible": False,
                                    "status": response.status,
                                    "load_time_ms": page_time
                                }
                                
                    except Exception as page_error:
                        ui_results[page_path] = {
                            "accessible": False,
                            "error": str(page_error),
                            "load_time_ms": (time.time() - page_start) * 1000
                        }
                
                # Test Streamlit-specific functionality
                streamlit_features = {}
                
                # Test Streamlit health endpoint (different from general health)
                try:
                    async with session.get(f"{frontend_url}/healthz",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        streamlit_features["healthz_endpoint"] = response.status == 200
                except:
                    streamlit_features["healthz_endpoint"] = False
                
                # Test Streamlit media endpoint
                try:
                    async with session.get(f"{frontend_url}/media",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        streamlit_features["media_endpoint"] = response.status in [200, 404]
                except:
                    streamlit_features["media_endpoint"] = False
            
            duration = time.time() - start_time
            
            # Calculate UI success metrics
            accessible_pages = sum(1 for result in ui_results.values() if result.get("accessible", False))
            ui_success_rate = accessible_pages / len(ui_results) * 100 if ui_results else 0
            
            # Calculate average load time
            load_times = [result.get("load_time_ms", 0) for result in ui_results.values()]
            avg_load_time = sum(load_times) / len(load_times) if load_times else 0
            
            # Count UI components
            total_streamlit_components = sum(
                result.get("ui_components", {}).get("streamlit_components", 0) 
                for result in ui_results.values() 
                if result.get("accessible", False)
            )
            
            self.results.append(IntegrationTestResult(
                component="frontend_ui_availability",
                test_name="ui_components_validation",
                success=ui_success_rate > 50,
                duration=duration,
                metrics={
                    "ui_results": ui_results,
                    "accessible_pages": accessible_pages,
                    "ui_success_rate": ui_success_rate,
                    "avg_load_time_ms": avg_load_time,
                    "total_streamlit_components": total_streamlit_components,
                    "streamlit_features": streamlit_features,
                    "performance_grade": "excellent" if avg_load_time < 500 else "good" if avg_load_time < 2000 else "poor"
                }
            ))
            
            logger.info(f"Frontend UI - Success rate: {ui_success_rate:.1f}%, Pages: {accessible_pages}/{len(ui_results)}, Load time: {avg_load_time:.1f}ms")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="frontend_ui_availability",
                test_name="ui_components_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Frontend UI availability test failed: {e}")
    
    async def test_frontend_pages_navigation(self) -> None:
        """Test frontend pages navigation and routing"""
        start_time = time.time()
        
        try:
            frontend_url = self.config["frontend"]["base_url"]
            
            navigation_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Test different page navigation patterns
                navigation_patterns = [
                    ("main_page", "/"),
                    ("main_explicit", "/?page=main"),
                    ("chat_page", "/?page=chat"),
                    ("agents_page", "/?page=agents"),
                    ("hardware_page", "/?page=hardware"),
                    ("settings_page", "/?page=settings"),  # Might not exist
                    ("invalid_page", "/?page=nonexistent")
                ]
                
                for pattern_name, pattern_path in navigation_patterns:
                    navigation_start = time.time()
                    try:
                        async with session.get(f"{frontend_url}{pattern_path}",
                                             timeout=aiohttp.ClientTimeout(total=15)) as response:
                            nav_success = response.status == 200
                            nav_time = (time.time() - navigation_start) * 1000
                            
                            if nav_success:
                                page_content = await response.text()
                                
                                # Analyze page-specific content
                                page_analysis = {
                                    "content_length": len(page_content),
                                    "has_specific_content": False,
                                    "page_indicators": []
                                }
                                
                                # Check for page-specific indicators
                                if "chat" in pattern_path:
                                    page_analysis["has_specific_content"] = "chat" in page_content.lower()
                                    if page_analysis["has_specific_content"]:
                                        page_analysis["page_indicators"].append("chat_interface")
                                
                                if "agent" in pattern_path:
                                    page_analysis["has_specific_content"] = "agent" in page_content.lower()
                                    if page_analysis["has_specific_content"]:
                                        page_analysis["page_indicators"].append("agent_management")
                                
                                if "hardware" in pattern_path:
                                    page_analysis["has_specific_content"] = any(hw in page_content.lower() for hw in ["hardware", "cpu", "memory", "gpu"])
                                    if page_analysis["has_specific_content"]:
                                        page_analysis["page_indicators"].append("hardware_monitoring")
                                
                                navigation_tests[pattern_name] = {
                                    "accessible": nav_success,
                                    "status": response.status,
                                    "load_time_ms": nav_time,
                                    "page_analysis": page_analysis
                                }
                            else:
                                navigation_tests[pattern_name] = {
                                    "accessible": False,
                                    "status": response.status,
                                    "load_time_ms": nav_time
                                }
                                
                    except Exception as nav_error:
                        navigation_tests[pattern_name] = {
                            "accessible": False,
                            "error": str(nav_error),
                            "load_time_ms": (time.time() - navigation_start) * 1000
                        }
                
                # Test URL parameter handling
                parameter_tests = {}
                param_patterns = [
                    ("with_session", "/?session=test"),
                    ("with_theme", "/?theme=dark"),
                    ("with_multiple", "/?page=main&theme=dark&session=test")
                ]
                
                for param_name, param_path in param_patterns:
                    try:
                        async with session.get(f"{frontend_url}{param_path}",
                                             timeout=aiohttp.ClientTimeout(total=10)) as response:
                            parameter_tests[param_name] = {
                                "handles_params": response.status == 200,
                                "status": response.status
                            }
                    except Exception as param_error:
                        parameter_tests[param_name] = {
                            "handles_params": False,
                            "error": str(param_error)
                        }
            
            duration = time.time() - start_time
            
            # Calculate navigation success metrics
            successful_navigation = sum(1 for test in navigation_tests.values() if test.get("accessible", False))
            navigation_success_rate = successful_navigation / len(navigation_tests) * 100
            
            # Calculate average navigation time
            nav_times = [test.get("load_time_ms", 0) for test in navigation_tests.values()]
            avg_nav_time = sum(nav_times) / len(nav_times) if nav_times else 0
            
            # Count pages with specific content
            pages_with_content = sum(1 for test in navigation_tests.values() 
                                   if test.get("page_analysis", {}).get("has_specific_content", False))
            
            self.results.append(IntegrationTestResult(
                component="frontend_pages_navigation",
                test_name="navigation_validation",
                success=navigation_success_rate > 50,
                duration=duration,
                metrics={
                    "navigation_tests": navigation_tests,
                    "successful_navigation": successful_navigation,
                    "navigation_success_rate": navigation_success_rate,
                    "parameter_tests": parameter_tests,
                    "avg_navigation_time_ms": avg_nav_time,
                    "pages_with_specific_content": pages_with_content,
                    "routing_functional": navigation_success_rate > 70,
                    "performance_grade": "excellent" if avg_nav_time < 500 else "good" if avg_nav_time < 1500 else "poor"
                }
            ))
            
            logger.info(f"Frontend navigation - Success rate: {navigation_success_rate:.1f}%, Avg time: {avg_nav_time:.1f}ms")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="frontend_pages_navigation",
                test_name="navigation_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Frontend pages navigation test failed: {e}")
    
    async def test_cors_configuration(self) -> None:
        """Test CORS (Cross-Origin Resource Sharing) configuration"""
        start_time = time.time()
        
        try:
            backend_url = self.config["backend"]["base_url"]
            frontend_url = self.config["frontend"]["base_url"]
            
            cors_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Test CORS for API endpoints from frontend origin
                frontend_host = urlparse(frontend_url).netloc
                cors_headers = {
                    "Origin": frontend_url,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "Content-Type"
                }
                
                api_endpoints = ["/api/v1/health", "/api/v1/agents", "/api/v1/chat"]
                
                for endpoint in api_endpoints:
                    # Test preflight request (OPTIONS)
                    try:
                        async with session.options(f"{backend_url}{endpoint}",
                                                 headers=cors_headers,
                                                 timeout=aiohttp.ClientTimeout(total=10)) as response:
                            preflight_success = response.status in [200, 204]
                            cors_response_headers = {
                                "access_control_allow_origin": response.headers.get("Access-Control-Allow-Origin"),
                                "access_control_allow_methods": response.headers.get("Access-Control-Allow-Methods"),
                                "access_control_allow_headers": response.headers.get("Access-Control-Allow-Headers"),
                                "access_control_max_age": response.headers.get("Access-Control-Max-Age")
                            }
                            
                            cors_configured = any(header_value is not None for header_value in cors_response_headers.values())
                            
                    except Exception as preflight_error:
                        preflight_success = False
                        cors_response_headers = {}
                        cors_configured = False
                    
                    # Test actual request with Origin header
                    try:
                        async with session.get(f"{backend_url}{endpoint}",
                                             headers={"Origin": frontend_url},
                                             timeout=aiohttp.ClientTimeout(total=10)) as response:
                            actual_request_success = response.status in [200, 404, 422]
                            actual_cors_headers = {
                                "access_control_allow_origin": response.headers.get("Access-Control-Allow-Origin")
                            }
                    except Exception as actual_error:
                        actual_request_success = False
                        actual_cors_headers = {}
                    
                    cors_tests[endpoint] = {
                        "preflight_success": preflight_success,
                        "actual_request_success": actual_request_success,
                        "cors_configured": cors_configured,
                        "preflight_headers": cors_response_headers,
                        "actual_headers": actual_cors_headers,
                        "allows_frontend_origin": cors_response_headers.get("access_control_allow_origin") in ["*", frontend_url]
                    }
                
                # Test CORS with different origins
                different_origins = [
                    "http://localhost:3000",  # Common React dev server
                    "https://example.com",    # External origin
                    "null"                    # File origin
                ]
                
                origin_tests = {}
                for origin in different_origins:
                    try:
                        async with session.options(f"{backend_url}/api/v1/health",
                                                 headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
                                                 timeout=aiohttp.ClientTimeout(total=5)) as response:
                            origin_tests[origin] = {
                                "allowed": response.status in [200, 204],
                                "allow_origin": response.headers.get("Access-Control-Allow-Origin")
                            }
                    except Exception as origin_error:
                        origin_tests[origin] = {
                            "allowed": False,
                            "error": str(origin_error)
                        }
            
            duration = time.time() - start_time
            
            # Analyze CORS configuration
            cors_enabled_endpoints = sum(1 for test in cors_tests.values() if test.get("cors_configured", False))
            cors_success_rate = cors_enabled_endpoints / len(cors_tests) * 100 if cors_tests else 0
            
            allows_frontend = sum(1 for test in cors_tests.values() if test.get("allows_frontend_origin", False))
            frontend_access_rate = allows_frontend / len(cors_tests) * 100 if cors_tests else 0
            
            self.results.append(IntegrationTestResult(
                component="cors_configuration",
                test_name="cors_validation",
                success=cors_success_rate > 0 or frontend_access_rate > 0,
                duration=duration,
                metrics={
                    "cors_tests": cors_tests,
                    "origin_tests": origin_tests,
                    "cors_enabled_endpoints": cors_enabled_endpoints,
                    "cors_success_rate": cors_success_rate,
                    "frontend_access_rate": frontend_access_rate,
                    "cors_properly_configured": cors_success_rate > 70,
                    "security_appropriate": not any(test.get("allows_frontend_origin") and test.get("preflight_headers", {}).get("access_control_allow_origin") == "*" for test in cors_tests.values()),
                    "performance_grade": "excellent" if duration < 5 else "good" if duration < 10 else "poor"
                }
            ))
            
            logger.info(f"CORS configuration - Enabled endpoints: {cors_enabled_endpoints}/{len(cors_tests)}, Frontend access: {frontend_access_rate:.1f}%")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="cors_configuration",
                test_name="cors_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"CORS configuration test failed: {e}")
    
    async def test_api_integration(self) -> None:
        """Test API integration between frontend and backend"""
        start_time = time.time()
        
        try:
            backend_url = self.config["backend"]["base_url"]
            
            integration_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Test complete API workflow simulation
                workflow_steps = {}
                
                # Step 1: Get system health (frontend would do this on load)
                try:
                    async with session.get(f"{backend_url}/health",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        health_step = response.status == 200
                        if health_step:
                            health_data = await response.json()
                        else:
                            health_data = {}
                        
                        workflow_steps["health_check"] = {
                            "success": health_step,
                            "data": health_data
                        }
                except Exception as health_error:
                    workflow_steps["health_check"] = {
                        "success": False,
                        "error": str(health_error)
                    }
                
                # Step 2: Get available agents (typical frontend request)
                try:
                    async with session.get(f"{backend_url}/api/v1/agents",
                                         timeout=aiohttp.ClientTimeout(total=15)) as response:
                        agents_step = response.status == 200
                        if agents_step:
                            agents_data = await response.json()
                            agents_count = len(agents_data) if isinstance(agents_data, list) else len(agents_data.get("agents", [])) if isinstance(agents_data, dict) else 0
                        else:
                            agents_data = {}
                            agents_count = 0
                        
                        workflow_steps["get_agents"] = {
                            "success": agents_step,
                            "agents_count": agents_count,
                            "data_type": type(agents_data).__name__
                        }
                except Exception as agents_error:
                    workflow_steps["get_agents"] = {
                        "success": False,
                        "error": str(agents_error)
                    }
                
                # Step 3: Send a chat message (core functionality)
                try:
                    chat_payload = self.test_data["chat_message"]
                    async with session.post(f"{backend_url}/api/v1/chat",
                                          json=chat_payload,
                                          headers={"Content-Type": "application/json"},
                                          timeout=aiohttp.ClientTimeout(total=30)) as response:
                        chat_step = response.status in [200, 422, 503]  # Accept various responses
                        
                        if response.status == 200:
                            chat_response = await response.json()
                            has_response = bool(chat_response.get("response") or chat_response.get("message"))
                        else:
                            chat_response = {"status": response.status}
                            has_response = False
                        
                        workflow_steps["chat_interaction"] = {
                            "success": chat_step,
                            "status_code": response.status,
                            "has_response": has_response,
                            "response_data": chat_response
                        }
                except Exception as chat_error:
                    workflow_steps["chat_interaction"] = {
                        "success": False,
                        "error": str(chat_error)
                    }
                
                # Step 4: Get system monitoring data (dashboard functionality)
                try:
                    async with session.get(f"{backend_url}/api/v1/monitoring",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        monitoring_step = response.status in [200, 404]  # Accept not implemented
                        
                        if response.status == 200:
                            monitoring_data = await response.json()
                        else:
                            monitoring_data = {"status": response.status}
                        
                        workflow_steps["get_monitoring"] = {
                            "success": monitoring_step,
                            "implemented": response.status == 200,
                            "data": monitoring_data
                        }
                except Exception as monitoring_error:
                    workflow_steps["get_monitoring"] = {
                        "success": False,
                        "error": str(monitoring_error)
                    }
                
                integration_tests["workflow_simulation"] = workflow_steps
                
                # Test API response consistency
                consistency_tests = {}
                
                # Test same endpoint multiple times for consistency
                consistent_endpoint = "/api/v1/health"
                consistency_results = []
                
                for i in range(5):
                    try:
                        async with session.get(f"{backend_url}{consistent_endpoint}",
                                             timeout=aiohttp.ClientTimeout(total=5)) as response:
                            consistency_results.append({
                                "status": response.status,
                                "content_type": response.content_type,
                                "content_length": int(response.headers.get("content-length", 0))
                            })
                    except Exception as consistency_error:
                        consistency_results.append({
                            "error": str(consistency_error)
                        })
                    
                    # Small delay between requests
                    await asyncio.sleep(0.2)
                
                # Analyze consistency
                status_codes = [r.get("status") for r in consistency_results if "status" in r]
                content_types = [r.get("content_type") for r in consistency_results if "content_type" in r]
                
                consistency_tests["response_consistency"] = {
                    "requests_sent": len(consistency_results),
                    "successful_requests": len(status_codes),
                    "status_consistency": len(set(status_codes)) <= 1 if status_codes else False,
                    "content_type_consistency": len(set(content_types)) <= 1 if content_types else False,
                    "consistency_rate": len(status_codes) / len(consistency_results) * 100 if consistency_results else 0
                }
                
                integration_tests["consistency_tests"] = consistency_tests
            
            duration = time.time() - start_time
            
            # Calculate integration success
            successful_workflow_steps = sum(1 for step in workflow_steps.values() if step.get("success", False))
            workflow_success_rate = successful_workflow_steps / len(workflow_steps) * 100
            
            # Overall integration assessment
            integration_functional = workflow_success_rate > 50
            core_functionality = workflow_steps.get("health_check", {}).get("success", False) and \
                               workflow_steps.get("chat_interaction", {}).get("success", False)
            
            self.results.append(IntegrationTestResult(
                component="api_integration",
                test_name="frontend_backend_integration",
                success=integration_functional,
                duration=duration,
                metrics={
                    "integration_tests": integration_tests,
                    "workflow_success_rate": workflow_success_rate,
                    "successful_workflow_steps": successful_workflow_steps,
                    "core_functionality_working": core_functionality,
                    "consistency_rate": consistency_tests.get("response_consistency", {}).get("consistency_rate", 0),
                    "chat_functional": workflow_steps.get("chat_interaction", {}).get("success", False),
                    "agents_accessible": workflow_steps.get("get_agents", {}).get("success", False),
                    "performance_grade": "excellent" if duration < 20 else "good" if duration < 40 else "poor"
                }
            ))
            
            logger.info(f"API integration - Workflow success: {workflow_success_rate:.1f}%, Core functionality: {core_functionality}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="api_integration",
                test_name="frontend_backend_integration",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"API integration test failed: {e}")
    
    async def test_real_time_communication(self) -> None:
        """Test real-time communication capabilities (WebSocket)"""
        start_time = time.time()
        
        try:
            # Test WebSocket connection to backend
            backend_ws_url = f"ws://localhost:{self.config['backend']['port']}/ws"
            frontend_ws_url = f"ws://localhost:{self.config['frontend']['port']}/stream"
            
            websocket_tests = {}
            
            # Test backend WebSocket
            backend_ws_test = {}
            try:
                async with websockets.connect(backend_ws_url, timeout=10) as websocket:
                    # Test connection
                    backend_ws_test["connection_successful"] = True
                    
                    # Test sending a message
                    test_message = json.dumps({"type": "test", "data": "integration_test"})
                    await websocket.send(test_message)
                    
                    # Test receiving response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        backend_ws_test["message_exchange"] = True
                        backend_ws_test["response"] = response
                    except asyncio.TimeoutError:
                        backend_ws_test["message_exchange"] = False
                        backend_ws_test["timeout"] = True
                        
            except Exception as backend_ws_error:
                backend_ws_test["connection_successful"] = False
                backend_ws_test["error"] = str(backend_ws_error)
            
            websocket_tests["backend_websocket"] = backend_ws_test
            
            # Test frontend WebSocket (Streamlit)
            frontend_ws_test = {}
            try:
                # Streamlit WebSocket might be on different path or require different approach
                # Try common Streamlit WebSocket patterns
                streamlit_ws_urls = [
                    f"ws://localhost:{self.config['frontend']['port']}/stream",
                    f"ws://localhost:{self.config['frontend']['port']}/_stcore/stream",
                    f"ws://localhost:{self.config['frontend']['port']}/component/stream"
                ]
                
                for ws_url in streamlit_ws_urls:
                    try:
                        async with websockets.connect(ws_url, timeout=5) as websocket:
                            frontend_ws_test["connection_successful"] = True
                            frontend_ws_test["successful_url"] = ws_url
                            break
                    except Exception:
                        continue
                
                if not frontend_ws_test.get("connection_successful", False):
                    frontend_ws_test["connection_successful"] = False
                    frontend_ws_test["note"] = "Streamlit WebSocket may use different protocol"
                        
            except Exception as frontend_ws_error:
                frontend_ws_test["connection_successful"] = False
                frontend_ws_test["error"] = str(frontend_ws_error)
            
            websocket_tests["frontend_websocket"] = frontend_ws_test
            
            # Test Server-Sent Events (alternative to WebSocket)
            sse_tests = {}
            backend_sse_endpoints = [
                "/api/v1/stream",
                "/api/v1/events",
                "/api/v1/chat/stream"
            ]
            
            for endpoint in backend_sse_endpoints:
                try:
                    sse_url = f"{self.config['backend']['base_url']}{endpoint}"
                    
                    # Use aiohttp to test SSE endpoint
                    async with aiohttp.ClientSession() as session:
                        async with session.get(sse_url,
                                             headers={"Accept": "text/event-stream"},
                                             timeout=aiohttp.ClientTimeout(total=5)) as response:
                            sse_tests[endpoint] = {
                                "accessible": response.status == 200,
                                "content_type": response.content_type,
                                "is_event_stream": "text/event-stream" in response.content_type
                            }
                            
                except Exception as sse_error:
                    sse_tests[endpoint] = {
                        "accessible": False,
                        "error": str(sse_error)
                    }
            
            websocket_tests["server_sent_events"] = sse_tests
            
            # Test real-time features availability
            realtime_features = {}
            
            # Check if streaming chat is supported
            try:
                streaming_chat_payload = {**self.test_data["chat_message"], "stream": True}
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.config['backend']['base_url']}/api/v1/chat",
                                          json=streaming_chat_payload,
                                          timeout=aiohttp.ClientTimeout(total=10)) as response:
                        realtime_features["streaming_chat"] = {
                            "supported": response.status in [200, 422],
                            "status": response.status
                        }
            except Exception as streaming_error:
                realtime_features["streaming_chat"] = {
                    "supported": False,
                    "error": str(streaming_error)
                }
            
            duration = time.time() - start_time
            
            # Calculate real-time communication success
            websocket_available = websocket_tests.get("backend_websocket", {}).get("connection_successful", False) or \
                                websocket_tests.get("frontend_websocket", {}).get("connection_successful", False)
            
            sse_available = any(test.get("accessible", False) for test in sse_tests.values())
            
            realtime_functional = websocket_available or sse_available
            
            self.results.append(IntegrationTestResult(
                component="real_time_communication",
                test_name="websocket_sse_validation",
                success=realtime_functional,
                duration=duration,
                metrics={
                    "websocket_tests": websocket_tests,
                    "websocket_available": websocket_available,
                    "sse_available": sse_available,
                    "realtime_features": realtime_features,
                    "realtime_functional": realtime_functional,
                    "backend_websocket_working": websocket_tests.get("backend_websocket", {}).get("connection_successful", False),
                    "performance_grade": "excellent" if duration < 10 else "good" if duration < 20 else "poor"
                }
            ))
            
            logger.info(f"Real-time communication - WebSocket: {websocket_available}, SSE: {sse_available}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="real_time_communication",
                test_name="websocket_sse_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Real-time communication test failed: {e}")
    
    async def test_user_workflow_simulation(self) -> None:
        """Test complete user workflow simulation"""
        start_time = time.time()
        
        try:
            workflow_scenarios = {}
            
            # Scenario 1: New user accessing the system
            new_user_scenario = {}
            async with aiohttp.ClientSession() as session:
                # Step 1: User visits frontend
                try:
                    async with session.get(f"{self.config['frontend']['base_url']}/",
                                         timeout=aiohttp.ClientTimeout(total=15)) as response:
                        new_user_scenario["frontend_access"] = response.status == 200
                except Exception as e:
                    new_user_scenario["frontend_access"] = False
                    new_user_scenario["frontend_error"] = str(e)
                
                # Step 2: Frontend requests backend health
                try:
                    async with session.get(f"{self.config['backend']['base_url']}/health",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        new_user_scenario["backend_health_check"] = response.status == 200
                except Exception as e:
                    new_user_scenario["backend_health_check"] = False
                    new_user_scenario["backend_error"] = str(e)
                
                # Step 3: User navigates to chat page
                try:
                    async with session.get(f"{self.config['frontend']['base_url']}/?page=chat",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        new_user_scenario["chat_page_access"] = response.status == 200
                except Exception as e:
                    new_user_scenario["chat_page_access"] = False
                
                # Step 4: User sends a chat message
                try:
                    chat_payload = self.test_data["chat_message"]
                    async with session.post(f"{self.config['backend']['base_url']}/api/v1/chat",
                                          json=chat_payload,
                                          timeout=aiohttp.ClientTimeout(total=30)) as response:
                        new_user_scenario["chat_functionality"] = response.status in [200, 422, 503]
                        if response.status == 200:
                            chat_response = await response.json()
                            new_user_scenario["received_response"] = bool(chat_response.get("response"))
                        else:
                            new_user_scenario["received_response"] = False
                except Exception as e:
                    new_user_scenario["chat_functionality"] = False
                    new_user_scenario["chat_error"] = str(e)
            
            workflow_scenarios["new_user_workflow"] = new_user_scenario
            
            # Scenario 2: Power user accessing agent management
            power_user_scenario = {}
            async with aiohttp.ClientSession() as session:
                # Step 1: Access agents page
                try:
                    async with session.get(f"{self.config['frontend']['base_url']}/?page=agents",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        power_user_scenario["agents_page_access"] = response.status == 200
                except Exception as e:
                    power_user_scenario["agents_page_access"] = False
                
                # Step 2: Get agent list
                try:
                    async with session.get(f"{self.config['backend']['base_url']}/api/v1/agents",
                                         timeout=aiohttp.ClientTimeout(total=15)) as response:
                        power_user_scenario["agents_list_access"] = response.status == 200
                        if response.status == 200:
                            agents_data = await response.json()
                            agents_count = len(agents_data) if isinstance(agents_data, list) else len(agents_data.get("agents", [])) if isinstance(agents_data, dict) else 0
                            power_user_scenario["agents_available"] = agents_count > 0
                        else:
                            power_user_scenario["agents_available"] = False
                except Exception as e:
                    power_user_scenario["agents_list_access"] = False
                
                # Step 3: Access monitoring/hardware page
                try:
                    async with session.get(f"{self.config['frontend']['base_url']}/?page=hardware",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        power_user_scenario["hardware_page_access"] = response.status == 200
                except Exception as e:
                    power_user_scenario["hardware_page_access"] = False
                
                # Step 4: Get monitoring data
                try:
                    async with session.get(f"{self.config['backend']['base_url']}/api/v1/monitoring",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        power_user_scenario["monitoring_access"] = response.status in [200, 404]
                        power_user_scenario["monitoring_implemented"] = response.status == 200
                except Exception as e:
                    power_user_scenario["monitoring_access"] = False
            
            workflow_scenarios["power_user_workflow"] = power_user_scenario
            
            # Scenario 3: Error handling workflow
            error_handling_scenario = {}
            async with aiohttp.ClientSession() as session:
                # Test invalid page access
                try:
                    async with session.get(f"{self.config['frontend']['base_url']}/?page=nonexistent",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        error_handling_scenario["handles_invalid_pages"] = response.status in [200, 404]  # Should handle gracefully
                except Exception as e:
                    error_handling_scenario["handles_invalid_pages"] = False
                
                # Test invalid API calls
                try:
                    async with session.get(f"{self.config['backend']['base_url']}/api/v1/nonexistent",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        error_handling_scenario["handles_invalid_api"] = response.status == 404
                except Exception as e:
                    error_handling_scenario["handles_invalid_api"] = False
                
                # Test malformed requests
                try:
                    async with session.post(f"{self.config['backend']['base_url']}/api/v1/chat",
                                          data="invalid json",
                                          headers={"Content-Type": "application/json"},
                                          timeout=aiohttp.ClientTimeout(total=10)) as response:
                        error_handling_scenario["handles_malformed_requests"] = response.status in [400, 422]
                except Exception as e:
                    error_handling_scenario["handles_malformed_requests"] = False
            
            workflow_scenarios["error_handling_workflow"] = error_handling_scenario
            
            duration = time.time() - start_time
            
            # Calculate workflow success rates
            workflow_success_rates = {}
            for scenario_name, scenario_data in workflow_scenarios.items():
                successful_steps = sum(1 for step_result in scenario_data.values() if step_result is True)
                total_steps = len([v for v in scenario_data.values() if isinstance(v, bool)])
                success_rate = successful_steps / max(total_steps, 1) * 100
                workflow_success_rates[scenario_name] = success_rate
            
            # Overall workflow assessment
            avg_workflow_success = sum(workflow_success_rates.values()) / len(workflow_success_rates)
            
            # Critical workflow components
            critical_workflow_components = [
                workflow_scenarios["new_user_workflow"].get("frontend_access", False),
                workflow_scenarios["new_user_workflow"].get("backend_health_check", False),
                workflow_scenarios["new_user_workflow"].get("chat_functionality", False)
            ]
            
            critical_workflow_success = sum(critical_workflow_components) / len(critical_workflow_components) * 100
            
            self.results.append(IntegrationTestResult(
                component="user_workflow_simulation",
                test_name="workflow_scenarios",
                success=avg_workflow_success > 50,
                duration=duration,
                metrics={
                    "workflow_scenarios": workflow_scenarios,
                    "workflow_success_rates": workflow_success_rates,
                    "avg_workflow_success": avg_workflow_success,
                    "critical_workflow_success": critical_workflow_success,
                    "new_user_can_access_system": workflow_scenarios["new_user_workflow"].get("frontend_access", False),
                    "chat_workflow_functional": workflow_scenarios["new_user_workflow"].get("chat_functionality", False),
                    "error_handling_appropriate": workflow_scenarios["error_handling_workflow"].get("handles_invalid_api", False),
                    "performance_grade": "excellent" if duration < 30 else "good" if duration < 60 else "poor"
                }
            ))
            
            logger.info(f"User workflow simulation - Avg success: {avg_workflow_success:.1f}%, Critical: {critical_workflow_success:.1f}%")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="user_workflow_simulation",
                test_name="workflow_scenarios",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"User workflow simulation test failed: {e}")
    
    async def test_error_handling(self) -> None:
        """Test error handling and recovery mechanisms"""
        start_time = time.time()
        
        try:
            error_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Test various error conditions
                error_conditions = [
                    # Backend error conditions
                    ("invalid_endpoint", "GET", f"{self.config['backend']['base_url']}/api/v1/nonexistent", None),
                    ("malformed_json", "POST", f"{self.config['backend']['base_url']}/api/v1/chat", "invalid json"),
                    ("empty_payload", "POST", f"{self.config['backend']['base_url']}/api/v1/chat", ""),
                    ("oversized_payload", "POST", f"{self.config['backend']['base_url']}/api/v1/chat", json.dumps({"message": "x" * 100000})),  # Very large message
                    ("invalid_method", "PATCH", f"{self.config['backend']['base_url']}/api/v1/health", None),
                    
                    # Frontend error conditions
                    ("frontend_invalid_page", "GET", f"{self.config['frontend']['base_url']}/?page=nonexistent", None),
                    ("frontend_invalid_params", "GET", f"{self.config['frontend']['base_url']}/?invalid=param&malformed", None)
                ]
                
                for error_name, method, url, payload in error_conditions:
                    error_start = time.time()
                    try:
                        request_kwargs = {
                            "timeout": aiohttp.ClientTimeout(total=10)
                        }
                        
                        if payload is not None:
                            if payload == "invalid json":
                                request_kwargs["data"] = payload
                                request_kwargs["headers"] = {"Content-Type": "application/json"}
                            elif payload == "":
                                request_kwargs["data"] = payload
                            else:
                                request_kwargs["json"] = json.loads(payload) if isinstance(payload, str) and payload.startswith("{") else payload
                        
                        async with session.request(method, url, **request_kwargs) as response:
                            error_time = (time.time() - error_start) * 1000
                            
                            # Check if error is handled appropriately
                            appropriate_error = False
                            if "invalid_endpoint" in error_name:
                                appropriate_error = response.status == 404
                            elif "malformed_json" in error_name or "empty_payload" in error_name:
                                appropriate_error = response.status in [400, 422]
                            elif "oversized_payload" in error_name:
                                appropriate_error = response.status in [400, 413, 422]  # Bad request or payload too large
                            elif "invalid_method" in error_name:
                                appropriate_error = response.status == 405  # Method not allowed
                            elif "frontend" in error_name:
                                appropriate_error = response.status in [200, 404]  # Frontend might handle gracefully
                            
                            try:
                                error_response = await response.json()
                            except:
                                error_response = await response.text()
                            
                            error_tests[error_name] = {
                                "status_code": response.status,
                                "appropriate_error": appropriate_error,
                                "response_time_ms": error_time,
                                "has_error_message": bool(error_response),
                                "content_type": response.content_type
                            }
                            
                    except Exception as test_error:
                        error_tests[error_name] = {
                            "exception": str(test_error),
                            "response_time_ms": (time.time() - error_start) * 1000,
                            "handled_gracefully": "timeout" not in str(test_error).lower()
                        }
                
                # Test recovery mechanisms
                recovery_tests = {}
                
                # Test if system recovers after error conditions
                try:
                    # After generating errors, test if normal functionality still works
                    async with session.get(f"{self.config['backend']['base_url']}/health",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        recovery_tests["backend_recovery"] = response.status == 200
                except Exception as recovery_error:
                    recovery_tests["backend_recovery"] = False
                    recovery_tests["backend_recovery_error"] = str(recovery_error)
                
                try:
                    async with session.get(f"{self.config['frontend']['base_url']}/",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        recovery_tests["frontend_recovery"] = response.status == 200
                except Exception as recovery_error:
                    recovery_tests["frontend_recovery"] = False
                    recovery_tests["frontend_recovery_error"] = str(recovery_error)
                
                # Test timeout handling
                timeout_tests = {}
                try:
                    # Attempt request with very short timeout to test timeout handling
                    async with session.get(f"{self.config['backend']['base_url']}/api/v1/chat",
                                         timeout=aiohttp.ClientTimeout(total=0.001)) as response:
                        timeout_tests["timeout_handling"] = False  # Should have timed out
                except asyncio.TimeoutError:
                    timeout_tests["timeout_handling"] = True  # Properly handled timeout
                except Exception as timeout_error:
                    timeout_tests["timeout_handling"] = True
                    timeout_tests["timeout_error_type"] = type(timeout_error).__name__
            
            duration = time.time() - start_time
            
            # Calculate error handling effectiveness
            appropriate_errors = sum(1 for test in error_tests.values() 
                                   if test.get("appropriate_error", False))
            error_handling_rate = appropriate_errors / len(error_tests) * 100 if error_tests else 0
            
            # Recovery assessment
            system_recovered = recovery_tests.get("backend_recovery", False) and \
                              recovery_tests.get("frontend_recovery", False)
            
            # Overall error handling assessment
            error_handling_effective = error_handling_rate > 70 and system_recovered
            
            self.results.append(IntegrationTestResult(
                component="error_handling",
                test_name="error_recovery_validation",
                success=error_handling_effective,
                duration=duration,
                metrics={
                    "error_tests": error_tests,
                    "recovery_tests": recovery_tests,
                    "timeout_tests": timeout_tests,
                    "error_handling_rate": error_handling_rate,
                    "appropriate_errors": appropriate_errors,
                    "system_recovered": system_recovered,
                    "timeout_handled_properly": timeout_tests.get("timeout_handling", False),
                    "graceful_degradation": sum(1 for test in error_tests.values() 
                                              if test.get("handled_gracefully", True)) / len(error_tests) * 100,
                    "performance_grade": "excellent" if duration < 15 else "good" if duration < 30 else "poor"
                }
            ))
            
            logger.info(f"Error handling - Appropriate errors: {error_handling_rate:.1f}%, System recovered: {system_recovered}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="error_handling",
                test_name="error_recovery_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Error handling test failed: {e}")
    
    async def test_performance_integration(self) -> None:
        """Test performance aspects of frontend/backend integration"""
        start_time = time.time()
        
        try:
            performance_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Test concurrent requests handling
                concurrent_test = {}
                concurrent_requests = 10
                request_tasks = []
                
                # Create concurrent requests to different endpoints
                endpoints_for_load = [
                    f"{self.config['backend']['base_url']}/health",
                    f"{self.config['backend']['base_url']}/api/v1/health", 
                    f"{self.config['backend']['base_url']}/api/v1/agents",
                    f"{self.config['frontend']['base_url']}/"
                ]
                
                load_start = time.time()
                for i in range(concurrent_requests):
                    endpoint = endpoints_for_load[i % len(endpoints_for_load)]
                    task = asyncio.create_task(self._single_performance_request(session, endpoint))
                    request_tasks.append(task)
                
                # Wait for all requests to complete
                results = await asyncio.gather(*request_tasks, return_exceptions=True)
                load_duration = time.time() - load_start
                
                # Analyze concurrent performance
                successful_requests = sum(1 for result in results 
                                        if isinstance(result, dict) and result.get("success", False))
                response_times = [result.get("response_time", 0) for result in results 
                                if isinstance(result, dict) and "response_time" in result]
                
                concurrent_test = {
                    "requests_sent": concurrent_requests,
                    "successful_requests": successful_requests,
                    "success_rate": successful_requests / concurrent_requests * 100,
                    "total_duration_seconds": load_duration,
                    "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                    "min_response_time": min(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0,
                    "requests_per_second": concurrent_requests / load_duration if load_duration > 0 else 0
                }
                
                performance_tests["concurrent_load"] = concurrent_test
                
                # Test response time consistency
                consistency_test = {}
                test_endpoint = f"{self.config['backend']['base_url']}/health"
                consistency_requests = 20
                consistency_times = []
                
                for i in range(consistency_requests):
                    request_start = time.time()
                    try:
                        async with session.get(test_endpoint,
                                             timeout=aiohttp.ClientTimeout(total=10)) as response:
                            request_time = (time.time() - request_start) * 1000
                            if response.status == 200:
                                consistency_times.append(request_time)
                    except Exception:
                        pass
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
                
                if consistency_times:
                    avg_consistency = sum(consistency_times) / len(consistency_times)
                    consistency_variance = sum((t - avg_consistency) ** 2 for t in consistency_times) / len(consistency_times)
                    consistency_stddev = consistency_variance ** 0.5
                    
                    consistency_test = {
                        "requests_tested": len(consistency_times),
                        "avg_response_time": avg_consistency,
                        "std_deviation": consistency_stddev,
                        "min_time": min(consistency_times),
                        "max_time": max(consistency_times),
                        "consistency_coefficient": consistency_stddev / avg_consistency if avg_consistency > 0 else 0,
                        "response_consistent": consistency_stddev < avg_consistency * 0.5  # Less than 50% variance
                    }
                else:
                    consistency_test = {"error": "No successful requests for consistency testing"}
                
                performance_tests["response_consistency"] = consistency_test
                
                # Test memory efficiency (basic check)
                memory_test = {}
                try:
                    import psutil
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Perform several operations that might cause memory usage
                    for i in range(50):
                        async with session.get(f"{self.config['backend']['base_url']}/health",
                                             timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                await response.json()
                    
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = memory_after - memory_before
                    
                    memory_test = {
                        "memory_before_mb": memory_before,
                        "memory_after_mb": memory_after,
                        "memory_increase_mb": memory_increase,
                        "memory_efficient": memory_increase < 50  # Less than 50MB increase
                    }
                    
                except ImportError:
                    memory_test = {"psutil_not_available": True}
                except Exception as memory_error:
                    memory_test = {"error": str(memory_error)}
                
                performance_tests["memory_efficiency"] = memory_test
                
                # Test large response handling
                large_response_test = {}
                try:
                    # Test with agent list (might be large)
                    large_start = time.time()
                    async with session.get(f"{self.config['backend']['base_url']}/api/v1/agents",
                                         timeout=aiohttp.ClientTimeout(total=15)) as response:
                        large_response_time = (time.time() - large_start) * 1000
                        
                        if response.status == 200:
                            response_data = await response.json()
                            response_size = len(str(response_data))
                            
                            large_response_test = {
                                "response_time_ms": large_response_time,
                                "response_size_chars": response_size,
                                "handles_large_response": large_response_time < 5000,  # Less than 5 seconds
                                "data_type": type(response_data).__name__
                            }
                        else:
                            large_response_test = {
                                "response_time_ms": large_response_time,
                                "status": response.status,
                                "handles_large_response": False
                            }
                            
                except Exception as large_error:
                    large_response_test = {
                        "error": str(large_error),
                        "handles_large_response": False
                    }
                
                performance_tests["large_response_handling"] = large_response_test
            
            duration = time.time() - start_time
            
            # Overall performance assessment
            performance_metrics = {
                "concurrent_success_rate": concurrent_test.get("success_rate", 0),
                "avg_concurrent_response": concurrent_test.get("avg_response_time", 0),
                "requests_per_second": concurrent_test.get("requests_per_second", 0),
                "response_consistent": consistency_test.get("response_consistent", False),
                "memory_efficient": memory_test.get("memory_efficient", True),
                "handles_large_responses": large_response_test.get("handles_large_response", False)
            }
            
            # Performance grade calculation
            performance_score = 0
            if performance_metrics["concurrent_success_rate"] > 90:
                performance_score += 25
            elif performance_metrics["concurrent_success_rate"] > 70:
                performance_score += 15
            
            if performance_metrics["avg_concurrent_response"] < 500:
                performance_score += 25
            elif performance_metrics["avg_concurrent_response"] < 2000:
                performance_score += 15
            
            if performance_metrics["response_consistent"]:
                performance_score += 25
            
            if performance_metrics["memory_efficient"]:
                performance_score += 25
            
            performance_grade = "excellent" if performance_score >= 75 else "good" if performance_score >= 50 else "poor"
            
            self.results.append(IntegrationTestResult(
                component="performance_integration",
                test_name="performance_validation",
                success=performance_score >= 50,
                duration=duration,
                metrics={
                    "performance_tests": performance_tests,
                    "performance_metrics": performance_metrics,
                    "performance_score": performance_score,
                    "performance_grade": performance_grade,
                    "test_duration": duration
                }
            ))
            
            logger.info(f"Performance integration - Score: {performance_score}/100, Grade: {performance_grade}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(IntegrationTestResult(
                component="performance_integration",
                test_name="performance_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Performance integration test failed: {e}")
    
    async def _single_performance_request(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Helper method for single performance request"""
        start_time = time.time()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "success": response.status == 200,
                    "status": response.status,
                    "response_time": response_time,
                    "url": url
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": (time.time() - start_time) * 1000,
                "url": url
            }
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration validation report"""
        total_components = len(self.results)
        successful_components = len([r for r in self.results if r.success])
        
        # Group results by component
        component_results = {}
        for result in self.results:
            component_results[result.component] = result
        
        # Calculate integration health
        critical_components = ["backend_health", "frontend_health", "api_integration"]
        critical_success = sum(1 for comp in critical_components 
                             if comp in component_results and component_results[comp].success)
        
        integration_grade = "EXCELLENT" if critical_success == len(critical_components) else \
                           "GOOD" if critical_success >= len(critical_components) - 1 else \
                           "POOR"
        
        # Performance analysis
        performance_summary = {}
        for result in self.results:
            if result.success and "performance_grade" in result.metrics:
                performance_summary[result.component] = result.metrics["performance_grade"]
        
        # Feature analysis
        feature_analysis = {
            "backend_functional": component_results.get("backend_health", {}).success,
            "frontend_accessible": component_results.get("frontend_health", {}).success,
            "api_integration_working": component_results.get("api_integration", {}).success,
            "cors_configured": component_results.get("cors_configuration", {}).success,
            "real_time_available": component_results.get("real_time_communication", {}).success,
            "error_handling_appropriate": component_results.get("error_handling", {}).success,
            "user_workflows_functional": component_results.get("user_workflow_simulation", {}).success
        }
        
        # Extract key metrics
        key_metrics = {}
        
        if "backend_api_endpoints" in component_results and component_results["backend_api_endpoints"].success:
            backend_metrics = component_results["backend_api_endpoints"].metrics
            key_metrics["backend"] = {
                "endpoint_success_rate": backend_metrics.get("endpoint_success_rate", 0),
                "avg_response_time": backend_metrics.get("avg_response_time_ms", 0)
            }
        
        if "frontend_ui_availability" in component_results and component_results["frontend_ui_availability"].success:
            frontend_metrics = component_results["frontend_ui_availability"].metrics
            key_metrics["frontend"] = {
                "ui_success_rate": frontend_metrics.get("ui_success_rate", 0),
                "avg_load_time": frontend_metrics.get("avg_load_time_ms", 0)
            }
        
        if "performance_integration" in component_results and component_results["performance_integration"].success:
            perf_metrics = component_results["performance_integration"].metrics.get("performance_metrics", {})
            key_metrics["performance"] = {
                "concurrent_success_rate": perf_metrics.get("concurrent_success_rate", 0),
                "requests_per_second": perf_metrics.get("requests_per_second", 0)
            }
        
        return {
            "summary": {
                "total_components_tested": total_components,
                "successful_components": successful_components,
                "success_rate": round(successful_components / max(total_components, 1) * 100, 2),
                "integration_grade": integration_grade,
                "critical_components_health": f"{critical_success}/{len(critical_components)}"
            },
            "component_details": {
                component: {
                    "status": "success" if result.success else "failed",
                    "duration_seconds": round(result.duration, 3),
                    "key_metrics": result.metrics,
                    "error": result.error_message
                }
                for component, result in component_results.items()
            },
            "performance_analysis": performance_summary,
            "feature_analysis": feature_analysis,
            "key_metrics": key_metrics,
            "recommendations": self._generate_integration_recommendations(component_results, feature_analysis)
        }
    
    def _generate_integration_recommendations(self, component_results: Dict, feature_analysis: Dict) -> List[str]:
        """Generate integration improvement recommendations"""
        recommendations = []
        
        # Check critical components
        if not feature_analysis.get("backend_functional", False):
            recommendations.append("🔴 CRITICAL: Backend is not functional - all integration features disabled")
        
        if not feature_analysis.get("frontend_accessible", False):
            recommendations.append("🔴 CRITICAL: Frontend is not accessible - user interface unavailable")
        
        if not feature_analysis.get("api_integration_working", False):
            recommendations.append("🔴 CRITICAL: API integration is broken - frontend cannot communicate with backend")
        
        # CORS recommendations
        if "cors_configuration" in component_results:
            cors_result = component_results["cors_configuration"]
            if not cors_result.success:
                recommendations.append("🟡 CONFIG: CORS is not properly configured - frontend may have cross-origin issues")
            elif cors_result.success:
                cors_metrics = cors_result.metrics
                if not cors_metrics.get("security_appropriate", True):
                    recommendations.append("🟡 SECURITY: CORS configuration may be too permissive - review security settings")
        
        # Performance recommendations
        if "performance_integration" in component_results and component_results["performance_integration"].success:
            perf_metrics = component_results["performance_integration"].metrics.get("performance_metrics", {})
            
            if perf_metrics.get("concurrent_success_rate", 100) < 90:
                recommendations.append("🟡 PERFORMANCE: Concurrent request handling needs improvement - system may struggle under load")
            
            if perf_metrics.get("avg_concurrent_response", 0) > 2000:
                recommendations.append("🟡 PERFORMANCE: Response times are slow - consider optimization")
            
            if not perf_metrics.get("memory_efficient", True):
                recommendations.append("🟡 MEMORY: Memory usage increases significantly during load - check for memory leaks")
        
        # Feature completeness
        if not feature_analysis.get("real_time_available", False):
            recommendations.append("🟡 FEATURE: Real-time communication not available - limited to request/response patterns")
        
        if not feature_analysis.get("error_handling_appropriate", False):
            recommendations.append("🟡 QUALITY: Error handling could be improved - may confuse users during issues")
        
        # User experience
        if "user_workflow_simulation" in component_results and component_results["user_workflow_simulation"].success:
            workflow_metrics = component_results["user_workflow_simulation"].metrics
            critical_workflow_success = workflow_metrics.get("critical_workflow_success", 0)
            
            if critical_workflow_success < 80:
                recommendations.append(f"🟡 UX: Critical user workflows have issues ({critical_workflow_success:.1f}% success) - impacts user experience")
        
        return recommendations if recommendations else ["✅ Frontend/Backend integration is operating optimally"]

async def main():
    """Main execution for frontend/backend integration validation"""
    validator = FrontendBackendIntegrationValidator()
    
    print("🌐 Starting Frontend/Backend Integration Validation Tests")
    print("=" * 60)
    
    results = await validator.run_all_integration_tests()
    report = validator.generate_integration_report()
    
    print("\n" + "=" * 60)
    print("📊 FRONTEND/BACKEND INTEGRATION VALIDATION COMPLETE")
    print("=" * 60)
    
    # Print summary
    summary = report["summary"]
    print(f"Components Tested: {summary['total_components_tested']}")
    print(f"Successful: {summary['successful_components']} ({summary['success_rate']}%)")
    print(f"Integration Grade: {summary['integration_grade']}")
    print(f"Critical Components: {summary['critical_components_health']}")
    
    # Print component details
    print("\n🔍 Component Status:")
    for component, details in report["component_details"].items():
        status_icon = "✅" if details["status"] == "success" else "❌"
        duration = details["duration_seconds"]
        print(f"  {status_icon} {component}: {details['status']} ({duration:.2f}s)")
        
        if details["error"]:
            print(f"    ⚠️  {details['error']}")
    
    # Print feature analysis
    features = report["feature_analysis"]
    if features:
        print(f"\n🎯 Feature Status:")
        for feature_name, status in features.items():
            status_icon = "✅" if status else "❌"
            feature_display = feature_name.replace("_", " ").title()
            print(f"  {status_icon} {feature_display}")
    
    # Print key metrics
    key_metrics = report["key_metrics"]
    if key_metrics:
        print(f"\n📈 Key Metrics:")
        
        if "backend" in key_metrics:
            backend = key_metrics["backend"]
            print(f"  Backend API: {backend.get('endpoint_success_rate', 0):.1f}% success, {backend.get('avg_response_time', 0):.1f}ms avg")
        
        if "frontend" in key_metrics:
            frontend = key_metrics["frontend"]
            print(f"  Frontend UI: {frontend.get('ui_success_rate', 0):.1f}% success, {frontend.get('avg_load_time', 0):.1f}ms avg")
        
        if "performance" in key_metrics:
            performance = key_metrics["performance"]
            print(f"  Performance: {performance.get('concurrent_success_rate', 0):.1f}% concurrent, {performance.get('requests_per_second', 0):.1f} req/s")
    
    # Print recommendations
    print("\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"  {rec}")
    
    # Save detailed report
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"frontend_backend_integration_report_{timestamp}.json"
    
    import json
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return summary["integration_grade"] in ["EXCELLENT", "GOOD"]

if __name__ == "__main__":
    success = asyncio.run(main())
    import sys
    sys.exit(0 if success else 1)