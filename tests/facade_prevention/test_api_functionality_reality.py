#!/usr/bin/env python3
"""
API Functionality Reality Tests - Facade Prevention Framework
============================================================

This module implements comprehensive tests to prevent facade implementations in API endpoints.
Tests verify that APIs actually perform the operations they claim to, not just return fake responses.

CRITICAL PURPOSE: Prevent facade APIs that claim to work but don't actually perform the intended operations,
which was a major issue in the recent facade implementation problems.
"""

import asyncio
import pytest
import json
import time
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import httpx
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIFunctionalityRealityTester:
    """
    Tests that verify APIs actually perform operations rather than just claiming to.
    
    FACADE PREVENTION: These tests catch APIs that return success responses 
    but don't actually perform the requested operations.
    """
    
    def __init__(self, base_url: str = "http://localhost:10010"):
        self.base_url = base_url
        self.client = None
        self.test_session_id = str(uuid.uuid4())
        
        # Define API endpoints to test
        self.api_endpoints = {
            # Core system endpoints
            "health": {
                "method": "GET",
                "path": "/health",
                "expected_status": 200,
                "reality_check": self._verify_health_reality
            },
            "system_info": {
                "method": "GET", 
                "path": "/api/v1/system/",
                "expected_status": 200,
                "reality_check": self._verify_system_info_reality
            },
            
            # Service mesh endpoints
            "service_discovery": {
                "method": "GET",
                "path": "/api/v1/mesh/v2/services",
                "expected_status": 200,
                "reality_check": self._verify_service_discovery_reality
            },
            "service_registration": {
                "method": "POST",
                "path": "/api/v1/mesh/v2/register",
                "expected_status": [200, 201],
                "reality_check": self._verify_service_registration_reality
            },
            
            # AI/Chat endpoints
            "chat": {
                "method": "POST",
                "path": "/api/v1/chat/",
                "expected_status": 200,
                "reality_check": self._verify_chat_reality
            },
            "models_list": {
                "method": "GET",
                "path": "/api/v1/models/",
                "expected_status": 200,
                "reality_check": self._verify_models_reality
            },
            
            # Agent endpoints
            "agents_list": {
                "method": "GET",
                "path": "/api/v1/agents/",
                "expected_status": 200,
                "reality_check": self._verify_agents_reality
            },
            
            # Hardware monitoring
            "hardware_status": {
                "method": "GET",
                "path": "/api/v1/hardware/",
                "expected_status": 200,
                "reality_check": self._verify_hardware_reality
            },
            
            # Cache operations
            "cache_status": {
                "method": "GET",
                "path": "/api/v1/cache/",
                "expected_status": 200,
                "reality_check": self._verify_cache_reality
            },
            
            # Document management
            "documents": {
                "method": "GET",
                "path": "/api/v1/documents/",
                "expected_status": 200,
                "reality_check": self._verify_documents_reality
            },
            
            # Feature flags
            "features": {
                "method": "GET",
                "path": "/api/v1/features/",
                "expected_status": 200,
                "reality_check": self._verify_features_reality
            }
        }
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def test_api_endpoint_reality(self, endpoint_name: str, config: Dict) -> Dict:
        """
        FACADE TEST: Verify an API endpoint actually performs its claimed function.
        
        PREVENTS: APIs that return success but don't actually work.
        """
        logger.info(f"🔍 Testing {endpoint_name} API reality...")
        
        method = config["method"]
        path = config["path"]
        expected_status = config["expected_status"]
        reality_check = config["reality_check"]
        
        try:
            # Step 1: Test basic API response
            url = f"{self.base_url}{path}"
            
            if method == "GET":
                response = await self.client.get(url)
            elif method == "POST":
                # Prepare test data based on endpoint
                test_data = await self._prepare_test_data(endpoint_name)
                response = await self.client.post(url, json=test_data)
            elif method == "PUT":
                test_data = await self._prepare_test_data(endpoint_name)
                response = await self.client.put(url, json=test_data)
            elif method == "DELETE":
                response = await self.client.delete(url)
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported method: {method}",
                    "test_passed": False
                }
            
            # Check basic response
            status_ok = (
                response.status_code in expected_status if isinstance(expected_status, list)
                else response.status_code == expected_status
            )
            
            if not status_ok:
                return {
                    "status": "failed",
                    "error": f"Unexpected status code: {response.status_code}",
                    "expected_status": expected_status,
                    "actual_status": response.status_code,
                    "test_passed": False
                }
            
            # Step 2: Perform reality check
            try:
                response_data = response.json() if response.content else {}
            except:
                response_data = {"raw_content": response.text}
            
            reality_result = await reality_check(response_data, response)
            
            return {
                "status": "tested",
                "basic_response_ok": status_ok,
                "response_status": response.status_code,
                "reality_check": reality_result,
                "test_passed": status_ok and reality_result.get("reality_verified", False),
                "is_facade": status_ok and not reality_result.get("reality_verified", False)
            }
            
        except Exception as e:
            logger.error(f"API test failed for {endpoint_name}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "test_passed": False
            }
    
    async def _prepare_test_data(self, endpoint_name: str) -> Dict:
        """Prepare test data for POST/PUT requests."""
        if endpoint_name == "service_registration":
            return {
                "name": f"facade-test-{self.test_session_id}",
                "address": "test-address",
                "port": 9999,
                "tags": ["test", "facade-prevention"],
                "metadata": {"test": "true", "session": self.test_session_id}
            }
        elif endpoint_name == "chat":
            return {
                "message": "Hello, this is a facade prevention test",
                "model": "tinyllama",
                "session_id": self.test_session_id
            }
        else:
            return {"test": True, "session_id": self.test_session_id}
    
    # Reality check methods for each endpoint
    async def _verify_health_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify health endpoint actually reflects system health."""
        try:
            # Check if health data makes sense
            status = response_data.get("status")
            if status not in ["healthy", "ok", "up"]:
                return {"reality_verified": False, "reason": "invalid_health_status"}
            
            # If health claims to be good, verify some basic services are accessible
            if status in ["healthy", "ok"]:
                # Test if database is actually accessible
                try:
                    db_response = await self.client.get(f"{self.base_url}/api/v1/system/")
                    db_accessible = db_response.status_code == 200
                except:
                    db_accessible = False
                
                if not db_accessible:
                    return {
                        "reality_verified": False, 
                        "reason": "health_claims_good_but_system_inaccessible"
                    }
            
            return {"reality_verified": True, "health_status": status}
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_system_info_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify system info endpoint returns actual system information."""
        try:
            # Check if response contains expected system info fields
            expected_fields = ["version", "status", "uptime", "services"]
            missing_fields = [field for field in expected_fields if field not in response_data]
            
            if missing_fields:
                return {
                    "reality_verified": False, 
                    "reason": "missing_system_info_fields",
                    "missing_fields": missing_fields
                }
            
            # Verify uptime makes sense (should be a positive number)
            uptime = response_data.get("uptime")
            if uptime is not None and uptime < 0:
                return {"reality_verified": False, "reason": "invalid_uptime"}
            
            return {"reality_verified": True, "system_info": response_data}
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_service_discovery_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify service discovery returns actual services."""
        try:
            services = response_data.get("services", [])
            count = response_data.get("count", 0)
            
            # Check if count matches actual services
            if len(services) != count:
                return {
                    "reality_verified": False,
                    "reason": "service_count_mismatch",
                    "claimed_count": count,
                    "actual_count": len(services)
                }
            
            # Check if services have required fields
            for service in services[:3]:  # Check first 3 services
                required_fields = ["id", "name", "address", "port"]
                missing_fields = [field for field in required_fields if field not in service]
                if missing_fields:
                    return {
                        "reality_verified": False,
                        "reason": "service_missing_required_fields",
                        "missing_fields": missing_fields
                    }
            
            # REALITY CHECK: Try to contact one of the discovered services
            if services:
                test_service = services[0]
                address = test_service.get("address")
                port = test_service.get("port")
                
                if address and port:
                    # Try to connect to the service
                    import socket
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(3.0)
                        result = sock.connect_ex((address, port))
                        sock.close()
                        service_reachable = result == 0
                    except:
                        service_reachable = False
                    
                    if not service_reachable:
                        return {
                            "reality_verified": False,
                            "reason": "discovered_service_not_reachable",
                            "test_service": f"{address}:{port}"
                        }
            
            return {
                "reality_verified": True,
                "services_count": len(services),
                "services_valid": True
            }
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_service_registration_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify service registration actually registers the service."""
        try:
            # After registration, check if service appears in discovery
            await asyncio.sleep(1)  # Give time for registration to propagate
            
            discovery_response = await self.client.get(f"{self.base_url}/api/v1/mesh/v2/services")
            if discovery_response.status_code != 200:
                return {
                    "reality_verified": False,
                    "reason": "cannot_verify_registration_discovery_failed"
                }
            
            services = discovery_response.json().get("services", [])
            test_service_name = f"facade-test-{self.test_session_id}"
            
            # Check if our test service appears in discovery
            registered_service = None
            for service in services:
                if service.get("name") == test_service_name:
                    registered_service = service
                    break
            
            if not registered_service:
                return {
                    "reality_verified": False,
                    "reason": "service_not_found_in_discovery_after_registration"
                }
            
            # Clean up - try to deregister
            try:
                await self.client.delete(f"{self.base_url}/api/v1/mesh/v2/services/{test_service_name}")
            except:
                pass  # Cleanup failure is not critical for test
            
            return {
                "reality_verified": True,
                "registered_service": registered_service
            }
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_chat_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify chat endpoint actually processes messages."""
        try:
            # Check if response contains a message
            if "response" not in response_data and "message" not in response_data:
                return {
                    "reality_verified": False,
                    "reason": "no_response_message_in_chat_response"
                }
            
            # Check if response is not empty
            response_text = response_data.get("response") or response_data.get("message", "")
            if not response_text or len(response_text.strip()) < 5:
                return {
                    "reality_verified": False,
                    "reason": "chat_response_too_short_or_empty"
                }
            
            # Check if response is not just an error message
            error_indicators = ["error", "failed", "exception", "unavailable"]
            if any(indicator in response_text.lower() for indicator in error_indicators):
                return {
                    "reality_verified": False,
                    "reason": "chat_response_indicates_error",
                    "response": response_text
                }
            
            return {
                "reality_verified": True,
                "response_length": len(response_text),
                "chat_working": True
            }
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_models_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify models endpoint returns actual available models."""
        try:
            # Check if models are listed
            models = response_data.get("models", [])
            if not models:
                # Try alternative response format
                if "data" in response_data:
                    models = response_data["data"]
            
            if not models:
                return {
                    "reality_verified": False,
                    "reason": "no_models_returned"
                }
            
            # Check if models have reasonable structure
            for model in models[:3]:  # Check first 3 models
                if isinstance(model, dict):
                    if "name" not in model and "id" not in model:
                        return {
                            "reality_verified": False,
                            "reason": "model_missing_name_or_id"
                        }
            
            # REALITY CHECK: Try to verify at least one model exists in Ollama
            try:
                ollama_response = await self.client.get("http://localhost:10104/api/tags")
                if ollama_response.status_code == 200:
                    ollama_models = ollama_response.json().get("models", [])
                    if not ollama_models and models:
                        return {
                            "reality_verified": False,
                            "reason": "api_claims_models_but_ollama_has_none"
                        }
            except:
                pass  # Ollama check is optional
            
            return {
                "reality_verified": True,
                "models_count": len(models)
            }
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_agents_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify agents endpoint returns actual agent information."""
        try:
            # Check response format
            agents = response_data.get("agents", [])
            if not agents and isinstance(response_data, list):
                agents = response_data
            
            if not agents:
                return {
                    "reality_verified": False,
                    "reason": "no_agents_returned"
                }
            
            # Check agent structure
            for agent in agents[:3]:  # Check first 3 agents
                if isinstance(agent, dict):
                    required_fields = ["name", "status"]
                    missing_fields = [field for field in required_fields if field not in agent]
                    if missing_fields:
                        return {
                            "reality_verified": False,
                            "reason": "agent_missing_required_fields",
                            "missing_fields": missing_fields
                        }
            
            return {
                "reality_verified": True,
                "agents_count": len(agents)
            }
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_hardware_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify hardware endpoint returns actual hardware information."""
        try:
            # Check if hardware metrics are present
            expected_metrics = ["cpu", "memory", "disk"]
            found_metrics = []
            
            for metric in expected_metrics:
                if metric in response_data or any(metric in key.lower() for key in response_data.keys()):
                    found_metrics.append(metric)
            
            if not found_metrics:
                return {
                    "reality_verified": False,
                    "reason": "no_hardware_metrics_found"
                }
            
            # Check if metrics have reasonable values
            for key, value in response_data.items():
                if isinstance(value, (int, float)):
                    if "cpu" in key.lower() and (value < 0 or value > 100):
                        return {
                            "reality_verified": False,
                            "reason": "invalid_cpu_metric_value",
                            "value": value
                        }
                    elif "memory" in key.lower() and value < 0:
                        return {
                            "reality_verified": False,
                            "reason": "invalid_memory_metric_value", 
                            "value": value
                        }
            
            return {
                "reality_verified": True,
                "metrics_found": found_metrics
            }
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_cache_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify cache endpoint returns actual cache information."""
        try:
            # Check if cache information is present
            cache_fields = ["status", "size", "hits", "misses", "keys"]
            found_fields = [field for field in cache_fields if field in response_data]
            
            if not found_fields:
                return {
                    "reality_verified": False,
                    "reason": "no_cache_information_found"
                }
            
            return {
                "reality_verified": True,
                "cache_fields": found_fields
            }
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_documents_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify documents endpoint returns actual document information."""
        try:
            # Check response structure
            documents = response_data.get("documents", [])
            if not documents and isinstance(response_data, list):
                documents = response_data
            
            # It's OK for documents to be empty, but structure should be valid
            if not isinstance(documents, list):
                return {
                    "reality_verified": False,
                    "reason": "documents_not_a_list"
                }
            
            return {
                "reality_verified": True,
                "documents_count": len(documents)
            }
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def _verify_features_reality(self, response_data: Dict, response: httpx.Response) -> Dict:
        """Verify features endpoint returns actual feature flag information."""
        try:
            # Check if features are present
            features = response_data.get("features", {})
            if not features and isinstance(response_data, dict):
                # Features might be at the root level
                features = response_data
            
            if not features:
                return {
                    "reality_verified": False,
                    "reason": "no_features_found"
                }
            
            # Check if features have boolean values (typical for feature flags)
            boolean_features = 0
            for key, value in features.items():
                if isinstance(value, bool):
                    boolean_features += 1
            
            return {
                "reality_verified": True,
                "features_count": len(features),
                "boolean_features": boolean_features
            }
            
        except Exception as e:
            return {"reality_verified": False, "error": str(e)}
    
    async def run_comprehensive_api_tests(self) -> Dict:
        """Run all API functionality reality tests and return comprehensive results."""
        logger.info("🚀 Starting comprehensive API functionality reality tests...")
        
        start_time = datetime.now()
        
        results = {
            "test_suite": "api_functionality_facade_prevention",
            "timestamp": start_time.isoformat(),
            "session_id": self.test_session_id,
            "tests": {}
        }
        
        passed_tests = 0
        total_tests = len(self.api_endpoints)
        facade_apis = 0
        
        for endpoint_name, config in self.api_endpoints.items():
            try:
                logger.info(f"Testing API endpoint: {endpoint_name}")
                test_result = await self.test_api_endpoint_reality(endpoint_name, config)
                results["tests"][endpoint_name] = test_result
                
                if test_result.get("test_passed", False):
                    passed_tests += 1
                
                if test_result.get("is_facade", False):
                    facade_apis += 1
                    
            except Exception as e:
                logger.error(f"API test failed for {endpoint_name}: {e}")
                results["tests"][endpoint_name] = {
                    "status": "error",
                    "error": str(e),
                    "test_passed": False
                }
        
        # Calculate overall results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results.update({
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "facade_apis": facade_apis,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "duration_seconds": duration
            },
            "overall_status": "passed" if passed_tests >= total_tests * 0.8 else "failed",  # 80% pass rate required
            "facade_issues_detected": facade_apis
        })
        
        logger.info(f"API functionality reality tests completed: {passed_tests}/{total_tests} passed")
        return results


# Pytest integration
@pytest.mark.asyncio
async def test_apis_are_not_facades():
    """
    Main facade prevention test for API functionality.
    
    This test MUST pass for deployment to prevent facade implementations.
    """
    async with APIFunctionalityRealityTester() as tester:
        results = await tester.run_comprehensive_api_tests()
        
        # CRITICAL: Fail if any facade APIs detected
        assert results["facade_issues_detected"] == 0, f"API FACADE IMPLEMENTATIONS DETECTED: {results}"
        assert results["overall_status"] == "passed", f"API functionality reality tests failed: {results}"
        
        # Log results for monitoring
        logger.info(f"✅ API functionality reality verification passed: {results['summary']}")


@pytest.mark.asyncio
async def test_critical_apis_work():
    """Test that critical APIs actually work."""
    async with APIFunctionalityRealityTester() as tester:
        critical_apis = ["health", "system_info", "service_discovery"]
        
        for api_name in critical_apis:
            if api_name in tester.api_endpoints:
                config = tester.api_endpoints[api_name]
                result = await tester.test_api_endpoint_reality(api_name, config)
                assert result["test_passed"], f"Critical API {api_name} failed: {result}"


@pytest.mark.asyncio
async def test_no_empty_responses():
    """Test that APIs don't return empty responses when they should have data."""
    async with APIFunctionalityRealityTester() as tester:
        # Test APIs that should return data
        data_apis = ["service_discovery", "models_list", "agents_list"]
        
        for api_name in data_apis:
            if api_name in tester.api_endpoints:
                config = tester.api_endpoints[api_name]
                result = await tester.test_api_endpoint_reality(api_name, config)
                
                # These APIs should not be facades
                assert not result.get("is_facade", False), f"API {api_name} is a facade: {result}"


if __name__ == "__main__":
    async def main():
        async with APIFunctionalityRealityTester() as tester:
            results = await tester.run_comprehensive_api_tests()
            print(json.dumps(results, indent=2))
            
            if results["facade_issues_detected"] > 0:
                print(f"\n❌ API FACADE ISSUES DETECTED: {results['facade_issues_detected']}")
                exit(1)
            else:
                print(f"\n✅ All API functionality reality tests passed!")
                exit(0)
    
    asyncio.run(main())