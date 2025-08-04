#!/usr/bin/env python3
"""
SutazAI Multi-Agent Orchestration Test Suite
Comprehensive testing of the multi-agent orchestration system with
real-world scenarios and performance validation.
"""

import asyncio
import json
import time
import logging
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestrationTester:
    """
    Comprehensive test suite for the SutazAI multi-agent orchestration system
    """
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.orchestration_url = f"{base_url}/api/v1/orchestration"
        self.session = None
        
        # Test results
        self.test_results = {
            "agent_discovery": {"status": "pending", "details": {}},
            "task_routing": {"status": "pending", "details": {}},
            "workflow_execution": {"status": "pending", "details": {}},
            "load_balancing": {"status": "pending", "details": {}},
            "message_passing": {"status": "pending", "details": {}},
            "consensus": {"status": "pending", "details": {}},
            "resource_allocation": {"status": "pending", "details": {}},
            "performance": {"status": "pending", "details": {}},
            "fault_tolerance": {"status": "pending", "details": {}},
            "monitoring": {"status": "pending", "details": {}}
        }
        
        # Test configurations
        self.test_workflows = self._define_test_workflows()
        
    async def initialize(self):
        """Initialize the test session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        logger.info("Orchestration test suite initialized")
    
    async def cleanup(self):
        """Cleanup test session"""
        if self.session:
            await self.session.close()
    
    def _define_test_workflows(self) -> Dict[str, Dict]:
        """Define test workflows for validation"""
        return {
            "simple_sequential": {
                "name": "Simple Sequential Workflow",
                "description": "Basic sequential task execution",
                "nodes": [
                    {
                        "id": "task1",
                        "name": "Data Preparation",
                        "type": "task",
                        "agent_type": "senior_ai_engineer",
                        "task": {
                            "type": "data_analysis",
                            "description": "Prepare test data",
                            "input": {"data": "sample_data"}
                        },
                        "dependencies": []
                    },
                    {
                        "id": "task2",
                        "name": "Analysis",
                        "type": "task",
                        "agent_type": "testing_qa_validator",
                        "task": {
                            "type": "analysis",
                            "description": "Analyze prepared data",
                            "input": {"data": "processed_data"}
                        },
                        "dependencies": ["task1"]
                    },
                    {
                        "id": "task3",
                        "name": "Report Generation",
                        "type": "task",
                        "agent_type": "infrastructure_devops_manager",
                        "task": {
                            "type": "report",
                            "description": "Generate analysis report",
                            "input": {"analysis": "results"}
                        },
                        "dependencies": ["task2"]
                    }
                ]
            },
            "parallel_execution": {
                "name": "Parallel Execution Workflow",
                "description": "Multiple parallel tasks with convergence",
                "nodes": [
                    {
                        "id": "init",
                        "name": "Initialize",
                        "type": "task",
                        "agent_type": "ai_agent_orchestrator",
                        "task": {
                            "type": "initialization",
                            "description": "Initialize parallel workflow"
                        },
                        "dependencies": []
                    },
                    {
                        "id": "parallel1",
                        "name": "Parallel Task 1",
                        "type": "task",
                        "agent_type": "senior_ai_engineer",
                        "task": {
                            "type": "ml_analysis",
                            "description": "Machine learning analysis"
                        },
                        "dependencies": ["init"]
                    },
                    {
                        "id": "parallel2",
                        "name": "Parallel Task 2",
                        "type": "task",
                        "agent_type": "security_pentesting_specialist",
                        "task": {
                            "type": "security_scan",
                            "description": "Security analysis"
                        },
                        "dependencies": ["init"]
                    },
                    {
                        "id": "parallel3",
                        "name": "Parallel Task 3",
                        "type": "task",
                        "agent_type": "infrastructure_devops_manager",
                        "task": {
                            "type": "deployment_check",
                            "description": "Infrastructure validation"
                        },
                        "dependencies": ["init"]
                    },
                    {
                        "id": "convergence",
                        "name": "Merge Results",
                        "type": "task",
                        "agent_type": "ai_agent_orchestrator",
                        "task": {
                            "type": "merge",
                            "description": "Merge parallel results"
                        },
                        "dependencies": ["parallel1", "parallel2", "parallel3"]
                    }
                ],
                "parallel_limit": 3
            },
            "complex_conditional": {
                "name": "Complex Conditional Workflow",
                "description": "Workflow with conditional branching",
                "nodes": [
                    {
                        "id": "assessment",
                        "name": "Initial Assessment",
                        "type": "task",
                        "agent_type": "ai_product_manager",
                        "task": {
                            "type": "assessment",
                            "description": "Assess requirements"
                        },
                        "dependencies": []
                    },
                    {
                        "id": "condition1",
                        "name": "Security Check Required",
                        "type": "condition",
                        "agent_type": "ai_agent_orchestrator",
                        "task": {
                            "condition": "assessment_result.security_required == true"
                        },
                        "dependencies": ["assessment"]
                    },
                    {
                        "id": "security_scan",
                        "name": "Security Scan",
                        "type": "task",
                        "agent_type": "security_pentesting_specialist",
                        "task": {
                            "type": "security_analysis",
                            "description": "Perform security scan"
                        },
                        "dependencies": ["condition1"]
                    },
                    {
                        "id": "development",
                        "name": "Development Task",
                        "type": "task",
                        "agent_type": "senior_backend_developer",
                        "task": {
                            "type": "development",
                            "description": "Development work"
                        },
                        "dependencies": ["assessment"]
                    },
                    {
                        "id": "final_validation",
                        "name": "Final Validation",
                        "type": "task",
                        "agent_type": "testing_qa_validator",
                        "task": {
                            "type": "validation",
                            "description": "Final validation"
                        },
                        "dependencies": ["development", "security_scan"]
                    }
                ]
            },
            "resource_intensive": {
                "name": "Resource Intensive Workflow",
                "description": "Workflow testing resource allocation",
                "nodes": [
                    {
                        "id": "resource_task1",
                        "name": "High CPU Task",
                        "type": "task",
                        "agent_type": "senior_ai_engineer",
                        "task": {
                            "type": "ml_training",
                            "description": "Machine learning training",
                            "resource_requirements": {
                                "cpu": 80,
                                "memory": 60
                            }
                        },
                        "dependencies": []
                    },
                    {
                        "id": "resource_task2",
                        "name": "Memory Intensive Task",
                        "type": "task",
                        "agent_type": "document_knowledge_manager",
                        "task": {
                            "type": "document_processing",
                            "description": "Large document processing",
                            "resource_requirements": {
                                "memory": 70,
                                "storage": 50
                            }
                        },
                        "dependencies": []
                    }
                ]
            }
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all orchestration tests"""
        logger.info("Starting comprehensive orchestration test suite")
        start_time = time.time()
        
        try:
            # Test 1: Agent Discovery
            await self.test_agent_discovery()
            
            # Test 2: Task Routing
            await self.test_task_routing()
            
            # Test 3: Workflow Execution
            await self.test_workflow_execution()
            
            # Test 4: Load Balancing
            await self.test_load_balancing()
            
            # Test 5: Message Passing
            await self.test_message_passing()
            
            # Test 6: Consensus Mechanisms
            await self.test_consensus()
            
            # Test 7: Resource Allocation
            await self.test_resource_allocation()
            
            # Test 8: Performance Testing
            await self.test_performance()
            
            # Test 9: Fault Tolerance
            await self.test_fault_tolerance()
            
            # Test 10: Monitoring
            await self.test_monitoring()
            
            total_time = time.time() - start_time
            
            # Generate summary report
            summary = self._generate_test_summary(total_time)
            
            logger.info(f"All tests completed in {total_time:.2f} seconds")
            return summary
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_agent_discovery(self):
        """Test agent discovery functionality"""
        logger.info("Testing agent discovery...")
        
        try:
            # Test 1: List all agents
            async with self.session.get(f"{self.orchestration_url}/agents") as response:
                if response.status == 200:
                    data = await response.json()
                    agent_count = data.get("total_count", 0)
                    healthy_count = data.get("healthy_count", 0)
                    
                    self.test_results["agent_discovery"]["details"]["total_agents"] = agent_count
                    self.test_results["agent_discovery"]["details"]["healthy_agents"] = healthy_count
                    
                    if agent_count > 0:
                        self.test_results["agent_discovery"]["status"] = "passed"
                        logger.info(f"Agent discovery passed: {agent_count} agents found, {healthy_count} healthy")
                    else:
                        self.test_results["agent_discovery"]["status"] = "failed"
                        self.test_results["agent_discovery"]["details"]["error"] = "No agents discovered"
                else:
                    self.test_results["agent_discovery"]["status"] = "failed"
                    self.test_results["agent_discovery"]["details"]["error"] = f"HTTP {response.status}"
            
            # Test 2: Trigger discovery
            async with self.session.post(f"{self.orchestration_url}/agents/discover") as response:
                if response.status == 200:
                    data = await response.json()
                    self.test_results["agent_discovery"]["details"]["discovery_triggered"] = True
                else:
                    self.test_results["agent_discovery"]["details"]["discovery_triggered"] = False
            
            # Test 3: Get healthy agents
            async with self.session.get(f"{self.orchestration_url}/agents/healthy") as response:
                if response.status == 200:
                    data = await response.json()
                    healthy_agents = data.get("count", 0)
                    self.test_results["agent_discovery"]["details"]["healthy_agents_endpoint"] = healthy_agents
                
        except Exception as e:
            self.test_results["agent_discovery"]["status"] = "error"
            self.test_results["agent_discovery"]["details"]["error"] = str(e)
            logger.error(f"Agent discovery test failed: {e}")
    
    async def test_task_routing(self):
        """Test task routing functionality"""
        logger.info("Testing task routing...")
        
        try:
            # Test 1: Submit simple task
            task_data = {
                "type": "test_task",
                "description": "Test task for routing validation",
                "priority": "normal",
                "capabilities_required": ["general_task"],
                "resource_requirements": {"cpu": 10, "memory": 10}
            }
            
            async with self.session.post(
                f"{self.orchestration_url}/tasks/submit",
                json=task_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    task_id = data.get("task_id")
                    self.test_results["task_routing"]["details"]["task_submitted"] = True
                    self.test_results["task_routing"]["details"]["task_id"] = task_id
                    
                    # Test 2: Check queue status
                    async with self.session.get(f"{self.orchestration_url}/tasks/queue/status") as queue_response:
                        if queue_response.status == 200:
                            queue_data = await queue_response.json()
                            self.test_results["task_routing"]["details"]["queue_size"] = queue_data.get("queue_size", 0)
                            self.test_results["task_routing"]["status"] = "passed"
                        else:
                            self.test_results["task_routing"]["status"] = "partial"
                else:
                    self.test_results["task_routing"]["status"] = "failed"
                    self.test_results["task_routing"]["details"]["error"] = f"HTTP {response.status}"
            
            # Test 3: Submit high priority task
            priority_task_data = {
                "type": "priority_test",
                "description": "High priority task test",
                "priority": "critical",
                "capabilities_required": ["general_task"]
            }
            
            async with self.session.post(
                f"{self.orchestration_url}/tasks/submit",
                json=priority_task_data
            ) as response:
                if response.status == 200:
                    self.test_results["task_routing"]["details"]["priority_task_submitted"] = True
                    
        except Exception as e:
            self.test_results["task_routing"]["status"] = "error"
            self.test_results["task_routing"]["details"]["error"] = str(e)
            logger.error(f"Task routing test failed: {e}")
    
    async def test_workflow_execution(self):
        """Test workflow execution"""
        logger.info("Testing workflow execution...")
        
        try:
            results = {}
            
            # Test each workflow type
            for workflow_name, workflow_def in self.test_workflows.items():
                logger.info(f"Testing workflow: {workflow_name}")
                
                start_time = time.time()
                
                async with self.session.post(
                    f"{self.orchestration_url}/workflows/create",
                    json=workflow_def
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        workflow_id = data.get("workflow_id")
                        execution_id = data.get("execution_id")
                        
                        # Monitor workflow execution
                        workflow_result = await self._monitor_workflow(workflow_id)
                        execution_time = time.time() - start_time
                        
                        results[workflow_name] = {
                            "workflow_id": workflow_id,
                            "execution_id": execution_id,
                            "status": workflow_result.get("status", "unknown"),
                            "execution_time": execution_time,
                            "progress": workflow_result.get("progress", 0.0)
                        }
                    else:
                        results[workflow_name] = {
                            "status": "failed",
                            "error": f"HTTP {response.status}"
                        }
            
            # Determine overall status
            successful_workflows = sum(1 for r in results.values() if r.get("status") == "completed")
            total_workflows = len(results)
            
            if successful_workflows == total_workflows:
                self.test_results["workflow_execution"]["status"] = "passed"
            elif successful_workflows > 0:
                self.test_results["workflow_execution"]["status"] = "partial"
            else:
                self.test_results["workflow_execution"]["status"] = "failed"
            
            self.test_results["workflow_execution"]["details"] = {
                "workflows_tested": total_workflows,
                "successful_workflows": successful_workflows,
                "results": results
            }
            
        except Exception as e:
            self.test_results["workflow_execution"]["status"] = "error"
            self.test_results["workflow_execution"]["details"]["error"] = str(e)
            logger.error(f"Workflow execution test failed: {e}")
    
    async def _monitor_workflow(self, workflow_id: str, timeout: int = 60) -> Dict[str, Any]:
        """Monitor workflow execution until completion or timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with self.session.get(
                    f"{self.orchestration_url}/workflows/{workflow_id}/status"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status")
                        
                        if status in ["completed", "failed", "cancelled"]:
                            return data
                    
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring workflow {workflow_id}: {e}")
                break
        
        return {"status": "timeout", "progress": 0.0}
    
    async def test_load_balancing(self):
        """Test load balancing functionality"""
        logger.info("Testing load balancing...")
        
        try:
            # Test 1: Get available algorithms
            async with self.session.get(f"{self.orchestration_url}/load-balancing/algorithms") as response:
                if response.status == 200:
                    data = await response.json()
                    algorithms = data.get("algorithms", [])
                    self.test_results["load_balancing"]["details"]["available_algorithms"] = len(algorithms)
                    
                    # Test 2: Configure different algorithms
                    test_results = {}
                    for algorithm in algorithms[:3]:  # Test first 3 algorithms
                        config_data = {
                            "algorithm": algorithm["name"],
                            "parameters": {}
                        }
                        
                        async with self.session.post(
                            f"{self.orchestration_url}/load-balancing/configure",
                            json=config_data
                        ) as config_response:
                            test_results[algorithm["name"]] = config_response.status == 200
                    
                    successful_configs = sum(test_results.values())
                    if successful_configs > 0:
                        self.test_results["load_balancing"]["status"] = "passed"
                    else:
                        self.test_results["load_balancing"]["status"] = "failed"
                    
                    self.test_results["load_balancing"]["details"]["algorithm_tests"] = test_results
                else:
                    self.test_results["load_balancing"]["status"] = "failed"
                    
        except Exception as e:
            self.test_results["load_balancing"]["status"] = "error"
            self.test_results["load_balancing"]["details"]["error"] = str(e)
            logger.error(f"Load balancing test failed: {e}")
    
    async def test_message_passing(self):
        """Test message passing functionality"""
        logger.info("Testing message passing...")
        
        try:
            # Test 1: Send direct message
            message_data = {
                "type": "direct_message",
                "recipient_id": "test_agent",
                "content": {"message": "Test message", "timestamp": datetime.now().isoformat()},
                "priority": 1,
                "requires_response": False
            }
            
            async with self.session.post(
                f"{self.orchestration_url}/messages/send",
                json=message_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    message_id = data.get("message_id")
                    self.test_results["message_passing"]["details"]["direct_message_sent"] = True
                    self.test_results["message_passing"]["details"]["message_id"] = message_id
                else:
                    self.test_results["message_passing"]["details"]["direct_message_sent"] = False
            
            # Test 2: Send broadcast message
            broadcast_data = {
                "message": "Test broadcast message",
                "data": {"type": "test", "timestamp": datetime.now().isoformat()}
            }
            
            async with self.session.post(
                f"{self.orchestration_url}/messages/broadcast",
                json=broadcast_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.test_results["message_passing"]["details"]["broadcast_sent"] = True
                    self.test_results["message_passing"]["details"]["broadcast_id"] = data.get("message_id")
                else:
                    self.test_results["message_passing"]["details"]["broadcast_sent"] = False
            
            # Determine overall status
            if (self.test_results["message_passing"]["details"].get("direct_message_sent", False) and
                self.test_results["message_passing"]["details"].get("broadcast_sent", False)):
                self.test_results["message_passing"]["status"] = "passed"
            elif (self.test_results["message_passing"]["details"].get("direct_message_sent", False) or
                  self.test_results["message_passing"]["details"].get("broadcast_sent", False)):
                self.test_results["message_passing"]["status"] = "partial"
            else:
                self.test_results["message_passing"]["status"] = "failed"
                
        except Exception as e:
            self.test_results["message_passing"]["status"] = "error"
            self.test_results["message_passing"]["details"]["error"] = str(e)
            logger.error(f"Message passing test failed: {e}")
    
    async def test_consensus(self):
        """Test consensus mechanisms"""
        logger.info("Testing consensus mechanisms...")
        
        try:
            # Test consensus request
            consensus_data = {
                "topic": "test_decision",
                "data": {"proposal": "Test consensus proposal"},
                "agents": ["agent1", "agent2", "agent3"],
                "threshold": 0.6
            }
            
            async with self.session.post(
                f"{self.orchestration_url}/coordination/consensus",
                json=consensus_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    consensus_id = data.get("consensus_id")
                    self.test_results["consensus"]["details"]["consensus_initiated"] = True
                    self.test_results["consensus"]["details"]["consensus_id"] = consensus_id
                    self.test_results["consensus"]["details"]["agents_contacted"] = data.get("agents_contacted", [])
                    self.test_results["consensus"]["status"] = "passed"
                else:
                    self.test_results["consensus"]["status"] = "failed"
                    self.test_results["consensus"]["details"]["error"] = f"HTTP {response.status}"
                    
        except Exception as e:
            self.test_results["consensus"]["status"] = "error"
            self.test_results["consensus"]["details"]["error"] = str(e)
            logger.error(f"Consensus test failed: {e}")
    
    async def test_resource_allocation(self):
        """Test resource allocation"""
        logger.info("Testing resource allocation...")
        
        try:
            # This test would typically interact with resource allocation endpoints
            # For now, we'll test by submitting resource-intensive workflows
            
            workflow = self.test_workflows["resource_intensive"]
            
            async with self.session.post(
                f"{self.orchestration_url}/workflows/create",
                json=workflow
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    workflow_id = data.get("workflow_id")
                    
                    # Monitor for resource allocation
                    result = await self._monitor_workflow(workflow_id, timeout=30)
                    
                    if result.get("status") == "completed":
                        self.test_results["resource_allocation"]["status"] = "passed"
                    else:
                        self.test_results["resource_allocation"]["status"] = "partial"
                    
                    self.test_results["resource_allocation"]["details"] = {
                        "workflow_id": workflow_id,
                        "result": result
                    }
                else:
                    self.test_results["resource_allocation"]["status"] = "failed"
                    
        except Exception as e:
            self.test_results["resource_allocation"]["status"] = "error"
            self.test_results["resource_allocation"]["details"]["error"] = str(e)
            logger.error(f"Resource allocation test failed: {e}")
    
    async def test_performance(self):
        """Test system performance under load"""
        logger.info("Testing performance under load...")
        
        try:
            # Submit multiple tasks concurrently
            start_time = time.time()
            task_count = 10
            
            tasks = []
            for i in range(task_count):
                task_data = {
                    "type": f"perf_test_{i}",
                    "description": f"Performance test task {i}",
                    "priority": "normal",
                    "capabilities_required": ["general_task"]
                }
                
                task = self.session.post(
                    f"{self.orchestration_url}/tasks/submit",
                    json=task_data
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_submissions = 0
            for response in responses:
                if not isinstance(response, Exception) and response.status == 200:
                    successful_submissions += 1
            
            total_time = time.time() - start_time
            throughput = successful_submissions / total_time if total_time > 0 else 0
            
            self.test_results["performance"]["details"] = {
                "tasks_submitted": task_count,
                "successful_submissions": successful_submissions,
                "total_time": total_time,
                "throughput": throughput,
                "success_rate": successful_submissions / task_count
            }
            
            if successful_submissions >= task_count * 0.8:  # 80% success rate
                self.test_results["performance"]["status"] = "passed"
            else:
                self.test_results["performance"]["status"] = "failed"
                
        except Exception as e:
            self.test_results["performance"]["status"] = "error"
            self.test_results["performance"]["details"]["error"] = str(e)
            logger.error(f"Performance test failed: {e}")
    
    async def test_fault_tolerance(self):
        """Test system fault tolerance"""
        logger.info("Testing fault tolerance...")
        
        try:
            # Test 1: Submit task with invalid agent type
            invalid_task = {
                "type": "fault_test",
                "description": "Test with invalid agent",
                "priority": "normal",
                "capabilities_required": ["nonexistent_capability"]
            }
            
            async with self.session.post(
                f"{self.orchestration_url}/tasks/submit",
                json=invalid_task
            ) as response:
                # System should handle this gracefully
                invalid_task_handled = response.status in [200, 400]  # Either accept or reject gracefully
            
            # Test 2: Submit workflow with circular dependencies (should be rejected)
            circular_workflow = {
                "name": "Circular Dependency Test",
                "description": "Test circular dependency handling",
                "nodes": [
                    {
                        "id": "task_a",
                        "name": "Task A",
                        "type": "task",
                        "agent_type": "general",
                        "task": {"type": "test"},
                        "dependencies": ["task_b"]
                    },
                    {
                        "id": "task_b",
                        "name": "Task B",
                        "type": "task",
                        "agent_type": "general",
                        "task": {"type": "test"},
                        "dependencies": ["task_a"]
                    }
                ]
            }
            
            async with self.session.post(
                f"{self.orchestration_url}/workflows/create",
                json=circular_workflow
            ) as response:
                # Should reject circular dependencies
                circular_dependency_handled = response.status != 200
            
            self.test_results["fault_tolerance"]["details"] = {
                "invalid_task_handled": invalid_task_handled,
                "circular_dependency_handled": circular_dependency_handled
            }
            
            if invalid_task_handled and circular_dependency_handled:
                self.test_results["fault_tolerance"]["status"] = "passed"
            else:
                self.test_results["fault_tolerance"]["status"] = "failed"
                
        except Exception as e:
            self.test_results["fault_tolerance"]["status"] = "error"
            self.test_results["fault_tolerance"]["details"]["error"] = str(e)
            logger.error(f"Fault tolerance test failed: {e}")
    
    async def test_monitoring(self):
        """Test monitoring and metrics"""
        logger.info("Testing monitoring and metrics...")
        
        try:
            # Test 1: Get system status
            async with self.session.get(f"{self.orchestration_url}/system/status") as response:
                if response.status == 200:
                    data = await response.json()
                    self.test_results["monitoring"]["details"]["system_status"] = data.get("status")
                    self.test_results["monitoring"]["details"]["components"] = data.get("components", {})
                else:
                    self.test_results["monitoring"]["details"]["system_status"] = "unavailable"
            
            # Test 2: Get metrics
            async with self.session.get(f"{self.orchestration_url}/system/metrics") as response:
                if response.status == 200:
                    data = await response.json()
                    self.test_results["monitoring"]["details"]["metrics_available"] = True
                    self.test_results["monitoring"]["details"]["metric_categories"] = list(data.keys())
                else:
                    self.test_results["monitoring"]["details"]["metrics_available"] = False
            
            # Test 3: Health check
            async with self.session.get(f"{self.orchestration_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.test_results["monitoring"]["details"]["health_status"] = data.get("status")
                    self.test_results["monitoring"]["details"]["component_health"] = data.get("components", {})
                else:
                    self.test_results["monitoring"]["details"]["health_status"] = "unavailable"
            
            # Determine overall monitoring status
            if (self.test_results["monitoring"]["details"].get("system_status") == "running" and
                self.test_results["monitoring"]["details"].get("metrics_available", False) and
                self.test_results["monitoring"]["details"].get("health_status") == "healthy"):
                self.test_results["monitoring"]["status"] = "passed"
            else:
                self.test_results["monitoring"]["status"] = "partial"
                
        except Exception as e:
            self.test_results["monitoring"]["status"] = "error"
            self.test_results["monitoring"]["details"]["error"] = str(e)
            logger.error(f"Monitoring test failed: {e}")
    
    def _generate_test_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "passed")
        partial_tests = sum(1 for result in self.test_results.values() if result["status"] == "partial")
        failed_tests = sum(1 for result in self.test_results.values() if result["status"] == "failed")
        error_tests = sum(1 for result in self.test_results.values() if result["status"] == "error")
        total_tests = len(self.test_results)
        
        success_rate = (passed_tests + partial_tests * 0.5) / total_tests if total_tests > 0 else 0
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "partial": partial_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": success_rate
            },
            "overall_status": "passed" if success_rate >= 0.8 else "failed" if success_rate < 0.5 else "partial",
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        for test_name, result in self.test_results.items():
            if result["status"] == "failed":
                recommendations.append(f"Fix {test_name} functionality - critical for orchestration")
            elif result["status"] == "error":
                recommendations.append(f"Investigate {test_name} errors - system stability issue")
            elif result["status"] == "partial":
                recommendations.append(f"Improve {test_name} reliability - partial functionality detected")
        
        if not recommendations:
            recommendations.append("All tests passed - orchestration system is functioning well")
        
        return recommendations

async def main():
    """Main test execution function"""
    logger.info("Starting SutazAI Multi-Agent Orchestration Test Suite")
    
    tester = OrchestrationTester()
    
    try:
        await tester.initialize()
        
        # Run all tests
        results = await tester.run_all_tests()
        
        # Print summary
        print("\n" + "="*80)
        print("SUTAZAI MULTI-AGENT ORCHESTRATION TEST RESULTS")
        print("="*80)
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Success Rate: {results['test_summary']['success_rate']:.1%}")
        print(f"Total Execution Time: {results['total_execution_time']:.2f} seconds")
        print(f"Tests Passed: {results['test_summary']['passed']}/{results['test_summary']['total_tests']}")
        print(f"Tests Failed: {results['test_summary']['failed']}/{results['test_summary']['total_tests']}")
        print(f"Tests with Errors: {results['test_summary']['errors']}/{results['test_summary']['total_tests']}")
        
        print("\nDETAILED RESULTS:")
        print("-"*40)
        for test_name, result in results["detailed_results"].items():
            status_symbol = "✓" if result["status"] == "passed" else "⚠" if result["status"] == "partial" else "✗"
            print(f"{status_symbol} {test_name.replace('_', ' ').title()}: {result['status'].upper()}")
        
        print("\nRECOMMENDATIONS:")
        print("-"*40)
        for i, recommendation in enumerate(results["recommendations"], 1):
            print(f"{i}. {recommendation}")
        
        # Save detailed results to file
        with open(f"/opt/sutazaiapp/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\nTest execution failed: {e}")
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())