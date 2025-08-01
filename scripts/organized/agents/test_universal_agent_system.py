#!/usr/bin/env python3
"""
Universal Agent System Test Suite
=================================

This script tests all components of the Universal Agent System to ensure
proper functionality and integration.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/opt/sutazaiapp')

from backend.ai_agents.core import (
    UniversalAgentSystem,
    AgentCapability,
    create_agent,
    send_message,
    get_agent_registry,
    get_orchestration_controller
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("universal_agent_test")


class UniversalAgentSystemTests:
    """Comprehensive test suite for the Universal Agent System"""
    
    def __init__(self):
        self.system: UniversalAgentSystem = None
        self.test_results = []
        self.start_time = datetime.utcnow()
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {details}")
    
    async def test_system_initialization(self) -> bool:
        """Test system initialization"""
        try:
            logger.info("Testing system initialization...")
            
            self.system = UniversalAgentSystem(
                redis_url="redis://localhost:6380",  # Use test Redis port
                ollama_url="http://localhost:11435",  # Use test Ollama port
                namespace="test_sutazai"
            )
            
            success = await self.system.initialize()
            
            self.log_test_result(
                "System Initialization",
                success,
                "System components initialized successfully" if success else "System initialization failed"
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("System Initialization", False, f"Exception: {str(e)}")
            return False
    
    async def test_agent_creation(self) -> bool:
        """Test creating agents of different types"""
        success_count = 0
        total_tests = 0
        
        agent_types = ["orchestrator", "code_generator", "generic"]
        
        for agent_type in agent_types:
            try:
                total_tests += 1
                agent_id = f"test-{agent_type}-{int(time.time())}"
                
                agent = await self.system.create_agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    config_overrides={
                        "model_config": {
                            "model": "llama2",
                            "temperature": 0.5
                        }
                    }
                )
                
                if agent:
                    success_count += 1
                    self.log_test_result(
                        f"Create {agent_type} Agent",
                        True,
                        f"Agent {agent_id} created successfully"
                    )
                else:
                    self.log_test_result(
                        f"Create {agent_type} Agent",
                        False,
                        f"Failed to create agent {agent_id}"
                    )
                    
            except Exception as e:
                self.log_test_result(
                    f"Create {agent_type} Agent",
                    False,
                    f"Exception creating {agent_type}: {str(e)}"
                )
        
        overall_success = success_count == total_tests
        self.log_test_result(
            "Agent Creation Suite",
            overall_success,
            f"{success_count}/{total_tests} agent types created successfully"
        )
        
        return overall_success
    
    async def test_agent_registry(self) -> bool:
        """Test agent registry functionality"""
        try:
            registry = self.system.registry
            
            # Test registry stats
            stats = registry.get_registry_stats()
            has_agents = stats.get("active_agents", 0) > 0
            
            # Test agent discovery
            all_agents = registry.get_all_agents()
            
            # Test capability-based search
            code_agents = registry.find_agents_by_capability([AgentCapability.CODE_GENERATION])
            
            success = has_agents and len(all_agents) > 0
            
            self.log_test_result(
                "Agent Registry",
                success,
                f"Registry has {len(all_agents)} agents, {len(code_agents)} with code generation capability"
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("Agent Registry", False, f"Exception: {str(e)}")
            return False
    
    async def test_message_bus(self) -> bool:
        """Test message bus functionality"""
        try:
            message_bus = self.system.message_bus
            
            # Test message sending
            message_id = await send_message(
                sender_id="test_sender",
                receiver_id="test_receiver",
                message_type="test_message",
                content={"test": "data", "timestamp": datetime.utcnow().isoformat()}
            )
            
            # Test broadcast
            broadcast_id = await message_bus.broadcast_system_message(
                message_type="test_broadcast",
                content={"message": "Test broadcast message"}
            )
            
            # Get message bus stats
            stats = message_bus.get_stats()
            
            success = bool(message_id and broadcast_id)
            
            self.log_test_result(
                "Message Bus",
                success,
                f"Sent {stats.total_sent} messages, received {stats.total_received}"
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("Message Bus", False, f"Exception: {str(e)}")
            return False
    
    async def test_workflow_orchestration(self) -> bool:
        """Test workflow orchestration"""
        try:
            controller = self.system.orchestrator
            
            # Create a simple test workflow
            workflow_spec = {
                "name": "Test Workflow",
                "description": "Simple test workflow",
                "tasks": [
                    {
                        "id": "test-task-1",
                        "name": "analyze_test_data",
                        "description": "Analyze some test data",
                        "task_type": "analyze",
                        "priority": 3,
                        "required_capabilities": ["reasoning"],
                        "input_data": {
                            "content": "This is test data for analysis",
                            "analysis_type": "general"
                        },
                        "dependencies": [],
                        "max_retries": 2,
                        "timeout_seconds": 300
                    }
                ]
            }
            
            # Create workflow
            workflow_id = await controller.create_workflow(workflow_spec)
            
            # Start workflow
            started = await controller.start_workflow(workflow_id)
            
            # Wait a moment for processing
            await asyncio.sleep(2)
            
            # Check workflow status
            workflow_status = controller.get_workflow_status(workflow_id)
            
            success = bool(workflow_id and started and workflow_status)
            
            self.log_test_result(
                "Workflow Orchestration",
                success,
                f"Workflow {workflow_id} created and started, status: {workflow_status.get('status') if workflow_status else 'unknown'}"
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("Workflow Orchestration", False, f"Exception: {str(e)}")
            return False
    
    async def test_agent_task_execution(self) -> bool:
        """Test agent task execution"""
        try:
            # Get a generic agent for testing
            registry = self.system.registry
            agents = registry.find_agents_by_capability([AgentCapability.REASONING])
            
            if not agents:
                self.log_test_result("Agent Task Execution", False, "No agents with reasoning capability found")
                return False
            
            # Use the first available agent
            test_agent_registration = agents[0]
            test_agent = self.system.factory.get_agent(test_agent_registration.agent_id)
            
            if not test_agent:
                self.log_test_result("Agent Task Execution", False, "Could not retrieve agent instance")
                return False
            
            # Execute a simple task
            task_result = await test_agent.execute_task(
                task_id="test-execution-task",
                task_data={
                    "task_type": "general",
                    "description": "Analyze the benefits of AI automation",
                    "input": "AI automation in business processes",
                    "context": "Business process optimization"
                }
            )
            
            success = task_result.get("success", False)
            
            self.log_test_result(
                "Agent Task Execution",
                success,
                f"Task executed by {test_agent.agent_id}: {task_result.get('result', {}).get('output', 'No output')[:100]}..."
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("Agent Task Execution", False, f"Exception: {str(e)}")
            return False
    
    async def test_system_health(self) -> bool:
        """Test overall system health"""
        try:
            system_status = self.system.get_system_status()
            
            components_healthy = (
                system_status.get("status") == "running" and
                system_status.get("registry_stats", {}).get("active_agents", 0) > 0 and
                system_status.get("factory_stats", {}).get("currently_active", 0) > 0
            )
            
            self.log_test_result(
                "System Health",
                components_healthy,
                f"System status: {system_status.get('status')}, "
                f"Active agents: {system_status.get('registry_stats', {}).get('active_agents', 0)}"
            )
            
            return components_healthy
            
        except Exception as e:
            self.log_test_result("System Health", False, f"Exception: {str(e)}")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all tests and generate report"""
        logger.info("🚀 Starting Universal Agent System Test Suite")
        
        test_functions = [
            self.test_system_initialization,
            self.test_agent_creation,
            self.test_agent_registry,
            self.test_message_bus,
            self.test_workflow_orchestration,
            self.test_agent_task_execution,
            self.test_system_health
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_func in test_functions:
            try:
                result = await test_func()
                if result:
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test {test_func.__name__} failed with exception: {e}")
        
        # Generate test report
        await self.generate_test_report(passed_tests, total_tests)
        
        success_rate = (passed_tests / total_tests) * 100
        overall_success = success_rate >= 80  # 80% pass rate required
        
        logger.info(f"🎯 Test Suite Complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if overall_success:
            logger.info("🎉 Universal Agent System is functioning correctly!")
        else:
            logger.error("❌ Universal Agent System has issues that need to be addressed")
        
        return overall_success
    
    async def generate_test_report(self, passed: int, total: int):
        """Generate detailed test report"""
        report = {
            "test_suite": "Universal Agent System",
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "summary": {
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "success_rate": (passed / total) * 100
            },
            "test_results": self.test_results
        }
        
        # Save report to file
        report_path = Path("/opt/sutazaiapp/logs/universal_agent_test_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved to {report_path}")
    
    async def cleanup(self):
        """Clean up test resources"""
        try:
            if self.system:
                logger.info("🧹 Cleaning up test resources...")
                await self.system.shutdown()
                logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


async def main():
    """Main test execution function"""
    test_suite = UniversalAgentSystemTests()
    
    try:
        # Run all tests
        success = await test_suite.run_all_tests()
        
        # Return appropriate exit code
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Test suite failed with exception: {e}")
        return 1
        
    finally:
        # Always attempt cleanup
        await test_suite.cleanup()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error in test suite: {e}")
        sys.exit(1)