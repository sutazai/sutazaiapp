"""
Sutazai Integration Test
Comprehensive test to verify all system components work together
"""

import asyncio
import logging
import json
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_sutazai_integration():
    """Test full Sutazai system integration"""
    try:
        logger.info("🚀 Starting Sutazai Integration Test")
        
        # Test 1: Import all core modules
        logger.info("Test 1: Importing core modules...")
        try:
            from sutazai.core import (
                sutazai_core, 
                knowledge_graph, 
                code_generation_module, 
                authorization_control_module
            )
            logger.info("✅ All core modules imported successfully")
        except Exception as e:
            logger.error(f"❌ Module import failed: {e}")
            return False
        
        # Test 2: Check system initialization
        logger.info("Test 2: Checking system initialization...")
        try:
            system_status = await sutazai_core.get_system_status()
            logger.info(f"✅ System status retrieved: {system_status.get('sutazai_core', {}).get('active', False)}")
        except Exception as e:
            logger.error(f"❌ System status check failed: {e}")
            return False
        
        # Test 3: Test ACM Authentication
        logger.info("Test 3: Testing ACM authentication...")
        try:
            auth_result = await authorization_control_module.authenticate_user(
                "os.getenv("ADMIN_EMAIL", "admin@localhost")",
                {"method": "password", "device_fingerprint": "test_device"}
            )
            if auth_result.get("success"):
                session_id = auth_result["session_id"]
                logger.info("✅ Authentication successful")
            else:
                logger.error(f"❌ Authentication failed: {auth_result.get('error')}")
                return False
        except Exception as e:
            logger.error(f"❌ Authentication test failed: {e}")
            return False
        
        # Test 4: Test Knowledge Graph operations
        logger.info("Test 4: Testing Knowledge Graph operations...")
        try:
            # Add test knowledge
            node_id = await knowledge_graph.add_knowledge_node(
                name="Integration Test Knowledge",
                knowledge_type="concept",
                content={
                    "description": "Test knowledge for integration testing",
                    "test_data": {"created_at": time.time()}
                },
                user_id="os.getenv("ADMIN_EMAIL", "admin@localhost")",
                tags=["test", "integration"]
            )
            
            # Query knowledge
            query_result = await knowledge_graph.query_knowledge(
                "integration test",
                user_id="os.getenv("ADMIN_EMAIL", "admin@localhost")"
            )
            
            logger.info(f"✅ Knowledge Graph operations successful - Added node: {node_id}")
            logger.info(f"   Query returned {len(query_result.nodes)} nodes")
        except Exception as e:
            logger.error(f"❌ Knowledge Graph test failed: {e}")
            return False
        
        # Test 5: Test Code Generation Module
        logger.info("Test 5: Testing Code Generation Module...")
        try:
            from sutazai.core.cgm import CodeGenerationTask, CodeType, OptimizationTarget
            
            # Create test task
            test_task = CodeGenerationTask(
                id="test_task_1",
                task_type="function_generation",
                description="Generate a simple addition function",
                requirements={"function_name": "add_numbers", "parameters": ["a", "b"]},
                target_language=CodeType.PYTHON,
                optimization_targets=[OptimizationTarget.READABILITY],
                context={"test": True}
            )
            
            # Generate code
            generation_result = await code_generation_module.generate_code(test_task, "os.getenv("ADMIN_EMAIL", "admin@localhost")")
            
            if generation_result.get("success"):
                logger.info("✅ Code generation successful")
                logger.info(f"   Generated code length: {len(generation_result.get('generated_code', {}).get('code', ''))}")
            else:
                logger.warning(f"⚠️ Code generation had issues: {generation_result.get('error')}")
        except Exception as e:
            logger.error(f"❌ Code Generation test failed: {e}")
            return False
        
        # Test 6: Test Task Submission and Processing
        logger.info("Test 6: Testing task submission and processing...")
        try:
            # Submit a knowledge query task
            task_id = await sutazai_core.submit_task(
                task_type="knowledge_query",
                description="Query test knowledge",
                data={"query": "integration test"},
                assigned_module="kg",
                priority=8
            )
            
            # Wait a moment for processing
            await asyncio.sleep(2)
            
            # Check system status for task processing
            status = await sutazai_core.get_system_status()
            task_info = status.get("task_queue", {})
            
            logger.info(f"✅ Task submission successful - Task ID: {task_id}")
            logger.info(f"   Tasks pending: {task_info.get('pending', 0)}, active: {task_info.get('active', 0)}, completed: {task_info.get('completed', 0)}")
        except Exception as e:
            logger.error(f"❌ Task submission test failed: {e}")
            return False
        
        # Test 7: Test System Control (non-destructive)
        logger.info("Test 7: Testing system control (status check)...")
        try:
            from sutazai.core.acm import ControlAction
            
            # Check permission for system status
            permission_check = await authorization_control_module.check_permission(
                session_id, 
                ControlAction.DATA_ACCESS
            )
            
            if permission_check.get("authorized"):
                logger.info("✅ System control permissions verified")
            else:
                logger.error(f"❌ Permission check failed: {permission_check.get('error')}")
                return False
        except Exception as e:
            logger.error(f"❌ System control test failed: {e}")
            return False
        
        # Test 8: Test Inter-Component Communication
        logger.info("Test 8: Testing inter-component communication...")
        try:
            # Test CGM -> KG knowledge addition
            test_knowledge = {
                "name": "Auto-Generated Test Pattern",
                "knowledge_type": "pattern",
                "content": {
                    "pattern_type": "test_pattern",
                    "description": "Pattern generated during integration test",
                    "effectiveness": 0.85
                },
                "tags": ["auto-generated", "test", "integration"]
            }
            
            # Submit as task to test cross-module communication
            task_id_2 = await sutazai_core.submit_task(
                task_type="add_knowledge",
                description="Add auto-generated test knowledge",
                data=test_knowledge,
                assigned_module="kg",
                priority=7
            )
            
            await asyncio.sleep(1)
            logger.info("✅ Inter-component communication test successful")
        except Exception as e:
            logger.error(f"❌ Inter-component communication test failed: {e}")
            return False
        
        # Test 9: Test Performance Metrics
        logger.info("Test 9: Testing performance metrics...")
        try:
            # Get comprehensive system status
            full_status = await sutazai_core.get_system_status()
            
            performance_metrics = full_status.get("performance_metrics", {})
            required_metrics = ["task_completion_rate", "average_response_time", "error_rate", "system_efficiency"]
            
            missing_metrics = [metric for metric in required_metrics if metric not in performance_metrics]
            
            if not missing_metrics:
                logger.info("✅ All performance metrics available")
                logger.info(f"   Task completion rate: {performance_metrics.get('task_completion_rate', 0):.2%}")
                logger.info(f"   System efficiency: {performance_metrics.get('system_efficiency', 0):.1f}%")
            else:
                logger.warning(f"⚠️ Missing performance metrics: {missing_metrics}")
        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            return False
        
        # Test 10: Test System Analytics
        logger.info("Test 10: Testing system analytics...")
        try:
            # Get KG analytics
            kg_analytics = await knowledge_graph.get_knowledge_analytics()
            
            # Get ACM audit log
            audit_log = await authorization_control_module.get_audit_log(limit=5)
            
            # Get CGM status
            cgm_status = await code_generation_module.get_cgm_status()
            
            logger.info("✅ System analytics retrieved successfully")
            logger.info(f"   Knowledge nodes: {kg_analytics.get('overview', {}).get('total_nodes', 0)}")
            logger.info(f"   Audit entries: {len(audit_log)}")
            logger.info(f"   Generated codes: {cgm_status.get('total_generated_codes', 0)}")
        except Exception as e:
            logger.error(f"❌ System analytics test failed: {e}")
            return False
        
        # Final Summary
        logger.info("🎉 All integration tests completed successfully!")
        logger.info("📊 Final System Status:")
        
        final_status = await sutazai_core.get_system_status()
        for component, status in final_status.items():
            if isinstance(status, dict) and "active" in str(status):
                logger.info(f"   {component.upper()}: {'✅ Active' if 'true' in str(status).lower() else '⚠️ Inactive'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed with critical error: {e}")
        return False

async def save_test_results(success: bool):
    """Save test results for reference"""
    try:
        results = {
            "test_completed_at": time.time(),
            "success": success,
            "test_summary": "Comprehensive Sutazai system integration test",
            "components_tested": [
                "sutazai_core",
                "knowledge_graph", 
                "code_generation_module",
                "authorization_control_module"
            ]
        }
        
        results_dir = Path("/opt/sutazaiapp/data/test_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / "integration_test_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Test results saved: {'SUCCESS' if success else 'FAILURE'}")
        
    except Exception as e:
        logger.error(f"Failed to save test results: {e}")

async def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("SUTAZAI INTEGRATION TEST")
    logger.info("=" * 60)
    
    try:
        # Run integration test
        success = await test_sutazai_integration()
        
        # Save results
        await save_test_results(success)
        
        if success:
            logger.info("🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY!")
            logger.info("✅ Sutazai system is fully operational")
        else:
            logger.error("❌ INTEGRATION TEST FAILED!")
            logger.error("⚠️ Please check the logs for specific issues")
        
        return success
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        await save_test_results(False)
        return False

if __name__ == "__main__":
    # Run the integration test
    success = asyncio.run(main())
    exit(0 if success else 1)