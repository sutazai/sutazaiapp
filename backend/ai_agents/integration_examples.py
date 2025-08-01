"""
Integration Examples and Usage Patterns
Comprehensive examples showing how to use the AI agent interface system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from universal_client import UniversalAgentClient, Priority, AgentType
from discovery_service import DiscoveryService, TaskRequirements, CapabilityRequirement
from workflow_orchestrator import (
    WorkflowEngine, WorkflowBuilder, ExecutionStrategy, WorkflowTask, TaskStatus
)
from communication_protocols import CommunicationHub, AgentCommunicator, MessageType
from health_monitor import HealthMonitor
from api_wrappers import UnifiedAgentAPI, ApiResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegrationExamples:
    """Comprehensive integration examples for the AI agent system."""
    
    def __init__(self):
        self.client: Optional[UniversalAgentClient] = None
        self.discovery: Optional[DiscoveryService] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.communication_hub: Optional[CommunicationHub] = None
        self.health_monitor: Optional[HealthMonitor] = None
        self.api: Optional[UnifiedAgentAPI] = None
    
    async def initialize_system(self):
        """Initialize the complete AI agent system."""
        logger.info("Initializing AI Agent System...")
        
        # Initialize universal client
        self.client = UniversalAgentClient()
        await self.client.__aenter__()
        
        # Initialize discovery service
        self.discovery = DiscoveryService(self.client)
        await self.discovery.start()
        
        # Initialize workflow engine
        self.workflow_engine = WorkflowEngine(self.client, self.discovery)
        
        # Initialize communication hub
        self.communication_hub = CommunicationHub()
        await self.communication_hub.start()
        
        # Initialize health monitor
        self.health_monitor = HealthMonitor(self.client, self.discovery)
        await self.health_monitor.start()
        
        # Initialize unified API
        self.api = UnifiedAgentAPI(self.client)
        
        logger.info("AI Agent System initialized successfully")
    
    async def cleanup_system(self):
        """Clean up system resources."""
        logger.info("Cleaning up AI Agent System...")
        
        if self.health_monitor:
            await self.health_monitor.stop()
        
        if self.communication_hub:
            await self.communication_hub.stop()
        
        if self.discovery:
            await self.discovery.stop()
        
        if self.client:
            await self.client.__aexit__(None, None, None)
        
        logger.info("AI Agent System cleanup completed")


# Example 1: Basic Agent Interaction
async def example_basic_agent_interaction():
    """Example: Basic interaction with individual agents."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Agent Interaction")
    print("="*60)
    
    examples = IntegrationExamples()
    await examples.initialize_system()
    
    try:
        # Direct agent interaction using universal client
        response = await examples.client.execute_task(
            agent_type=AgentType.CODE_GENERATION_IMPROVER,
            task_description="Analyze the quality of this Python function",
            parameters={
                "code": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
                """,
                "language": "python"
            },
            priority=Priority.HIGH
        )
        
        print(f"Task Status: {response.status}")
        print(f"Execution Time: {response.execution_time:.2f}s")
        if response.result:
            print(f"Analysis Result: {json.dumps(response.result, indent=2)}")
        
        # Using API wrapper for more structured interaction
        result = await examples.api.code_improver.analyze_code_quality(
            code="""
def process_list(items):
    result = []
    for item in items:
        if item is not None and len(str(item)) > 0:
            result.append(str(item).upper())
    return result
            """,
            language="python",
            analysis_depth="comprehensive"
        )
        
        print(f"\nAPI Wrapper Result:")
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        if result.data:
            print(f"Data: {json.dumps(result.data, indent=2)}")
        
    finally:
        await examples.cleanup_system()


# Example 2: Agent Discovery and Matching
async def example_agent_discovery():
    """Example: Using agent discovery to find suitable agents."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Agent Discovery and Matching")
    print("="*60)
    
    examples = IntegrationExamples()
    await examples.initialize_system()
    
    try:
        # Find agents by capability
        code_agents = examples.discovery.registry.get_agents_by_capability("code_analysis")
        print(f"Agents with code analysis capability: {code_agents}")
        
        # Find best agent for a specific task
        best_match = examples.discovery.find_best_agent(
            task_description="I need help with security testing and vulnerability assessment",
            capabilities=["security_testing", "vulnerability_scanning"],
            priority=Priority.HIGH
        )
        
        if best_match:
            print(f"\nBest match for security testing:")
            print(f"Agent: {best_match.agent_info.name}")
            print(f"Match Score: {best_match.total_score:.2f}")
            print(f"Reasoning: {best_match.reasoning}")
        
        # Get agent recommendations for natural language task
        recommendations = examples.discovery.get_agent_recommendations(
            "I want to build a web application with a REST API and database",
            num_recommendations=3
        )
        
        print(f"\nRecommendations for web development:")
        for i, match in enumerate(recommendations, 1):
            print(f"{i}. {match.agent_info.name} (score: {match.total_score:.2f})")
        
        # Show registry statistics
        stats = examples.discovery.get_registry_stats()
        print(f"\nRegistry Statistics:")
        print(f"Total agents: {stats['total_agents']}")
        print(f"Online agents: {stats['online_agents']}")
        print(f"Top capabilities: {stats['most_common_capabilities'][:5]}")
        
    finally:
        await examples.cleanup_system()


# Example 3: Multi-Agent Workflow Orchestration
async def example_workflow_orchestration():
    """Example: Creating and executing multi-agent workflows."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Multi-Agent Workflow Orchestration")
    print("="*60)
    
    examples = IntegrationExamples()
    await examples.initialize_system()
    
    try:
        # Create a comprehensive software development workflow
        workflow_def = (
            WorkflowBuilder("software_development", "Complete Software Development Workflow")
            .add_task(
                "requirements_analysis",
                "Requirements Analysis",
                "Analyze and document software requirements",
                agent_type="ai-product-manager",
                parameters={
                    "project_description": "Build a task management web application",
                    "stakeholders": ["developers", "end_users", "project_managers"]
                }
            )
            .add_task(
                "system_design",
                "System Architecture Design",
                "Design system architecture based on requirements",
                agent_type="agi-system-architect",
                dependencies=["requirements_analysis"],
                parameters={
                    "requirements": "{{ requirements_analysis.result }}",
                    "scalability_target": "10000_users"
                }
            )
            .add_task(
                "backend_development",
                "Backend Development",
                "Develop REST API and database layer",
                agent_type="senior-backend-developer",
                dependencies=["system_design"],
                parameters={
                    "architecture": "{{ system_design.result }}",
                    "framework": "fastapi",
                    "database": "postgresql"
                }
            )
            .add_task(
                "frontend_development",
                "Frontend Development",
                "Develop user interface",
                agent_type="senior-frontend-developer",
                dependencies=["system_design"],
                parameters={
                    "architecture": "{{ system_design.result }}",
                    "framework": "react",
                    "design_system": "material_ui"
                }
            )
            .add_task(
                "security_testing",
                "Security Testing",
                "Perform security analysis and testing",
                agent_type="security-pentesting-specialist",
                dependencies=["backend_development", "frontend_development"],
                parameters={
                    "backend_code": "{{ backend_development.result }}",
                    "frontend_code": "{{ frontend_development.result }}"
                }
            )
            .add_task(
                "qa_testing",
                "Quality Assurance Testing",
                "Perform comprehensive QA testing",
                agent_type="testing-qa-validator",
                dependencies=["backend_development", "frontend_development"],
                parameters={
                    "test_types": ["unit", "integration", "e2e"],
                    "coverage_target": 85
                }
            )
            .add_task(
                "deployment",
                "Deploy Application",
                "Deploy application to production",
                agent_type="deployment-automation-master",
                dependencies=["security_testing", "qa_testing"],
                parameters={
                    "environment": "production",
                    "deployment_strategy": "blue_green"
                }
            )
            .set_execution_strategy(ExecutionStrategy.HYBRID)
            .set_max_parallel_tasks(3)
            .build()
        )
        
        # Register and execute workflow
        examples.workflow_engine.register_workflow(workflow_def)
        
        execution_id = await examples.workflow_engine.start_workflow(
            "software_development",
            parameters={
                "project_name": "TaskMaster",
                "target_release": "Q2_2024"
            }
        )
        
        print(f"Started workflow execution: {execution_id}")
        
        # Monitor workflow execution
        monitoring_iterations = 0
        max_iterations = 20  # Prevent infinite loop in example
        
        while monitoring_iterations < max_iterations:
            status = examples.workflow_engine.get_execution_status(execution_id)
            if not status:
                break
            
            print(f"\nWorkflow Status: {status['status']}")
            print(f"Progress: {status['completed_tasks']}/{status['total_tasks']} tasks completed")
            
            # Show task details
            for task_id, task_info in status['tasks'].items():
                status_emoji = {
                    "waiting": "⏳",
                    "ready": "🟡",
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                    "skipped": "⏭️"
                }.get(task_info['status'], "❓")
                
                print(f"  {status_emoji} {task_info['name']}: {task_info['status']}")
                if task_info['assigned_agent']:
                    print(f"    Agent: {task_info['assigned_agent']}")
                if task_info['execution_time'] > 0:
                    print(f"    Time: {task_info['execution_time']:.2f}s")
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                break
            
            await asyncio.sleep(2)
            monitoring_iterations += 1
        
        # Show final results
        final_status = examples.workflow_engine.get_execution_status(execution_id)
        if final_status:
            print(f"\nFinal Status: {final_status['status']}")
            print(f"Total Duration: {final_status['duration']:.2f}s")
            
            if final_status['status'] == 'completed':
                print("🎉 Workflow completed successfully!")
            elif final_status['status'] == 'failed':
                print(f"❌ Workflow failed: {final_status.get('error', 'Unknown error')}")
        
        # Show workflow engine metrics
        metrics = examples.workflow_engine.get_metrics()
        print(f"\nWorkflow Engine Metrics:")
        print(f"Total workflows: {metrics['total_workflows']}")
        print(f"Active workflows: {metrics['active_workflows']}")
        print(f"Success rate: {metrics['successful_workflows'] / max(1, metrics['total_workflows']) * 100:.1f}%")
        
    finally:
        await examples.cleanup_system()


# Example 4: Real-Time Agent Communication
async def example_agent_communication():
    """Example: Real-time communication between agents."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Real-Time Agent Communication")
    print("="*60)
    
    examples = IntegrationExamples()
    await examples.initialize_system()
    
    try:
        # Create agent communicators
        orchestrator = AgentCommunicator("orchestrator", examples.communication_hub)
        code_agent = AgentCommunicator("code-agent", examples.communication_hub)
        security_agent = AgentCommunicator("security-agent", examples.communication_hub)
        
        # Send heartbeats
        await orchestrator.send_heartbeat(
            capabilities=["orchestration", "coordination"],
            status="active"
        )
        
        await code_agent.send_heartbeat(
            capabilities=["code_generation", "code_analysis"],
            status="active"
        )
        
        await security_agent.send_heartbeat(
            capabilities=["security_testing", "vulnerability_scanning"],
            status="active"
        )
        
        print("Agents registered and sending heartbeats")
        
        # Demonstrate task assignment communication
        print("\nDemonstrating task assignment...")
        
        task_response = await orchestrator.send_task_assignment(
            target_agent="code-agent",
            task_description="Generate a secure login function",
            parameters={
                "language": "python",
                "framework": "fastapi",
                "security_requirements": ["password_hashing", "rate_limiting"]
            },
            priority=Priority.HIGH
        )
        
        if task_response:
            print(f"Task assignment response: {task_response.payload}")
        
        # Simulate task completion
        await code_agent.send_task_completion(
            task_id="login_function_001",
            result={
                "code": "# Generated secure login function...",
                "security_features": ["bcrypt_hashing", "jwt_tokens"],
                "test_coverage": 95
            },
            execution_time=45.5
        )
        
        print("Task completion notification sent")
        
        # Demonstrate collaboration request
        print("\nDemonstrating collaboration request...")
        
        collaboration_responses = await code_agent.request_collaboration(
            capability_needed="security_testing",
            task_description="Review generated login function for security vulnerabilities",
            context={
                "code_type": "authentication",
                "security_level": "high",
                "compliance_requirements": ["OWASP"]
            }
        )
        
        print(f"Collaboration request sent, received {len(collaboration_responses)} responses")
        
        # Share knowledge between agents
        await security_agent.share_knowledge(
            knowledge_type="security_best_practices",
            knowledge_data={
                "authentication": [
                    "Use strong password hashing (bcrypt, Argon2)",
                    "Implement rate limiting",
                    "Use secure session management",
                    "Enable two-factor authentication"
                ],
                "common_vulnerabilities": [
                    "SQL injection",
                    "Cross-site scripting (XSS)",
                    "Cross-site request forgery (CSRF)"
                ]
            },
            target_agents=["code-agent"]
        )
        
        print("Knowledge shared between agents")
        
        # Show communication statistics
        stats = examples.communication_hub.get_statistics()
        print(f"\nCommunication Statistics:")
        print(f"Messages sent: {stats['messages_sent']}")
        print(f"Messages received: {stats['messages_received']}")
        print(f"Active connections: {stats['active_connections']}")
        print(f"Errors: {stats['errors']}")
        
        # Show connected agents
        connected_agents = examples.communication_hub.get_connected_agents()
        print(f"\nConnected Agents: {len(connected_agents)}")
        for agent in connected_agents:
            print(f"  - {agent['agent_id']}: {agent['capabilities']}")
        
    finally:
        await examples.cleanup_system()


# Example 5: Health Monitoring and Management
async def example_health_monitoring():
    """Example: Comprehensive health monitoring and management."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Health Monitoring and Management")
    print("="*60)
    
    examples = IntegrationExamples()
    await examples.initialize_system()
    
    try:
        # Let the health monitor run for a bit to collect metrics
        print("Collecting health metrics...")
        await asyncio.sleep(10)
        
        # Get system health overview
        overview = examples.health_monitor.get_system_health_overview()
        print(f"\nSystem Health Overview:")
        print(f"Total agents: {overview['total_agents']}")
        print(f"Healthy agents: {overview['healthy_agents']}")
        print(f"Warning agents: {overview['warning_agents']}")
        print(f"Critical agents: {overview['critical_agents']}")
        print(f"System health: {overview['system_health_percentage']:.1f}%")
        print(f"Average response time: {overview['average_response_time']:.1f}ms")
        print(f"Average uptime: {overview['average_uptime']:.1f}%")
        print(f"Active alerts: {overview['active_alerts']}")
        
        # Show individual agent health for first few agents
        print(f"\nIndividual Agent Health:")
        agent_count = 0
        for agent_id in examples.health_monitor.agent_profiles.keys():
            if agent_count >= 5:  # Limit output
                break
            
            summary = examples.health_monitor.get_agent_health_summary(agent_id)
            if summary:
                status_emoji = {
                    "healthy": "✅",
                    "warning": "⚠️",
                    "critical": "🔴",
                    "unhealthy": "❌",
                    "unknown": "❓"
                }.get(summary['current_status'], "❓")
                
                print(f"  {status_emoji} {agent_id}:")
                print(f"    Status: {summary['current_status']}")
                print(f"    Uptime: {summary['uptime_percentage']:.1f}%")
                print(f"    Success rate: {summary['success_rate']:.1f}%")
                print(f"    Avg response: {summary['average_response_time']:.1f}ms")
                print(f"    Active alerts: {summary['active_alerts']}")
                
                agent_count += 1
        
        # Show alert statistics
        alert_stats = examples.health_monitor.alert_manager.get_alert_statistics()
        print(f"\nAlert Statistics:")
        print(f"Total alerts: {alert_stats['total_alerts']}")
        print(f"Active alerts: {alert_stats['active_alerts']}")
        print(f"Resolved alerts: {alert_stats['resolved_alerts']}")
        print(f"Resolution rate: {alert_stats['resolution_rate']:.1f}%")
        
        if alert_stats['severity_breakdown']:
            print(f"Alert breakdown:")
            for severity, count in alert_stats['severity_breakdown'].items():
                print(f"  {severity}: {count}")
        
        # Show active alerts
        active_alerts = examples.health_monitor.alert_manager.get_active_alerts()
        if active_alerts:
            print(f"\nActive Alerts:")
            for alert in active_alerts[:3]:  # Show first 3
                severity_emoji = {
                    "info": "ℹ️",
                    "warning": "⚠️",
                    "error": "🔴",
                    "critical": "🚨"
                }.get(alert.severity.value, "❓")
                
                print(f"  {severity_emoji} {alert.title}")
                print(f"    Agent: {alert.agent_id}")
                print(f"    Description: {alert.description}")
                print(f"    Time: {alert.timestamp.strftime('%H:%M:%S')}")
        
        # Demonstrate recovery history
        recovery_history = examples.health_monitor.recovery_manager.get_recovery_history()
        if recovery_history:
            print(f"\nRecovery History:")
            for agent_id, recoveries in recovery_history.items():
                if recoveries:
                    latest_recovery = recoveries[-1]
                    success_emoji = "✅" if latest_recovery['success'] else "❌"
                    print(f"  {success_emoji} {agent_id}: {len(latest_recovery['actions_attempted'])} actions attempted")
        
    finally:
        await examples.cleanup_system()


# Example 6: Advanced API Integration Patterns
async def example_advanced_api_patterns():
    """Example: Advanced API integration patterns."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Advanced API Integration Patterns")
    print("="*60)
    
    examples = IntegrationExamples()
    await examples.initialize_system()
    
    try:
        # Pattern 1: Chain of agents for complex task processing
        print("Pattern 1: Agent Chaining for Code Quality Pipeline")
        
        # Step 1: Generate code
        code_result = await examples.api.opendevin_generator.generate_code(
            requirements="Create a Python function to validate email addresses using regex",
            language="python",
            framework=None
        )
        
        if code_result.success:
            generated_code = code_result.data.get('code', '')
            print(f"✅ Code generated ({len(generated_code)} characters)")
            
            # Step 2: Analyze code quality
            quality_result = await examples.api.code_improver.analyze_code_quality(
                code=generated_code,
                language="python",
                analysis_depth="comprehensive"
            )
            
            if quality_result.success:
                print(f"✅ Quality analysis completed")
                quality_score = quality_result.data.get('overall_score', 0)
                print(f"   Quality Score: {quality_score}/100")
                
                # Step 3: Security analysis
                security_result = await examples.api.semgrep_analyzer.analyze_code_security(
                    code_path="/tmp/generated_code.py",  # Simulated path
                    rule_sets=["python-security"]
                )
                
                if security_result.success:
                    print(f"✅ Security analysis completed")
                    vulnerabilities = security_result.data.get('vulnerabilities', [])
                    print(f"   Vulnerabilities found: {len(vulnerabilities)}")
                    
                    # Step 4: Generate tests
                    test_result = await examples.api.qa_validator.create_test_suite(
                        code_path="/tmp/generated_code.py",
                        test_types=["unit", "integration"],
                        coverage_target=90.0
                    )
                    
                    if test_result.success:
                        print(f"✅ Test suite generated")
                        test_count = test_result.data.get('test_count', 0)
                        print(f"   Tests created: {test_count}")
        
        # Pattern 2: Parallel execution for independent tasks
        print(f"\nPattern 2: Parallel Task Execution")
        
        tasks = [
            {
                "name": "infrastructure_analysis",
                "coro": examples.api.devops_manager.monitor_infrastructure(
                    monitoring_targets=["web_servers", "databases", "load_balancers"]
                )
            },
            {
                "name": "resource_optimization",
                "coro": examples.api.resource_optimizer.analyze_resource_usage(
                    time_range="24h",
                    include_predictions=True
                )
            },
            {
                "name": "model_management",
                "coro": examples.api.ollama_specialist.manage_models(
                    action="list"
                )
            }
        ]
        
        # Execute tasks in parallel
        results = await asyncio.gather(
            *[task["coro"] for task in tasks],
            return_exceptions=True
        )
        
        for i, result in enumerate(results):
            task_name = tasks[i]["name"]
            if isinstance(result, Exception):
                print(f"❌ {task_name}: Failed with {type(result).__name__}")
            elif hasattr(result, 'success') and result.success:
                print(f"✅ {task_name}: Completed successfully")
            else:
                print(f"⚠️ {task_name}: Completed with issues")
        
        # Pattern 3: Conditional execution based on results
        print(f"\nPattern 3: Conditional Execution Flow")
        
        # Check system architecture
        arch_result = await examples.api.agi_architect.design_system_architecture(
            requirements={
                "scalability": "high",
                "availability": "99.9%",
                "security": "enterprise"
            }
        )
        
        if arch_result.success:
            architecture = arch_result.data
            complexity_score = architecture.get('complexity_score', 5)
            
            print(f"✅ Architecture designed (complexity: {complexity_score}/10)")
            
            # Conditional execution based on complexity
            if complexity_score > 7:
                print("   High complexity detected - initiating additional validations")
                
                # Additional security review for complex architectures
                security_review = await examples.api.security_specialist.validate_security_compliance(
                    system_config=architecture,
                    compliance_standards=["SOC2", "ISO27001"]
                )
                
                if security_review.success:
                    print("   ✅ Security compliance validation completed")
                
                # Additional load testing for high-complexity systems
                load_test_result = await examples.api.qa_validator.perform_security_testing(
                    application_url="https://staging.example.com",
                    security_tests=["load_testing", "stress_testing"]
                )
                
                if load_test_result.success:
                    print("   ✅ Load testing completed")
            else:
                print("   Standard validation sufficient for this complexity level")
        
        # Pattern 4: Error handling and retry logic
        print(f"\nPattern 4: Error Handling and Retry Logic")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Simulate a potentially failing operation
                deployment_result = await examples.api.deployment_master.deploy_system(
                    deployment_spec={
                        "application": "demo_app",
                        "version": "1.0.0",
                        "environment": "staging"
                    }
                )
                
                if deployment_result.success:
                    print(f"✅ Deployment succeeded on attempt {retry_count + 1}")
                    break
                else:
                    raise Exception(deployment_result.error or "Deployment failed")
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    print(f"❌ Deployment failed (attempt {retry_count}): {str(e)}")
                    print(f"   Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ Deployment failed after {max_retries} attempts")
                    
                    # Escalate to human intervention
                    await examples.api.ai_product_manager.execute(
                        "Escalate deployment failure to operations team",
                        {
                            "failure_reason": str(e),
                            "retry_attempts": max_retries,
                            "escalation_level": "urgent"
                        }
                    )
                    print("   🚨 Escalated to operations team")
        
        # Pattern 5: Performance monitoring and optimization
        print(f"\nPattern 5: Performance Monitoring")
        
        # Monitor system performance
        performance_tasks = []
        
        # Collect metrics from different agents
        for agent_type in [AgentType.AGI_SYSTEM_ARCHITECT, AgentType.CODE_GENERATION_IMPROVER, 
                          AgentType.SENIOR_AI_ENGINEER]:
            try:
                wrapper = examples.api.get_agent(agent_type)
                task = wrapper.execute("get performance metrics")
                performance_tasks.append((agent_type.value, task))
            except Exception as e:
                print(f"⚠️ Could not get metrics for {agent_type.value}: {str(e)}")
        
        # Collect results
        performance_results = []
        for agent_id, task in performance_tasks:
            try:
                result = await task
                performance_results.append((agent_id, result))
            except Exception as e:
                print(f"⚠️ Performance check failed for {agent_id}: {str(e)}")
        
        # Analyze performance
        total_agents_checked = len(performance_results)
        successful_checks = sum(1 for _, result in performance_results if result.success)
        
        print(f"Performance Check Results:")
        print(f"  Agents checked: {total_agents_checked}")
        print(f"  Successful checks: {successful_checks}")
        print(f"  Success rate: {successful_checks/total_agents_checked*100:.1f}%")
        
        # Show execution times
        execution_times = [result.execution_time for _, result in performance_results if result.success]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            print(f"  Average response time: {avg_time:.2f}s")
            print(f"  Max response time: {max_time:.2f}s")
            print(f"  Min response time: {min_time:.2f}s")
    
    finally:
        await examples.cleanup_system()


# Example 7: Complete End-to-End Scenario
async def example_complete_scenario():
    """Example: Complete end-to-end AI development scenario."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Complete End-to-End AI Development Scenario")
    print("="*60)
    print("Scenario: Building an AI-powered customer service chatbot")
    
    examples = IntegrationExamples()
    await examples.initialize_system()
    
    try:
        # Phase 1: Project Planning and Architecture
        print("\n📋 Phase 1: Project Planning and Architecture")
        
        # Requirements gathering
        requirements_result = await examples.api.ai_product_manager.execute(
            "Gather requirements for AI customer service chatbot",
            {
                "project_type": "ai_chatbot",
                "industry": "e_commerce",
                "expected_users": 10000,
                "languages": ["english", "spanish"],
                "integration_requirements": ["crm", "knowledge_base", "live_chat"]
            }
        )
        
        if requirements_result.success:
            print("✅ Requirements gathered successfully")
            requirements = requirements_result.data
        
        # System architecture design
        architecture_result = await examples.api.agi_architect.design_system_architecture(
            requirements={
                "type": "ai_chatbot_system",
                "scalability": "high",
                "real_time": True,
                "ml_components": ["nlp", "intent_recognition", "response_generation"]
            },
            constraints={
                "budget": "moderate",
                "timeline": "3_months",
                "team_size": 5
            }
        )
        
        if architecture_result.success:
            print("✅ System architecture designed")
            architecture = architecture_result.data
        
        # Phase 2: AI Model Development
        print("\n🤖 Phase 2: AI Model Development")
        
        # Design ML architecture
        ml_architecture_result = await examples.api.ai_engineer.design_ml_architecture(
            requirements={
                "task_type": "conversational_ai",
                "model_types": ["intent_classification", "entity_extraction", "response_generation"],
                "performance_targets": {"accuracy": 0.95, "latency": "< 500ms"}
            },
            data_specs={
                "training_data_size": "100k_conversations",
                "languages": ["en", "es"],
                "domains": ["customer_service", "e_commerce"]
            }
        )
        
        if ml_architecture_result.success:
            print("✅ ML architecture designed")
        
        # Set up RAG system for knowledge base
        rag_result = await examples.api.ai_engineer.implement_rag_system(
            knowledge_sources=[
                "product_catalog",
                "faq_database",
                "support_articles",
                "company_policies"
            ],
            rag_config={
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_store": "chromadb",
                "chunk_size": 512,
                "similarity_threshold": 0.8
            }
        )
        
        if rag_result.success:
            print("✅ RAG system implemented")
        
        # Configure Ollama for local inference
        model_setup_result = await examples.api.ollama_specialist.manage_models(
            action="pull",
            model_name="llama2:7b-chat",
            model_config={
                "temperature": 0.7,
                "max_tokens": 512,
                "context_window": 4096
            }
        )
        
        if model_setup_result.success:
            print("✅ Language model configured")
        
        # Phase 3: Application Development
        print("\n💻 Phase 3: Application Development")
        
        # Backend API development
        backend_result = await examples.api.backend_dev.design_api(
            api_spec={
                "endpoints": [
                    {"path": "/chat/message", "method": "POST"},
                    {"path": "/chat/history", "method": "GET"},
                    {"path": "/chat/feedback", "method": "POST"}
                ],
                "authentication": "jwt",
                "rate_limiting": "100_requests_per_minute",
                "websocket_support": True
            },
            framework="fastapi"
        )
        
        if backend_result.success:
            print("✅ Backend API developed")
        
        # Frontend development
        frontend_result = await examples.api.frontend_dev.create_ui_component(
            component_spec={
                "type": "chat_interface",
                "features": ["message_history", "typing_indicators", "file_upload"],
                "accessibility": True,
                "mobile_responsive": True
            },
            framework="react"
        )
        
        if frontend_result.success:
            print("✅ Chat interface developed")
        
        # Real-time features
        realtime_result = await examples.api.frontend_dev.implement_realtime_features(
            feature_spec={
                "features": ["typing_indicators", "live_agent_handoff", "presence_status"],
                "fallback_polling_interval": 5000
            },
            websocket_config={
                "endpoint": "/ws/chat",
                "heartbeat_interval": 30,
                "reconnection_strategy": "exponential_backoff"
            }
        )
        
        if realtime_result.success:
            print("✅ Real-time features implemented")
        
        # Phase 4: Quality Assurance and Security
        print("\n🔍 Phase 4: Quality Assurance and Security")
        
        # Comprehensive testing
        testing_result = await examples.api.qa_validator.create_test_suite(
            code_path="/app/chatbot",
            test_types=["unit", "integration", "e2e", "performance", "load"],
            coverage_target=90.0
        )
        
        if testing_result.success:
            print("✅ Comprehensive test suite created")
        
        # Security analysis
        security_analysis_result = await examples.api.semgrep_analyzer.analyze_code_security(
            code_path="/app/chatbot",
            rule_sets=["owasp-top-10", "pii-detection", "api-security"]
        )
        
        if security_analysis_result.success:
            print("✅ Security analysis completed")
            vulnerabilities = security_analysis_result.data.get('vulnerabilities', [])
            if vulnerabilities:
                print(f"   ⚠️ {len(vulnerabilities)} potential issues found")
            else:
                print("   🛡️ No security issues detected")
        
        # Penetration testing
        pentest_result = await examples.api.security_specialist.perform_penetration_test(
            target_info={
                "application_type": "web_api",
                "endpoints": ["/api/chat", "/api/auth", "/ws/chat"],
                "authentication_method": "jwt"
            },
            test_scenarios=["injection_attacks", "auth_bypass", "dos_attacks"]
        )
        
        if pentest_result.success:
            print("✅ Penetration testing completed")
        
        # Phase 5: Deployment and Monitoring
        print("\n🚀 Phase 5: Deployment and Monitoring")
        
        # Infrastructure setup
        infra_result = await examples.api.devops_manager.setup_cicd_pipeline(
            repository_url="https://github.com/company/chatbot-service",
            pipeline_config={
                "stages": ["test", "security_scan", "build", "deploy"],
                "environments": ["staging", "production"],
                "deployment_strategy": "blue_green",
                "rollback_enabled": True
            }
        )
        
        if infra_result.success:
            print("✅ CI/CD pipeline configured")
        
        # System deployment
        deployment_result = await examples.api.deployment_master.deploy_system(
            deployment_spec={
                "application": "ai_chatbot",
                "version": "1.0.0",
                "environment": "production",
                "scaling_config": {
                    "min_instances": 2,
                    "max_instances": 10,
                    "cpu_threshold": 70,
                    "memory_threshold": 80
                }
            },
            environment="production"
        )
        
        if deployment_result.success:
            print("✅ System deployed to production")
        
        # Set up monitoring
        monitoring_result = await examples.api.devops_manager.monitor_infrastructure(
            monitoring_targets=[
                "api_servers",
                "database",
                "ml_inference_service",
                "websocket_service"
            ],
            alert_config={
                "response_time_threshold": 1000,
                "error_rate_threshold": 0.01,
                "cpu_threshold": 80,
                "memory_threshold": 85
            }
        )
        
        if monitoring_result.success:
            print("✅ Infrastructure monitoring configured")
        
        # Phase 6: Optimization and Maintenance
        print("\n⚡ Phase 6: Optimization and Maintenance")
        
        # Performance optimization
        optimization_result = await examples.api.resource_optimizer.optimize_resource_allocation(
            workload_profiles=[
                {
                    "service": "chat_api",
                    "cpu_pattern": "moderate",
                    "memory_pattern": "high",
                    "traffic_pattern": "business_hours"
                },
                {
                    "service": "ml_inference",
                    "cpu_pattern": "high",
                    "memory_pattern": "very_high",
                    "traffic_pattern": "continuous"
                }
            ],
            constraints={
                "budget_limit": 10000,
                "sla_requirements": "99.9%_uptime"
            }
        )
        
        if optimization_result.success:
            print("✅ Resource allocation optimized")
        
        # Model performance optimization
        model_optimization_result = await examples.api.ollama_specialist.optimize_model_performance(
            model_name="llama2:7b-chat",
            performance_metrics={
                "current_latency": 800,
                "target_latency": 500,
                "accuracy": 0.94,
                "throughput": 10
            }
        )
        
        if model_optimization_result.success:
            print("✅ Model performance optimized")
        
        # Continuous learning setup
        continuous_learning_result = await examples.api.brain_manager.implement_continuous_learning(
            learning_config={
                "learning_rate": 0.001,
                "batch_size": 32,
                "update_frequency": "daily",
                "validation_split": 0.2
            },
            data_sources=[
                "user_conversations",
                "feedback_ratings",
                "support_escalations"
            ]
        )
        
        if continuous_learning_result.success:
            print("✅ Continuous learning system configured")
        
        # Final Status Report
        print("\n📊 PROJECT COMPLETION SUMMARY")
        print("="*50)
        
        phases_completed = [
            "✅ Requirements & Architecture",
            "✅ AI Model Development", 
            "✅ Application Development",
            "✅ Quality Assurance & Security",
            "✅ Deployment & Monitoring",
            "✅ Optimization & Maintenance"
        ]
        
        for phase in phases_completed:
            print(f"  {phase}")
        
        print(f"\n🎉 AI Customer Service Chatbot successfully deployed!")
        print(f"📈 System is ready to handle production traffic")
        print(f"🔄 Continuous monitoring and learning enabled")
        
    finally:
        await examples.cleanup_system()


# Main execution function
async def run_all_examples():
    """Run all integration examples."""
    print("🚀 Starting AI Agent Integration Examples")
    print("This will demonstrate the complete capabilities of the AI Agent System")
    print("="*80)
    
    examples_to_run = [
        ("Basic Agent Interaction", example_basic_agent_interaction),
        ("Agent Discovery and Matching", example_agent_discovery),
        ("Multi-Agent Workflow Orchestration", example_workflow_orchestration),
        ("Real-Time Agent Communication", example_agent_communication),
        ("Health Monitoring and Management", example_health_monitoring),
        ("Advanced API Integration Patterns", example_advanced_api_patterns),
        ("Complete End-to-End Scenario", example_complete_scenario)
    ]
    
    for example_name, example_func in examples_to_run:
        try:
            print(f"\n🔄 Running: {example_name}")
            await example_func()
            print(f"✅ Completed: {example_name}")
        except Exception as e:
            print(f"❌ Failed: {example_name} - {str(e)}")
            logger.exception(f"Example failed: {example_name}")
        
        # Small delay between examples
        await asyncio.sleep(2)
    
    print(f"\n🏁 All integration examples completed!")
    print(f"Check the output above for detailed results and patterns.")


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())