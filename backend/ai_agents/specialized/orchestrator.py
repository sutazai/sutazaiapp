"""
Orchestrator Agent - Master Coordinator for Multi-Agent Systems
==============================================================

This agent acts as the central coordinator for complex multi-agent workflows.
It can break down high-level tasks, distribute them to specialized agents,
and coordinate their execution to achieve complex goals autonomously.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum

from ..core.base_agent import BaseAgent, AgentMessage, AgentStatus, AgentCapability
from ..core.agent_registry import get_agent_registry
from ..core.agent_message_bus import get_message_bus


class WorkflowType(Enum):
    """Types of workflows the orchestrator can handle"""
    CODE_DEVELOPMENT = "code_development"
    SECURITY_AUDIT = "security_audit"
    SYSTEM_DEPLOYMENT = "system_deployment"
    DATA_PROCESSING = "data_processing"
    TESTING_PIPELINE = "testing_pipeline"
    DOCUMENTATION = "documentation"
    RESEARCH_ANALYSIS = "research_analysis"
    CUSTOM = "custom"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class OrchestratorAgent(BaseAgent):
    """
    Master Orchestrator Agent
    
    Capabilities:
    - Break down complex tasks into subtasks
    - Coordinate multiple agents
    - Monitor workflow progress
    - Handle failures and recovery
    - Optimize resource allocation
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Workflow management
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_templates: Dict[WorkflowType, Dict[str, Any]] = {}
        self.task_dependencies: Dict[str, Set[str]] = {}
        
        # Agent coordination
        self.agent_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.agent_capabilities_cache: Dict[str, Set[AgentCapability]] = {}
        
        # Performance tracking
        self.workflow_stats = {
            "total_workflows": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "average_completion_time": 0.0,
            "agent_utilization": {}
        }
        
        self._initialize_workflow_templates()
    
    def _initialize_workflow_templates(self):
        """Initialize predefined workflow templates"""
        
        # Code Development Workflow
        self.workflow_templates[WorkflowType.CODE_DEVELOPMENT] = {
            "name": "Code Development Pipeline",
            "description": "Complete code development from specification to deployment",
            "phases": [
                {
                    "name": "analysis",
                    "tasks": [
                        {
                            "name": "analyze_requirements",
                            "agent_type": "orchestrator",
                            "capabilities": ["reasoning"],
                            "priority": TaskPriority.HIGH
                        },
                        {
                            "name": "create_architecture",
                            "agent_type": "orchestrator",
                            "capabilities": ["reasoning", "code_generation"],
                            "priority": TaskPriority.HIGH,
                            "depends_on": ["analyze_requirements"]
                        }
                    ]
                },
                {
                    "name": "implementation",
                    "tasks": [
                        {
                            "name": "generate_code",
                            "agent_type": "code_generator",
                            "capabilities": ["code_generation"],
                            "priority": TaskPriority.NORMAL,
                            "depends_on": ["create_architecture"]
                        },
                        {
                            "name": "create_tests",
                            "agent_type": "test_agent",
                            "capabilities": ["testing"],
                            "priority": TaskPriority.NORMAL,
                            "depends_on": ["generate_code"]
                        }
                    ]
                },
                {
                    "name": "validation",
                    "tasks": [
                        {
                            "name": "security_scan",
                            "agent_type": "security_analyzer",
                            "capabilities": ["security_analysis"],
                            "priority": TaskPriority.HIGH,
                            "depends_on": ["generate_code"]
                        },
                        {
                            "name": "run_tests",
                            "agent_type": "test_agent",
                            "capabilities": ["testing"],
                            "priority": TaskPriority.HIGH,
                            "depends_on": ["create_tests"]
                        }
                    ]
                },
                {
                    "name": "deployment",
                    "tasks": [
                        {
                            "name": "deploy_application",
                            "agent_type": "deployment_agent",
                            "capabilities": ["deployment"],
                            "priority": TaskPriority.CRITICAL,
                            "depends_on": ["security_scan", "run_tests"]
                        }
                    ]
                }
            ]
        }
        
        # Security Audit Workflow
        self.workflow_templates[WorkflowType.SECURITY_AUDIT] = {
            "name": "Comprehensive Security Audit",
            "description": "Complete security analysis of codebase",
            "phases": [
                {
                    "name": "preparation",
                    "tasks": [
                        {
                            "name": "scan_codebase",
                            "agent_type": "security_analyzer",
                            "capabilities": ["security_analysis", "code_analysis"],
                            "priority": TaskPriority.HIGH
                        },
                        {
                            "name": "identify_dependencies",
                            "agent_type": "security_analyzer",
                            "capabilities": ["security_analysis"],
                            "priority": TaskPriority.NORMAL
                        }
                    ]
                },
                {
                    "name": "analysis",
                    "tasks": [
                        {
                            "name": "vulnerability_assessment",
                            "agent_type": "security_analyzer",
                            "capabilities": ["security_analysis"],
                            "priority": TaskPriority.CRITICAL,
                            "depends_on": ["scan_codebase"]
                        },
                        {
                            "name": "dependency_audit",
                            "agent_type": "security_analyzer",
                            "capabilities": ["security_analysis"],
                            "priority": TaskPriority.HIGH,
                            "depends_on": ["identify_dependencies"]
                        }
                    ]
                },
                {
                    "name": "remediation",
                    "tasks": [
                        {
                            "name": "generate_security_fixes",
                            "agent_type": "code_generator",
                            "capabilities": ["code_generation", "security_analysis"],
                            "priority": TaskPriority.CRITICAL,
                            "depends_on": ["vulnerability_assessment"]
                        },
                        {
                            "name": "create_security_report",
                            "agent_type": "orchestrator",
                            "capabilities": ["reasoning"],
                            "priority": TaskPriority.NORMAL,
                            "depends_on": ["vulnerability_assessment", "dependency_audit"]
                        }
                    ]
                }
            ]
        }
    
    async def on_initialize(self):
        """Initialize orchestrator specific components"""
        self.logger.info("Initializing Orchestrator Agent")
        
        # Register message handlers
        self.register_message_handler("create_workflow", self._handle_create_workflow)
        self.register_message_handler("execute_workflow", self._handle_execute_workflow)
        self.register_message_handler("task_completed", self._handle_task_completed)
        self.register_message_handler("task_failed", self._handle_task_failed)
        self.register_message_handler("workflow_status", self._handle_workflow_status)
        self.register_message_handler("cancel_workflow", self._handle_cancel_workflow)
        
        # Add orchestration capabilities
        self.add_capability(AgentCapability.ORCHESTRATION)
        self.add_capability(AgentCapability.COMMUNICATION)
        self.add_capability(AgentCapability.REASONING)
        self.add_capability(AgentCapability.AUTONOMOUS_EXECUTION)
        
        # Start monitoring tasks
        asyncio.create_task(self._workflow_monitor())
        asyncio.create_task(self._performance_tracker())
        
        self.logger.info("Orchestrator Agent initialized successfully")
    
    async def on_task_execute(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute orchestration task"""
        task_type = task_data.get("task_type", "orchestrate")
        
        try:
            if task_type == "create_workflow":
                return await self._create_workflow_from_request(task_id, task_data)
            elif task_type == "execute_workflow":
                return await self._execute_workflow_by_id(task_id, task_data)
            elif task_type == "analyze_requirements":
                return await self._analyze_requirements(task_id, task_data)
            elif task_type == "create_architecture":
                return await self._create_architecture(task_id, task_data)
            elif task_type == "coordinate_agents":
                return await self._coordinate_agents(task_id, task_data)
            else:
                return await self._handle_custom_orchestration(task_id, task_data)
                
        except Exception as e:
            self.logger.error(f"Orchestration task {task_id} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def _create_workflow_from_request(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a workflow from a high-level request"""
        request = task_data.get("request", "")
        workflow_type = task_data.get("workflow_type", "custom")
        
        if not request:
            return {
                "success": False,
                "error": "No request provided",
                "task_id": task_id
            }
        
        # Analyze the request to determine workflow structure
        workflow_analysis = await self._analyze_workflow_request(request, workflow_type)
        
        if not workflow_analysis["success"]:
            return workflow_analysis
        
        # Create workflow from analysis
        workflow_id = str(uuid.uuid4())
        workflow = {
            "id": workflow_id,
            "name": workflow_analysis["workflow_name"],
            "description": workflow_analysis["description"],
            "type": workflow_type,
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "request": request,
            "tasks": workflow_analysis["tasks"],
            "dependencies": workflow_analysis["dependencies"],
            "estimated_duration": workflow_analysis.get("estimated_duration", 0),
            "progress": 0.0
        }
        
        self.active_workflows[workflow_id] = workflow
        self.workflow_stats["total_workflows"] += 1
        
        return {
            "success": True,
            "result": {
                "workflow_id": workflow_id,
                "workflow": workflow,
                "analysis": workflow_analysis
            },
            "task_id": task_id
        }
    
    async def _analyze_workflow_request(self, request: str, workflow_type: str) -> Dict[str, Any]:
        """Analyze a high-level request and break it down into tasks"""
        
        system_prompt = """You are an expert workflow orchestrator. Analyze the given request and break it down into specific, actionable tasks.

Guidelines:
- Identify the main goal and sub-goals
- Break down complex tasks into smaller, manageable subtasks
- Determine task dependencies and proper execution order
- Assign appropriate agent types based on required capabilities
- Estimate effort and duration for each task
- Consider error handling and recovery scenarios

Output Format: JSON structure with tasks, dependencies, and metadata."""

        user_prompt = f"""Analyze the following request and create a detailed workflow plan:

Request: {request}
Workflow Type: {workflow_type}

Please provide a comprehensive breakdown including:
1. Workflow name and description
2. List of tasks with details (name, description, agent_type, capabilities, priority, estimated_duration)
3. Task dependencies
4. Overall workflow estimated duration
5. Success criteria
6. Risk assessment and mitigation strategies

Format the response as a JSON structure."""

        analysis_result = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=3000
        )
        
        if not analysis_result:
            return {
                "success": False,
                "error": "Failed to analyze workflow request"
            }
        
        try:
            # Parse the JSON response
            analysis_data = json.loads(analysis_result)
            
            return {
                "success": True,
                "workflow_name": analysis_data.get("workflow_name", "Generated Workflow"),
                "description": analysis_data.get("description", ""),
                "tasks": analysis_data.get("tasks", []),
                "dependencies": analysis_data.get("dependencies", {}),
                "estimated_duration": analysis_data.get("estimated_duration", 0),
                "success_criteria": analysis_data.get("success_criteria", []),
                "risks": analysis_data.get("risks", [])
            }
            
        except json.JSONDecodeError:
            # Fallback: create a simple workflow structure
            return await self._create_fallback_workflow(request, workflow_type)
    
    async def _create_fallback_workflow(self, request: str, workflow_type: str) -> Dict[str, Any]:
        """Create a simple fallback workflow when analysis fails"""
        
        # Use predefined template if available
        if workflow_type in self.workflow_templates:
            template = self.workflow_templates[WorkflowType(workflow_type)]
            
            tasks = []
            dependencies = {}
            
            for phase in template["phases"]:
                for task_template in phase["tasks"]:
                    task_id = str(uuid.uuid4())
                    task = {
                        "id": task_id,
                        "name": task_template["name"],
                        "description": f"Execute {task_template['name']} for: {request}",
                        "agent_type": task_template["agent_type"],
                        "capabilities": task_template["capabilities"],
                        "priority": task_template["priority"].value,
                        "estimated_duration": 300  # 5 minutes default
                    }
                    tasks.append(task)
                    
                    if "depends_on" in task_template:
                        dependencies[task_id] = task_template["depends_on"]
            
            return {
                "success": True,
                "workflow_name": template["name"],
                "description": template["description"],
                "tasks": tasks,
                "dependencies": dependencies,
                "estimated_duration": len(tasks) * 300
            }
        
        # Create a basic single-task workflow
        task_id = str(uuid.uuid4())
        return {
            "success": True,
            "workflow_name": "Custom Task Execution",
            "description": f"Execute custom request: {request}",
            "tasks": [
                {
                    "id": task_id,
                    "name": "execute_request",
                    "description": request,
                    "agent_type": "generic",
                    "capabilities": ["reasoning"],
                    "priority": TaskPriority.NORMAL.value,
                    "estimated_duration": 600
                }
            ],
            "dependencies": {},
            "estimated_duration": 600
        }
    
    async def _execute_workflow_by_id(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow by ID"""
        workflow_id = task_data.get("workflow_id", "")
        
        if not workflow_id or workflow_id not in self.active_workflows:
            return {
                "success": False,
                "error": "Workflow not found",
                "task_id": task_id
            }
        
        workflow = self.active_workflows[workflow_id]
        
        if workflow["status"] != "created":
            return {
                "success": False,
                "error": "Workflow is not in created state",
                "task_id": task_id
            }
        
        # Start workflow execution
        workflow["status"] = "running"
        workflow["started_at"] = datetime.utcnow().isoformat()
        
        # Schedule initial tasks (tasks with no dependencies)
        ready_tasks = self._get_ready_tasks(workflow)
        
        for task in ready_tasks:
            await self._schedule_task(workflow_id, task)
        
        return {
            "success": True,
            "result": {
                "workflow_id": workflow_id,
                "status": "running",
                "scheduled_tasks": len(ready_tasks)
            },
            "task_id": task_id
        }
    
    def _get_ready_tasks(self, workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get tasks that are ready to execute (dependencies met)"""
        ready_tasks = []
        
        for task in workflow["tasks"]:
            if task.get("status", "pending") != "pending":
                continue
            
            # Check if all dependencies are completed
            task_id = task["id"]
            dependencies = workflow["dependencies"].get(task_id, [])
            
            dependencies_met = True
            for dep_name in dependencies:
                # Find dependency task by name
                dep_task = next((t for t in workflow["tasks"] if t["name"] == dep_name), None)
                if not dep_task or dep_task.get("status") != "completed":
                    dependencies_met = False
                    break
            
            if dependencies_met:
                ready_tasks.append(task)
        
        return ready_tasks
    
    async def _schedule_task(self, workflow_id: str, task: Dict[str, Any]):
        """Schedule a task for execution by finding appropriate agent"""
        
        # Find suitable agent
        registry = get_agent_registry()
        if not registry:
            self.logger.error("Agent registry not available")
            return
        
        # Convert capability strings to enums
        required_capabilities = []
        for cap_str in task.get("capabilities", []):
            try:
                required_capabilities.append(AgentCapability(cap_str))
            except ValueError:
                self.logger.warning(f"Unknown capability: {cap_str}")
        
        # Select best agent
        selected_agent = registry.select_best_agent(
            required_capabilities=required_capabilities,
            agent_type=task.get("agent_type")
        )
        
        if not selected_agent:
            self.logger.error(f"No suitable agent found for task {task['name']}")
            task["status"] = "failed"
            task["error"] = "No suitable agent available"
            return
        
        # Send task to agent
        task["status"] = "assigned"
        task["assigned_agent"] = selected_agent.agent_id
        task["assigned_at"] = datetime.utcnow().isoformat()
        
        self.agent_assignments[task["id"]] = selected_agent.agent_id
        
        # Create task message
        task_message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=selected_agent.agent_id,
            message_type="execute_task",
            content={
                "workflow_id": workflow_id,
                "task_id": task["id"],
                "task_name": task["name"],
                "task_description": task["description"],
                "task_data": task,
                "orchestrator_id": self.agent_id
            },
            priority=task.get("priority", 3)
        )
        
        message_bus = get_message_bus()
        if message_bus:
            await message_bus.send_message(task_message)
            task["status"] = "running"
            task["started_at"] = datetime.utcnow().isoformat()
            
            self.logger.info(f"Scheduled task {task['name']} to agent {selected_agent.agent_id}")
    
    async def _analyze_requirements(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze requirements for a project"""
        requirements = task_data.get("requirements", "")
        
        if not requirements:
            return {
                "success": False,
                "error": "No requirements provided",
                "task_id": task_id
            }
        
        system_prompt = """You are an expert business analyst and software architect. Analyze the given requirements and provide detailed insights.

Guidelines:
- Identify functional and non-functional requirements
- Determine scope and complexity
- Identify potential challenges and risks
- Suggest technical approaches and technologies
- Estimate effort and resources needed
- Identify stakeholders and success criteria"""

        user_prompt = f"""Analyze the following requirements:

{requirements}

Please provide a comprehensive analysis including:
1. Requirements breakdown (functional vs non-functional)
2. Scope and complexity assessment
3. Technical considerations and approaches
4. Potential challenges and risks
5. Resource and timeline estimates
6. Success criteria and acceptance criteria
7. Recommended next steps"""

        analysis = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2500
        )
        
        if not analysis:
            return {
                "success": False,
                "error": "Failed to analyze requirements",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "requirements": requirements,
                "analysis": analysis,
                "analyzed_at": datetime.utcnow().isoformat()
            },
            "task_id": task_id
        }
    
    async def _create_architecture(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create system architecture based on requirements"""
        requirements_analysis = task_data.get("requirements_analysis", "")
        project_type = task_data.get("project_type", "web_application")
        
        if not requirements_analysis:
            return {
                "success": False,
                "error": "No requirements analysis provided",
                "task_id": task_id
            }
        
        system_prompt = """You are an expert software architect. Create a comprehensive system architecture based on the requirements analysis.

Guidelines:
- Design scalable and maintainable architecture
- Consider security, performance, and reliability
- Choose appropriate technologies and patterns
- Define clear component boundaries and interfaces
- Consider deployment and operational aspects
- Document architectural decisions and rationale"""

        user_prompt = f"""Create a system architecture for the following project:

Project Type: {project_type}

Requirements Analysis:
{requirements_analysis}

Please provide a detailed architecture including:
1. High-level system overview
2. Component architecture and relationships
3. Technology stack recommendations
4. Data architecture and storage strategy
5. Security architecture considerations
6. Deployment and infrastructure requirements
7. Performance and scalability considerations
8. Integration points and APIs
9. Development and testing strategy"""

        architecture = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=3000
        )
        
        if not architecture:
            return {
                "success": False,
                "error": "Failed to create architecture",
                "task_id": task_id
            }
        
        return {
            "success": True,
            "result": {
                "project_type": project_type,
                "architecture": architecture,
                "created_at": datetime.utcnow().isoformat()
            },
            "task_id": task_id
        }
    
    async def _coordinate_agents(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents for complex tasks"""
        coordination_plan = task_data.get("coordination_plan", {})
        target_agents = task_data.get("target_agents", [])
        
        if not coordination_plan:
            return {
                "success": False,
                "error": "No coordination plan provided",
                "task_id": task_id
            }
        
        coordination_results = []
        
        for agent_task in coordination_plan.get("agent_tasks", []):
            agent_id = agent_task.get("agent_id")
            task_details = agent_task.get("task_details", {})
            
            if agent_id:
                # Send coordination message to agent
                coord_message = AgentMessage(
                    sender_id=self.agent_id,
                    receiver_id=agent_id,
                    message_type="coordinate_task",
                    content={
                        "coordination_id": task_id,
                        "task_details": task_details,
                        "orchestrator_id": self.agent_id
                    }
                )
                
                message_bus = get_message_bus()
                if message_bus:
                    await message_bus.send_message(coord_message)
                    coordination_results.append({
                        "agent_id": agent_id,
                        "status": "coordinated",
                        "message_id": coord_message.id
                    })
        
        return {
            "success": True,
            "result": {
                "coordination_id": task_id,
                "coordinated_agents": len(coordination_results),
                "results": coordination_results
            },
            "task_id": task_id
        }
    
    async def _handle_custom_orchestration(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom orchestration tasks"""
        custom_request = task_data.get("custom_request", "")
        
        if not custom_request:
            return {
                "success": False,
                "error": "No custom request provided",
                "task_id": task_id
            }
        
        system_prompt = """You are an expert orchestrator capable of handling any type of coordination task. Analyze the request and provide appropriate orchestration."""

        user_prompt = f"""Handle the following custom orchestration request:

{custom_request}

Please provide:
1. Analysis of the request
2. Orchestration strategy
3. Required resources and agents
4. Execution plan
5. Success criteria"""

        orchestration_plan = await self.query_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=2000
        )
        
        return {
            "success": True,
            "result": {
                "custom_request": custom_request,
                "orchestration_plan": orchestration_plan,
                "handled_at": datetime.utcnow().isoformat()
            },
            "task_id": task_id
        }
    
    # Background monitoring tasks
    async def _workflow_monitor(self):
        """Monitor active workflows"""
        while self.status != AgentStatus.OFFLINE:
            try:
                for workflow_id, workflow in list(self.active_workflows.items()):
                    if workflow["status"] == "running":
                        await self._check_workflow_progress(workflow_id, workflow)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Workflow monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _check_workflow_progress(self, workflow_id: str, workflow: Dict[str, Any]):
        """Check progress of a running workflow"""
        total_tasks = len(workflow["tasks"])
        completed_tasks = sum(1 for task in workflow["tasks"] 
                            if task.get("status") == "completed")
        failed_tasks = sum(1 for task in workflow["tasks"] 
                         if task.get("status") == "failed")
        
        # Update progress
        workflow["progress"] = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Check if workflow is complete
        if completed_tasks + failed_tasks == total_tasks:
            if failed_tasks == 0:
                workflow["status"] = "completed"
                workflow["completed_at"] = datetime.utcnow().isoformat()
                self.workflow_stats["completed_workflows"] += 1
                
                self.logger.info(f"Workflow {workflow_id} completed successfully")
            else:
                workflow["status"] = "failed"
                workflow["completed_at"] = datetime.utcnow().isoformat()
                workflow["error"] = f"{failed_tasks} tasks failed"
                self.workflow_stats["failed_workflows"] += 1
                
                self.logger.error(f"Workflow {workflow_id} failed with {failed_tasks} failed tasks")
        
        # Schedule new tasks if available
        ready_tasks = self._get_ready_tasks(workflow)
        for task in ready_tasks:
            await self._schedule_task(workflow_id, task)
    
    async def _performance_tracker(self):
        """Track orchestrator performance"""
        while self.status != AgentStatus.OFFLINE:
            try:
                # Calculate average completion time
                completed_workflows = [w for w in self.active_workflows.values() 
                                     if w["status"] == "completed"]
                
                if completed_workflows:
                    total_duration = 0
                    for workflow in completed_workflows:
                        if "started_at" in workflow and "completed_at" in workflow:
                            start_time = datetime.fromisoformat(workflow["started_at"])
                            end_time = datetime.fromisoformat(workflow["completed_at"])
                            duration = (end_time - start_time).total_seconds()
                            total_duration += duration
                    
                    self.workflow_stats["average_completion_time"] = (
                        total_duration / len(completed_workflows)
                    )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance tracker error: {e}")
                await asyncio.sleep(300)
    
    # Message handlers
    async def _handle_create_workflow(self, message: AgentMessage):
        """Handle workflow creation request"""
        content = message.content
        
        result = await self._create_workflow_from_request(
            message.id,
            {
                "request": content.get("request", ""),
                "workflow_type": content.get("workflow_type", "custom")
            }
        )
        
        await self.send_message(
            message.sender_id,
            "workflow_created",
            result
        )
    
    async def _handle_execute_workflow(self, message: AgentMessage):
        """Handle workflow execution request"""
        content = message.content
        
        result = await self._execute_workflow_by_id(
            message.id,
            {"workflow_id": content.get("workflow_id", "")}
        )
        
        await self.send_message(
            message.sender_id,
            "workflow_execution_started",
            result
        )
    
    async def _handle_task_completed(self, message: AgentMessage):
        """Handle task completion notification"""
        content = message.content
        task_id = content.get("task_id", "")
        workflow_id = content.get("workflow_id", "")
        
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            
            # Find and update task
            for task in workflow["tasks"]:
                if task["id"] == task_id:
                    task["status"] = "completed"
                    task["completed_at"] = datetime.utcnow().isoformat()
                    task["result"] = content.get("result")
                    break
            
            # Remove from agent assignments
            self.agent_assignments.pop(task_id, None)
            
            self.logger.info(f"Task {task_id} completed in workflow {workflow_id}")
    
    async def _handle_task_failed(self, message: AgentMessage):
        """Handle task failure notification"""
        content = message.content
        task_id = content.get("task_id", "")
        workflow_id = content.get("workflow_id", "")
        error = content.get("error", "Unknown error")
        
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            
            # Find and update task
            for task in workflow["tasks"]:
                if task["id"] == task_id:
                    task["status"] = "failed"
                    task["completed_at"] = datetime.utcnow().isoformat()
                    task["error"] = error
                    break
            
            # Remove from agent assignments
            self.agent_assignments.pop(task_id, None)
            
            self.logger.error(f"Task {task_id} failed in workflow {workflow_id}: {error}")
    
    async def _handle_workflow_status(self, message: AgentMessage):
        """Handle workflow status request"""
        content = message.content
        workflow_id = content.get("workflow_id", "")
        
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            
            await self.send_message(
                message.sender_id,
                "workflow_status_response",
                {
                    "workflow_id": workflow_id,
                    "workflow": workflow
                }
            )
        else:
            await self.send_message(
                message.sender_id,
                "workflow_status_response",
                {
                    "workflow_id": workflow_id,
                    "error": "Workflow not found"
                }
            )
    
    async def _handle_cancel_workflow(self, message: AgentMessage):
        """Handle workflow cancellation request"""
        content = message.content
        workflow_id = content.get("workflow_id", "")
        
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow["status"] = "cancelled"
            workflow["completed_at"] = datetime.utcnow().isoformat()
            
            # Cancel running tasks
            for task in workflow["tasks"]:
                if task.get("status") == "running":
                    task["status"] = "cancelled"
                    
                    # Notify assigned agent
                    agent_id = self.agent_assignments.get(task["id"])
                    if agent_id:
                        await self.send_message(
                            agent_id,
                            "cancel_task",
                            {"task_id": task["id"]}
                        )
            
            await self.send_message(
                message.sender_id,
                "workflow_cancelled",
                {"workflow_id": workflow_id, "success": True}
            )
            
            self.logger.info(f"Cancelled workflow {workflow_id}")
        else:
            await self.send_message(
                message.sender_id,
                "workflow_cancelled",
                {"workflow_id": workflow_id, "success": False, "error": "Workflow not found"}
            )
    
    async def on_message_received(self, message: AgentMessage):
        """Handle unknown message types"""
        self.logger.warning(f"Received unknown message type: {message.message_type}")
    
    async def on_shutdown(self):
        """Cleanup when shutting down"""
        self.logger.info("Orchestrator Agent shutting down")
        
        # Cancel all active workflows
        for workflow_id, workflow in self.active_workflows.items():
            if workflow["status"] == "running":
                workflow["status"] = "cancelled"
                workflow["completed_at"] = datetime.utcnow().isoformat()
    
    # Public query methods
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        return self.workflow_stats.copy()
    
    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get all active workflows"""
        return self.active_workflows.copy()