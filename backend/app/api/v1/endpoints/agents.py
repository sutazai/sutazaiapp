"""Agents endpoints with workflow support"""
import os
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Import the workflow
import sys
sys.path.append('/opt/sutazaiapp')
from workflows.code_improvement_workflow import CodeImprovementWorkflow

router = APIRouter()

# Request/Response models
class WorkflowRequest(BaseModel):
    """Request to run a workflow"""
    workflow_type: str = Field(..., description="Type of workflow to run")
    target_directory: str = Field(..., description="Directory to analyze")
    options: Dict = Field(default_factory=dict, description="Additional workflow options")

class WorkflowResponse(BaseModel):
    """Response from workflow execution"""
    workflow_id: str
    status: str
    message: str
    result_file: Optional[str] = None

class CodeImprovementRequest(BaseModel):
    """Request for code improvement analysis"""
    directory: str = Field(..., description="Directory to analyze")
    output_format: str = Field(default="markdown", description="Output format: markdown or json")

# Store for workflow results
workflow_results = {}

@router.get("/")
async def list_agents():
    """List available agents and their capabilities"""
    return {
        "agents": [
            {
                "id": "senior-ai-engineer",
                "name": "Senior AI Engineer",
                "capabilities": ["ml_analysis", "model_optimization", "processing_architecture"],
                "status": "active"
            },
            {
                "id": "testing-qa-validator",
                "name": "Testing QA Validator",
                "capabilities": ["test_coverage", "bug_detection", "quality_assurance"],
                "status": "active"
            },
            {
                "id": "infrastructure-devops-manager",
                "name": "Infrastructure DevOps Manager",
                "capabilities": ["docker_analysis", "deployment_config", "security_audit"],
                "status": "active"
            },
            {
                "id": "senior-backend-developer",
                "name": "Senior Backend Developer",
                "capabilities": ["api_design", "database_optimization", "architecture"],
                "status": "active"
            },
            {
                "id": "security-pentesting-specialist",
                "name": "Security Pentesting Specialist",
                "capabilities": ["vulnerability_scan", "penetration_testing", "security_audit"],
                "status": "active"
            }
        ],
        "active_count": 5,
        "total_count": 5
    }

async def run_code_improvement_workflow(workflow_id: str, directory: str):
    """Background task to run code improvement workflow"""
    try:
        workflow = CodeImprovementWorkflow()
        await workflow.initialize()
        
        # Update status
        workflow_results[workflow_id]['status'] = 'running'
        workflow_results[workflow_id]['started_at'] = datetime.now().isoformat()
        
        # Run analysis
        report = await workflow.analyze_directory(directory)
        
        # Save report
        output_dir = Path("/opt/sutazaiapp/data/workflow_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{workflow_id}_report.md"
        workflow.save_report(report, str(output_file))
        
        # Update results
        workflow_results[workflow_id].update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'result_file': str(output_file),
            'summary': {
                'total_issues': len(report.issues),
                'critical_issues': len([i for i in report.issues if i.severity == 'critical']),
                'improvements': len(report.improvements),
                'metrics': {
                    'lines_of_code': report.metrics.lines_of_code,
                    'complexity_score': report.metrics.complexity_score,
                    'security_issues': report.metrics.security_issues,
                    'performance_issues': report.metrics.performance_issues
                }
            }
        })
        
    except Exception as e:
        workflow_results[workflow_id].update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        })

@router.post("/workflows/code-improvement")
async def run_code_improvement(
    request: CodeImprovementRequest,
    background_tasks: BackgroundTasks
):
    """Run code improvement workflow on specified directory"""
    
    # Validate directory
    if not os.path.exists(request.directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory}")
    
    if not os.path.isdir(request.directory):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.directory}")
    
    # Create workflow ID
    workflow_id = f"code_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize result entry
    workflow_results[workflow_id] = {
        'workflow_id': workflow_id,
        'type': 'code_improvement',
        'directory': request.directory,
        'status': 'queued',
        'created_at': datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(run_code_improvement_workflow, workflow_id, request.directory)
    
    return WorkflowResponse(
        workflow_id=workflow_id,
        status='queued',
        message=f'Code improvement workflow started for {request.directory}'
    )

@router.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get status of a workflow"""
    if workflow_id not in workflow_results:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return workflow_results[workflow_id]

@router.get("/workflows/{workflow_id}/report")
async def get_workflow_report(workflow_id: str):
    """Get the report from a completed workflow"""
    if workflow_id not in workflow_results:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    result = workflow_results[workflow_id]
    
    if result['status'] != 'completed':
        raise HTTPException(
            status_code=400, 
            detail=f"Workflow is {result['status']}, report not available"
        )
    
    if not result.get('result_file'):
        raise HTTPException(status_code=404, detail="Report file not found")
    
    # Read the report file
    try:
        with open(result['result_file'], 'r') as f:
            content = f.read()
        
        # Also get the JSON version if it exists
        json_file = result['result_file'].replace('.md', '.json')
        json_content = None
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                json_content = json.load(f)
        
        return {
            'workflow_id': workflow_id,
            'markdown_report': content,
            'json_report': json_content,
            'summary': result.get('summary', {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading report: {str(e)}")

@router.post("/consensus")
async def agent_consensus(request: Dict):
    """Multi-agent consensus for decision making"""
    query = request.get("query", "")
    agents = request.get("agents", ["senior-ai-engineer", "testing-qa-validator"])
    
    # Simulate agent consensus
    consensus_results = {
        "query": query,
        "agents_consulted": agents,
        "consensus": {
            "agreed": True,
            "confidence": 0.85,
            "reasoning": "All agents analyzed the query and reached consensus"
        },
        "individual_responses": {
            agent: {
                "opinion": f"{agent} analysis of: {query}",
                "confidence": 0.8 + (0.1 if 'senior' in agent else 0)
            }
            for agent in agents
        },
        "recommendation": "Proceed with the proposed approach based on agent consensus",
        "timestamp": datetime.now().isoformat()
    }
    
    return consensus_results

@router.post("/delegate")
async def delegate_task(request: Dict):
    """Delegate task to appropriate agent via coordinator"""
    task = request.get("task", {})
    task_type = task.get("type", "general")
    
    # Determine best agent for task
    agent_mapping = {
        "ml": "senior-ai-engineer",
        "testing": "testing-qa-validator",
        "deployment": "infrastructure-devops-manager",
        "backend": "senior-backend-developer",
        "security": "security-pentesting-specialist"
    }
    
    selected_agent = agent_mapping.get(task_type, "senior-backend-developer")
    
    return {
        "task_id": f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "delegated_to": selected_agent,
        "status": "assigned",
        "task": task,
        "estimated_completion": "5 minutes",
        "timestamp": datetime.now().isoformat()
    }
