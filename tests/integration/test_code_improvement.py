#!/usr/bin/env python3
"""
Test script for code improvement workflow
Demonstrates analyzing the backend/app directory and getting improvement suggestions
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import asyncio
import httpx
import json
import time
from datetime import datetime


async def test_workflow():
    """Test the code improvement workflow via API"""
    
    base_url = "http://localhost:8000"
    
    # Test 1: List available agents
    logger.info("=== Testing Agent List ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/api/v1/agents/")
        if response.status_code == 200:
            agents = response.json()
            logger.info(f"Found {agents['active_count']} active agents:")
            for agent in agents['agents']:
                logger.info(f"  - {agent['name']}: {', '.join(agent['capabilities'])}")
        else:
            logger.error(f"Error listing agents: {response.status_code}")
    
    logger.info("\n=== Testing Code Improvement Workflow ===")
    
    # Test 2: Start code improvement workflow
    workflow_request = {
        "directory": "/opt/sutazaiapp/backend/app"
    }
    
    async with httpx.AsyncClient() as client:
        logger.info(f"Starting analysis of: {workflow_request['directory']}")
        response = await client.post(
            f"{base_url}/api/v1/agents/workflows/code-improvement",
            json=workflow_request
        )
        
        if response.status_code == 200:
            result = response.json()
            workflow_id = result['workflow_id']
            logger.info(f"Workflow started: {workflow_id}")
            logger.info(f"Status: {result['status']}")
            logger.info(f"Message: {result['message']}")
            
            # Test 3: Poll for workflow status
            logger.info("\nMonitoring workflow progress...")
            max_attempts = 30
            attempt = 0
            
            while attempt < max_attempts:
                await asyncio.sleep(2)
                status_response = await client.get(
                    f"{base_url}/api/v1/agents/workflows/{workflow_id}"
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    logger.info(f"  Status: {status['status']}", end="")
                    
                    if status['status'] == 'completed':
                        logger.info(f"\n  Completed at: {status['completed_at']}")
                        logger.info(f"\n  Summary:")
                        summary = status.get('summary', {})
                        logger.info(f"    Total issues: {summary.get('total_issues', 0)}")
                        logger.error(f"    Critical issues: {summary.get('critical_issues', 0)}")
                        logger.info(f"    Improvements suggested: {summary.get('improvements', 0)}")
                        
                        metrics = summary.get('metrics', {})
                        logger.info(f"\n  Metrics:")
                        logger.info(f"    Lines of code: {metrics.get('lines_of_code', 0):,}")
                        logger.info(f"    Complexity score: {metrics.get('complexity_score', 0):.2f}")
                        logger.info(f"    Security issues: {metrics.get('security_issues', 0)}")
                        logger.info(f"    Performance issues: {metrics.get('performance_issues', 0)}")
                        
                        # Test 4: Get the full report
                        logger.info("\n=== Fetching Full Report ===")
                        report_response = await client.get(
                            f"{base_url}/api/v1/agents/workflows/{workflow_id}/report"
                        )
                        
                        if report_response.status_code == 200:
                            report = report_response.json()
                            
                            # Display markdown report preview
                            markdown_report = report['markdown_report']
                            logger.info("\nReport Preview (first 1000 chars):")
                            logger.info("-" * 50)
                            logger.info(markdown_report[:1000])
                            logger.info("-" * 50)
                            
                            # Display top issues from JSON report
                            json_report = report.get('json_report')
                            if json_report and 'issues' in json_report:
                                logger.info(f"\nTop 5 Issues Found:")
                                for i, issue in enumerate(json_report['issues'][:5], 1):
                                    logger.info(f"\n{i}. [{issue['severity'].upper()}] {issue['description']}")
                                    logger.info(f"   File: {issue['file']}:{issue['line']}")
                                    logger.info(f"   Type: {issue['type']}")
                                    if issue.get('fix'):
                                        logger.info(f"   Fix: {issue['fix']}")
                        break
                    
                    elif status['status'] == 'failed':
                        logger.error(f"\n  Workflow failed: {status.get('error', 'Unknown error')}")
                        break
                    else:
                        logger.info(f" (attempt {attempt + 1}/{max_attempts})")
                
                attempt += 1
            
            if attempt >= max_attempts:
                logger.info("\nWorkflow timed out")
        
        else:
            logger.error(f"Error starting workflow: {response.status_code}")
            logger.info(response.text)
    
    # Test 5: Test agent consensus
    logger.info("\n\n=== Testing Agent Consensus ===")
    consensus_request = {
        "query": "Should we refactor the authentication module to use JWT tokens?",
        "agents": ["senior-backend-developer", "security-pentesting-specialist", "testing-qa-validator"]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/api/v1/agents/consensus",
            json=consensus_request
        )
        
        if response.status_code == 200:
            consensus = response.json()
            logger.info(f"Query: {consensus['query']}")
            logger.info(f"Agents consulted: {', '.join(consensus['agents_consulted'])}")
            logger.info(f"Consensus reached: {consensus['consensus']['agreed']}")
            logger.info(f"Confidence: {consensus['consensus']['confidence']}")
            logger.info(f"Recommendation: {consensus['recommendation']}")
    
    # Test 6: Test task delegation
    logger.info("\n\n=== Testing Task Delegation ===")
    delegation_request = {
        "task": {
            "type": "security",
            "description": "Perform security audit on authentication endpoints",
            "priority": "high"
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/api/v1/agents/delegate",
            json=delegation_request
        )
        
        if response.status_code == 200:
            delegation = response.json()
            logger.info(f"Task ID: {delegation['task_id']}")
            logger.info(f"Delegated to: {delegation['delegated_to']}")
            logger.info(f"Status: {delegation['status']}")
            logger.info(f"Estimated completion: {delegation['estimated_completion']}")


async def test_direct_workflow():
    """Test the workflow directly without API"""
    logger.info("\n\n=== Testing Direct Workflow Execution ===")
    
    from workflows.code_improvement_workflow import CodeImprovementWorkflow
    
    workflow = CodeImprovementWorkflow()
    await workflow.initialize()
    
    # Analyze a small directory for faster results
    test_dir = "/opt/sutazaiapp/backend/app/api/v1/endpoints"
    logger.info(f"Analyzing directory: {test_dir}")
    
    report = await workflow.analyze_directory(test_dir)
    
    logger.info(f"\nAnalysis Complete!")
    logger.info(f"Total issues found: {len(report.issues)}")
    logger.info(f"Lines of code analyzed: {report.metrics.lines_of_code:,}")
    logger.info(f"Complexity score: {report.metrics.complexity_score:.2f}")
    
    # Show top 3 issues
    logger.info("\nTop 3 Issues:")
    for i, issue in enumerate(report.issues[:3], 1):
        logger.info(f"\n{i}. [{issue.severity.upper()}] {issue.description}")
        logger.info(f"   File: {issue.file_path}:{issue.line_number}")
        logger.info(f"   Found by: {issue.agent}")
        if issue.suggested_fix:
            logger.info(f"   Fix: {issue.suggested_fix}")
    
    # Save test report
    output_file = "/opt/sutazaiapp/data/test_improvement_report.md"
    workflow.save_report(report, output_file)
    logger.info(f"\nTest report saved to: {output_file}")


async def main():
    """Main test function"""
    logger.info("Code Improvement Workflow Test")
    logger.info("=" * 50)
    
    # Check if backend is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                logger.info("Backend is running, testing via API...")
                await test_workflow()
            else:
                logger.info("Backend not responding, testing direct workflow...")
                await test_direct_workflow()
    except Exception as e:
        logger.error(f"Unexpected exception: {e}", exc_info=True)
        logger.info("Backend not available, testing direct workflow...")
        await test_direct_workflow()


if __name__ == "__main__":
