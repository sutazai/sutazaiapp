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
    print("=== Testing Agent List ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/api/v1/agents/")
        if response.status_code == 200:
            agents = response.json()
            print(f"Found {agents['active_count']} active agents:")
            for agent in agents['agents']:
                print(f"  - {agent['name']}: {', '.join(agent['capabilities'])}")
        else:
            print(f"Error listing agents: {response.status_code}")
    
    print("\n=== Testing Code Improvement Workflow ===")
    
    # Test 2: Start code improvement workflow
    workflow_request = {
        "directory": "/opt/sutazaiapp/backend/app"
    }
    
    async with httpx.AsyncClient() as client:
        print(f"Starting analysis of: {workflow_request['directory']}")
        response = await client.post(
            f"{base_url}/api/v1/agents/workflows/code-improvement",
            json=workflow_request
        )
        
        if response.status_code == 200:
            result = response.json()
            workflow_id = result['workflow_id']
            print(f"Workflow started: {workflow_id}")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            
            # Test 3: Poll for workflow status
            print("\nMonitoring workflow progress...")
            max_attempts = 30
            attempt = 0
            
            while attempt < max_attempts:
                await asyncio.sleep(2)
                status_response = await client.get(
                    f"{base_url}/api/v1/agents/workflows/{workflow_id}"
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"  Status: {status['status']}", end="")
                    
                    if status['status'] == 'completed':
                        print(f"\n  Completed at: {status['completed_at']}")
                        print(f"\n  Summary:")
                        summary = status.get('summary', {})
                        print(f"    Total issues: {summary.get('total_issues', 0)}")
                        print(f"    Critical issues: {summary.get('critical_issues', 0)}")
                        print(f"    Improvements suggested: {summary.get('improvements', 0)}")
                        
                        metrics = summary.get('metrics', {})
                        print(f"\n  Metrics:")
                        print(f"    Lines of code: {metrics.get('lines_of_code', 0):,}")
                        print(f"    Complexity score: {metrics.get('complexity_score', 0):.2f}")
                        print(f"    Security issues: {metrics.get('security_issues', 0)}")
                        print(f"    Performance issues: {metrics.get('performance_issues', 0)}")
                        
                        # Test 4: Get the full report
                        print("\n=== Fetching Full Report ===")
                        report_response = await client.get(
                            f"{base_url}/api/v1/agents/workflows/{workflow_id}/report"
                        )
                        
                        if report_response.status_code == 200:
                            report = report_response.json()
                            
                            # Display markdown report preview
                            markdown_report = report['markdown_report']
                            print("\nReport Preview (first 1000 chars):")
                            print("-" * 50)
                            print(markdown_report[:1000])
                            print("-" * 50)
                            
                            # Display top issues from JSON report
                            json_report = report.get('json_report')
                            if json_report and 'issues' in json_report:
                                print(f"\nTop 5 Issues Found:")
                                for i, issue in enumerate(json_report['issues'][:5], 1):
                                    print(f"\n{i}. [{issue['severity'].upper()}] {issue['description']}")
                                    print(f"   File: {issue['file']}:{issue['line']}")
                                    print(f"   Type: {issue['type']}")
                                    if issue.get('fix'):
                                        print(f"   Fix: {issue['fix']}")
                        break
                    
                    elif status['status'] == 'failed':
                        print(f"\n  Workflow failed: {status.get('error', 'Unknown error')}")
                        break
                    else:
                        print(f" (attempt {attempt + 1}/{max_attempts})")
                
                attempt += 1
            
            if attempt >= max_attempts:
                print("\nWorkflow timed out")
        
        else:
            print(f"Error starting workflow: {response.status_code}")
            print(response.text)
    
    # Test 5: Test agent consensus
    print("\n\n=== Testing Agent Consensus ===")
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
            print(f"Query: {consensus['query']}")
            print(f"Agents consulted: {', '.join(consensus['agents_consulted'])}")
            print(f"Consensus reached: {consensus['consensus']['agreed']}")
            print(f"Confidence: {consensus['consensus']['confidence']}")
            print(f"Recommendation: {consensus['recommendation']}")
    
    # Test 6: Test task delegation
    print("\n\n=== Testing Task Delegation ===")
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
            print(f"Task ID: {delegation['task_id']}")
            print(f"Delegated to: {delegation['delegated_to']}")
            print(f"Status: {delegation['status']}")
            print(f"Estimated completion: {delegation['estimated_completion']}")


async def test_direct_workflow():
    """Test the workflow directly without API"""
    print("\n\n=== Testing Direct Workflow Execution ===")
    
    from workflows.code_improvement_workflow import CodeImprovementWorkflow
    
    workflow = CodeImprovementWorkflow()
    await workflow.initialize()
    
    # Analyze a small directory for faster results
    test_dir = "/opt/sutazaiapp/backend/app/api/v1/endpoints"
    print(f"Analyzing directory: {test_dir}")
    
    report = await workflow.analyze_directory(test_dir)
    
    print(f"\nAnalysis Complete!")
    print(f"Total issues found: {len(report.issues)}")
    print(f"Lines of code analyzed: {report.metrics.lines_of_code:,}")
    print(f"Complexity score: {report.metrics.complexity_score:.2f}")
    
    # Show top 3 issues
    print("\nTop 3 Issues:")
    for i, issue in enumerate(report.issues[:3], 1):
        print(f"\n{i}. [{issue.severity.upper()}] {issue.description}")
        print(f"   File: {issue.file_path}:{issue.line_number}")
        print(f"   Found by: {issue.agent}")
        if issue.suggested_fix:
            print(f"   Fix: {issue.suggested_fix}")
    
    # Save test report
    output_file = "/opt/sutazaiapp/data/test_improvement_report.md"
    workflow.save_report(report, output_file)
    print(f"\nTest report saved to: {output_file}")


async def main():
    """Main test function"""
    print("Code Improvement Workflow Test")
    print("=" * 50)
    
    # Check if backend is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("Backend is running, testing via API...")
                await test_workflow()
            else:
                print("Backend not responding, testing direct workflow...")
                await test_direct_workflow()
    except Exception as e:
        # TODO: Review this exception handling
        logger.error(f"Unexpected exception: {e}", exc_info=True)
        print("Backend not available, testing direct workflow...")
        await test_direct_workflow()


if __name__ == "__main__":
    asyncio.run(main())