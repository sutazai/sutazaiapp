#!/usr/bin/env python3
"""
Practical workflow examples for SutazAI task automation platform
Demonstrates real-world use cases without fantasy elements
"""

import asyncio
import httpx
from typing import Dict, List, Any
from datetime import datetime
import json

class SutazAIWorkflows:
    """Practical workflow implementations"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def code_review_workflow(self, 
                                  repository_path: str,
                                  branch: str = "main") -> Dict[str, Any]:
        """
        Automated code review workflow
        
        Steps:
        1. Analyze code quality
        2. Check for security issues
        3. Validate test coverage
        4. Generate improvement suggestions
        """
        print("🔍 Starting code review workflow...")
        
        # Step 1: Code quality analysis
        quality_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/workflows/code-improvement",
            json={
                "path": repository_path,
                "branch": branch,
                "agents": ["code-generation-improver", "testing-qa-validator"]
            }
        )
        quality_data = quality_result.json()
        
        # Step 2: Security analysis
        security_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/workflows/security-scan",
            json={
                "path": repository_path,
                "agents": ["semgrep-security-analyzer", "kali-security-specialist"]
            }
        )
        security_data = security_result.json()
        
        # Step 3: Test coverage
        test_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/workflows/test-coverage",
            json={
                "path": repository_path,
                "agents": ["testing-qa-validator"]
            }
        )
        test_data = test_result.json()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "repository": repository_path,
            "branch": branch,
            "quality_issues": quality_data.get("issues", []),
            "security_vulnerabilities": security_data.get("vulnerabilities", []),
            "test_coverage": test_data.get("coverage", 0),
            "recommendations": self._generate_recommendations(
                quality_data, security_data, test_data
            )
        }
    
    async def deployment_pipeline(self,
                                 service_name: str,
                                 environment: str = "staging") -> Dict[str, Any]:
        """
        Automated deployment pipeline
        
        Steps:
        1. Run tests
        2. Build containers
        3. Deploy to environment
        4. Validate deployment
        """
        print(f"🚀 Starting deployment pipeline for {service_name} to {environment}...")
        
        # Step 1: Run tests
        test_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "testing-qa-validator",
                "task": "run_all_tests",
                "params": {"service": service_name}
            }
        )
        
        if not test_result.json().get("success"):
            return {"status": "failed", "reason": "tests_failed", "details": test_result.json()}
        
        # Step 2: Build container
        build_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "infrastructure-devops-manager",
                "task": "build_container",
                "params": {"service": service_name, "tag": f"{environment}-latest"}
            }
        )
        
        # Step 3: Deploy
        deploy_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "deployment-automation-master",
                "task": "deploy_service",
                "params": {
                    "service": service_name,
                    "environment": environment,
                    "strategy": "rolling"
                }
            }
        )
        
        # Step 4: Validate
        validation_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "testing-qa-validator",
                "task": "validate_deployment",
                "params": {"service": service_name, "environment": environment}
            }
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "service": service_name,
            "environment": environment,
            "status": "success" if validation_result.json().get("healthy") else "failed",
            "deployment_id": deploy_result.json().get("deployment_id"),
            "health_checks": validation_result.json().get("checks", [])
        }
    
    async def documentation_generation(self,
                                     project_path: str,
                                     output_format: str = "markdown") -> Dict[str, Any]:
        """
        Automated documentation generation
        
        Steps:
        1. Analyze codebase
        2. Extract API schemas
        3. Generate documentation
        4. Create examples
        """
        print("📚 Generating documentation...")
        
        # Step 1: Analyze codebase
        analysis_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "document-knowledge-manager",
                "task": "analyze_codebase",
                "params": {"path": project_path}
            }
        )
        
        # Step 2: Extract API schemas
        api_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "senior-backend-developer",
                "task": "extract_api_schemas",
                "params": {"path": project_path}
            }
        )
        
        # Step 3: Generate documentation
        doc_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "document-knowledge-manager",
                "task": "generate_docs",
                "params": {
                    "analysis": analysis_result.json(),
                    "apis": api_result.json(),
                    "format": output_format
                }
            }
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "project": project_path,
            "format": output_format,
            "files_generated": doc_result.json().get("files", []),
            "api_endpoints": api_result.json().get("endpoints", []),
            "coverage": doc_result.json().get("coverage", 0)
        }
    
    async def performance_optimization(self,
                                     service_name: str,
                                     metrics_window: str = "1h") -> Dict[str, Any]:
        """
        Performance optimization workflow
        
        Steps:
        1. Collect performance metrics
        2. Identify bottlenecks
        3. Generate optimization suggestions
        4. Apply optimizations
        """
        print("⚡ Running performance optimization...")
        
        # Step 1: Collect metrics
        metrics_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "hardware-resource-optimizer",
                "task": "collect_metrics",
                "params": {"service": service_name, "window": metrics_window}
            }
        )
        
        # Step 2: Identify bottlenecks
        analysis_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "code-generation-improver",
                "task": "analyze_performance",
                "params": {"metrics": metrics_result.json()}
            }
        )
        
        # Step 3: Generate suggestions
        suggestions_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "senior-backend-developer",
                "task": "optimize_code",
                "params": {"bottlenecks": analysis_result.json()}
            }
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "service": service_name,
            "current_performance": metrics_result.json().get("summary", {}),
            "bottlenecks": analysis_result.json().get("bottlenecks", []),
            "optimizations": suggestions_result.json().get("suggestions", []),
            "estimated_improvement": suggestions_result.json().get("improvement", "0%")
        }
    
    async def multi_agent_data_pipeline(self,
                                       data_source: str,
                                       transformations: List[str]) -> Dict[str, Any]:
        """
        Multi-agent data processing pipeline
        
        Steps:
        1. Validate data source
        2. Apply transformations
        3. Quality check
        4. Store results
        """
        print("🔄 Running data pipeline...")
        
        # Coordinate multiple agents
        coordinator_result = await self.client.post(
            f"{self.api_url}/api/v1/agents/execute",
            json={
                "agent": "ai-agent-orchestrator",
                "task": "coordinate_pipeline",
                "params": {
                    "pipeline": [
                        {
                            "agent": "private-data-analyst",
                            "task": "validate_source",
                            "params": {"source": data_source}
                        },
                        {
                            "agent": "senior-ai-engineer",
                            "task": "apply_transformations",
                            "params": {"transformations": transformations}
                        },
                        {
                            "agent": "testing-qa-validator",
                            "task": "validate_output",
                            "params": {"quality_checks": ["completeness", "accuracy"]}
                        }
                    ]
                }
            }
        )
        
        return coordinator_result.json()
    
    def _generate_recommendations(self, quality, security, test) -> List[str]:
        """Generate actionable recommendations from analysis results"""
        recommendations = []
        
        if quality.get("issues", []):
            recommendations.append(f"Fix {len(quality['issues'])} code quality issues")
        
        if security.get("vulnerabilities", []):
            high_priority = [v for v in security["vulnerabilities"] if v.get("severity") == "high"]
            if high_priority:
                recommendations.append(f"Address {len(high_priority)} high-priority security vulnerabilities")
        
        coverage = test.get("coverage", 0)
        if coverage < 80:
            recommendations.append(f"Increase test coverage from {coverage}% to at least 80%")
        
        return recommendations


async def main():
    """Example usage of workflows"""
    workflows = SutazAIWorkflows()
    
    # Example 1: Code review
    print("\n=== Code Review Workflow ===")
    review_result = await workflows.code_review_workflow("/opt/sutazaiapp")
    print(f"Found {len(review_result['quality_issues'])} quality issues")
    print(f"Found {len(review_result['security_vulnerabilities'])} security issues")
    print(f"Test coverage: {review_result['test_coverage']}%")
    
    # Example 2: Deployment
    print("\n=== Deployment Pipeline ===")
    deploy_result = await workflows.deployment_pipeline("backend-api", "staging")
    print(f"Deployment status: {deploy_result['status']}")
    
    # Example 3: Documentation
    print("\n=== Documentation Generation ===")
    doc_result = await workflows.documentation_generation("/opt/sutazaiapp")
    print(f"Generated {len(doc_result['files_generated'])} documentation files")
    
    # Example 4: Performance optimization
    print("\n=== Performance Optimization ===")
    perf_result = await workflows.performance_optimization("backend-api")
    print(f"Found {len(perf_result['bottlenecks'])} performance bottlenecks")
    print(f"Estimated improvement: {perf_result['estimated_improvement']}")
    
    # Example 5: Data pipeline
    print("\n=== Multi-Agent Data Pipeline ===")
    pipeline_result = await workflows.multi_agent_data_pipeline(
        "database://production",
        ["normalize", "aggregate", "anonymize"]
    )
    print(f"Pipeline status: {pipeline_result.get('status', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())