#!/usr/bin/env python3
"""
Comprehensive Testing Suite for SutazAI AGI/ASI System
Tests all components and integrations
"""

import asyncio
import httpx
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

class AGISystemTester:
    def __init__(self, base_urls: Dict[str, str] = None):
        """Initialize the tester with service URLs"""
        self.base_urls = base_urls or {
            "orchestrator": "http://localhost:8200",
            "ollama": "http://localhost:11434",
            "litellm": "http://localhost:4000",
            "chromadb": "http://localhost:8000",
            "faiss": "http://localhost:8100",
            "letta": "http://localhost:8283",
            "autogpt": "http://localhost:8080",
            "localagi": "http://localhost:8090",
            "tabbyml": "http://localhost:8085",
            "semgrep": "http://localhost:8087",
            "langchain": "http://localhost:8095",
            "streamlit": "http://localhost:8501"
        }
        self.results = {}
        self.client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def test_service_health(self, service_name: str, url: str) -> Dict[str, Any]:
        """Test if a service is healthy"""
        try:
            start_time = time.time()
            response = await self.client.get(f"{url}/health")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return {
                    "status": "✅ Healthy",
                    "response_time": f"{response_time:.2f}s",
                    "details": response.json()
                }
            else:
                return {
                    "status": "❌ Unhealthy",
                    "response_time": f"{response_time:.2f}s",
                    "error": f"Status code: {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "❌ Unreachable",
                "response_time": "N/A",
                "error": str(e)
            }

    async def test_all_services_health(self) -> Dict[str, Any]:
        """Test health of all services"""
        console.print("\n[bold cyan]🏥 Testing Service Health...[/bold cyan]")
        
        health_results = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Checking services...", total=len(self.base_urls))
            
            for service_name, url in self.base_urls.items():
                health_results[service_name] = await self.test_service_health(service_name, url)
                progress.advance(task)
        
        # Display results in a table
        table = Table(title="Service Health Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status")
        table.add_column("Response Time", style="yellow")
        table.add_column("Details", style="dim")
        
        for service, result in health_results.items():
            details = ""
            if "details" in result:
                if isinstance(result["details"], dict):
                    details = json.dumps(result["details"], indent=2)[:50] + "..."
                else:
                    details = str(result["details"])[:50]
            elif "error" in result:
                details = result["error"][:50]
            
            table.add_row(
                service,
                result["status"],
                result["response_time"],
                details
            )
        
        console.print(table)
        return health_results

    async def test_model_inference(self) -> Dict[str, Any]:
        """Test model inference capabilities"""
        console.print("\n[bold cyan]🤖 Testing Model Inference...[/bold cyan]")
        
        test_prompts = [
            {
                "model": "deepseek-r1",
                "prompt": "What is 2+2?",
                "expected_contains": ["4", "four"]
            },
            {
                "model": "codellama",
                "prompt": "Write a Python function to add two numbers",
                "expected_contains": ["def", "return", "+"]
            }
        ]
        
        results = {}
        
        # Test through LiteLLM
        for test in test_prompts:
            try:
                response = await self.client.post(
                    f"{self.base_urls['litellm']}/chat/completions",
                    json={
                        "model": test["model"],
                        "messages": [{"role": "user", "content": test["prompt"]}],
                        "max_tokens": 100
                    },
                    headers={"Authorization": "Bearer sk-sutazai-local-key"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Check if expected content is present
                    success = any(expected.lower() in content.lower() 
                                for expected in test["expected_contains"])
                    
                    results[test["model"]] = {
                        "status": "✅ Success" if success else "⚠️  Partial",
                        "response": content[:100] + "..." if len(content) > 100 else content
                    }
                else:
                    results[test["model"]] = {
                        "status": "❌ Failed",
                        "error": f"Status: {response.status_code}"
                    }
            except Exception as e:
                results[test["model"]] = {
                    "status": "❌ Error",
                    "error": str(e)
                }
        
        # Display results
        for model, result in results.items():
            console.print(f"\n[bold]{model}:[/bold]")
            console.print(f"  Status: {result['status']}")
            if "response" in result:
                console.print(f"  Response: {result['response']}")
            if "error" in result:
                console.print(f"  Error: [red]{result['error']}[/red]")
        
        return results

    async def test_vector_operations(self) -> Dict[str, Any]:
        """Test vector database operations"""
        console.print("\n[bold cyan]🔍 Testing Vector Operations...[/bold cyan]")
        
        results = {}
        
        # Test vector for indexing and search
        test_vector = [0.1] * 768  # 768-dimensional vector
        test_id = f"test_vector_{int(time.time())}"
        
        # Test FAISS
        try:
            # Index a vector
            response = await self.client.post(
                f"{self.base_urls['faiss']}/index",
                json={"id": test_id, "vector": test_vector}
            )
            
            if response.status_code == 200:
                # Search for similar vectors
                search_response = await self.client.post(
                    f"{self.base_urls['faiss']}/search",
                    json={"vector": test_vector, "k": 5}
                )
                
                if search_response.status_code == 200:
                    search_results = search_response.json()
                    results["faiss"] = {
                        "status": "✅ Success",
                        "indexed": True,
                        "search_results": len(search_results.get("results", []))
                    }
                else:
                    results["faiss"] = {"status": "⚠️  Partial", "error": "Search failed"}
            else:
                results["faiss"] = {"status": "❌ Failed", "error": "Indexing failed"}
        except Exception as e:
            results["faiss"] = {"status": "❌ Error", "error": str(e)}
        
        # Test ChromaDB
        try:
            # Get or create collection
            response = await self.client.post(
                f"{self.base_urls['chromadb']}/api/v1/collections",
                json={"name": "test_collection", "get_or_create": True}
            )
            
            if response.status_code in [200, 201]:
                results["chromadb"] = {
                    "status": "✅ Success",
                    "collection_created": True
                }
            else:
                results["chromadb"] = {"status": "❌ Failed", "error": f"Status: {response.status_code}"}
        except Exception as e:
            results["chromadb"] = {"status": "❌ Error", "error": str(e)}
        
        # Display results
        for db, result in results.items():
            console.print(f"\n[bold]{db}:[/bold]")
            console.print(f"  Status: {result['status']}")
            for key, value in result.items():
                if key != "status":
                    console.print(f"  {key}: {value}")
        
        return results

    async def test_agent_capabilities(self) -> Dict[str, Any]:
        """Test AI agent capabilities"""
        console.print("\n[bold cyan]🤖 Testing AI Agents...[/bold cyan]")
        
        results = {}
        
        # Test LangChain orchestrator
        try:
            response = await self.client.post(
                f"{self.base_urls['langchain']}/execute",
                json={
                    "task": "Create a simple task plan for building a web app",
                    "tools": ["Task_Planning"]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                results["langchain"] = {
                    "status": "✅ Success",
                    "result": result.get("result", "")[:100] + "..."
                }
            else:
                results["langchain"] = {"status": "❌ Failed", "error": f"Status: {response.status_code}"}
        except Exception as e:
            results["langchain"] = {"status": "❌ Error", "error": str(e)}
        
        # Test Semgrep code analysis
        try:
            test_code = """
def insecure_function(user_input):
    # Potential SQL injection
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return query
"""
            response = await self.client.post(
                f"{self.base_urls['semgrep']}/analyze/code",
                json={
                    "code": test_code,
                    "language": "python"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                findings = result.get("total_findings", 0)
                results["semgrep"] = {
                    "status": "✅ Success",
                    "findings": findings,
                    "summary": result.get("summary", {})
                }
            else:
                results["semgrep"] = {"status": "❌ Failed", "error": f"Status: {response.status_code}"}
        except Exception as e:
            results["semgrep"] = {"status": "❌ Error", "error": str(e)}
        
        # Display results
        for agent, result in results.items():
            console.print(f"\n[bold]{agent}:[/bold]")
            console.print(f"  Status: {result['status']}")
            for key, value in result.items():
                if key != "status":
                    console.print(f"  {key}: {value}")
        
        return results

    async def test_orchestrator_integration(self) -> Dict[str, Any]:
        """Test the AGI orchestrator integration"""
        console.print("\n[bold cyan]🎭 Testing Orchestrator Integration...[/bold cyan]")
        
        test_tasks = [
            {
                "name": "Code Generation",
                "request": {
                    "task_type": "code_generation",
                    "prompt": "Write a Python function to calculate fibonacci numbers",
                    "agents": ["litellm", "tabbyml"]
                }
            },
            {
                "name": "Code Analysis",
                "request": {
                    "task_type": "code_analysis",
                    "prompt": "Analyze this code for security issues: def get_user(id): return db.query(f'SELECT * FROM users WHERE id={id}')",
                    "agents": ["semgrep", "langchain"]
                }
            }
        ]
        
        results = {}
        
        for task in test_tasks:
            try:
                response = await self.client.post(
                    f"{self.base_urls['orchestrator']}/execute",
                    json=task["request"]
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results[task["name"]] = {
                        "status": "✅ Success",
                        "agents_used": result.get("agents_used", []),
                        "has_result": bool(result.get("result"))
                    }
                else:
                    results[task["name"]] = {
                        "status": "❌ Failed",
                        "error": f"Status: {response.status_code}"
                    }
            except Exception as e:
                results[task["name"]] = {
                    "status": "❌ Error",
                    "error": str(e)
                }
        
        # Display results
        table = Table(title="Orchestrator Integration Tests")
        table.add_column("Task", style="cyan")
        table.add_column("Status")
        table.add_column("Agents Used")
        table.add_column("Result")
        
        for task_name, result in results.items():
            agents = ", ".join(result.get("agents_used", [])) if "agents_used" in result else "N/A"
            has_result = "✅" if result.get("has_result") else "❌"
            
            table.add_row(
                task_name,
                result["status"],
                agents,
                has_result
            )
        
        console.print(table)
        return results

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate report"""
        console.print("[bold green]🚀 Starting AGI/ASI System Tests[/bold green]")
        console.print(f"[dim]Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n")
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Run tests
        all_results["tests"]["health"] = await self.test_all_services_health()
        all_results["tests"]["inference"] = await self.test_model_inference()
        all_results["tests"]["vectors"] = await self.test_vector_operations()
        all_results["tests"]["agents"] = await self.test_agent_capabilities()
        all_results["tests"]["orchestrator"] = await self.test_orchestrator_integration()
        
        # Calculate summary
        total_tests = 0
        passed_tests = 0
        
        for test_category, results in all_results["tests"].items():
            for service, result in results.items():
                total_tests += 1
                if isinstance(result, dict) and result.get("status", "").startswith("✅"):
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Display summary
        console.print("\n[bold green]📊 Test Summary[/bold green]")
        console.print(f"Total Tests: {total_tests}")
        console.print(f"Passed: {passed_tests}")
        console.print(f"Failed: {total_tests - passed_tests}")
        console.print(f"Success Rate: {success_rate:.1f}%")
        
        # Generate report
        report_path = f"agi_test_report_{int(time.time())}.json"
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        
        console.print(f"\n[dim]Full report saved to: {report_path}[/dim]")
        
        return all_results

async def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description="Test AGI/ASI System")
    parser.add_argument("--orchestrator-url", default="http://localhost:8200",
                       help="AGI Orchestrator URL")
    parser.add_argument("--test", choices=["health", "inference", "vectors", "agents", "orchestrator", "all"],
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    async with AGISystemTester() as tester:
        if args.orchestrator_url != "http://localhost:8200":
            tester.base_urls["orchestrator"] = args.orchestrator_url
        
        if args.test == "all":
            await tester.run_all_tests()
        elif args.test == "health":
            await tester.test_all_services_health()
        elif args.test == "inference":
            await tester.test_model_inference()
        elif args.test == "vectors":
            await tester.test_vector_operations()
        elif args.test == "agents":
            await tester.test_agent_capabilities()
        elif args.test == "orchestrator":
            await tester.test_orchestrator_integration()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Test failed with error: {e}[/red]")
        sys.exit(1)