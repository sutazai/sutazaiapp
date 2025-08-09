#!/usr/bin/env python3
"""
ULTRA-DEEP System Analysis for SutazAI
Comprehensive verification of actual system state vs documentation claims
"""

import subprocess
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

class UltraDeepAnalyzer:
    def __init__(self):
        self.findings = {
            "timestamp": datetime.now().isoformat(),
            "containers": {},
            "services": {},
            "agents": {},
            "databases": {},
            "api_endpoints": {},
            "rule_violations": [],
            "critical_issues": [],
            "recommendations": []
        }
        
    def run_command(self, cmd: str, timeout: int = 5) -> Tuple[bool, str]:
        """Execute shell command and return success/output"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0, result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
            
    def test_endpoint(self, url: str, timeout: int = 3) -> Dict[str, Any]:
        """Test HTTP endpoint"""
        try:
            response = requests.get(url, timeout=timeout)
            return {
                "status": "reachable",
                "status_code": response.status_code,
                "response": response.text[:500] if response.text else None
            }
        except requests.exceptions.Timeout:
            return {"status": "timeout"}
        except requests.exceptions.ConnectionError:
            return {"status": "connection_error"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
            
    def analyze_containers(self):
        """Analyze Docker containers"""
        print("[1/7] Analyzing Docker containers...")
        
        # Count running containers
        success, output = self.run_command("docker ps --format '{{.Names}}' | wc -l")
        if success:
            self.findings["containers"]["running_count"] = int(output.strip())
            
        # Get container details
        success, output = self.run_command(
            "docker ps --format '{{json .}}' | jq -s '.'"
        )
        if success:
            try:
                containers = json.loads(output)
                for container in containers:
                    self.findings["containers"][container["Names"]] = {
                        "status": container.get("Status"),
                        "ports": container.get("Ports", ""),
                        "state": container.get("State", "")
                    }
            except:
                pass
                
    def analyze_services(self):
        """Test all service endpoints"""
        print("[2/7] Testing service endpoints...")
        
        services = {
            "backend": "http://127.0.0.1:10010/health",
            "frontend": "http://127.0.0.1:10011",
            "ollama": "http://127.0.0.1:10104/api/tags",
            "postgres": "postgresql://sutazai:sutazai_password@127.0.0.1:10000/sutazai",
            "redis": "redis://127.0.0.1:10001",
            "neo4j": "http://127.0.0.1:10002",
            "prometheus": "http://127.0.0.1:10200/-/healthy",
            "grafana": "http://127.0.0.1:10201/api/health",
            "chromadb": "http://127.0.0.1:10100/api/v1/heartbeat",
            "qdrant": "http://127.0.0.1:10101/collections"
        }
        
        for name, url in services.items():
            if url.startswith("http"):
                self.findings["services"][name] = self.test_endpoint(url)
            else:
                # Test non-HTTP services differently
                if "postgresql" in url:
                    success, _ = self.run_command(
                        "docker exec sutazai-postgres psql -U sutazai -d sutazai -c '\\dt' 2>/dev/null"
                    )
                    self.findings["services"][name] = {
                        "status": "reachable" if success else "error",
                        "has_tables": success
                    }
                elif "redis" in url:
                    success, _ = self.run_command(
                        "docker exec sutazai-redis redis-cli ping 2>/dev/null"
                    )
                    self.findings["services"][name] = {
                        "status": "reachable" if success else "error"
                    }
                    
    def analyze_agents(self):
        """Test agent endpoints"""
        print("[3/7] Testing agent services...")
        
        # Known agent ports from documentation
        agent_ports = {
            "task-assignment": 8551,
            "ai-orchestrator": 8589,
            "multi-agent-coordinator": 8587,
            "resource-arbitration": 8588,
            "hardware-optimizer": 8002,
            "ollama-integration": 11015,
            "metrics-exporter": 11063,
            "jarvis-automation": 11102,
            "jarvis-hardware": 11110
        }
        
        for name, port in agent_ports.items():
            health_url = f"http://127.0.0.1:{port}/health"
            process_url = f"http://127.0.0.1:{port}/process"
            
            self.findings["agents"][name] = {
                "port": port,
                "health": self.test_endpoint(health_url, timeout=2),
                "process": self.test_endpoint(process_url, timeout=2)
            }
            
    def analyze_api_endpoints(self):
        """Test backend API endpoints"""
        print("[4/7] Testing API endpoints...")
        
        base_url = "http://127.0.0.1:10010"
        endpoints = [
            "/docs",
            "/api/v1/agents",
            "/api/v1/chat",
            "/api/v1/models",
            "/api/v1/system/status",
            "/api/v1/documents",
            "/api/v1/features",
            "/api/v1/orchestration/status"
        ]
        
        for endpoint in endpoints:
            url = base_url + endpoint
            self.findings["api_endpoints"][endpoint] = self.test_endpoint(url)
            
    def check_rule_compliance(self):
        """Check compliance with CLAUDE.md rules"""
        print("[5/7] Checking rule compliance...")
        
        # Rule 1: No Fantasy Elements
        success, output = self.run_command(
            "grep -r 'quantum\\|AGI\\|ASI\\|magic' /opt/sutazaiapp/backend/app 2>/dev/null | wc -l"
        )
        if success and int(output.strip()) > 0:
            self.findings["rule_violations"].append({
                "rule": "Rule 1: No Fantasy Elements",
                "violation": f"Found {output.strip()} references to fantasy terms"
            })
            
        # Rule 16: Local LLM only
        if "ollama" in self.findings["services"]:
            if self.findings["services"]["ollama"].get("status") != "reachable":
                self.findings["rule_violations"].append({
                    "rule": "Rule 16: Use Local LLMs via Ollama",
                    "violation": "Ollama service not accessible"
                })
                
        # Check for hardcoded credentials
        success, output = self.run_command(
            "grep -r 'password.*=.*[\"\\']' /opt/sutazaiapp/backend/app/core 2>/dev/null | wc -l"
        )
        if success and int(output.strip()) > 0:
            self.findings["critical_issues"].append(
                f"Found {output.strip()} hardcoded password references"
            )
            
    def identify_critical_issues(self):
        """Identify critical system issues"""
        print("[6/7] Identifying critical issues...")
        
        # Check database schema
        if not self.findings["services"].get("postgres", {}).get("has_tables"):
            self.findings["critical_issues"].append(
                "PostgreSQL has no tables - database schema not applied"
            )
            
        # Check authentication
        backend_health = self.findings["services"].get("backend", {})
        if backend_health.get("status") == "reachable":
            # Test if endpoints require auth
            test_url = "http://127.0.0.1:10010/api/v1/agents"
            response = self.test_endpoint(test_url)
            if response.get("status_code") == 200:
                self.findings["critical_issues"].append(
                    "No authentication required for API endpoints"
                )
                
        # Check agent functionality
        working_agents = sum(
            1 for agent in self.findings["agents"].values()
            if agent.get("health", {}).get("status") == "reachable"
        )
        if working_agents < 3:
            self.findings["critical_issues"].append(
                f"Only {working_agents} agents are responding (expected 7+)"
            )
            
    def generate_recommendations(self):
        """Generate architectural recommendations"""
        print("[7/7] Generating recommendations...")
        
        # Based on findings
        if not self.findings["services"].get("postgres", {}).get("has_tables"):
            self.findings["recommendations"].append({
                "priority": "CRITICAL",
                "action": "Apply database schema",
                "command": "docker exec sutazai-postgres psql -U sutazai -d sutazai -f /init.sql"
            })
            
        if len(self.findings["critical_issues"]) > 0:
            self.findings["recommendations"].append({
                "priority": "HIGH",
                "action": "Implement authentication layer",
                "details": "Add JWT authentication middleware to all API endpoints"
            })
            
        # Check container health
        unhealthy = [
            name for name, info in self.findings["containers"].items()
            if isinstance(info, dict) and "unhealthy" in info.get("status", "").lower()
        ]
        if unhealthy:
            self.findings["recommendations"].append({
                "priority": "HIGH",
                "action": f"Fix unhealthy containers: {', '.join(unhealthy)}"
            })
            
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*80)
        print("ULTRA-DEEP SYSTEM ANALYSIS REPORT")
        print("="*80)
        
        # Executive Summary
        print("\n## EXECUTIVE SUMMARY")
        print(f"Timestamp: {self.findings['timestamp']}")
        print(f"Running Containers: {self.findings['containers'].get('running_count', 0)}")
        print(f"Critical Issues: {len(self.findings['critical_issues'])}")
        print(f"Rule Violations: {len(self.findings['rule_violations'])}")
        
        # Critical Issues
        if self.findings["critical_issues"]:
            print("\n## CRITICAL ISSUES")
            for i, issue in enumerate(self.findings["critical_issues"], 1):
                print(f"{i}. {issue}")
                
        # Service Status
        print("\n## SERVICE STATUS")
        for name, status in self.findings["services"].items():
            status_str = status.get("status", "unknown")
            print(f"- {name}: {status_str}")
            
        # Agent Status
        print("\n## AGENT STATUS")
        for name, info in self.findings["agents"].items():
            health = info.get("health", {}).get("status", "unknown")
            print(f"- {name}: {health}")
            
        # Rule Violations
        if self.findings["rule_violations"]:
            print("\n## RULE VIOLATIONS")
            for violation in self.findings["rule_violations"]:
                print(f"- {violation['rule']}: {violation['violation']}")
                
        # Recommendations
        if self.findings["recommendations"]:
            print("\n## RECOMMENDATIONS")
            for rec in self.findings["recommendations"]:
                print(f"\n[{rec['priority']}] {rec['action']}")
                if "details" in rec:
                    print(f"  Details: {rec['details']}")
                if "command" in rec:
                    print(f"  Command: {rec['command']}")
                    
        # Save full report
        with open("/opt/sutazaiapp/ultra_deep_analysis_report.json", "w") as f:
            json.dump(self.findings, f, indent=2)
        print(f"\nFull report saved to: /opt/sutazaiapp/ultra_deep_analysis_report.json")
        
    def run(self):
        """Execute full analysis"""
        print("Starting ULTRA-DEEP System Analysis...")
        print("This will test all services, agents, and compliance rules.\n")
        
        self.analyze_containers()
        self.analyze_services()
        self.analyze_agents()
        self.analyze_api_endpoints()
        self.check_rule_compliance()
        self.identify_critical_issues()
        self.generate_recommendations()
        self.generate_report()
        
if __name__ == "__main__":
    analyzer = UltraDeepAnalyzer()
    analyzer.run()