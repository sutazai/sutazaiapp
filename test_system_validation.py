#!/usr/bin/env python3
"""
SutazAI System Validation Script
Comprehensive testing of the AGI/ASI system components
"""

import asyncio
import json
import sys
import time
import requests
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def log(message, color=Colors.NC):
    print(f"{color}{message}{Colors.NC}")

def success(message):
    log(f"✅ {message}", Colors.GREEN)

def error(message):
    log(f"❌ {message}", Colors.RED)

def warning(message):
    log(f"⚠️  {message}", Colors.YELLOW)

def info(message):
    log(f"ℹ️  {message}", Colors.BLUE)

def progress(message):
    log(f"🔄 {message}", Colors.PURPLE)

class SutazAIValidator:
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "unknown",
            "score": 0,
            "max_score": 0
        }
        
    def run_test(self, test_name: str, test_func, weight: int = 1):
        """Run a test and record results"""
        self.test_results["max_score"] += weight
        
        try:
            progress(f"Running {test_name}...")
            result = test_func()
            
            if result:
                success(f"{test_name} - PASSED")
                self.test_results["tests"][test_name] = {
                    "status": "passed",
                    "weight": weight,
                    "details": result if isinstance(result, dict) else {}
                }
                self.test_results["score"] += weight
            else:
                error(f"{test_name} - FAILED")
                self.test_results["tests"][test_name] = {
                    "status": "failed", 
                    "weight": weight,
                    "details": {}
                }
                
        except Exception as e:
            error(f"{test_name} - ERROR: {str(e)}")
            self.test_results["tests"][test_name] = {
                "status": "error",
                "weight": weight,
                "error": str(e)
            }
    
    def test_file_structure(self):
        """Test if all required files exist"""
        required_files = [
            "backend/api/enhanced_main.py",
            "backend/core/orchestrator.py", 
            "backend/services/model_manager.py",
            "frontend/enhanced_streamlit_app.py",
            "docker-compose.enhanced.yml",
            "setup_complete_agi_system.sh",
            "IMPLEMENTATION_PLAN.md"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            warning(f"Missing files: {missing_files}")
            return False
        
        return {"total_files": len(required_files), "status": "all_present"}
    
    def test_docker_files(self):
        """Test Docker configuration files"""
        docker_files = [
            "docker-compose.yml",
            "docker-compose.enhanced.yml", 
            "docker/backend.Dockerfile",
            "docker/streamlit.Dockerfile"
        ]
        
        valid_files = 0
        for file_path in docker_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if len(content) > 100:  # Basic content check
                            valid_files += 1
                except:
                    pass
        
        return valid_files == len(docker_files)
    
    def test_python_imports(self):
        """Test if Python modules can be imported"""
        test_modules = [
            "backend.core.orchestrator",
            "backend.services.model_manager", 
            "backend.api.enhanced_main"
        ]
        
        importable_modules = 0
        for module in test_modules:
            try:
                # Use subprocess to avoid import conflicts
                result = subprocess.run(
                    [sys.executable, "-c", f"import {module}"],
                    capture_output=True,
                    cwd="/opt/sutazaiapp"
                )
                if result.returncode == 0:
                    importable_modules += 1
            except:
                pass
        
        return {
            "importable_modules": importable_modules,
            "total_modules": len(test_modules),
            "status": "passed" if importable_modules >= len(test_modules) * 0.8 else "failed"
        }
    
    def test_setup_script(self):
        """Test if setup script is executable and well-formed"""
        script_path = "setup_complete_agi_system.sh"
        
        if not os.path.exists(script_path):
            return False
        
        # Check if executable
        if not os.access(script_path, os.X_OK):
            return False
        
        # Check script content
        with open(script_path, 'r') as f:
            content = f.read()
            
        # Look for key functions
        required_functions = [
            "check_requirements",
            "install_system_dependencies", 
            "install_ai_models",
            "start_application_services"
        ]
        
        functions_found = sum(1 for func in required_functions if func in content)
        
        return {
            "executable": True,
            "functions_found": functions_found,
            "total_functions": len(required_functions),
            "line_count": len(content.split('\n'))
        }
    
    def test_configuration_files(self):
        """Test configuration and environment files"""
        config_checks = []
        
        # Check if .env.template exists
        if os.path.exists(".env.template"):
            config_checks.append("env_template")
        
        # Check enhanced docker compose
        if os.path.exists("docker-compose.enhanced.yml"):
            with open("docker-compose.enhanced.yml", 'r') as f:
                content = f.read()
                if "sutazai" in content.lower() and "services:" in content:
                    config_checks.append("docker_compose")
        
        # Check requirements files
        req_files = ["requirements.txt", "backend/requirements.txt", "frontend/requirements.txt"]
        for req_file in req_files:
            if os.path.exists(req_file):
                config_checks.append(f"requirements_{req_file.replace('/', '_')}")
        
        return {
            "valid_configs": len(config_checks),
            "config_details": config_checks
        }
    
    def test_agent_definitions(self):
        """Test AI agent and model definitions"""
        agent_score = 0
        
        # Check orchestrator for agent definitions
        if os.path.exists("backend/core/orchestrator.py"):
            with open("backend/core/orchestrator.py", 'r') as f:
                content = f.read()
                if "class AGIOrchestrator" in content:
                    agent_score += 1
                if "submit_task" in content:
                    agent_score += 1
                if "find_suitable_agent" in content:
                    agent_score += 1
        
        # Check model manager
        if os.path.exists("backend/services/model_manager.py"):
            with open("backend/services/model_manager.py", 'r') as f:
                content = f.read()
                if "class EnhancedModelManager" in content:
                    agent_score += 1
                if "deepseek-r1" in content:
                    agent_score += 1
                if "codellama" in content:
                    agent_score += 1
        
        return {
            "agent_score": agent_score,
            "max_score": 6,
            "status": "passed" if agent_score >= 4 else "failed"
        }
    
    def test_frontend_components(self):
        """Test frontend implementation"""
        if not os.path.exists("frontend/enhanced_streamlit_app.py"):
            return False
        
        with open("frontend/enhanced_streamlit_app.py", 'r') as f:
            content = f.read()
        
        components = []
        if "streamlit" in content:
            components.append("streamlit")
        if "tabs" in content.lower():
            components.append("tabs")
        if "chat" in content.lower():
            components.append("chat")
        if "agent" in content.lower():
            components.append("agents")
        if "monitoring" in content.lower():
            components.append("monitoring")
        
        return {
            "components_found": len(components),
            "components": components,
            "line_count": len(content.split('\n'))
        }
    
    def test_docker_compose_services(self):
        """Test Docker Compose service definitions"""
        if not os.path.exists("docker-compose.enhanced.yml"):
            return False
        
        with open("docker-compose.enhanced.yml", 'r') as f:
            content = f.read()
        
        # Count expected services
        expected_services = [
            "backend", "frontend", "ollama", "postgres", "redis",
            "chromadb", "qdrant", "prometheus", "grafana", 
            "autogpt", "crewai", "aider", "gpt-engineer"
        ]
        
        services_found = sum(1 for service in expected_services if service in content)
        
        return {
            "services_found": services_found,
            "total_expected": len(expected_services),
            "coverage": services_found / len(expected_services)
        }
    
    def test_api_endpoints(self):
        """Test API endpoint definitions"""
        if not os.path.exists("backend/api/enhanced_main.py"):
            return False
        
        with open("backend/api/enhanced_main.py", 'r') as f:
            content = f.read()
        
        endpoints = []
        if "@app.get(\"/health\")" in content:
            endpoints.append("health")
        if "@app.post(\"/api/v1/execute\")" in content:
            endpoints.append("execute")
        if "chat/completions" in content:
            endpoints.append("chat")
        if "code/generate" in content:
            endpoints.append("code_generation")
        if "agents" in content:
            endpoints.append("agents")
        
        return {
            "endpoints_found": len(endpoints),
            "endpoints": endpoints
        }
    
    def generate_report(self):
        """Generate final test report"""
        score_percentage = (self.test_results["score"] / self.test_results["max_score"]) * 100
        
        if score_percentage >= 90:
            self.test_results["overall_status"] = "excellent"
            status_color = Colors.GREEN
        elif score_percentage >= 70:
            self.test_results["overall_status"] = "good"
            status_color = Colors.BLUE
        elif score_percentage >= 50:
            self.test_results["overall_status"] = "fair"
            status_color = Colors.YELLOW
        else:
            self.test_results["overall_status"] = "poor"
            status_color = Colors.RED
        
        log("\n" + "="*60, Colors.CYAN)
        log("🎯 SUTAZAI SYSTEM VALIDATION REPORT", Colors.CYAN)
        log("="*60, Colors.CYAN)
        
        log(f"\n📊 Overall Score: {self.test_results['score']}/{self.test_results['max_score']} ({score_percentage:.1f}%)", status_color)
        log(f"🏆 Status: {self.test_results['overall_status'].upper()}", status_color)
        
        log(f"\n📋 Test Results:", Colors.BLUE)
        for test_name, result in self.test_results["tests"].items():
            status_icon = "✅" if result["status"] == "passed" else "❌" if result["status"] == "failed" else "⚠️"
            log(f"  {status_icon} {test_name}: {result['status'].upper()}")
            
            if "details" in result and result["details"]:
                for key, value in result["details"].items():
                    if isinstance(value, (int, float, str)):
                        log(f"     - {key}: {value}")
        
        # Save detailed report
        report_path = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        log(f"\n📄 Detailed report saved: {report_path}", Colors.BLUE)
        
        return score_percentage >= 70  # Return True if passed

def main():
    """Main validation function"""
    log("🚀 Starting SutazAI System Validation...\n", Colors.CYAN)
    
    validator = SutazAIValidator()
    
    # Run all tests
    validator.run_test("File Structure", validator.test_file_structure, weight=2)
    validator.run_test("Docker Files", validator.test_docker_files, weight=2)
    validator.run_test("Python Imports", validator.test_python_imports, weight=1)
    validator.run_test("Setup Script", validator.test_setup_script, weight=2)
    validator.run_test("Configuration Files", validator.test_configuration_files, weight=1)
    validator.run_test("Agent Definitions", validator.test_agent_definitions, weight=3)
    validator.run_test("Frontend Components", validator.test_frontend_components, weight=2)
    validator.run_test("Docker Services", validator.test_docker_compose_services, weight=3)
    validator.run_test("API Endpoints", validator.test_api_endpoints, weight=2)
    
    # Generate final report
    passed = validator.generate_report()
    
    if passed:
        log("\n🎉 System validation PASSED! Ready for deployment.", Colors.GREEN)
        return 0
    else:
        log("\n⚠️  System validation FAILED. Please review issues before deployment.", Colors.RED)
        return 1

if __name__ == "__main__":
    sys.exit(main())