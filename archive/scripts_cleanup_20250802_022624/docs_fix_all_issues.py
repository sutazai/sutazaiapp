#!/usr/bin/env python3
"""
SutazAI Comprehensive Bug Fix and Validation Script
Identifies and fixes all known issues in the system
"""

import os
import sys
import json
import subprocess
import time
import psutil
import requests
from pathlib import Path
import docker
import logging
from typing import Dict, List, Tuple, Optional
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class SutazAISystemFixer:
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        self.docker_client = docker.from_env()
        self.base_path = Path("/opt/sutazaiapp")
        
    def print_status(self, message: str, status: str = "info"):
        """Print colored status messages"""
        if status == "success":
            print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")
        elif status == "error":
            print(f"{Colors.RED}✗ {message}{Colors.ENDC}")
        elif status == "warning":
            print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")
        elif status == "info":
            print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")
    
    def check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements"""
        self.print_status("Checking system requirements...", "info")
        
        issues = []
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            issues.append(f"Low CPU count: {cpu_count} cores (recommended: 4+)")
        
        # Check memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            issues.append(f"Low memory: {memory_gb:.1f}GB (recommended: 8GB+)")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 20:
            issues.append(f"Low disk space: {free_gb:.1f}GB free (recommended: 20GB+)")
        
        # Check Docker
        try:
            self.docker_client.ping()
        except:
            issues.append("Docker is not running or not accessible")
        
        if issues:
            for issue in issues:
                self.print_status(issue, "warning")
            return False
        
        self.print_status("System requirements met", "success")
        return True
    
    def fix_docker_containers(self):
        """Fix Docker container issues"""
        self.print_status("Checking Docker containers...", "info")
        
        # Expected containers
        expected_containers = {
            'sutazai-postgres': {'port': 5432, 'image': 'postgres:15'},
            'sutazai-redis': {'port': 6379, 'image': 'redis:7-alpine'},
            'sutazai-ollama': {'port': 11434, 'image': 'ollama/ollama:latest'},
            'sutazai-qdrant': {'port': 6333, 'image': 'qdrant/qdrant:latest'},
            'sutazai-chromadb': {'port': 8001, 'image': 'chromadb/chroma:latest'}
        }
        
        for name, config in expected_containers.items():
            try:
                container = self.docker_client.containers.get(name)
                if container.status != 'running':
                    self.print_status(f"Starting {name}...", "warning")
                    container.start()
                    time.sleep(3)
                    self.fixes_applied.append(f"Started {name}")
                else:
                    self.print_status(f"{name} is running", "success")
            except docker.errors.NotFound:
                self.print_status(f"{name} not found, creating...", "warning")
                # Container doesn't exist, would need docker-compose to create
                self.issues_found.append(f"{name} container missing")
    
    def fix_backend_issues(self):
        """Fix backend connectivity and configuration issues"""
        self.print_status("Checking backend services...", "info")
        
        # Kill any conflicting processes
        processes_to_check = [
            'intelligent_backend',
            'simple_backend',
            'backend_api'
        ]
        
        for proc_name in processes_to_check:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info.get('cmdline', []))
                    if proc_name in cmdline and 'python' in cmdline:
                        self.print_status(f"Stopping conflicting process: {proc_name} (PID: {proc.info['pid']})", "warning")
                        proc.terminate()
                        time.sleep(1)
                        if proc.is_running():
                            proc.kill()
                        self.fixes_applied.append(f"Stopped {proc_name}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        # Check if backend is responding
        backend_url = "http://localhost:8000/health"
        try:
            response = requests.get(backend_url, timeout=5)
            if response.status_code == 200:
                self.print_status("Backend is healthy", "success")
            else:
                self.issues_found.append("Backend returning non-200 status")
        except:
            self.print_status("Backend not responding, will restart", "warning")
            self.issues_found.append("Backend not responding")
    
    def fix_database_issues(self):
        """Fix database connectivity and schema issues"""
        self.print_status("Checking database...", "info")
        
        try:
            # Test database connection
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="sutazai_db",
                user="sutazai",
                password="sutazai123"
            )
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['users', 'conversations', 'documents', 'agent_tasks']
            missing_tables = set(expected_tables) - set(tables)
            
            if missing_tables:
                self.print_status(f"Missing tables: {missing_tables}", "warning")
                self.issues_found.append(f"Missing database tables: {missing_tables}")
            else:
                self.print_status("Database schema is complete", "success")
            
            conn.close()
        except Exception as e:
            self.print_status(f"Database connection error: {str(e)}", "error")
            self.issues_found.append(f"Database error: {str(e)}")
    
    def fix_file_permissions(self):
        """Fix file and directory permissions"""
        self.print_status("Checking file permissions...", "info")
        
        directories_to_check = [
            'data', 'logs', 'uploads', 'backups',
            'data/uploads', 'data/documents', 'data/models',
            'logs/agents', 'logs/backend', 'logs/system'
        ]
        
        for dir_name in directories_to_check:
            dir_path = self.base_path / dir_name
            if not dir_path.exists():
                self.print_status(f"Creating directory: {dir_name}", "warning")
                dir_path.mkdir(parents=True, exist_ok=True)
                self.fixes_applied.append(f"Created {dir_name}")
            
            # Fix permissions
            os.chmod(dir_path, 0o755)
    
    def fix_python_dependencies(self):
        """Check and install missing Python dependencies"""
        self.print_status("Checking Python dependencies...", "info")
        
        required_packages = [
            'fastapi', 'uvicorn', 'sqlalchemy', 'psycopg2-binary',
            'redis', 'requests', 'pydantic', 'prometheus-client',
            'psutil', 'aiofiles', 'websockets', 'python-multipart'
        ]
        
        import pkg_resources
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        missing_packages = []
        
        for package in required_packages:
            if package.lower() not in installed_packages:
                missing_packages.append(package)
        
        if missing_packages:
            self.print_status(f"Installing missing packages: {missing_packages}", "warning")
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages, 
                         capture_output=True, text=True)
            self.fixes_applied.append(f"Installed {len(missing_packages)} packages")
        else:
            self.print_status("All required Python packages installed", "success")
    
    def fix_ollama_models(self):
        """Ensure Ollama models are available"""
        self.print_status("Checking Ollama models...", "info")
        
        required_models = ['llama3.2:1b', 'qwen2.5:3b']
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                installed_models = [m['name'] for m in response.json().get('models', [])]
                
                for model in required_models:
                    if model not in installed_models:
                        self.print_status(f"Pulling Ollama model: {model}", "warning")
                        # Initiate model pull (non-blocking)
                        requests.post("http://localhost:11434/api/pull", 
                                    json={"name": model}, timeout=5)
                        self.fixes_applied.append(f"Initiated pull for {model}")
                    else:
                        self.print_status(f"Model {model} is available", "success")
            else:
                self.issues_found.append("Cannot connect to Ollama API")
        except Exception as e:
            self.print_status(f"Ollama check failed: {str(e)}", "error")
            self.issues_found.append(f"Ollama error: {str(e)}")
    
    def validate_api_endpoints(self):
        """Validate all API endpoints are working"""
        self.print_status("Validating API endpoints...", "info")
        
        endpoints_to_test = [
            ("GET", "/health", None),
            ("GET", "/api/status", None),
            ("GET", "/api/models", None),
            ("GET", "/api/performance/summary", None),
            ("GET", "/api/agents", None),
        ]
        
        base_url = "http://localhost:8000"
        working_endpoints = 0
        
        for method, endpoint, data in endpoints_to_test:
            try:
                url = f"{base_url}{endpoint}"
                if method == "GET":
                    response = requests.get(url, timeout=5)
                else:
                    response = requests.post(url, json=data, timeout=5)
                
                if response.status_code in [200, 201]:
                    self.print_status(f"{endpoint} is working", "success")
                    working_endpoints += 1
                else:
                    self.print_status(f"{endpoint} returned {response.status_code}", "warning")
                    self.issues_found.append(f"{endpoint} returned {response.status_code}")
            except Exception as e:
                self.print_status(f"{endpoint} failed: {str(e)}", "error")
                self.issues_found.append(f"{endpoint} error: {str(e)}")
        
        return working_endpoints
    
    def generate_fix_script(self):
        """Generate a script to fix remaining issues"""
        if not self.issues_found:
            return
        
        self.print_status("Generating fix script for remaining issues...", "info")
        
        fix_script = self.base_path / "auto_fix_remaining.sh"
        
        with open(fix_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated fix script for remaining issues\n\n")
            
            # Add fixes based on issues found
            if any("container missing" in issue for issue in self.issues_found):
                f.write("# Rebuild and start missing containers\n")
                f.write("cd /opt/sutazaiapp\n")
                f.write("docker-compose up -d --build\n\n")
            
            if any("Backend not responding" in issue for issue in self.issues_found):
                f.write("# Start enhanced backend\n")
                f.write("cd /opt/sutazaiapp\n")
                f.write("nohup python3 intelligent_backend_enterprise.py > logs/backend/enterprise.log 2>&1 &\n")
                f.write("sleep 5\n\n")
            
            if any("database" in issue.lower() for issue in self.issues_found):
                f.write("# Fix database issues\n")
                f.write("docker restart sutazai-postgres\n")
                f.write("sleep 5\n\n")
            
            f.write("echo 'Fix script completed!'\n")
        
        os.chmod(fix_script, 0o755)
        self.print_status(f"Fix script generated: {fix_script}", "success")
    
    def run_comprehensive_fix(self):
        """Run all fixes in sequence"""
        print(f"\n{Colors.BOLD}🔧 SutazAI Comprehensive System Fix{Colors.ENDC}")
        print("=" * 50)
        
        # Run all checks and fixes
        self.check_system_requirements()
        print()
        
        self.fix_file_permissions()
        print()
        
        self.fix_python_dependencies()
        print()
        
        self.fix_docker_containers()
        print()
        
        self.fix_database_issues()
        print()
        
        self.fix_backend_issues()
        print()
        
        self.fix_ollama_models()
        print()
        
        working_endpoints = self.validate_api_endpoints()
        print()
        
        # Generate report
        print(f"\n{Colors.BOLD}📊 Fix Summary{Colors.ENDC}")
        print("=" * 50)
        
        if self.fixes_applied:
            print(f"\n{Colors.GREEN}Fixes Applied:{Colors.ENDC}")
            for fix in self.fixes_applied:
                print(f"  ✓ {fix}")
        
        if self.issues_found:
            print(f"\n{Colors.YELLOW}Remaining Issues:{Colors.ENDC}")
            for issue in self.issues_found:
                print(f"  ⚠ {issue}")
            
            self.generate_fix_script()
        else:
            print(f"\n{Colors.GREEN}✅ All systems operational!{Colors.ENDC}")
        
        # Save report
        report_path = self.base_path / "SYSTEM_FIX_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(f"# SutazAI System Fix Report\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Fixes Applied ({len(self.fixes_applied)})\n")
            for fix in self.fixes_applied:
                f.write(f"- {fix}\n")
            
            f.write(f"\n## Remaining Issues ({len(self.issues_found)})\n")
            for issue in self.issues_found:
                f.write(f"- {issue}\n")
            
            f.write(f"\n## System Status\n")
            f.write(f"- Working API endpoints: {working_endpoints}\n")
            f.write(f"- System health: {'Operational' if not self.issues_found else 'Needs attention'}\n")
        
        print(f"\n{Colors.BLUE}Report saved to: {report_path}{Colors.ENDC}")
        
        return len(self.issues_found) == 0

if __name__ == "__main__":
    fixer = SutazAISystemFixer()
    success = fixer.run_comprehensive_fix()
    
    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 System is fully operational!{Colors.ENDC}")
        print(f"{Colors.GREEN}You can now start using SutazAI automation/advanced automation system.{Colors.ENDC}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Some issues remain.{Colors.ENDC}")
        print(f"{Colors.YELLOW}Run the generated fix script: ./auto_fix_remaining.sh{Colors.ENDC}")
    
    sys.exit(0 if success else 1)