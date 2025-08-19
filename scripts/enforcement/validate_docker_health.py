"""
Docker Health Validation Script
Validates that Docker infrastructure is working correctly
Date: 2025-08-18
"""

import subprocess
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

class DockerHealthValidator:
    def __init__(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.required_containers = [
            "sutazai-backend",
            "sutazai-chromadb",
            "sutazai-consul",
            "sutazai-prometheus",
            "sutazai-kong",
            "sutazai-qdrant"
        ]
        self.required_ports = [
            10000,  # PostgreSQL
            10001,  # Redis
            10005,  # Kong
            10006,  # Consul
            10010,  # Backend
            10100,  # ChromaDB
            10200,  # Prometheus
        ]
        self.test_results = []
        
    def check_docker_daemon(self) -> bool:
        """Check if Docker daemon is running"""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, text=True, check=True
            )
            self.test_results.append(("Docker Daemon", "PASS", "Docker is running"))
            return True
        except subprocess.CalledProcessError as e:
            self.test_results.append(("Docker Daemon", "FAIL", f"Docker not running: {e}"))
            return False
    
    def check_docker_compose(self) -> bool:
        """Check if docker-compose is available and config is valid"""
        try:
            result = subprocess.run(
                ["docker-compose", "version"],
                capture_output=True, text=True, check=True
            )
            self.test_results.append(("Docker Compose", "PASS", "docker-compose available"))
            
            result = subprocess.run(
                ["docker-compose", "config"],
                capture_output=True, text=True, check=True,
                cwd=str(self.project_root)
            )
            self.test_results.append(("Compose Config", "PASS", "Configuration is valid"))
            return True
        except subprocess.CalledProcessError as e:
            self.test_results.append(("Docker Compose", "FAIL", f"Error: {e}"))
            return False
    
    def check_running_containers(self) -> Dict[str, str]:
        """Check which containers are running"""
        container_status = {}
        
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}:{{.Status}}"],
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    name, status = line.split(':', 1)
                    container_status[name] = status
                    
                    if name in self.required_containers:
                        if "Up" in status:
                            if "unhealthy" in status:
                                self.test_results.append((f"Container: {name}", "WARN", "Running but unhealthy"))
                            else:
                                self.test_results.append((f"Container: {name}", "PASS", "Running"))
                        else:
                            self.test_results.append((f"Container: {name}", "FAIL", "Not running"))
            
            for required in self.required_containers:
                if required not in container_status:
                    self.test_results.append((f"Container: {required}", "FAIL", "Not found"))
                    
        except subprocess.CalledProcessError as e:
            self.test_results.append(("Container Check", "FAIL", f"Error: {e}"))
            
        return container_status
    
    def check_port_availability(self) -> List[int]:
        """Check which required ports are accessible"""
        accessible_ports = []
        
        for port in self.required_ports:
            try:
                result = subprocess.run(
                    ["nc", "-zv", "localhost", str(port)],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    accessible_ports.append(port)
                    self.test_results.append((f"Port {port}", "PASS", "Accessible"))
                else:
                    self.test_results.append((f"Port {port}", "FAIL", "Not accessible"))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                self.test_results.append((f"Port {port}", "FAIL", "Connection failed"))
        
        return accessible_ports
    
    def check_docker_network(self) -> bool:
        """Check if sutazai-network exists"""
        try:
            result = subprocess.run(
                ["docker", "network", "inspect", "sutazai-network"],
                capture_output=True, text=True, check=True
            )
            network_info = json.loads(result.stdout)
            if network_info:
                self.test_results.append(("Docker Network", "PASS", "sutazai-network exists"))
                return True
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            self.test_results.append(("Docker Network", "FAIL", "sutazai-network not found"))
            return False
    
    def check_docker_volumes(self) -> List[str]:
        """Check Docker volumes"""
        volumes = []
        try:
            result = subprocess.run(
                ["docker", "volume", "ls", "--format", "{{.Name}}"],
                capture_output=True, text=True, check=True
            )
            volumes = [v for v in result.stdout.strip().split('\n') if v.startswith("sutazai")]
            
            if volumes:
                self.test_results.append(("Docker Volumes", "PASS", f"Found {len(volumes)} volumes"))
            else:
                self.test_results.append(("Docker Volumes", "WARN", "No sutazai volumes found"))
                
        except subprocess.CalledProcessError as e:
            self.test_results.append(("Docker Volumes", "FAIL", f"Error: {e}"))
            
        return volumes
    
    def check_api_health(self) -> bool:
        """Check if backend API is responding"""
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:10010/health"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout == "200":
                self.test_results.append(("Backend API", "PASS", "Responding with 200"))
                return True
            else:
                self.test_results.append(("Backend API", "WARN", f"HTTP {result.stdout}"))
                return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            self.test_results.append(("Backend API", "FAIL", "Not responding"))
            return False
    
    def run_validation(self) -> bool:
        """Run complete validation suite"""
        print("=" * 60)
        print("Docker Health Validation")
        print("=" * 60)
        
        docker_ok = self.check_docker_daemon()
        compose_ok = self.check_docker_compose()
        containers = self.check_running_containers()
        ports = self.check_port_availability()
        network_ok = self.check_docker_network()
        volumes = self.check_docker_volumes()
        api_ok = self.check_api_health()
        
        print("\n📊 Validation Results:")
        print("-" * 40)
        
        pass_count = 0
        fail_count = 0
        warn_count = 0
        
        for test, status, message in self.test_results:
            if status == "PASS":
                symbol = "✓"
                color = "\033[32m"  # Green
                pass_count += 1
            elif status == "WARN":
                symbol = "⚠"
                color = "\033[33m"  # Yellow
                warn_count += 1
            else:
                symbol = "✗"
                color = "\033[31m"  # Red
                fail_count += 1
            
            print(f"{color}{symbol}\033[0m {test:30} {message}")
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"✓ Passed: {pass_count}")
        print(f"⚠ Warnings: {warn_count}")
        print(f"✗ Failed: {fail_count}")
        
        if fail_count == 0:
            print("\n🎉 Docker infrastructure is healthy!")
            return True
        else:
            print(f"\n❌ Docker infrastructure has {fail_count} issues that need attention")
            return False
    
    def save_report(self, filename: str = None):
        """Save validation report to file"""
        if filename is None:
            filename = f"/opt/sutazaiapp/docs/reports/docker_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(filename, 'w') as f:
            f.write("# Docker Health Validation Report\n")
            f.write(f"Date: {datetime.now().isoformat()}\n\n")
            
            f.write("## Test Results\n\n")
            f.write("| Test | Status | Message |\n")
            f.write("|------|--------|------|\n")
            
            for test, status, message in self.test_results:
                f.write(f"| {test} | {status} | {message} |\n")
            
            f.write("\n## Summary\n")
            pass_count = sum(1 for _, s, _ in self.test_results if s == "PASS")
            warn_count = sum(1 for _, s, _ in self.test_results if s == "WARN")
            fail_count = sum(1 for _, s, _ in self.test_results if s == "FAIL")
            
            f.write(f"- Passed: {pass_count}\n")
            f.write(f"- Warnings: {warn_count}\n")
            f.write(f"- Failed: {fail_count}\n")
        
        print(f"\n📄 Report saved to: {filename}")

def main():
    """Main entry point"""
    validator = DockerHealthValidator()
    success = validator.run_validation()
    
    validator.save_report()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()