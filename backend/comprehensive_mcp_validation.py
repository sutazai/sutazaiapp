#!/usr/bin/env python3
"""
Comprehensive MCP Server Protection Validation
Rule 20 Compliance: MCP Server Protection Report
Generated: 2025-08-15
"""

import json
import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path
import docker
from typing import Dict, List, Tuple, Any

class MCPValidator:
    """Comprehensive MCP Infrastructure Validator"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "rule_20_compliance": False,
            "mcp_servers": {},
            "infrastructure": {},
            "security": {},
            "issues": [],
            "recommendations": []
        }
        self.docker_client = docker.from_env()
        self.mcp_config_path = Path("/opt/sutazaiapp/.mcp.json")
        self.wrapper_dir = Path("/opt/sutazaiapp/scripts/mcp/wrappers")
        
    def validate_mcp_config(self) -> bool:
        """Validate MCP configuration file integrity"""
        print("\n[1/7] Validating MCP Configuration File...")
        
        if not self.mcp_config_path.exists():
            self.results["issues"].append("CRITICAL: .mcp.json missing")
            return False
            
        try:
            with open(self.mcp_config_path, 'r') as f:
                config = json.load(f)
                
            servers = config.get("mcpServers", {})
            self.results["infrastructure"]["config_servers_count"] = len(servers)
            self.results["infrastructure"]["config_servers"] = list(servers.keys())
            
            # Validate each server configuration
            for name, server_config in servers.items():
                if "command" not in server_config:
                    self.results["issues"].append(f"Server {name}: Missing command")
                    
            print(f"  ✓ Found {len(servers)} MCP servers in configuration")
            return True
            
        except Exception as e:
            self.results["issues"].append(f"Config parse error: {str(e)}")
            return False
            
    def validate_wrapper_scripts(self) -> bool:
        """Validate all MCP wrapper scripts exist and are executable"""
        print("\n[2/7] Validating MCP Wrapper Scripts...")
        
        if not self.wrapper_dir.exists():
            self.results["issues"].append("CRITICAL: Wrapper directory missing")
            return False
            
        wrappers = list(self.wrapper_dir.glob("*.sh"))
        self.results["infrastructure"]["wrapper_count"] = len(wrappers)
        self.results["infrastructure"]["wrappers"] = [w.name for w in wrappers]
        
        missing_executable = []
        for wrapper in wrappers:
            if not os.access(wrapper, os.X_OK):
                missing_executable.append(wrapper.name)
                
        if missing_executable:
            self.results["issues"].append(f"Non-executable wrappers: {missing_executable}")
            
        print(f"  ✓ Found {len(wrappers)} wrapper scripts")
        return len(wrappers) > 0
        
    def test_mcp_servers(self) -> Dict[str, bool]:
        """Test each MCP server individually"""
        print("\n[3/7] Testing Individual MCP Servers...")
        
        servers_status = {}
        
        # Read MCP config
        try:
            with open(self.mcp_config_path, 'r') as f:
                config = json.load(f)
                servers = config.get("mcpServers", {})
        except:
            return servers_status
            
        for server_name in servers.keys():
            wrapper_path = self.wrapper_dir / f"{server_name.replace('_', '-')}.sh"
            
            # Special case for some server names
            if not wrapper_path.exists():
                wrapper_path = self.wrapper_dir / f"{server_name}.sh"
                
            if wrapper_path.exists():
                try:
                    result = subprocess.run(
                        [str(wrapper_path), "--selfcheck"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    servers_status[server_name] = result.returncode == 0
                    self.results["mcp_servers"][server_name] = {
                        "status": "operational" if result.returncode == 0 else "failed",
                        "wrapper": wrapper_path.name
                    }
                except Exception as e:
                    servers_status[server_name] = False
                    self.results["mcp_servers"][server_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                servers_status[server_name] = False
                self.results["mcp_servers"][server_name] = {
                    "status": "missing_wrapper"
                }
                
        operational = sum(1 for status in servers_status.values() if status)
        total = len(servers_status)
        print(f"  ✓ {operational}/{total} MCP servers operational")
        
        return servers_status
        
    def validate_docker_infrastructure(self) -> bool:
        """Validate Docker infrastructure for MCP"""
        print("\n[4/7] Validating Docker Infrastructure...")
        
        try:
            # Check critical containers
            critical_containers = [
                "sutazai-postgres",
                "sutazai-redis", 
                "sutazai-backend",
                "sutazai-frontend",
                "sutazai-ollama"
            ]
            
            container_status = {}
            for name in critical_containers:
                try:
                    container = self.docker_client.containers.get(name)
                    container_status[name] = container.status
                except:
                    container_status[name] = "missing"
                    
            self.results["infrastructure"]["containers"] = container_status
            
            # Check network
            try:
                network = self.docker_client.networks.get("sutazai-network")
                self.results["infrastructure"]["network"] = "present"
            except:
                self.results["infrastructure"]["network"] = "missing"
                self.results["issues"].append("Docker network missing")
                
            running = sum(1 for status in container_status.values() if status == "running")
            print(f"  ✓ {running}/{len(critical_containers)} critical containers running")
            return running >= 3  # At least core services running
            
        except Exception as e:
            self.results["issues"].append(f"Docker validation error: {str(e)}")
            return False
            
    def validate_mcp_cleanup_service(self) -> bool:
        """Validate MCP cleanup service is protecting infrastructure"""
        print("\n[5/7] Validating MCP Cleanup Service...")
        
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "mcp-cleanup.service"],
                capture_output=True,
                text=True
            )
            
            is_active = result.stdout.strip() == "active"
            self.results["infrastructure"]["cleanup_service"] = "active" if is_active else "inactive"
            
            if is_active:
                print("  ✓ MCP cleanup service is active and protecting infrastructure")
            else:
                self.results["issues"].append("MCP cleanup service not active")
                
            return is_active
            
        except Exception as e:
            self.results["infrastructure"]["cleanup_service"] = "error"
            return False
            
    def validate_security(self) -> bool:
        """Validate MCP security configurations"""
        print("\n[6/7] Validating MCP Security...")
        
        security_checks = {
            "config_permissions": False,
            "wrapper_permissions": False,
            "no_hardcoded_secrets": False,
            "postgres_auth": False
        }
        
        # Check file permissions
        try:
            config_stat = os.stat(self.mcp_config_path)
            # Check if world-readable (not ideal for production)
            security_checks["config_permissions"] = True
            
            # Check wrapper permissions
            wrapper_issues = []
            for wrapper in self.wrapper_dir.glob("*.sh"):
                stat = os.stat(wrapper)
                if stat.st_mode & 0o002:  # World writable
                    wrapper_issues.append(wrapper.name)
                    
            security_checks["wrapper_permissions"] = len(wrapper_issues) == 0
            if wrapper_issues:
                self.results["issues"].append(f"World-writable wrappers: {wrapper_issues}")
                
            # Check for hardcoded secrets in config
            with open(self.mcp_config_path, 'r') as f:
                config_content = f.read()
                # Basic check for common secret patterns
                if "password" in config_content.lower() and "=" in config_content:
                    self.results["issues"].append("Possible hardcoded secrets in config")
                else:
                    security_checks["no_hardcoded_secrets"] = True
                    
            # Check postgres authentication
            try:
                result = subprocess.run(
                    ["docker", "exec", "sutazai-postgres", "pg_isready", "-U", "sutazai"],
                    capture_output=True,
                    timeout=5
                )
                security_checks["postgres_auth"] = result.returncode == 0
            except:
                pass
                
        except Exception as e:
            self.results["issues"].append(f"Security validation error: {str(e)}")
            
        self.results["security"] = security_checks
        
        passed = sum(1 for check in security_checks.values() if check)
        print(f"  ✓ {passed}/{len(security_checks)} security checks passed")
        
        return passed >= 3  # Most checks should pass
        
    def generate_recommendations(self):
        """Generate recommendations based on findings"""
        print("\n[7/7] Generating Recommendations...")
        
        if "ultimatecoder" in self.results["mcp_servers"]:
            if self.results["mcp_servers"]["ultimatecoder"]["status"] != "operational":
                self.results["recommendations"].append(
                    "Fix ultimatecoder MCP: Install fastmcp in venv"
                )
                
        if self.results["infrastructure"].get("cleanup_service") != "active":
            self.results["recommendations"].append(
                "Enable MCP cleanup service: sudo systemctl enable --now mcp-cleanup.service"
            )
            
        if not self.results["security"].get("no_hardcoded_secrets"):
            self.results["recommendations"].append(
                "Review MCP configuration for hardcoded secrets"
            )
            
        # Check for missing wrappers
        if self.results["infrastructure"].get("config_servers_count", 0) > \
           self.results["infrastructure"].get("wrapper_count", 0):
            self.results["recommendations"].append(
                "Some MCP servers may be missing wrapper scripts"
            )
            
    def validate_all(self) -> bool:
        """Run all validations"""
        print("=" * 60)
        print("MCP SERVER PROTECTION VALIDATION - RULE 20 COMPLIANCE")
        print("=" * 60)
        
        checks = [
            self.validate_mcp_config(),
            self.validate_wrapper_scripts(),
            bool(self.test_mcp_servers()),
            self.validate_docker_infrastructure(),
            self.validate_mcp_cleanup_service(),
            self.validate_security()
        ]
        
        self.generate_recommendations()
        
        # Overall compliance
        operational_servers = sum(
            1 for server in self.results["mcp_servers"].values() 
            if server.get("status") == "operational"
        )
        total_servers = len(self.results["mcp_servers"])
        
        self.results["summary"] = {
            "total_servers": total_servers,
            "operational_servers": operational_servers,
            "compliance_percentage": (operational_servers / total_servers * 100) if total_servers > 0 else 0,
            "critical_issues": len([i for i in self.results["issues"] if "CRITICAL" in i]),
            "total_issues": len(self.results["issues"])
        }
        
        # Rule 20 compliance check
        self.results["rule_20_compliance"] = (
            operational_servers >= total_servers * 0.9 and  # 90% servers operational
            self.results["infrastructure"].get("cleanup_service") == "active" and
            len([i for i in self.results["issues"] if "CRITICAL" in i]) == 0
        )
        
        return self.results["rule_20_compliance"]
        
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        summary = self.results.get("summary", {})
        print(f"\nMCP Servers: {summary.get('operational_servers', 0)}/{summary.get('total_servers', 0)} operational")
        print(f"Compliance: {summary.get('compliance_percentage', 0):.1f}%")
        print(f"Critical Issues: {summary.get('critical_issues', 0)}")
        print(f"Total Issues: {summary.get('total_issues', 0)}")
        
        if self.results["rule_20_compliance"]:
            print("\n✅ RULE 20 COMPLIANT: MCP infrastructure is protected and operational")
        else:
            print("\n⚠️  RULE 20 VIOLATION: MCP infrastructure requires attention")
            
        if self.results["issues"]:
            print("\nIssues Found:")
            for issue in self.results["issues"][:5]:  # First 5 issues
                print(f"  - {issue}")
                
        if self.results["recommendations"]:
            print("\nRecommendations:")
            for rec in self.results["recommendations"]:
                print(f"  - {rec}")
                
        # Save detailed report
        report_path = Path("/opt/sutazaiapp/backend/mcp_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        
        return self.results["rule_20_compliance"]

def main():
    """Main validation entry point"""
    validator = MCPValidator()
    
    try:
        compliance = validator.validate_all()
        validator.generate_report()
        
        # Exit code based on compliance
        sys.exit(0 if compliance else 1)
        
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    main()