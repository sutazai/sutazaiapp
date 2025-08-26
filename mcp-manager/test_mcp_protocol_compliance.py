#!/usr/bin/env python3
"""
MCP Protocol Compliance Validation Script
Tests actual MCP server implementations against protocol specification
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import uuid

class MCPProtocolValidator:
    """Validates MCP server protocol compliance"""
    
    def __init__(self):
        self.servers_config_path = Path("/opt/sutazaiapp/mcp-manager/config/mcp-servers.json")
        self.test_results = {}
        self.protocol_version = "2024-11-05"
        
    def load_server_config(self) -> Dict[str, Any]:
        """Load server configuration"""
        with open(self.servers_config_path) as f:
            return json.load(f)
    
    async def send_message(self, process: subprocess.Popen, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send JSON-RPC message and receive response"""
        try:
            # Add JSON-RPC fields
            if "jsonrpc" not in message:
                message["jsonrpc"] = "2.0"
            if "id" not in message:
                message["id"] = str(uuid.uuid4())
            
            # Send message
            message_str = json.dumps(message) + "\n"
            process.stdin.write(message_str.encode())
            process.stdin.flush()
            
            # Wait for response with timeout
            process.stdout.flush()
            
            # Read response
            response_line = process.stdout.readline()
            if response_line:
                return json.loads(response_line.decode().strip())
            return None
            
        except (json.JSONDecodeError, IOError, OSError) as e:
            print(f"Error communicating with process: {e}")
            return None
    
    async def test_server_initialization(self, server_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test server initialization protocol"""
        results = {
            "server": server_name,
            "tests": {},
            "compliance_score": 0,
            "errors": []
        }
        
        try:
            # Start server process
            cmd = config.get("command")
            args = config.get("args", [])
            
            # Handle wrapper scripts
            if cmd.endswith(".sh"):
                full_cmd = [cmd] + args
            else:
                full_cmd = [cmd] + args
            
            print(f"\n Testing {server_name}...")
            print(f"  Command: {' '.join(full_cmd)}")
            
            process = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False
            )
            
            # Allow process to start
            await asyncio.sleep(0.5)
            
            # Test 1: Initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "mcp-validator",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = await self.send_message(process, init_request)
            
            if response:
                results["tests"]["initialize"] = {
                    "passed": "result" in response,
                    "response": response
                }
                
                if "result" in response:
                    result = response["result"]
                    
                    # Check protocol version
                    if "protocolVersion" in result:
                        results["tests"]["protocol_version"] = {
                            "passed": result["protocolVersion"] == self.protocol_version,
                            "value": result["protocolVersion"]
                        }
                    
                    # Check capabilities
                    if "capabilities" in result:
                        results["tests"]["capabilities"] = {
                            "passed": True,
                            "value": result["capabilities"]
                        }
                    
                    # Check server info
                    if "serverInfo" in result:
                        results["tests"]["server_info"] = {
                            "passed": True,
                            "value": result["serverInfo"]
                        }
            else:
                results["tests"]["initialize"] = {
                    "passed": False,
                    "error": "No response received"
                }
            
            # Test 2: List tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": "tools-1",
                "method": "tools/list",
                "params": {}
            }
            
            tools_response = await self.send_message(process, tools_request)
            
            if tools_response:
                results["tests"]["tools_list"] = {
                    "passed": "result" in tools_response,
                    "response": tools_response
                }
                
                if "result" in tools_response and "tools" in tools_response["result"]:
                    tools = tools_response["result"]["tools"]
                    results["tests"]["tools_count"] = {
                        "passed": len(tools) > 0,
                        "count": len(tools),
                        "tools": [t.get("name") for t in tools] if tools else []
                    }
            else:
                results["tests"]["tools_list"] = {
                    "passed": False,
                    "error": "No response received"
                }
            
            # Test 3: List resources (if supported)
            resources_request = {
                "jsonrpc": "2.0",
                "id": "resources-1",
                "method": "resources/list",
                "params": {}
            }
            
            resources_response = await self.send_message(process, resources_request)
            
            if resources_response:
                if "result" in resources_response:
                    results["tests"]["resources_list"] = {
                        "passed": True,
                        "response": resources_response
                    }
                elif "error" in resources_response:
                    # Not all servers support resources
                    results["tests"]["resources_list"] = {
                        "passed": None,
                        "note": "Resources not supported"
                    }
            
            # Test 4: Error handling
            invalid_request = {
                "jsonrpc": "2.0",
                "id": "error-1",
                "method": "invalid/method",
                "params": {}
            }
            
            error_response = await self.send_message(process, invalid_request)
            
            if error_response and "error" in error_response:
                results["tests"]["error_handling"] = {
                    "passed": True,
                    "response": error_response
                }
            else:
                results["tests"]["error_handling"] = {
                    "passed": False,
                    "error": "No error response for invalid method"
                }
            
            # Calculate compliance score
            passed_tests = sum(1 for test in results["tests"].values() 
                             if test.get("passed") is True)
            total_tests = len([t for t in results["tests"].values() 
                             if t.get("passed") is not None])
            
            if total_tests > 0:
                results["compliance_score"] = (passed_tests / total_tests) * 100
            
            # Cleanup
            process.terminate()
            await asyncio.sleep(0.5)
            if process.poll() is None:
                process.kill()
            
        except Exception as e:
            results["errors"].append(str(e))
            results["compliance_score"] = 0
        
        return results
    
    async def validate_all_servers(self) -> Dict[str, Any]:
        """Validate all configured servers"""
        config = self.load_server_config()
        servers = config.get("mcpServers", {})
        
        validation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "protocol_version": self.protocol_version,
            "servers": {},
            "summary": {
                "total": 0,
                "compliant": 0,
                "partial": 0,
                "non_compliant": 0,
                "disabled": 0
            }
        }
        
        for server_name, server_config in servers.items():
            if not server_config.get("enabled", True):
                validation_results["summary"]["disabled"] += 1
                continue
            
            validation_results["summary"]["total"] += 1
            
            result = await self.test_server_initialization(server_name, server_config)
            validation_results["servers"][server_name] = result
            
            # Categorize compliance
            score = result["compliance_score"]
            if score >= 75:
                validation_results["summary"]["compliant"] += 1
            elif score >= 25:
                validation_results["summary"]["partial"] += 1
            else:
                validation_results["summary"]["non_compliant"] += 1
        
        return validation_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate compliance report"""
        report = []
        report.append("=" * 80)
        report.append("MCP PROTOCOL COMPLIANCE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Protocol Version: {results['protocol_version']}")
        report.append("")
        
        # Summary
        summary = results['summary']
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Servers: {summary['total']}")
        report.append(f"Compliant (≥75%): {summary['compliant']}")
        report.append(f"Partial (25-74%): {summary['partial']}")
        report.append(f"Non-Compliant (<25%): {summary['non_compliant']}")
        report.append(f"Disabled: {summary['disabled']}")
        report.append("")
        
        # Compliance percentage
        if summary['total'] > 0:
            compliance_rate = (summary['compliant'] / summary['total']) * 100
            report.append(f"Overall Compliance Rate: {compliance_rate:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        
        for server_name, result in results['servers'].items():
            report.append(f"\n{server_name.upper()}")
            report.append(f"  Compliance Score: {result['compliance_score']:.1f}%")
            
            for test_name, test_result in result['tests'].items():
                if test_result.get("passed") is True:
                    status = "✅ PASS"
                elif test_result.get("passed") is False:
                    status = "❌ FAIL"
                else:
                    status = "⚠️  N/A"
                
                report.append(f"  {test_name}: {status}")
                
                if test_result.get("error"):
                    report.append(f"    Error: {test_result['error']}")
                if test_result.get("value"):
                    report.append(f"    Value: {test_result['value']}")
                if test_result.get("count") is not None:
                    report.append(f"    Count: {test_result['count']}")
                if test_result.get("tools"):
                    report.append(f"    Tools: {', '.join(test_result['tools'])}")
            
            if result.get("errors"):
                report.append(f"  Errors: {', '.join(result['errors'])}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main validation routine"""
    validator = MCPProtocolValidator()
    
    print("Starting MCP Protocol Compliance Validation...")
    print("-" * 80)
    
    results = await validator.validate_all_servers()
    
    # Generate and print report
    report = validator.generate_report(results)
    print(report)
    
    # Save detailed results
    results_file = Path("/opt/sutazaiapp/mcp-manager/mcp_protocol_validation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Return exit code based on compliance
    if results['summary']['total'] > 0:
        compliance_rate = (results['summary']['compliant'] / results['summary']['total']) * 100
        if compliance_rate >= 75:
            return 0  # Success
        elif compliance_rate >= 50:
            return 1  # Partial success
        else:
            return 2  # Failure
    
    return 3  # No servers tested

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)