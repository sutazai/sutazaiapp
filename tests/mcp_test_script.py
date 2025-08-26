#!/usr/bin/env python3
"""
MCP Server Testing Script
Tests each MCP server for actual functionality
"""

import json
import socket
import subprocess
import sys
import time
from typing import Dict, Any, Tuple

class MCPTester:
    def __init__(self):
        self.servers = {
            'mcp-claude-flow': 3001,
            'mcp-ruv-swarm': 3002,
            'mcp-files': 3003,
            'mcp-context7': 3004,
            'mcp-http-fetch': 3005,
            'mcp-ddg': 3006,
            'mcp-extended-memory': 3009,
            'mcp-ssh': 3010,
            'mcp-ultimatecoder': 3011,
            'mcp-knowledge-graph-mcp': 3014,
            'mcp-github': 3016,
            'mcp-language-server': 3018,
            'mcp-claude-task-runner': 3019
        }
        self.results = {}
    
    def check_port_listening(self, host: str, port: int, timeout: int = 3) -> bool:
        """Check if a port is listening"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def test_mcp_protocol(self, port: int) -> Tuple[bool, str]:
        """Test MCP protocol response"""
        try:
            # MCP initialize request
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            # Use curl to test HTTP-based MCP server
            cmd = [
                'curl', '-s', '-X', 'POST',
                f'http://localhost:{port}',
                '-H', 'Content-Type: application/json',
                '-d', json.dumps(mcp_request),
                '--max-time', '5'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout:
                try:
                    response = json.loads(result.stdout)
                    if 'result' in response or 'error' in response:
                        return True, f"Valid MCP response: {response.get('result', response.get('error'))}"
                    else:
                        return False, f"Invalid response format: {result.stdout[:200]}"
                except json.JSONDecodeError:
                    return False, f"Non-JSON response: {result.stdout[:200]}"
            else:
                return False, f"Connection failed: {result.stderr[:200]}"
                
        except Exception as e:
            return False, f"Exception: {str(e)}"
    
    def test_server_tools(self, server_name: str, port: int) -> Tuple[bool, str]:
        """Test actual server functionality by calling tools/resources"""
        try:
            # Test tools list endpoint
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            
            cmd = [
                'curl', '-s', '-X', 'POST',
                f'http://localhost:{port}',
                '-H', 'Content-Type: application/json',
                '-d', json.dumps(tools_request),
                '--max-time', '5'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout:
                try:
                    response = json.loads(result.stdout)
                    if 'result' in response and 'tools' in response['result']:
                        tools = response['result']['tools']
                        return True, f"Found {len(tools)} tools: {[t.get('name', 'unnamed') for t in tools[:3]]}"
                    elif 'error' in response:
                        return False, f"Tools list error: {response['error']}"
                    else:
                        return False, f"No tools found in response: {result.stdout[:200]}"
                except json.JSONDecodeError:
                    return False, f"Invalid JSON in tools response: {result.stdout[:200]}"
            else:
                return False, f"Tools request failed: {result.stderr[:200]}"
                
        except Exception as e:
            return False, f"Tools test exception: {str(e)}"
    
    def test_server(self, server_name: str, port: int) -> Dict[str, Any]:
        """Comprehensive test of a single MCP server"""
        print(f"\n=== Testing {server_name} (port {port}) ===")
        
        result = {
            'server': server_name,
            'port': port,
            'port_listening': False,
            'mcp_protocol_valid': False,
            'mcp_response': '',
            'tools_working': False,
            'tools_response': '',
            'overall_status': 'BROKEN'
        }
        
        # Test 1: Port listening
        result['port_listening'] = self.check_port_listening('localhost', port)
        print(f"  Port {port} listening: {result['port_listening']}")
        
        if not result['port_listening']:
            result['overall_status'] = 'NOT_RUNNING'
            return result
        
        # Test 2: MCP Protocol
        mcp_valid, mcp_response = self.test_mcp_protocol(port)
        result['mcp_protocol_valid'] = mcp_valid
        result['mcp_response'] = mcp_response
        print(f"  MCP Protocol: {mcp_valid} - {mcp_response[:100]}")
        
        # Test 3: Tools functionality  
        tools_valid, tools_response = self.test_server_tools(server_name, port)
        result['tools_working'] = tools_valid
        result['tools_response'] = tools_response
        print(f"  Tools working: {tools_valid} - {tools_response[:100]}")
        
        # Determine overall status
        if result['port_listening'] and result['mcp_protocol_valid'] and result['tools_working']:
            result['overall_status'] = 'WORKING'
        elif result['port_listening'] and result['mcp_protocol_valid']:
            result['overall_status'] = 'PARTIAL'
        else:
            result['overall_status'] = 'BROKEN'
            
        return result
    
    def run_all_tests(self):
        """Test all MCP servers"""
        print("=== MCP SERVER COMPREHENSIVE TESTING ===")
        print(f"Testing {len(self.servers)} servers...")
        
        for server_name, port in self.servers.items():
            self.results[server_name] = self.test_server(server_name, port)
            time.sleep(1)  # Brief pause between tests
    
    def generate_report(self):
        """Generate detailed test report"""
        print("\n" + "="*80)
        print("MCP SERVER TEST REPORT")
        print("="*80)
        
        working = []
        partial = []
        broken = []
        not_running = []
        
        for server_name, result in self.results.items():
            status = result['overall_status']
            if status == 'WORKING':
                working.append(server_name)
            elif status == 'PARTIAL':
                partial.append(server_name)
            elif status == 'NOT_RUNNING':
                not_running.append(server_name)
            else:
                broken.append(server_name)
        
        print(f"\n✅ FULLY WORKING ({len(working)}): {', '.join(working)}")
        print(f"⚠️  PARTIALLY WORKING ({len(partial)}): {', '.join(partial)}")
        print(f"❌ BROKEN ({len(broken)}): {', '.join(broken)}")
        print(f"🔴 NOT RUNNING ({len(not_running)}): {', '.join(not_running)}")
        
        print(f"\n📊 SUMMARY: {len(working)}/13 servers fully functional")
        
        # Detailed breakdown
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for server_name, result in self.results.items():
            print(f"\n🔧 {server_name} (port {result['port']})")
            print(f"   Status: {result['overall_status']}")
            print(f"   Port Listening: {result['port_listening']}")
            print(f"   MCP Protocol: {result['mcp_protocol_valid']}")
            print(f"   Tools Working: {result['tools_working']}")
            if result['mcp_response']:
                print(f"   MCP Response: {result['mcp_response'][:150]}...")
            if result['tools_response']:
                print(f"   Tools Response: {result['tools_response'][:150]}...")

if __name__ == "__main__":
    tester = MCPTester()
    tester.run_all_tests()
    tester.generate_report()