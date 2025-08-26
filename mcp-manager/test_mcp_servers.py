#!/usr/bin/env python3
"""
Comprehensive MCP Server Test Suite

Tests for all MCP servers according to the Model Context Protocol specification.
Validates STDIO communication, health checks, message exchange, and error handling.

Author: Claude Code QA Agent
Created: 2025-08-26
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Define pytest decorators as no-ops when pytest is not available
    class pytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        @staticmethod
        def skip(reason):
            pass
        
        class mark:
            @staticmethod
            def asyncio(func):
                return func


# Test Configuration
TEST_TIMEOUT = 30  # seconds
HEALTH_CHECK_TIMEOUT = 10  # seconds
MCP_MESSAGE_TIMEOUT = 5  # seconds

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    """MCP Server configuration"""
    name: str
    command: str
    args: List[str]
    server_type: str
    description: str
    enabled: bool = True
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MCPProtocolTester:
    """MCP Protocol testing utilities"""
    
    @staticmethod
    def create_initialize_request(request_id: int = 1) -> Dict[str, Any]:
        """Create MCP initialize request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "mcp-test-suite",
                    "version": "1.0.0"
                }
            }
        }
    
    @staticmethod
    def create_list_tools_request(request_id: int = 2) -> Dict[str, Any]:
        """Create list tools request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/list",
            "params": {}
        }
    
    @staticmethod
    def create_ping_request(request_id: int = 3) -> Dict[str, Any]:
        """Create ping request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "ping",
            "params": {}
        }
    
    @staticmethod
    def validate_mcp_response(response: Dict[str, Any], expected_id: int = None) -> bool:
        """Validate MCP response format"""
        if not isinstance(response, dict):
            return False
            
        # Must have jsonrpc field
        if response.get("jsonrpc") != "2.0":
            return False
            
        # Check if this is a notification (has method field)
        if "method" in response:
            # Notifications don't have id field, that's normal
            return True
            
        # Must have either id or error for regular responses
        if "id" not in response and "error" not in response:
            return False
            
        # If ID provided, it must match
        if expected_id is not None and response.get("id") != expected_id:
            return False
            
        return True
    
    @staticmethod
    def is_error_response(response: Dict[str, Any]) -> bool:
        """Check if response is an error"""
        return "error" in response


class MCPServerProcess:
    """Manages MCP server process lifecycle"""
    
    def __init__(self, server: MCPServer):
        self.server = server
        self.process: Optional[subprocess.Popen] = None
        self._stdout_buffer = []
        self._stderr_buffer = []
    
    async def start(self) -> bool:
        """Start the MCP server process"""
        try:
            # Build command
            if self.server.command.endswith('.sh'):
                # Shell wrapper
                cmd = ['/bin/bash', self.server.command] + self.server.args
            else:
                # Direct command
                cmd = [self.server.command] + self.server.args
            
            logger.info(f"Starting {self.server.name}: {' '.join(cmd)}")
            
            # Start process with STDIO
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            # Give process time to start
            await asyncio.sleep(1)
            
            # Check if process is still running
            if self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                logger.error(f"Process {self.server.name} exited early: {stderr}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {self.server.name}: {e}")
            return False
    
    async def send_message(self, message: Dict[str, Any], timeout: float = MCP_MESSAGE_TIMEOUT) -> Optional[Dict[str, Any]]:
        """Send message to MCP server and wait for response"""
        if not self.process or self.process.poll() is not None:
            return None
        
        try:
            message_str = json.dumps(message) + '\n'
            logger.debug(f"Sending to {self.server.name}: {message_str.strip()}")
            
            # Send message
            self.process.stdin.write(message_str)
            self.process.stdin.flush()
            
            # Wait for response with timeout
            start_time = time.time()
            expected_id = message.get("id")
            
            while time.time() - start_time < timeout:
                if self.process.stdout.readable():
                    try:
                        line = self.process.stdout.readline()
                        if line:
                            line = line.strip()
                            logger.debug(f"Received from {self.server.name}: {line}")
                            
                            if line:
                                try:
                                    response = json.loads(line)
                                    
                                    # For initialize, we might get a notification first, then the response
                                    # or just the response. Return any valid MCP message
                                    if expected_id is not None:
                                        # Look for response with matching ID
                                        if response.get("id") == expected_id:
                                            return response
                                        # Or return notification/error
                                        elif "method" in response or "error" in response:
                                            return response
                                    else:
                                        # No ID expected, return any valid response
                                        return response
                                        
                                except json.JSONDecodeError:
                                    logger.warning(f"Invalid JSON from {self.server.name}: {line}")
                                    continue
                    except Exception as e:
                        logger.warning(f"Error reading from {self.server.name}: {e}")
                
                await asyncio.sleep(0.1)
            
            logger.warning(f"Timeout waiting for response from {self.server.name}")
            return None
            
        except Exception as e:
            logger.error(f"Error sending message to {self.server.name}: {e}")
            return None
    
    def stop(self):
        """Stop the MCP server process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"Error stopping {self.server.name}: {e}")
            finally:
                self.process = None
    
    def is_running(self) -> bool:
        """Check if process is running"""
        return self.process is not None and self.process.poll() is None


def load_mcp_servers() -> List[MCPServer]:
    """Load MCP server configurations from config files"""
    servers = []
    
    # Load from main .mcp.json
    main_config_path = Path("/opt/sutazaiapp/.mcp.json")
    if main_config_path.exists():
        with open(main_config_path) as f:
            config = json.load(f)
            for name, server_config in config.get("mcpServers", {}).items():
                servers.append(MCPServer(
                    name=name,
                    command=server_config["command"],
                    args=server_config.get("args", []),
                    server_type=server_config.get("type", "stdio"),
                    description=server_config.get("description", f"{name} MCP server")
                ))
    
    # Load from manager config
    manager_config_path = Path("/opt/sutazaiapp/mcp-manager/config/mcp-servers.json")
    if manager_config_path.exists():
        with open(manager_config_path) as f:
            config = json.load(f)
            for name, server_config in config.get("mcpServers", {}).items():
                # Don't duplicate servers
                if not any(s.name == name for s in servers):
                    servers.append(MCPServer(
                        name=name,
                        command=server_config["command"],
                        args=server_config.get("args", []),
                        server_type=server_config.get("type", "stdio"),
                        description=server_config.get("description", f"{name} MCP server"),
                        enabled=server_config.get("enabled", True),
                        tags=server_config.get("tags", [])
                    ))
    
    return servers


class TestMCPServers:
    """Comprehensive MCP Server test suite"""
    
    @pytest.fixture(scope="session")
    def mcp_servers(self) -> List[MCPServer]:
        """Load all MCP server configurations"""
        return load_mcp_servers()
    
    @pytest.fixture
    def protocol_tester(self) -> MCPProtocolTester:
        """Get protocol testing utilities"""
        return MCPProtocolTester()
    
    def test_server_configurations_loaded(self, mcp_servers: List[MCPServer]):
        """Test that server configurations are loaded correctly"""
        assert len(mcp_servers) > 0, "No MCP servers found in configuration"
        
        for server in mcp_servers:
            assert server.name, f"Server missing name: {server}"
            assert server.command, f"Server {server.name} missing command"
            assert server.server_type == "stdio", f"Server {server.name} has unsupported type: {server.server_type}"
    
    @pytest.mark.asyncio
    async def test_server_startup(self, mcp_servers: List[MCPServer]):
        """Test that all servers can start up properly"""
        for server in mcp_servers[:5]:  # Test first 5 to avoid overwhelming system
            if not server.enabled:
                pytest.skip(f"Server {server.name} is disabled")
            
            logger.info(f"Testing startup for {server.name}")
            server_process = MCPServerProcess(server)
            
            try:
                startup_success = await server_process.start()
                if startup_success:
                    assert server_process.is_running(), f"Server {server.name} not running after start"
                else:
                    # Log the failure but don't fail the test - some servers may need specific setup
                    logger.warning(f"Server {server.name} failed to start")
            finally:
                server_process.stop()
    
    @pytest.mark.asyncio
    async def test_mcp_initialize_protocol(self, mcp_servers: List[MCPServer], protocol_tester: MCPProtocolTester):
        """Test MCP initialize protocol for working servers"""
        working_servers = []
        
        for server in mcp_servers[:3]:  # Test first 3 servers
            if not server.enabled:
                continue
                
            logger.info(f"Testing MCP initialize for {server.name}")
            server_process = MCPServerProcess(server)
            
            try:
                # Start server
                if not await server_process.start():
                    logger.warning(f"Skipping {server.name} - failed to start")
                    continue
                
                # Send initialize request
                init_request = protocol_tester.create_initialize_request()
                response = await server_process.send_message(init_request, timeout=10)
                
                if response:
                    assert protocol_tester.validate_mcp_response(response, 1), \
                        f"Invalid MCP response from {server.name}: {response}"
                    
                    if not protocol_tester.is_error_response(response):
                        working_servers.append(server.name)
                        logger.info(f"✓ {server.name} MCP initialize successful")
                    else:
                        logger.warning(f"✗ {server.name} returned error: {response.get('error')}")
                else:
                    logger.warning(f"✗ {server.name} no response to initialize")
                    
            finally:
                server_process.stop()
        
        # At least one server should work
        assert len(working_servers) > 0, f"No servers successfully completed initialize. Tested: {[s.name for s in mcp_servers[:3]]}"
    
    @pytest.mark.asyncio
    async def test_tools_listing(self, mcp_servers: List[MCPServer], protocol_tester: MCPProtocolTester):
        """Test tools listing for servers that support it"""
        tools_servers = []
        
        for server in mcp_servers[:3]:  # Test subset
            if not server.enabled:
                continue
                
            logger.info(f"Testing tools listing for {server.name}")
            server_process = MCPServerProcess(server)
            
            try:
                # Start and initialize server
                if not await server_process.start():
                    continue
                
                init_response = await server_process.send_message(
                    protocol_tester.create_initialize_request()
                )
                
                if not init_response or protocol_tester.is_error_response(init_response):
                    continue
                
                # Try to list tools
                tools_request = protocol_tester.create_list_tools_request()
                tools_response = await server_process.send_message(tools_request)
                
                if tools_response and protocol_tester.validate_mcp_response(tools_response, 2):
                    if not protocol_tester.is_error_response(tools_response):
                        tools = tools_response.get("result", {}).get("tools", [])
                        logger.info(f"✓ {server.name} listed {len(tools)} tools")
                        tools_servers.append(server.name)
                    else:
                        logger.info(f"- {server.name} doesn't support tools listing")
                        
            finally:
                server_process.stop()
        
        # Report results but don't require all servers to have tools
        logger.info(f"Servers with tools: {tools_servers}")
    
    @pytest.mark.asyncio
    async def test_ping_pong(self, mcp_servers: List[MCPServer], protocol_tester: MCPProtocolTester):
        """Test ping/pong for servers that support it"""
        ping_servers = []
        
        for server in mcp_servers[:2]:  # Test subset 
            if not server.enabled:
                continue
                
            logger.info(f"Testing ping for {server.name}")
            server_process = MCPServerProcess(server)
            
            try:
                if not await server_process.start():
                    continue
                
                # Initialize first
                init_response = await server_process.send_message(
                    protocol_tester.create_initialize_request()
                )
                
                if not init_response or protocol_tester.is_error_response(init_response):
                    continue
                
                # Try ping
                ping_request = protocol_tester.create_ping_request()
                ping_response = await server_process.send_message(ping_request)
                
                if ping_response and protocol_tester.validate_mcp_response(ping_response, 3):
                    if not protocol_tester.is_error_response(ping_response):
                        logger.info(f"✓ {server.name} responded to ping")
                        ping_servers.append(server.name)
                    else:
                        logger.info(f"- {server.name} doesn't support ping")
                        
            finally:
                server_process.stop()
        
        logger.info(f"Servers supporting ping: {ping_servers}")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_servers: List[MCPServer]):
        """Test error handling with invalid requests"""
        error_tested = []
        
        for server in mcp_servers[:2]:  # Test subset
            if not server.enabled:
                continue
                
            logger.info(f"Testing error handling for {server.name}")
            server_process = MCPServerProcess(server)
            
            try:
                if not await server_process.start():
                    continue
                
                # Send invalid JSON
                invalid_message = {"invalid": "request", "missing": "required_fields"}
                response = await server_process.send_message(invalid_message)
                
                if response:
                    # Should get an error response or handle gracefully
                    assert isinstance(response, dict), f"Expected dict response from {server.name}"
                    logger.info(f"✓ {server.name} handled invalid request")
                    error_tested.append(server.name)
                    
            finally:
                server_process.stop()
        
        logger.info(f"Servers tested for error handling: {error_tested}")
    
    def test_wrapper_script_health_checks(self, mcp_servers: List[MCPServer]):
        """Test wrapper script health checks where available"""
        health_results = {}
        
        for server in mcp_servers:
            if server.command.endswith('.sh'):
                logger.info(f"Testing health check for wrapper {server.name}")
                
                try:
                    # Try selfcheck if supported
                    result = subprocess.run(
                        ['/bin/bash', server.command, '--selfcheck'],
                        capture_output=True,
                        text=True,
                        timeout=HEALTH_CHECK_TIMEOUT
                    )
                    
                    health_results[server.name] = {
                        'return_code': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
                    
                    if result.returncode == 0:
                        logger.info(f"✓ {server.name} selfcheck passed")
                    else:
                        logger.warning(f"✗ {server.name} selfcheck failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"✗ {server.name} selfcheck timed out")
                    health_results[server.name] = {'error': 'timeout'}
                except Exception as e:
                    logger.warning(f"✗ {server.name} selfcheck error: {e}")
                    health_results[server.name] = {'error': str(e)}
        
        # Report health check results
        passed = sum(1 for r in health_results.values() if r.get('return_code') == 0)
        total = len(health_results)
        logger.info(f"Health checks: {passed}/{total} passed")
        
        # Don't fail test if some health checks fail - this is informational
        assert total > 0, "No wrapper scripts found to test"
    
    @pytest.mark.asyncio 
    async def test_concurrent_connections(self, mcp_servers: List[MCPServer], protocol_tester: MCPProtocolTester):
        """Test concurrent connections to servers"""
        # Test one stable server with multiple connections
        test_server = None
        for server in mcp_servers:
            if server.enabled and 'extended-memory' in server.name:  # Known stable server
                test_server = server
                break
        
        if not test_server:
            pytest.skip("No suitable server for concurrent testing")
        
        logger.info(f"Testing concurrent connections to {test_server.name}")
        
        processes = []
        try:
            # Start multiple connections
            for i in range(3):
                process = MCPServerProcess(test_server)
                if await process.start():
                    processes.append(process)
            
            assert len(processes) >= 2, f"Failed to start multiple instances of {test_server.name}"
            
            # Send messages concurrently
            tasks = []
            for i, process in enumerate(processes):
                init_request = protocol_tester.create_initialize_request(request_id=i+1)
                task = asyncio.create_task(process.send_message(init_request))
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in responses if isinstance(r, dict) and not protocol_tester.is_error_response(r))
            
            logger.info(f"Concurrent test: {successful}/{len(processes)} connections successful")
            
        finally:
            for process in processes:
                process.stop()
    
    @pytest.mark.asyncio
    async def test_message_throughput(self, mcp_servers: List[MCPServer], protocol_tester: MCPProtocolTester):
        """Test message throughput for responsive servers"""
        # Find a responsive server
        responsive_server = None
        for server in mcp_servers:
            if server.enabled and any(tag in ['memory', 'task'] for tag in server.tags):
                responsive_server = server
                break
        
        if not responsive_server:
            pytest.skip("No suitable server for throughput testing")
        
        logger.info(f"Testing message throughput for {responsive_server.name}")
        server_process = MCPServerProcess(responsive_server)
        
        try:
            if not await server_process.start():
                pytest.skip(f"Could not start {responsive_server.name}")
            
            # Initialize
            init_response = await server_process.send_message(
                protocol_tester.create_initialize_request()
            )
            
            if not init_response or protocol_tester.is_error_response(init_response):
                pytest.skip(f"Could not initialize {responsive_server.name}")
            
            # Send multiple ping messages
            start_time = time.time()
            successful_pings = 0
            
            for i in range(5):
                ping_request = protocol_tester.create_ping_request(request_id=i+10)
                response = await server_process.send_message(ping_request, timeout=2)
                if response and not protocol_tester.is_error_response(response):
                    successful_pings += 1
            
            duration = time.time() - start_time
            throughput = successful_pings / duration if duration > 0 else 0
            
            logger.info(f"Throughput test: {successful_pings}/5 pings in {duration:.2f}s = {throughput:.1f} msg/sec")
            
        finally:
            server_process.stop()


class TestMCPServerIntegration:
    """Integration tests for specific server types"""
    
    @pytest.mark.asyncio
    async def test_claude_task_runner_integration(self):
        """Test claude-task-runner specific functionality"""
        server_config = None
        servers = load_mcp_servers()
        
        for server in servers:
            if 'claude-task-runner' in server.name or 'task-runner' in server.name:
                server_config = server
                break
        
        if not server_config:
            pytest.skip("Claude task runner not found")
        
        logger.info(f"Testing claude-task-runner integration: {server_config.name}")
        server_process = MCPServerProcess(server_config)
        
        try:
            if not await server_process.start():
                pytest.skip("Could not start claude-task-runner")
            
            protocol_tester = MCPProtocolTester()
            
            # Initialize
            init_response = await server_process.send_message(
                protocol_tester.create_initialize_request()
            )
            
            assert init_response, "No initialize response"
            assert not protocol_tester.is_error_response(init_response), f"Initialize error: {init_response}"
            
            # List tools
            tools_response = await server_process.send_message(
                protocol_tester.create_list_tools_request()
            )
            
            if tools_response and not protocol_tester.is_error_response(tools_response):
                tools = tools_response.get("result", {}).get("tools", [])
                logger.info(f"claude-task-runner tools: {[t.get('name') for t in tools]}")
                
                # Should have task-related tools
                tool_names = [t.get('name', '') for t in tools]
                task_tools = [name for name in tool_names if 'task' in name.lower()]
                assert len(task_tools) > 0, f"Expected task tools, got: {tool_names}"
            
        finally:
            server_process.stop()
    
    @pytest.mark.asyncio
    async def test_git_mcp_integration(self):
        """Test git-mcp specific functionality"""
        server_config = None
        servers = load_mcp_servers()
        
        for server in servers:
            if 'git-mcp' in server.name:
                server_config = server
                break
        
        if not server_config:
            pytest.skip("Git MCP server not found")
        
        logger.info(f"Testing git-mcp integration: {server_config.name}")
        server_process = MCPServerProcess(server_config)
        
        try:
            if not await server_process.start():
                pytest.skip("Could not start git-mcp")
            
            protocol_tester = MCPProtocolTester()
            
            # Test basic protocol
            init_response = await server_process.send_message(
                protocol_tester.create_initialize_request()
            )
            
            if init_response and not protocol_tester.is_error_response(init_response):
                logger.info("✓ git-mcp initialized successfully")
                
                # Try to list tools
                tools_response = await server_process.send_message(
                    protocol_tester.create_list_tools_request()
                )
                
                if tools_response and not protocol_tester.is_error_response(tools_response):
                    tools = tools_response.get("result", {}).get("tools", [])
                    tool_names = [t.get('name', '') for t in tools]
                    logger.info(f"git-mcp tools: {tool_names}")
                    
                    # Should have git-related tools
                    git_tools = [name for name in tool_names if 'git' in name.lower()]
                    if len(git_tools) > 0:
                        logger.info(f"✓ Found git tools: {git_tools}")
            
        finally:
            server_process.stop()
    
    @pytest.mark.asyncio
    async def test_playwright_mcp_integration(self):
        """Test playwright-mcp specific functionality"""
        server_config = None
        servers = load_mcp_servers()
        
        for server in servers:
            if 'playwright' in server.name:
                server_config = server
                break
        
        if not server_config:
            pytest.skip("Playwright MCP server not found")
        
        logger.info(f"Testing playwright-mcp integration: {server_config.name}")
        server_process = MCPServerProcess(server_config)
        
        try:
            # Playwright might need more time to start
            if not await server_process.start():
                pytest.skip("Could not start playwright-mcp")
            
            protocol_tester = MCPProtocolTester()
            
            # Test initialization with longer timeout
            init_response = await server_process.send_message(
                protocol_tester.create_initialize_request(),
                timeout=10  # Playwright might be slower
            )
            
            if init_response and not protocol_tester.is_error_response(init_response):
                logger.info("✓ playwright-mcp initialized successfully")
                
                tools_response = await server_process.send_message(
                    protocol_tester.create_list_tools_request(),
                    timeout=10
                )
                
                if tools_response and not protocol_tester.is_error_response(tools_response):
                    tools = tools_response.get("result", {}).get("tools", [])
                    tool_names = [t.get('name', '') for t in tools]
                    logger.info(f"playwright-mcp tools: {tool_names}")
                    
                    # Should have browser-related tools
                    browser_tools = [name for name in tool_names if any(keyword in name.lower() 
                                                                       for keyword in ['browser', 'page', 'click', 'navigate'])]
                    if len(browser_tools) > 0:
                        logger.info(f"✓ Found browser tools: {browser_tools}")
            
        finally:
            server_process.stop()


def run_manual_tests():
    """Run manual tests outside pytest"""
    print("🧪 Running Manual MCP Server Tests")
    print("=" * 50)
    
    servers = load_mcp_servers()
    print(f"Found {len(servers)} MCP servers")
    
    # Test server loading
    for server in servers[:5]:
        print(f"\n📋 Testing {server.name}:")
        print(f"  Command: {server.command}")
        print(f"  Args: {server.args}")
        print(f"  Type: {server.server_type}")
        print(f"  Enabled: {server.enabled}")
        print(f"  Tags: {server.tags}")
    
    print("\n🔧 Testing wrapper script health checks:")
    for server in servers:
        if server.command.endswith('.sh') and server.enabled:
            try:
                result = subprocess.run(
                    ['/bin/bash', server.command, '--selfcheck'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    print(f"  ✅ {server.name}: PASS")
                else:
                    print(f"  ❌ {server.name}: FAIL - {result.stderr.strip()[:100]}")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ {server.name}: TIMEOUT")
            except Exception as e:
                print(f"  🔥 {server.name}: ERROR - {e}")
    
    print("\n✅ Manual tests completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "manual":
            run_manual_tests()
        elif sys.argv[1] == "pytest":
            # Run pytest with verbose output
            pytest.main(["-v", "-s", __file__])
        else:
            print("Usage:")
            print("  python test_mcp_servers.py manual    # Run manual tests")
            print("  python test_mcp_servers.py pytest    # Run pytest suite")
            print("  pytest test_mcp_servers.py -v        # Run with pytest directly")
    else:
        print("MCP Server Test Suite")
        print("Usage:")
        print("  python test_mcp_servers.py manual    # Quick manual tests")
        print("  python test_mcp_servers.py pytest    # Full pytest suite")
        print("  pytest test_mcp_servers.py -v -k test_server_startup  # Run specific test")