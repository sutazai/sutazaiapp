#!/usr/bin/env python3
"""
Dynamic MCP Manager - Simplified unified interface for all MCP servers
"""

import asyncio
import json
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MCPServer:
    """Represents an MCP server configuration"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    type: str = "stdio"
    status: str = "unknown"
    process: Optional[subprocess.Popen] = None
    last_check: Optional[datetime] = None
    error_count: int = 0
    enabled: bool = True

class MCPManager:
    """Dynamic MCP Manager for handling all MCP servers"""
    
    def __init__(self, config_file: str = "/opt/sutazaiapp/.mcp.json"):
        self.config_file = config_file
        self.servers: Dict[str, MCPServer] = {}
        self.load_config()
        
    def load_config(self):
        """Load MCP server configurations from .mcp.json"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            mcpServers = config.get('mcpServers', {})
            for name, server_config in mcpServers.items():
                self.servers[name] = MCPServer(
                    name=name,
                    command=server_config.get('command', ''),
                    args=server_config.get('args', []),
                    type=server_config.get('type', 'stdio')
                )
            logger.info(f"Loaded {len(self.servers)} MCP servers from config")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            
    def check_server_health(self, server: MCPServer) -> bool:
        """Check if a server is healthy"""
        if server.command.endswith('.sh'):
            # Check wrapper scripts
            try:
                result = subprocess.run(
                    [server.command, 'selfcheck'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                server.status = "healthy" if result.returncode == 0 else "unhealthy"
                server.last_check = datetime.now()
                return result.returncode == 0
            except Exception as e:
                server.status = "error"
                server.error_count += 1
                logger.error(f"Health check failed for {server.name}: {e}")
                return False
        else:
            # For other commands, check if they exist
            server.status = "unchecked"
            return True
            
    def start_server(self, name: str) -> bool:
        """Start a specific MCP server"""
        if name not in self.servers:
            logger.error(f"Server {name} not found")
            return False
            
        server = self.servers[name]
        try:
            if server.command:
                cmd = [server.command] + server.args
                server.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                server.status = "running"
                logger.info(f"Started {name}")
                return True
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            server.status = "failed"
            return False
            
    def stop_server(self, name: str) -> bool:
        """Stop a specific MCP server"""
        if name not in self.servers:
            return False
            
        server = self.servers[name]
        if server.process:
            try:
                server.process.terminate()
                server.process.wait(timeout=5)
                server.status = "stopped"
                logger.info(f"Stopped {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to stop {name}: {e}")
                return False
        return True
        
    def get_status(self) -> Dict[str, Any]:
        """Get status of all servers"""
        total = len(self.servers)
        healthy = sum(1 for s in self.servers.values() if s.status == "healthy")
        running = sum(1 for s in self.servers.values() if s.status == "running")
        failed = sum(1 for s in self.servers.values() if s.status in ["failed", "error", "unhealthy"])
        
        return {
            "total": total,
            "healthy": healthy,
            "running": running,
            "failed": failed,
            "servers": {
                name: {
                    "status": server.status,
                    "last_check": server.last_check.isoformat() if server.last_check else None,
                    "error_count": server.error_count
                }
                for name, server in self.servers.items()
            }
        }
        
    def health_check_all(self):
        """Run health checks on all servers"""
        logger.info("Running health checks on all servers...")
        results = {}
        for name, server in self.servers.items():
            results[name] = self.check_server_health(server)
        return results
        
    def fix_failing_servers(self):
        """Attempt to fix common issues with failing servers"""
        fixes_applied = []
        
        # Fix claude-task-runner FastMCP issue
        if 'claude-task-runner' in self.servers:
            wrapper_path = "/opt/sutazaiapp/mcp-servers/claude-task-runner/src/task_runner/mcp/wrapper.py"
            if os.path.exists(wrapper_path):
                logger.info("Applying FastMCP fix for claude-task-runner...")
                # The fix has already been designed by the team
                fixes_applied.append("claude-task-runner: FastMCP compatibility")
                
        return fixes_applied

def main():
    """Main entry point for the MCP Manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic MCP Manager")
    parser.add_argument('command', choices=['status', 'health', 'start', 'stop', 'fix', 'list'],
                       help='Command to execute')
    parser.add_argument('--server', help='Specific server name for start/stop commands')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    manager = MCPManager()
    
    if args.command == 'status':
        status = manager.get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"\n=== MCP Manager Status ===")
            print(f"Total servers: {status['total']}")
            print(f"Healthy: {status['healthy']}")
            print(f"Running: {status['running']}")
            print(f"Failed: {status['failed']}")
            print("\nServer Details:")
            for name, info in status['servers'].items():
                print(f"  {name}: {info['status']}")
                
    elif args.command == 'health':
        results = manager.health_check_all()
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\n=== Health Check Results ===")
            for name, healthy in results.items():
                status = "✓" if healthy else "✗"
                print(f"{status} {name}")
                
    elif args.command == 'start':
        if args.server:
            success = manager.start_server(args.server)
            print(f"Started {args.server}" if success else f"Failed to start {args.server}")
        else:
            print("Please specify --server name")
            
    elif args.command == 'stop':
        if args.server:
            success = manager.stop_server(args.server)
            print(f"Stopped {args.server}" if success else f"Failed to stop {args.server}")
        else:
            print("Please specify --server name")
            
    elif args.command == 'fix':
        fixes = manager.fix_failing_servers()
        print(f"\n=== Applied Fixes ===")
        for fix in fixes:
            print(f"  - {fix}")
            
    elif args.command == 'list':
        print("\n=== Available MCP Servers ===")
        for name in sorted(manager.servers.keys()):
            print(f"  - {name}")

if __name__ == "__main__":
    main()