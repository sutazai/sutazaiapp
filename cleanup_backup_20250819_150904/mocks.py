#!/usr/bin/env python3
"""
Mock Services for MCP Testing

Comprehensive Mock services and utilities for MCP automation testing.
Provides realistic Mock implementations for external dependencies,
services, and components used in MCP testing scenarios.

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0
"""

import asyncio
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from unittest.Mock import AsyncMock, Mock
from dataclasses import dataclass, field


@dataclass
class MockHealthCheckResponse:
    """Mock health check response structure."""
    server_name: str
    healthy: bool
    response_time: float
    version: Optional[str] = None
    memory_usage_mb: Optional[int] = None
    uptime_seconds: Optional[int] = None
    error: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockDownloadResponse:
    """Mock download response structure."""
    package: str
    version: str
    size_bytes: int
    install_path: Path
    checksum_verified: bool
    download_time: float
    expected_checksum: Optional[str] = None
    actual_checksum: Optional[str] = None


class MockHealthChecker:
    """Mock health checker with configurable responses."""
    
    def __init__(self):
        self.responses: Dict[str, MockHealthCheckResponse] = {}
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
        self.default_response_time = 2.0
        self.failure_rate = 0.0  # 0.0 = never fail, 1.0 = always fail
        
    def set_server_response(
        self,
        server_name: str,
        healthy: bool = True,
        response_time: float = 2.0,
        version: str = "1.0.0",
        memory_usage_mb: int = 50,
        error: Optional[str] = None
    ) -> None:
        """Set predefined response for a specific server."""
        self.responses[server_name] = MockHealthCheckResponse(
            server_name=server_name,
            healthy=healthy,
            response_time=response_time,
            version=version,
            memory_usage_mb=memory_usage_mb,
            uptime_seconds=random.randint(100, 10000),
            error=error
        )
    
    def set_failure_rate(self, failure_rate: float) -> None:
        """Set random failure rate for health checks."""
        self.failure_rate = max(0.0, min(1.0, failure_rate))
    
    async def check_health(
        self,
        server_name: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Mock health check implementation."""
        self.call_count += 1
        start_time = time.time()
        
        # Simulate network delay
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Record call
        self.call_history.append({
            "server_name": server_name,
            "timeout": timeout,
            "timestamp": start_time,
            "call_number": self.call_count
        })
        
        # Check for predefined response
        if server_name in self.responses:
            response = self.responses[server_name]
        else:
            # Generate random response
            should_fail = random.random() < self.failure_rate
            response = MockHealthCheckResponse(
                server_name=server_name,
                healthy=not should_fail,
                response_time=random.uniform(1.0, 5.0),
                version=f"1.{random.randint(0, 5)}.{random.randint(0, 10)}",
                memory_usage_mb=random.randint(30, 100),
                uptime_seconds=random.randint(100, 10000),
                error="Health check failed" if should_fail else None
            )
        
        # Simulate timeout
        if response.response_time > timeout:
            return {
                "server_name": server_name,
                "healthy": False,
                "error": f"Health check timed out after {timeout}s",
                "response_time": timeout
            }
        
        # Build response dictionary
        result = {
            "server_name": response.server_name,
            "healthy": response.healthy,
            "response_time": response.response_time
        }
        
        if response.healthy:
            result.update({
                "version": response.version,
                "memory_usage_mb": response.memory_usage_mb,
                "uptime_seconds": response.uptime_seconds
            })
            if response.additional_data:
                result.update(response.additional_data)
        else:
            result["error"] = response.error or "Health check failed"
        
        return result
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of all health check calls."""
        return self.call_history.copy()
    
    def reset(self) -> None:
        """Reset call count and history."""
        self.call_count = 0
        self.call_history.clear()


class MockDownloadManager:
    """Mock download manager with configurable responses."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.downloads: Dict[str, MockDownloadResponse] = {}
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
        self.failure_rate = 0.0
        self.download_speed_mbps = 10.0  # Simulated download speed
        
    def set_package_response(
        self,
        package_name: str,
        version: str,
        size_bytes: int = 1024,
        checksum_verified: bool = True,
        install_path: Optional[Path] = None
    ) -> None:
        """Set predefined response for a specific package."""
        if install_path is None:
            install_path = self.base_path / f"{package_name.replace('/', '_')}-{version}"
        
        self.downloads[f"{package_name}@{version}"] = MockDownloadResponse(
            package=package_name,
            version=version,
            size_bytes=size_bytes,
            install_path=install_path,
            checksum_verified=checksum_verified,
            download_time=size_bytes / (self.download_speed_mbps * 1024 * 1024),  # Convert to seconds
            expected_checksum=f"sha256:{'a' * 64}",
            actual_checksum=f"sha256:{'a' * 64}" if checksum_verified else f"sha256:{'b' * 64}"
        )
    
    def set_failure_rate(self, failure_rate: float) -> None:
        """Set random failure rate for downloads."""
        self.failure_rate = max(0.0, min(1.0, failure_rate))
    
    def set_download_speed(self, speed_mbps: float) -> None:
        """Set simulated download speed in Mbps."""
        self.download_speed_mbps = max(0.1, speed_mbps)
    
    async def download_package(
        self,
        package_name: str,
        version: str,
        target_dir: Path
    ) -> Dict[str, Any]:
        """Mock package download implementation."""
        self.call_count += 1
        start_time = time.time()
        
        # Record call
        self.call_history.append({
            "package_name": package_name,
            "version": version,
            "target_dir": str(target_dir),
            "timestamp": start_time,
            "call_number": self.call_count
        })
        
        # Check for random failure
        if random.random() < self.failure_rate:
            await asyncio.sleep(random.uniform(1.0, 3.0))  # Simulate failed download time
            raise Exception(f"Mock download failure for {package_name}@{version}")
        
        # Check for predefined response
        package_key = f"{package_name}@{version}"
        if package_key in self.downloads:
            response = self.downloads[package_key]
        else:
            # Generate random response
            size_bytes = random.randint(1000, 100000)
            download_time = size_bytes / (self.download_speed_mbps * 1024 * 1024)
            
            response = MockDownloadResponse(
                package=package_name,
                version=version,
                size_bytes=size_bytes,
                install_path=target_dir / f"{package_name.split('/')[-1]}-{version}",
                checksum_verified=True,
                download_time=download_time,
                expected_checksum=f"sha256:{'c' * 64}",
                actual_checksum=f"sha256:{'c' * 64}"
            )
        
        # Simulate download time
        await asyncio.sleep(min(response.download_time, 0.5))  # Cap simulation time
        
        # Create Mock package files
        package_dir = response.install_path
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Create package.json
        package_json = {
            "name": package_name,
            "version": version,
            "description": f"Mock package {package_name}",
            "main": "index.js"
        }
        (package_dir / "package.json").write_text(json.dumps(package_json, indent=2))
        
        # Create main file
        (package_dir / "index.js").write_text(f"// Mock package {package_name} v{version}")
        
        return {
            "package": response.package,
            "version": response.version,
            "size_bytes": response.size_bytes,
            "install_path": response.install_path,
            "checksum_verified": response.checksum_verified,
            "download_time": response.download_time,
            "expected_checksum": response.expected_checksum,
            "actual_checksum": response.actual_checksum
        }
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of all download calls."""
        return self.call_history.copy()
    
    def reset(self) -> None:
        """Reset call count and history."""
        self.call_count = 0
        self.call_history.clear()


class MockProcessRunner:
    """Mock process runner for shell commands."""
    
    def __init__(self):
        self.commands: Dict[str, Dict[str, Any]] = {}
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
        self.default_success = True
        
    def set_command_response(
        self,
        command_pattern: str,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        execution_time: float = 1.0
    ) -> None:
        """Set predefined response for command pattern."""
        self.commands[command_pattern] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": execution_time
        }
    
    async def run_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Mock command execution."""
        self.call_count += 1
        start_time = time.time()
        
        command_str = " ".join(command)
        
        # Record call
        self.call_history.append({
            "command": command_str,
            "cwd": str(cwd) if cwd else None,
            "timeout": timeout,
            "timestamp": start_time,
            "call_number": self.call_count
        })
        
        # Find matching response
        response = None
        for pattern, resp in self.commands.items():
            if pattern in command_str:
                response = resp
                break
        
        # Default response if no match
        if response is None:
            if "npm" in command_str:
                if "install" in command_str:
                    response = {
                        "returncode": 0,
                        "stdout": "Mock npm install successful",
                        "stderr": "",
                        "execution_time": 2.0
                    }
                elif "list" in command_str:
                    response = {
                        "returncode": 0,
                        "stdout": json.dumps({"dependencies": {}}, indent=2),
                        "stderr": "",
                        "execution_time": 1.0
                    }
                else:
                    response = {
                        "returncode": 0,
                        "stdout": "Mock npm command successful",
                        "stderr": "",
                        "execution_time": 1.0
                    }
            else:
                response = {
                    "returncode": 0 if self.default_success else 1,
                    "stdout": f"Mock command output: {command_str}",
                    "stderr": "" if self.default_success else "Mock command failed",
                    "execution_time": 0.5
                }
        
        # Simulate execution time
        await asyncio.sleep(min(response["execution_time"], 0.1))  # Cap simulation time
        
        # Check for timeout
        if response["execution_time"] > timeout:
            return {
                "returncode": 124,  # Timeout exit code
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "execution_time": timeout
            }
        
        return response
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of all command calls."""
        return self.call_history.copy()
    
    def reset(self) -> None:
        """Reset call count and history."""
        self.call_count = 0
        self.call_history.clear()


class MockVersionManager:
    """Mock version manager for testing version operations."""
    
    def __init__(self):
        self.server_versions: Dict[str, Dict[str, Any]] = {}
        self.version_history: Dict[str, List[Dict[str, Any]]] = {}
        self.call_count = 0
        
    def set_server_version(
        self,
        server_name: str,
        current_version: str,
        staging_version: Optional[str] = None,
        available_versions: Optional[List[str]] = None
    ) -> None:
        """Set version information for a server."""
        if available_versions is None:
            available_versions = [current_version]
        
        self.server_versions[server_name] = {
            "current_version": current_version,
            "staging_version": staging_version,
            "available_versions": available_versions,
            "last_update": time.time()
        }
    
    async def get_version_status(self, server_name: str) -> Dict[str, Any]:
        """Get version status for a server."""
        self.call_count += 1
        
        if server_name in self.server_versions:
            return self.server_versions[server_name].copy()
        else:
            return {
                "current_version": "1.0.0",
                "staging_version": None,
                "available_versions": ["1.0.0"],
                "last_update": time.time()
            }
    
    async def record_activation(self, server_name: str, version: str) -> None:
        """Record version activation."""
        self.call_count += 1
        
        # Update current version
        if server_name not in self.server_versions:
            self.server_versions[server_name] = {
                "current_version": version,
                "staging_version": None,
                "available_versions": [version],
                "last_update": time.time()
            }
        else:
            self.server_versions[server_name]["current_version"] = version
            self.server_versions[server_name]["last_update"] = time.time()
        
        # Add to history
        if server_name not in self.version_history:
            self.version_history[server_name] = []
        
        self.version_history[server_name].append({
            "version": version,
            "timestamp": time.time(),
            "action": "activation"
        })
    
    async def rollback_server(
        self,
        server_name: str,
        target_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mock rollback operation."""
        self.call_count += 1
        
        current_status = await self.get_version_status(server_name)
        current_version = current_status["current_version"]
        
        # Determine rollback target
        if target_version is None:
            # Find previous version from history
            history = self.version_history.get(server_name, [])
            if len(history) > 1:
                target_version = history[-2]["version"]
            else:
                target_version = "1.0.0"  # Default fallback
        
        # Simulate rollback delay
        await asyncio.sleep(0.1)
        
        # Update version
        await self.record_activation(server_name, target_version)
        
        return {
            "success": True,
            "server_name": server_name,
            "rolled_back_from": current_version,
            "rolled_back_to": target_version,
            "rollback_time": 0.1
        }
    
    def get_call_count(self) -> int:
        """Get total number of calls made."""
        return self.call_count
    
    def reset(self) -> None:
        """Reset all state."""
        self.server_versions.clear()
        self.version_history.clear()
        self.call_count = 0


class MockServices:
    """Centralized Mock services manager."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.health_checker = MockHealthChecker()
        self.download_manager = MockDownloadManager(base_path)
        self.process_runner = MockProcessRunner()
        self.version_manager = MockVersionManager()
        
        self.setup_default_responses()
    
    def setup_default_responses(self) -> None:
        """Setup default responses for common scenarios."""
        # Health checker defaults
        self.health_checker.set_server_response("files", healthy=True, version="1.2.3")
        self.health_checker.set_server_response("postgres", healthy=True, version="2.1.0")
        self.health_checker.set_server_response("test-server", healthy=True, version="0.5.0")
        
        # Download manager defaults
        self.download_manager.set_package_response(
            "@modelcontextprotocol/server-filesystem",
            "1.2.3",
            size_bytes=50000
        )
        self.download_manager.set_package_response(
            "@modelcontextprotocol/server-postgres",
            "2.1.0", 
            size_bytes=75000
        )
        
        # Process runner defaults
        self.process_runner.set_command_response(
            "npm install",
            returncode=0,
            stdout="Mock npm install successful",
            execution_time=2.0
        )
        self.process_runner.set_command_response(
            "npm list",
            returncode=0,
            stdout=json.dumps({"dependencies": {}}, indent=2),
            execution_time=1.0
        )
        
        # Version manager defaults
        self.version_manager.set_server_version("files", "1.2.3", available_versions=["1.0.0", "1.1.0", "1.2.3"])
        self.version_manager.set_server_version("postgres", "2.1.0", available_versions=["2.0.0", "2.1.0"])
    
    def enable_failure_mode(
        self,
        health_failure_rate: float = 0.1,
        download_failure_rate: float = 0.05
    ) -> None:
        """Enable random failures for testing resilience."""
        self.health_checker.set_failure_rate(health_failure_rate)
        self.download_manager.set_failure_rate(download_failure_rate)
    
    def get_all_call_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get call history from all Mock services."""
        return {
            "health_checker": self.health_checker.get_call_history(),
            "download_manager": self.download_manager.get_call_history(),
            "process_runner": self.process_runner.get_call_history()
        }
    
    def reset_all(self) -> None:
        """Reset all Mock services."""
        self.health_checker.reset()
        self.download_manager.reset()
        self.process_runner.reset()
        self.version_manager.reset()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics from all Mock services."""
        return {
            "health_checker_calls": self.health_checker.call_count,
            "download_manager_calls": self.download_manager.call_count,
            "process_runner_calls": self.process_runner.call_count,
            "version_manager_calls": self.version_manager.get_call_count(),
            "total_calls": (
                self.health_checker.call_count +
                self.download_manager.call_count +
                self.process_runner.call_count +
                self.version_manager.get_call_count()
            )
        }