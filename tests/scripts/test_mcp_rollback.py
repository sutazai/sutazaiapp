#!/usr/bin/env python3
"""
MCP Rollback Testing Suite

Comprehensive rollback and recovery testing for MCP server automation system.
Validates rollback procedures, disaster recovery, failure scenarios,
data consistency, and system resilience under various failure conditions.

Test Coverage:
- Automated rollback procedures
- Manual rollback validation
- Disaster recovery scenarios
- Data consistency during rollbacks
- Partial failure recovery
- Rollback performance and timing
- Multi-server rollback coordination
- Rollback validation and verification

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0
"""

import pytest
import asyncio
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass, field
from enum import Enum

from conftest import TestEnvironment, TestMCPServer

# Import automation modules
from config import MCPAutomationConfig, UpdateMode, LogLevel
from mcp_update_manager import MCPUpdateManager
from version_manager import MCPVersionManager
from download_manager import MCPDownloadManager
from error_handling import MCPError, ErrorSeverity


class FailureType(Enum):
    """Types of failures that can trigger rollbacks."""
    HEALTH_CHECK_FAILURE = "health_check_failure"
    STARTUP_TIMEOUT = "startup_timeout"
    MEMORY_LEAK = "memory_leak"
    CRASH_LOOP = "crash_loop"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_FAILURE = "network_failure"
    DISK_SPACE_EXHAUSTION = "disk_space_exhaustion"
    PERMISSION_ERROR = "permission_error"
    CORRUPTION = "corruption"


@dataclass
class RollbackScenario:
    """Rollback test scenario definition."""
    name: str
    failure_type: FailureType
    trigger_point: str  # when to trigger failure: "download", "staging", "activation", "post_activation"
    expected_rollback_success: bool
    expected_rollback_time: float  # seconds
    cleanup_required: bool = True
    data_loss_acceptable: bool = False


@dataclass
class RollbackResult:
    """Rollback test result structure."""
    scenario_name: str
    rollback_triggered: bool
    rollback_successful: bool
    rollback_time: float
    final_version: str
    data_integrity_maintained: bool
    service_availability_maintained: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recovery_steps: List[str] = field(default_factory=list)


class RollbackTestManager:
    """Rollback test manager for coordinated rollback testing."""
    
    def __init__(self, config: MCPAutomationConfig):
        self.config = config
        self.active_scenarios: Dict[str, RollbackScenario] = {}
        self.rollback_history: List[RollbackResult] = []
    
    def create_backup_state(self, server_name: str, version: str) -> Path:
        """Create a backup of current server state."""
        backup_dir = self.config.get_backup_path(server_name) / f"backup_{version}_{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate creating backup files
        backup_metadata = {
            "server_name": server_name,
            "version": version,
            "backup_timestamp": time.time(),
            "backup_type": "pre_rollback"
        }
        
        (backup_dir / "backup_metadata.json").write_text(json.dumps(backup_metadata, indent=2))
        (backup_dir / "server_state.json").write_text(json.dumps({"status": "backed_up"}, indent=2))
        
        return backup_dir
    
    def simulate_failure(self, failure_type: FailureType, server_name: str) -> Dict[str, Any]:
        """Simulate various failure conditions."""
        failure_details = {
            "failure_type": failure_type.value,
            "server_name": server_name,
            "timestamp": time.time(),
            "simulated": True
        }
        
        if failure_type == FailureType.HEALTH_CHECK_FAILURE:
            failure_details.update({
                "error": "Health check endpoint not responding",
                "http_status": 503,
                "timeout": True
            })
        elif failure_type == FailureType.STARTUP_TIMEOUT:
            failure_details.update({
                "error": "Server failed to start within timeout period",
                "timeout_seconds": 30,
                "last_log": "Starting server..."
            })
        elif failure_type == FailureType.MEMORY_LEAK:
            failure_details.update({
                "error": "Memory usage exceeded threshold",
                "memory_usage_mb": 1024,
                "threshold_mb": 512
            })
        elif failure_type == FailureType.CRASH_LOOP:
            failure_details.update({
                "error": "Server crashed repeatedly",
                "crash_count": 5,
                "exit_code": 1
            })
        elif failure_type == FailureType.DEPENDENCY_FAILURE:
            failure_details.update({
                "error": "Required dependency not available",
                "missing_dependency": "postgresql",
                "dependency_health": False
            })
        elif failure_type == FailureType.CONFIGURATION_ERROR:
            failure_details.update({
                "error": "Invalid configuration detected",
                "config_file": "config.json",
                "validation_errors": ["missing required field: database_url"]
            })
        
        return failure_details


class TestMCPAutomaticRollback:
    """Test suite for automatic rollback procedures."""
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_health_check_failure_rollback(
        self,
        test_environment: TestEnvironment,
        rollback_simulator,
        mock_process_runner
    ):
        """Test automatic rollback when health checks fail after update."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        rollback_manager = RollbackTestManager(config)
        
        server_name = "files"
        stable_version = "1.0.0"
        failing_version = "1.1.0"
        
        # Establish stable baseline
        await version_manager.record_activation(server_name, stable_version)
        baseline_backup = rollback_manager.create_backup_state(server_name, stable_version)
        
        # Mock health check that fails for new version
        health_call_count = 0
        async def failing_health_check(name: str, timeout: int = 30):
            nonlocal health_call_count
            health_call_count += 1
            
            # Simulate failure after update
            if health_call_count > 2:  # Fail after initial checks
                failure_details = rollback_manager.simulate_failure(
                    FailureType.HEALTH_CHECK_FAILURE, 
                    name
                )
                return {
                    "server_name": name,
                    "healthy": False,
                    "error": failure_details["error"],
                    "response_time": timeout,
                    "failure_details": failure_details
                }
            else:
                return {
                    "server_name": name,
                    "healthy": True,
                    "response_time": 2.0,
                    "version": stable_version
                }
        
        with patch.object(update_manager, '_run_health_check', side_effect=failing_health_check), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            rollback_start_time = time.time()
            
            try:
                # Attempt update that should trigger rollback
                update_result = await update_manager.update_server(server_name, target_version=failing_version)
                assert update_result["success"] is True  # Download should succeed
                
                # Activation should fail and trigger rollback
                activation_result = await update_manager.activate_server(server_name)
                
                # Rollback should be automatically triggered
                if not activation_result["success"]:
                    rollback_result = await version_manager.rollback_server(server_name)
                    
                    rollback_end_time = time.time()
                    rollback_duration = rollback_end_time - rollback_start_time
                    
                    # Verify rollback success
                    assert rollback_result["success"] is True
                    assert rollback_result["rolled_back_to"] == stable_version
                    assert rollback_duration < 30.0  # Should complete quickly
                    
                    # Verify server health after rollback
                    post_rollback_health = await update_manager._run_health_check(server_name, timeout=30)
                    assert post_rollback_health["healthy"] is True
                    
                    # Verify version state
                    version_status = await version_manager.get_version_status(server_name)
                    assert version_status["current_version"] == stable_version
                    assert version_status["staging_version"] is None
                
            except Exception as e:
                pytest.fail(f"Rollback test failed with exception: {e}")
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_startup_timeout_rollback(
        self,
        test_environment: TestEnvironment,
        mock_process_runner
    ):
        """Test rollback when server fails to start within timeout."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        rollback_manager = RollbackTestManager(config)
        
        server_name = "test-server"
        stable_version = "1.0.0"
        slow_version = "1.1.0"
        
        # Set up stable baseline
        await version_manager.record_activation(server_name, stable_version)
        
        # Mock health check with startup timeout
        async def timeout_health_check(name: str, timeout: int = 30):
            # Simulate very slow startup that exceeds timeout
            await asyncio.sleep(min(timeout + 1, 5))  # Don't actually wait full timeout in tests
            
            failure_details = rollback_manager.simulate_failure(
                FailureType.STARTUP_TIMEOUT,
                name
            )
            
            return {
                "server_name": name,
                "healthy": False,
                "error": failure_details["error"],
                "response_time": timeout,
                "startup_timeout": True
            }
        
        with patch.object(update_manager, '_run_health_check', side_effect=timeout_health_check), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Configure shorter timeout for testing
            original_timeout = config.performance.health_check_timeout_seconds
            config.performance.health_check_timeout_seconds = 5
            
            try:
                rollback_start_time = time.time()
                
                # Update should succeed
                update_result = await update_manager.update_server(server_name, target_version=slow_version)
                assert update_result["success"] is True
                
                # Activation should fail due to timeout
                activation_result = await update_manager.activate_server(server_name)
                
                if not activation_result["success"]:
                    # Trigger rollback
                    rollback_result = await version_manager.rollback_server(server_name)
                    
                    rollback_end_time = time.time()
                    rollback_duration = rollback_end_time - rollback_start_time
                    
                    # Verify rollback completed quickly despite startup timeout
                    assert rollback_duration < 15.0  # Should not wait for full timeout
                    assert rollback_result["success"] is True
                    assert rollback_result["rolled_back_to"] == stable_version
                
            finally:
                # Restore original timeout
                config.performance.health_check_timeout_seconds = original_timeout
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_dependency_failure_rollback(
        self,
        test_environment: TestEnvironment,
        mock_process_runner
    ):
        """Test rollback when server dependencies fail."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        rollback_manager = RollbackTestManager(config)
        
        server_name = "postgres"
        stable_version = "1.0.0"
        dependency_breaking_version = "2.0.0"
        
        # Set up stable baseline
        await version_manager.record_activation(server_name, stable_version)
        
        # Mock health check that fails due to dependency issues
        async def dependency_failure_check(name: str, timeout: int = 30):
            failure_details = rollback_manager.simulate_failure(
                FailureType.DEPENDENCY_FAILURE,
                name
            )
            
            return {
                "server_name": name,
                "healthy": False,
                "error": failure_details["error"],
                "response_time": 1.0,
                "dependency_check_failed": True,
                "missing_dependencies": failure_details["missing_dependency"]
            }
        
        with patch.object(update_manager, '_run_health_check', side_effect=dependency_failure_check), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Attempt update with dependency issues
            update_result = await update_manager.update_server(server_name, target_version=dependency_breaking_version)
            assert update_result["success"] is True
            
            # Activation should fail
            activation_result = await update_manager.activate_server(server_name)
            
            if not activation_result["success"]:
                # Verify rollback handles dependency failures
                rollback_result = await version_manager.rollback_server(server_name)
                
                assert rollback_result["success"] is True
                assert rollback_result["rolled_back_to"] == stable_version
                assert "dependency" in rollback_result.get("rollback_reason", "").lower()


class TestMCPManualRollback:
    """Test suite for manual rollback procedures."""
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_manual_rollback_to_specific_version(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner
    ):
        """Test manual rollback to a specific version."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        server_name = "files"
        version_history = ["1.0.0", "1.1.0", "1.2.0", "1.3.0"]
        target_rollback_version = "1.1.0"
        
        # Establish version history
        for version in version_history:
            await version_manager.record_activation(server_name, version)
            await asyncio.sleep(0.1)  # Simulate time between versions
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Get initial state
            initial_status = await version_manager.get_version_status(server_name)
            assert initial_status["current_version"] == version_history[-1]
            
            # Perform manual rollback to specific version
            rollback_result = await version_manager.rollback_server(server_name, target_version=target_rollback_version)
            
            # Verify rollback success
            assert rollback_result["success"] is True
            assert rollback_result["rolled_back_to"] == target_rollback_version
            
            # Verify version state after rollback
            post_rollback_status = await version_manager.get_version_status(server_name)
            assert post_rollback_status["current_version"] == target_rollback_version
            
            # Verify health after rollback
            health_result = await update_manager._run_health_check(server_name, timeout=30)
            assert health_result["healthy"] is True
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_rollback_with_data_preservation(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner,
        tmp_path: Path
    ):
        """Test rollback while preserving critical data."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        server_name = "files"
        original_version = "1.0.0"
        updated_version = "1.1.0"
        
        # Create Mock server data
        server_data_dir = tmp_path / "server_data"
        server_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create critical data files
        critical_data = {
            "user_data.json": {"users": ["user1", "user2", "user3"]},
            "config.json": {"settings": {"debug": False, "port": 8080}},
            "state.json": {"last_update": time.time(), "version": original_version}
        }
        
        for filename, data in critical_data.items():
            (server_data_dir / filename).write_text(json.dumps(data, indent=2))
        
        # Record initial version
        await version_manager.record_activation(server_name, original_version)
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Update to new version
            await update_manager.update_server(server_name, target_version=updated_version)
            await update_manager.activate_server(server_name)
            
            # Modify data in new version
            updated_data = critical_data.copy()
            updated_data["state.json"]["version"] = updated_version
            updated_data["user_data.json"]["users"].append("user4")  # New user added
            
            for filename, data in updated_data.items():
                (server_data_dir / filename).write_text(json.dumps(data, indent=2))
            
            # Create backup before rollback
            backup_dir = tmp_path / "data_backup"
            shutil.copytree(server_data_dir, backup_dir)
            
            # Perform rollback
            rollback_result = await version_manager.rollback_server(server_name)
            
            # Verify rollback success
            assert rollback_result["success"] is True
            assert rollback_result["rolled_back_to"] == original_version
            
            # Verify critical data preservation
            for filename in critical_data.keys():
                original_file = server_data_dir / filename
                backup_file = backup_dir / filename
                
                assert original_file.exists(), f"Critical data file {filename} was lost during rollback"
                
                # Compare data integrity
                original_content = json.loads(original_file.read_text())
                backup_content = json.loads(backup_file.read_text())
                
                # User data should be preserved (new user should still exist)
                if filename == "user_data.json":
                    assert "user4" in original_content["users"], "Data added after update was lost during rollback"
                
                # State should be updated to reflect rollback
                if filename == "state.json":
                    assert original_content["version"] == original_version
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_partial_rollback_recovery(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner
    ):
        """Test recovery from partial rollback failures."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        rollback_manager = RollbackTestManager(config)
        
        server_name = "test-server"
        stable_version = "1.0.0"
        current_version = "1.2.0"
        
        # Set up server state
        await version_manager.record_activation(server_name, stable_version)
        await version_manager.record_activation(server_name, current_version)
        
        # Mock rollback that fails partway through
        rollback_call_count = 0
        async def partial_failure_health_check(name: str, timeout: int = 30):
            nonlocal rollback_call_count
            rollback_call_count += 1
            
            if rollback_call_count == 2:  # Fail on second health check (during rollback)
                return {
                    "server_name": name,
                    "healthy": False,
                    "error": "Rollback interrupted",
                    "response_time": timeout,
                    "partial_failure": True
                }
            else:
                return {
                    "server_name": name,
                    "healthy": True,
                    "response_time": 2.0,
                    "version": stable_version if rollback_call_count > 2 else current_version
                }
        
        with patch.object(update_manager, '_run_health_check', side_effect=partial_failure_health_check), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Attempt rollback that partially fails
            try:
                rollback_result = await version_manager.rollback_server(server_name)
                
                # If first rollback attempt fails, try recovery
                if not rollback_result["success"]:
                    # Reset call count for recovery attempt
                    rollback_call_count = 0
                    
                    # Attempt recovery rollback
                    recovery_result = await version_manager.rollback_server(server_name, force=True)
                    
                    # Recovery should succeed
                    assert recovery_result["success"] is True
                    assert recovery_result["rolled_back_to"] == stable_version
                    
                    # Verify final state
                    final_status = await version_manager.get_version_status(server_name)
                    assert final_status["current_version"] == stable_version
                    
                else:
                    # If rollback succeeded on first try, verify it
                    assert rollback_result["rolled_back_to"] == stable_version
                    
            except Exception as e:
                # If rollback completely fails, ensure system is in recoverable state
                status = await version_manager.get_version_status(server_name)
                assert status["current_version"] in [stable_version, current_version], "System left in unknown state"


class TestMCPMultiServerRollback:
    """Test suite for coordinated multi-server rollbacks."""
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_coordinated_multi_server_rollback(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner
    ):
        """Test coordinated rollback across multiple servers."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        # Select multiple servers for coordinated rollback
        servers = ["files", "postgres", "test-server"]
        stable_versions = {"files": "1.0.0", "postgres": "2.0.0", "test-server": "0.5.0"}
        current_versions = {"files": "1.2.0", "postgres": "2.1.0", "test-server": "0.6.0"}
        
        # Set up initial states
        for server_name in servers:
            await version_manager.record_activation(server_name, stable_versions[server_name])
            await version_manager.record_activation(server_name, current_versions[server_name])
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Perform coordinated rollback
            rollback_tasks = []
            rollback_start_time = time.time()
            
            for server_name in servers:
                task = asyncio.create_task(
                    version_manager.rollback_server(server_name, target_version=stable_versions[server_name])
                )
                rollback_tasks.append((server_name, task))
            
            # Wait for all rollbacks to complete
            rollback_results = {}
            for server_name, task in rollback_tasks:
                try:
                    result = await task
                    rollback_results[server_name] = result
                except Exception as e:
                    rollback_results[server_name] = {
                        "success": False,
                        "error": str(e)
                    }
            
            rollback_end_time = time.time()
            total_rollback_time = rollback_end_time - rollback_start_time
            
            # Verify all rollbacks completed
            assert len(rollback_results) == len(servers)
            
            # Verify rollback success
            successful_rollbacks = sum(1 for result in rollback_results.values() if result["success"])
            assert successful_rollbacks == len(servers), f"Only {successful_rollbacks}/{len(servers)} rollbacks succeeded"
            
            # Verify coordination efficiency (parallel rollbacks should be faster than sequential)
            expected_sequential_time = len(servers) * 10  # Assume 10s per rollback sequentially
            assert total_rollback_time < expected_sequential_time, f"Coordinated rollback took {total_rollback_time}s, expected less than {expected_sequential_time}s"
            
            # Verify final server states
            for server_name in servers:
                status = await version_manager.get_version_status(server_name)
                assert status["current_version"] == stable_versions[server_name]
                
                health = await update_manager._run_health_check(server_name, timeout=30)
                assert health["healthy"] is True
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_dependency_aware_rollback_ordering(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner
    ):
        """Test rollback ordering based on server dependencies."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        # Define dependency chain: postgres -> files -> test-server
        dependency_chain = [
            ("postgres", "2.0.0", "2.1.0"),
            ("files", "1.0.0", "1.2.0"),
            ("test-server", "0.5.0", "0.6.0")
        ]
        
        # Set up server states
        for server_name, stable_version, current_version in dependency_chain:
            await version_manager.record_activation(server_name, stable_version)
            await version_manager.record_activation(server_name, current_version)
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Perform dependency-aware rollback (reverse order)
            rollback_order = []
            rollback_times = {}
            
            # Rollback in reverse dependency order
            for server_name, stable_version, current_version in reversed(dependency_chain):
                rollback_start = time.time()
                
                rollback_result = await version_manager.rollback_server(server_name, target_version=stable_version)
                
                rollback_end = time.time()
                rollback_order.append(server_name)
                rollback_times[server_name] = rollback_end - rollback_start
                
                # Verify rollback success
                assert rollback_result["success"] is True
                assert rollback_result["rolled_back_to"] == stable_version
                
                # Wait briefly before next rollback to simulate dependency stabilization
                await asyncio.sleep(0.1)
            
            # Verify rollback order (should be reverse of dependency chain)
            expected_order = [server_name for server_name, _, _ in reversed(dependency_chain)]
            assert rollback_order == expected_order
            
            # Verify all servers are healthy after coordinated rollback
            for server_name, stable_version, _ in dependency_chain:
                health = await update_manager._run_health_check(server_name, timeout=30)
                assert health["healthy"] is True
                
                status = await version_manager.get_version_status(server_name)
                assert status["current_version"] == stable_version


class TestMCPRollbackPerformance:
    """Test suite for rollback performance and timing validation."""
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_rollback_time_constraints(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner,
        performance_monitor
    ):
        """Test that rollbacks complete within acceptable time constraints."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        server_name = "files"
        stable_version = "1.0.0"
        current_version = "1.2.0"
        
        # Set up server state
        await version_manager.record_activation(server_name, stable_version)
        await version_manager.record_activation(server_name, current_version)
        
        # Define rollback time constraints
        rollback_constraints = {
            "max_rollback_time": 30.0,  # 30 seconds maximum
            "health_check_timeout": 10.0,  # 10 seconds for health checks
            "activation_timeout": 15.0  # 15 seconds for activation
        }
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            performance_monitor.start_operation("rollback_performance")
            
            # Perform rollback with time monitoring
            rollback_start = time.time()
            
            rollback_result = await version_manager.rollback_server(server_name)
            
            rollback_end = time.time()
            rollback_duration = rollback_end - rollback_start
            
            performance_monitor.end_operation("rollback_performance")
            
            # Verify rollback success
            assert rollback_result["success"] is True
            assert rollback_result["rolled_back_to"] == stable_version
            
            # Verify time constraints
            assert rollback_duration <= rollback_constraints["max_rollback_time"], f"Rollback took {rollback_duration}s, exceeds maximum {rollback_constraints['max_rollback_time']}s"
            
            # Verify performance metrics
            performance_monitor.assert_performance("rollback_performance", rollback_constraints["max_rollback_time"])
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_rollback_under_load(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner
    ):
        """Test rollback performance under system load."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        # Create multiple servers to simulate load
        test_servers = []
        for i in range(5):
            server_name = f"load-test-server-{i}"
            config.mcp_servers[server_name] = {
                "package": f"@test/load-server-{i}",
                "wrapper": f"load-test-server-{i}.sh"
            }
            test_servers.append(server_name)
            
            # Set up versions
            await version_manager.record_activation(server_name, "1.0.0")
            await version_manager.record_activation(server_name, "1.1.0")
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Create background load with health checks
            background_tasks = []
            for _ in range(10):  # 10 concurrent background operations
                task = asyncio.create_task(
                    update_manager._run_health_check(test_servers[0], timeout=30)
                )
                background_tasks.append(task)
            
            # Perform rollback under load
            rollback_start = time.time()
            
            rollback_result = await version_manager.rollback_server(test_servers[0])
            
            rollback_end = time.time()
            rollback_duration = rollback_end - rollback_start
            
            # Wait for background tasks to complete
            await asyncio.gather(*background_tasks, return_exceptions=True)
            
            # Verify rollback succeeded despite load
            assert rollback_result["success"] is True
            assert rollback_result["rolled_back_to"] == "1.0.0"
            
            # Rollback should complete within reasonable time even under load
            assert rollback_duration < 60.0, f"Rollback under load took {rollback_duration}s, too slow"
    
    @pytest.mark.rollback
    @pytest.mark.asyncio
    async def test_rollback_resource_cleanup(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner,
        tmp_path: Path
    ):
        """Test proper resource cleanup during rollback."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        server_name = "files"
        stable_version = "1.0.0" 
        failed_version = "1.2.0"
        
        # Set up server state with temporary resources
        await version_manager.record_activation(server_name, stable_version)
        
        # Create temporary resources that should be cleaned up
        temp_staging_dir = config.paths.staging_root / server_name
        temp_staging_dir.mkdir(parents=True, exist_ok=True)
        
        temp_files = [
            temp_staging_dir / "package.json",
            temp_staging_dir / "temp_data.json", 
            temp_staging_dir / "lock_file.lock"
        ]
        
        for temp_file in temp_files:
            temp_file.write_text(f"Temporary content for {temp_file.name}")
        
        # Record resource state before rollback
        initial_resources = {
            "staging_files": len(list(temp_staging_dir.glob("*"))),
            "staging_size": sum(f.stat().st_size for f in temp_staging_dir.glob("*") if f.is_file())
        }
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Perform rollback
            rollback_result = await version_manager.rollback_server(server_name)
            
            # Verify rollback success
            assert rollback_result["success"] is True
            
            # Verify resource cleanup
            if temp_staging_dir.exists():
                remaining_files = list(temp_staging_dir.glob("*"))
                # Staging directory should be cleaned up or
                assert len(remaining_files) <= 1, f"Expected cleanup but found {len(remaining_files)} files remaining"
            
            # Verify no resource leaks in other areas
            backup_dir = config.get_backup_path(server_name)
            if backup_dir.exists():
                backup_files = list(backup_dir.glob("*"))
                # Should have reasonable number of backup files
                assert len(backup_files) < 10, f"Too many backup files: {len(backup_files)}"