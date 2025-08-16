#!/usr/bin/env python3
"""
MCP Integration Testing Suite

Comprehensive integration tests for MCP server automation system.
Tests end-to-end workflows including server installation, updates,
health validation, and coordination between system components.

Test Coverage:
- MCP server deployment workflows
- Update and rollback procedures
- Configuration management
- Service coordination
- Error handling and recovery
- System integration points

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test import AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, patch

from conftest import (
    TestEnvironment, TestMCPServer, test_mcp_server,
    isolated_test_environment
)

# Import automation modules
from config import MCPAutomationConfig, UpdateMode, LogLevel
from mcp_update_manager import MCPUpdateManager
from version_manager import MCPVersionManager
from download_manager import MCPDownloadManager
from error_handling import MCPError, ErrorSeverity


class TestMCPIntegration:
    """Integration test suite for MCP automation system."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_server_deployment_workflow(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner,
        performance_monitor
    ):
        """Test complete MCP server deployment from start to finish."""
        performance_monitor.start_operation("complete_deployment")
        
        config = test_environment.config
        server_name = "files"
        
        # Initialize managers
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        download_manager = MCPDownloadManager(config)
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test external dependencies
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Step 1: Check initial state
            initial_status = await update_manager.get_server_status(server_name)
            assert initial_status["server_name"] == server_name
            
            # Step 2: Download and stage server
            download_result = await update_manager.update_server(
                server_name, 
                target_version="1.2.3",
                dry_run=False
            )
            
            assert download_result["success"] is True
            assert download_result["server_name"] == server_name
            assert download_result["version"] == "1.2.3"
            assert download_result["staging_path"].exists()
            
            # Step 3: Validate staging environment
            staging_path = config.get_staging_path(server_name)
            assert staging_path.exists()
            assert (staging_path / "package.json").exists()
            
            # Step 4: Activate staged server
            activation_result = await update_manager.activate_server(server_name)
            assert activation_result["success"] is True
            assert activation_result["activated_version"] == "1.2.3"
            
            # Step 5: Verify deployment success
            final_status = await update_manager.get_server_status(server_name)
            assert final_status["active_version"] == "1.2.3"
            assert final_status["health_status"]["healthy"] is True
            
        performance_monitor.end_operation("complete_deployment")
        performance_monitor.assert_performance("complete_deployment", 60.0)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_server_updates(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test updating multiple MCP servers concurrently."""
        config = test_environment.config
        server_names = ["files", "postgres", "test-server"]
        
        update_manager = MCPUpdateManager(config)
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Start concurrent updates
            update_tasks = []
            for server_name in server_names:
                task = asyncio.create_task(
                    update_manager.update_server(server_name, target_version="1.0.0")
                )
                update_tasks.append(task)
            
            # Wait for all updates to complete
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            
            # Verify all updates succeeded
            for i, result in enumerate(results):
                assert not isinstance(result, Exception), f"Update failed for {server_names[i]}: {result}"
                assert result["success"] is True
                assert result["server_name"] == server_names[i]
                
                # Verify staging paths exist
                staging_path = config.get_staging_path(server_names[i])
                assert staging_path.exists()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_update_with_dependency_chain(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test updating servers with dependency relationships."""
        config = test_environment.config
        
        # Define dependency chain: postgres -> files -> test-server
        dependency_chain = [
            ("postgres", "2.1.0"),
            ("files", "1.2.3"),  
            ("test-server", "0.5.0")
        ]
        
        update_manager = MCPUpdateManager(config)
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Update servers in dependency order
            for server_name, version in dependency_chain:
                result = await update_manager.update_server(server_name, target_version=version)
                assert result["success"] is True
                
                # Activate immediately to satisfy dependencies
                activation_result = await update_manager.activate_server(server_name)
                assert activation_result["success"] is True
                
                # Verify health after each activation
                health_status = await update_manager._run_health_check(server_name, timeout=30)
                assert health_status["healthy"] is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_configuration_management_integration(
        self,
        test_environment: TestEnvironment,
        tmp_path: Path
    ):
        """Test configuration management and environment integration."""
        config = test_environment.config
        
        # Test configuration serialization
        config_file = tmp_path / "test_config.json"
        config.save_config(config_file)
        assert config_file.exists()
        
        # Verify configuration content
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        assert "security" in saved_config
        assert "performance" in saved_config
        assert "paths" in saved_config
        assert "mcp_servers" in saved_config
        
        # Test configuration loading
        new_config = MCPAutomationConfig(config_file)
        assert new_config.update_mode == config.update_mode
        assert new_config.security.verify_checksums == config.security.verify_checksums
        assert len(new_config.mcp_servers) == len(config.mcp_servers)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test comprehensive error handling and recovery mechanisms."""
        config = test_environment.config
        server_name = "failing-server"
        
        update_manager = MCPUpdateManager(config)
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test failing health check
        async def failing_health_check(name: str, timeout: int = 30):
            if name == "failing-server":
                return {
                    "server_name": name,
                    "healthy": False,
                    "error": "Health check failed",
                    "response_time": timeout
                }
            return await Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker(name, timeout)
        
        with patch.object(update_manager, '_run_health_check', side_effect=failing_health_check), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Attempt update that should fail during health check
            result = await update_manager.update_server(server_name, target_version="1.0.0")
            
            # Update should succeed but activation should fail
            assert result["success"] is True  # Download succeeds
            
            # Activation should fail due to health check
            activation_result = await update_manager.activate_server(server_name)
            assert activation_result["success"] is False
            assert "health check failed" in activation_result.get("error", "").lower()
            
            # Verify rollback was triggered
            status = await update_manager.get_server_status(server_name)
            assert status["staging_version"] is None  # Staging should be cleaned up
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_monitoring_and_metrics_integration(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test monitoring and metrics collection during operations."""
        config = test_environment.config
        server_name = "files"
        
        update_manager = MCPUpdateManager(config)
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Track metrics during update
            start_time = time.time()
            
            result = await update_manager.update_server(server_name, target_version="1.2.3")
            
            end_time = time.time()
            operation_duration = end_time - start_time
            
            # Verify operation completed in reasonable time
            assert operation_duration < 30.0  # Should complete quickly in test environment
            
            # Verify result contains metrics
            assert "download_time" in result
            assert "staging_time" in result
            assert result["download_time"] > 0
            
            # Get system status for monitoring integration
            system_status = await update_manager.get_system_status()
            assert "total_servers" in system_status
            assert "healthy_servers" in system_status
            assert "pending_updates" in system_status
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_version_management_integration(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test version management and tracking integration."""
        config = test_environment.config
        server_name = "files"
        
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Get initial version state
            initial_versions = await version_manager.get_version_status(server_name)
            
            # Perform update
            await update_manager.update_server(server_name, target_version="1.2.3")
            
            # Check version tracking
            staging_versions = await version_manager.get_version_status(server_name)
            assert staging_versions["staging_version"] == "1.2.3"
            assert staging_versions["current_version"] != "1.2.3"  # Not yet activated
            
            # Activate and verify version tracking
            await update_manager.activate_server(server_name)
            
            final_versions = await version_manager.get_version_status(server_name)
            assert final_versions["current_version"] == "1.2.3"
            assert final_versions["staging_version"] is None
            
            # Test version history
            version_history = await version_manager.get_version_history(server_name)
            assert len(version_history) > 0
            assert version_history[0]["version"] == "1.2.3"
            assert "timestamp" in version_history[0]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_backup_and_recovery_integration(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test backup creation and recovery integration."""
        config = test_environment.config
        server_name = "files"
        
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Simulate existing installation
            current_version = "1.0.0"
            await version_manager.record_activation(server_name, current_version)
            
            # Update to new version
            await update_manager.update_server(server_name, target_version="1.2.3")
            await update_manager.activate_server(server_name)
            
            # Verify backup was created
            backup_path = config.get_backup_path(server_name)
            assert backup_path.exists()
            
            # Test recovery from backup
            recovery_result = await version_manager.rollback_server(server_name)
            assert recovery_result["success"] is True
            assert recovery_result["rolled_back_to"] == current_version
            
            # Verify system state after rollback
            status = await update_manager.get_server_status(server_name)
            assert status["active_version"] == current_version
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_system_load_integration(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test system behavior under load conditions."""
        config = test_environment.config
        
        # Reduce timeouts for faster testing
        config.performance.download_timeout_seconds = 10
        config.performance.health_check_timeout_seconds = 5
        
        update_manager = MCPUpdateManager(config)
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Create multiple concurrent operations
            operations = []
            
            # Start multiple updates
            for i in range(5):
                server_name = f"test-server-{i}"
                # Add to config for testing
                config.mcp_servers[server_name] = {
                    "package": f"@test/server-{i}",
                    "wrapper": f"test-server-{i}.sh"
                }
                
                operation = asyncio.create_task(
                    update_manager.update_server(server_name, target_version="1.0.0")
                )
                operations.append(operation)
            
            # Wait for all operations with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*operations, return_exceptions=True),
                    timeout=60.0
                )
                
                # Verify all operations completed
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        pytest.fail(f"Operation {i} failed: {result}")
                    assert result["success"] is True
                    
            except asyncio.TimeoutError:
                pytest.fail("Load test operations timed out")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_security_integration_workflow(
        self,
        test_environment: TestEnvironment,
        security_scanner,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test security validation integration in update workflow."""
        config = test_environment.config
        server_name = "files"
        
        update_manager = MCPUpdateManager(config)
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner), \
             patch.object(update_manager.download_manager, '_scan_security', side_effect=security_scanner.scan_package):
            
            # Enable security scanning
            config.security.verify_checksums = True
            
            # Perform update with security validation
            result = await update_manager.update_server(server_name, target_version="1.2.3")
            
            assert result["success"] is True
            assert "security_scan" in result
            assert result["security_scan"]["risk_level"] == "low"
            
            # Verify staging path contains scanned package
            staging_path = config.get_staging_path(server_name)
            assert staging_path.exists()


class TestMCPSystemIntegration:
    """System-level integration tests for complete MCP automation system."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_system_lifecycle(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner,
        performance_monitor
    ):
        """Test complete system lifecycle from initialization to shutdown."""
        performance_monitor.start_operation("full_system_lifecycle")
        
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Phase 1: System initialization
            system_status = await update_manager.get_system_status()
            assert system_status["status"] == "healthy"
            assert system_status["total_servers"] > 0
            
            # Phase 2: Bulk server updates
            all_servers = config.get_all_servers()[:3]  # Test first 3 servers
            
            for server_name in all_servers:
                await update_manager.update_server(server_name, target_version="1.0.0")
                await update_manager.activate_server(server_name)
            
            # Phase 3: System validation
            final_status = await update_manager.get_system_status()
            assert final_status["healthy_servers"] >= len(all_servers)
            
            # Phase 4: System metrics
            assert "uptime" in final_status
            assert "total_operations" in final_status
        
        performance_monitor.end_operation("full_system_lifecycle")
        performance_monitor.assert_performance("full_system_lifecycle", 120.0)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_disaster_recovery_integration(
        self,
        test_environment: TestEnvironment,
        rollback_simulator,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test disaster recovery and system resilience."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        server_name = "files"
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Establish baseline
            await update_manager.update_server(server_name, target_version="1.0.0")
            await update_manager.activate_server(server_name)
            
            baseline_status = await update_manager.get_server_status(server_name)
            assert baseline_status["active_version"] == "1.0.0"
            
            # Simulate disaster scenarios
            for scenario in rollback_simulator.get_scenarios()[:2]:  # Test first 2 scenarios
                # Trigger scenario
                rollback_simulator.trigger_scenario(scenario, server_name)
                
                # Attempt update (should fail)
                try:
                    await update_manager.update_server(server_name, target_version="2.0.0")
                    await update_manager.activate_server(server_name)
                except Exception:
                    pass  # Expected failure
                
                # Verify automatic recovery
                recovery_result = await version_manager.rollback_server(server_name)
                assert recovery_result["success"] is True
                
                # Verify system returned to baseline
                recovered_status = await update_manager.get_server_status(server_name)
                assert recovered_status["active_version"] == "1.0.0"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_component_data_flow(
        self,
        test_environment: TestEnvironment,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker,
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner
    ):
        """Test data flow between system components."""
        config = test_environment.config
        server_name = "files"
        
        # Initialize all managers
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        download_manager = MCPDownloadManager(config)
        
        with patch.object(update_manager, '_run_health_check', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_health_checker), \
             patch.object(download_manager, '_run_command', side_effect=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_process_runner):
            
            # Test data flow: Update Manager -> Version Manager
            await update_manager.update_server(server_name, target_version="1.2.3")
            
            version_status = await version_manager.get_version_status(server_name)
            assert version_status["staging_version"] == "1.2.3"
            
            # Test data flow: Version Manager -> Download Manager
            download_result = await download_manager.download_package(
                config.mcp_servers[server_name]["package"],
                "1.2.3",
                config.get_staging_path(server_name)
            )
            assert download_result["version"] == "1.2.3"
            
            # Test data flow: All components -> System status
            system_status = await update_manager.get_system_status()
            assert system_status["pending_updates"] >= 1  # At least one staged update
            
            # Complete activation and verify data consistency
            await update_manager.activate_server(server_name)
            
            final_version_status = await version_manager.get_version_status(server_name)
            final_system_status = await update_manager.get_system_status()
            
            assert final_version_status["current_version"] == "1.2.3"
            assert final_version_status["staging_version"] is None
            assert final_system_status["pending_updates"] == 0