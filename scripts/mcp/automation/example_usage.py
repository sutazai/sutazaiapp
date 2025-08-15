#!/usr/bin/env python3
"""
MCP Automation System Usage Examples

Demonstrates how to use the MCP automation system for common tasks.
Shows best practices for configuration, error handling, and monitoring.

Author: Claude AI Assistant (python-architect.md)
Created: 2025-08-15 11:55:00 UTC
Version: 1.0.0
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# Import MCP automation components
from mcp.automation import (
    MCPUpdateManager,
    VersionManager,
    DownloadManager,
    get_config,
    UpdatePriority,
    UpdateMode,
    LogLevel
)


async def example_basic_update():
    """Example: Basic MCP server update."""
    print("=== Basic MCP Server Update Example ===")
    
    try:
        # Initialize configuration
        config = get_config()
        
        # Use the update manager
        async with MCPUpdateManager(config) as manager:
            # Schedule an update for the files server
            job_id = await manager.schedule_update(
                server_name="files",
                priority=UpdatePriority.HIGH
            )
            
            print(f"Scheduled update job: {job_id}")
            
            # Monitor progress
            while True:
                status = await manager.get_update_status(job_id)
                if status:
                    print(f"Status: {status.status.value} - Progress: {status.progress_percentage:.1f}%")
                    
                    if status.status.value in ['completed', 'failed', 'cancelled']:
                        break
                
                await asyncio.sleep(2)
            
            print(f"Update completed with status: {status.status.value}")
            if status.error_message:
                print(f"Error: {status.error_message}")
    
    except Exception as e:
        print(f"Error during update: {e}")


async def example_bulk_updates():
    """Example: Bulk update multiple servers."""
    print("\\n=== Bulk MCP Server Updates Example ===")
    
    try:
        config = get_config()
        
        async with MCPUpdateManager(config) as manager:
            # Schedule updates for multiple servers
            servers_to_update = ["files", "postgres", "playwright-mcp"]
            job_ids = []
            
            for server_name in servers_to_update:
                job_id = await manager.schedule_update(
                    server_name=server_name,
                    priority=UpdatePriority.NORMAL
                )
                job_ids.append(job_id)
                print(f"Scheduled update for {server_name}: {job_id}")
            
            # Get summary
            summary = await manager.get_update_summary()
            print(f"\\nUpdate Summary:")
            print(f"  Total servers: {summary.total_servers}")
            print(f"  Pending updates: {summary.pending_updates}")
            print(f"  Successful updates: {summary.successful_updates}")
            print(f"  Failed updates: {summary.failed_updates}")
    
    except Exception as e:
        print(f"Error during bulk updates: {e}")


async def example_version_management():
    """Example: Version management operations."""
    print("\\n=== Version Management Example ===")
    
    try:
        config = get_config()
        version_manager = VersionManager(config)
        
        # Check current versions
        servers = ["files", "postgres", "playwright-mcp"]
        
        for server_name in servers:
            current = await version_manager.get_current_version(server_name)
            available = await version_manager.get_available_version(server_name)
            
            print(f"{server_name}:")
            print(f"  Current: {current or 'Not installed'}")
            print(f"  Available: {available or 'Unknown'}")
            
            # Check version history
            history = version_manager.get_version_history(server_name)
            print(f"  History: {len(history)} versions tracked")
        
        # Show recent operations
        operations = version_manager.get_operation_history(limit=5)
        print(f"\\nRecent operations ({len(operations)}):")
        for op in operations:
            print(f"  {op.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
                  f"{op.operation_type.value} {op.server_name} -> {op.to_version} "
                  f"({'✓' if op.success else '✗'})")
    
    except Exception as e:
        print(f"Error during version management: {e}")


async def example_download_with_progress():
    """Example: Download with progress monitoring."""
    print("\\n=== Download with Progress Example ===")
    
    try:
        config = get_config()
        
        def progress_callback(progress):
            print(f"\\r  Progress: {progress.percentage:.1f}% "
                  f"({progress.speed_mbps:.2f} MB/s, "
                  f"ETA: {progress.eta_seconds:.0f}s)", end="")
        
        async with DownloadManager(config) as dm:
            print("Downloading @modelcontextprotocol/server-filesystem...")
            
            result = await dm.download_package(
                server_name="files",
                package_name="@modelcontextprotocol/server-filesystem",
                progress_callback=progress_callback
            )
            
            print("\\n")  # New line after progress
            
            if result.success:
                print(f"✓ Download successful:")
                print(f"  File: {result.file_path}")
                print(f"  Size: {result.size_bytes:,} bytes")
                print(f"  Time: {result.download_time_seconds:.2f}s")
                print(f"  Checksum: {result.checksum}")
            else:
                print(f"✗ Download failed: {result.error_message}")
    
    except Exception as e:
        print(f"Error during download: {e}")


async def example_configuration_management():
    """Example: Configuration management."""
    print("\\n=== Configuration Management Example ===")
    
    try:
        # Load configuration
        config = get_config()
        
        print("Current configuration:")
        print(f"  Update mode: {config.update_mode.value}")
        print(f"  Log level: {config.log_level.value}")
        print(f"  Dry run: {config.dry_run}")
        print(f"  Max concurrent downloads: {config.performance.max_concurrent_downloads}")
        print(f"  Download timeout: {config.performance.download_timeout_seconds}s")
        print(f"  Verify checksums: {config.security.verify_checksums}")
        print(f"  Max download size: {config.security.max_download_size_mb}MB")
        
        # Show paths
        print(f"\\nPaths:")
        print(f"  MCP root: {config.paths.mcp_root}")
        print(f"  Staging: {config.paths.staging_root}")
        print(f"  Backups: {config.paths.backup_root}")
        print(f"  Logs: {config.paths.logs_root}")
        
        # Show configured servers
        print(f"\\nConfigured servers ({len(config.mcp_servers)}):")
        for server_name, server_config in config.mcp_servers.items():
            print(f"  {server_name}: {server_config['package']}")
        
        # Save configuration example
        config_file = config.paths.automation_root / "example_config.json"
        config.save_config(config_file)
        print(f"\\nConfiguration saved to: {config_file}")
    
    except Exception as e:
        print(f"Error with configuration: {e}")


async def example_error_handling():
    """Example: Error handling and recovery."""
    print("\\n=== Error Handling Example ===")
    
    try:
        from mcp.automation.error_handling import get_error_tracker, with_error_handling
        
        # Get error tracker
        error_tracker = get_error_tracker()
        
        # Simulate some errors for demonstration
        try:
            raise ValueError("Example error for demonstration")
        except Exception as e:
            error_id = error_tracker.record_error(
                error=e,
                component="example",
                function_name="example_error_handling"
            )
            print(f"Recorded error: {error_id}")
        
        # Get error statistics
        stats = error_tracker.get_error_statistics()
        print(f"\\nError Statistics:")
        print(f"  Total errors: {stats.get('total_errors', 0)}")
        print(f"  Recent errors (24h): {stats.get('recent_errors_24h', 0)}")
        print(f"  Resolution rate: {stats.get('resolution_rate_percent', 0)}%")
        
        # Show error patterns
        patterns = stats.get('error_patterns', {})
        if patterns:
            print(f"\\nTop error patterns:")
            for pattern, count in list(patterns.items())[:3]:
                print(f"  {pattern}: {count} occurrences")
    
    except Exception as e:
        print(f"Error during error handling example: {e}")


async def main():
    """Run all examples."""
    print("MCP Automation System Usage Examples")
    print("=" * 50)
    
    examples = [
        example_configuration_management,
        example_version_management,
        example_download_with_progress,
        example_error_handling,
        # Note: Commented out to avoid actual updates during demonstration
        # example_basic_update,
        # example_bulk_updates,
    ]
    
    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\\nError in {example_func.__name__}: {e}")
        
        # Small delay between examples
        await asyncio.sleep(1)
    
    print("\\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())