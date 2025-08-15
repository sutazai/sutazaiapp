#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
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
    logger.info("=== Basic MCP Server Update Example ===")
    
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
            
            logger.info(f"Scheduled update job: {job_id}")
            
            # Monitor progress
            while True:
                status = await manager.get_update_status(job_id)
                if status:
                    logger.info(f"Status: {status.status.value} - Progress: {status.progress_percentage:.1f}%")
                    
                    if status.status.value in ['completed', 'failed', 'cancelled']:
                        break
                
                await asyncio.sleep(2)
            
            logger.info(f"Update completed with status: {status.status.value}")
            if status.error_message:
                logger.error(f"Error: {status.error_message}")
    
    except Exception as e:
        logger.error(f"Error during update: {e}")


async def example_bulk_updates():
    """Example: Bulk update multiple servers."""
    logger.info("\\n=== Bulk MCP Server Updates Example ===")
    
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
                logger.info(f"Scheduled update for {server_name}: {job_id}")
            
            # Get summary
            summary = await manager.get_update_summary()
            logger.info(f"\\nUpdate Summary:")
            logger.info(f"  Total servers: {summary.total_servers}")
            logger.info(f"  Pending updates: {summary.pending_updates}")
            logger.info(f"  Successful updates: {summary.successful_updates}")
            logger.error(f"  Failed updates: {summary.failed_updates}")
    
    except Exception as e:
        logger.error(f"Error during bulk updates: {e}")


async def example_version_management():
    """Example: Version management operations."""
    logger.info("\\n=== Version Management Example ===")
    
    try:
        config = get_config()
        version_manager = VersionManager(config)
        
        # Check current versions
        servers = ["files", "postgres", "playwright-mcp"]
        
        for server_name in servers:
            current = await version_manager.get_current_version(server_name)
            available = await version_manager.get_available_version(server_name)
            
            logger.info(f"{server_name}:")
            logger.info(f"  Current: {current or 'Not installed'}")
            logger.info(f"  Available: {available or 'Unknown'}")
            
            # Check version history
            history = version_manager.get_version_history(server_name)
            logger.info(f"  History: {len(history)} versions tracked")
        
        # Show recent operations
        operations = version_manager.get_operation_history(limit=5)
        logger.info(f"\\nRecent operations ({len(operations)}):")
        for op in operations:
            logger.info(f"  {op.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - "
                  f"{op.operation_type.value} {op.server_name} -> {op.to_version} "
                  f"({'✓' if op.success else '✗'})")
    
    except Exception as e:
        logger.error(f"Error during version management: {e}")


async def example_download_with_progress():
    """Example: Download with progress monitoring."""
    logger.info("\\n=== Download with Progress Example ===")
    
    try:
        config = get_config()
        
        def progress_callback(progress):
            logger.info(f"\\r  Progress: {progress.percentage:.1f}% "
                  f"({progress.speed_mbps:.2f} MB/s, "
                  f"ETA: {progress.eta_seconds:.0f}s)", end="")
        
        async with DownloadManager(config) as dm:
            logger.info("Downloading @modelcontextprotocol/server-filesystem...")
            
            result = await dm.download_package(
                server_name="files",
                package_name="@modelcontextprotocol/server-filesystem",
                progress_callback=progress_callback
            )
            
            logger.info("\\n")  # New line after progress
            
            if result.success:
                logger.info(f"✓ Download successful:")
                logger.info(f"  File: {result.file_path}")
                logger.info(f"  Size: {result.size_bytes:,} bytes")
                logger.info(f"  Time: {result.download_time_seconds:.2f}s")
                logger.info(f"  Checksum: {result.checksum}")
            else:
                logger.error(f"✗ Download failed: {result.error_message}")
    
    except Exception as e:
        logger.error(f"Error during download: {e}")


async def example_configuration_management():
    """Example: Configuration management."""
    logger.info("\\n=== Configuration Management Example ===")
    
    try:
        # Load configuration
        config = get_config()
        
        logger.info("Current configuration:")
        logger.info(f"  Update mode: {config.update_mode.value}")
        logger.info(f"  Log level: {config.log_level.value}")
        logger.info(f"  Dry run: {config.dry_run}")
        logger.info(f"  Max concurrent downloads: {config.performance.max_concurrent_downloads}")
        logger.info(f"  Download timeout: {config.performance.download_timeout_seconds}s")
        logger.info(f"  Verify checksums: {config.security.verify_checksums}")
        logger.info(f"  Max download size: {config.security.max_download_size_mb}MB")
        
        # Show paths
        logger.info(f"\\nPaths:")
        logger.info(f"  MCP root: {config.paths.mcp_root}")
        logger.info(f"  Staging: {config.paths.staging_root}")
        logger.info(f"  Backups: {config.paths.backup_root}")
        logger.info(f"  Logs: {config.paths.logs_root}")
        
        # Show configured servers
        logger.info(f"\\nConfigured servers ({len(config.mcp_servers)}):")
        for server_name, server_config in config.mcp_servers.items():
            logger.info(f"  {server_name}: {server_config['package']}")
        
        # Save configuration example
        config_file = config.paths.automation_root / "example_config.json"
        config.save_config(config_file)
        logger.info(f"\\nConfiguration saved to: {config_file}")
    
    except Exception as e:
        logger.error(f"Error with configuration: {e}")


async def example_error_handling():
    """Example: Error handling and recovery."""
    logger.error("\\n=== Error Handling Example ===")
    
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
            logger.error(f"Recorded error: {error_id}")
        
        # Get error statistics
        stats = error_tracker.get_error_statistics()
        logger.error(f"\\nError Statistics:")
        logger.error(f"  Total errors: {stats.get('total_errors', 0)}")
        logger.error(f"  Recent errors (24h): {stats.get('recent_errors_24h', 0)}")
        logger.info(f"  Resolution rate: {stats.get('resolution_rate_percent', 0)}%")
        
        # Show error patterns
        patterns = stats.get('error_patterns', {})
        if patterns:
            logger.error(f"\\nTop error patterns:")
            for pattern, count in list(patterns.items())[:3]:
                logger.info(f"  {pattern}: {count} occurrences")
    
    except Exception as e:
        logger.error(f"Error during error handling example: {e}")


async def main():
    """Run all examples."""
    logger.info("MCP Automation System Usage Examples")
    logger.info("=" * 50)
    
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
            logger.error(f"\\nError in {example_func.__name__}: {e}")
        
        # Small delay between examples
        await asyncio.sleep(1)
    
    logger.info("\\n" + "=" * 50)
    logger.info("Examples completed!")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())