#!/usr/bin/env python3
"""
MCP Cleanup System

Intelligent cleanup system for MCP versions and artifacts with comprehensive
safety validation, audit logging, and zero-impact operations following
Enforcement Rules compliance.

Author: Claude AI Assistant (garbage-collector.md)
Created: 2025-08-15 20:30:00 UTC
Version: 1.0.0
"""

from .cleanup_manager import CleanupManager, CleanupStatus, CleanupPriority
from .version_cleanup import VersionCleanupService, VersionCleanupResult
from .artifact_cleanup import ArtifactCleanupService, ArtifactType
from .retention_policies import RetentionPolicyManager, RetentionPolicy
from .safety_validator import SafetyValidator, SafetyResult
from .audit_logger import AuditLogger, AuditEvent
from .cleanup_scheduler import CleanupScheduler, ScheduleType

__version__ = "1.0.0"
__author__ = "Claude AI Assistant (garbage-collector.md)"

__all__ = [
    "CleanupManager",
    "CleanupStatus", 
    "CleanupPriority",
    "VersionCleanupService",
    "VersionCleanupResult",
    "ArtifactCleanupService",
    "ArtifactType",
    "RetentionPolicyManager",
    "RetentionPolicy",
    "SafetyValidator",
    "SafetyResult",
    "AuditLogger",
    "AuditEvent",
    "CleanupScheduler",
    "ScheduleType"
]