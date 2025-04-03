"""
Self-Modification Monitoring Module for SutazAI

This module provides monitoring and verification for the self-modification engine,
tracking code changes, dual-execution verification, and maintaining audit trails
of system modifications.
"""

import os
import time
import json
import hashlib
import difflib
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Import sys for executable path
import sys

# Try to import optional dependencies
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Import our logging setup
from utils.logging_setup import get_app_logger

logger = get_app_logger()

# Define self-modification metrics
if PROMETHEUS_AVAILABLE:
    # Modification metrics
    MODIFICATION_COUNT = Counter(
        "sutazai_self_mod_count_total",
        "Total number of self-modifications",
        ["system_id", "component", "type"],
    )

    MODIFICATION_SIZE = Histogram(
        "sutazai_self_mod_size_bytes",
        "Size of self-modifications in bytes",
        ["system_id", "component"],
    )

    # Verification metrics
    VERIFICATION_SUCCESS = Counter(
        "sutazai_self_mod_verification_total",
        "Total number of verification attempts",
        ["system_id", "component", "result"],
    )

    VERIFICATION_TIME = Histogram(
        "sutazai_self_mod_verification_seconds",
        "Time taken to verify modifications",
        ["system_id", "component"],
    )

    # Sandbox metrics
    SANDBOX_EXECUTION_TIME = Histogram(
        "sutazai_self_mod_sandbox_seconds",
        "Time taken to execute modifications in sandbox",
        ["system_id", "component"],
    )

    SANDBOX_RESOURCE_USAGE = Gauge(
        "sutazai_self_mod_sandbox_resources",
        "Resource usage during sandbox execution",
        ["system_id", "component", "resource_type"],
    )


class ModificationType(Enum):
    """Types of self-modifications."""

    CODE = "code"  # Source code changes
    CONFIG = "configuration"  # Configuration changes
    MODEL = "model"  # Neural model architecture changes
    WEIGHT = "weight"  # Weight/parameter changes
    RULE = "rule"  # Rule-based system changes
    DATA = "data"  # Data structure changes
    GOAL = "goal"  # Goal/objective changes
    OTHER = "other"  # Other changes


class VerificationResult(Enum):
    """Results of verification checks."""

    PASS = "pass"  # Verification passed
    FAIL = "fail"  # Verification failed
    INCONCLUSIVE = "inconclusive"  # Verification was inconclusive
    ERROR = "error"  # Error during verification


@dataclass
class ModificationEvent:
    """Record of a system self-modification event."""

    id: str
    timestamp: float = field(default_factory=time.time)
    system_id: str = ""
    component: str = ""
    modification_type: ModificationType = ModificationType.OTHER
    description: str = ""
    size_bytes: int = 0
    diff: Optional[str] = None
    verification_result: VerificationResult = VerificationResult.INCONCLUSIVE
    verification_details: Dict[str, Any] = field(default_factory=dict)
    sandbox_results: Dict[str, Any] = field(default_factory=dict)
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None
    is_applied: bool = False
    rollback_info: Optional[Dict[str, Any]] = None


class SelfModificationMonitor:
    """Monitor and verify system self-modifications."""

    def __init__(
        self,
        system_id: str,
        base_dir: str,
        log_dir: Optional[str] = None,
        components: Optional[List[str]] = None,
    ):
        """
        Initialize the self-modification monitor.

        Args:
            system_id: System identifier
            base_dir: Base directory to monitor for changes
            log_dir: Directory to store modification logs
            components: List of components to monitor (None for all)
        """
        self.system_id = system_id
        self.base_dir = os.path.abspath(base_dir)
        self.components = components or []
        self.logger = logger

        # Set up logging directory
        self.log_dir = log_dir or os.path.join(
            os.environ.get("SUTAZAI_LOG_DIR", "/opt/sutazaiapp/logs"), "self_mod"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up sandbox directory for testing modifications
        self.sandbox_dir = os.path.join(
            os.environ.get("SUTAZAI_SANDBOX_DIR", "/opt/sutazaiapp/sandbox"),
            f"self_mod_{system_id}",
        )
        os.makedirs(self.sandbox_dir, exist_ok=True)

        # Track modifications
        self.modifications: List[ModificationEvent] = []
        self.component_checksums: Dict[str, Dict[str, str]] = {}

        # Initialize checksums
        self._initialize_checksums()

        self.logger.info(
            f"Initialized self-modification monitor for system {system_id}"
        )

    def _initialize_checksums(self) -> None:
        """Initialize checksums for all monitored components."""
        for component in self.components:
            component_path = os.path.join(self.base_dir, component)
            if os.path.exists(component_path):
                self.component_checksums[component] = self._compute_checksums_for_path(
                    component_path
                )

    def _compute_checksums_for_path(self, path: str) -> Dict[str, str]:
        """
        Compute checksums for all files in a directory.

        Args:
            path: Directory or file path

        Returns:
            Dictionary mapping file paths to their checksums
        """
        checksums = {}
        if os.path.isfile(path):
            checksums[path] = self._compute_file_checksum(path)
        else:
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    checksums[file_path] = self._compute_file_checksum(file_path)
        return checksums

    def _compute_file_checksum(self, file_path: str) -> str:
        """
        Compute checksum for a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 checksum of the file
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error computing checksum for {file_path}: {e}")
            return ""

    def detect_changes(self, component: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Detect changes in the monitored components.

        Args:
            component: Specific component to check, or None for all

        Returns:
            Dictionary mapping change types to lists of modified files
        """
        changes: Dict[str, List[str]] = {"added": [], "modified": [], "deleted": []}

        components_to_check = [component] if component else self.components

        for comp in components_to_check:
            comp_path = os.path.join(self.base_dir, comp)
            if not os.path.exists(comp_path):
                continue

            # Get current checksums
            current_checksums = self._compute_checksums_for_path(comp_path)

            # Get previous checksums
            previous_checksums = self.component_checksums.get(comp, {})

            # Find added and modified files
            for file_path, checksum in current_checksums.items():
                if file_path not in previous_checksums:
                    changes["added"].append(file_path)
                elif previous_checksums[file_path] != checksum:
                    changes["modified"].append(file_path)

            # Find deleted files
            for file_path in previous_checksums:
                if file_path not in current_checksums:
                    changes["deleted"].append(file_path)

            # Update checksums
            self.component_checksums[comp] = current_checksums

        return changes

    def compute_diff(self, file_path: str, previous_content: str) -> str:
        """
        Compute diff between previous and current content.

        Args:
            file_path: Path to the modified file
            previous_content: Previous file content

        Returns:
            Unified diff string
        """
        try:
            with open(file_path, "r") as f:
                current_content = f.read()

            prev_lines = previous_content.splitlines(keepends=True)
            curr_lines = current_content.splitlines(keepends=True)

            diff = difflib.unified_diff(
                prev_lines,
                curr_lines,
                fromfile=f"previous/{os.path.basename(file_path)}",
                tofile=f"current/{os.path.basename(file_path)}",
                n=3,
            )

            return "".join(diff)
        except Exception as e:
            self.logger.error(f"Error computing diff for {file_path}: {e}")
            return f"Error: {e}"

    def detect_modification_type(self, file_path: str) -> ModificationType:
        """
        Detect the type of modification based on the file path.

        Args:
            file_path: Path to the modified file

        Returns:
            ModificationType enum value
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".py", ".c", ".cpp", ".h", ".js", ".ts"]:
            return ModificationType.CODE
        elif ext in [".json", ".yaml", ".yml", ".toml", ".ini", ".conf"]:
            return ModificationType.CONFIG
        elif ext in [".pt", ".pth", ".h5", ".ckpt", ".weights"]:
            return ModificationType.WEIGHT
        elif "model" in file_path.lower() and ext in [".py", ".json"]:
            return ModificationType.MODEL
        elif "rule" in file_path.lower() or "policy" in file_path.lower():
            return ModificationType.RULE
        elif "goal" in file_path.lower() or "objective" in file_path.lower():
            return ModificationType.GOAL
        elif ext in [".csv", ".db", ".sqlite", ".parquet", ".jsonl"]:
            return ModificationType.DATA
        else:
            return ModificationType.OTHER

    def prepare_sandbox(self, component: str) -> bool:
        """
        Prepare sandbox environment for testing a component modification.

        Args:
            component: Component being modified

        Returns:
            True if preparation was successful, False otherwise
        """
        try:
            # Clear previous sandbox
            sandbox_comp_dir = os.path.join(self.sandbox_dir, component)
            if os.path.exists(sandbox_comp_dir):
                shutil.rmtree(sandbox_comp_dir, ignore_errors=True)

            # Copy component to sandbox
            component_path = os.path.join(self.base_dir, component)
            shutil.copytree(component_path, sandbox_comp_dir, dirs_exist_ok=True)

            return True
        except Exception as e:
            self.logger.error(f"Error preparing sandbox for {component}: {e}")
            return False

    def verify_in_sandbox(
        self, component: str, tests: Optional[List[str]] = None
    ) -> Dict[str, Union[bool, int, float, Optional[str], Dict[str, Any]]]:
        """
        Verify a modification in the sandbox environment.

        Args:
            component: Component being verified
            tests: List of tests to run (None for default tests)

        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        sandbox_comp_dir = os.path.join(self.sandbox_dir, component)

        results: Dict[str, Union[bool, int, float, Optional[str], Dict[str, Any]]] = {
            "success": False,
            "tests_run": 0,
            "tests_passed": 0,
            "execution_time": 0,
            "error": None,
            "details": {},
        }

        try:
            # Check if component has tests
            test_dir = os.path.join(sandbox_comp_dir, "tests")
            if not os.path.exists(test_dir) and not tests:
                results["error"] = f"No tests found for component {component}"
                return results

            # Run tests
            tests_to_run = tests or ["pytest", test_dir]
            # Safer way to run tests using the current python executable
            if tests_to_run[0] == "pytest":  # Check if using default
                tests_to_run = [sys.executable, "-m", "pytest", test_dir]

            process = subprocess.run(
                tests_to_run, cwd=sandbox_comp_dir, capture_output=True, text=True
            )

            results["execution_time"] = time.time() - start_time
            results["return_code"] = process.returncode
            results["stdout"] = process.stdout
            results["stderr"] = process.stderr

            # Parse test results
            if "passed" in process.stdout:
                results["success"] = process.returncode == 0
                # Extract number of tests from pytest output
                import re

                results["tests_run"] = 0 # Initialize tests_run

                match = re.search(r"(\d+) passed", process.stdout)
                if match:
                    results["tests_passed"] = int(match.group(1))
                    results["tests_run"] = results["tests_passed"] # Assign passed tests

                match = re.search(r"(\d+) failed", process.stdout)
                if match:
                    failed_count: int = int(match.group(1))
                    # Ensure results["tests_run"] is treated as int
                    current_tests_run = results.get("tests_run", 0)
                    if isinstance(current_tests_run, (int, float)): # Check if numeric
                       results["tests_run"] = int(current_tests_run) + failed_count
                    else:
                        # Handle case where tests_run might be non-numeric (shouldn't happen with init)
                        logger.warning(f"Non-numeric value found for tests_run: {current_tests_run}. Setting failed count directly.")
                        results["tests_run"] = failed_count # Or handle as error

            if PROMETHEUS_AVAILABLE:
                # Record verification time
                VERIFICATION_TIME.labels(
                    system_id=self.system_id, component=component
                ).observe(results["execution_time"])

                # Record sandbox resource usage (could be expanded with actual measurements)
                SANDBOX_EXECUTION_TIME.labels(
                    system_id=self.system_id, component=component
                ).observe(results["execution_time"])

                # Record verification result
                VERIFICATION_SUCCESS.labels(
                    system_id=self.system_id,
                    component=component,
                    result="success" if results["success"] else "fail",
                ).inc()

            return results

        except Exception as e:
            error_msg = f"Error verifying {component} in sandbox: {e}"
            self.logger.error(error_msg)
            results["error"] = error_msg
            results["execution_time"] = time.time() - start_time

            if PROMETHEUS_AVAILABLE:
                VERIFICATION_SUCCESS.labels(
                    system_id=self.system_id, component=component, result="error"
                ).inc()

            return results

    def record_modification(
        self,
        component: str,
        modified_files: List[str],
        description: str,
        mod_type: Optional[ModificationType] = None,
        verification_result: Optional[Dict[str, Any]] = None,
    ) -> ModificationEvent:
        """
        Record a modification event.

        Args:
            component: Modified component
            modified_files: List of modified files
            description: Description of the modification
            mod_type: Type of modification (auto-detected if None)
            verification_result: Results of verification (None if not verified)

        Returns:
            Created ModificationEvent object
        """
        # Generate modification ID
        mod_id = f"{self.system_id}_{component}_{int(time.time())}_{hashlib.sha256(description.encode()).hexdigest()[:8]}"

        # Determine modification type
        if mod_type is None and modified_files:
            mod_type = self.detect_modification_type(modified_files[0])
        else:
            mod_type = mod_type or ModificationType.OTHER

        # Calculate total size
        total_size = sum(
            os.path.getsize(file) for file in modified_files if os.path.exists(file)
        )

        # Create verification result
        if verification_result:
            ver_result = (
                VerificationResult.PASS
                if verification_result.get("success", False)
                else VerificationResult.FAIL
            )
        else:
            ver_result = VerificationResult.INCONCLUSIVE

        # Create modification event
        event = ModificationEvent(
            id=mod_id,
            system_id=self.system_id,
            component=component,
            modification_type=mod_type,
            description=description,
            size_bytes=total_size,
            verification_result=ver_result,
            verification_details=verification_result or {},
            is_applied=False,
        )

        # Add to modifications list
        self.modifications.append(event)

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            MODIFICATION_COUNT.labels(
                system_id=self.system_id, component=component, type=mod_type.value
            ).inc()

            MODIFICATION_SIZE.labels(
                system_id=self.system_id, component=component
            ).observe(total_size)

        # Log the modification
        self.logger.info(
            f"Recorded modification {mod_id} to component {component}, "
            f"type={mod_type.value}, size={total_size} bytes, "
            f"verified={ver_result.value}"
        )

        # Write to log file
        self._log_modification(event)

        return event

    def approve_modification(
        self, mod_id: str, approver: str
    ) -> Optional[ModificationEvent]:
        """
        Approve a pending modification.

        Args:
            mod_id: Modification ID
            approver: Identity of the approver

        Returns:
            Updated ModificationEvent or None if not found
        """
        for i, mod in enumerate(self.modifications):
            if mod.id == mod_id:
                mod.approved_by = approver
                mod.approved_at = time.time()

                # Update the modification in the list
                self.modifications[i] = mod

                # Log the approval
                self.logger.info(f"Modification {mod_id} approved by {approver}")

                # Update log file
                self._log_modification(mod)

                return mod

        return None

    def mark_as_applied(
        self, mod_id: str, rollback_info: Optional[Dict[str, Any]] = None
    ) -> Optional[ModificationEvent]:
        """
        Mark a modification as applied to the system.

        Args:
            mod_id: Modification ID
            rollback_info: Information needed for rollback (if available)

        Returns:
            Updated ModificationEvent or None if not found
        """
        for i, mod in enumerate(self.modifications):
            if mod.id == mod_id:
                mod.is_applied = True
                mod.rollback_info = rollback_info

                # Update the modification in the list
                self.modifications[i] = mod

                # Log the application
                self.logger.info(f"Modification {mod_id} marked as applied")

                # Update log file
                self._log_modification(mod)

                return mod

        return None

    def get_modification(self, mod_id: str) -> Optional[ModificationEvent]:
        """
        Get a modification by ID.

        Args:
            mod_id: Modification ID

        Returns:
            ModificationEvent or None if not found
        """
        for mod in self.modifications:
            if mod.id == mod_id:
                return mod
        return None

    def get_modifications(
        self,
        component: Optional[str] = None,
        mod_type: Optional[ModificationType] = None,
        applied_only: bool = False,
        approved_only: bool = False,
        limit: int = 100,
    ) -> List[ModificationEvent]:
        """
        Get filtered list of modifications.

        Args:
            component: Filter by component
            mod_type: Filter by modification type
            applied_only: Only return applied modifications
            approved_only: Only return approved modifications
            limit: Maximum number of modifications to return

        Returns:
            List of ModificationEvent objects
        """
        result = []
        for mod in sorted(self.modifications, key=lambda m: m.timestamp, reverse=True):
            if component and mod.component != component:
                continue
            if mod_type and mod.modification_type != mod_type:
                continue
            if applied_only and not mod.is_applied:
                continue
            if approved_only and not mod.approved_by:
                continue

            result.append(mod)
            if len(result) >= limit:
                break

        return result

    def _log_modification(self, mod: ModificationEvent) -> None:
        """Log a modification event to the log file."""
        # Convert the modification to a serializable dict
        log_entry = {
            "id": mod.id,
            "timestamp": datetime.fromtimestamp(mod.timestamp).isoformat(),
            "system_id": mod.system_id,
            "component": mod.component,
            "modification_type": mod.modification_type.value,
            "description": mod.description,
            "size_bytes": mod.size_bytes,
            "verification_result": mod.verification_result.value,
            "verification_details": mod.verification_details,
            "sandbox_results": mod.sandbox_results,
            "approved_by": mod.approved_by,
            "approved_at": datetime.fromtimestamp(mod.approved_at).isoformat()
            if mod.approved_at
            else None,
            "is_applied": mod.is_applied,
            "rollback_info": mod.rollback_info,
        }

        # Generate filename based on month
        month_str = datetime.now().strftime("%Y-%m")
        filename = f"{month_str}_modifications.jsonl"
        filepath = os.path.join(self.log_dir, filename)

        # Append to log file
        with open(filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


# Factory function
def create_self_mod_monitor(
    system_id: str,
    base_dir: str,
    components: Optional[List[str]] = None,
    log_dir: Optional[str] = None,
) -> SelfModificationMonitor:
    """Create and return a new self-modification monitor."""
    return SelfModificationMonitor(
        system_id=system_id,
        base_dir=base_dir,
        components=components or ["models", "configs", "rules", "utils"],
        log_dir=log_dir,
    )
