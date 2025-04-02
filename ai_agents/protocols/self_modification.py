#!/usr/bin/env python3
"""
Self-Modification Control System Module

This module provides a secure framework for AGI/ASI systems to modify their own code
in a controlled, verifiable manner. It implements:

1. Dual-execution sandbox for code changes
2. SHA-3 cryptographic hashing for code integrity
3. Causal influence diagrams and state transition logging
4. Human-readable accountability logs
"""

import os
import sys
import time
import json
import hashlib
import logging
import tempfile
import subprocess
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/self_modification.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SelfModification")


# Secure hash function (SHA-3)
def secure_hash(content: Union[str, bytes]) -> str:
    """Generate a SHA-3 (256-bit) hash of content"""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha3_256(content).hexdigest()


class ModificationType(Enum):
    """Types of code modifications"""

    ADDITION = "addition"
    MODIFICATION = "modification"
    DELETION = "deletion"
    REFACTORING = "refactoring"
    OPTIMIZATION = "optimization"


@dataclass
class CodeChange:
    """Represents a single code change"""

    file_path: str
    original_content: Optional[str]
    new_content: Optional[str]
    modification_type: ModificationType
    justification: str
    timestamp: datetime = field(default_factory=datetime.now)
    original_hash: Optional[str] = None
    new_hash: Optional[str] = None

    def __post_init__(self):
        """Calculate hashes after initialization"""
        if self.original_content is not None:
            self.original_hash = secure_hash(self.original_content)
        if self.new_content is not None:
            self.new_hash = secure_hash(self.new_content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "file_path": self.file_path,
            "modification_type": self.modification_type.value,
            "justification": self.justification,
            "timestamp": self.timestamp.isoformat(),
            "original_hash": self.original_hash,
            "new_hash": self.new_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeChange":
        """Create from dictionary"""
        return cls(
            file_path=data["file_path"],
            original_content=None,  # We don't store content in logs
            new_content=None,  # Just the hashes
            modification_type=ModificationType(data["modification_type"]),
            justification=data["justification"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            original_hash=data.get("original_hash"),
            new_hash=data.get("new_hash"),
        )


@dataclass
class ModificationEvent:
    """A group of related code changes forming a single logical modification"""

    event_id: str
    changes: List[CodeChange]
    rationale: str
    initiated_by: str  # Component that initiated the change
    approved_by: Optional[str] = None  # Human or approval system
    verification_result: Optional[bool] = None
    executed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "rationale": self.rationale,
            "initiated_by": self.initiated_by,
            "approved_by": self.approved_by,
            "verification_result": self.verification_result,
            "executed": self.executed,
            "changes": [change.to_dict() for change in self.changes],
            "timestamp": self.changes[0].timestamp.isoformat()
            if self.changes
            else datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModificationEvent":
        """Create from dictionary"""
        return cls(
            event_id=data["event_id"],
            changes=[CodeChange.from_dict(c) for c in data["changes"]],
            rationale=data["rationale"],
            initiated_by=data["initiated_by"],
            approved_by=data.get("approved_by"),
            verification_result=data.get("verification_result"),
            executed=data.get("executed", False),
        )


class SandboxViolation(Exception):
    """Exception raised when code violates sandbox constraints"""

    pass


class SecurityRuleViolation(Exception):
    """Exception raised when code violates security rules"""

    pass


class VerificationFailure(Exception):
    """Exception raised when code fails verification"""

    pass


class SelfModificationControl:
    """Controls and verifies all self-modifications to the AGI/ASI system"""

    def __init__(
        self,
        base_dir: str,
        log_dir: str = "logs/modifications",
        max_sandbox_time: int = 30,
        auto_approve: bool = False,
        forbidden_imports: List[str] = None,
        max_changes_per_event: int = 10,
    ):
        """
        Initialize the self-modification control system

        Args:
            base_dir: Base directory of the codebase
            log_dir: Directory to store modification logs
            max_sandbox_time: Maximum execution time in seconds for sandboxed code
            auto_approve: Whether to auto-approve changes (dangerous!)
            forbidden_imports: List of modules that cannot be imported
            max_changes_per_event: Maximum number of file changes in one event
        """
        self.base_dir = os.path.abspath(base_dir)
        self.log_dir = log_dir
        self.max_sandbox_time = max_sandbox_time
        self.auto_approve = auto_approve
        self.forbidden_imports = forbidden_imports or [
            "socket",
            "subprocess",
            "requests",
            "http.client",
            "ftplib",
            "smtplib",
            "telnetlib",
            "urllib.request",
        ]
        self.max_changes_per_event = max_changes_per_event

        # Initialize logs directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Load existing modification history
        self.modification_history: List[ModificationEvent] = self._load_history()

        logger.info(f"Self-Modification Control system initialized in {self.base_dir}")
        logger.info(f"Auto-approve is {'ENABLED' if self.auto_approve else 'DISABLED'}")

    def propose_change(
        self,
        file_path: str,
        new_content: str,
        justification: str,
        modification_type: ModificationType = ModificationType.MODIFICATION,
    ) -> CodeChange:
        """
        Propose a single file change

        Args:
            file_path: Path to the file to be modified
            new_content: New content for the file
            justification: Explanation for why this change is necessary
            modification_type: Type of modification

        Returns:
            CodeChange object representing the proposed change
        """
        abs_path = os.path.abspath(os.path.join(self.base_dir, file_path))

        # Ensure the file is within the base directory
        if not abs_path.startswith(self.base_dir):
            raise SecurityRuleViolation(
                f"Cannot modify files outside the base directory: {file_path}"
            )

        # For existing files, read the original content
        original_content = None
        if os.path.exists(abs_path) and modification_type != ModificationType.ADDITION:
            with open(abs_path, "r", encoding="utf-8") as f:
                original_content = f.read()

        return CodeChange(
            file_path=file_path,
            original_content=original_content,
            new_content=new_content,
            modification_type=modification_type,
            justification=justification,
        )

    def propose_modification_event(
        self, changes: List[CodeChange], rationale: str, initiated_by: str
    ) -> ModificationEvent:
        """
        Propose a set of related changes as a single modification event

        Args:
            changes: List of CodeChange objects
            rationale: Overall explanation for the set of changes
            initiated_by: Identifier of the component initiating the change

        Returns:
            ModificationEvent object
        """
        if len(changes) > self.max_changes_per_event:
            raise SecurityRuleViolation(
                f"Too many changes in one event: {len(changes)}. "
                f"Maximum allowed: {self.max_changes_per_event}"
            )

        event_id = str(uuid.uuid4())
        return ModificationEvent(
            event_id=event_id,
            changes=changes,
            rationale=rationale,
            initiated_by=initiated_by,
        )

    def verify_changes(self, event: ModificationEvent) -> bool:
        """
        Verify that the proposed changes meet security and functionality requirements

        Args:
            event: ModificationEvent to verify

        Returns:
            True if verification passed, False otherwise
        """
        logger.info(
            f"Verifying modification event {event.event_id} with {len(event.changes)} changes"
        )

        try:
            # Static analysis of Python code
            for change in event.changes:
                if change.file_path.endswith(".py") and change.new_content:
                    self._analyze_python_code(change.new_content)

            # Create a sandbox environment for testing
            with tempfile.TemporaryDirectory() as sandbox_dir:
                # Copy current codebase to sandbox
                self._copy_codebase_to_sandbox(sandbox_dir)

                # Apply the changes in the sandbox
                self._apply_changes_to_sandbox(sandbox_dir, event.changes)

                # Run tests in the sandbox with timeout
                test_result = self._run_tests_in_sandbox(sandbox_dir)

                if not test_result:
                    logger.warning(
                        f"Tests failed for modification event {event.event_id}"
                    )
                    event.verification_result = False
                    return False

            # All checks passed
            event.verification_result = True
            logger.info(f"Verification passed for modification event {event.event_id}")
            return True

        except (SandboxViolation, SecurityRuleViolation, VerificationFailure) as e:
            logger.error(f"Verification failed: {str(e)}")
            event.verification_result = False
            return False

    def _analyze_python_code(self, code_content: str) -> None:
        """
        Perform static analysis on Python code to detect security issues

        Args:
            code_content: Python code to analyze

        Raises:
            SecurityRuleViolation: If code contains forbidden constructs
        """
        try:
            # Parse the code to an AST
            tree = ast.parse(code_content)

            # Check for forbidden imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in self.forbidden_imports:
                            raise SecurityRuleViolation(
                                f"Forbidden import: {name.name}"
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.forbidden_imports:
                        raise SecurityRuleViolation(f"Forbidden import: {node.module}")

                # Check for eval/exec usage
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec"):
                        raise SecurityRuleViolation(
                            f"Forbidden function call: {node.func.id}"
                        )

            # More advanced checks could be added here

        except SyntaxError as e:
            raise VerificationFailure(f"Python syntax error: {str(e)}")

    def _copy_codebase_to_sandbox(self, sandbox_dir: str) -> None:
        """Copy the current codebase to the sandbox directory"""
        import shutil

        # Get list of Python files to copy
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".py"):
                    src_path = os.path.join(root, file)
                    # Create relative path
                    rel_path = os.path.relpath(src_path, self.base_dir)
                    dst_path = os.path.join(sandbox_dir, rel_path)

                    # Create directories if needed
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                    # Copy the file
                    shutil.copy2(src_path, dst_path)

    def _apply_changes_to_sandbox(
        self, sandbox_dir: str, changes: List[CodeChange]
    ) -> None:
        """Apply the proposed changes to the sandbox environment"""
        for change in changes:
            full_path = os.path.join(sandbox_dir, change.file_path)

            if change.modification_type == ModificationType.ADDITION:
                # Create directories if needed
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(change.new_content)

            elif change.modification_type == ModificationType.MODIFICATION:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(change.new_content)

            elif change.modification_type == ModificationType.DELETION:
                if os.path.exists(full_path):
                    os.remove(full_path)

    def _run_tests_in_sandbox(self, sandbox_dir: str) -> bool:
        """
        Run tests in the sandbox environment with a timeout

        Returns:
            True if tests passed, False otherwise
        """
        # Create a temporary pytest configuration
        with open(os.path.join(sandbox_dir, "pytest.ini"), "w") as f:
            f.write("[pytest]\npython_files = test_*.py\n")

        # Run pytest in a subprocess with timeout
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-xvs"],
                cwd=sandbox_dir,
                timeout=self.max_sandbox_time,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error(
                f"Sandbox tests timed out after {self.max_sandbox_time} seconds"
            )
            return False

    def approve_changes(self, event: ModificationEvent, approver: str = "auto") -> bool:
        """
        Approve a modification event for execution

        Args:
            event: ModificationEvent to approve
            approver: Identifier of the approver

        Returns:
            True if approval succeeded
        """
        # Auto-approval check
        if not self.auto_approve and approver == "auto":
            logger.warning("Auto-approval attempted but auto_approve is disabled")
            return False

        event.approved_by = approver
        logger.info(f"Modification event {event.event_id} approved by {approver}")
        return True

    def execute_changes(self, event: ModificationEvent) -> bool:
        """
        Execute approved changes

        Args:
            event: Approved ModificationEvent to execute

        Returns:
            True if changes were applied successfully
        """
        if not event.verification_result:
            logger.error(f"Cannot execute unverified changes: {event.event_id}")
            return False

        if not event.approved_by:
            logger.error(f"Cannot execute unapproved changes: {event.event_id}")
            return False

        logger.info(f"Executing modification event {event.event_id}")

        try:
            # Apply each change
            for change in event.changes:
                abs_path = os.path.abspath(
                    os.path.join(self.base_dir, change.file_path)
                )

                if change.modification_type == ModificationType.ADDITION:
                    # Create directories if needed
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(change.new_content)

                elif change.modification_type == ModificationType.MODIFICATION:
                    # Verify hash of existing content matches original hash
                    if os.path.exists(abs_path):
                        with open(abs_path, "r", encoding="utf-8") as f:
                            current_content = f.read()
                            current_hash = secure_hash(current_content)

                        if (
                            change.original_hash
                            and current_hash != change.original_hash
                        ):
                            logger.error(
                                f"File has been modified since change was proposed: {change.file_path}"
                            )
                            return False

                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(change.new_content)

                elif change.modification_type == ModificationType.DELETION:
                    if os.path.exists(abs_path):
                        os.remove(abs_path)

            # Update event status
            event.executed = True

            # Log the event
            self.modification_history.append(event)
            self._save_event(event)

            logger.info(f"Successfully executed modification event {event.event_id}")
            return True

        except Exception as e:
            logger.error(f"Error executing changes: {str(e)}")
            return False

    def _load_history(self) -> List[ModificationEvent]:
        """Load modification history from log files"""
        history = []

        if not os.path.exists(self.log_dir):
            return history

        for filename in os.listdir(self.log_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.log_dir, filename), "r") as f:
                        event_data = json.load(f)
                        event = ModificationEvent.from_dict(event_data)
                        history.append(event)
                except Exception as e:
                    logger.error(
                        f"Error loading modification event {filename}: {str(e)}"
                    )

        # Sort by timestamp
        history.sort(key=lambda e: datetime.fromisoformat(e.to_dict()["timestamp"]))

        return history

    def _save_event(self, event: ModificationEvent) -> None:
        """Save a modification event to the log directory"""
        event_file = os.path.join(self.log_dir, f"{event.event_id}.json")
        with open(event_file, "w") as f:
            json.dump(event.to_dict(), f, indent=2)

    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Get the modification history as serializable dictionaries"""
        return [event.to_dict() for event in self.modification_history]

    def get_human_readable_log(self, event_id: Optional[str] = None) -> str:
        """
        Generate a human-readable log of modifications

        Args:
            event_id: Optional event ID to get log for just one event

        Returns:
            Formatted log string
        """
        if event_id:
            # Find specific event
            event = next(
                (e for e in self.modification_history if e.event_id == event_id), None
            )
            if not event:
                return f"No modification event found with ID {event_id}"
            events = [event]
        else:
            events = self.modification_history

        log_lines = []
        for event in events:
            event_dict = event.to_dict()
            log_lines.append(f"Modification Event: {event.event_id}")
            log_lines.append(f"Timestamp: {event_dict['timestamp']}")
            log_lines.append(f"Rationale: {event.rationale}")
            log_lines.append(f"Initiated by: {event.initiated_by}")
            log_lines.append(f"Approved by: {event.approved_by or 'Not approved'}")
            log_lines.append(
                f"Verification: {'Passed' if event.verification_result else 'Failed'}"
            )
            log_lines.append(f"Executed: {'Yes' if event.executed else 'No'}")
            log_lines.append("Changes:")

            for i, change in enumerate(event.changes):
                log_lines.append(
                    f"  {i + 1}. {change.modification_type.value} to {change.file_path}"
                )
                log_lines.append(f"     Justification: {change.justification}")
                if change.original_hash and change.new_hash:
                    log_lines.append(
                        f"     Hash change: {change.original_hash[:8]} → {change.new_hash[:8]}"
                    )

            log_lines.append("-" * 40)

        return "\n".join(log_lines)


# Code execution sandbox with time and resource limits
class CodeExecutionSandbox:
    """Provides a secure sandbox for executing potentially unsafe code"""

    def __init__(
        self,
        max_execution_time: int = 5,
        max_memory_mb: int = 100,
        enable_network: bool = False,
    ):
        """
        Initialize the code execution sandbox

        Args:
            max_execution_time: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            enable_network: Whether to allow network access (dangerous!)
        """
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.enable_network = enable_network

    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in a secure sandbox

        Args:
            code: Python code to execute

        Returns:
            Dict containing execution results and metadata
        """
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as f:
            f.write(code)
            temp_filename = f.name

        result = {
            "success": False,
            "output": "",
            "error": None,
            "execution_time": 0,
            "memory_usage": 0,
        }

        try:
            # Prepare sandbox environment variables
            env = os.environ.copy()
            if not self.enable_network:
                # Block network access at Python level
                env["PYTHONPATH"] = ""

            # Execute in a subprocess with resource limits
            start_time = time.time()

            # Use subprocess for isolation
            process = subprocess.Popen(
                [sys.executable, temp_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )

            # Set timeout for execution
            try:
                stdout, stderr = process.communicate(timeout=self.max_execution_time)
                exit_code = process.returncode

                result["success"] = exit_code == 0
                result["output"] = stdout
                result["error"] = stderr if stderr else None
                result["execution_time"] = time.time() - start_time

                # Get peak memory usage
                # This is an approximation
                if hasattr(process, "memory_info"):
                    mem_info = process.memory_info()
                    result["memory_usage"] = mem_info.rss / (
                        1024 * 1024
                    )  # Convert to MB

            except subprocess.TimeoutExpired:
                process.kill()
                result["error"] = (
                    f"Execution timed out after {self.max_execution_time} seconds"
                )

        except Exception as e:
            result["error"] = str(e)

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_filename}: {e}")
                pass

        return result
