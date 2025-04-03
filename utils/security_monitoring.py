"""
Security Monitoring Module for SutazAI

This module provides security monitoring for air-gapped environments,
including system integrity checking, access controls, anomaly detection,
and security event logging.
"""

import os
import time
import json
import hashlib
import hmac
import base64
import logging
import threading
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import uuid

# Try to import optional dependencies
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Import our logging setup
from utils.logging_setup import get_app_logger

logger = get_app_logger()

# Define security metrics
if PROMETHEUS_AVAILABLE:
    # Security event metrics
    SECURITY_EVENTS = Counter(
        "sutazai_security_events_total",
        "Total number of security events",
        ["system_id", "event_type", "severity"],
    )

    ACCESS_ATTEMPTS = Counter(
        "sutazai_access_attempts_total",
        "Total number of access attempts",
        ["system_id", "resource_type", "result"],
    )

    # Integrity check metrics
    INTEGRITY_CHECK_STATUS = Gauge(
        "sutazai_integrity_check_status",
        "Status of integrity check (1=pass, 0=fail)",
        ["system_id", "component"],
    )

    INTEGRITY_CHECK_DURATION = Histogram(
        "sutazai_integrity_check_seconds",
        "Time taken to perform integrity check",
        ["system_id", "component"],
    )

    # Anomaly detection metrics
    ANOMALY_SCORE = Gauge(
        "sutazai_anomaly_score",
        "Anomaly score for system behavior",
        ["system_id", "component", "detector_type"],
    )

    # Air gap metrics
    AIRGAP_STATUS = Gauge(
        "sutazai_airgap_status",
        "Status of air gap verification (1=verified, 0=compromised)",
        ["system_id"],
    )

    NETWORK_CONNECTIONS = Gauge(
        "sutazai_network_connections",
        "Number of active network connections",
        ["system_id", "connection_type"],
    )


class SecurityEventType(Enum):
    """Types of security events."""

    ACCESS_ATTEMPT = "access_attempt"
    INTEGRITY_VIOLATION = "integrity_violation"
    CONFIGURATION_CHANGE = "configuration_change"
    ANOMALY_DETECTED = "anomaly_detected"
    AIRGAP_VIOLATION = "airgap_violation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SYSTEM_CHANGE = "system_change"
    DATA_ACCESS = "data_access"
    EXTERNAL_COMMUNICATION = "external_communication"


class SecuritySeverity(Enum):
    """Severity levels for security events."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Record of a security event."""

    id: str
    timestamp: float
    event_type: SecurityEventType
    severity: SecuritySeverity
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    user_id: Optional[str] = None
    resource_id: Optional[str] = None
    result: Optional[str] = None


@dataclass
class IntegrityCheck:
    """Result of a system integrity check."""

    id: str
    timestamp: float
    component: str
    passed: bool
    details: Dict[str, Any]
    duration_seconds: float
    checksum_type: str = "sha256"


class SecurityMonitor:
    """Monitor and log security events in an air-gapped environment."""

    def __init__(
        self,
        system_id: str,
        base_dir: str,
        log_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the security monitor.

        Args:
            system_id: System identifier
            base_dir: Base directory to monitor
            log_dir: Directory to store security logs
            config: Configuration options
        """
        self.system_id = system_id
        self.base_dir = os.path.abspath(base_dir)
        self.logger = logger
        self.config = config or {}

        # Set up logging directory
        self.log_dir = log_dir or os.path.join(
            os.environ.get("SUTAZAI_LOG_DIR", "/opt/sutazaiapp/logs"), "security"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Track security events
        self.events: List[SecurityEvent] = []
        self.integrity_checks: List[IntegrityCheck] = []

        # Initialize file manifest for integrity checks
        self.file_manifest: Dict[str, Dict[str, str]] = {}
        self.manifest_path = os.path.join(self.log_dir, "file_manifest.json")
        self._initialize_manifest()

        # Set up HMAC key for secure logging
        self.hmac_key = self.config.get("hmac_key", os.urandom(32))

        # Track the last airgap verification
        self.last_airgap_check: Optional[float] = None
        self.airgap_status: bool = True  # Assume valid until proven otherwise

        # Initialize monitoring threads
        self._stop_event = threading.Event()
        self._last_check_time: Optional[float] = None
        self._monitor_thread: Optional[threading.Thread] = None

        self.logger.info(f"Initialized security monitor for system {system_id}")

    def _initialize_manifest(self) -> None:
        """Initialize or load file manifest for integrity checking."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r") as f:
                    self.file_manifest = json.load(f)
                self.logger.info(
                    f"Loaded file manifest with {sum(len(files) for files in self.file_manifest.values())} files"
                )
            except Exception as e:
                self.logger.error(f"Error loading file manifest: {e}")
                self.file_manifest = {}
        else:
            self.file_manifest = {}
            self.logger.info("Initialized new file manifest")

    def start(self, check_interval: float = 300.0) -> None:
        """
        Start security monitoring in a background thread.

        Args:
            check_interval: Interval in seconds between checks
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.warning("Security monitoring already running")
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, args=(check_interval,), daemon=True)
        if self._monitor_thread:
            self._monitor_thread.start()
        logger.info("Security monitoring started")

    def stop(self) -> None:
        """Stop the security monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)
        self.logger.info(f"Stopped security monitoring for system {self.system_id}")

    def _monitor_loop(self, check_interval: float) -> None:
        """Main monitoring loop that runs in a background thread."""
        last_full_check = 0.0
        last_quick_check = 0.0
        quick_check_interval = max(
            60.0, check_interval / 5
        )  # At least 60 seconds between quick checks

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Run a full system integrity check less frequently (default: 5 minutes)
                if current_time - last_full_check >= check_interval:
                    logger.info("Running full system integrity check")
                    self.verify_system_integrity()
                    self.verify_airgap()
                    last_full_check = current_time

                # Run quick checks less frequently too
                if current_time - last_quick_check >= quick_check_interval:
                    logger.info("Running quick anomaly check")
                    self.check_for_anomalies()
                    last_quick_check = current_time

                # Sleep for a longer period
                time.sleep(
                    60.0
                )  # Sleep for a minute between checking if we need to run scans
            except Exception as e:
                logger.error(f"Error in security monitoring loop: {e}")
                time.sleep(120.0)  # Sleep longer on error

    def log_event(
        self,
        event_type: SecurityEventType,
        severity: SecuritySeverity,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "",
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        result: Optional[str] = None,
    ) -> SecurityEvent:
        """
        Log a security event.

        Args:
            event_type: Type of security event
            severity: Severity level
            summary: Brief description
            details: Additional details
            source: Source of the event
            user_id: User identifier if applicable
            resource_id: Resource identifier if applicable
            result: Result of the event

        Returns:
            Created SecurityEvent object
        """
        # Generate event ID
        event_id = str(uuid.uuid4())

        # Create event
        event = SecurityEvent(
            id=event_id,
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            summary=summary,
            details=details or {},
            source=source,
            user_id=user_id,
            resource_id=resource_id,
            result=result,
        )

        # Add to events list
        self.events.append(event)

        # Log based on severity
        log_level = logging.INFO
        if severity == SecuritySeverity.HIGH:
            log_level = logging.WARNING
        elif severity == SecuritySeverity.CRITICAL:
            log_level = logging.CRITICAL

        self.logger.log(
            log_level,
            f"Security event: [{severity.value.upper()}] {event_type.value}: {summary}",
        )

        # Write to secure log file
        self._secure_log_event(event)

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            SECURITY_EVENTS.labels(
                system_id=self.system_id,
                event_type=event_type.value,
                severity=severity.value,
            ).inc()

            if event_type == SecurityEventType.ACCESS_ATTEMPT and resource_id:
                ACCESS_ATTEMPTS.labels(
                    system_id=self.system_id,
                    resource_type=resource_id.split(":")[0]
                    if ":" in resource_id
                    else "unknown",
                    result=result or "unknown",
                ).inc()

        return event

    def _secure_log_event(self, event: SecurityEvent) -> None:
        """Write a security event to a secure log file with HMAC signature."""
        # Convert to serializable dict
        log_entry = {
            "id": event.id,
            "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
            "system_id": self.system_id,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "summary": event.summary,
            "details": event.details,
            "source": event.source,
            "user_id": event.user_id,
            "resource_id": event.resource_id,
            "result": event.result,
        }

        # Convert to JSON string
        json_str = json.dumps(log_entry)

        # Generate HMAC signature
        signature = hmac.new(self.hmac_key, json_str.encode(), hashlib.sha256).digest()

        # Encode signature as base64
        b64_signature = base64.b64encode(signature).decode()

        # Add signature to log entry
        signed_entry = json_str + f"|HMAC:{b64_signature}"

        # Generate filename based on date
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_str}_security_events.log"
        filepath = os.path.join(self.log_dir, filename)

        # Append to log file
        with open(filepath, "a") as f:
            f.write(signed_entry + "\n")

    def verify_log_integrity(self, log_path: str) -> Tuple[bool, List[int]]:
        """
        Verify the integrity of a security log file.

        Args:
            log_path: Path to log file

        Returns:
            Tuple of (integrity_valid, list_of_invalid_line_numbers)
        """
        invalid_lines = []

        try:
            with open(log_path, "r") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # Split on HMAC signature delimiter
                    parts = line.split("|HMAC:")
                    if len(parts) != 2:
                        invalid_lines.append(i)
                        continue

                    json_str, b64_signature = parts

                    # Decode signature
                    try:
                        signature = base64.b64decode(b64_signature)
                    except Exception:
                        invalid_lines.append(i)
                        continue

                    # Verify HMAC
                    expected_sig = hmac.new(
                        self.hmac_key, json_str.encode(), hashlib.sha256
                    ).digest()

                    if not hmac.compare_digest(signature, expected_sig):
                        invalid_lines.append(i)

            is_valid = len(invalid_lines) == 0
            return is_valid, invalid_lines
        except Exception as e:
            self.logger.error(f"Error verifying log integrity: {e}")
            return False, []

    def build_file_manifest(
        self,
        component: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Build a file manifest for integrity checking.

        Args:
            component: Component identifier
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude

        Returns:
            Dictionary mapping file paths to checksums
        """
        manifest: Dict[str, str] = {}

        # Default patterns
        include_patterns = include_patterns or [
            "**/*.py",
            "**/*.sh",
            "**/*.json",
            "**/*.yml",
            "**/*.yaml",
        ]
        exclude_patterns = exclude_patterns or [
            "**/__pycache__/**",
            "**/.git/**",
            "**/logs/**",
            "**/tmp/**",
        ]

        component_path = os.path.join(self.base_dir, component)
        if not os.path.exists(component_path):
            self.logger.warning(f"Component path does not exist: {component_path}")
            return manifest

        # Find files matching patterns
        matching_files = set()
        for pattern in include_patterns:
            for path in Path(component_path).glob(pattern):
                if path.is_file():
                    matching_files.add(str(path))

        # Apply exclusions
        for pattern in exclude_patterns:
            for path in Path(component_path).glob(pattern):
                if str(path) in matching_files:
                    matching_files.remove(str(path))

        # Compute checksums
        for file_path in sorted(matching_files):
            try:
                checksum = self._compute_file_checksum(file_path)
                rel_path = os.path.relpath(file_path, self.base_dir)
                manifest[rel_path] = checksum
            except Exception as e:
                self.logger.error(f"Error computing checksum for {file_path}: {e}")

        # Update the manifest
        self.file_manifest[component] = manifest

        # Save to disk
        with open(self.manifest_path, "w") as f:
            json.dump(self.file_manifest, f, indent=2)

        self.logger.info(
            f"Built file manifest for component {component} with {len(manifest)} files"
        )

        return manifest

    def _compute_file_checksum(self, file_path: str) -> str:
        """
        Compute checksum for a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 checksum of the file
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def verify_system_integrity(
        self, components: Optional[List[str]] = None
    ) -> Dict[str, IntegrityCheck]:
        """
        Verify system integrity by checking file checksums.

        Args:
            components: List of components to check (None for all)

        Returns:
            Dictionary mapping components to integrity check results
        """
        start_time = time.time()

        # Determine components to check
        components_to_check = components or list(self.file_manifest.keys())
        if not components_to_check:
            # Default components if none specified or in manifest
            components_to_check = ["app", "utils", "config", "models"]

        results = {}

        for component in components_to_check:
            check_start = time.time()

            # Build manifest if it doesn't exist
            if component not in self.file_manifest:
                self.build_file_manifest(component)

                # Skip integrity check for newly built manifests
                check_id = str(uuid.uuid4())
                check = IntegrityCheck(
                    id=check_id,
                    timestamp=time.time(),
                    component=component,
                    passed=True,
                    details={
                        "status": "manifest_created",
                        "files_added": len(self.file_manifest.get(component, {})),
                    },
                    duration_seconds=time.time() - check_start,
                )
                results[component] = check
                self.integrity_checks.append(check)
                continue

            manifest = self.file_manifest[component]
            modified_files: List[str] = []
            missing_files: List[str] = []
            new_files: List[str] = []

            # Check existing files
            for rel_path, expected_checksum in manifest.items():
                full_path = os.path.join(self.base_dir, rel_path)

                if not os.path.exists(full_path):
                    missing_files.append(rel_path)
                    continue

                current_checksum = self._compute_file_checksum(full_path)
                if current_checksum != expected_checksum:
                    modified_files.append(rel_path)

            # Look for new files
            component_path = os.path.join(self.base_dir, component)
            if os.path.exists(component_path):
                for root, _, files in os.walk(component_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, self.base_dir)

                        if rel_path not in manifest:
                            # Check exclusions
                            excluded = False
                            for pattern in [
                                "**/__pycache__/**",
                                "**/.git/**",
                                "**/logs/**",
                                "**/tmp/**",
                            ]:
                                if Path(rel_path).match(pattern):
                                    excluded = True
                                    break

                            if not excluded:
                                new_files.append(rel_path)

            # Create integrity check result
            passed = not (
                modified_files or missing_files
            )  # New files don't fail integrity
            check_id = str(uuid.uuid4())
            check = IntegrityCheck(
                id=check_id,
                timestamp=time.time(),
                component=component,
                passed=passed,
                details={
                    "modified_files": modified_files,
                    "missing_files": missing_files,
                    "new_files": new_files,
                    "total_files_checked": len(manifest),
                },
                duration_seconds=time.time() - check_start,
            )

            results[component] = check
            self.integrity_checks.append(check)

            # Record metrics
            if PROMETHEUS_AVAILABLE:
                INTEGRITY_CHECK_STATUS.labels(
                    system_id=self.system_id, component=component
                ).set(1 if passed else 0)

                INTEGRITY_CHECK_DURATION.labels(
                    system_id=self.system_id, component=component
                ).observe(check.duration_seconds)

            # Log the result
            log_level = logging.INFO if passed else logging.WARNING
            self.logger.log(
                log_level,
                f"Integrity check for {component}: {'PASSED' if passed else 'FAILED'} "
                f"({len(modified_files)} modified, {len(missing_files)} missing, {len(new_files)} new)",
            )

            # Log security event for failures
            if not passed:
                self.log_event(
                    event_type=SecurityEventType.INTEGRITY_VIOLATION,
                    severity=SecuritySeverity.HIGH,
                    summary=f"Integrity check failed for component {component}",
                    details={
                        "modified_files": modified_files,
                        "missing_files": missing_files,
                        "check_id": check_id,
                    },
                    source="integrity_check",
                )

        self.logger.info(
            f"Completed system integrity verification in {time.time() - start_time:.2f} seconds"
        )

        return results

    def verify_airgap(self) -> Dict[str, Any]:
        """
        Verify that the system is properly air-gapped.

        Returns:
            Dictionary with verification results
        """
        start_time = time.time()

        result: Dict[str, Any] = {
            "timestamp": start_time,
            "is_airgapped": True,
            "active_connections": [],
            "network_interfaces": [],
            "issues": [],
        }

        # 1. Check network interfaces
        try:
            ip_path = shutil.which("ip")
            if not ip_path:
                self.logger.error(
                    "'ip' command not found. Cannot check network interfaces."
                )
                raise FileNotFoundError("'ip' command not found")
            interfaces_output = subprocess.run(
                [ip_path, "link", "show", "up"],
                capture_output=True,
                text=True,
                check=False,
            )

            interfaces = []
            for line in interfaces_output.stdout.splitlines():
                if ":" in line and "<" in line and ">" in line:
                    iface_name = line.split(":")[1].strip()
                    if iface_name != "lo":  # Ignore loopback
                        interfaces.append(iface_name)
                        result["network_interfaces"].append(iface_name)

            # Exclude certain secure interfaces
            allowed_interfaces = self.config.get(
                "allowed_interfaces", ["docker0", "veth"]
            )
            external_interfaces = [
                iface
                for iface in interfaces
                if not any(allowed in iface for allowed in allowed_interfaces)
            ]

            if external_interfaces:
                result["is_airgapped"] = bool(False)
                result["issues"].append(
                    f"Found active external network interfaces: {', '.join(external_interfaces)}"
                )
        except Exception as e:
            self.logger.error(f"Error checking network interfaces: {e}")
            result["issues"].append(f"Error checking network interfaces: {str(e)}")

        # 2. Check active connections
        try:
            ss_path = shutil.which("ss")
            if not ss_path:
                self.logger.error(
                    "'ss' command not found. Cannot check network connections."
                )
                raise FileNotFoundError("'ss' command not found")
            netstat_output = subprocess.run(
                [ss_path, "-tuln"], capture_output=True, text=True, check=False
            )

            connections: List[Dict[str, str]] = []
            for line in netstat_output.stdout.splitlines()[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 5:
                    proto = parts[0]
                    local_addr = parts[4]

                    # Ignore localhost-only connections
                    if not local_addr.startswith(
                        "127.0.0.1:"
                    ) and not local_addr.startswith("[::1]:"):
                        connections.append(
                            {"protocol": proto, "local_address": local_addr}
                        )

            result["active_connections"] = connections

            # Check for non-localhost listeners on restricted ports
            restricted_ports = self.config.get("restricted_ports", [22, 80, 443, 8080])
            for conn in connections:
                addr = conn["local_address"]
                if ":" in addr:
                    try:
                        port = int(addr.split(":")[-1])
                        if (
                            port in restricted_ports
                            and not addr.startswith("127.0.0.1:")
                            and not addr.startswith("[::1]:")
                        ):
                            result["is_airgapped"] = False
                            result["issues"].append(
                                f"Found listener on restricted port: {addr}"
                            )
                    except ValueError:
                        pass
        except Exception as e:
            self.logger.error(f"Error checking network connections: {e}")
            result["issues"].append(f"Error checking network connections: {str(e)}")

        # 3. Additional checks for air gap
        # Check for proxy settings
        for env_var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
            if env_var in os.environ:
                result["is_airgapped"] = False
                result["issues"].append(f"Found proxy environment variable: {env_var}")

        # Record the result
        self.last_airgap_check = time.time()
        self.airgap_status = result["is_airgapped"]

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            AIRGAP_STATUS.labels(system_id=self.system_id).set(
                1 if self.airgap_status else 0
            )

            assert isinstance(result["active_connections"], list)
            NETWORK_CONNECTIONS.labels(
                system_id=self.system_id, connection_type="total"
            ).set(len(result["active_connections"]))

        # Log the result
        log_level = logging.INFO if self.airgap_status else logging.WARNING
        self.logger.log(
            log_level,
            f"Air gap verification: {'VERIFIED' if self.airgap_status else 'COMPROMISED'}",
        )

        # Log security event for compromised air gap
        if not self.airgap_status:
            self.log_event(
                event_type=SecurityEventType.AIRGAP_VIOLATION,
                severity=SecuritySeverity.HIGH,
                summary="Air gap verification failed",
                details=result,
                source="airgap_check",
            )

        return result

    def check_for_anomalies(self) -> Dict[str, float]:
        """
        Check for security anomalies in system behavior.

        Returns:
            Dictionary mapping anomaly types to scores (0-1)
        """
        # This is a simplified implementation
        # A real implementation would use machine learning models
        # or statistical methods to detect anomalies

        anomaly_scores = {
            "process_anomaly": 0.0,
            "user_behavior": 0.0,
            "file_activity": 0.0,
            "resource_usage": 0.0,
        }

        # Check for unusual processes
        try:
            ps_path = shutil.which("ps")
            if not ps_path:
                self.logger.error(
                    "'ps' command not found. Cannot check for process anomalies."
                )
                raise FileNotFoundError("'ps' command not found")
            ps_output = subprocess.run(
                [ps_path, "-eo", "user,pid,ppid,cmd", "--sort=-pcpu"],
                capture_output=True,
                text=True,
                check=False,
            )

            suspicious_processes = []
            for line in ps_output.stdout.splitlines()[1:]:  # Skip header
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    cmd = parts[3]

                    # Simple check for suspicious process names
                    suspicious_keywords = [
                        "netcat",
                        "nc ",
                        "nmap",
                        "wireshark",
                        "tcpdump",
                        "wget",
                    ]
                    if any(keyword in cmd.lower() for keyword in suspicious_keywords):
                        suspicious_processes.append(cmd)

            # Calculate anomaly score based on suspicious processes
            if suspicious_processes:
                anomaly_scores["process_anomaly"] = min(
                    1.0, len(suspicious_processes) / 3.0
                )

                if anomaly_scores["process_anomaly"] > 0.3:
                    self.log_event(
                        event_type=SecurityEventType.ANOMALY_DETECTED,
                        severity=SecuritySeverity.MEDIUM,
                        summary="Detected suspicious processes",
                        details={"suspicious_processes": suspicious_processes},
                        source="anomaly_detection",
                    )
        except Exception as e:
            self.logger.error(f"Error checking for process anomalies: {e}")

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            for anomaly_type, score in anomaly_scores.items():
                ANOMALY_SCORE.labels(
                    system_id=self.system_id,
                    component="system",
                    detector_type=anomaly_type,
                ).set(score)

        return anomaly_scores

    def log_access_attempt(
        self,
        resource_id: str,
        user_id: str,
        action: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> SecurityEvent:
        """
        Log an access attempt to a resource.

        Args:
            resource_id: Resource identifier
            user_id: User identifier
            action: Action attempted
            result: Result of the attempt
            details: Additional details

        Returns:
            Created SecurityEvent object
        """
        # Determine severity based on result
        severity = SecuritySeverity.INFO
        if result in ["denied", "failed"]:
            severity = SecuritySeverity.MEDIUM
        elif result == "error":
            severity = SecuritySeverity.HIGH

        # Create security event
        event = self.log_event(
            event_type=SecurityEventType.ACCESS_ATTEMPT,
            severity=severity,
            summary=f"Access attempt: {action} on {resource_id} by {user_id}: {result}",
            details=details or {},
            source="access_control",
            user_id=user_id,
            resource_id=resource_id,
            result=result,
        )

        return event


# Factory function
def create_security_monitor(
    system_id: str,
    base_dir: str,
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None,
) -> SecurityMonitor:
    """Create and return a new security monitor."""
    return SecurityMonitor(
        system_id=system_id, base_dir=base_dir, config=config, log_dir=log_dir
    )
