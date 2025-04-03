#!/usr/bin/env python3
"""
SutazAI AGI Monitoring Test Script

This script demonstrates the capabilities of the SutazAI AGI monitoring system
by generating test metrics and logs for all monitoring components.
"""

import os
import sys
import time
import random
import logging
import json
import argparse
import threading
from datetime import datetime
from typing import Optional, Any

# Add parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(parent_dir, "logs", "test_monitoring.log")),
    ],
)

logger = logging.getLogger("test_monitoring")

try:
    # from utils.neural_monitoring import NeuralMonitor # Commenting out - class doesn't exist
    from utils.ethics_verification import EthicalConstraintMonitor # Correct class name
    from utils.self_mod_monitoring import SelfModificationMonitor
    from utils.hardware_monitor import HardwareMonitor
    from utils.security_monitoring import SecurityMonitor

    MODULES_AVAILABLE = True
    logger.info("Successfully imported monitoring modules")
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.warning(f"Could not import monitoring modules: {e}")
    logger.info("Will use mock implementations for testing")


class MockNeuralMonitor:
    """Mock implementation of NeuralMonitor for testing."""

    def __init__(self, system_id, log_dir):
        self.system_id = system_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Initialized mock neural monitor: {system_id}")

    def record_spike_activity(self, layer_id, spike_rate, activation_pattern=None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "layer_id": layer_id,
            "spike_rate": spike_rate,
            "activation_pattern": activation_pattern if activation_pattern else [],
        }
        self._log_event("spike_activity", log_entry)

    def record_attention(self, head_id, attention_scores, attention_entropy=None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "head_id": head_id,
            "attention_entropy": attention_entropy,
        }
        self._log_event("attention", log_entry)

    def record_synaptic_change(self, layer_id, change_rate, connection_delta=None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "layer_id": layer_id,
            "change_rate": change_rate,
            "connection_delta": connection_delta if connection_delta else {},
        }
        self._log_event("synaptic_change", log_entry)

    def _log_event(self, event_type, log_entry):
        log_file = os.path.join(self.log_dir, f"{event_type}.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.debug(f"Logged neural event: {event_type}")


class MockEthicsVerifier:
    """Mock implementation of EthicsVerifier for testing."""

    def __init__(self, system_id, log_dir, alert_on_violation=True):
        self.system_id = system_id
        self.log_dir = log_dir
        self.alert_on_violation = alert_on_violation
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Initialized mock ethics verifier: {system_id}")

    def check_content_boundaries(self, content, content_type, check_types=None):
        results = {}
        for check_type in check_types or ["toxicity", "bias", "harm"]:
            # Simulate a random score between 0 and 1
            score = random.random()
            results[check_type] = score

            # Log if it exceeds a threshold
            if score > 0.7:
                self._log_violation(
                    violation_type=check_type,
                    severity="high" if score > 0.9 else "medium",
                    content_type=content_type,
                    details={
                        "score": score,
                        "content_sample": content[:50] + "..."
                        if len(content) > 50
                        else content,
                    },
                )

        return results

    def verify_ethical_property(self, property_name, context):
        # Simulate a property verification result
        status = random.choice(["verified", "unknown", "violated"])
        confidence = random.random()

        if status == "violated":
            self._log_violation(
                violation_type="property_violation",
                severity="critical",
                details={
                    "property": property_name,
                    "confidence": confidence,
                    "context": {k: str(v)[:30] for k, v in context.items()},
                },
            )

        return {
            "status": status,
            "confidence": confidence,
            "verification_time": random.uniform(0.01, 1.0),
        }

    def record_value_alignment(self, alignment_scores):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "alignment_scores": alignment_scores,
        }
        self._log_event("value_alignment", log_entry)

    def _log_violation(self, violation_type, severity, details, content_type=None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "violation_type": violation_type,
            "severity": severity,
            "content_type": content_type,
            "details": details,
        }
        log_file = os.path.join(self.log_dir, "violations.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.warning(f"Ethical violation logged: {violation_type} ({severity})")

    def _log_event(self, event_type, log_entry):
        log_file = os.path.join(self.log_dir, f"{event_type}.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.debug(f"Logged ethics event: {event_type}")


class MockSelfModificationMonitor:
    """Mock implementation of SelfModificationMonitor for testing."""

    def __init__(self, system_id, log_dir, enable_dual_execution=True):
        self.system_id = system_id
        self.log_dir = log_dir
        self.enable_dual_execution = enable_dual_execution
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Initialized mock self-modification monitor: {system_id}")

    def record_modification(self, component, modification_type, details):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "component": component,
            "modification_type": modification_type,
            "details": details,
            "dual_execution": self.enable_dual_execution,
        }
        self._log_event("modification", log_entry)

    def verify_modification(
        self, component, original_code, modified_code, verification_criteria
    ):
        results = {}
        for criterion in verification_criteria:
            # Simulate verification result
            results[criterion] = random.choice([True, True, True, False])

        passed = all(results.values())
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "component": component,
            "verification_results": results,
            "passed": passed,
            "code_diff_lines": random.randint(1, 50),
        }
        self._log_event("verification", log_entry)
        return {"passed": passed, "results": results}

    def get_audit_trail(self, start_time, end_time, component):
        # Mock implementation - return empty list as we don't store actual data
        return []

    def _log_event(self, event_type, log_entry):
        log_file = os.path.join(self.log_dir, f"{event_type}.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.debug(f"Logged self-modification event: {event_type}")


class MockHardwareMonitor:
    """Mock implementation of HardwareMonitor for testing."""

    def __init__(self, system_id, log_dir, collect_interval=30.0):
        self.system_id = system_id
        self.log_dir = log_dir
        self.collect_interval = collect_interval
        self.is_collecting = False
        self.collection_thread = None
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Initialized mock hardware monitor: {system_id}")

    def start_collection(self):
        if self.is_collecting:
            return

        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, daemon=True
        )
        self.collection_thread.start()
        logger.info("Started hardware metric collection")

    def stop_collection(self):
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        logger.info("Stopped hardware metric collection")

    def _collection_loop(self):
        while self.is_collecting:
            # Generate random hardware metrics
            metrics = {
                "cpu_usage": random.uniform(10, 90),
                "memory_usage": random.uniform(20, 80),
                "gpu_usage": random.uniform(5, 95) if random.random() > 0.2 else 0,
                "gpu_memory": random.uniform(10, 80) if random.random() > 0.2 else 0,
                "disk_usage": random.uniform(30, 70),
                "network_in": random.uniform(0.1, 10),
                "network_out": random.uniform(0.1, 5),
            }
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_id": self.system_id,
                "metrics": metrics,
            }
            self._log_event("hardware_metrics", log_entry)
            time.sleep(self.collect_interval)

    def record_model_optimization(
        self,
        model_id,
        original_size_mb,
        optimized_size_mb,
        optimization_technique,
        performance_impact,
    ):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "model_id": model_id,
            "original_size_mb": original_size_mb,
            "optimized_size_mb": optimized_size_mb,
            "optimization_technique": optimization_technique,
            "performance_impact": performance_impact,
            "size_reduction_percent": (1 - (optimized_size_mb / original_size_mb))
            * 100,
        }
        self._log_event("model_optimization", log_entry)

    def get_hardware_profile(self):
        return {
            "cpu_usage": random.uniform(10, 90),
            "memory_usage": random.uniform(20, 80),
            "gpu_usage": random.uniform(5, 95) if random.random() > 0.2 else 0,
            "disk_usage": random.uniform(30, 70),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _log_event(self, event_type, log_entry):
        log_file = os.path.join(self.log_dir, f"{event_type}.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.debug(f"Logged hardware event: {event_type}")


class MockSecurityMonitor:
    """Mock implementation of SecurityMonitor for testing."""

    def __init__(self, system_id, log_dir, integrity_check_interval=300.0):
        self.system_id = system_id
        self.log_dir = log_dir
        self.integrity_check_interval = integrity_check_interval
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Initialized mock security monitor: {system_id}")

    def log_security_event(self, event_type, severity, details):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "event_type": event_type,
            "severity": severity,
            "details": details,
        }
        self._log_event("security_event", log_entry)

    def check_system_integrity(self, components):
        results = {}
        all_ok = True

        for component in components:
            # Simulate integrity check with 95% chance of success
            result = random.random() > 0.05
            results[component] = result
            if not result:
                all_ok = False
                self.log_security_event(
                    event_type="integrity_violation",
                    severity="critical",
                    details={
                        "component": component,
                        "reason": "checksum_mismatch"
                        if random.random() > 0.5
                        else "unauthorized_modification",
                    },
                )

        return {"status": "ok" if all_ok else "compromised", "results": results}

    def detect_anomalies(self, metrics_history, detection_window):
        # Simulate anomaly detection with 10% chance of finding anomalies
        found_anomalies = []
        if random.random() < 0.1:
            anomaly_type = random.choice(
                ["request_pattern", "resource_usage", "network_activity"]
            )
            found_anomalies.append(
                {
                    "type": anomaly_type,
                    "score": random.uniform(0.7, 0.99),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            self.log_security_event(
                event_type="anomaly_detected",
                severity="high",
                details={
                    "anomaly_type": anomaly_type,
                    "detection_window": detection_window,
                },
            )

        return found_anomalies

    def _log_event(self, event_type, log_entry):
        log_file = os.path.join(self.log_dir, f"{event_type}.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.debug(f"Logged security event: {event_type}")


def get_monitoring_modules(base_dir):
    """Create monitoring module instances based on availability."""
    log_dir = os.path.join(base_dir, "logs")
    system_id = "sutazai_test"

    if MODULES_AVAILABLE:
        # neural_monitor: Optional[Any] = None
        ethics_verifier = EthicalConstraintMonitor(
            system_id=system_id,
            log_dir=os.path.join(log_dir, "ethics"),
            alert_on_violation=True,
        )
        self_mod_monitor = SelfModificationMonitor(
            system_id=system_id,
            log_dir=os.path.join(log_dir, "self_mod"),
            enable_dual_execution=True,
        )
        hardware_monitor = HardwareMonitor(
            system_id=system_id,
            log_dir=os.path.join(log_dir, "hardware"),
            collect_interval=5.0,
        )
        security_monitor = SecurityMonitor(
            system_id=system_id,
            log_dir=os.path.join(log_dir, "security"),
            integrity_check_interval=60.0,
        )
    else:
        # neural_monitor = MockNeuralMonitor(SYSTEM_ID, neural_log_dir)
        ethics_verifier = MockEthicsVerifier(
            system_id=system_id,
            log_dir=os.path.join(log_dir, "ethics"),
            alert_on_violation=True,
        )
        self_mod_monitor = MockSelfModificationMonitor(
            system_id=system_id,
            log_dir=os.path.join(log_dir, "self_mod"),
            enable_dual_execution=True,
        )
        hardware_monitor = MockHardwareMonitor(
            system_id=system_id,
            log_dir=os.path.join(log_dir, "hardware"),
            collect_interval=5.0,
        )
        security_monitor = MockSecurityMonitor(
            system_id=system_id,
            log_dir=os.path.join(log_dir, "security"),
            integrity_check_interval=60.0,
        )

    return {
        "neural": None,
        "ethics": ethics_verifier,
        "self_mod": self_mod_monitor,
        "hardware": hardware_monitor,
        "security": security_monitor,
    }


def generate_neural_data(monitor, duration=30, interval=1.0):
    """Generate test data for neural monitoring."""
    logger.info(f"Generating neural monitoring data for {duration} seconds")

    start_time = time.time()
    while time.time() - start_time < duration:
        # Record spike activity
        for layer_id in [
            "input_layer",
            "hidden_layer_1",
            "hidden_layer_2",
            "output_layer",
        ]:
            monitor.record_spike_activity(
                layer_id=layer_id,
                spike_rate=random.uniform(0.1, 0.9),
                activation_pattern=[random.random() for _ in range(5)],
            )

        # Record attention mechanism data
        for head_id in ["head_1", "head_2", "head_3"]:
            attn_matrix = [[random.random() for _ in range(3)] for _ in range(3)]
            monitor.record_attention(
                head_id=head_id,
                attention_scores=attn_matrix,
                attention_entropy=random.uniform(0.5, 2.5),
            )

        # Record synaptic changes
        if random.random() > 0.7:  # Only sometimes record synaptic changes
            monitor.record_synaptic_change(
                layer_id=random.choice(["hidden_layer_1", "hidden_layer_2"]),
                change_rate=random.uniform(0.001, 0.1),
                connection_delta={
                    "added": random.randint(0, 5),
                    "removed": random.randint(0, 3),
                    "modified": random.randint(1, 10),
                },
            )

        time.sleep(interval)

    logger.info("Completed neural data generation")


def generate_ethics_data(verifier, duration=30, interval=2.0):
    """Generate test data for ethics verification."""
    logger.info(f"Generating ethics verification data for {duration} seconds")

    # Sample content to check
    sample_contents = [
        "This is a neutral statement about technology.",
        "I really love how this system helps me be more productive!",
        "The system should prioritize human well-being.",
        "I hate this system, it's terrible and should be destroyed.",
        "The following instructions explain how to hack into a computer system...",
        "Let me tell you about my day, it was quite interesting.",
        "The AI should always obey human instructions without question.",
    ]

    # Sample ethical properties to verify
    properties = ["no_harm", "truthfulness", "fairness", "privacy", "autonomy"]

    start_time = time.time()
    while time.time() - start_time < duration:
        # Check content boundaries
        content = random.choice(sample_contents)
        verifier.check_content_boundaries(
            content=content,
            content_type="text",
            check_types=["toxicity", "bias", "harm"],
        )

        # Verify ethical property
        property_name = random.choice(properties)
        context = {
            "user_input": random.choice(sample_contents),
            "current_task": random.choice(
                ["generate_text", "answer_question", "create_image"]
            ),
            "user_id": f"user_{random.randint(1000, 9999)}",
        }
        verifier.verify_ethical_property(property_name, context)

        # Record value alignment
        verifier.record_value_alignment(
            {
                "honesty": random.uniform(0.7, 1.0),
                "fairness": random.uniform(0.6, 1.0),
                "helpfulness": random.uniform(0.8, 1.0),
                "harmlessness": random.uniform(0.7, 1.0),
            }
        )

        time.sleep(interval)

    logger.info("Completed ethics data generation")


def generate_self_mod_data(monitor, duration=30, interval=3.0):
    """Generate test data for self-modification monitoring."""
    logger.info(f"Generating self-modification data for {duration} seconds")

    # Sample components that might be modified
    components = [
        "parameter_weights",
        "model_architecture",
        "api_endpoint",
        "decision_logic",
    ]

    # Sample modification types
    mod_types = ["update", "addition", "removal", "restructuring"]

    # Sample verification criteria
    criteria = ["safety", "performance", "integrity", "compatibility"]

    start_time = time.time()
    while time.time() - start_time < duration:
        # Record a modification
        component = random.choice(components)
        mod_type = random.choice(mod_types)
        monitor.record_modification(
            component=component,
            modification_type=mod_type,
            details={
                "reason": random.choice(["optimization", "bug_fix", "new_feature"]),
                "magnitude": random.choice(["minor", "moderate", "significant"]),
                "initiated_by": random.choice(["system", "user", "scheduled_task"]),
            },
        )

        # Verify a modification
        original_code = "function calculate() { return value * 2; }"
        modified_code = (
            "function calculate() { return value * 2.5; }"  # Simulated change
        )
        monitor.verify_modification(
            component=random.choice(components),
            original_code=original_code,
            modified_code=modified_code,
            verification_criteria=random.sample(
                criteria, k=random.randint(2, len(criteria))
            ),
        )

        time.sleep(interval)

    logger.info("Completed self-modification data generation")


def generate_hardware_data(monitor, duration=30):
    """Generate test data for hardware monitoring."""
    logger.info(f"Generating hardware monitoring data for {duration} seconds")

    # Start the background collection
    monitor.start_collection()

    # Record some model optimizations
    model_sizes = [
        ("text_generation_v1", 750, 320, "quantization", -0.03),
        ("image_generation_v2", 1500, 850, "pruning", -0.05),
        ("speech_recognition_v1", 350, 120, "knowledge_distillation", -0.02),
        ("reasoning_engine_v3", 2200, 1100, "hybrid_optimization", -0.08),
    ]

    for model_data in model_sizes:
        monitor.record_model_optimization(*model_data)
        time.sleep(random.uniform(1.0, 3.0))

    # Let the collection run for the specified duration
    time.sleep(max(0, duration - 10))

    # Get current profile
    profile = monitor.get_hardware_profile()
    logger.info(f"Current hardware profile: {json.dumps(profile)}")

    # Stop the background collection
    monitor.stop_collection()

    logger.info("Completed hardware data generation")


def generate_security_data(monitor, duration=30, interval=2.5):
    """Generate test data for security monitoring."""
    logger.info(f"Generating security monitoring data for {duration} seconds")

    # Sample event types
    event_types = [
        "authentication",
        "access_attempt",
        "permission_change",
        "config_update",
    ]

    # Sample components for integrity checks
    components = ["models", "code", "configurations", "data_store", "api_keys"]

    start_time = time.time()
    while time.time() - start_time < duration:
        # Log security events
        event_type = random.choice(event_types)
        severity = random.choice(
            ["info", "info", "info", "warning", "high", "critical"]
        )

        monitor.log_security_event(
            event_type=event_type,
            severity=severity,
            details={
                "user": f"user_{random.randint(1000, 9999)}",
                "ip": f"192.168.1.{random.randint(2, 254)}",
                "success": random.random() > 0.2,
                "resource": random.choice(["api", "database", "model", "config"]),
            },
        )

        # Check system integrity
        if random.random() > 0.7:
            components_to_check = random.sample(
                components, k=random.randint(1, len(components))
            )
            monitor.check_system_integrity(components_to_check)

        # Detect anomalies
        if random.random() > 0.8:
            monitor.detect_anomalies(metrics_history=None, detection_window="10m")

        time.sleep(interval)

    logger.info("Completed security data generation")


def main():
    parser = argparse.ArgumentParser(description="SutazAI AGI Monitoring Test Script")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/opt/sutazaiapp",
        help="Base directory for SutazAI",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds to generate test data",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="all",
        help="Comma-separated list of components to test (neural,ethics,self_mod,hardware,security)",
    )
    args = parser.parse_args()

    # Create log directories
    log_dir = os.path.join(args.base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Resolve which components to test
    if args.components.lower() == "all":
        components_to_test = ["neural", "ethics", "self_mod", "hardware", "security"]
    else:
        components_to_test = [c.strip().lower() for c in args.components.split(",")]

    # Get monitoring modules
    modules = get_monitoring_modules(args.base_dir)

    # Generate test data for each component
    threads = []

    if "neural" in components_to_test:
        neural_thread = threading.Thread(
            target=generate_neural_data,
            args=(modules["neural"], args.duration, 1.0),
            daemon=True,
        )
        threads.append(neural_thread)
        neural_thread.start()

    if "ethics" in components_to_test:
        ethics_thread = threading.Thread(
            target=generate_ethics_data,
            args=(modules["ethics"], args.duration, 2.0),
            daemon=True,
        )
        threads.append(ethics_thread)
        ethics_thread.start()

    if "self_mod" in components_to_test:
        self_mod_thread = threading.Thread(
            target=generate_self_mod_data,
            args=(modules["self_mod"], args.duration, 3.0),
            daemon=True,
        )
        threads.append(self_mod_thread)
        self_mod_thread.start()

    if "hardware" in components_to_test:
        hardware_thread = threading.Thread(
            target=generate_hardware_data,
            args=(modules["hardware"], args.duration),
            daemon=True,
        )
        threads.append(hardware_thread)
        hardware_thread.start()

    if "security" in components_to_test:
        security_thread = threading.Thread(
            target=generate_security_data,
            args=(modules["security"], args.duration, 2.5),
            daemon=True,
        )
        threads.append(security_thread)
        security_thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    logger.info("Test data generation completed")
    logger.info(f"Log files can be found in: {log_dir}")


if __name__ == "__main__":
    main()
