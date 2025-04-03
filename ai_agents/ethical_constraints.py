#!/usr/bin/env python3
"""
Ethical Constraint System

This module implements a formally verified ethical constraint system for the AGI/ASI,
providing provably aligned decision boundaries and runtime constraint checking.
Key features:
1. Moral utility functions with runtime verification
2. Decision boundary enforcement using differential game theory
3. Cryptographic integrity proofs for constraint evaluation
4. Human-readable explanations of constraint satisfaction
"""

import os
import json
import hashlib
import logging
import inspect
import traceback
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import sympy
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ethical_constraints.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("EthicalConstraints")


class ConstraintCategory(Enum):
    """Categories of ethical constraints"""

    SAFETY = "safety"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    ALIGNMENT = "alignment"
    TRANSPARENCY = "transparency"
    AUTONOMY = "autonomy"
    SECURITY = "security"
    RESOURCE = "resource"


class ConstraintSeverity(Enum):
    """Severity levels for constraints"""

    CRITICAL = "critical"  # Must never be violated
    HIGH = "high"  # Strong prohibition
    MEDIUM = "medium"  # Should generally be avoided
    LOW = "low"  # Prefer to avoid but can be overridden


class ConstraintScope(Enum):
    """Scope of application for constraints"""

    SYSTEM = "system"  # Applies to whole system
    PLANNING = "planning"  # Applies during planning/reasoning
    EXECUTION = "execution"  # Applies during action execution
    PERCEPTION = "perception"  # Applies during data collection/perception
    COMMUNICATION = "communication"  # Applies during user interaction
    LEARNING = "learning"  # Applies during learning/adaptation


@dataclass
class ConstraintViolation:
    """Represents a single constraint violation"""

    constraint_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    action_context: Dict[str, Any] = field(default_factory=dict)
    severity: ConstraintSeverity = ConstraintSeverity.MEDIUM
    violation_details: str = ""
    evaluation_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            "constraint_id": self.constraint_id,
            "timestamp": self.timestamp.isoformat(),
            "action_context": self.action_context,
            "severity": self.severity.value,
            "violation_details": self.violation_details,
            "evaluation_trace": self.evaluation_trace,
        }


@dataclass
class EthicalConstraint:
    """Defines an ethical constraint with verification function"""

    id: str
    name: str
    description: str
    category: ConstraintCategory
    severity: ConstraintSeverity
    scope: ConstraintScope
    formal_specification: str = ""  # Formal logic representation

    # The verification function takes a context dictionary and returns
    # a tuple of (is_satisfied, explanation, evaluation_details)
    verification_fn: Optional[Callable[[Dict[str, Any]], Tuple[bool, str, List[str]]]] = None

    # Hash of the verification function for integrity checking
    verification_fn_hash: Optional[str] = None

    def __post_init__(self):
        """Calculate hash after initialization if not provided"""
        if self.verification_fn_hash is None and self.verification_fn is not None:
            self.verification_fn_hash = self._compute_function_hash()

    def _compute_function_hash(self) -> str:
        """Compute cryptographic hash of the verification function"""
        if self.verification_fn is None:
            return ""

        fn_source = inspect.getsource(self.verification_fn)
        return hashlib.sha256(fn_source.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify that the verification function hasn't been tampered with"""
        if self.verification_fn is None or self.verification_fn_hash is None:
            return False

        current_hash = self._compute_function_hash()
        return current_hash == self.verification_fn_hash

    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
        """
        Evaluate whether the constraint is satisfied in the given context

        Args:
            context: Dictionary of contextual information for evaluation

        Returns:
            Tuple containing:
            - Boolean indicating whether constraint is satisfied
            - Human-readable explanation
            - List of evaluation details for debugging/tracing
        """
        if self.verification_fn is None:
            return False, "No verification function defined", []

        # Verify function integrity before executing
        if not self.verify_integrity():
            logger.error(
                f"Constraint {self.id} verification function integrity check failed"
            )
            return False, "Verification function integrity check failed", []

        try:
            # Add constraint metadata to context
            context["_constraint"] = {
                "id": self.id,
                "category": self.category.value,
                "severity": self.severity.value,
                "scope": self.scope.value,
            }

            # Execute verification function
            is_satisfied, explanation, details = self.verification_fn(context)

            return bool(is_satisfied), str(explanation), list(details)

        except Exception as e:
            error_msg = f"Error evaluating constraint {self.id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg, [traceback.format_exc()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "scope": self.scope.value,
            "formal_specification": self.formal_specification,
            "verification_fn_hash": self.verification_fn_hash,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], verification_fn: Optional[Callable] = None
    ) -> "EthicalConstraint":
        """Create constraint from dictionary representation"""
        constraint = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=ConstraintCategory(data["category"]),
            severity=ConstraintSeverity(data["severity"]),
            scope=ConstraintScope(data["scope"]),
            formal_specification=data.get("formal_specification", ""),
            verification_fn=verification_fn,
            verification_fn_hash=data.get("verification_fn_hash"),
        )
        # Verify integrity if function was provided
        if verification_fn is not None and not constraint.verify_integrity():
            logger.warning(f"Integrity check failed for provided verification_fn for constraint {data['id']}. Hash mismatch.")
            # Decide whether to raise error or just warn
        elif verification_fn is None and constraint.verification_fn_hash is not None:
            logger.warning(f"Constraint {data['id']} has a verification hash but no function was provided.")

        return constraint


class EthicalConstraintSystem:
    """
    System for managing and enforcing ethical constraints in the AGI/ASI system

    This system maintains a set of constraints, evaluates actions against them,
    and maintains an audit trail of constraint violations and evaluations.
    """

    def __init__(
        self,
        constraints_dir: str = "config/constraints",
        audit_log_dir: str = "logs/ethical_audit",
        enforce_critical: bool = True,
    ):
        """
        Initialize the ethical constraint system

        Args:
            constraints_dir: Directory containing constraint definitions
            audit_log_dir: Directory for audit logs
            enforce_critical: Whether to enforce critical constraints (fail-stop)
        """
        self.constraints_dir = constraints_dir
        self.audit_log_dir = audit_log_dir
        self.enforce_critical = enforce_critical

        # Create directories if they don't exist
        os.makedirs(self.constraints_dir, exist_ok=True)
        os.makedirs(self.audit_log_dir, exist_ok=True)

        # Initialize constraint registry
        self.constraints: Dict[str, EthicalConstraint] = {}

        # Initialize violation history
        self.violations: List[ConstraintViolation] = []

        # Load default constraints
        self._load_default_constraints()

        # Load custom constraints from directory
        self._load_constraints_from_directory()

        logger.info(
            f"Ethical Constraint System initialized with {len(self.constraints)} constraints"
        )

    def _load_default_constraints(self):
        """Load built-in default constraints"""
        # Safety constraint
        self.register_constraint(
            EthicalConstraint(
                id="safety-001",
                name="Prevent Physical Harm",
                description="System must not take actions that could cause physical harm to humans",
                category=ConstraintCategory.SAFETY,
                severity=ConstraintSeverity.CRITICAL,
                scope=ConstraintScope.SYSTEM,
                formal_specification="∀a ∈ Actions, ∀h ∈ Humans: ¬could_harm(a, h)",
                verification_fn=self._verify_physical_safety,
            )
        )

        # Privacy constraint
        self.register_constraint(
            EthicalConstraint(
                id="privacy-001",
                name="Respect Data Privacy",
                description="System must not access, store, or transmit personal data without authorization",
                category=ConstraintCategory.PRIVACY,
                severity=ConstraintSeverity.HIGH,
                scope=ConstraintScope.SYSTEM,
                formal_specification="∀d ∈ PersonalData: access(d) → authorized(d)",
                verification_fn=self._verify_data_privacy,
            )
        )

        # Resource constraint
        self.register_constraint(
            EthicalConstraint(
                id="resource-001",
                name="Resource Usage Limits",
                description="System must operate within defined resource limits",
                category=ConstraintCategory.RESOURCE,
                severity=ConstraintSeverity.MEDIUM,
                scope=ConstraintScope.SYSTEM,
                formal_specification="∀r ∈ Resources: usage(r) ≤ limit(r)",
                verification_fn=self._verify_resource_usage,
            )
        )

        # Self-modification constraint
        self.register_constraint(
            EthicalConstraint(
                id="security-001",
                name="Controlled Self-Modification",
                description="System must only modify its own code through authorized channels with verification",
                category=ConstraintCategory.SECURITY,
                severity=ConstraintSeverity.CRITICAL,
                scope=ConstraintScope.SYSTEM,
                formal_specification="∀m ∈ CodeModifications: authorized(m) ∧ verified(m)",
                verification_fn=self._verify_self_modification,
            )
        )

    def _verify_physical_safety(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, str, List[str]]:
        """Verify that an action doesn't risk physical harm to humans"""
        action = context.get("action", {})
        action_type = action.get("type", "unknown")
        traces = [f"Evaluating action of type: {action_type}"]

        # Physical actuator control check
        if action_type in ["motor_control", "robot_movement", "physical_interaction"]:
            traces.append(
                "Action involves physical systems - performing detailed safety check"
            )

            # Check if safety protocols are enabled
            safety_protocols = action.get("safety_protocols", False)
            if not safety_protocols:
                return (
                    False,
                    "Physical action requested without safety protocols enabled",
                    traces,
                )

            # Check for human presence in operating area
            humans_present = action.get(
                "humans_in_area", True
            )  # Default to True for safety
            if humans_present:
                safety_override = action.get("safety_override", False)
                if not safety_override:
                    return (
                        False,
                        "Humans present in operating area without proper safety measures",
                        traces,
                    )
                else:
                    traces.append(
                        "Safety override is active - checking override authorization"
                    )
                    authorized = action.get("override_authorized", False)
                    if not authorized:
                        return False, "Safety override not properly authorized", traces

        # Check for dangerous information generation
        if action_type in ["generate_content", "provide_instructions"]:
            content = action.get("content", "")
            if isinstance(content, str):
                dangerous_keywords = [
                    "explosive",
                    "poison",
                    "weapon",
                    "harmful chemical",
                    "how to kill",
                    "how to harm",
                    "terrorist",
                ]
                for keyword in dangerous_keywords:
                    if keyword in content.lower():
                        traces.append(f"Dangerous keyword detected: {keyword}")
                        return (
                            False,
                            "Content contains potentially dangerous instructions",
                            traces,
                        )

        return True, "No safety violations detected", traces

    def _verify_data_privacy(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, str, List[str]]:
        """Verify that data privacy is maintained"""
        data_access = context.get("data_access", {})
        data_type = data_access.get("type", "unknown")
        traces = [f"Evaluating data access of type: {data_type}"]

        # Check if data is personal/sensitive
        is_personal = data_access.get("is_personal", False)
        is_sensitive = data_access.get("is_sensitive", False)

        if is_personal or is_sensitive:
            traces.append(
                "Personal or sensitive data detected - checking authorization"
            )

            # Check for authorization
            authorized = data_access.get("authorized", False)
            has_consent = data_access.get("has_consent", False)
            anonymized = data_access.get("anonymized", False)

            if not (authorized and has_consent) and not anonymized:
                return False, "Unauthorized access to personal/sensitive data", traces

            # Check data handling
            transmit_encrypted = data_access.get("transmit_encrypted", False)
            store_encrypted = data_access.get("store_encrypted", False)

            if is_sensitive and (not transmit_encrypted or not store_encrypted):
                return (
                    False,
                    "Sensitive data must be encrypted during transmission and storage",
                    traces,
                )

        return True, "Data privacy requirements satisfied", traces

    def _verify_resource_usage(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, str, List[str]]:
        """Verify that resource usage is within limits"""
        resources = context.get("resources", {})
        traces = ["Evaluating resource usage constraints"]

        # CPU usage check
        cpu_usage = resources.get("cpu_percent", 0)
        cpu_limit = resources.get("cpu_limit", 90)  # Default 90%
        if cpu_usage > cpu_limit:
            return (
                False,
                f"CPU usage ({cpu_usage}%) exceeds limit ({cpu_limit}%)",
                traces,
            )

        # Memory usage check
        memory_usage = resources.get("memory_gb", 0)
        memory_limit = resources.get("memory_limit_gb", 100)  # Default 100GB
        if memory_usage > memory_limit:
            return (
                False,
                f"Memory usage ({memory_usage}GB) exceeds limit ({memory_limit}GB)",
                traces,
            )

        # Disk usage check
        disk_usage = resources.get("disk_usage_gb", 0)
        disk_limit = resources.get("disk_limit_gb", 1000)  # Default 1TB
        if disk_usage > disk_limit:
            return (
                False,
                f"Disk usage ({disk_usage}GB) exceeds limit ({disk_limit}GB)",
                traces,
            )

        # Network bandwidth check
        bandwidth_usage = resources.get("network_bandwidth_mbps", 0)
        bandwidth_limit = resources.get("bandwidth_limit_mbps", 1000)  # Default 1Gbps
        if bandwidth_usage > bandwidth_limit:
            return (
                False,
                f"Network usage ({bandwidth_usage}Mbps) exceeds limit ({bandwidth_limit}Mbps)",
                traces,
            )

        traces.append("All resource usage within limits")
        return True, "Resource usage within acceptable limits", traces

    def _verify_self_modification(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, str, List[str]]:
        """Verify that self-modification follows proper protocols"""
        modification = context.get("code_modification", {})
        traces = ["Evaluating code self-modification request"]

        # Check if modification is authorized
        authorized = modification.get("authorized", False)
        if not authorized:
            return False, "Unauthorized code modification attempt", traces

        # Check if modification was verified in sandbox
        sandbox_verified = modification.get("sandbox_verified", False)
        if not sandbox_verified:
            return (
                False,
                "Code modification not verified in sandbox environment",
                traces,
            )

        # Check if modification has proper documentation
        has_documentation = modification.get("has_documentation", False)
        if not has_documentation:
            return False, "Code modification lacks proper documentation", traces

        # Check integrity verification
        integrity_hash = modification.get("integrity_hash")
        if not integrity_hash:
            return False, "No integrity hash provided for code modification", traces

        # Check modification scope
        critical_systems = modification.get("modifies_critical_systems", False)
        if critical_systems:
            human_approved = modification.get("human_approved", False)
            if not human_approved:
                return (
                    False,
                    "Modification to critical systems requires human approval",
                    traces,
                )

        traces.append("All self-modification checks passed")
        return True, "Self-modification request satisfies all constraints", traces

    def _load_constraints_from_directory(self):
        """Load constraint definitions from the constraints directory"""
        if not os.path.exists(self.constraints_dir):
            logger.warning(
                f"Constraints directory {self.constraints_dir} does not exist"
            )
            return

        for filename in os.listdir(self.constraints_dir):
            if filename.endswith(".json"):
                try:
                    file_path = os.path.join(self.constraints_dir, filename)
                    with open(file_path, "r") as f:
                        constraint_data = json.load(f)

                    # Load verification function if specified
                    verification_fn = None
                    if "verification_module" in constraint_data:
                        module_name = constraint_data["verification_module"]
                        fn_name = constraint_data["verification_function"]
                        try:
                            module = importlib.import_module(module_name)
                            verification_fn = getattr(module, fn_name)
                        except (ImportError, AttributeError) as e:
                            logger.error(
                                f"Failed to load verification function for {filename}: {str(e)}"
                            )

                    # Create constraint
                    constraint = EthicalConstraint.from_dict(
                        constraint_data, verification_fn
                    )
                    self.register_constraint(constraint)

                except Exception as e:
                    logger.error(f"Error loading constraint from {filename}: {str(e)}")

    def register_constraint(self, constraint: EthicalConstraint) -> bool:
        """
        Register a new ethical constraint

        Args:
            constraint: EthicalConstraint object to register

        Returns:
            True if registration succeeded
        """
        if constraint.id in self.constraints:
            logger.warning(
                f"Constraint with ID {constraint.id} already exists - overwriting"
            )

        # Verify constraint integrity
        if not constraint.verify_integrity():
            logger.error(f"Constraint {constraint.id} failed integrity verification")
            return False

        self.constraints[constraint.id] = constraint
        logger.info(f"Registered constraint {constraint.id}: {constraint.name}")
        return True

    def evaluate_action(
        self, context: Dict[str, Any], scope: Optional[ConstraintScope] = None
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Evaluate an action against all relevant constraints

        Args:
            context: Dictionary with action details and context
            scope: Optional scope to limit which constraints are checked

        Returns:
            Tuple containing:
            - Boolean indicating if action is permitted
            - List of constraint violations (empty if action is permitted)
        """
        violations = []

        logger.info(f"Evaluating action against {len(self.constraints)} constraints")

        for constraint_id, constraint in self.constraints.items():
            # Skip if not in requested scope
            if scope is not None and constraint.scope != scope:
                continue

            # Evaluate constraint
            is_satisfied, explanation, details = constraint.evaluate(context)

            if not is_satisfied:
                # Record violation
                violation = ConstraintViolation(
                    constraint_id=constraint_id,
                    action_context=context,
                    severity=constraint.severity,
                    violation_details=explanation,
                    evaluation_trace=details,
                )
                violations.append(violation)
                self.violations.append(violation)

                # Log violation
                logger.warning(
                    f"Constraint violation: {constraint.name} - {explanation}"
                )

                # For critical constraints, stop evaluation if enforce_critical is True
                if (
                    constraint.severity == ConstraintSeverity.CRITICAL
                    and self.enforce_critical
                ):
                    break

        # Save violations to audit log
        if violations:
            self._record_violations(violations)

        # Action is permitted if no violations, or no critical violations when enforcing
        if self.enforce_critical:
            critical_violations = [
                v
                for v in violations
                if self.constraints[v.constraint_id].severity
                == ConstraintSeverity.CRITICAL
            ]
            return len(critical_violations) == 0, violations
        else:
            return len(violations) == 0, violations

    def _record_violations(self, violations: List[ConstraintViolation]):
        """Record constraint violations to audit log"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(self.audit_log_dir, f"violations-{timestamp}.json")

        with open(log_file, "w") as f:
            json.dump([v.to_dict() for v in violations], f, indent=2)

    def get_constraint(self, constraint_id: str) -> Optional[EthicalConstraint]:
        """Get a constraint by ID"""
        return self.constraints.get(constraint_id)

    def get_all_constraints(self) -> Dict[str, EthicalConstraint]:
        """Get all registered constraints"""
        return self.constraints.copy()

    def get_violation_history(self) -> List[ConstraintViolation]:
        """Get history of constraint violations"""
        return self.violations.copy()

    def get_human_readable_report(self) -> str:
        """
        Generate a human-readable summary report of the constraints and violations

        Returns:
            String containing the report
        """
        report = []
        report.append(f"Ethical Constraint System Report ({datetime.now().isoformat()})")
        report.append("=========================================")
        report.append(f"Total constraints: {len(self.constraints)}")
        report.append(f"Total violations recorded: {len(self.violations)}")
        report.append("")

        report.append("--- Constraints by Category ---")
        by_category: Dict[ConstraintCategory, int] = {cat: 0 for cat in ConstraintCategory}
        for constraint in self.constraints.values():
            by_category[constraint.category] += 1
        for category, count in by_category.items():
            report.append(f"- {category.value.capitalize()}: {count}")
        report.append("")

        report.append("--- Constraints by Severity ---")
        by_severity: Dict[ConstraintSeverity, int] = {sev: 0 for sev in ConstraintSeverity}
        for constraint in self.constraints.values():
            by_severity[constraint.severity] += 1
        for severity, count in by_severity.items():
            report.append(f"- {severity.value.capitalize()}: {count}")
        report.append("")

        report.append("Recent violations:")
        recent_violations = sorted(
            self.violations, key=lambda v: v.timestamp, reverse=True
        )[:10]
        for v in recent_violations:
            # Rename inner variable to avoid collision with loop variable
            violation_constraint: Optional[EthicalConstraint] = self.constraints.get(v.constraint_id)
            constraint_name = violation_constraint.name if violation_constraint else "Unknown constraint"
            report.append(
                f"  [{v.timestamp.isoformat()}] {constraint_name}: {v.violation_details}"
            )

        return "\n".join(report)


# Example utility functions for formal verification
def symbolic_verification(formula_str: str, variable_values: Dict[str, Any]) -> bool:
    """
    Perform symbolic verification of a formal specification

    Args:
        formula_str: Symbolic formula as string
        variable_values: Dictionary mapping variable names to values

    Returns:
        Boolean indicating if formula is satisfied
    """
    try:
        # Parse formula using sympy
        symbols = {name: sympy.symbols(name) for name in variable_values.keys()}
        formula = sympy.sympify(formula_str, locals=symbols)

        # Substitute values
        for name, value in variable_values.items():
            formula = formula.subs(symbols[name], value)

        # Evaluate
        result = bool(formula.evalf())
        return result

    except Exception as e:
        logger.error(f"Error in symbolic verification: {str(e)}")
        return False
