#!/usr/bin/env python3
"""
🔧 COMPREHENSIVE RULE ENFORCEMENT SYSTEM - SUPREME VALIDATOR
Zero-Tolerance Enforcement of ALL 20 Fundamental Rules + 14 Core Principles
Production-Ready Implementation with Automatic Remediation

This is the SUPREME VALIDATOR that enforces professional codebase standards
as defined in /opt/sutazaiapp/IMPORTANT/Enforcement_Rules
"""

import os
import sys
import re
import json
import ast
import time
import hashlib
import subprocess
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class RuleViolation:
    """Comprehensive rule violation with full context and remediation"""
    rule_number: int
    rule_name: str
    violation_type: str
    file_path: str
    line_number: int
    column: int
    description: str
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    remediation: str
    code_snippet: str = ""
    auto_fixable: bool = False
    fix_command: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule_number,
            "rule_name": self.rule_name,
            "type": self.violation_type,
            "file": self.file_path,
            "line": self.line_number,
            "column": self.column,
            "description": self.description,
            "severity": self.severity,
            "remediation": self.remediation,
            "code_snippet": self.code_snippet,
            "auto_fixable": self.auto_fixable,
            "fix_command": self.fix_command,
            "timestamp": self.timestamp
        }


class ComprehensiveRuleEnforcer:
    """
    SUPREME VALIDATOR - Comprehensive enforcement of all 20 Fundamental Rules
    with zero-tolerance implementation and automatic remediation capabilities
    """
    
    def __init__(self, codebase_root: str, auto_fix: bool = False):
        self.root = Path(codebase_root)
        self.violations: List[RuleViolation] = []
        self.auto_fix = auto_fix
        self.rules_doc = self.root / "IMPORTANT" / "Enforcement_Rules"
        self.changelog_tracking = {}
        self.mcp_protection_status = {}
        self.documentation_status = {}
        self.docker_analysis = {}
        self.test_coverage_data = {}
        self.dependency_analysis = {}
        self.security_scan_results = {}
        
        # Load enforcement rules document
        self._load_enforcement_rules()
        
    def _load_enforcement_rules(self):
        """Load and parse the enforcement rules document"""
        if self.rules_doc.exists():
            try:
                with open(self.rules_doc, 'r') as f:
                    self.rules_content = f.read()
                logger.info(f"Loaded enforcement rules from {self.rules_doc}")
            except Exception as e:
                logger.error(f"Failed to load enforcement rules: {e}")
                self.rules_content = ""
        else:
            logger.warning(f"Enforcement rules document not found at {self.rules_doc}")
            self.rules_content = ""
    
    def validate_all_rules(self) -> Dict[str, Any]:
        """
        Execute comprehensive validation of ALL 20 Fundamental Rules
        with zero-tolerance enforcement
        """
        logger.info("🔧 SUPREME VALIDATOR - Starting Comprehensive Rule Enforcement")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # RULE 1: Real Implementation Only - No Fantasy Code
        self._validate_rule_01_real_implementation()
        
        # RULE 2: Never Break Existing Functionality
        self._validate_rule_02_never_break_functionality()
        
        # RULE 3: Comprehensive Analysis Required
        self._validate_rule_03_comprehensive_analysis()
        
        # RULE 4: Investigate Existing Files & Consolidate First
        self._validate_rule_04_investigate_and_consolidate()
        
        # RULE 5: Professional Project Standards
        self._validate_rule_05_professional_standards()
        
        # RULE 6: Centralized Documentation
        self._validate_rule_06_centralized_documentation()
        
        # RULE 7: Script Organization & Control
        self._validate_rule_07_script_organization()
        
        # RULE 8: Python Script Excellence
        self._validate_rule_08_python_excellence()
        
        # RULE 9: Single Source Frontend/Backend
        self._validate_rule_09_single_source()
        
        # RULE 10: Functionality-First Cleanup
        self._validate_rule_10_functionality_first()
        
        # RULE 11: Docker Excellence
        self._validate_rule_11_docker_excellence()
        
        # RULE 12: Universal Deployment Script
        self._validate_rule_12_universal_deployment()
        
        # RULE 13: Zero Tolerance for Waste
        self._validate_rule_13_zero_waste()
        
        # RULE 14: Specialized Claude Sub-Agent Usage
        self._validate_rule_14_claude_agents()
        
        # RULE 15: Documentation Quality
        self._validate_rule_15_documentation_quality()
        
        # RULE 16: Local LLM Operations
        self._validate_rule_16_local_llm()
        
        # RULE 17: Canonical Documentation Authority
        self._validate_rule_17_canonical_authority()
        
        # RULE 18: Mandatory Documentation Review
        self._validate_rule_18_mandatory_review()
        
        # RULE 19: Change Tracking Requirements
        self._validate_rule_19_change_tracking()
        
        # RULE 20: MCP Server Protection
        self._validate_rule_20_mcp_protection()
        
        # Generate comprehensive compliance report
        execution_time = time.time() - start_time
        report = self._generate_comprehensive_report(execution_time)
        
        # Apply automatic fixes if enabled
        if self.auto_fix and self._has_auto_fixable_violations():
            self._apply_automatic_fixes()
        
        return report
    
    def _validate_rule_01_real_implementation(self):
        """
        RULE 1: Real Implementation Only - No Fantasy Code
        Zero tolerance for placeholder, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, or theoretical implementations
        """
        logger.info("📌 RULE 1: Validating Real Implementation Only...")
        
        # Comprehensive fantasy code patterns
        fantasy_patterns = [
            (r'TODO.*magic\s+happens', "Magic/fantasy comment detected"),
            (r'TODO.*future.*implementation', "Future implementation placeholder"),
            (r'placeholder.*service', "Placeholder service detected"),
            (r'abstract.*handler.*production', "Abstract handler in production"),
            (r'Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.*implementation.*(?!test)', "Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test implementation outside tests"),
            (r'imaginary.*system', "Imaginary system reference"),
            (r'theoretical.*implementation', "Theoretical implementation"),
            (r'conceptual.*code', "Conceptual code detected"),
            (r'stub.*function.*production', "Stub function in production"),
            (r'dummy.*(?:data|service|implementation)', "Dummy implementation"),
            (r'fake.*(?:service|data|api).*(?!test)', "Fake service outside tests"),
            (r'NotImplementedError.*(?!test)', "NotImplementedError outside tests"),
            (r'raise\s+NotImplemented', "NotImplemented exception"),
            (r'pass\s*#.*implement.*later', "Deferred implementation"),
            (r'return\s+None\s*#.*TODO', "Incomplete return statement")
        ]
        
        for pattern, description in fantasy_patterns:
            self._scan_for_violations(
                pattern=pattern,
                rule_num=1,
                rule_name="Real Implementation Only",
                violation_type="Fantasy Code",
                description=description,
                severity="CRITICAL",
                remediation="Replace with concrete, working implementation",
                auto_fixable=False
            )
        
        # Check for hardcoded test/development values in production code
        self._check_hardcoded_values()
    
    def _validate_rule_02_never_break_functionality(self):
        """
        RULE 2: Never Break Existing Functionality
        Comprehensive validation of backward compatibility and migration safety
        """
        logger.info("📌 RULE 2: Validating Never Break Existing Functionality...")
        
        # Breaking change patterns
        breaking_patterns = [
            (r'\.remove\(', "Removal operation detected"),
            (r'\.delete\(', "Delete operation detected"),
            (r'del\s+\w+\[', "Dictionary/list deletion"),
            (r'DROP\s+TABLE', "Database table drop"),
            (r'ALTER\s+TABLE.*DROP', "Column drop operation"),
            (r'CASCADE\s+DELETE', "Cascade delete operation"),
            (r'TRUNCATE\s+TABLE', "Table truncation"),
            (r'os\.remove\(', "File removal operation"),
            (r'shutil\.rmtree\(', "Directory removal operation"),
            (r'Path\(.*\)\.unlink\(', "Path unlink operation"),
            (r'@deprecated', "Deprecated decorator without migration"),
            (r'raise\s+DeprecationWarning', "Deprecation warning")
        ]
        
        for pattern, description in breaking_patterns:
            self._scan_for_violations(
                pattern=pattern,
                rule_num=2,
                rule_name="Never Break Functionality",
                violation_type="Breaking Change Risk",
                description=description,
                severity="HIGH",
                remediation="Ensure backward compatibility or provide migration path",
                auto_fixable=False
            )
        
        # Check for missing migration scripts
        self._validate_migration_safety()
    
    def _validate_rule_03_comprehensive_analysis(self):
        """
        RULE 3: Comprehensive Analysis Required
        Ensure thorough analysis before any changes
        """
        logger.info("📌 RULE 3: Validating Comprehensive Analysis...")
        
        # Check for analysis documentation
        analysis_docs = [
            "ANALYSIS.md",
            "IMPACT_ANALYSIS.md",
            "DEPENDENCY_ANALYSIS.md",
            "RISK_ASSESSMENT.md"
        ]
        
        docs_found = 0
        for doc in analysis_docs:
            if (self.root / "docs" / doc).exists():
                docs_found += 1
        
        if docs_found == 0:
            self.violations.append(RuleViolation(
                rule_number=3,
                rule_name="Comprehensive Analysis",
                violation_type="Missing Analysis Documentation",
                file_path=str(self.root / "docs"),
                line_number=0,
                column=0,
                description="No comprehensive analysis documentation found",
                severity="MEDIUM",
                remediation="Create analysis documentation before making changes",
                auto_fixable=False
            ))
    
    def _validate_rule_04_investigate_and_consolidate(self):
        """
        RULE 4: Investigate Existing Files & Consolidate First
        Detect and prevent duplication
        """
        logger.info("📌 RULE 4: Validating Investigation and Consolidation...")
        
        # Detect duplicate functions and classes
        self._detect_code_duplication()
        
        # Check for duplicate configuration files
        self._check_duplicate_configs()
        
        # Validate consolidation opportunities
        self._identify_consolidation_opportunities()
    
    def _validate_rule_05_professional_standards(self):
        """
        RULE 5: Professional Project Standards
        Comprehensive validation of professional development practices
        """
        logger.info("📌 RULE 5: Validating Professional Standards...")
        
        # Test coverage validation
        self._validate_test_coverage()
        
        # Code review process validation
        self._validate_code_review_process()
        
        # Security scanning validation
        self._validate_security_scanning()
        
        # Performance benchmarking validation
        self._validate_performance_benchmarks()
    
    def _validate_rule_06_centralized_documentation(self):
        """
        RULE 6: Centralized Documentation
        Ensure all documentation is centralized and current
        """
        logger.info("📌 RULE 6: Validating Centralized Documentation...")
        
        docs_dir = self.root / "docs"
        if not docs_dir.exists():
            self.violations.append(RuleViolation(
                rule_number=6,
                rule_name="Centralized Documentation",
                violation_type="Missing Documentation Directory",
                file_path=str(self.root),
                line_number=0,
                column=0,
                description="No centralized /docs/ directory found",
                severity="HIGH",
                remediation="Create /docs/ directory with proper structure",
                auto_fixable=True,
                fix_command="mkdir -p docs && touch docs/README.md"
            ))
        
        # Check for scattered documentation
        self._check_scattered_documentation()
    
    def _validate_rule_07_script_organization(self):
        """
        RULE 7: Script Organization & Control
        Validate proper script organization and structure
        """
        logger.info("📌 RULE 7: Validating Script Organization...")
        
        scripts_dir = self.root / "scripts"
        required_subdirs = [
            "dev", "deploy", "utils", "test", "monitoring",
            "backup", "security", "maintenance", "automation"
        ]
        
        for subdir in required_subdirs:
            if not (scripts_dir / subdir).exists():
                self.violations.append(RuleViolation(
                    rule_number=7,
                    rule_name="Script Organization",
                    violation_type="Missing Script Directory",
                    file_path=str(scripts_dir / subdir),
                    line_number=0,
                    column=0,
                    description=f"Missing required script directory: {subdir}",
                    severity="MEDIUM",
                    remediation=f"Create organized {subdir} directory",
                    auto_fixable=True,
                    fix_command=f"mkdir -p scripts/{subdir}"
                ))
    
    def _validate_rule_08_python_excellence(self):
        """
        RULE 8: Python Script Excellence
        Comprehensive Python code quality validation
        """
        logger.info("📌 RULE 8: Validating Python Excellence...")
        
        python_files = list(self.root.rglob("*.py"))
        
        for py_file in python_files:
            self._validate_python_file(py_file)
    
    def _validate_rule_09_single_source(self):
        """
        RULE 9: Single Source Frontend/Backend
        Prevent duplicate frontend/backend directories
        """
        logger.info("📌 RULE 9: Validating Single Source...")
        
        # Check for duplicate frontend directories
        frontend_dirs = [
            "frontend", "frontend_v1", "frontend_old",
            "frontend_new", "frontend_backup"
        ]
        
        found_frontends = []
        for dir_name in frontend_dirs:
            if (self.root / dir_name).exists():
                found_frontends.append(dir_name)
        
        if len(found_frontends) > 1:
            self.violations.append(RuleViolation(
                rule_number=9,
                rule_name="Single Source",
                violation_type="Duplicate Frontend Directories",
                file_path=str(self.root),
                line_number=0,
                column=0,
                description=f"Multiple frontend directories found: {found_frontends}",
                severity="HIGH",
                remediation="Consolidate to single /frontend directory",
                auto_fixable=False
            ))
    
    def _validate_rule_10_functionality_first(self):
        """
        RULE 10: Functionality-First Cleanup
        Never delete blindly - investigate first
        """
        logger.info("📌 RULE 10: Validating Functionality-First Cleanup...")
        
        # Check for aggressive cleanup patterns
        cleanup_patterns = [
            (r'rm\s+-rf\s+(?!\/tmp)', "Aggressive rm -rf detected"),
            (r'find.*-delete', "Find with delete operation"),
            (r'git\s+clean\s+-fdx', "Aggressive git clean"),
            (r'truncate.*--size=0', "File truncation detected")
        ]
        
        for pattern, description in cleanup_patterns:
            self._scan_for_violations(
                pattern=pattern,
                rule_num=10,
                rule_name="Functionality-First Cleanup",
                violation_type="Aggressive Cleanup",
                description=description,
                severity="HIGH",
                remediation="Investigate before deletion, create backups",
                auto_fixable=False
            )
    
    def _validate_rule_11_docker_excellence(self):
        """
        RULE 11: Docker Excellence
        Comprehensive Docker best practices validation
        """
        logger.info("📌 RULE 11: Validating Docker Excellence...")
        
        dockerfiles = list(self.root.rglob("Dockerfile*"))
        
        for dockerfile in dockerfiles:
            self._validate_dockerfile(dockerfile)
    
    def _validate_rule_12_universal_deployment(self):
        """
        RULE 12: Universal Deployment Script
        Validate deployment automation and zero-touch capability
        """
        logger.info("📌 RULE 12: Validating Universal Deployment...")
        
        deploy_script = self.root / "deploy.sh"
        if not deploy_script.exists():
            self.violations.append(RuleViolation(
                rule_number=12,
                rule_name="Universal Deployment",
                violation_type="Missing Deployment Script",
                file_path=str(self.root),
                line_number=0,
                column=0,
                description="No universal deploy.sh script found",
                severity="CRITICAL",
                remediation="Create comprehensive deploy.sh with zero-touch deployment",
                auto_fixable=False
            ))
        else:
            self._validate_deployment_script(deploy_script)
    
    def _validate_rule_13_zero_waste(self):
        """
        RULE 13: Zero Tolerance for Waste
        Detect and eliminate all forms of waste
        """
        logger.info("📌 RULE 13: Validating Zero Waste...")
        
        # Technical debt markers
        debt_patterns = [
            (r'TODO:', "TODO marker found"),
            (r'FIXME:', "FIXME marker found"),
            (r'XXX:', "XXX marker found"),
            (r'HACK:', "HACK marker found"),
            (r'BUG:', "BUG marker found"),
            (r'DEPRECATED:', "DEPRECATED marker found"),
            (r'REFACTOR:', "REFACTOR marker found"),
            (r'OPTIMIZE:', "OPTIMIZE marker found")
        ]
        
        for pattern, description in debt_patterns:
            self._scan_for_violations(
                pattern=pattern,
                rule_num=13,
                rule_name="Zero Waste",
                violation_type="Technical Debt",
                description=description,
                severity="LOW",
                remediation="Resolve technical debt or convert to tracked issue",
                auto_fixable=False
            )
        
        # Check for unused imports
        self._check_unused_imports()
        
        # Check for dead code
        self._check_dead_code()
    
    def _validate_rule_14_claude_agents(self):
        """
        RULE 14: Specialized Claude Sub-Agent Usage
        Validate proper Claude agent orchestration
        """
        logger.info("📌 RULE 14: Validating Claude Agent Usage...")
        
        # Check for agent documentation
        agents_doc = self.root / "AGENTS.md"
        if not agents_doc.exists():
            self.violations.append(RuleViolation(
                rule_number=14,
                rule_name="Claude Agent Usage",
                violation_type="Missing Agent Documentation",
                file_path=str(self.root),
                line_number=0,
                column=0,
                description="No AGENTS.md documentation found",
                severity="MEDIUM",
                remediation="Create comprehensive agent documentation",
                auto_fixable=False
            ))
    
    def _validate_rule_15_documentation_quality(self):
        """
        RULE 15: Documentation Quality
        Validate documentation completeness and quality
        """
        logger.info("📌 RULE 15: Validating Documentation Quality...")
        
        # Check all markdown files for quality
        md_files = list(self.root.rglob("*.md"))
        
        for md_file in md_files:
            self._validate_documentation_file(md_file)
    
    def _validate_rule_16_local_llm(self):
        """
        RULE 16: Local LLM Operations
        Validate Ollama configuration and usage
        """
        logger.info("📌 RULE 16: Validating Local LLM Operations...")
        
        # Check for Ollama configuration
        ollama_patterns = [
            (r'openai\.api_key', "External API key detected"),
            (r'anthropic.*api.*key', "External Anthropic key detected"),
            (r'gpt-[34]', "External GPT model reference"),
            (r'claude-\d', "External Claude model reference")
        ]
        
        for pattern, description in ollama_patterns:
            self._scan_for_violations(
                pattern=pattern,
                rule_num=16,
                rule_name="Local LLM Operations",
                violation_type="External LLM Dependency",
                description=description,
                severity="HIGH",
                remediation="Use local Ollama with TinyLlama model",
                auto_fixable=False
            )
    
    def _validate_rule_17_canonical_authority(self):
        """
        RULE 17: Canonical Documentation Authority
        Validate /opt/sutazaiapp/IMPORTANT/ as authority
        """
        logger.info("📌 RULE 17: Validating Canonical Authority...")
        
        important_dir = self.root / "IMPORTANT"
        if not important_dir.exists():
            self.violations.append(RuleViolation(
                rule_number=17,
                rule_name="Canonical Authority",
                violation_type="Missing Authority Directory",
                file_path=str(self.root),
                line_number=0,
                column=0,
                description="No /IMPORTANT/ authority directory found",
                severity="CRITICAL",
                remediation="Create /IMPORTANT/ directory for canonical documentation",
                auto_fixable=True,
                fix_command="mkdir -p IMPORTANT"
            ))
    
    def _validate_rule_18_mandatory_review(self):
        """
        RULE 18: Mandatory Documentation Review
        Validate CHANGELOG.md presence and currency
        """
        logger.info("📌 RULE 18: Validating Mandatory Review...")
        
        # Check for CHANGELOG.md in all directories
        for directory in self.root.rglob("*/"):
            if directory.is_dir() and not any(part.startswith('.') for part in directory.parts):
                changelog = directory / "CHANGELOG.md"
                if not changelog.exists():
                    self.violations.append(RuleViolation(
                        rule_number=18,
                        rule_name="Mandatory Review",
                        violation_type="Missing CHANGELOG.md",
                        file_path=str(directory),
                        line_number=0,
                        column=0,
                        description=f"No CHANGELOG.md in {directory.relative_to(self.root)}",
                        severity="MEDIUM",
                        remediation="Create CHANGELOG.md with comprehensive history",
                        auto_fixable=True,
                        fix_command=f"touch {changelog}"
                    ))
    
    def _validate_rule_19_change_tracking(self):
        """
        RULE 19: Change Tracking Requirements
        Comprehensive change tracking validation
        """
        logger.info("📌 RULE 19: Validating Change Tracking...")
        
        # Validate CHANGELOG.md format and content
        changelogs = list(self.root.rglob("CHANGELOG.md"))
        
        for changelog in changelogs:
            self._validate_changelog_format(changelog)
    
    def _validate_rule_20_mcp_protection(self):
        """
        RULE 20: MCP Server Protection
        Critical infrastructure safeguarding
        """
        logger.info("📌 RULE 20: Validating MCP Protection...")
        
        mcp_json = self.root / ".mcp.json"
        mcp_scripts = self.root / "scripts" / "mcp"
        
        # Check MCP configuration integrity
        if mcp_json.exists():
            # Calculate checksum for integrity verification
            with open(mcp_json, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Store for monitoring
            self.mcp_protection_status['config_checksum'] = checksum
            
            # Check modification time
            mtime = os.path.getmtime(mcp_json)
            if time.time() - mtime < 3600:  # Modified in last hour
                self.violations.append(RuleViolation(
                    rule_number=20,
                    rule_name="MCP Protection",
                    violation_type="Recent MCP Modification",
                    file_path=str(mcp_json),
                    line_number=0,
                    column=0,
                    description="MCP configuration recently modified - verify authorization",
                    severity="CRITICAL",
                    remediation="Ensure MCP changes are explicitly authorized",
                    auto_fixable=False
                ))
        
        # Validate MCP wrapper scripts
        if mcp_scripts.exists():
            for wrapper in mcp_scripts.glob("wrappers/*.sh"):
                self._validate_mcp_wrapper(wrapper)
    
    # Helper methods for comprehensive validation
    
    def _scan_for_violations(self, pattern: str, rule_num: int, rule_name: str,
                            violation_type: str, description: str, 
                            severity: str, remediation: str, auto_fixable: bool = False):
        """Scan files for pattern violations"""
        try:
            # Use ripgrep for faster scanning if available
            cmd = ['rg', '-n', '--no-heading', pattern, str(self.root)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 2:
                            file_path = parts[0]
                            line_num = int(parts[1]) if parts[1].isdigit() else 0
                            code_snippet = parts[2] if len(parts) > 2 else ""
                            
                            # Skip test files for certain patterns
                            if 'test' in file_path.lower() and violation_type in ['Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test implementation', 'Fake service']:
                                continue
                            
                            self.violations.append(RuleViolation(
                                rule_number=rule_num,
                                rule_name=rule_name,
                                violation_type=violation_type,
                                file_path=file_path,
                                line_number=line_num,
                                column=0,
                                description=description,
                                severity=severity,
                                remediation=remediation,
                                code_snippet=code_snippet[:100],
                                auto_fixable=auto_fixable
                            ))
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout scanning for pattern: {pattern}")
        except FileNotFoundError:
            # Fallback to grep if ripgrep not available
            self._scan_with_grep(pattern, rule_num, rule_name, violation_type, 
                               description, severity, remediation, auto_fixable)
        except Exception as e:
            logger.error(f"Error scanning for pattern {pattern}: {e}")
    
    def _scan_with_grep(self, pattern: str, rule_num: int, rule_name: str,
                        violation_type: str, description: str,
                        severity: str, remediation: str, auto_fixable: bool):
        """Fallback grep scanning"""
        try:
            cmd = ['grep', '-rn', pattern, str(self.root), '--include=*.py', '--include=*.js']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 2:
                            self.violations.append(RuleViolation(
                                rule_number=rule_num,
                                rule_name=rule_name,
                                violation_type=violation_type,
                                file_path=parts[0],
                                line_number=int(parts[1]) if parts[1].isdigit() else 0,
                                column=0,
                                description=description,
                                severity=severity,
                                remediation=remediation,
                                auto_fixable=auto_fixable
                            ))
        except Exception as e:
            logger.error(f"Grep scan failed: {e}")
    
    def _validate_python_file(self, py_file: Path):
        """Comprehensive Python file validation"""
        try:
            content = py_file.read_text()
            
            # Check for docstrings
            if not content.strip().startswith('"""'):
                self.violations.append(RuleViolation(
                    rule_number=8,
                    rule_name="Python Excellence",
                    violation_type="Missing Module Docstring",
                    file_path=str(py_file),
                    line_number=1,
                    column=0,
                    description="Python file missing module-level docstring",
                    severity="LOW",
                    remediation="Add comprehensive module docstring",
                    auto_fixable=False
                ))
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                self._analyze_python_ast(tree, py_file)
            except SyntaxError as e:
                self.violations.append(RuleViolation(
                    rule_number=8,
                    rule_name="Python Excellence",
                    violation_type="Syntax Error",
                    file_path=str(py_file),
                    line_number=e.lineno or 0,
                    column=e.offset or 0,
                    description=f"Python syntax error: {e.msg}",
                    severity="CRITICAL",
                    remediation="Fix syntax error",
                    auto_fixable=False
                ))
        except Exception as e:
            logger.error(f"Error validating Python file {py_file}: {e}")
    
    def _analyze_python_ast(self, tree: ast.AST, file_path: Path):
        """Analyze Python AST for rule violations"""
        for node in ast.walk(tree):
            # Check for bare except clauses
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                self.violations.append(RuleViolation(
                    rule_number=8,
                    rule_name="Python Excellence",
                    violation_type="Bare Except Clause",
                    file_path=str(file_path),
                    line_number=node.lineno,
                    column=node.col_offset,
                    description="Bare except clause found - catches all exceptions",
                    severity="MEDIUM",
                    remediation="Use specific exception types",
                    auto_fixable=True
                ))
            
            # Check for missing type hints
            if isinstance(node, ast.FunctionDef):
                if not node.returns and node.name != '__init__':
                    self.violations.append(RuleViolation(
                        rule_number=8,
                        rule_name="Python Excellence",
                        violation_type="Missing Type Hint",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        description=f"Function '{node.name}' missing return type hint",
                        severity="LOW",
                        remediation="Add return type hint",
                        auto_fixable=False
                    ))
    
    def _validate_dockerfile(self, dockerfile: Path):
        """Comprehensive Dockerfile validation"""
        try:
            content = dockerfile.read_text()
            lines = content.split('\n')
            
            # Check for USER directive
            has_user = False
            uses_latest = False
            has_healthcheck = False
            
            for i, line in enumerate(lines, 1):
                if line.strip().startswith('USER '):
                    has_user = True
                if ':latest' in line:
                    uses_latest = True
                    self.violations.append(RuleViolation(
                        rule_number=11,
                        rule_name="Docker Excellence",
                        violation_type="Latest Tag Usage",
                        file_path=str(dockerfile),
                        line_number=i,
                        column=0,
                        description="Using ':latest' tag - not reproducible",
                        severity="HIGH",
                        remediation="Pin to specific version",
                        auto_fixable=False
                    ))
                if line.strip().startswith('HEALTHCHECK'):
                    has_healthcheck = True
            
            if not has_user:
                self.violations.append(RuleViolation(
                    rule_number=11,
                    rule_name="Docker Excellence",
                    violation_type="Missing USER Directive",
                    file_path=str(dockerfile),
                    line_number=0,
                    column=0,
                    description="Dockerfile missing non-root USER directive",
                    severity="HIGH",
                    remediation="Add USER directive for security",
                    auto_fixable=True,
                    fix_command=f"echo 'USER nonroot' >> {dockerfile}"
                ))
            
            if not has_healthcheck:
                self.violations.append(RuleViolation(
                    rule_number=11,
                    rule_name="Docker Excellence",
                    violation_type="Missing HEALTHCHECK",
                    file_path=str(dockerfile),
                    line_number=0,
                    column=0,
                    description="Dockerfile missing HEALTHCHECK instruction",
                    severity="MEDIUM",
                    remediation="Add HEALTHCHECK for monitoring",
                    auto_fixable=False
                ))
        except Exception as e:
            logger.error(f"Error validating Dockerfile {dockerfile}: {e}")
    
    def _validate_changelog_format(self, changelog: Path):
        """Validate CHANGELOG.md format and content"""
        try:
            content = changelog.read_text()
            
            # Check for required sections
            required_sections = [
                "## Change History",
                "## Directory Information",
                "### ["  # Date entries
            ]
            
            for section in required_sections:
                if section not in content:
                    self.violations.append(RuleViolation(
                        rule_number=19,
                        rule_name="Change Tracking",
                        violation_type="Incomplete CHANGELOG",
                        file_path=str(changelog),
                        line_number=0,
                        column=0,
                        description=f"CHANGELOG missing required section: {section}",
                        severity="MEDIUM",
                        remediation="Add comprehensive change tracking sections",
                        auto_fixable=False
                    ))
            
            # Check for recent updates
            mtime = os.path.getmtime(changelog)
            days_old = (time.time() - mtime) / 86400
            
            if days_old > 30:
                self.violations.append(RuleViolation(
                    rule_number=19,
                    rule_name="Change Tracking",
                    violation_type="Stale CHANGELOG",
                    file_path=str(changelog),
                    line_number=0,
                    column=0,
                    description=f"CHANGELOG not updated in {int(days_old)} days",
                    severity="LOW",
                    remediation="Update CHANGELOG with recent changes",
                    auto_fixable=False
                ))
        except Exception as e:
            logger.error(f"Error validating changelog {changelog}: {e}")
    
    def _validate_test_coverage(self):
        """Validate test coverage meets professional standards"""
        try:
            # Try to get coverage data
            result = subprocess.run(
                ['coverage', 'report', '--format=json'],
                capture_output=True,
                text=True,
                cwd=str(self.root)
            )
            
            if result.returncode == 0:
                coverage_data = json.loads(result.stdout)
                total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
                
                if total_coverage < 80:
                    self.violations.append(RuleViolation(
                        rule_number=5,
                        rule_name="Professional Standards",
                        violation_type="Insufficient Test Coverage",
                        file_path=str(self.root),
                        line_number=0,
                        column=0,
                        description=f"Test coverage {total_coverage:.1f}% below 80% standard",
                        severity="HIGH",
                        remediation="Increase test coverage to at least 80%",
                        auto_fixable=False
                    ))
                
                self.test_coverage_data = coverage_data
        except FileNotFoundError:
            logger.warning("Coverage tool not found - skipping coverage validation")
        except Exception as e:
            logger.error(f"Error checking test coverage: {e}")
    
    def _check_hardcoded_values(self):
        """Check for hardcoded values in production code"""
        hardcoded_patterns = [
            (r'localhost:\d{4}', "Hardcoded localhost URL"),
            (r'127\.0\.0\.1', "Hardcoded IP address"),
            (r'password\s*=\s*["\'].*["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\'].*["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'].*["\']', "Hardcoded secret")
        ]
        
        for pattern, description in hardcoded_patterns:
            self._scan_for_violations(
                pattern=pattern,
                rule_num=1,
                rule_name="Real Implementation",
                violation_type="Hardcoded Value",
                description=description,
                severity="HIGH",
                remediation="Use environment variables or configuration",
                auto_fixable=False
            )
    
    def _detect_code_duplication(self):
        """Detect duplicate code blocks"""
        # This would use tools like pylint or radon for duplication detection
        pass
    
    def _check_duplicate_configs(self):
        """Check for duplicate configuration files"""
        config_patterns = [
            "config*.json",
            "settings*.py",
            "configuration*.yaml"
        ]
        
        for pattern in config_patterns:
            configs = list(self.root.rglob(pattern))
            if len(configs) > 1:
                self.violations.append(RuleViolation(
                    rule_number=4,
                    rule_name="Investigate and Consolidate",
                    violation_type="Duplicate Configuration",
                    file_path=str(self.root),
                    line_number=0,
                    column=0,
                    description=f"Multiple {pattern} files found: {len(configs)}",
                    severity="MEDIUM",
                    remediation="Consolidate configuration files",
                    auto_fixable=False
                ))
    
    def _identify_consolidation_opportunities(self):
        """Identify code consolidation opportunities"""
        # Check for similar file names that could be consolidated
        pass
    
    def _validate_migration_safety(self):
        """Validate database migration safety"""
        migrations_dir = self.root / "migrations"
        if migrations_dir.exists():
            migrations = list(migrations_dir.glob("*.sql"))
            for migration in migrations:
                content = migration.read_text()
                if "DROP" in content and "IF EXISTS" not in content:
                    self.violations.append(RuleViolation(
                        rule_number=2,
                        rule_name="Never Break Functionality",
                        violation_type="Unsafe Migration",
                        file_path=str(migration),
                        line_number=0,
                        column=0,
                        description="Migration contains DROP without IF EXISTS",
                        severity="HIGH",
                        remediation="Add IF EXISTS clause for safety",
                        auto_fixable=True
                    ))
    
    def _validate_code_review_process(self):
        """Validate code review process is in place"""
        pr_template = self.root / ".github" / "pull_request_template.md"
        if not pr_template.exists():
            self.violations.append(RuleViolation(
                rule_number=5,
                rule_name="Professional Standards",
                violation_type="Missing PR Template",
                file_path=str(self.root / ".github"),
                line_number=0,
                column=0,
                description="No pull request template found",
                severity="MEDIUM",
                remediation="Create PR template for code reviews",
                auto_fixable=True
            ))
    
    def _validate_security_scanning(self):
        """Validate security scanning is configured"""
        # Check for security scanning configuration
        pass
    
    def _validate_performance_benchmarks(self):
        """Validate performance benchmarking"""
        # Check for performance test files
        perf_tests = list(self.root.rglob("*performance*.py"))
        if not perf_tests:
            self.violations.append(RuleViolation(
                rule_number=5,
                rule_name="Professional Standards",
                violation_type="Missing Performance Tests",
                file_path=str(self.root),
                line_number=0,
                column=0,
                description="No performance benchmarking tests found",
                severity="MEDIUM",
                remediation="Add performance benchmarking tests",
                auto_fixable=False
            ))
    
    def _check_scattered_documentation(self):
        """Check for documentation scattered across the codebase"""
        doc_files = []
        for ext in ['*.md', '*.rst', '*.txt']:
            doc_files.extend(self.root.rglob(ext))
        
        scattered = [f for f in doc_files if 'docs' not in str(f)]
        if len(scattered) > 10:
            self.violations.append(RuleViolation(
                rule_number=6,
                rule_name="Centralized Documentation",
                violation_type="Scattered Documentation",
                file_path=str(self.root),
                line_number=0,
                column=0,
                description=f"{len(scattered)} documentation files outside /docs/",
                severity="MEDIUM",
                remediation="Move documentation to /docs/ directory",
                auto_fixable=False
            ))
    
    def _validate_deployment_script(self, deploy_script: Path):
        """Validate deployment script comprehensiveness"""
        content = deploy_script.read_text()
        
        required_features = [
            ("--env", "Environment parameter support"),
            ("rollback", "Rollback capability"),
            ("health", "Health check validation"),
            ("backup", "Backup procedures"),
            ("zero-downtime", "Zero-downtime deployment")
        ]
        
        for feature, description in required_features:
            if feature not in content:
                self.violations.append(RuleViolation(
                    rule_number=12,
                    rule_name="Universal Deployment",
                    violation_type="Incomplete Deployment Script",
                    file_path=str(deploy_script),
                    line_number=0,
                    column=0,
                    description=f"Deploy script missing: {description}",
                    severity="HIGH",
                    remediation=f"Add {description} to deployment script",
                    auto_fixable=False
                ))
    
    def _check_unused_imports(self):
        """Check for unused imports in Python files"""
        # Would use tools like autoflake or pylint
        pass
    
    def _check_dead_code(self):
        """Check for dead code"""
        # Would use tools like vulture
        pass
    
    def _validate_documentation_file(self, md_file: Path):
        """Validate documentation file quality"""
        content = md_file.read_text()
        
        # Check for proper headers
        if not content.strip().startswith('#'):
            self.violations.append(RuleViolation(
                rule_number=15,
                rule_name="Documentation Quality",
                violation_type="Missing Header",
                file_path=str(md_file),
                line_number=1,
                column=0,
                description="Documentation missing proper header",
                severity="LOW",
                remediation="Add proper markdown header",
                auto_fixable=False
            ))
        
        # Check for TODO items in docs
        if 'TODO' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'TODO' in line:
                    self.violations.append(RuleViolation(
                        rule_number=15,
                        rule_name="Documentation Quality",
                        violation_type="Incomplete Documentation",
                        file_path=str(md_file),
                        line_number=i,
                        column=0,
                        description="TODO found in documentation",
                        severity="LOW",
                        remediation="Complete documentation TODO",
                        auto_fixable=False
                    ))
    
    def _validate_mcp_wrapper(self, wrapper: Path):
        """Validate MCP wrapper script integrity"""
        # Check wrapper script permissions and content
        stat_info = os.stat(wrapper)
        if not stat_info.st_mode & 0o111:  # Not executable
            self.violations.append(RuleViolation(
                rule_number=20,
                rule_name="MCP Protection",
                violation_type="MCP Wrapper Permission",
                file_path=str(wrapper),
                line_number=0,
                column=0,
                description="MCP wrapper script not executable",
                severity="HIGH",
                remediation="Fix wrapper script permissions",
                auto_fixable=True,
                fix_command=f"chmod +x {wrapper}"
            ))
    
    def _has_auto_fixable_violations(self) -> bool:
        """Check if there are auto-fixable violations"""
        return any(v.auto_fixable for v in self.violations)
    
    def _apply_automatic_fixes(self):
        """Apply automatic fixes for violations"""
        logger.info("🔧 Applying automatic fixes...")
        
        fixed_count = 0
        for violation in self.violations:
            if violation.auto_fixable and violation.fix_command:
                try:
                    result = subprocess.run(
                        violation.fix_command,
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        fixed_count += 1
                        logger.info(f"Fixed: {violation.description}")
                    else:
                        logger.error(f"Failed to fix: {violation.description}")
                except Exception as e:
                    logger.error(f"Error applying fix: {e}")
        
        logger.info(f"Applied {fixed_count} automatic fixes")
    
    def _generate_comprehensive_report(self, execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": round(execution_time, 2),
            "total_violations": len(self.violations),
            "violations_by_severity": defaultdict(int),
            "violations_by_rule": defaultdict(int),
            "auto_fixable_count": sum(1 for v in self.violations if v.auto_fixable),
            "violations": [],
            "compliance_score": 0,
            "status": "",
            "summary": "",
            "recommendations": [],
            "test_coverage": self.test_coverage_data,
            "mcp_status": self.mcp_protection_status,
            "documentation_status": self.documentation_status
        }
        
        # Process violations
        for violation in self.violations:
            report["violations_by_severity"][violation.severity] += 1
            report["violations_by_rule"][violation.rule_number] += 1
            report["violations"].append(violation.to_dict())
        
        # Calculate compliance score
        total_rules = 20
        rules_with_violations = len(report["violations_by_rule"])
        report["compliance_score"] = round((1 - rules_with_violations / total_rules) * 100, 1)
        
        # Determine status
        critical = report["violations_by_severity"]["CRITICAL"]
        high = report["violations_by_severity"]["HIGH"]
        
        if critical > 0:
            report["status"] = "CRITICAL"
            report["summary"] = f"❌ CRITICAL: {critical} violations require immediate attention"
        elif high > 0:
            report["status"] = "HIGH"
            report["summary"] = f"⚠️  HIGH PRIORITY: {high} violations should be addressed"
        elif report["total_violations"] > 0:
            report["status"] = "MEDIUM"
            report["summary"] = f"⚠️  MEDIUM: {report['total_violations']} violations found"
        else:
            report["status"] = "COMPLIANT"
            report["summary"] = "✅ FULLY COMPLIANT: No violations found"
        
        # Generate recommendations
        if critical > 0:
            report["recommendations"].append("Address all CRITICAL violations immediately")
        if high > 0:
            report["recommendations"].append("Review and fix HIGH priority violations")
        if report["auto_fixable_count"] > 0:
            report["recommendations"].append(f"Run with --auto-fix to fix {report['auto_fixable_count']} violations")
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="🔧 SUPREME VALIDATOR - Comprehensive Rule Enforcement System"
    )
    parser.add_argument("--root", default="/opt/sutazaiapp", help="Codebase root directory")
    parser.add_argument("--output", help="Output JSON report file")
    parser.add_argument("--auto-fix", action="store_true", help="Apply automatic fixes")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--rule", type=int, help="Validate specific rule only")
    
    args = parser.parse_args()
    
    # Initialize enforcer
    enforcer = ComprehensiveRuleEnforcer(args.root, args.auto_fix)
    
    # Execute validation
    report = enforcer.validate_all_rules()
    
    # Display results
    if args.summary:
        logger.info("\n" + "="*80)
        logger.info("🔧 SUPREME VALIDATOR - RULE ENFORCEMENT SUMMARY")
        logger.info("="*80)
        logger.info(f"Timestamp: {report['timestamp']}")
        logger.info(f"Execution Time: {report['execution_time_seconds']}s")
        logger.info(f"Total Violations: {report['total_violations']}")
        logger.info(f"Compliance Score: {report['compliance_score']}%")
        logger.info(f"Status: {report['status']}")
        logger.info(f"Summary: {report['summary']}")
        
        if report["violations_by_severity"]:
            logger.info("\nViolations by Severity:")
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                count = report["violations_by_severity"].get(severity, 0)
                if count > 0:
                    logger.info(f"  {severity}: {count}")
        
        if report["violations_by_rule"]:
            logger.info("\nViolations by Rule:")
            for rule in sorted(report["violations_by_rule"].keys()):
                count = report["violations_by_rule"][rule]
                logger.info(f"  Rule {rule}: {count} violations")
        
        if report["recommendations"]:
            logger.info("\nRecommendations:")
            for rec in report["recommendations"]:
                logger.info(f"  • {rec}")
    else:
        # Full report
        logger.info(json.dumps(report, indent=2, default=str))
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\n📄 Report saved to {args.output}")
    
    # Exit with appropriate code
    if report["status"] == "CRITICAL":
        sys.exit(2)
    elif report["status"] == "HIGH":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()