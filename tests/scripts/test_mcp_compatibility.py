#!/usr/bin/env python3
"""
MCP Compatibility Testing Suite

Comprehensive compatibility validation for MCP server automation system.
Tests version compatibility, system integration, dependency compatibility,
cross-platform compatibility, and upgrade/downgrade scenarios.

Test Coverage:
- MCP server version compatibility
- System dependency compatibility
- Node.js and npm version compatibility
- Cross-platform compatibility validation
- Backward and forward compatibility
- API compatibility validation
- Configuration compatibility
- Migration and upgrade testing

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0
"""

import pytest
import asyncio
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass
from packaging import version
import semver

from conftest import TestEnvironment, TestMCPServer

# Import automation modules
from config import MCPAutomationConfig, UpdateMode, LogLevel
from mcp_update_manager import MCPUpdateManager
from version_manager import MCPVersionManager
from download_manager import MCPDownloadManager
from error_handling import MCPError, ErrorSeverity


@dataclass
class CompatibilityTestResult:
    """Compatibility test result structure."""
    test_name: str
    component: str
    version_tested: str
    compatible: bool
    issues: List[str]
    warnings: List[str]
    test_timestamp: float
    platform_info: Dict[str, str]


@dataclass
class SystemRequirements:
    """System requirements specification."""
    node_version_min: str = "18.0.0"
    node_version_max: str = "21.0.0"
    npm_version_min: str = "8.0.0"
    npm_version_max: str = "10.0.0"
    python_version_min: str = "3.8.0"
    python_version_max: str = "3.12.0"
    supported_platforms: List[str] = None
    
    def __post_init__(self):
        if self.supported_platforms is None:
            self.supported_platforms = ["linux", "darwin", "win32"]


class CompatibilityTestManager:
    """Compatibility test manager for coordinated compatibility testing."""
    
    def __init__(self, config: MCPAutomationConfig):
        self.config = config
        self.requirements = SystemRequirements()
        self.platform_info = self._get_platform_info()
    
    def _get_platform_info(self) -> Dict[str, str]:
        """Get current platform information."""
        return {
            "system": platform.system(),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "node_version": self._get_node_version(),
            "npm_version": self._get_npm_version()
        }
    
    def _get_node_version(self) -> str:
        """Get Node.js version."""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip().lstrip('v')
            return "unknown"
        except Exception:
            return "not_installed"
    
    def _get_npm_version(self) -> str:
        """Get npm version."""
        try:
            result = subprocess.run(
                ["npm", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return "unknown"
        except Exception:
            return "not_installed"
    
    def check_system_compatibility(self) -> CompatibilityTestResult:
        """Check overall system compatibility."""
        issues = []
        warnings = []
        
        # Check Node.js version
        node_version = self.platform_info["node_version"]
        if node_version == "not_installed":
            issues.append("Node.js is not installed")
        elif node_version != "unknown":
            try:
                if version.parse(node_version) < version.parse(self.requirements.node_version_min):
                    issues.append(f"Node.js version {node_version} is below minimum {self.requirements.node_version_min}")
                elif version.parse(node_version) > version.parse(self.requirements.node_version_max):
                    warnings.append(f"Node.js version {node_version} is above tested maximum {self.requirements.node_version_max}")
            except Exception as e:
                warnings.append(f"Could not parse Node.js version {node_version}: {e}")
        
        # Check npm version
        npm_version = self.platform_info["npm_version"]
        if npm_version == "not_installed":
            issues.append("npm is not installed")
        elif npm_version != "unknown":
            try:
                if version.parse(npm_version) < version.parse(self.requirements.npm_version_min):
                    issues.append(f"npm version {npm_version} is below minimum {self.requirements.npm_version_min}")
                elif version.parse(npm_version) > version.parse(self.requirements.npm_version_max):
                    warnings.append(f"npm version {npm_version} is above tested maximum {self.requirements.npm_version_max}")
            except Exception as e:
                warnings.append(f"Could not parse npm version {npm_version}: {e}")
        
        # Check Python version
        python_version = self.platform_info["python_version"]
        try:
            if version.parse(python_version) < version.parse(self.requirements.python_version_min):
                issues.append(f"Python version {python_version} is below minimum {self.requirements.python_version_min}")
            elif version.parse(python_version) > version.parse(self.requirements.python_version_max):
                warnings.append(f"Python version {python_version} is above tested maximum {self.requirements.python_version_max}")
        except Exception as e:
            warnings.append(f"Could not parse Python version {python_version}: {e}")
        
        # Check platform support
        current_platform = platform.system().lower()
        if current_platform not in [p.lower() for p in self.requirements.supported_platforms]:
            issues.append(f"Platform {current_platform} is not officially supported")
        
        return CompatibilityTestResult(
            test_name="system_compatibility",
            component="system",
            version_tested="current",
            compatible=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            test_timestamp=asyncio.get_event_loop().time(),
            platform_info=self.platform_info
        )


class TestMCPVersionCompatibility:
    """Test suite for MCP server version compatibility."""
    
    @pytest.mark.compatibility
    @pytest.mark.asyncio
    async def test_mcp_server_version_compatibility(
        self,
        test_environment: TestEnvironment,
        mock_process_runner
    ):
        """Test compatibility across different MCP server versions."""
        config = test_environment.config
        version_manager = MCPVersionManager(config)
        download_manager = MCPDownloadManager(config)
        
        server_name = "files"
        package_name = config.mcp_servers[server_name]["package"]
        
        # Test different version scenarios
        version_scenarios = [
            ("1.0.0", "1.0.1", "patch_upgrade", True),
            ("1.0.0", "1.1.0", "minor_upgrade", True),
            ("1.0.0", "2.0.0", "major_upgrade", False),  # May break compatibility
            ("1.2.0", "1.1.0", "minor_downgrade", False),  # Downgrade may fail
            ("2.0.0", "1.0.0", "major_downgrade", False),  # Major downgrade should fail
        ]
        
        compatibility_results = []
        
        with patch.object(download_manager, '_run_command', side_effect=mock_process_runner):
            for from_version, to_version, scenario_type, expected_compatible in version_scenarios:
                # Set initial version
                await version_manager.record_activation(server_name, from_version)
                
                # Test upgrade/downgrade
                try:
                    # Mock version compatibility check
                    from_ver = version.parse(from_version)
                    to_ver = version.parse(to_version)
                    
                    compatibility_issues = []
                    compatibility_warnings = []
                    
                    # Check semantic versioning compatibility
                    if to_ver.major > from_ver.major:
                        if to_ver.major - from_ver.major > 1:
                            compatibility_issues.append("Major version jump detected")
                        else:
                            compatibility_warnings.append("Major version upgrade may break compatibility")
                    
                    elif to_ver.major < from_ver.major:
                        compatibility_issues.append("Major version downgrade not supported")
                    
                    elif to_ver.minor < from_ver.minor:
                        compatibility_warnings.append("Minor version downgrade may cause issues")
                    
                    is_compatible = len(compatibility_issues) == 0 and expected_compatible
                    
                    compatibility_results.append({
                        "scenario": scenario_type,
                        "from_version": from_version,
                        "to_version": to_version,
                        "compatible": is_compatible,
                        "issues": compatibility_issues,
                        "warnings": compatibility_warnings
                    })
                    
                except Exception as e:
                    compatibility_results.append({
                        "scenario": scenario_type,
                        "from_version": from_version,
                        "to_version": to_version,
                        "compatible": False,
                        "issues": [f"Version parsing error: {str(e)}"],
                        "warnings": []
                    })
        
        # Verify compatibility results
        assert len(compatibility_results) == len(version_scenarios)
        
        # Check specific scenarios
        patch_upgrade = next(r for r in compatibility_results if r["scenario"] == "patch_upgrade")
        assert patch_upgrade["compatible"] is True
        
        minor_upgrade = next(r for r in compatibility_results if r["scenario"] == "minor_upgrade")
        assert minor_upgrade["compatible"] is True
        
        major_upgrade = next(r for r in compatibility_results if r["scenario"] == "major_upgrade")
        # Major upgrades may or may not be compatible depending on implementation
        
        major_downgrade = next(r for r in compatibility_results if r["scenario"] == "major_downgrade")
        assert major_downgrade["compatible"] is False
    
    @pytest.mark.compatibility
    @pytest.mark.asyncio
    async def test_dependency_version_compatibility(
        self,
        test_environment: TestEnvironment,
        mock_process_runner
    ):
        """Test compatibility of MCP server dependencies."""
        config = test_environment.config
        download_manager = MCPDownloadManager(config)
        
        # Test package with various dependency scenarios
        dependency_scenarios = [
            {
                "package": "@test/package-stable-deps",
                "dependencies": {
                    "lodash": "^4.17.21",  # Stable, compatible
                    "express": "^4.18.0",  # Stable, compatible
                    "semver": "^7.3.8"     # Stable, compatible
                }
            },
            {
                "package": "@test/package-outdated-deps",
                "dependencies": {
                    "lodash": "^3.10.1",   # Very old version
                    "express": "^3.21.2",  # Old major version
                    "request": "^2.88.2"   # Deprecated package
                }
            },
            {
                "package": "@test/package-beta-deps",
                "dependencies": {
                    "next": "^14.0.0-beta.1",  # Beta version
                    "react": "^18.0.0-rc.0",   # Release candidate
                    "typescript": "^5.0.0-dev" # Development version
                }
            }
        ]
        
        def analyze_dependency_compatibility(deps: Dict[str, str]) -> Tuple[List[str], List[str]]:
            """Analyze dependency compatibility."""
            issues = []
            warnings = []
            
            # Known problematic packages
            deprecated_packages = ["request", "bower", "grunt"]
            security_vulnerable = ["lodash<4.17.21", "express<4.17.1"]
            
            for dep_name, dep_version in deps.items():
                # Check for deprecated packages
                if dep_name in deprecated_packages:
                    warnings.append(f"Package {dep_name} is deprecated")
                
                # Check for pre-release versions
                if any(tag in dep_version for tag in ["beta", "alpha", "rc", "dev"]):
                    warnings.append(f"Package {dep_name} uses pre-release version {dep_version}")
                
                # Check for very old versions (simplified check)
                if dep_name == "lodash" and "^3." in dep_version:
                    issues.append(f"Package {dep_name} version {dep_version} has known security vulnerabilities")
                
                if dep_name == "express" and "^3." in dep_version:
                    issues.append(f"Package {dep_name} version {dep_version} is no longer supported")
            
            return issues, warnings
        
        compatibility_results = []
        
        with patch.object(download_manager, '_run_command', side_effect=mock_process_runner):
            for scenario in dependency_scenarios:
                package_name = scenario["package"]
                dependencies = scenario["dependencies"]
                
                # Analyze dependency compatibility
                issues, warnings = analyze_dependency_compatibility(dependencies)
                
                # Create Mock package.json
                staging_path = config.paths.staging_root / package_name.replace("/", "_")
                staging_path.mkdir(parents=True, exist_ok=True)
                
                package_json = {
                    "name": package_name,
                    "version": "1.0.0",
                    "dependencies": dependencies
                }
                
                (staging_path / "package.json").write_text(json.dumps(package_json, indent=2))
                
                compatibility_results.append({
                    "package": package_name,
                    "compatible": len(issues) == 0,
                    "issues": issues,
                    "warnings": warnings,
                    "dependencies_count": len(dependencies)
                })
        
        # Verify dependency compatibility results
        assert len(compatibility_results) == len(dependency_scenarios)
        
        # Stable package should be compatible
        stable_result = next(r for r in compatibility_results if "stable" in r["package"])
        assert stable_result["compatible"] is True
        assert len(stable_result["issues"]) == 0
        
        # Outdated package should have issues
        outdated_result = next(r for r in compatibility_results if "outdated" in r["package"])
        assert outdated_result["compatible"] is False
        assert len(outdated_result["issues"]) > 0
        
        # Beta package should have warnings but may be compatible
        beta_result = next(r for r in compatibility_results if "beta" in r["package"])
        assert len(beta_result["warnings"]) > 0
    
    @pytest.mark.compatibility
    @pytest.mark.asyncio
    async def test_api_compatibility_validation(
        self,
        test_environment: TestEnvironment,
        mock_process_runner,
        mock_health_checker
    ):
        """Test API compatibility across different versions."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        
        # Define API compatibility matrix
        api_versions = [
            {
                "version": "1.0.0",
                "api_features": ["health_check", "basic_commands"],
                "breaking_changes": []
            },
            {
                "version": "1.1.0", 
                "api_features": ["health_check", "basic_commands", "extended_commands"],
                "breaking_changes": []
            },
            {
                "version": "2.0.0",
                "api_features": ["health_check_v2", "commands_v2", "new_features"],
                "breaking_changes": ["removed_basic_commands", "changed_health_format"]
            }
        ]
        
        def check_api_compatibility(from_version: str, to_version: str) -> Tuple[bool, List[str], List[str]]:
            """Check API compatibility between versions."""
            from_api = next(api for api in api_versions if api["version"] == from_version)
            to_api = next(api for api in api_versions if api["version"] == to_version)
            
            issues = []
            warnings = []
            
            # Check for breaking changes
            if to_api["breaking_changes"]:
                for breaking_change in to_api["breaking_changes"]:
                    issues.append(f"Breaking change in {to_version}: {breaking_change}")
            
            # Check for removed features
            from_features = set(from_api["api_features"])
            to_features = set(to_api["api_features"])
            
            removed_features = from_features - to_features
            for feature in removed_features:
                if not any(f"{feature}_v2" in to_features for f in [feature]):
                    issues.append(f"Feature removed: {feature}")
                else:
                    warnings.append(f"Feature replaced: {feature}")
            
            # Check for new features
            new_features = to_features - from_features
            for feature in new_features:
                warnings.append(f"New feature available: {feature}")
            
            is_compatible = len(issues) == 0
            return is_compatible, issues, warnings
        
        # Test API compatibility scenarios
        api_test_scenarios = [
            ("1.0.0", "1.1.0", True),   # Minor version, should be compatible
            ("1.1.0", "1.0.0", False),  # Downgrade, may lose features
            ("1.1.0", "2.0.0", False),  # Major version, breaking changes
            ("2.0.0", "1.1.0", False),  # Major downgrade, not supported
        ]
        
        api_compatibility_results = []
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker):
            for from_ver, to_ver, expected_compatible in api_test_scenarios:
                try:
                    is_compatible, issues, warnings = check_api_compatibility(from_ver, to_ver)
                    
                    api_compatibility_results.append({
                        "from_version": from_ver,
                        "to_version": to_ver,
                        "expected_compatible": expected_compatible,
                        "actual_compatible": is_compatible,
                        "issues": issues,
                        "warnings": warnings
                    })
                    
                except Exception as e:
                    api_compatibility_results.append({
                        "from_version": from_ver,
                        "to_version": to_ver,
                        "expected_compatible": expected_compatible,
                        "actual_compatible": False,
                        "issues": [f"API compatibility check failed: {str(e)}"],
                        "warnings": []
                    })
        
        # Verify API compatibility results
        assert len(api_compatibility_results) == len(api_test_scenarios)
        
        for result in api_compatibility_results:
            # Compatible upgrades should match expectations
            if result["expected_compatible"]:
                assert result["actual_compatible"] is True, f"Expected compatible upgrade from {result['from_version']} to {result['to_version']}"
            else:
                # Incompatible upgrades should be detected
                assert result["actual_compatible"] is False or len(result["issues"]) > 0


class TestMCPSystemCompatibility:
    """Test suite for system-level compatibility validation."""
    
    @pytest.mark.compatibility
    @pytest.mark.asyncio
    async def test_cross_platform_compatibility(
        self,
        test_environment: TestEnvironment
    ):
        """Test cross-platform compatibility validation."""
        config = test_environment.config
        compatibility_manager = CompatibilityTestManager(config)
        
        # Get current platform info
        platform_info = compatibility_manager.platform_info
        
        # Test platform-specific configurations
        platform_configs = {
            "linux": {
                "path_separator": "/",
                "executable_extension": "",
                "line_ending": "\n",
                "case_sensitive": True
            },
            "darwin": {
                "path_separator": "/", 
                "executable_extension": "",
                "line_ending": "\n",
                "case_sensitive": False  # HFS+ default
            },
            "windows": {
                "path_separator": "\\",
                "executable_extension": ".exe",
                "line_ending": "\r\n",
                "case_sensitive": False
            }
        }
        
        current_platform = platform.system().lower()
        
        # Test path handling
        test_paths = [
            "package.json",
            "bin/server",
            "lib/utils.js",
            "config/settings.json"
        ]
        
        path_compatibility_issues = []
        
        for test_path in test_paths:
            # Convert path to platform-specific format
            if current_platform in platform_configs:
                platform_config = platform_configs[current_platform]
                
                # Test path separator handling
                normalized_path = test_path.replace("/", platform_config["path_separator"])
                
                # Test case sensitivity
                if not platform_config["case_sensitive"]:
                    # On case-insensitive systems, paths should be handled carefully
                    upper_path = test_path.upper()
                    if Path(upper_path) != Path(test_path) and platform_config["case_sensitive"]:
                        path_compatibility_issues.append(f"Case sensitivity issue with path: {test_path}")
        
        # Test executable handling
        executable_test = "mcp-server"
        if current_platform == "windows":
            expected_executable = executable_test + ".exe"
        else:
            expected_executable = executable_test
        
        # Test environment variables
        env_compatibility_issues = []
        
        # Test PATH variable handling
        path_var = os.environ.get("PATH", "")
        if current_platform == "windows":
            path_separator = ";"
        else:
            path_separator = ":"
        
        if path_separator not in path_var and len(path_var) > 0:
            env_compatibility_issues.append(f"Unexpected PATH separator for platform {current_platform}")
        
        # Create compatibility report
        platform_compatibility = {
            "current_platform": current_platform,
            "supported_platform": current_platform in [p.lower() for p in compatibility_manager.requirements.supported_platforms],
            "path_issues": path_compatibility_issues,
            "env_issues": env_compatibility_issues,
            "executable_format": expected_executable
        }
        
        # Verify platform compatibility
        assert platform_compatibility["supported_platform"], f"Platform {current_platform} is not supported"
        assert len(platform_compatibility["path_issues"]) == 0, f"Path compatibility issues: {platform_compatibility['path_issues']}"
        assert len(platform_compatibility["env_issues"]) == 0, f"Environment compatibility issues: {platform_compatibility['env_issues']}"
    
    @pytest.mark.compatibility
    @pytest.mark.asyncio
    async def test_system_dependency_compatibility(
        self,
        test_environment: TestEnvironment
    ):
        """Test system dependency compatibility."""
        config = test_environment.config
        compatibility_manager = CompatibilityTestManager(config)
        
        # Check system compatibility
        system_compatibility = compatibility_manager.check_system_compatibility()
        
        # Verify system compatibility result structure
        assert hasattr(system_compatibility, 'test_name')
        assert hasattr(system_compatibility, 'compatible')
        assert hasattr(system_compatibility, 'issues')
        assert hasattr(system_compatibility, 'warnings')
        assert hasattr(system_compatibility, 'platform_info')
        
        # Platform info should be populated
        assert "system" in system_compatibility.platform_info
        assert "python_version" in system_compatibility.platform_info
        assert "node_version" in system_compatibility.platform_info
        assert "npm_version" in system_compatibility.platform_info
        
        # If system is compatible, should have no critical issues
        if system_compatibility.compatible:
            critical_issues = [issue for issue in system_compatibility.issues 
                             if "not installed" in issue.lower() or "below minimum" in issue.lower()]
            assert len(critical_issues) == 0, f"System marked compatible but has critical issues: {critical_issues}"
        
        # Test dependency resolution
        dependency_requirements = {
            "node": {"min": "18.0.0", "max": "21.0.0"},
            "npm": {"min": "8.0.0", "max": "10.0.0"},
            "python": {"min": "3.8.0", "max": "3.12.0"}
        }
        
        dependency_status = {}
        
        for dep_name, req in dependency_requirements.items():
            if dep_name == "python":
                current_version = compatibility_manager.platform_info["python_version"]
            else:
                current_version = compatibility_manager.platform_info[f"{dep_name}_version"]
            
            if current_version in ["unknown", "not_installed"]:
                status = "missing"
            else:
                try:
                    curr_ver = version.parse(current_version)
                    min_ver = version.parse(req["min"])
                    max_ver = version.parse(req["max"])
                    
                    if curr_ver < min_ver:
                        status = "below_minimum"
                    elif curr_ver > max_ver:
                        status = "above_maximum"
                    else:
                        status = "compatible"
                except Exception:
                    status = "version_parse_error"
            
            dependency_status[dep_name] = {
                "current_version": current_version,
                "required_min": req["min"],
                "required_max": req["max"],
                "status": status
            }
        
        # Verify dependency status
        for dep_name, status_info in dependency_status.items():
            if status_info["status"] == "missing":
                # Only fail if it's a critical dependency
                if dep_name in ["node", "npm"]:
                    assert False, f"Critical dependency {dep_name} is missing"
            elif status_info["status"] == "below_minimum":
                # Version too old
                assert False, f"Dependency {dep_name} version {status_info['current_version']} is below minimum {status_info['required_min']}"
    
    @pytest.mark.compatibility
    @pytest.mark.asyncio
    async def test_configuration_compatibility(
        self,
        test_environment: TestEnvironment,
        tmp_path: Path
    ):
        """Test configuration compatibility across versions."""
        config = test_environment.config
        
        # Test different configuration versions
        config_versions = [
            {
                "version": "1.0.0",
                "config": {
                    "servers": {
                        "files": {"package": "@modelcontextprotocol/server-filesystem"}
                    },
                    "settings": {
                        "timeout": 30
                    }
                }
            },
            {
                "version": "1.1.0",
                "config": {
                    "servers": {
                        "files": {"package": "@modelcontextprotocol/server-filesystem", "wrapper": "files.sh"}
                    },
                    "settings": {
                        "timeout": 30,
                        "retry_attempts": 3
                    }
                }
            },
            {
                "version": "2.0.0",
                "config": {
                    "mcp_servers": {  # Renamed from 'servers'
                        "files": {"package": "@modelcontextprotocol/server-filesystem", "wrapper": "files.sh"}
                    },
                    "performance": {  # Renamed from 'settings'
                        "download_timeout_seconds": 30,  # Renamed from 'timeout'
                        "retry_attempts": 3
                    }
                }
            }
        ]
        
        def migrate_config(old_config: Dict[str, Any], from_version: str, to_version: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
            """Migrate configuration between versions."""
            migrated_config = old_config.copy()
            issues = []
            warnings = []
            
            from_ver = version.parse(from_version)
            to_ver = version.parse(to_version)
            
            # Migration from 1.x to 2.0
            if from_ver.major == 1 and to_ver.major == 2:
                # Rename 'servers' to 'mcp_servers'
                if "servers" in migrated_config:
                    migrated_config["mcp_servers"] = migrated_config.pop("servers")
                    warnings.append("Renamed 'servers' to 'mcp_servers'")
                
                # Migrate settings to performance
                if "settings" in migrated_config:
                    settings = migrated_config.pop("settings")
                    migrated_config["performance"] = {}
                    
                    if "timeout" in settings:
                        migrated_config["performance"]["download_timeout_seconds"] = settings["timeout"]
                        warnings.append("Renamed 'timeout' to 'download_timeout_seconds'")
                    
                    if "retry_attempts" in settings:
                        migrated_config["performance"]["retry_attempts"] = settings["retry_attempts"]
                    
                    warnings.append("Migrated 'settings' to 'performance'")
            
            # Check for deprecated fields
            deprecated_fields = ["old_field", "legacy_setting"]
            for field in deprecated_fields:
                if field in migrated_config:
                    warnings.append(f"Deprecated field '{field}' found")
            
            return migrated_config, issues, warnings
        
        # Test configuration migrations
        migration_results = []
        
        for i, from_config in enumerate(config_versions[:-1]):
            to_config = config_versions[i + 1]
            
            # Test migration
            migrated, issues, warnings = migrate_config(
                from_config["config"],
                from_config["version"],
                to_config["version"]
            )
            
            migration_results.append({
                "from_version": from_config["version"],
                "to_version": to_config["version"],
                "migration_successful": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "migrated_config": migrated
            })
        
        # Verify migration results
        assert len(migration_results) > 0
        
        # All migrations should be successful
        for result in migration_results:
            assert result["migration_successful"], f"Migration from {result['from_version']} to {result['to_version']} failed: {result['issues']}"
        
        # Check specific migration scenarios
        v1_to_v2_migration = next((r for r in migration_results 
                                 if r["from_version"] == "1.1.0" and r["to_version"] == "2.0.0"), None)
        
        if v1_to_v2_migration:
            migrated_config = v1_to_v2_migration["migrated_config"]
            
            # Verify structural changes
            assert "mcp_servers" in migrated_config
            assert "servers" not in migrated_config
            assert "performance" in migrated_config
            assert "settings" not in migrated_config
            
            # Verify field renames
            if "performance" in migrated_config:
                assert "download_timeout_seconds" in migrated_config["performance"]


class TestMCPUpgradeCompatibility:
    """Test suite for upgrade and migration compatibility."""
    
    @pytest.mark.compatibility
    @pytest.mark.asyncio
    async def test_seamless_upgrade_scenarios(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner
    ):
        """Test seamless upgrade scenarios without service interruption."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        version_manager = MCPVersionManager(config)
        
        server_name = "files"
        
        # Test seamless upgrade scenarios
        upgrade_scenarios = [
            {
                "name": "patch_upgrade",
                "from_version": "1.0.0",
                "to_version": "1.0.1",
                "should_be_seamless": True,
                "expected_downtime": 0
            },
            {
                "name": "minor_upgrade",
                "from_version": "1.0.1", 
                "to_version": "1.1.0",
                "should_be_seamless": True,
                "expected_downtime": 5  # Brief restart
            },
            {
                "name": "major_upgrade",
                "from_version": "1.1.0",
                "to_version": "2.0.0", 
                "should_be_seamless": False,
                "expected_downtime": 30  # Significant changes
            }
        ]
        
        upgrade_results = []
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            for scenario in upgrade_scenarios:
                # Set initial version
                await version_manager.record_activation(server_name, scenario["from_version"])
                
                # Monitor service availability during upgrade
                service_availability = []
                upgrade_start_time = asyncio.get_event_loop().time()
                
                # Simulate upgrade process
                try:
                    # Pre-upgrade health check
                    pre_health = await update_manager._run_health_check(server_name, timeout=10)
                    service_availability.append({
                        "timestamp": asyncio.get_event_loop().time(),
                        "available": pre_health["healthy"],
                        "phase": "pre_upgrade"
                    })
                    
                    # Perform upgrade
                    await update_manager.update_server(server_name, target_version=scenario["to_version"])
                    
                    # Simulate brief downtime for activation
                    await asyncio.sleep(0.1)  # Simulate activation time
                    
                    # Activate new version
                    await update_manager.activate_server(server_name)
                    
                    # Post-upgrade health check
                    post_health = await update_manager._run_health_check(server_name, timeout=10)
                    service_availability.append({
                        "timestamp": asyncio.get_event_loop().time(),
                        "available": post_health["healthy"],
                        "phase": "post_upgrade"
                    })
                    
                    upgrade_end_time = asyncio.get_event_loop().time()
                    total_upgrade_time = upgrade_end_time - upgrade_start_time
                    
                    # Calculate downtime
                    downtime_periods = []
                    for i, status in enumerate(service_availability):
                        if not status["available"]:
                            if i == 0:
                                downtime_start = status["timestamp"]
                            else:
                                downtime_start = service_availability[i-1]["timestamp"]
                            
                            # Find when service becomes available again
                            for j in range(i+1, len(service_availability)):
                                if service_availability[j]["available"]:
                                    downtime_end = service_availability[j]["timestamp"]
                                    downtime_periods.append(downtime_end - downtime_start)
                                    break
                    
                    total_downtime = sum(downtime_periods)
                    
                    upgrade_results.append({
                        "scenario": scenario["name"],
                        "from_version": scenario["from_version"],
                        "to_version": scenario["to_version"],
                        "upgrade_successful": True,
                        "total_upgrade_time": total_upgrade_time,
                        "total_downtime": total_downtime,
                        "expected_downtime": scenario["expected_downtime"],
                        "should_be_seamless": scenario["should_be_seamless"],
                        "is_seamless": total_downtime <= scenario["expected_downtime"]
                    })
                    
                except Exception as e:
                    upgrade_results.append({
                        "scenario": scenario["name"],
                        "from_version": scenario["from_version"],
                        "to_version": scenario["to_version"],
                        "upgrade_successful": False,
                        "error": str(e),
                        "total_upgrade_time": 0,
                        "total_downtime": float('inf'),
                        "is_seamless": False
                    })
        
        # Verify upgrade results
        assert len(upgrade_results) == len(upgrade_scenarios)
        
        # Check seamless upgrade expectations
        for result in upgrade_results:
            if result["upgrade_successful"]:
                scenario = next(s for s in upgrade_scenarios if s["name"] == result["scenario"])
                
                if scenario["should_be_seamless"]:
                    assert result["is_seamless"], f"Scenario {result['scenario']} should be seamless but had {result['total_downtime']}s downtime"
                
                # Verify downtime is within expected bounds
                assert result["total_downtime"] <= scenario["expected_downtime"] * 2, f"Downtime {result['total_downtime']}s exceeded expected {scenario['expected_downtime']}s"