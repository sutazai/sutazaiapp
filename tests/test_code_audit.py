"""Tests for the code audit system."""
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from scripts.code_audit import CodeAuditor

@pytest.fixture
def auditor(tmp_path):
    return CodeAuditor(project_root=tmp_path)

def test_run_security_scan(auditor):
    with patch("subprocess.run") as mock_run:
        # Mock successful Bandit scan
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps([{"severity": "HIGH", "issue": "Test vulnerability"}]),
        )

        auditor.run_security_scan()

        assert len(auditor.audit_results["security"]) == 1
        assert auditor.audit_results["security"][0]["severity"] == "HIGH"

        # Mock failed scan
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Error running scan",
        )

        auditor.audit_results["security"] = []
        auditor.run_security_scan()

        assert len(auditor.audit_results["security"]) == 0

def test_run_quality_checks(auditor):
    with patch("subprocess.run") as mock_run:
        # Mock successful Pylint check
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps([{"type": "error", "message": "Test error"}]),
        )

        auditor.run_quality_checks()

        assert len(auditor.audit_results["quality"]) == 1
        assert auditor.audit_results["quality"][0]["type"] == "error"

        # Mock failed check
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Error running check",
        )

        auditor.audit_results["quality"] = []
        auditor.run_quality_checks()

        assert len(auditor.audit_results["quality"]) == 0

def test_check_dependencies(auditor, tmp_path):
    # Create test requirements.txt
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pytest==7.4.3\npylint==3.0.2\n")

    with patch("subprocess.run") as mock_run:
        # Mock pip list output
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps([
                {"name": "pytest", "version": "7.4.3", "latest": "7.5.0"},
            ]),
        )

        auditor.check_dependencies()

        assert len(auditor.audit_results["dependencies"]) == 2
        assert "requirements.txt" in auditor.audit_results["dependencies"][0]["file"]
        assert len(auditor.audit_results["dependencies"][0]["dependencies"]) == 2

def test_check_performance(auditor):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        # Create test modules
        for module in ["backend/main.py", "ai_agents/base_agent.py"]:
            module_path = auditor.project_root / module
            module_path.parent.mkdir(parents=True, exist_ok=True)
            module_path.write_text("print('test')")

        auditor.check_performance()

        assert len(auditor.audit_results["performance"]) == 2
        for result in auditor.audit_results["performance"]:
            assert "profile_" in result["profile"]

def test_check_documentation(auditor):
    with patch("subprocess.run") as mock_run:
        # Mock successful pydocstyle check
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps([{"code": "D100", "message": "Missing docstring"}]),
        )

        # Create test README files
        readme = auditor.project_root / "README.md"
        readme.write_text("# Test Project")

        auditor.check_documentation()

        assert len(auditor.audit_results["documentation"]) >= 2
        assert any(d["type"] == "readme" for d in auditor.audit_results["documentation"])
        assert any(d.get("code") == "D100" for d in auditor.audit_results["documentation"])

def test_generate_report(auditor, tmp_path):
    # Add some test results
    auditor.audit_results = {
        "security": [{"severity": "HIGH", "issue": "Test vulnerability"}],
        "quality": [{"type": "error", "message": "Test error"}],
        "dependencies": [{"name": "pytest", "version": "7.4.3"}],
        "performance": [{"module": "test.py", "profile": "profile_test.prof"}],
        "documentation": [{"type": "readme", "file": "README.md"}],
    }

    auditor.generate_report()

    # Check if report was generated
    report_files = list(auditor.log_dir.glob("audit_report_*.json"))
    assert len(report_files) == 1

    # Verify report contents
    with open(report_files[0]) as f:
        report = json.load(f)
        assert report == auditor.audit_results
