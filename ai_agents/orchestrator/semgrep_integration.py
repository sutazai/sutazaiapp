"""
SutazAI Semgrep Integration Module
Interfaces with Semgrep for static code analysis and vulnerability detection
"""

import os
import subprocess
import json
import tempfile
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional
from functools import lru_cache

# Configure logging
logger = logging.getLogger("semgrep_integration")

# Cache for semgrep results to avoid duplicate scans
RESULTS_CACHE = {}
CACHE_TTL = 3600  # 1 hour cache TTL


class SemgrepAnalyzer:
    """
    Class for analyzing code using Semgrep to detect vulnerabilities,
    security issues and code quality problems.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Semgrep analyzer

        Args:
            config_path: Path to custom Semgrep rules (optional)
        """
        self.config_path = config_path
        self._check_semgrep_installation()

    def _check_semgrep_installation(self):
        """Check if Semgrep is installed"""
        try:
            result = subprocess.run(
                ["semgrep", "--version"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                logger.warning("Semgrep may not be properly installed")
            else:
                logger.info(f"Semgrep version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error(
                "Semgrep not found in PATH. Please install it using 'pip install semgrep'"
            )
            raise RuntimeError("Semgrep not installed")

    def analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze code using Semgrep to find issues

        Args:
            code: Code string to analyze
            language: Programming language of the code

        Returns:
            Dictionary with analysis results
        """
        # Generate cache key based on code content and language
        cache_key = f"{hashlib.md5(code.encode()).hexdigest()}_{language}"

        # Check if we have a cached result that's still valid
        if cache_key in RESULTS_CACHE:
            cached_result, timestamp = RESULTS_CACHE[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                logger.info(f"Using cached semgrep results for {language} code")
                return cached_result

        with tempfile.NamedTemporaryFile(
            suffix=f".{self._get_file_extension(language)}", mode="w", delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            # Build the semgrep command with optimized settings
            cmd = ["semgrep", "--json", "--timeout", "10", "--max-memory", "512"]

            # Use custom config if provided, otherwise use "auto" but with optimizations
            if self.config_path:
                cmd.extend(["--config", self.config_path])
            else:
                # Use p/ci optimized ruleset instead of the more intensive "auto"
                cmd.extend(["--config", "p/ci"])

            # Add the temporary file to scan
            cmd.append(temp_file_path)

            logger.info(f"Running semgrep with command: {' '.join(cmd)}")

            # Run semgrep with resource limits
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            # Process the results
            if result.returncode != 0 and result.returncode != 1:
                # Return code 1 means issues were found, which is normal
                logger.warning(f"Semgrep returned non-zero code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Semgrep error: {result.stderr}")

            try:
                # Parse JSON output
                semgrep_output = json.loads(result.stdout)

                # Transform to a more usable format
                findings = self._parse_semgrep_output(semgrep_output)

                analysis_result = {
                    "success": True,
                    "findings": findings,
                    "summary": {
                        "total_issues": len(findings),
                        "error_count": sum(
                            1 for f in findings if f["severity"] == "ERROR"
                        ),
                        "warning_count": sum(
                            1 for f in findings if f["severity"] == "WARNING"
                        ),
                        "info_count": sum(
                            1 for f in findings if f["severity"] == "INFO"
                        ),
                    },
                }

                # Cache the result
                RESULTS_CACHE[cache_key] = (analysis_result, time.time())

                # Clean cache if it gets too large (>100 entries)
                if len(RESULTS_CACHE) > 100:
                    # Remove oldest entries
                    oldest_keys = sorted(
                        RESULTS_CACHE.keys(), key=lambda k: RESULTS_CACHE[k][1]
                    )[:20]
                    for key in oldest_keys:
                        del RESULTS_CACHE[key]

                return analysis_result

            except json.JSONDecodeError:
                logger.error("Failed to parse Semgrep JSON output")
                return {
                    "success": False,
                    "error": "Failed to parse Semgrep output",
                    "raw_output": result.stdout,
                }

        except Exception as e:
            logger.error(f"Error running Semgrep: {str(e)}")
            return {"success": False, "error": str(e)}
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze an existing file using Semgrep

        Args:
            file_path: Path to the file to analyze

        Returns:
            Dictionary with analysis results
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"success": False, "error": f"File not found: {file_path}"}

        # Generate cache key based on file path and modification time
        try:
            file_mtime = os.path.getmtime(file_path)
            cache_key = f"{file_path}_{file_mtime}"

            # Check if we have a cached result that's still valid
            if cache_key in RESULTS_CACHE:
                cached_result, timestamp = RESULTS_CACHE[cache_key]
                if time.time() - timestamp < CACHE_TTL:
                    logger.info(f"Using cached semgrep results for {file_path}")
                    return cached_result
        except Exception as e:
            logger.warning(f"Error getting file modification time: {e}")
            cache_key = f"{file_path}_{time.time()}"  # Fallback

        try:
            # Build the semgrep command with optimized settings
            cmd = ["semgrep", "--json", "--timeout", "10", "--max-memory", "512"]

            # Use custom config if provided, otherwise use "p/ci" for better performance
            if self.config_path:
                cmd.extend(["--config", self.config_path])
            else:
                cmd.extend(["--config", "p/ci"])

            # Add the file to scan
            cmd.append(file_path)

            logger.info(f"Running semgrep with command: {' '.join(cmd)}")

            # Run semgrep
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            # Process the results
            if result.returncode != 0 and result.returncode != 1:
                # Return code 1 means issues were found, which is normal
                logger.warning(f"Semgrep returned non-zero code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Semgrep error: {result.stderr}")

            try:
                # Parse JSON output
                semgrep_output = json.loads(result.stdout)

                # Transform to a more usable format
                findings = self._parse_semgrep_output(semgrep_output)

                analysis_result = {
                    "success": True,
                    "file_path": file_path,
                    "findings": findings,
                    "summary": {
                        "total_issues": len(findings),
                        "error_count": sum(
                            1 for f in findings if f["severity"] == "ERROR"
                        ),
                        "warning_count": sum(
                            1 for f in findings if f["severity"] == "WARNING"
                        ),
                        "info_count": sum(
                            1 for f in findings if f["severity"] == "INFO"
                        ),
                    },
                }

                # Cache the result
                RESULTS_CACHE[cache_key] = (analysis_result, time.time())

                return analysis_result

            except json.JSONDecodeError:
                logger.error("Failed to parse Semgrep JSON output")
                return {
                    "success": False,
                    "error": "Failed to parse Semgrep output",
                    "raw_output": result.stdout,
                }

        except Exception as e:
            logger.error(f"Error running Semgrep: {str(e)}")
            return {"success": False, "error": str(e)}

    def _parse_semgrep_output(
        self, semgrep_output: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse Semgrep JSON output into a more usable format

        Args:
            semgrep_output: Raw Semgrep JSON output

        Returns:
            List of findings in a simplified format
        """
        findings = []

        # Extract findings from the results
        if "results" in semgrep_output:
            for result in semgrep_output["results"]:
                finding = {
                    "check_id": result.get("check_id", ""),
                    "path": result.get("path", ""),
                    "line": result.get("start", {}).get("line", 0),
                    "column": result.get("start", {}).get("col", 0),
                    "message": result.get("extra", {}).get("message", ""),
                    "severity": result.get("extra", {}).get("severity", "WARNING"),
                    "code": result.get("extra", {}).get("lines", ""),
                    "fix": result.get("extra", {}).get("fix", ""),
                }
                findings.append(finding)

        # Sort findings by severity and line number
        severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        findings.sort(key=lambda x: (severity_order.get(x["severity"], 999), x["line"]))

        return findings

    def _get_file_extension(self, language: str) -> str:
        """
        Get the file extension for a given language

        Args:
            language: Programming language name

        Returns:
            File extension for the language
        """
        # Map of language names to file extensions
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "java": "java",
            "csharp": "cs",
            "c": "c",
            "cpp": "cpp",
            "go": "go",
            "ruby": "rb",
            "php": "php",
            "rust": "rs",
            "swift": "swift",
            "kotlin": "kt",
            "html": "html",
            "css": "css",
            "yaml": "yaml",
            "json": "json",
        }

        return extensions.get(language.lower(), "txt")


# Helper functions for easier usage
@lru_cache(maxsize=32)
def analyze_code(
    code: str, language: str, config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze code snippet using Semgrep

    Args:
        code: Code string to analyze
        language: Programming language of the code
        config_path: Optional path to custom Semgrep rules

    Returns:
        Dictionary with analysis results
    """
    analyzer = SemgrepAnalyzer(config_path)
    return analyzer.analyze_code(code, language)


@lru_cache(maxsize=32)
def analyze_file(file_path: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze file using Semgrep

    Args:
        file_path: Path to the file to analyze
        config_path: Optional path to custom Semgrep rules

    Returns:
        Dictionary with analysis results
    """
    analyzer = SemgrepAnalyzer(config_path)
    return analyzer.analyze_file(file_path)


def code_to_fix_suggestions(code: str, language: str) -> List[str]:
    """
    Analyze code and convert findings to fix suggestions

    Args:
        code: Code string to analyze
        language: Programming language of the code

    Returns:
        List of natural language fix suggestions
    """
    analysis = analyze_code(code, language)
    if not analysis.get("success", False):
        return ["Unable to analyze code"]

    findings = analysis.get("findings", [])
    suggestions = []

    for finding in findings:
        message = finding.get("message", "")
        line = finding.get("line", 0)
        fix = finding.get("fix", "")

        suggestion = f"Line {line}: {message}"
        if fix:
            suggestion += f" Fix: {fix}"

        suggestions.append(suggestion)

    return suggestions
