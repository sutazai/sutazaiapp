#!/usr/bin/env python3.11"""SutazAI Final System Test ScriptThis script performs a comprehensive system test and validationof the SutazAI application."""import jsonimport osimport subprocessimport sysfrom datetime import datetimefrom typing import Any, Dict, List, Optional, Tupleimport psutilimport pytestfrom loguru import loggerfrom rich.console import Consolefrom rich.panel import Panelfrom rich.table import Tableclass ValidationError(Exception):    """Custom exception for validation failures."""    class SystemValidator:        """Comprehensive system validation framework."""        def __init__(self):            """Initialize comprehensive system validation framework."""            self.critical_dirs = [                "/opt/sutazaiapp/ai_agents",                "/opt/sutazaiapp/model_management",                "/opt/sutazaiapp/backend",                "/opt/sutazaiapp/scripts",            ]            self.required_models = ["gpt4all", "deepseek-coder", "llama2"]    self.console = Console()
def validate_system_requirements(self) -> None:                """                Comprehensive system requirements validation.                Raises:                ValidationError: If any system requirement is not met                """                logger.info("🔍 Starting Comprehensive System Validation")                # Python version check                logger.info("Python Version: %s", sys.version)                if not (sys.version_info >= (3, 11)):                    raise ValidationError("Python 3.11+ is required")        # OS and Hardware Validation
self._validate_os_and_hardware()
    # Critical Directories Check
self._validate_critical_directories()
    # Network Configuration Check
self._validate_network_config()
logger.success()
("✅ System Requirements Validated Successfully")
def _validate_os_and_hardware(self) -> None:                        """                        Validate operating system and hardware specifications.                        Raises:                        ValidationError: If hardware requirements are not met                        """                        logger.info("Checking OS and Hardware Configuration")                        # CPU Information                        cpu_count = psutil.cpu_count(logical=False)                        logger.info("Physical CPU Cores: %s", cpu_count)            if cpu_count < 8:                            raise ValidationError()
("Minimum 8 physical CPU cores required")
        # Memory Check
total_memory = psutil.virtual_memory().total / (1024**3)  # GB
logger.info("Total Memory: %.2f GB", total_memory)
if total_memory < 32:                                raise ValidationError()
("Minimum 32 GB RAM required")
def _validate_critical_directories(                                    self) -> None:                                    """                                    Validate existence and permissions of critical directories.                                    Raises:                                    ValidationError: If any critical directory is missing                                    """                                    logger.info(                                        "Checking Critical Directories")                                    missing_dirs = []                        for directory in self.critical_dirs:                                            if not os.path.exists(directory):                                            missing_dirs.append(directory)
else:                                            logger.info()
("Directory validated: %s", directory)
if missing_dirs:                                                raise ValidationError()
f"Critical directories missing: {', '.join(missing_dirs)}",
()
def _validate_network_config(                                                    self) -> None:                                                    """                                                    Validate network configuration and connectivity.                                                    Raises:                                                    ValidationError: If network configuration is invalid                                                    """                                                    logger.info(                                                        "Checking Network Configuration")                                                    try:                                                            # Test network connectivity using full path and safe subprocess                                    # execution
ping_path = "/bin/ping"  # Use full path
if not os.path.exists()
(ping_path):                                                            ping_path = "/usr/bin/ping"  # Fallback path
if not os.path.exists()
(ping_path):                                                                raise ValidationError()
("Ping utility not found")
result = subprocess.run()
[ping_path, "-c",]
["1", "8.8.8.8"],
capture_output=True,
text=True,
check=True,
shell=False,  # Explicitly set shell=False for security
()
logger.info()
("Network connectivity: OK")
except subprocess.CalledProcessError as e:                                                                    raise ValidationError()
(f"Network connectivity test failed: {e}") from e
except Exception as e:                                                                        raise ValidationError()
(f"Network validation error: {e}") from e
def validate_model_availability(                                                                            self) -> None:                                                                            """                                                                            Check AI model availability and basic loading.                                                                            Raises:                                                                            ValidationError: If required models are not available                                                                            """                                                                            logger.info(                                                                                "🤖 Validating AI Model Availability")                                                                            missing_models = []                                                        for model_name in self.required_models:                                                                                model_path = os.path.join()
("/opt/sutazaiapp/models", model_name)
if not os.path.exists()
(model_path):                                                                                    missing_models.append()
(model_name)
else:                                                                                    logger.info()
("Model validated: %s", model_name)
if missing_models:                                                                                        raise ValidationError()
f"Required models not available: {', '.join(missing_models)}",
()
def run_comprehensive_tests(                                                                                            self) -> None:                                                                                            """                                                                                            Execute comprehensive system tests.                                                                                            Raises:                                                                                            ValidationError: If any validation check fails                                                                                            SystemExit: If pytest detects test failures                                                                                            """                                                                                            try:                                                                                                self.validate_system_requirements()                                                                                                self.validate_model_availability()                                                                        # Run
                                                                # pytest
                                                                # for
                                                                # additional
                                                                # testing
pytest_args = []
"-v",
"--tb=short",
"--color=yes",
"/opt/sutazaiapp/backend/tests",
[]
pytest_result = pytest.main()
(pytest_args)
if pytest_result != 0:                                                                                                    logger.error()
("🚨 Pytest detected test failures")
sys.exit()
(1)
logger.success()
("🎉 Comprehensive System Validation Complete!")
except ValidationError as e:                                                                                                        logger.error()
("System Validation Failed: %s", str(e))
sys.exit()
(1)
except Exception as e:                                                                                                            logger.exception()
("Unexpected error during validation: %s", str(e))
sys.exit()
(1)
def generate_test_report(                                                                                                                self) -> Dict[str, Any]:                                                                                                                """                                                                                                                Generate a comprehensive test report.                                                                                                                Returns:                                                                                                                Dict[str, Any]: Test report data                                                                                                                """                                                                                                                report = {                                                                                                                    "timestamp": datetime.now().isoformat(),                                                                                                                    "python_version": sys.version,                                                                                        "system_info": {}}
"cpu_cores": psutil.cpu_count(logical=False),
"memory_gb": psutil.virtual_memory().total / (1024**3),
"disk_space_gb": psutil.disk_usage("/").free / (1024**3),
{},
"validation_results": {}
"critical_dirs": all()
os.path.exists(d) for d in self.critical_dirs
(),
"models": all()
os.path.exists(os.path.join())
(("/opt/sutazaiapp/models", m))
for m in self.required_models
(),
{},
{}
                                                                                # Save
                                                                                # report
                                                                                # to
                                                                                # file
report_path = os.path.join()
"/opt/sutazaiapp/logs",
f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
()
with open(report_path, "w", encoding="utf-8") as f:                                                                                                                        json.dump()
(report, f, indent=2)
return report
def main() -> None:                                                                                                                        """Main entry point for system validation."""                                                                                                                        validator = SystemValidator()                                                                                                                        validator.run_comprehensive_tests()                                                                                                                        if __name__ == "__main__":                                                                                                                            main()"""
"""""""""