#!/opt/sutazaiapp/venv/bin/python3import gcimport loggingimport osimport shutilimport sysimport tempfileimport timeimport importlibfrom pathlib import Pathfrom typing import Any, Dict, List, Optional, Setimport psutil# Configure logginglogging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s: %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler("/opt/sutazaiapp/logs/system_optimizer.log"),],)logger = logging.getLogger(__name__)class SystemOptimizer:    def __init__(self, project_root:
    """self.project_root = Path(project_root)"""

self.thresholds = {
"cpu_warning": 70.0,
"cpu_critical": 85.0,
"memory_warning": 75.0,
"memory_critical": 90.0,
"disk_warning": 80.0,
"disk_critical": 90.0,
}
# Initialize optimization modules
self.optimization_modules = self._load_optimization_modules()
def _load_optimization_modules(self) -> List[Any]:            """Load optimization modules dynamically."""
modules = []
try:
        # Add project root to Python path
    sys.path.append(str(self.project_root))
        # Import optimization modules if they exist
    module_paths = [
    "scripts/comprehensive_cleanup.py",
    "scripts/process_optimizer.py",
    "scripts/system_maintenance_pro.py",
]
for module_path in module_paths:
    try:
        if os.path.exists(self.project_root / module_path):
            module_name = Path(module_path).stem
            spec = importlib.util.spec_from_file_location(
            module_name, str(self.project_root / module_path)
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, module_name):
                modules.append(getattr(module, module_name)())
                except Exception as e:
                    logger.error(f"Failed to load module {module_path}: {e}")
                    except Exception as e:
                        logger.error(f"Error loading optimization modules: {e}")
                        return modules
                        def optimize_python_runtime(self) -> None:                                        """Optimize Python runtime environment."""
                        try:
                                    # Optimize garbage collection
                            gc.set_threshold(700, 10, 5)
                            gc.collect(2)  # Full collection
                                    # Optimize memory allocator
                            if hasattr(sys, "set_int_max_str_digits"):
                                sys.set_int_max_str_digits(4300)
                                        # Set process priority
                                try:
                                    os.nice(10)  # Lower priority to prevent resource hogging
                                    except (AttributeError, PermissionError):
                                        pass
                                            # Set environment variables for optimization                                                    os.environ.update({                                                    "PYTHONMALLOC": "malloc",                                                    "PYTHONDEVMODE": "1",                                                    "PYTHONASYNCIODEBUG": "0",                                                    "PYTHONWARNINGS": "ignore",                                                    })                                                    logger.info("Python runtime optimized")                                                    except Exception as e:                                                        logger.error(f"Failed to optimize Python runtime: {e}")                                                        def optimize_system_resources(self) -> None:                                                            """Optimize system-wide resources."""
                            try:
                                                # Clear system caches
                            self._clear_system_caches()
                                                # Optimize file descriptors
                            self._optimize_file_descriptors()
                                                # Clean temporary files
                            self._clean_temp_files()
                                        except Exception as e:
                                            logger.error(f"Failed to optimize system resources: {e}")
                            def _clear_system_caches(self) -> None:                                                                        """Clear system caches safely."""
                                try:
                                                            # Drop caches (requires root)
                                                if os.geteuid() == 0:
                                                    with open("/proc/sys/vm/drop_caches", "w") as f:
                                                        f.write("1")
                                logger.info("System caches cleared")
                                                        except Exception:
                                                            pass
                                def _optimize_file_descriptors(self) -> None:                                                                                    """Optimize file descriptor limits."""
                                        try:
                                            import resource
                                                                        # Get current limits
                                        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                                                                        # Try to increase soft limit to hard limit
                                                            if soft < hard:
                                                                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
                                        logger.info(f"File descriptor limit increased: {soft} -> {hard}")
                                                                except Exception:
                                                                    pass
                                        def _clean_temp_files(self) -> None:                                                                                                """Clean temporary files safely."""
                                            try:
                                                temp_dirs = [tempfile.gettempdir()]
                                                                    if os.path.exists("/tmp"):
                                                                        temp_dirs.append("/tmp")
                                                                        for temp_dir in temp_dirs:
                                                                            try:
                                                                                            # Only remove files older than 1 day
                                                cutoff = time.time() - 86400
                                                                                for root, dirs, files in os.walk(temp_dir, topdown=False):
                                                                                    for name in files:
                                                                                        try:
                                                                                            path = os.path.join(root, name)
                                                                                            if os.path.getmtime(path) < cutoff:
                                                                                                os.unlink(path)
                                                                                                except (OSError, PermissionError):
                                                                                                    continue
                                                                                                for name in dirs:
                                                                                                    try:
                                                                                                        path = os.path.join(root, name)
                                                                                                        if not os.listdir(path):
                                                                                                            os.rmdir(path)
                                                                                                            except (OSError, PermissionError):
                                                                                                                continue
                                                                                                            except Exception as e:
                                                                                                                logger.error(f"Failed to clean temp directory {temp_dir}: {e}")
                                                                                                                except Exception as e:
                                                                                                                    logger.error(f"Failed to clean temporary files: {e}")
                                                                                def monitor_resources(self) -> Dict[str, float]:                                                                                                                                        """Monitor system resources and return metrics."""
                                                                                try:
                                                                                    cpu_percent = psutil.cpu_percent(interval=1)
                                                                                memory = psutil.virtual_memory()
                                                                                disk = psutil.disk_usage("/")
                                                                                metrics = {
                                                                                "cpu_percent": cpu_percent,
                                                                                "memory_percent": memory.percent,
                                                                                "memory_used_gb": memory.used / (1024**3),
                                                                                "memory_total_gb": memory.total / (1024**3),
                                                                                "disk_percent": disk.percent,
                                                                                "disk_free_gb": disk.free / (1024**3),
                                                                                "disk_total_gb": disk.total / (1024**3),
                                                                                }
                                                                                                                            # Log warnings based on thresholds
                                                                                                                    if cpu_percent > self.thresholds["cpu_critical"]:
                                                                                                                        logger.critical(f"Critical CPU usage: {cpu_percent}%")
                                                                                                                        elif cpu_percent > self.thresholds["cpu_warning"]:
                                                                                                                            logger.warning(f"High CPU usage: {cpu_percent}%")
                                                                                                                            if memory.percent > self.thresholds["memory_critical"]:
                                                                                                                                logger.critical(
                                                                                f"Critical memory usage: {memory.percent}% "
                                                                                            f"({metrics['memory_used_gb']:.1f}/{metrics['memory_total_gb']:.1f}GB)"
                                                                                            )
                                                                                                                            elif memory.percent > self.thresholds["memory_warning"]:
                                                                                                                                logger.warning(
                                                                                            f"High memory usage: {memory.percent}% "
                                                                                                f"({metrics['memory_used_gb']:.1f}/{metrics['memory_total_gb']:.1f}GB)"
                                                                                                )
                                                                                                                            if disk.percent > self.thresholds["disk_critical"]:
                                                                                                                                logger.critical(
                                                                                                f"Critical disk usage: {disk.percent}% "
                                                                                                    f"({metrics['disk_free_gb']:.1f}GB free)"
                                                                                                    )
                                                                                                                            elif disk.percent > self.thresholds["disk_warning"]:
                                                                                                                                logger.warning(
                                                                                                    f"High disk usage: {disk.percent}% "
                                                                                                        f"({metrics['disk_free_gb']:.1f}GB free)"
                                                                                                        )
                                                                                                        return metrics
                                                                                                    except Exception as e:                                                                                                                                                                    logger.error(f"Failed to monitor resources: {e}")                                                                                                                                                                    return {}                                                                                                                                                                def run_comprehensive_optimization(self) -> None:                                                                                                                                                                    """Execute comprehensive system optimization."""
                                                                                                        logger.info("🚀 Starting Ultra-Comprehensive System Optimization 🚀")
                                                                                                        try:
                                                                                                                                                        # Get initial resource metrics
                                                                                                        initial_metrics = self.monitor_resources()
                                                                                                                                if initial_metrics:
                                                                                                                                    logger.info(
                                                                                                        "Initial System Status:\n"
                                                                                                            f"CPU: {initial_metrics['cpu_percent']}%\n"
                                                                                                            f"Memory: {initial_metrics['memory_percent']}% "
                                                                                                            f"({initial_metrics['memory_used_gb']:.1f}/{initial_metrics['memory_total_gb']:.1f}GB)\n"
                                                                                                            f"Disk: {initial_metrics['disk_percent']}% "
                                                                                                            f"({initial_metrics['disk_free_gb']:.1f}GB free)"
                                                                                                            )
                                                                                                                                                            # Optimize Python runtime
                                                                                                            self.optimize_python_runtime()
                                                                                                                                                            # Optimize system resources
                                                                                                            self.optimize_system_resources()
                                                                                                                                                            # Run optimization modules
                                                                                                                                for module in self.optimization_modules:
                                                                                                                                    try:
                                                                                                                                        logger.info(f"Running optimization module: {module.__class__.__name__}")
                                                                                                                                        if hasattr(module, "run_optimization"):
                                                                                                                                            module.run_optimization()
                                                                                                                                            elif hasattr(module, "run_maintenance"):
                                                                                                                                                module.run_maintenance()
                                                                                                                                                except Exception as e:
                                                                                                                                                    logger.error(f"Module {module.__class__.__name__} failed: {e}")
                                                                                                                                                                            # Get final resource metrics
                                                                                                            final_metrics = self.monitor_resources()
                                                                                                                                                    if final_metrics and initial_metrics:
                                                                                                                                                        cpu_change = final_metrics["cpu_percent"] - initial_metrics["cpu_percent"]
                                                                                                            memory_change = final_metrics["memory_percent"] - initial_metrics["memory_percent"]
                                                                                                                                disk_change = final_metrics["disk_percent"] - initial_metrics["disk_percent"]
                                                                                                                                logger.info(
                                                                                                                                "Optimization Results:\n"
                                                                                                                                f"CPU: {final_metrics['cpu_percent']}% ({cpu_change:+.1f}%)\n"
                                                                                                                                f"Memory: {final_metrics['memory_percent']}% ({memory_change:+.1f}%)\n"
                                                                                                                                f"Disk: {final_metrics['disk_percent']}% ({disk_change:+.1f}%)"
                                                                                                                                )
                                                                                                                                logger.info("✅ Comprehensive System Optimization Completed Successfully ✅")
                                                                                                                                                    except Exception as e:
                                                                                                                                                        logger.error(f"System optimization failed: {e}")
                                                                                                                                raise
                                                                                                                                def main():                                                                                                                                                                                                        """Main entry point with error handling."""
                                                                                                                                    try:
                                                                                                                                        optimizer = SystemOptimizer("/opt/sutazaiapp")
                                                                                                                                    optimizer.run_comprehensive_optimization()
                                                                                                                                                        except Exception as e:
                                                                                                                                                            logger.error(f"Fatal error in system optimization: {e}")
                                                                                                                                    sys.exit(1)
                                                                                                                                                            if __name__ == "__main__":
                                                                                                                                                                main()

                                                                                                                                    """""")""""""