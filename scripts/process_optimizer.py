#!/opt/sutazaiapp/venv/bin/python3import loggingimport osimport sysfrom typing import Any, Dict, List, Setimport psutillogging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s: %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler("/opt/sutazaiapp/logs/process_optimizer.log"),],)logger = logging.getLogger(__name__)class ProcessOptimizer:    def __init__(self, project_root:
"""self.project_root = project_root"""

self.critical_processes: Set[str] = {
"autonomous_monitor.py",
"performance_manager.py",
}
self.process_limits = {
"cpu_percent": 80.0,
"memory_percent": 70.0,
"max_instances": 1,
}
def get_process_info(self, proc: psutil.Process) -> Dict[str, Any]:            """Get detailed process information safely."""
try:
    with proc.oneshot():
        return {
        "pid": proc.pid,            "name": proc.name(),            "cpu_percent": proc.cpu_percent(),            "memory_percent": proc.memory_percent(),            "status": proc.status(),            "create_time": proc.create_time(),            "cmdline": " ".join(proc.cmdline()),            }            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):                return {}            def find_redundant_processes(self) -> List[psutil.Process]:                """Find redundant or resource-heavy processes."""
        process_counts: Dict[str, List[psutil.Process]] = {}
        redundant_processes: List[psutil.Process] = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_percent"]):
            try:
        proc_info = self.get_process_info(proc)
                if not proc_info:
            continue
        cmdline = proc_info["cmdline"]
                if not cmdline or "python" not in cmdline.lower():
            continue
        # Check for critical processes
                for critical_proc in self.critical_processes:
                    if critical_proc in cmdline:
                        if critical_proc not in process_counts:
                    process_counts[critical_proc] = []
                    process_counts[critical_proc].append(proc)
                # Check resource usage
                    if (proc_info["cpu_percent"] > self.process_limits["cpu_percent"] or
                    proc_info["memory_percent"] > self.process_limits["memory_percent"]):
                    redundant_processes.append(proc)
                                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                    # Check for multiple instances of critical processes
                                for proc_name, procs in process_counts.items():
                                    if len(procs) > self.process_limits["max_instances"]:
                        # Keep the newest instance, mark others as redundant
                            sorted_procs = sorted(procs, key=lambda p: p.create_time(), reverse=True)
                            redundant_processes.extend(sorted_procs[1:])
                            return list(set(redundant_processes))
                            def terminate_redundant_processes(self, processes: List[psutil.Process]) -> None:                                    """Terminate redundant processes with proper cleanup."""
                                        for proc in processes:
                                try:
                                proc_info = self.get_process_info(proc)
                                                if not proc_info:
                                    continue
                                logger.warning(f"Terminating process: PID {proc_info['pid']} ({proc_info['cmdline']})")
                                proc.terminate()
                                try:
                                proc.wait(timeout=5)
                                                    except psutil.TimeoutExpired:
                                    logger.error(f"Force killing process: PID {proc_info['pid']}")
                                    proc.kill()
                                                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                                        continue
                                    def optimize_system_resources(self) -> None:                                                    """Optimize system-wide resource usage."""
                                    try:
                                            # Set process niceness for better CPU scheduling
                                    os.nice(10)
                                            # Optimize Python memory allocator
                                                            if hasattr(sys, "set_int_max_str_digits"):
                                        sys.set_int_max_str_digits(4300)
                                                # Optimize garbage collection
                                        import gc
                                        gc.set_threshold(700, 10, 5)
                                                # Set environment variables for optimization
                                        os.environ["PYTHONMALLOC"] = "malloc"
                                        os.environ["PYTHONDEVMODE"] = "1"
                                        os.environ["PYTHONASYNCIODEBUG"] = "0"
                                                                except Exception as e:
                                            logger.error(f"Failed to optimize system resources: {e}")
                                            def run_optimization(self) -> None:                                                                    """Run comprehensive process optimization."""
                                            logger.info("Starting process optimization")
                                            try:
                                                            # Optimize system resources
                                            self.optimize_system_resources()
                                                            # Find and terminate redundant processes
                                            redundant_processes = self.find_redundant_processes()
                                                                        if redundant_processes:
                                                self.terminate_redundant_processes(redundant_processes)
                                                                # Monitor system resources
                                                cpu_percent = psutil.cpu_percent(interval=1)
                                                memory = psutil.virtual_memory()
                                                logger.info(
                                                f"System Status - CPU: {cpu_percent}%, "
                                                f"Memory: {memory.percent}% (Used: {memory.used / 1024**3:.1f}GB/"
                                                f"Total: {memory.total / 1024**3:.1f}GB)",
                                                )
                                                                        except Exception as e:
                                                    logger.error(f"Process optimization failed: {e}")
                                                else:
                                                    logger.info("Process optimization completed successfully")
                                                                                def main():
                                                        """try:"""

                                                        optimizer = ProcessOptimizer("/opt/sutazaiapp")
                                                        optimizer.run_optimization()
                                                                                    except Exception as e:
                                                            logger.error(f"Main execution failed: {e}")
                                                            sys.exit(1)
                                                                                        if __name__ == "__main__":
                                                                main()

                                                                """""")""""""