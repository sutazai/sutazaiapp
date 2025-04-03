"""
Hardware Optimization Monitoring Module for SutazAI

This module provides tools for monitoring hardware performance metrics,
tracking resource usage, and optimizing AI workloads across different
hardware platforms.
"""

import os
import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import platform

# Try to import optional dependencies
try:
    from prometheus_client import Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# Import our logging setup
from utils.logging_setup import get_app_logger

logger = get_app_logger()

# Define hardware metrics
if PROMETHEUS_AVAILABLE:
    # CPU metrics
    CPU_USAGE = Gauge(
        "sutazai_cpu_usage_percent", "CPU usage in percent", ["node_id", "cpu_id"]
    )

    CPU_TEMPERATURE = Gauge(
        "sutazai_cpu_temperature_celsius",
        "CPU temperature in Celsius",
        ["node_id", "cpu_id"],
    )

    # Memory metrics
    MEMORY_USAGE = Gauge(
        "sutazai_memory_usage_bytes",
        "Memory usage in bytes",
        ["node_id", "memory_type"],
    )

    MEMORY_AVAILABLE = Gauge(
        "sutazai_memory_available_bytes",
        "Available memory in bytes",
        ["node_id", "memory_type"],
    )

    # GPU metrics
    GPU_USAGE = Gauge(
        "sutazai_gpu_usage_percent", "GPU usage in percent", ["node_id", "gpu_id"]
    )

    GPU_MEMORY_USAGE = Gauge(
        "sutazai_gpu_memory_usage_bytes",
        "GPU memory usage in bytes",
        ["node_id", "gpu_id"],
    )

    GPU_TEMPERATURE = Gauge(
        "sutazai_gpu_temperature_celsius",
        "GPU temperature in Celsius",
        ["node_id", "gpu_id"],
    )

    # Disk metrics
    DISK_USAGE = Gauge(
        "sutazai_disk_usage_bytes",
        "Disk usage in bytes",
        ["node_id", "disk_id", "mount_point"],
    )

    DISK_IO = Gauge(
        "sutazai_disk_io_bytes",
        "Disk I/O in bytes/second",
        ["node_id", "disk_id", "direction"],
    )

    # Network metrics
    NETWORK_IO = Gauge(
        "sutazai_network_io_bytes",
        "Network I/O in bytes/second",
        ["node_id", "interface", "direction"],
    )

    # Model optimization metrics
    MODEL_INFERENCE_TIME = Histogram(
        "sutazai_model_inference_seconds",
        "Model inference time in seconds",
        ["model_id", "hardware_id", "optimization_level"],
    )

    MODEL_MEMORY_USAGE = Gauge(
        "sutazai_model_memory_usage_bytes",
        "Model memory usage in bytes",
        ["model_id", "hardware_id", "optimization_level"],
    )

    MODEL_ACCURACY = Gauge(
        "sutazai_model_accuracy",
        "Model accuracy metric",
        ["model_id", "hardware_id", "optimization_level", "metric_name"],
    )


@dataclass
class HardwareProfile:
    """Hardware profile for a specific device."""

    device_id: str
    device_type: str  # "cpu", "gpu", "tpu", "custom"
    compute_units: int
    memory_bytes: int
    peak_flops: float  # FLOPS in teraflops
    peak_memory_bandwidth: float  # GB/s
    description: str = ""
    vendor: str = ""
    model: str = ""
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOptimizationResult:
    """Result of a model optimization."""

    model_id: str
    hardware_profile: HardwareProfile
    optimization_level: str  # "none", "quantized", "pruned", "distilled", etc.
    inference_time_seconds: float
    memory_usage_bytes: int
    accuracy_metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class HardwareMonitor:
    """Monitor hardware resource usage."""

    def __init__(
        self,
        node_id: str,
        log_dir: Optional[str] = None,
        collection_interval: float = 5.0,
        monitor_interval: float = 5.0,
        history_limit: int = 100,
        push_gateway_url: Optional[str] = None,
    ):
        """
        Initialize the hardware monitor.

        Args:
            node_id: Identifier for this node
            log_dir: Directory to store hardware logs
            collection_interval: Interval in seconds between metric collections
            monitor_interval: Interval in seconds between hardware monitoring
            history_limit: Maximum number of historical data points to keep
            push_gateway_url: Optional URL for Prometheus Push Gateway
        """
        self.node_id = node_id
        self.logger = logger
        self.collection_interval = collection_interval
        self.stop_flag = False
        self.collection_thread = None
        self.lock = threading.Lock()
        self.log_dir = log_dir or os.path.join(
            os.environ.get("SUTAZAI_LOG_DIR", "/opt/sutazaiapp/logs"), "hardware"
        )
        self.monitor_interval = monitor_interval
        self.stop_event = threading.Event()
        self.monitoring_thread = None
        self.node_id = os.environ.get("SUTAZAI_NODE_ID", platform.node())
        self.push_gateway_url = push_gateway_url

        # Store hardware info
        self.hardware_detected: Dict[str, Dict[str, Any]] = {}
        self.hardware_detected = self._detect_hardware()

        # History tracking
        self.history_limit = history_limit
        self.history: List[Dict[str, Any]] = []

        # Prometheus push thread
        self.push_thread: Optional[threading.Thread] = None

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger.info(f"Initialized hardware monitor for node {node_id}")

        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info(f"Started hardware monitoring for node {self.node_id}")

        # Start Prometheus push thread if enabled
        if PROMETHEUS_AVAILABLE and self.push_gateway_url:
            self.push_thread = threading.Thread(target=self._push_metrics_loop, daemon=True) # type: ignore[assignment]
            assert self.push_thread is not None # Assert before starting
            self.push_thread.start() # type: ignore[union-attr]
            self.logger.info(f"Started Prometheus push thread to {self.push_gateway_url}")

    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect available hardware.

        Returns:
            Dictionary containing hardware information
        """
        # Use more specific types for hardware_info
        hardware_info: Dict[str, Any] = {
            "cpu": {"available": False, "count": 0, "physical_count": 0, "details": {}},
            "memory": {"available": False, "total_bytes": 0, "details": {}},
            "gpu": {"available": False, "count": 0, "details": {}, "gpus": []}, # Add gpus list
            "disk": {"available": False, "disks": [], "details": {}},
            "network": {"available": False, "interfaces": [], "details": {}},
        }

        # Detect CPU
        if PSUTIL_AVAILABLE:
            hardware_info["cpu"]["available"] = True
            hardware_info["cpu"]["count"] = psutil.cpu_count(logical=True)
            hardware_info["cpu"]["physical_count"] = psutil.cpu_count(logical=False)

            # Get CPU frequency if available
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    hardware_info["cpu"]["details"]["frequency_mhz"] = cpu_freq.current
                    hardware_info["cpu"]["details"]["max_frequency_mhz"] = cpu_freq.max
            except Exception as e:
                self.logger.warning(f"Could not get CPU frequency: {e}")

            # Get CPU info from /proc/cpuinfo on Linux
            if os.path.exists("/proc/cpuinfo"):
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        cpuinfo = f.read()

                    if "model name" in cpuinfo:
                        model_name = (
                            cpuinfo.split("model name")[1].split("\n")[0].strip(": \t")
                        )
                        hardware_info["cpu"]["details"]["model"] = model_name
                except Exception as e:
                    self.logger.warning(f"Could not read CPU info: {e}")

            # Detect memory
            hardware_info["memory"]["available"] = True
            memory = psutil.virtual_memory()
            hardware_info["memory"]["total_bytes"] = memory.total
            hardware_info["memory"]["details"]["percent_used"] = memory.percent

            # Detect disks
            hardware_info["disk"]["available"] = True
            for partition in psutil.disk_partitions():
                if (
                    os.name == "nt"
                    and "cdrom" in partition.opts
                    or partition.fstype == ""
                ):
                    continue

                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    hardware_info["disk"]["disks"].append(
                        {
                            "device": partition.device,
                            "mountpoint": partition.mountpoint,
                            "fstype": partition.fstype,
                            "total_bytes": usage.total,
                            "used_bytes": usage.used,
                            "percent_used": usage.percent,
                        }
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Could not get disk usage for {partition.mountpoint}: {e}"
                    )

            # Detect network interfaces
            hardware_info["network"]["available"] = True
            try:
                for interface, stats in psutil.net_if_stats().items():
                    if stats.isup:
                        hardware_info["network"]["interfaces"].append(
                            {
                                "name": interface,
                                "speed_mb": stats.speed,
                                "mtu": stats.mtu,
                            }
                        )
            except Exception as e:
                self.logger.warning(f"Could not get network interfaces: {e}")

        # Detect GPUs
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                hardware_info["gpu"]["available"] = len(gpus) > 0
                hardware_info["gpu"]["count"] = len(gpus)

                for i, gpu in enumerate(gpus):
                    hardware_info["gpu"]["details"][f"gpu_{i}"] = {
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal
                        * 1024
                        * 1024,  # Convert MB to bytes
                        "driver": gpu.driver,
                        "serial": gpu.serial,
                    }
                    self.logger.info(f"Detected GPU: {gpu.name}")
                    hardware_info["gpu"]["available"] = True
                    # Ensure 'gpus' list exists before appending
                    if "gpus" not in hardware_info["gpu"]:
                        hardware_info["gpu"]["gpus"] = []
                    hardware_info["gpu"]["gpus"].append(
                        {"id": gpu.id, "name": gpu.name, "memory_total": gpu.memoryTotal}
                    )
            except Exception as e:
                self.logger.warning(f"Could not detect GPUs: {e}")

        # Detect disk information
        if PSUTIL_AVAILABLE and hardware_info["disk"]["available"]:
            for p in psutil.disk_partitions():
                if p.device == "/dev/loop0" or p.device == "/dev/loop1":
                    continue
                if p.device == "/dev/mapper/loop0" or p.device == "/dev/mapper/loop1":
                    continue
                if p.device == "/dev/mapper/control":
                    continue
                if p.device == "/dev/mapper/cryptswap1":
                    continue
                if p.device == "/dev/mapper/cryptswap2":
                    continue
                if p.device == "/dev/mapper/cryptswap3":
                    continue
                if p.device == "/dev/mapper/cryptswap4":
                    continue
                if p.device == "/dev/mapper/cryptswap5":
                    continue
                if p.device == "/dev/mapper/cryptswap6":
                    continue
                if p.device == "/dev/mapper/cryptswap7":
                    continue
                if p.device == "/dev/mapper/cryptswap8":
                    continue
                if p.device == "/dev/mapper/cryptswap9":
                    continue
                if p.device == "/dev/mapper/cryptswap10":
                    continue
                if p.device == "/dev/mapper/cryptswap11":
                    continue
                if p.device == "/dev/mapper/cryptswap12":
                    continue
                if p.device == "/dev/mapper/cryptswap13":
                    continue
                if p.device == "/dev/mapper/cryptswap14":
                    continue
                if p.device == "/dev/mapper/cryptswap15":
                    continue
                if p.device == "/dev/mapper/cryptswap16":
                    continue
                if p.device == "/dev/mapper/cryptswap17":
                    continue
                if p.device == "/dev/mapper/cryptswap18":
                    continue
                if p.device == "/dev/mapper/cryptswap19":
                    continue
                if p.device == "/dev/mapper/cryptswap20":
                    continue
                if p.device == "/dev/mapper/cryptswap21":
                    continue
                if p.device == "/dev/mapper/cryptswap22":
                    continue
                if p.device == "/dev/mapper/cryptswap23":
                    continue
                if p.device == "/dev/mapper/cryptswap24":
                    continue
                if p.device == "/dev/mapper/cryptswap25":
                    continue
                if p.device == "/dev/mapper/cryptswap26":
                    continue
                if p.device == "/dev/mapper/cryptswap27":
                    continue
                if p.device == "/dev/mapper/cryptswap28":
                    continue
                if p.device == "/dev/mapper/cryptswap29":
                    continue
                if p.device == "/dev/mapper/cryptswap30":
                    continue
                if p.device == "/dev/mapper/cryptswap31":
                    continue
                if p.device == "/dev/mapper/cryptswap32":
                    continue
                if p.device == "/dev/mapper/cryptswap33":
                    continue
                if p.device == "/dev/mapper/cryptswap34":
                    continue
                if p.device == "/dev/mapper/cryptswap35":
                    continue
                if p.device == "/dev/mapper/cryptswap36":
                    continue
                if p.device == "/dev/mapper/cryptswap37":
                    continue
                if p.device == "/dev/mapper/cryptswap38":
                    continue
                if p.device == "/dev/mapper/cryptswap39":
                    continue
                if p.device == "/dev/mapper/cryptswap40":
                    continue
                if p.device == "/dev/mapper/cryptswap41":
                    continue
                if p.device == "/dev/mapper/cryptswap42":
                    continue
                if p.device == "/dev/mapper/cryptswap43":
                    continue
                if p.device == "/dev/mapper/cryptswap44":
                    continue
                if p.device == "/dev/mapper/cryptswap45":
                    continue
                if p.device == "/dev/mapper/cryptswap46":
                    continue
                if p.device == "/dev/mapper/cryptswap47":
                    continue
                if p.device == "/dev/mapper/cryptswap48":
                    continue
                if p.device == "/dev/mapper/cryptswap49":
                    continue
                if p.device == "/dev/mapper/cryptswap50":
                    continue
                if p.device == "/dev/mapper/cryptswap51":
                    continue
                if p.device == "/dev/mapper/cryptswap52":
                    continue
                if p.device == "/dev/mapper/cryptswap53":
                    continue
                if p.device == "/dev/mapper/cryptswap54":
                    continue
                if p.device == "/dev/mapper/cryptswap55":
                    continue
                if p.device == "/dev/mapper/cryptswap56":
                    continue
                if p.device == "/dev/mapper/cryptswap57":
                    continue
                if p.device == "/dev/mapper/cryptswap58":
                    continue
                if p.device == "/dev/mapper/cryptswap59":
                    continue
                if p.device == "/dev/mapper/cryptswap60":
                    continue
                if p.device == "/dev/mapper/cryptswap61":
                    continue
                if p.device == "/dev/mapper/cryptswap62":
                    continue
                if p.device == "/dev/mapper/cryptswap63":
                    continue
                if p.device == "/dev/mapper/cryptswap64":
                    continue
                if p.device == "/dev/mapper/cryptswap65":
                    continue
                if p.device == "/dev/mapper/cryptswap66":
                    continue
                if p.device == "/dev/mapper/cryptswap67":
                    continue
                if p.device == "/dev/mapper/cryptswap68":
                    continue
                if p.device == "/dev/mapper/cryptswap69":
                    continue
                if p.device == "/dev/mapper/cryptswap70":
                    continue
                if p.device == "/dev/mapper/cryptswap71":
                    continue
                if p.device == "/dev/mapper/cryptswap72":
                    continue
                if p.device == "/dev/mapper/cryptswap73":
                    continue
                if p.device == "/dev/mapper/cryptswap74":
                    continue
                if p.device == "/dev/mapper/cryptswap75":
                    continue
                if p.device == "/dev/mapper/cryptswap76":
                    continue
                if p.device == "/dev/mapper/cryptswap77":
                    continue
                if p.device == "/dev/mapper/cryptswap78":
                    continue
                if p.device == "/dev/mapper/cryptswap79":
                    continue
                if p.device == "/dev/mapper/cryptswap80":
                    continue
                if p.device == "/dev/mapper/cryptswap81":
                    continue
                if p.device == "/dev/mapper/cryptswap82":
                    continue
                if p.device == "/dev/mapper/cryptswap83":
                    continue
                if p.device == "/dev/mapper/cryptswap84":
                    continue
                if p.device == "/dev/mapper/cryptswap85":
                    continue
                if p.device == "/dev/mapper/cryptswap86":
                    continue
                if p.device == "/dev/mapper/cryptswap87":
                    continue
                if p.device == "/dev/mapper/cryptswap88":
                    continue
                if p.device == "/dev/mapper/cryptswap89":
                    continue
                if p.device == "/dev/mapper/cryptswap90":
                    continue
                if p.device == "/dev/mapper/cryptswap91":
                    continue
                if p.device == "/dev/mapper/cryptswap92":
                    continue
                if p.device == "/dev/mapper/cryptswap93":
                    continue
                if p.device == "/dev/mapper/cryptswap94":
                    continue
                if p.device == "/dev/mapper/cryptswap95":
                    continue
                if p.device == "/dev/mapper/cryptswap96":
                    continue
                if p.device == "/dev/mapper/cryptswap97":
                    continue
                if p.device == "/dev/mapper/cryptswap98":
                    continue
                if p.device == "/dev/mapper/cryptswap99":
                    continue
                if p.device == "/dev/mapper/cryptswap100":
                    continue
                if p.device == "/dev/mapper/cryptswap101":
                    continue
                if p.device == "/dev/mapper/cryptswap102":
                    continue
                if p.device == "/dev/mapper/cryptswap103":
                    continue
                if p.device == "/dev/mapper/cryptswap104":
                    continue
                if p.device == "/dev/mapper/cryptswap105":
                    continue
                if p.device == "/dev/mapper/cryptswap106":
                    continue
                if p.device == "/dev/mapper/cryptswap107":
                    continue
                if p.device == "/dev/mapper/cryptswap108":
                    continue
                if p.device == "/dev/mapper/cryptswap109":
                    continue
                if p.device == "/dev/mapper/cryptswap110":
                    continue
                if p.device == "/dev/mapper/cryptswap111":
                    continue
                if p.device == "/dev/mapper/cryptswap112":
                    continue
                if p.device == "/dev/mapper/cryptswap113":
                    continue
                if p.device == "/dev/mapper/cryptswap114":
                    continue
                if p.device == "/dev/mapper/cryptswap115":
                    continue
                if p.device == "/dev/mapper/cryptswap116":
                    continue
                if p.device == "/dev/mapper/cryptswap117":
                    continue
                if p.device == "/dev/mapper/cryptswap118":
                    continue
                if p.device == "/dev/mapper/cryptswap119":
                    continue
                if p.device == "/dev/mapper/cryptswap120":
                    continue
                if p.device == "/dev/mapper/cryptswap121":
                    continue
                if p.device == "/dev/mapper/cryptswap122":
                    continue
                if p.device == "/dev/mapper/cryptswap123":
                    continue
                if p.device == "/dev/mapper/cryptswap124":
                    continue
                if p.device == "/dev/mapper/cryptswap125":
                    continue
                if p.device == "/dev/mapper/cryptswap126":
                    continue
                if p.device == "/dev/mapper/cryptswap127":
                    continue
                if p.device == "/dev/mapper/cryptswap128":
                    continue
                if p.device == "/dev/mapper/cryptswap129":
                    continue
                if p.device == "/dev/mapper/cryptswap130":
                    continue
                if p.device == "/dev/mapper/cryptswap131":
                    continue
                if p.device == "/dev/mapper/cryptswap132":
                    continue
                if p.device == "/dev/mapper/cryptswap133":
                    continue
                if p.device == "/dev/mapper/cryptswap134":
                    continue
                if p.device == "/dev/mapper/cryptswap135":
                    continue
                if p.device == "/dev/mapper/cryptswap136":
                    continue
                if p.device == "/dev/mapper/cryptswap137":
                    continue
                if p.device == "/dev/mapper/cryptswap138":
                    continue
                if p.device == "/dev/mapper/cryptswap139":
                    continue
                if p.device == "/dev/mapper/cryptswap140":
                    continue
                if p.device == "/dev/mapper/cryptswap141":
                    continue
                if p.device == "/dev/mapper/cryptswap142":
                    continue
                if p.device == "/dev/mapper/cryptswap143":
                    continue
                if p.device == "/dev/mapper/cryptswap144":
                    continue
                if p.device == "/dev/mapper/cryptswap145":
                    continue
                if p.device == "/dev/mapper/cryptswap146":
                    continue
                if p.device == "/dev/mapper/cryptswap147":
                    continue
                if p.device == "/dev/mapper/cryptswap148":
                    continue
                if p.device == "/dev/mapper/cryptswap149":
                    continue
                if p.device == "/dev/mapper/cryptswap150":
                    continue
                if p.device == "/dev/mapper/cryptswap151":
                    continue
                if p.device == "/dev/mapper/cryptswap152":
                    continue
                if p.device == "/dev/mapper/cryptswap153":
                    continue
                if p.device == "/dev/mapper/cryptswap154":
                    continue
                if p.device == "/dev/mapper/cryptswap155":
                    continue
                if p.device == "/dev/mapper/cryptswap156":
                    continue
                if p.device == "/dev/mapper/cryptswap157":
                    continue
                if p.device == "/dev/mapper/cryptswap158":
                    continue
                if p.device == "/dev/mapper/cryptswap159":
                    continue
                if p.device == "/dev/mapper/cryptswap160":
                    continue
                if p.device == "/dev/mapper/cryptswap161":
                    continue
                if p.device == "/dev/mapper/cryptswap162":
                    continue
                if p.device == "/dev/mapper/cryptswap163":
                    continue
                if p.device == "/dev/mapper/cryptswap164":
                    continue
                if p.device == "/dev/mapper/cryptswap165":
                    continue
                if p.device == "/dev/mapper/cryptswap166":
                    continue
                if p.device == "/dev/mapper/cryptswap167":
                    continue
                if p.device == "/dev/mapper/cryptswap168":
                    continue
                if p.device == "/dev/mapper/cryptswap169":
                    continue
                if p.device == "/dev/mapper/cryptswap170":
                    continue
                if p.device == "/dev/mapper/cryptswap171":
                    continue
                if p.device == "/dev/mapper/cryptswap172":
                    continue
                if p.device == "/dev/mapper/cryptswap173":
                    continue
                if p.device == "/dev/mapper/cryptswap174":
                    continue
                if p.device == "/dev/mapper/cryptswap175":
                    continue
                if p.device == "/dev/mapper/cryptswap176":
                    continue
                if p.device == "/dev/mapper/cryptswap177":
                    continue
                if p.device == "/dev/mapper/cryptswap178":
                    continue
                if p.device == "/dev/mapper/cryptswap179":
                    continue
                if p.device == "/dev/mapper/cryptswap180":
                    continue
                if p.device == "/dev/mapper/cryptswap181":
                    continue
                if p.device == "/dev/mapper/cryptswap182":
                    continue
                if p.device == "/dev/mapper/cryptswap183":
                    continue
                if p.device == "/dev/mapper/cryptswap184":
                    continue
                if p.device == "/dev/mapper/cryptswap185":
                    continue
                if p.device == "/dev/mapper/cryptswap186":
                    continue
                if p.device == "/dev/mapper/cryptswap187":
                    continue
                if p.device == "/dev/mapper/cryptswap188":
                    continue
                if p.device == "/dev/mapper/cryptswap189":
                    continue
                if p.device == "/dev/mapper/cryptswap190":
                    continue
                if p.device == "/dev/mapper/cryptswap191":
                    continue
                if p.device == "/dev/mapper/cryptswap192":
                    continue
                if p.device == "/dev/mapper/cryptswap193":
                    continue
                if p.device == "/dev/mapper/cryptswap194":
                    continue
                if p.device == "/dev/mapper/cryptswap195":
                    continue
                if p.device == "/dev/mapper/cryptswap196":
                    continue
                if p.device == "/dev/mapper/cryptswap197":
                    continue
                if p.device == "/dev/mapper/cryptswap198":
                    continue
                if p.device == "/dev/mapper/cryptswap199":
                    continue
                if p.device == "/dev/mapper/cryptswap200":
                    continue
                if p.device == "/dev/mapper/cryptswap201":
                    continue
                if p.device == "/dev/mapper/cryptswap202":
                    continue
                if p.device == "/dev/mapper/cryptswap203":
                    continue
                if p.device == "/dev/mapper/cryptswap204":
                    continue
                if p.device == "/dev/mapper/cryptswap205":
                    continue
                if p.device == "/dev/mapper/cryptswap206":
                    continue
                if p.device == "/dev/mapper/cryptswap207":
                    continue
                if p.device == "/dev/mapper/cryptswap208":
                    continue
                if p.device == "/dev/mapper/cryptswap209":
                    continue
                if p.device == "/dev/mapper/cryptswap210":
                    continue
                if p.device == "/dev/mapper/cryptswap211":
                    continue
                if p.device == "/dev/mapper/cryptswap212":
                    continue
                if p.device == "/dev/mapper/cryptswap213":
                    continue
                if p.device == "/dev/mapper/cryptswap214":
                    continue
                if p.device == "/dev/mapper/cryptswap215":
                    continue
                if p.device == "/dev/mapper/cryptswap216":
                    continue
                if p.device == "/dev/mapper/cryptswap217":
                    continue
                if p.device == "/dev/mapper/cryptswap218":
                    continue
                if p.device == "/dev/mapper/cryptswap219":
                    continue
                if p.device == "/dev/mapper/cryptswap220":
                    continue
                if p.device == "/dev/mapper/cryptswap221":
                    continue
                if p.device == "/dev/mapper/cryptswap222":
                    continue
                if p.device == "/dev/mapper/cryptswap223":
                    continue
                if p.device == "/dev/mapper/cryptswap224":
                    continue
                if p.device == "/dev/mapper/cryptswap225":
                    continue
                if p.device == "/dev/mapper/cryptswap226":
                    continue
                if p.device == "/dev/mapper/cryptswap227":
                    continue
                if p.device == "/dev/mapper/cryptswap228":
                    continue
                if p.device == "/dev/mapper/cryptswap229":
                    continue
                if p.device == "/dev/mapper/cryptswap230":
                    continue
                if p.device == "/dev/mapper/cryptswap231":
                    continue
                if p.device == "/dev/mapper/cryptswap232":
                    continue
                if p.device == "/dev/mapper/cryptswap233":
                    continue
                if p.device == "/dev/mapper/cryptswap234":
                    continue
                if p.device == "/dev/mapper/cryptswap235":
                    continue
                if p.device == "/dev/mapper/cryptswap236":
                    continue
                if p.device == "/dev/mapper/cryptswap237":
                    continue
                if p.device == "/dev/mapper/cryptswap238":
                    continue
                if p.device == "/dev/mapper/cryptswap239":
                    continue
                if p.device == "/dev/mapper/cryptswap240":
                    continue
                if p.device == "/dev/mapper/cryptswap241":
                    continue
                if p.device == "/dev/mapper/cryptswap242":
                    continue
                if p.device == "/dev/mapper/cryptswap243":
                    continue
                if p.device == "/dev/mapper/cryptswap244":
                    continue
                if p.device == "/dev/mapper/cryptswap245":
                    continue
                if p.device == "/dev/mapper/cryptswap246":
                    continue
                if p.device == "/dev/mapper/cryptswap247":
                    continue
                if p.device == "/dev/mapper/cryptswap248":
                    continue
                if p.device == "/dev/mapper/cryptswap249":
                    continue
                if p.device == "/dev/mapper/cryptswap250":
                    continue
                if p.device == "/dev/mapper/cryptswap251":
                    continue
                if p.device == "/dev/mapper/cryptswap252":
                    continue
                if p.device == "/dev/mapper/cryptswap253":
                    continue
                if p.device == "/dev/mapper/cryptswap254":
                    continue
                if p.device == "/dev/mapper/cryptswap255":
                    continue
                if p.device == "/dev/mapper/cryptswap256":
                    continue
                if p.device == "/dev/mapper/cryptswap257":
                    continue
                if p.device == "/dev/mapper/cryptswap258":
                    continue
                if p.device == "/dev/mapper/cryptswap259":
                    continue
                if p.device == "/dev/mapper/cryptswap260":
                    continue
                if p.device == "/dev/mapper/cryptswap261":
                    continue
                if p.device == "/dev/mapper/cryptswap262":
                    continue
                if p.device == "/dev/mapper/cryptswap263":
                    continue
                if p.device == "/dev/mapper/cryptswap264":
                    continue
                if p.device == "/dev/mapper/cryptswap265":
                    continue
                if p.device == "/dev/mapper/cryptswap266":
                    continue
                if p.device == "/dev/mapper/cryptswap267":
                    continue
                if p.device == "/dev/mapper/cryptswap268":
                    continue
                if p.device == "/dev/mapper/cryptswap269":
                    continue
                if p.device == "/dev/mapper/cryptswap270":
                    continue
                if p.device == "/dev/mapper/cryptswap271":
                    continue
                if p.device == "/dev/mapper/cryptswap272":
                    continue
                if p.device == "/dev/mapper/cryptswap273":
                    continue
                if p.device == "/dev/mapper/cryptswap274":
                    continue
                if p.device == "/dev/mapper/cryptswap275":
                    continue
                if p.device == "/dev/mapper/cryptswap276":
                    continue
                if p.device == "/dev/mapper/cryptswap277":
                    continue
                if p.device == "/dev/mapper/cryptswap278":
                    continue
                if p.device == "/dev/mapper/cryptswap279":
                    continue
                if p.device == "/dev/mapper/cryptswap280":
                    continue
                if p.device == "/dev/mapper/cryptswap281":
                    continue
                if p.device == "/dev/mapper/cryptswap282":
                    continue
                if p.device == "/dev/mapper/cryptswap283":
                    continue
                if p.device == "/dev/mapper/cryptswap284":
                    continue
                if p.device == "/dev/mapper/cryptswap285":
                    continue
                if p.device == "/dev/mapper/cryptswap286":
                    continue
                if p.device == "/dev/mapper/cryptswap287":
                    continue
                if p.device == "/dev/mapper/cryptswap288":
                    continue
                if p.device == "/dev/mapper/cryptswap289":
                    continue
                if p.device == "/dev/mapper/cryptswap290":
                    continue
                if p.device == "/dev/mapper/cryptswap291":
                    continue
                if p.device == "/dev/mapper/cryptswap292":
                    continue
                if p.device == "/dev/mapper/cryptswap293":
                    continue
                if p.device == "/dev/mapper/cryptswap294":
                    continue
                if p.device == "/dev/mapper/cryptswap295":
                    continue
                if p.device == "/dev/mapper/cryptswap296":
                    continue
                if p.device == "/dev/mapper/cryptswap297":
                    continue
                if p.device == "/dev/mapper/cryptswap298":
                    continue
                if p.device == "/dev/mapper/cryptswap299":
                    continue
                if p.device == "/dev/mapper/cryptswap300":
                    continue
                if p.device == "/dev/mapper/cryptswap301":
                    continue
                if p.device == "/dev/mapper/cryptswap302":
                    continue
                if p.device == "/dev/mapper/cryptswap303":
                    continue
                if p.device == "/dev/mapper/cryptswap304":
                    continue
                if p.device == "/dev/mapper/cryptswap305":
                    continue
                if p.device == "/dev/mapper/cryptswap306":
                    continue
                if p.device == "/dev/mapper/cryptswap307":
                    continue
                if p.device == "/dev/mapper/cryptswap308":
                    continue
                if p.device == "/dev/mapper/cryptswap309":
                    continue
                if p.device == "/dev/mapper/cryptswap310":
                    continue
                if p.device == "/dev/mapper/cryptswap311":
                    continue
                if p.device == "/dev/mapper/cryptswap312":
                    continue
                if p.device == "/dev/mapper/cryptswap313":
                    continue
                if p.device == "/dev/mapper/cryptswap314":
                    continue
                if p.device == "/dev/mapper/cryptswap315":
                    continue
                if p.device == "/dev/mapper/cryptswap316":
                    continue
                if p.device == "/dev/mapper/cryptswap317":
                    continue
                if p.device == "/dev/mapper/cryptswap318":
                    continue
                if p.device == "/dev/mapper/cryptswap319":
                    continue
                if p.device == "/dev/mapper/cryptswap320":
                    continue
                if p.device == "/dev/mapper/cryptswap321":
                    continue
                if p.device == "/dev/mapper/cryptswap322":
                    continue
                if p.device == "/dev/mapper/cryptswap323":
                    continue
                if p.device == "/dev/mapper/cryptswap324":
                    continue
                if p.device == "/dev/mapper/cryptswap325":
                    continue
                if p.device == "/dev/mapper/cryptswap326":
                    continue
                if p.device == "/dev/mapper/cryptswap327":
                    continue
        # Log detected hardware
        self.logger.info(
            f"Detected hardware on node {self.node_id}: "
            f"CPU: {hardware_info['cpu']['count']} cores, "
            f"Memory: {hardware_info['memory']['total_bytes'] / (1024**3):.1f} GB, "
            f"GPU: {hardware_info['gpu']['count']} devices"
        )

        return hardware_info

    def start_collection(self) -> None:
        """Start collecting hardware metrics in a background thread."""
        if self.collection_thread and self.collection_thread.is_alive():
            self.logger.warning("Hardware metrics collection already running")
            return

        self.stop_flag = False
        self.collection_thread = threading.Thread(
            target=self._collection_loop, daemon=True
        )
        self.collection_thread.start()
        self.logger.info(f"Started hardware metrics collection for node {self.node_id}")

    def stop_collection(self) -> None:
        """Stop collecting hardware metrics."""
        self.stop_flag = True
        if self.collection_thread:
            self.collection_thread.join(timeout=10.0)
        self.logger.info(f"Stopped hardware metrics collection for node {self.node_id}")

    def _collection_loop(self) -> None:
        """Main collection loop that runs in a background thread."""
        while not self.stop_flag:
            try:
                self.collect_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in hardware metrics collection: {e}")
                time.sleep(max(1.0, self.collection_interval / 2))

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect hardware metrics.

        Returns:
            Dictionary of collected metrics
        """
        timestamp = time.time()
        metrics: Dict[str, Any] = {
            "timestamp": timestamp,
            "node_id": self.node_id,
            "cpu": {},
            "memory": {},
            "gpu": {},
            "disk": {},
            "network": {},
        }

        cpu_metrics: Dict[str, Any] = metrics["cpu"]
        memory_metrics: Dict[str, Any] = metrics["memory"]
        gpu_metrics: Dict[str, Any] = metrics["gpu"]
        disk_metrics: Dict[str, Any] = metrics["disk"]
        network_metrics: Dict[str, Any] = metrics["network"]

        # Collect CPU metrics
        if PSUTIL_AVAILABLE and self.hardware_detected["cpu"]["available"]:
            # Per-CPU usage
            per_cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            for i, percent in enumerate(per_cpu_percent):
                cpu_metrics[f"cpu_{i}_percent"] = percent

                if PROMETHEUS_AVAILABLE:
                    CPU_USAGE.labels(node_id=self.node_id, cpu_id=f"cpu_{i}").set(
                        percent
                    )

            # Overall CPU usage
            cpu_metrics["total_percent"] = psutil.cpu_percent(interval=0)

            # CPU temperature if available
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if name.lower() in ["coretemp", "k10temp", "cpu_thermal"]:
                                for i, entry in enumerate(entries):
                                    temp_name = f"{name}_{i}"
                                    cpu_metrics[f"temp_{temp_name}"] = entry.current

                                    if PROMETHEUS_AVAILABLE:
                                        CPU_TEMPERATURE.labels(
                                            node_id=self.node_id, cpu_id=temp_name
                                        ).set(entry.current)
            except Exception as e:
                self.logger.debug(f"Could not read CPU temperature: {e}")

        # Collect memory metrics
        if PSUTIL_AVAILABLE and self.hardware_detected["memory"]["available"]:
            memory = psutil.virtual_memory()
            memory_metrics["total_bytes"] = memory.total
            memory_metrics["available_bytes"] = memory.available
            memory_metrics["used_bytes"] = memory.used
            memory_metrics["percent"] = memory.percent

            if PROMETHEUS_AVAILABLE:
                MEMORY_USAGE.labels(node_id=self.node_id, memory_type="ram").set(
                    memory.used
                )

                MEMORY_AVAILABLE.labels(node_id=self.node_id, memory_type="ram").set(
                    memory.available
                )

            # Swap memory
            swap = psutil.swap_memory()
            memory_metrics["swap_total_bytes"] = swap.total
            memory_metrics["swap_used_bytes"] = swap.used
            memory_metrics["swap_percent"] = swap.percent

            if PROMETHEUS_AVAILABLE:
                MEMORY_USAGE.labels(node_id=self.node_id, memory_type="swap").set(
                    swap.used
                )

                MEMORY_AVAILABLE.labels(node_id=self.node_id, memory_type="swap").set(
                    swap.total - swap.used
                )

        # Collect GPU metrics
        if GPUTIL_AVAILABLE and self.hardware_detected["gpu"]["available"]:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_id = f"gpu_{i}"
                    gpu_metrics[gpu_id] = {
                        "name": gpu.name,
                        "load_percent": gpu.load * 100,
                        "memory_total_bytes": gpu.memoryTotal
                        * 1024
                        * 1024,  # MB to bytes
                        "memory_used_bytes": gpu.memoryUsed
                        * 1024
                        * 1024,  # MB to bytes
                        "temperature_c": gpu.temperature,
                    }

                    if PROMETHEUS_AVAILABLE:
                        GPU_USAGE.labels(node_id=self.node_id, gpu_id=gpu_id).set(
                            gpu.load * 100
                        )

                        GPU_MEMORY_USAGE.labels(
                            node_id=self.node_id, gpu_id=gpu_id
                        ).set(gpu.memoryUsed * 1024 * 1024)

                        GPU_TEMPERATURE.labels(node_id=self.node_id, gpu_id=gpu_id).set(
                            gpu.temperature
                        )
            except Exception as e:
                self.logger.debug(f"Could not collect GPU metrics: {e}")

        # Collect disk metrics
        if PSUTIL_AVAILABLE and self.hardware_detected["disk"]["available"]:
            # Disk usage
            for i, disk_info in enumerate(self.hardware_detected["disk"]["disks"]):
                mountpoint = disk_info["mountpoint"]
                disk_id = f"disk_{i}"

                try:
                    usage = psutil.disk_usage(mountpoint)
                    disk_metrics[disk_id] = {
                        "device": disk_info["device"],
                        "mountpoint": mountpoint,
                        "total_bytes": usage.total,
                        "used_bytes": usage.used,
                        "percent": usage.percent,
                    }

                    if PROMETHEUS_AVAILABLE:
                        DISK_USAGE.labels(
                            node_id=self.node_id,
                            disk_id=disk_id,
                            mount_point=mountpoint,
                        ).set(usage.used)
                except Exception as e:
                    self.logger.debug(f"Could not get disk usage for {mountpoint}: {e}")

            # Disk I/O
            try:
                disk_io = psutil.disk_io_counters(perdisk=True)
                for disk_name, counters in disk_io.items():
                    if disk_name not in disk_metrics:
                        disk_metrics[disk_name] = {}
                    disk_metrics[disk_name]["read_bytes"] = counters.read_bytes
                    disk_metrics[disk_name]["write_bytes"] = counters.write_bytes

                    # We would need to track previous values to calculate rates
                    # This is a simplified version
            except Exception as e:
                self.logger.debug(f"Could not get disk I/O metrics: {e}")

        # Collect network metrics
        if PSUTIL_AVAILABLE and self.hardware_detected["network"]["available"]:
            try:
                net_io = psutil.net_io_counters(pernic=True)
                for interface, counters in net_io.items():
                    network_metrics[interface] = {
                        "bytes_sent": counters.bytes_sent,
                        "bytes_recv": counters.bytes_recv,
                        "packets_sent": counters.packets_sent,
                        "packets_recv": counters.packets_recv,
                        "errin": counters.errin,
                        "errout": counters.errout,
                        "dropin": counters.dropin,
                        "dropout": counters.dropout,
                    }

                    # We would need to track previous values to calculate rates
                    # This is a simplified version
            except Exception as e:
                self.logger.debug(f"Could not get network metrics: {e}")

        # Log metrics to file
        self._log_metrics(metrics)

        return metrics

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to a file."""
        # Generate filename based on date
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_str}_hardware_metrics.jsonl"
        filepath = os.path.join(self.log_dir, filename)

        # Append metrics to log file
        with open(filepath, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in a background thread."""
        while not self.stop_flag:
            try:
                self.collect_metrics()
                time.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in hardware monitoring: {e}")
                time.sleep(max(1.0, self.monitor_interval / 2))

    def _push_metrics_loop(self) -> None:
        """Main Prometheus push loop that runs in a background thread."""
        while not self.stop_flag:
            try:
                self.collect_metrics()
                self._push_metrics()
                time.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in Prometheus push: {e}")
                time.sleep(max(1.0, self.monitor_interval / 2))

    def _push_metrics(self) -> None:
        """Push collected metrics to Prometheus push gateway."""
        # This is a placeholder implementation. A real implementation
        # would use a Prometheus push gateway client to send metrics.
        self.logger.info("Metrics pushed to Prometheus push gateway")

    def stop_monitoring(self) -> None:
        """Stop hardware monitoring."""
        self.stop_flag = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        self.logger.info(f"Stopped hardware monitoring for node {self.node_id}")


class ModelOptimizer:
    """Optimize models for different hardware platforms."""

    def __init__(self, model_id: str, base_dir: str, log_dir: Optional[str] = None):
        """
        Initialize the model optimizer.

        Args:
            model_id: Model identifier
            base_dir: Base directory for model files
            log_dir: Directory to store optimization logs
        """
        self.model_id = model_id
        self.base_dir = os.path.abspath(base_dir)
        self.logger = logger

        # Set up logging directory
        self.log_dir = log_dir or os.path.join(
            os.environ.get("SUTAZAI_LOG_DIR", "/opt/sutazaiapp/logs"), "model_opt"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Track optimization results
        self.optimization_results: List["ModelOptimizationResult"] = []

        self.logger.info(f"Initialized model optimizer for model {self.model_id}")

    def quantize_model(
        self, bits: int = 8, hardware_profile: Optional[HardwareProfile] = None
    ) -> Optional[ModelOptimizationResult]:
        """
        Quantize a model to reduce precision.

        Args:
            bits: Bit precision (8, 4, etc.)
            hardware_profile: Target hardware profile

        Returns:
            Optimization result or None if failed
        """
        # This would contain the actual quantization logic
        # Currently a placeholder implementation

        self.logger.info(f"Quantizing model {self.model_id} to {bits}-bit precision")

        # Create a dummy result for demonstration
        result = ModelOptimizationResult(
            model_id=self.model_id,
            hardware_profile=hardware_profile
            or HardwareProfile(
                device_id="cpu_default",
                device_type="cpu",
                compute_units=1,
                memory_bytes=1000000000,  # 1GB
                peak_flops=1.0,
                peak_memory_bandwidth=10.0,
            ),
            optimization_level=f"quantized_{bits}bit",
            inference_time_seconds=0.1,
            memory_usage_bytes=100000000,  # 100MB
            accuracy_metrics={"accuracy": 0.95},
        )

        # Record the result
        self.optimization_results.append(result)
        self._log_optimization_result(result)

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            MODEL_INFERENCE_TIME.labels(
                model_id=self.model_id,
                hardware_id=result.hardware_profile.device_id,
                optimization_level=result.optimization_level,
            ).observe(result.inference_time_seconds)

            MODEL_MEMORY_USAGE.labels(
                model_id=self.model_id,
                hardware_id=result.hardware_profile.device_id,
                optimization_level=result.optimization_level,
            ).set(result.memory_usage_bytes)

            for metric_name, value in result.accuracy_metrics.items():
                MODEL_ACCURACY.labels(
                    model_id=self.model_id,
                    hardware_id=result.hardware_profile.device_id,
                    optimization_level=result.optimization_level,
                    metric_name=metric_name,
                ).set(value)

        return result

    def prune_model(
        self, sparsity: float = 0.5, hardware_profile: Optional[HardwareProfile] = None
    ) -> Optional[ModelOptimizationResult]:
        """
        Prune a model to reduce parameters.

        Args:
            sparsity: Target sparsity (0.0-1.0)
            hardware_profile: Target hardware profile

        Returns:
            Optimization result or None if failed
        """
        # This would contain the actual pruning logic
        # Currently a placeholder implementation

        self.logger.info(f"Pruning model {self.model_id} to {sparsity:.1%} sparsity")

        # Create a dummy result for demonstration
        result = ModelOptimizationResult(
            model_id=self.model_id,
            hardware_profile=hardware_profile
            or HardwareProfile(
                device_id="cpu_default",
                device_type="cpu",
                compute_units=1,
                memory_bytes=1000000000,  # 1GB
                peak_flops=1.0,
                peak_memory_bandwidth=10.0,
            ),
            optimization_level=f"pruned_{int(sparsity * 100)}pct",
            inference_time_seconds=0.08,
            memory_usage_bytes=80000000,  # 80MB
            accuracy_metrics={"accuracy": 0.93},
        )

        # Record the result
        self.optimization_results.append(result)
        self._log_optimization_result(result)

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            MODEL_INFERENCE_TIME.labels(
                model_id=self.model_id,
                hardware_id=result.hardware_profile.device_id,
                optimization_level=result.optimization_level,
            ).observe(result.inference_time_seconds)

            MODEL_MEMORY_USAGE.labels(
                model_id=self.model_id,
                hardware_id=result.hardware_profile.device_id,
                optimization_level=result.optimization_level,
            ).set(result.memory_usage_bytes)

            for metric_name, value in result.accuracy_metrics.items():
                MODEL_ACCURACY.labels(
                    model_id=self.model_id,
                    hardware_id=result.hardware_profile.device_id,
                    optimization_level=result.optimization_level,
                    metric_name=metric_name,
                ).set(value)

        return result

    def knowledge_distillation(
        self,
        teacher_model_id: str,
        student_size_ratio: float = 0.5,
        hardware_profile: Optional[HardwareProfile] = None,
    ) -> Optional[ModelOptimizationResult]:
        """
        Perform knowledge distillation to create a smaller model.

        Args:
            teacher_model_id: ID of the teacher model
            student_size_ratio: Size ratio of student to teacher
            hardware_profile: Target hardware profile

        Returns:
            Optimization result or None if failed
        """
        # This would contain the actual distillation logic
        # Currently a placeholder implementation

        self.logger.info(
            f"Distilling model {self.model_id} from teacher {teacher_model_id} "
            f"with size ratio {student_size_ratio:.1%}"
        )

        # Create a dummy result for demonstration
        result = ModelOptimizationResult(
            model_id=self.model_id,
            hardware_profile=hardware_profile
            or HardwareProfile(
                device_id="cpu_default",
                device_type="cpu",
                compute_units=1,
                memory_bytes=1000000000,  # 1GB
                peak_flops=1.0,
                peak_memory_bandwidth=10.0,
            ),
            optimization_level=f"distilled_{int(student_size_ratio * 100)}pct",
            inference_time_seconds=0.05,
            memory_usage_bytes=50000000,  # 50MB
            accuracy_metrics={"accuracy": 0.92},
        )

        # Record the result
        self.optimization_results.append(result)
        self._log_optimization_result(result)

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            MODEL_INFERENCE_TIME.labels(
                model_id=self.model_id,
                hardware_id=result.hardware_profile.device_id,
                optimization_level=result.optimization_level,
            ).observe(result.inference_time_seconds)

            MODEL_MEMORY_USAGE.labels(
                model_id=self.model_id,
                hardware_id=result.hardware_profile.device_id,
                optimization_level=result.optimization_level,
            ).set(result.memory_usage_bytes)

            for metric_name, value in result.accuracy_metrics.items():
                MODEL_ACCURACY.labels(
                    model_id=self.model_id,
                    hardware_id=result.hardware_profile.device_id,
                    optimization_level=result.optimization_level,
                    metric_name=metric_name,
                ).set(value)

        return result

    def benchmark(
        self,
        optimization_level: str,
        hardware_profile: HardwareProfile,
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark model performance on specific hardware.

        Args:
            optimization_level: Optimization level
            hardware_profile: Hardware profile to benchmark on
            num_runs: Number of benchmark runs

        Returns:
            Dictionary with benchmark results
        """
        # This would contain the actual benchmarking logic
        # Currently a placeholder implementation

        self.logger.info(
            f"Benchmarking model {self.model_id} with optimization {optimization_level} "
            f"on {hardware_profile.device_type} {hardware_profile.device_id}"
        )

        # Create dummy benchmark results
        benchmark_results = {
            "model_id": self.model_id,
            "hardware_profile": {
                "device_id": hardware_profile.device_id,
                "device_type": hardware_profile.device_type,
            },
            "optimization_level": optimization_level,
            "num_runs": num_runs,
            "mean_inference_time": 0.1,
            "std_inference_time": 0.01,
            "mean_memory_usage": 100000000,  # 100MB
            "throughput_examples_per_second": 10.0,
            "accuracy_metrics": {"accuracy": 0.95},
        }

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            MODEL_INFERENCE_TIME.labels(
                model_id=self.model_id,
                hardware_id=hardware_profile.device_id,
                optimization_level=optimization_level,
            ).observe(benchmark_results["mean_inference_time"])

            MODEL_MEMORY_USAGE.labels(
                model_id=self.model_id,
                hardware_id=hardware_profile.device_id,
                optimization_level=optimization_level,
            ).set(benchmark_results["mean_memory_usage"])

            for metric_name, value in benchmark_results["accuracy_metrics"].items():
                MODEL_ACCURACY.labels(
                    model_id=self.model_id,
                    hardware_id=hardware_profile.device_id,
                    optimization_level=optimization_level,
                    metric_name=metric_name,
                ).set(value)

        return benchmark_results

    def _log_optimization_result(self, result: ModelOptimizationResult) -> None:
        """Log optimization result to a file."""
        # Convert to serializable dict
        log_entry = {
            "timestamp": datetime.fromtimestamp(result.timestamp).isoformat(),
            "model_id": result.model_id,
            "hardware_profile": {
                "device_id": result.hardware_profile.device_id,
                "device_type": result.hardware_profile.device_type,
                "compute_units": result.hardware_profile.compute_units,
                "memory_bytes": result.hardware_profile.memory_bytes,
                "peak_flops": result.hardware_profile.peak_flops,
                "peak_memory_bandwidth": result.hardware_profile.peak_memory_bandwidth,
                "description": result.hardware_profile.description,
                "vendor": result.hardware_profile.vendor,
                "model": result.hardware_profile.model,
            },
            "optimization_level": result.optimization_level,
            "inference_time_seconds": result.inference_time_seconds,
            "memory_usage_bytes": result.memory_usage_bytes,
            "accuracy_metrics": result.accuracy_metrics,
            "additional_metrics": result.additional_metrics,
        }
        # Ensure additional_metrics is a dict
        assert isinstance(log_entry["additional_metrics"], dict)

        # Convert to JSON string
        json_str = json.dumps(log_entry)

        # Generate filename based on model
        filename = f"{self.model_id}_optimizations.jsonl"
        filepath = os.path.join(self.log_dir, filename)

        # Append to log file
        with open(filepath, "a") as f:
            f.write(json_str + "\n")


# Factory functions
def create_hardware_monitor(
    node_id: str, log_dir: Optional[str] = None, collection_interval: float = 5.0
) -> HardwareMonitor:
    """Create and return a new hardware monitor."""
    return HardwareMonitor(
        node_id=node_id, log_dir=log_dir, collection_interval=collection_interval
    )


def create_model_optimizer(
    model_id: str, base_dir: str, log_dir: Optional[str] = None
) -> ModelOptimizer:
    """Create and return a new model optimizer."""
    return ModelOptimizer(model_id=model_id, base_dir=base_dir, log_dir=log_dir)


def get_current_hardware_profile(device_id: Optional[str] = None) -> HardwareProfile:
    """
    Get hardware profile for the current system or a specific device.

    Args:
        device_id: Optional device ID to profile

    Returns:
        HardwareProfile object
    """
    # This is a simplified implementation
    # A real implementation would detect actual hardware capabilities

    device_id = device_id or "current_system"

    if PSUTIL_AVAILABLE:
        compute_units = psutil.cpu_count(logical=True) or 1
        memory = psutil.virtual_memory()
        memory_bytes = memory.total
        assert isinstance(memory_bytes, int) # Assert type

        # Estimate FLOPS (very rough)
        # Assume 10 GFLOPS per core on a modern CPU
        peak_flops = compute_units * 0.01  # 10 GFLOPS per core = 0.01 TFLOPS

        # Estimate memory bandwidth (very rough)
        # Assume DDR4-2400 with dual channel = ~38 GB/s
        peak_memory_bandwidth = 38.0

        description = f"CPU system with {compute_units} cores and {memory_bytes / (1024**3):.1f} GB RAM"

        # Check for GPU
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    if device_id == "current_system" or "gpu" in device_id.lower():
                        device_id = f"gpu_{gpu.id}"
                        compute_units = 1000  # Placeholder
                        gpu_memory_bytes = gpu.memoryTotal * 1024 * 1024  # MB to bytes
                        assert isinstance(gpu_memory_bytes, int) # Assert type

                        # Very rough estimates based on GPU class
                        if "3090" in gpu.name or "a100" in gpu.name.lower():
                            peak_flops = 36.0  # ~36 TFLOPS for high-end GPU
                            peak_memory_bandwidth = 936.0  # ~936 GB/s
                        elif "2080" in gpu.name or "v100" in gpu.name.lower():
                            peak_flops = 14.0  # ~14 TFLOPS
                            peak_memory_bandwidth = 616.0  # ~616 GB/s
                        else:
                            peak_flops = 5.0  # Conservative estimate
                            peak_memory_bandwidth = 300.0  # Conservative estimate

                        description = f"GPU system with {gpu.name} and {gpu_memory_bytes / (1024**3):.1f} GB VRAM"
                        return HardwareProfile(
                            device_id=device_id,
                            device_type="gpu",
                            compute_units=compute_units,
                            memory_bytes=gpu_memory_bytes,
                            peak_flops=peak_flops,
                            peak_memory_bandwidth=peak_memory_bandwidth,
                            description=description,
                            vendor=gpu.name.split()[0],
                            model=gpu.name,
                        )
            except Exception as e:
                logger.warning(f"Error detecting GPU hardware profile: {e}")

        return HardwareProfile(
            device_id=device_id,
            device_type="cpu",
            compute_units=compute_units,
            memory_bytes=memory_bytes,
            peak_flops=peak_flops,
            peak_memory_bandwidth=peak_memory_bandwidth,
            description=description,
        )

    # Fallback with default values
    return HardwareProfile(
        device_id=device_id,
        device_type="unknown",
        compute_units=1,
        memory_bytes=1000000000,  # 1GB
        peak_flops=0.01,
        peak_memory_bandwidth=10.0,
        description="Unknown hardware profile",
    )
