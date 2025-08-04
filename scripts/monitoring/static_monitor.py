#!/usr/bin/env python3
"""
Enhanced Static System Monitor - Production Ready
==============================================================

Comprehensive system and AI agent monitoring with adaptive features:
- Adaptive refresh rates based on system load
- Network I/O monitoring and bandwidth tracking  
- Complete AI agent health monitoring with response times
- Configuration file support for thresholds and preferences
- Visual trend indicators and enhanced color coding
- Optional logging for historical analysis
- Agent performance metrics and activity tracking

Maintains compact 25-line terminal format with zero errors.
"""

import sys
import time
import json
import psutil
import subprocess
import os
import logging
import threading
import select
import termios
import tty
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class EnhancedMonitor:
    """Enhanced monitor with adaptive features and AI agent monitoring"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Colors and terminal control
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.BLUE = '\033[94m'
        self.MAGENTA = '\033[95m'
        self.CYAN = '\033[96m'
        self.BOLD = '\033[1m'
        self.RESET = '\033[0m'
        self.CLEAR = '\033[2J'
        self.HOME = '\033[H'
        self.HIDE_CURSOR = '\033[?25l'
        self.SHOW_CURSOR = '\033[?25h'
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Historical data for trends
        self.history = {
            'cpu': deque(maxlen=60),
            'memory': deque(maxlen=60),
            'network': deque(maxlen=60),
            'agent_response_times': defaultdict(lambda: deque(maxlen=30))
        }
        
        # Network baseline for bandwidth calculation
        self.network_baseline = None
        self.last_network_check = time.time()
        
        # Agent monitoring
        self.agent_registry = self._load_agent_registry()
        self.agent_health = {}
        self.agent_last_activity = {}
        
        # GPU monitoring setup - initialize first
        self.gpu_driver_type = None
        self.gpu_info = {'name': 'Unknown', 'driver': 'None'}
        self.gpu_available = self._detect_gpu_capabilities()
        
        # Adaptive timing with manual control
        self.base_refresh_rate = self.config.get('refresh_rate', 2.0)
        self.current_refresh_rate = self.base_refresh_rate
        self.manual_refresh_rate = self.base_refresh_rate
        self.adaptive_mode = self.config.get('adaptive_refresh', True)
        self.last_activity_check = time.time()
        self.last_refresh_time = 0
        
        # Keyboard input handling
        self.old_settings = None
        self.input_buffer = []
        self._setup_keyboard_input()
        
        # Logging setup
        self.logger = self._setup_logging()
        
        # Performance session for agent checks
        self.session = requests.Session()
        retry_strategy = Retry(
            total=1,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Initialize CPU measurement baseline
        psutil.cpu_percent()  # Initialize CPU measurement
        
        # Initialize display
        print(self.CLEAR + self.HIDE_CURSOR, end='')
        
    def _setup_keyboard_input(self):
        """Setup non-blocking keyboard input"""
        if sys.stdin.isatty():
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setraw(sys.stdin.fileno())
            except Exception:
                pass
    
    def _get_keyboard_input(self) -> Optional[str]:
        """Get keyboard input without blocking"""
        if not sys.stdin.isatty():
            return None
            
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                if char:
                    return char
        except Exception:
            pass
        return None
    
    def _handle_keyboard_input(self, key: str) -> bool:
        """Handle keyboard shortcuts for timer control"""
        if key == '+':
            # Increase refresh rate (slower)
            self.manual_refresh_rate = min(10.0, self.manual_refresh_rate + 0.5)
            if not self.adaptive_mode:
                self.current_refresh_rate = self.manual_refresh_rate
            return True
        elif key == '-':
            # Decrease refresh rate (faster)
            self.manual_refresh_rate = max(0.5, self.manual_refresh_rate - 0.5)
            if not self.adaptive_mode:
                self.current_refresh_rate = self.manual_refresh_rate
            return True
        elif key == 'a' or key == 'A':
            # Toggle adaptive mode
            self.adaptive_mode = not self.adaptive_mode
            if not self.adaptive_mode:
                self.current_refresh_rate = self.manual_refresh_rate
            return True
        elif key == 'r' or key == 'R':
            # Reset to default
            self.manual_refresh_rate = self.base_refresh_rate
            self.current_refresh_rate = self.base_refresh_rate
            self.adaptive_mode = self.config.get('adaptive_refresh', True)
            return True
        elif key == 'q' or key == 'Q' or ord(key) == 3:  # q, Q, or Ctrl+C
            return False
        return True
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'refresh_rate': 2.0,
            'adaptive_refresh': True,
            'thresholds': {
                'cpu_warning': 70,
                'cpu_critical': 85,
                'memory_warning': 75,
                'memory_critical': 90,
                'disk_warning': 80,
                'disk_critical': 90,
                'response_time_warning': 1000,
                'response_time_critical': 5000
            },
            'agent_monitoring': {
                'enabled': True,
                'timeout': 2,
                'max_agents_display': 6
            },
            'logging': {
                'enabled': False,
                'file': '/tmp/monitor.log',
                'level': 'INFO'
            },
            'display': {
                'show_trends': True,
                'show_network': True,
                'compact_mode': False
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _load_agent_registry(self) -> Dict[str, Any]:
        """Load AI agent registry"""
        registry_path = Path('/opt/sutazaiapp/agents/agent_registry.json')
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'agents': {}}
    
    def _setup_logging(self) -> Optional[logging.Logger]:
        """Setup optional logging"""
        if not self.config['logging']['enabled']:
            return None
            
        logger = logging.getLogger('enhanced_monitor')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        handler = logging.FileHandler(self.config['logging']['file'])
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def move_to(self, line):
        """Move cursor to specific line"""
        return f'\033[{line};1H'
    
    def clear_line(self):
        """Clear from cursor to end of line"""
        return '\033[K'
    
    def create_bar(self, percent, width=20):
        """Create progress bar"""
        filled = int((percent / 100) * width)
        bar = '█' * filled + '░' * (width - filled)
        return bar
    
    def get_color(self, value, warning, critical):
        """Get color based on thresholds"""
        if value >= critical:
            return self.RED
        elif value >= warning:
            return self.YELLOW
        else:
            return self.GREEN
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics with network monitoring"""
        # Get fresh CPU data - use a very short interval to get current usage
        # On first call or if too much time has passed, use interval to get accurate reading
        current_time = time.time()
        if not hasattr(self, '_last_cpu_time') or current_time - self._last_cpu_time > 10:
            cpu = psutil.cpu_percent(interval=0.1)  # Short blocking call for accuracy
            self._last_cpu_time = current_time
        else:
            cpu = psutil.cpu_percent(interval=None)  # Non-blocking call
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Calculate network bandwidth
        network_stats = self._calculate_network_stats(network)
        
        # Get GPU statistics
        gpu_stats = self.get_gpu_stats()
        
        # Store historical data for trends
        self.history['cpu'].append(cpu)
        self.history['memory'].append(memory.percent)
        self.history['network'].append(network_stats['bandwidth_mbps'])
        
        # Store GPU data for trends (ensure history exists)
        if gpu_stats['available']:
            if 'gpu' not in self.history:
                self.history['gpu'] = deque(maxlen=60)
            self.history['gpu'].append(gpu_stats['usage'])
        
        # Calculate adaptive refresh rate
        self._update_refresh_rate(cpu, memory.percent)
        
        stats = {
            'cpu_percent': cpu,
            'cpu_cores': psutil.cpu_count(),
            'cpu_trend': self._get_trend(self.history['cpu']),
            'mem_percent': memory.percent,
            'mem_used': memory.used / 1024 / 1024 / 1024,
            'mem_total': memory.total / 1024 / 1024 / 1024,
            'mem_trend': self._get_trend(self.history['memory']),
            'swap_percent': swap.percent,
            'swap_used': swap.used / 1024 / 1024,
            'swap_total': swap.total / 1024 / 1024 / 1024,
            'disk_percent': disk.percent,
            'disk_free': disk.free / 1024 / 1024 / 1024,
            'network': network_stats,
            'connections': len(psutil.net_connections()),
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0),
            'gpu': gpu_stats
        }
        
        # Log metrics if enabled
        if self.logger:
            gpu_log = f"GPU={gpu_stats['usage']:.1f}%" if gpu_stats['available'] else "GPU=N/A"
            self.logger.info(f"System stats: CPU={cpu:.1f}%, MEM={memory.percent:.1f}%, "
                           f"NET={network_stats['bandwidth_mbps']:.1f}Mbps, {gpu_log}")
        
        return stats
    
    def _calculate_network_stats(self, network) -> Dict[str, Any]:
        """Calculate network statistics and bandwidth"""
        current_time = time.time()
        
        if self.network_baseline is None:
            self.network_baseline = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'timestamp': current_time
            }
            return {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'bandwidth_mbps': 0.0,
                'upload_mbps': 0.0,
                'download_mbps': 0.0
            }
        
        time_diff = current_time - self.network_baseline['timestamp']
        if time_diff < 1.0:  # Avoid division by zero
            time_diff = 1.0
        
        bytes_sent_diff = network.bytes_sent - self.network_baseline['bytes_sent']
        bytes_recv_diff = network.bytes_recv - self.network_baseline['bytes_recv']
        
        upload_mbps = (bytes_sent_diff * 8) / (time_diff * 1024 * 1024)
        download_mbps = (bytes_recv_diff * 8) / (time_diff * 1024 * 1024)
        bandwidth_mbps = upload_mbps + download_mbps
        
        # Update baseline
        self.network_baseline = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'timestamp': current_time
        }
        
        return {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'bandwidth_mbps': bandwidth_mbps,
            'upload_mbps': upload_mbps,
            'download_mbps': download_mbps
        }
    
    def _get_trend(self, data: deque) -> str:
        """Calculate trend arrow based on recent data"""
        if len(data) < 3:
            return "→"
        
        recent = list(data)[-3:]
        if recent[-1] > recent[-2] > recent[-3]:
            return "↑"
        elif recent[-1] < recent[-2] < recent[-3]:
            return "↓"
        else:
            return "→"
    
    def _update_refresh_rate(self, cpu: float, memory: float):
        """Adaptively update refresh rate based on system activity"""
        if not self.adaptive_mode:
            self.current_refresh_rate = self.manual_refresh_rate
            return
        
        high_activity = cpu > 50 or memory > 70
        very_high_activity = cpu > 80 or memory > 85
        
        base_rate = self.manual_refresh_rate
        
        if very_high_activity:
            self.current_refresh_rate = max(0.5, base_rate * 0.25)
        elif high_activity:
            self.current_refresh_rate = max(1.0, base_rate * 0.5)
        else:
            self.current_refresh_rate = min(5.0, base_rate * 1.5)
    
    def get_ai_agents_status(self) -> Tuple[List[str], int, int]:
        """Get AI agent status with health monitoring"""
        agents = []
        healthy_count = 0
        total_count = 0
        
        max_display = self.config['agent_monitoring'].get('max_agents_display', 6)
        timeout = self.config['agent_monitoring'].get('timeout', 2)
        
        try:
            for agent_id, agent_info in list(self.agent_registry.get('agents', {}).items())[:max_display]:
                total_count += 1
                
                # Check agent health
                health_status, response_time = self._check_agent_health(agent_id, agent_info, timeout)
                
                # Update agent tracking
                self.agent_health[agent_id] = health_status
                self.agent_last_activity[agent_id] = datetime.now()
                
                # Store response time for trends
                if response_time is not None:
                    self.history['agent_response_times'][agent_id].append(response_time)
                
                # Format display
                if health_status == 'healthy':
                    icon = '🟢'
                    healthy_count += 1
                    color = self.GREEN
                elif health_status == 'warning':
                    icon = '🟡'
                    color = self.YELLOW
                elif health_status == 'critical':
                    icon = '🔴'
                    color = self.RED
                elif health_status == 'offline':
                    icon = '🔴'
                    color = self.RED
                else:
                    icon = '🔘'
                    color = self.RESET
                
                # Get agent type from description or name
                agent_type = self._get_agent_type(agent_info)
                
                # Format response time
                rt_str = f"{response_time:.0f}ms" if response_time is not None else "--ms"
                
                # Create display line with improved name truncation and status
                display_name = self._get_display_name(agent_id)
                agent_line = f"{display_name:<14} {icon} {color}{health_status[:8]:<8}{self.RESET} {rt_str:>6}"
                agents.append(agent_line)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting agent status: {e}")
            agents.append("Error loading agents")
        
        return agents, healthy_count, total_count
    
    def _check_agent_health(self, agent_id: str, agent_info: Dict, timeout: int) -> Tuple[str, Optional[float]]:
        """Check individual agent health with enhanced response time measurement"""
        try:
            # First check if the agent is actually deployed
            if not self._is_agent_deployed(agent_id):
                return 'offline', None
            
            # Try to determine agent endpoint
            endpoint = self._get_agent_endpoint(agent_id, agent_info)
            if not endpoint:
                return 'offline', None
            
            start_time = time.time()
            response_time = None
            
            # Try multiple health check paths
            health_paths = ['/health', '/status', '/ping', '/api/health', '/heartbeat']
            last_exception = None
            
            for path in health_paths:
                try:
                    response = self.session.get(
                        f"{endpoint}{path}", 
                        timeout=timeout,
                        headers={'User-Agent': 'SutazAI-Monitor/1.0'}
                    )
                    
                    response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                    
                    # Check if response indicates healthy service
                    if response.status_code in [200, 201, 204]:
                        # Parse response for additional health info if available
                        health_status = self._parse_health_response(response, response_time)
                        return health_status, response_time
                    elif response.status_code in [400, 401, 403, 405]:
                        # Service is running but may not have health endpoint
                        # Check response time thresholds
                        if response_time > self.config['thresholds']['response_time_critical']:
                            return 'critical', response_time
                        elif response_time > self.config['thresholds']['response_time_warning']:
                            return 'warning', response_time
                        else:
                            return 'healthy', response_time
                    
                except requests.exceptions.Timeout:
                    return 'critical', timeout * 1000
                except requests.exceptions.ConnectionError as e:
                    last_exception = e
                    continue
                except Exception as e:
                    last_exception = e
                    continue
            
            # If all health paths failed, the service is likely down
            if last_exception:
                if isinstance(last_exception, requests.exceptions.ConnectionError):
                    return 'offline', None
                else:
                    return 'offline', response_time
            
            return 'offline', response_time
                
        except requests.exceptions.Timeout:
            return 'offline', timeout * 1000
        except requests.exceptions.ConnectionError:
            return 'offline', None
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Health check failed for {agent_id}: {e}")
            return 'offline', None
    
    def _parse_health_response(self, response, response_time: float) -> str:
        """Parse health response to determine detailed status"""
        try:
            # Try to parse JSON response for detailed health info
            if 'json' in response.headers.get('content-type', '').lower():
                data = response.json()
                
                # Check common health response patterns
                status = data.get('status', '').lower()
                if status in ['ok', 'healthy', 'up', 'running']:
                    # Check response time thresholds
                    if response_time > self.config['thresholds']['response_time_critical']:
                        return 'critical'
                    elif response_time > self.config['thresholds']['response_time_warning']:
                        return 'warning'
                    else:
                        return 'healthy'
                elif status in ['degraded', 'warning']:
                    return 'warning'
                elif status in ['down', 'error', 'critical', 'unhealthy']:
                    return 'critical'
        except Exception:
            pass  # Fall back to response time based assessment
        
        # Default to response time based assessment
        if response_time > self.config['thresholds']['response_time_critical']:
            return 'critical'
        elif response_time > self.config['thresholds']['response_time_warning']:
            return 'warning'
        else:
            return 'healthy'
    
    def _get_agent_endpoint(self, agent_id: str, agent_info: Dict) -> Optional[str]:
        """Determine agent endpoint for health checks with enhanced detection"""
        # Check if endpoint is specified in agent config
        config_path = agent_info.get('config_path')
        if config_path:
            endpoint = self._extract_endpoint_from_config(config_path)
            if endpoint:
                return endpoint
        
        # Try agent-specific port patterns based on agent type
        agent_type = self._get_agent_type(agent_info)
        port_ranges = self._get_port_ranges_by_type(agent_type)
        
        for port_range in port_ranges:
            for port in port_range:
                endpoint = f"http://localhost:{port}"
                if self._test_port_connection(port):
                    # Verify it's actually an agent endpoint
                    if self._verify_agent_endpoint(endpoint, agent_id):
                        return endpoint
        
        # Fallback to common ports
        common_ports = [8000, 8001, 8002, 8003, 8004, 8005, 8115, 3000, 5000, 9000]
        for port in common_ports:
            if self._test_port_connection(port):
                endpoint = f"http://localhost:{port}"
                if self._verify_agent_endpoint(endpoint, agent_id):
                    return endpoint
        
        return None
    
    def _extract_endpoint_from_config(self, config_path: str) -> Optional[str]:
        """Extract endpoint from agent configuration file"""
        try:
            full_path = Path('/opt/sutazaiapp/agents') / config_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    config = json.load(f)
                    return config.get('endpoint') or config.get('url') or config.get('base_url')
        except Exception:
            pass
        return None
    
    def _get_port_ranges_by_type(self, agent_type: str) -> List[List[int]]:
        """Get port ranges based on agent type"""
        type_port_map = {
            'BACK': [[8000, 8010], [5000, 5010]],    # Backend services
            'FRON': [[3000, 3010], [8080, 8090]],    # Frontend services  
            'AI': [[11434, 11444], [7860, 7870]],    # AI/ML services (Ollama, HuggingFace)
            'INFR': [[9000, 9010], [6000, 6010], [8110, 8120]],    # Infrastructure services (includes hardware-resource-optimizer on 8115)
            'SECU': [[8443, 8453], [9443, 9453]],    # Security services
            'DATA': [[5432, 5442], [6379, 6389]],    # Data services
        }
        
        ranges = type_port_map.get(agent_type, [[8000, 8020]])
        return [list(range(start, end)) for start, end in ranges]
    
    def _test_port_connection(self, port: int) -> bool:
        """Test if port is open and accepting connections"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.3)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _verify_agent_endpoint(self, endpoint: str, agent_id: str) -> bool:
        """Verify endpoint is actually an agent service"""
        try:
            # Try common health check paths
            health_paths = ['/health', '/status', '/ping', '/api/health', '/heartbeat']
            
            for path in health_paths:
                try:
                    response = self.session.get(
                        f"{endpoint}{path}",
                        timeout=1,
                        headers={'User-Agent': 'SutazAI-Monitor/1.0'}
                    )
                    if response.status_code in [200, 201, 204]:
                        return True
                except Exception:
                    continue
            
            # If no health endpoints, try root path
            try:
                response = self.session.get(
                    endpoint,
                    timeout=1,
                    headers={'User-Agent': 'SutazAI-Monitor/1.0'}
                )
                # Accept various success codes and even some error codes that indicate a service is running
                return response.status_code in [200, 201, 204, 400, 401, 403, 405, 501]
            except Exception:
                pass
            
        except Exception:
            pass
        
        return False
    
    def _get_agent_type(self, agent_info: Dict) -> str:
        """Extract agent type from agent information"""
        name = agent_info.get('name', '')
        description = agent_info.get('description', '')
        
        # Check agent name for direct matches first (most reliable)
        if 'hardware-resource-optimizer' in name:
            return 'INFR'
        
        # Extract key type indicators with word boundaries for precision
        type_indicators = {
            'backend': ['backend', 'api', 'server'],
            'frontend': ['frontend', 'web app', 'web ui'],  # Made more specific
            'ai': ['ai', 'ml', 'model'],  # Removed 'agent' as it's too generic
            'infra': ['infrastructure', 'deploy', 'container', 'hardware', 'resource', 'optimizer', 'monitoring'],
            'security': ['security', 'auth', 'vault'],
            'data': ['data', 'database', 'storage']
        }
        
        text = (name + ' ' + description).lower()
        
        # Count matches for each type to find the best match
        type_scores = {}
        for agent_type, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                type_scores[agent_type] = score
        
        # Return the type with the highest score
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            return best_type[:4].upper()
        
        return 'UTIL'
    
    def _is_agent_deployed(self, agent_id: str) -> bool:
        """Check if agent is actually deployed (has directory and/or container)"""
        try:
            # Check if agent directory exists
            agent_dir = Path('/opt/sutazaiapp/agents') / agent_id
            if agent_dir.exists():
                return True
            
            # Check if container exists
            import subprocess
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', f'name={agent_id}', '--format', '{{.Names}}'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
            
            return False
        except Exception:
            return False
    
    def _get_display_name(self, agent_id: str) -> str:
        """Get intelligently truncated display name"""
        if len(agent_id) <= 14:
            return agent_id
        
        # Special handling for hardware-resource-optimizer
        if agent_id == 'hardware-resource-optimizer':
            return 'hardware-optim'
        
        # Try to extract meaningful parts
        parts = agent_id.split('-')
        if len(parts) > 1:
            # Take first and last meaningful parts
            if len(parts[0]) <= 6 and len(parts[-1]) <= 6:
                truncated = f"{parts[0]}-{parts[-1]}"
                if len(truncated) <= 14:
                    return truncated
            
            # Try first part + abbreviated last
            if len(parts[0]) <= 8:
                remaining = 14 - len(parts[0]) - 1  # -1 for hyphen
                last_part = parts[-1][:remaining] if remaining > 0 else ""
                if last_part:
                    return f"{parts[0]}-{last_part}"
        
        # Fallback to simple truncation
        return agent_id[:14]
    
    def _detect_gpu_capabilities(self) -> bool:
        """Detect available GPU monitoring capabilities with WSL2 support"""
        # First check if we're in WSL2
        wsl_info = self._detect_wsl_environment()
        
        # Try NVIDIA first (native or WSL2)
        nvidia_detected = self._detect_nvidia_gpu(wsl_info)
        if nvidia_detected:
            return True
        
        # Try AMD (ROCm)
        amd_detected = self._detect_amd_gpu()
        if amd_detected:
            return True
        
        # Try Intel GPU
        intel_detected = self._detect_intel_gpu()
        if intel_detected:
            return True
        
        # WSL2-specific GPU detection
        if wsl_info['is_wsl2']:
            wsl_gpu_detected = self._detect_wsl2_gpu(wsl_info)
            if wsl_gpu_detected:
                return True
        
        # Generic GPU detection through system files
        generic_detected = self._detect_generic_gpu()
        if generic_detected:
            return True
        
        return False
    
    def _detect_wsl_environment(self) -> dict:
        """Detect WSL environment and GPU capabilities"""
        wsl_info = {
            'is_wsl': False,
            'is_wsl2': False,
            'gpu_passthrough': False,
            'version': None
        }
        
        try:
            # Check if running in WSL
            if os.path.exists('/proc/version'):
                with open('/proc/version', 'r') as f:
                    version_info = f.read().lower()
                    if 'microsoft' in version_info or 'wsl' in version_info:
                        wsl_info['is_wsl'] = True
                        if 'wsl2' in version_info or 'microsoft-standard-wsl2' in version_info:
                            wsl_info['is_wsl2'] = True
            
            # Check WSL environment variables
            if os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSLENV'):
                wsl_info['is_wsl'] = True
                # WSL2 is more likely if these specific env vars exist
                if os.environ.get('WSLG_GPU') or os.environ.get('WSL2_GUI_APPS_ENABLED'):
                    wsl_info['is_wsl2'] = True
                    wsl_info['gpu_passthrough'] = True
            
            # Check for WSL2 GPU passthrough indicators
            if wsl_info['is_wsl2']:
                # Check for DirectX GPU passthrough device
                if os.path.exists('/dev/dxg'):
                    wsl_info['gpu_passthrough'] = True
                
                # Check for WSLG (Windows Subsystem for Linux GUI) GPU support
                if os.environ.get('WAYLAND_DISPLAY') or os.environ.get('DISPLAY'):
                    wsl_info['gpu_passthrough'] = True
        
        except Exception:
            pass
        
        return wsl_info
    
    def _detect_nvidia_gpu(self, wsl_info: dict) -> bool:
        """Enhanced NVIDIA GPU detection with comprehensive WSL2 support"""
        # Try multiple nvidia-smi locations in priority order
        nvidia_smi_paths = self._get_nvidia_smi_paths(wsl_info)
        
        for nvidia_smi_path in nvidia_smi_paths:
            if self._test_nvidia_smi_path(nvidia_smi_path, wsl_info):
                return True
        
        # WSL2-specific fallback methods
        if wsl_info['is_wsl2']:
            if self._detect_nvidia_wsl2_fallbacks(wsl_info):
                return True
        
        # Try gpustat as final fallback
        if self._detect_nvidia_gpustat():
            return True
        
        return False
    
    def _get_nvidia_smi_paths(self, wsl_info: dict) -> list:
        """Get prioritized list of nvidia-smi paths to try"""
        paths = []
        
        if wsl_info['is_wsl2']:
            # WSL2 specific paths - Windows nvidia-smi.exe is most reliable
            paths.extend([
                '/mnt/c/Windows/System32/nvidia-smi.exe',
                '/mnt/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe',
                '/mnt/c/Windows/system32/nvidia-smi.exe',  # Case variations
                '/usr/lib/wsl/lib/nvidia-smi',  # WSL2 specific location
                'nvidia-smi.exe'  # Try in PATH
            ])
        
        # Standard paths (works for native Linux and sometimes WSL2)
        paths.extend([
            'nvidia-smi',  # Standard PATH lookup
            '/usr/bin/nvidia-smi',
            '/usr/local/cuda/bin/nvidia-smi'
        ])
        
        return paths
    
    def _test_nvidia_smi_path(self, nvidia_smi_path: str, wsl_info: dict) -> bool:
        """Test a specific nvidia-smi path and extract GPU info"""
        try:
            # First try XML output for structured parsing
            if self._test_nvidia_smi_xml(nvidia_smi_path, wsl_info):
                return True
            
            # Fallback to CSV format
            if self._test_nvidia_smi_csv(nvidia_smi_path, wsl_info):
                return True
                
        except Exception as e:
            if self.logger:
                self.logger.debug(f"nvidia-smi path {nvidia_smi_path} failed: {e}")
        
        return False
    
    def _test_nvidia_smi_xml(self, nvidia_smi_path: str, wsl_info: dict) -> bool:
        """Test nvidia-smi with XML output for comprehensive data"""
        try:
            result = subprocess.run([nvidia_smi_path, '-q', '-x'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                gpu_data = self._parse_nvidia_xml(result.stdout)
                if gpu_data:
                    self.gpu_info = {
                        'name': gpu_data.get('name', 'NVIDIA GPU'),
                        'driver': f"NVIDIA {gpu_data.get('driver_version', 'Unknown')}",
                        'wsl2': wsl_info['is_wsl2'],
                        'xml_support': True
                    }
                    self.gpu_driver_type = 'nvidia_xml' if not wsl_info['is_wsl2'] else 'nvidia_wsl2_xml'
                    self.nvidia_smi_path = nvidia_smi_path  # Store working path
                    return True
        except Exception:
            pass
        
        return False
    
    def _test_nvidia_smi_csv(self, nvidia_smi_path: str, wsl_info: dict) -> bool:
        """Test nvidia-smi with CSV output as fallback"""
        try:
            result = subprocess.run([nvidia_smi_path, '--query-gpu=name,driver_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(', ')
                    driver_version = parts[1].strip() if len(parts) > 1 else 'Unknown'
                    self.gpu_info = {
                        'name': parts[0].strip(), 
                        'driver': f"NVIDIA {driver_version}",
                        'wsl2': wsl_info['is_wsl2'],
                        'xml_support': False
                    }
                    self.gpu_driver_type = 'nvidia' if not wsl_info['is_wsl2'] else 'nvidia_wsl2'
                    self.nvidia_smi_path = nvidia_smi_path  # Store working path
                    return True
        except Exception:
            pass
        
        return False
    
    def _parse_nvidia_xml(self, xml_output: str) -> dict:
        """Parse nvidia-smi XML output for comprehensive GPU information"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_output)
            
            gpu_data = {}
            
            # Find first GPU
            gpu = root.find('.//gpu')
            if gpu is not None:
                # Extract basic info
                product_name = gpu.find('product_name')
                if product_name is not None:
                    gpu_data['name'] = product_name.text
                
                # Driver version
                driver_version = root.find('.//driver_version')
                if driver_version is not None:
                    gpu_data['driver_version'] = driver_version.text
                
                # GPU UUID for identification
                uuid = gpu.find('uuid')
                if uuid is not None:
                    gpu_data['uuid'] = uuid.text
                
                # CUDA version
                cuda_version = root.find('.//cuda_version')
                if cuda_version is not None:
                    gpu_data['cuda_version'] = cuda_version.text
            
            return gpu_data if gpu_data else None
        
        except Exception as e:
            if self.logger:
                self.logger.debug(f"XML parsing failed: {e}")
            return None
    
    def _detect_nvidia_wsl2_fallbacks(self, wsl_info: dict) -> bool:
        """WSL2-specific NVIDIA detection fallback methods"""
        try:
            # Check for NVIDIA device files that might exist in WSL2
            nvidia_devices = ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm']
            nvidia_files_found = [dev for dev in nvidia_devices if os.path.exists(dev)]
            
            if nvidia_files_found:
                # Try nvidia-ml-py (pynvml) as fallback
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                        driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                        self.gpu_info = {
                            'name': name,
                            'driver': f"NVIDIA {driver_version} (pynvml)",
                            'wsl2': True,
                            'pynvml_support': True
                        }
                        self.gpu_driver_type = 'nvidia_pynvml'
                        return True
                except ImportError:
                    if self.logger:
                        self.logger.debug("pynvml not available, consider installing: pip install nvidia-ml-py")
                except Exception as e:
                    if self.logger:
                        self.logger.debug(f"pynvml failed: {e}")
                
                # Check proc filesystem for NVIDIA info
                if self._detect_nvidia_proc_info():
                    return True
                
                # Fallback - we know NVIDIA files exist but can't get detailed info
                self.gpu_info = {
                    'name': 'NVIDIA GPU (WSL2)', 
                    'driver': 'WSL2 - Device files detected',
                    'wsl2': True,
                    'device_files': nvidia_files_found
                }
                self.gpu_driver_type = 'nvidia_wsl2_limited'
                return True
        except Exception:
            pass
        
        return False
    
    def _detect_nvidia_proc_info(self) -> bool:
        """Try to get NVIDIA GPU info from proc filesystem"""
        try:
            if os.path.exists('/proc/driver/nvidia'):
                nvidia_proc_files = os.listdir('/proc/driver/nvidia')
                if 'gpus' in nvidia_proc_files:
                    gpu_dirs = os.listdir('/proc/driver/nvidia/gpus')
                    if gpu_dirs:
                        # Try to read GPU info from proc
                        gpu_dir = gpu_dirs[0]
                        info_file = f'/proc/driver/nvidia/gpus/{gpu_dir}/information'
                        if os.path.exists(info_file):
                            with open(info_file, 'r') as f:
                                content = f.read()
                                # Extract GPU name from proc file
                                for line in content.split('\n'):
                                    if 'Model:' in line:
                                        gpu_name = line.split('Model:')[1].strip()
                                        self.gpu_info = {
                                            'name': gpu_name,
                                            'driver': 'NVIDIA (WSL2 proc)',
                                            'wsl2': True,
                                            'proc_info': True
                                        }
                                        self.gpu_driver_type = 'nvidia_wsl2_proc'
                                        return True
                        
                        # Fallback if we can't read specific info
                        self.gpu_info = {
                            'name': 'NVIDIA GPU (WSL2)',
                            'driver': 'WSL2 proc - Limited info',
                            'wsl2': True,
                            'proc_info': True
                        }
                        self.gpu_driver_type = 'nvidia_wsl2_proc'
                        return True
        except Exception:
            pass
        
        return False
    
    def _detect_nvidia_gpustat(self) -> bool:
        """Try to use gpustat as fallback GPU monitoring tool"""
        try:
            # First try to import gpustat
            result = subprocess.run(['gpustat', '--json'], capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                import json
                gpustat_data = json.loads(result.stdout)
                
                if 'gpus' in gpustat_data and gpustat_data['gpus']:
                    gpu = gpustat_data['gpus'][0]  # First GPU
                    self.gpu_info = {
                        'name': gpu.get('name', 'NVIDIA GPU'),
                        'driver': f"NVIDIA (gpustat)",
                        'wsl2': self._detect_wsl_environment()['is_wsl2'],
                        'gpustat_support': True
                    }
                    self.gpu_driver_type = 'nvidia_gpustat'
                    return True
        except Exception:
            # Try to install gpustat if it's not available
            try:
                if self.logger:
                    self.logger.info("Attempting to install gpustat for GPU monitoring")
                subprocess.run(['pip', 'install', 'gpustat'], capture_output=True, timeout=30)
                # Retry after installation
                result = subprocess.run(['gpustat', '--json'], capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    return self._detect_nvidia_gpustat()  # Recursive call after installation
            except Exception:
                pass
        
        return False
    
    def _detect_amd_gpu(self) -> bool:
        """Detect AMD GPU capabilities"""
        try:
            result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and 'GPU' in result.stdout:
                self.gpu_info = {'name': 'AMD GPU', 'driver': 'ROCm'}
                self.gpu_driver_type = 'amd'
                return True
        except Exception:
            pass
        return False
    
    def _detect_intel_gpu(self) -> bool:
        """Detect Intel GPU capabilities"""
        try:
            result = subprocess.run(['intel_gpu_top', '-l'], capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                self.gpu_info = {'name': 'Intel GPU', 'driver': 'Intel'}
                self.gpu_driver_type = 'intel'
                return True
        except Exception:
            pass
        return False
    
    def _detect_wsl2_gpu(self, wsl_info: dict) -> bool:
        """WSL2-specific GPU detection methods"""
        try:
            # Check for WSL2 DirectX GPU passthrough
            if os.path.exists('/dev/dxg'):
                self.gpu_info = {
                    'name': 'DirectX GPU (WSL2)',
                    'driver': 'WSL2 DirectX',
                    'wsl2': True
                }
                self.gpu_driver_type = 'wsl2_directx'
                return True
            
            # Check for NVIDIA proc files in WSL2
            if os.path.exists('/proc/driver/nvidia'):
                try:
                    nvidia_proc_files = os.listdir('/proc/driver/nvidia')
                    if 'gpus' in nvidia_proc_files:
                        gpu_dirs = os.listdir('/proc/driver/nvidia/gpus')
                        if gpu_dirs:
                            # Try to read GPU info from proc
                            gpu_dir = gpu_dirs[0]
                            info_file = f'/proc/driver/nvidia/gpus/{gpu_dir}/information'
                            if os.path.exists(info_file):
                                with open(info_file, 'r') as f:
                                    content = f.read()
                                    # Extract GPU name from proc file
                                    for line in content.split('\n'):
                                        if 'Model:' in line:
                                            gpu_name = line.split('Model:')[1].strip()
                                            self.gpu_info = {
                                                'name': gpu_name,
                                                'driver': 'NVIDIA (WSL2 proc)',
                                                'wsl2': True
                                            }
                                            self.gpu_driver_type = 'nvidia_wsl2_proc'
                                            return True
                            
                            # Fallback if we can't read specific info
                            self.gpu_info = {
                                'name': 'NVIDIA GPU (WSL2)',
                                'driver': 'WSL2 - Limited info',
                                'wsl2': True
                            }
                            self.gpu_driver_type = 'nvidia_wsl2_proc'
                            return True
                except Exception:
                    pass
            
            # If WSL2 but no GPU passthrough
            if wsl_info['is_wsl2'] and not wsl_info['gpu_passthrough']:
                return False  # Don't report GPU if passthrough not enabled
            
        except Exception:
            pass
        
        return False
    
    def _detect_generic_gpu(self) -> bool:
        """Generic GPU detection through system files"""
        try:
            if os.path.exists('/sys/class/drm'):
                drm_devices = os.listdir('/sys/class/drm')
                gpu_devices = [d for d in drm_devices if d.startswith('card') and not d.endswith('-')]
                if gpu_devices:
                    self.gpu_info = {'name': 'Generic GPU', 'driver': 'Generic'}
                    self.gpu_driver_type = 'generic'
                    return True
        except Exception:
            pass
        
        return False
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics with comprehensive WSL2 and fallback handling"""
        if not self.gpu_available:
            return {'available': False, 'usage': 0, 'memory': 0, 'temperature': 0, 'name': 'No GPU detected'}
        
        try:
            # All NVIDIA variants - use unified method with fallbacks
            if self.gpu_driver_type.startswith('nvidia'):
                # Try the primary method first
                if self.gpu_driver_type == 'nvidia_gpustat':
                    stats = self._get_nvidia_gpustat_stats()
                    if stats['available']:
                        return stats
                
                # Try standard nvidia-smi methods
                stats = self._get_nvidia_stats()
                if stats['available']:
                    return stats
                
                # Try pynvml if detected
                if self.gpu_info.get('pynvml_support'):
                    stats = self._get_nvidia_pynvml_stats()
                    if stats['available']:
                        return stats
                
                # Try gpustat as last resort for NVIDIA
                stats = self._get_nvidia_gpustat_stats()
                if stats['available']:
                    return stats
                
                # If all methods fail, return detection info
                return {
                    'available': True,
                    'usage': 0,
                    'memory': 0, 
                    'temperature': 0,
                    'name': self.gpu_info['name'],
                    'status': f"Detected but stats unavailable - {self.gpu_info.get('driver', 'Unknown driver')}"
                }
            elif self.gpu_driver_type == 'wsl2_directx':
                return self._get_wsl2_directx_stats()
            elif self.gpu_driver_type == 'amd':
                return self._get_amd_stats()
            elif self.gpu_driver_type == 'intel':
                return self._get_intel_stats()
            else:
                return self._get_generic_stats()
        except Exception as e:
            if self.logger:
                self.logger.debug(f"GPU stats error: {e}")
            return {'available': False, 'usage': 0, 'memory': 0, 'temperature': 0, 'name': self.gpu_info['name'], 'error': str(e)}
    
    def _get_nvidia_stats(self) -> Dict[str, Any]:
        """Get NVIDIA GPU statistics using the best available method"""
        if hasattr(self, 'nvidia_smi_path'):
            # Use stored working nvidia-smi path
            if self.gpu_info.get('xml_support', False):
                return self._get_nvidia_stats_xml()
            else:
                return self._get_nvidia_stats_csv()
        
        # Fallback to original method
        return self._get_nvidia_stats_csv()
    
    def _get_nvidia_stats_xml(self) -> Dict[str, Any]:
        """Get comprehensive NVIDIA GPU statistics using XML output"""
        try:
            result = subprocess.run([self.nvidia_smi_path, '-q', '-x'], 
                                  capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and result.stdout.strip():
                return self._parse_nvidia_xml_stats(result.stdout)
        except Exception as e:
            if self.logger:
                self.logger.debug(f"XML stats failed, falling back to CSV: {e}")
        
        # Fallback to CSV
        return self._get_nvidia_stats_csv()
    
    def _parse_nvidia_xml_stats(self, xml_output: str) -> Dict[str, Any]:
        """Parse nvidia-smi XML output for real-time statistics"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_output)
            
            # Find first GPU
            gpu = root.find('.//gpu')
            if gpu is not None:
                # GPU utilization
                utilization = gpu.find('.//utilization')
                gpu_util = 0
                if utilization is not None:
                    gpu_util_elem = utilization.find('gpu_util')
                    if gpu_util_elem is not None:
                        gpu_util = float(gpu_util_elem.text.replace('%', '').strip())
                
                # Memory utilization
                fb_memory_usage = gpu.find('.//fb_memory_usage')
                mem_used = 0
                mem_total = 1
                if fb_memory_usage is not None:
                    used_elem = fb_memory_usage.find('used')
                    total_elem = fb_memory_usage.find('total')
                    if used_elem is not None:
                        mem_used = float(used_elem.text.split()[0])  # Remove 'MiB' unit
                    if total_elem is not None:
                        mem_total = float(total_elem.text.split()[0])  # Remove 'MiB' unit
                
                # Temperature
                temperature = gpu.find('.//temperature')
                gpu_temp = 0
                if temperature is not None:
                    gpu_temp_elem = temperature.find('gpu_temp')
                    if gpu_temp_elem is not None:
                        gpu_temp = float(gpu_temp_elem.text.replace('C', '').strip())
                
                # Power usage
                power_readings = gpu.find('.//power_readings')
                power_draw = 0
                if power_readings is not None:
                    power_draw_elem = power_readings.find('power_draw')
                    if power_draw_elem is not None:
                        power_draw = float(power_draw_elem.text.replace('W', '').strip())
                
                # GPU name
                product_name = gpu.find('product_name')
                name = product_name.text if product_name is not None else self.gpu_info['name']
                
                mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                
                return {
                    'available': True,
                    'usage': gpu_util,
                    'memory': mem_percent,
                    'memory_used': mem_used / 1024,  # Convert to GB
                    'memory_total': mem_total / 1024,  # Convert to GB
                    'temperature': gpu_temp,
                    'power': power_draw,
                    'name': name[:20]  # Truncate long names
                }
        
        except Exception as e:
            if self.logger:
                self.logger.debug(f"XML stats parsing failed: {e}")
        
        return {'available': False, 'usage': 0, 'memory': 0, 'temperature': 0, 'name': self.gpu_info['name']}
    
    def _get_nvidia_stats_csv(self) -> Dict[str, Any]:
        """Get NVIDIA GPU statistics using CSV output"""
        try:
            nvidia_smi = getattr(self, 'nvidia_smi_path', 'nvidia-smi')
            result = subprocess.run([
                nvidia_smi, 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = [p.strip() for p in lines[0].split(', ')]
                    if len(parts) >= 4:
                        usage = float(parts[0]) if parts[0] not in ['[Not Supported]', 'N/A'] else 0
                        mem_used = float(parts[1]) if parts[1] not in ['[Not Supported]', 'N/A'] else 0
                        mem_total = float(parts[2]) if parts[2] not in ['[Not Supported]', 'N/A'] else 1
                        temp = float(parts[3]) if parts[3] not in ['[Not Supported]', 'N/A'] else 0
                        power = float(parts[4]) if len(parts) > 4 and parts[4] not in ['[Not Supported]', 'N/A'] else 0
                        name = parts[5] if len(parts) > 5 else self.gpu_info['name']
                        
                        mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                        
                        return {
                            'available': True,
                            'usage': usage,
                            'memory': mem_percent,
                            'memory_used': mem_used / 1024,  # Convert to GB
                            'memory_total': mem_total / 1024,  # Convert to GB
                            'temperature': temp,
                            'power': power,
                            'name': name[:20]  # Truncate long names
                        }
        except Exception as e:
            if self.logger:
                self.logger.debug(f"CSV stats failed: {e}")
        
        return {'available': False, 'usage': 0, 'memory': 0, 'temperature': 0, 'name': self.gpu_info['name']}
    
    def _get_amd_stats(self) -> Dict[str, Any]:
        """Get AMD GPU statistics"""
        try:
            # Try ROCm SMI
            usage_result = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True, timeout=2)
            temp_result = subprocess.run(['rocm-smi', '--showtemp'], capture_output=True, text=True, timeout=2)
            
            usage = 0
            temperature = 0
            
            if usage_result.returncode == 0:
                for line in usage_result.stdout.split('\n'):
                    if '%' in line and 'GPU' in line:
                        import re
                        match = re.search(r'(\d+)%', line)
                        if match:
                            usage = float(match.group(1))
                            break
            
            if temp_result.returncode == 0:
                for line in temp_result.stdout.split('\n'):
                    if 'c' in line.lower() and 'temp' in line.lower():
                        import re
                        match = re.search(r'(\d+\.?\d*)c', line, re.IGNORECASE)
                        if match:
                            temperature = float(match.group(1))
                            break
            
            return {
                'available': True,
                'usage': usage,
                'memory': 0,  # AMD memory usage is harder to get
                'temperature': temperature,
                'name': self.gpu_info['name']
            }
        except Exception:
            pass
        
        return {'available': False, 'usage': 0, 'memory': 0, 'temperature': 0, 'name': self.gpu_info['name']}
    
    def _get_intel_stats(self) -> Dict[str, Any]:
        """Get Intel GPU statistics"""
        try:
            result = subprocess.run(['intel_gpu_top', '-s', '100'], capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                # Parse intel_gpu_top output
                usage = 0
                for line in result.stdout.split('\n'):
                    if 'Render/3D' in line:
                        import re
                        match = re.search(r'(\d+\.?\d*)%', line)
                        if match:
                            usage = float(match.group(1))
                            break
                
                return {
                    'available': True,
                    'usage': usage,
                    'memory': 0,  # Intel integrated memory is shared
                    'temperature': 0,  # Temperature not easily available
                    'name': self.gpu_info['name']
                }
        except Exception:
            pass
        
        return {'available': False, 'usage': 0, 'memory': 0, 'temperature': 0, 'name': self.gpu_info['name']}
    
    def _get_generic_stats(self) -> Dict[str, Any]:
        """Get generic GPU statistics from system files"""
        try:
            # Try to read GPU frequency or usage from /sys
            gpu_usage = 0
            
            # Look for GPU frequency files
            gpu_freq_paths = ['/sys/class/drm/card0/gt_cur_freq_mhz', '/sys/class/drm/card0/device/gpu_usage']
            
            for path in gpu_freq_paths:
                try:
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            value = f.read().strip()
                            if value.isdigit():
                                # This is a rough approximation
                                gpu_usage = min(100, int(value) / 10)  # Arbitrary scaling
                                break
                except Exception:
                    continue
            
            return {
                'available': True,
                'usage': gpu_usage,
                'memory': 0,
                'temperature': 0,
                'name': self.gpu_info['name']
            }
        except Exception:
            pass
        
        return {'available': False, 'usage': 0, 'memory': 0, 'temperature': 0, 'name': self.gpu_info['name']}
    
    def _get_nvidia_gpustat_stats(self) -> Dict[str, Any]:
        """Get NVIDIA GPU statistics using gpustat"""
        try:
            result = subprocess.run(['gpustat', '--json'], capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                import json
                gpustat_data = json.loads(result.stdout)
                
                if 'gpus' in gpustat_data and gpustat_data['gpus']:
                    gpu = gpustat_data['gpus'][0]  # First GPU
                    
                    # Extract stats from gpustat format
                    usage = gpu.get('utilization.gpu', 0)
                    if isinstance(usage, str):
                        usage = float(usage.replace('%', '').strip()) if usage.replace('%', '').strip().isdigit() else 0
                    
                    mem_used = gpu.get('memory.used', 0)
                    mem_total = gpu.get('memory.total', 1)
                    if isinstance(mem_used, str):
                        mem_used = float(mem_used.replace('MB', '').replace('MiB', '').strip())
                    if isinstance(mem_total, str):
                        mem_total = float(mem_total.replace('MB', '').replace('MiB', '').strip())
                    
                    mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                    
                    temp = gpu.get('temperature.gpu', 0)
                    if isinstance(temp, str):
                        temp = float(temp.replace('C', '').strip()) if temp.replace('C', '').strip().replace('.', '').isdigit() else 0
                    
                    return {
                        'available': True,
                        'usage': usage,
                        'memory': mem_percent,
                        'memory_used': mem_used / 1024 if mem_used else 0,  # Convert to GB
                        'memory_total': mem_total / 1024 if mem_total else 0,  # Convert to GB
                        'temperature': temp,
                        'name': gpu.get('name', self.gpu_info['name'])[:20],
                        'method': 'gpustat'
                    }
        except Exception as e:
            if self.logger:
                self.logger.debug(f"gpustat stats failed: {e}")
        
        return {'available': False, 'usage': 0, 'memory': 0, 'temperature': 0, 'name': self.gpu_info['name']}
    
    def _get_nvidia_pynvml_stats(self) -> Dict[str, Any]:
        """Get NVIDIA GPU statistics using pynvml library"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Get utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except:
                    gpu_util = 0
                
                # Get memory info
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = mem_info.used / 1024 / 1024  # Convert to MB
                    mem_total = mem_info.total / 1024 / 1024  # Convert to MB
                    mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                except:
                    mem_used = mem_total = mem_percent = 0
                
                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = 0
                
                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to W
                except:
                    power = 0
                
                # Get name
                try:
                    name_raw = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name_raw, bytes):
                        name = name_raw.decode('utf-8')
                    else:
                        name = str(name_raw)  # Handle newer pynvml versions that return strings
                except:
                    name = self.gpu_info['name']
                
                return {
                    'available': True,
                    'usage': gpu_util,
                    'memory': mem_percent,
                    'memory_used': mem_used / 1024,  # Convert to GB
                    'memory_total': mem_total / 1024,  # Convert to GB
                    'temperature': temp,
                    'power': power,
                    'name': name[:20],
                    'method': 'pynvml'
                }
        except Exception as e:
            if self.logger:
                self.logger.debug(f"pynvml stats failed: {e}")
        
        return {'available': False, 'usage': 0, 'memory': 0, 'temperature': 0, 'name': self.gpu_info['name']}
    
    def _get_nvidia_wsl2_proc_stats(self) -> Dict[str, Any]:
        """Get NVIDIA GPU statistics from WSL2 proc files"""
        try:
            # Try to get some basic info from proc files
            usage = 0
            memory = 0
            
            # Attempt to read GPU utilization if available
            # Note: This is very limited in WSL2
            if os.path.exists('/proc/driver/nvidia/gpus'):
                try:
                    gpu_dirs = os.listdir('/proc/driver/nvidia/gpus')
                    if gpu_dirs:
                        gpu_dir = gpu_dirs[0]
                        # Try to read utilization - this might not be available
                        util_file = f'/proc/driver/nvidia/gpus/{gpu_dir}/utilization'
                        if os.path.exists(util_file):
                            with open(util_file, 'r') as f:
                                content = f.read()
                                # Very basic parsing - this format varies
                                if 'gpu' in content.lower():
                                    import re
                                    match = re.search(r'(\d+)%', content)
                                    if match:
                                        usage = float(match.group(1))
                except Exception:
                    pass
            
            return {
                'available': True,
                'usage': usage,
                'memory': memory,
                'temperature': 0,  # Temperature rarely available through proc
                'name': self.gpu_info['name'],
                'status': 'WSL2 proc info'
            }
        except Exception:
            pass
        
        return {
            'available': True,
            'usage': 0,
            'memory': 0,
            'temperature': 0,
            'name': self.gpu_info['name'],
            'status': 'WSL2 - Limited info'
        }
    
    def _get_wsl2_directx_stats(self) -> Dict[str, Any]:
        """Get DirectX GPU statistics for WSL2 with fallback attempts"""
        # Try to get some stats through alternative methods even for DirectX
        # First try Windows nvidia-smi if available
        wsl_nvidia_paths = [
            '/mnt/c/Windows/System32/nvidia-smi.exe',
            '/mnt/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe'
        ]
        
        for nvidia_path in wsl_nvidia_paths:
            if os.path.exists(nvidia_path):
                try:
                    result = subprocess.run([
                        nvidia_path, 
                        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name',
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split('\n')
                        if lines and lines[0]:
                            parts = [p.strip() for p in lines[0].split(', ')]
                            if len(parts) >= 4:
                                usage = float(parts[0]) if parts[0] not in ['[Not Supported]', 'N/A'] else 0
                                mem_used = float(parts[1]) if parts[1] not in ['[Not Supported]', 'N/A'] else 0
                                mem_total = float(parts[2]) if parts[2] not in ['[Not Supported]', 'N/A'] else 1
                                temp = float(parts[3]) if parts[3] not in ['[Not Supported]', 'N/A'] else 0
                                name = parts[4] if len(parts) > 4 else self.gpu_info['name']
                                
                                mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
                                
                                return {
                                    'available': True,
                                    'usage': usage,
                                    'memory': mem_percent,
                                    'memory_used': mem_used / 1024,
                                    'memory_total': mem_total / 1024,
                                    'temperature': temp,
                                    'name': name[:20],
                                    'method': 'WSL2 DirectX + Windows nvidia-smi'
                                }
                except Exception:
                    continue
        
        # DirectX GPU passthrough fallback - indicate detection but no stats
        return {
            'available': True,
            'usage': 0,  # DirectX doesn't expose usage stats directly
            'memory': 0,  # Memory usage not available through DirectX
            'temperature': 0,  # Temperature not available through DirectX
            'name': self.gpu_info['name'],
            'status': 'WSL2 DirectX - GPU detected, stats limited'
        }
    
    def get_docker_containers(self) -> Tuple[List[str], int, int]:
        """Get Docker container info (fallback when agents not available)"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}|{{.Status}}|{{.Image}}'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                containers = []
                running = 0
                for line in result.stdout.strip().split('\n'):
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) == 3:
                            name = parts[0][:18]
                            status = 'Up' in parts[1]
                            image = parts[2].split(':')[0].split('/')[-1][:18]
                            
                            if status:
                                icon = '🟢'
                                running += 1
                            else:
                                icon = '🔴'
                            
                            containers.append(f"{name:<18} {icon} {image}")
                
                return containers[:6], running, len(containers)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Docker check failed: {e}")
        
        return ["No containers found"], 0, 0
    
    def get_ollama_models(self):
        """Get Ollama models"""
        try:
            result = subprocess.run(
                ['docker', 'exec', 'sutazai-ollama', 'ollama', 'list'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    models = []
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            models.append(f"  • {model_name}")
                    
                    return models[:3]
        except:
            pass
        
        return ["  Unable to retrieve models"]
    
    def run(self):
        """Main enhanced monitor loop with adaptive features"""
        try:
            # Initial display
            print(self.HOME + "Initializing Enhanced Monitor...", end='', flush=True)
            
            while True:
                # Get comprehensive data
                stats = self.get_system_stats()
                
                # Choose between AI agents or Docker containers
                if self.config['agent_monitoring']['enabled'] and self.agent_registry.get('agents'):
                    agents, healthy, total = self.get_ai_agents_status()
                    display_type = "AI Agents"
                    icon = "🤖"
                else:
                    agents, healthy, total = self.get_docker_containers()
                    display_type = "Containers"
                    icon = "🐳"
                
                models = self.get_ollama_models()
                
                # Build enhanced display
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                refresh_mode = "ADAPTIVE" if self.adaptive_mode else "MANUAL"
                refresh_indicator = f"⚡{self.current_refresh_rate:.1f}s [{refresh_mode}]"
                
                # Line 1: Enhanced header with refresh rate
                print(f"{self.move_to(1)}{self.BOLD}🚀 SutazAI Enhanced Monitor{self.RESET} - {timestamp} [{refresh_indicator}]{self.clear_line()}", end='')
                
                # Line 2: Separator
                print(f"{self.move_to(2)}{'=' * 70}{self.clear_line()}", end='')
                
                # Lines 3-7: Enhanced system stats with trends and network
                cpu_color = self.get_color(stats['cpu_percent'], 
                                         self.config['thresholds']['cpu_warning'], 
                                         self.config['thresholds']['cpu_critical'])
                mem_color = self.get_color(stats['mem_percent'], 
                                         self.config['thresholds']['memory_warning'], 
                                         self.config['thresholds']['memory_critical'])
                disk_color = self.get_color(stats['disk_percent'], 
                                          self.config['thresholds']['disk_warning'], 
                                          self.config['thresholds']['disk_critical'])
                
                trend_cpu = stats['cpu_trend'] if self.config['display']['show_trends'] else ""
                trend_mem = stats['mem_trend'] if self.config['display']['show_trends'] else ""
                
                print(f"{self.move_to(3)}CPU:    {self.create_bar(stats['cpu_percent'])} {cpu_color}{stats['cpu_percent']:5.1f}%{self.RESET} {trend_cpu} ({stats['cpu_cores']}c) Load:{stats['load_avg'][0]:.2f}{self.clear_line()}", end='')
                print(f"{self.move_to(4)}Memory: {self.create_bar(stats['mem_percent'])} {mem_color}{stats['mem_percent']:5.1f}%{self.RESET} {trend_mem} ({stats['mem_used']:.1f}GB/{stats['mem_total']:.1f}GB){self.clear_line()}", end='')
                print(f"{self.move_to(5)}Disk:   {self.create_bar(stats['disk_percent'])} {disk_color}{stats['disk_percent']:5.1f}%{self.RESET} ({stats['disk_free']:.1f}GB free){self.clear_line()}", end='')
                
                # Line 6: GPU statistics (if available)
                gpu_line = 6
                if stats['gpu']['available']:
                    gpu_usage = stats['gpu']['usage']
                    gpu_color = self.get_color(gpu_usage, 70, 85)  # GPU thresholds
                    gpu_trend = self._get_trend(self.history['gpu']) if self.config['display']['show_trends'] and len(self.history['gpu']) > 2 else ""
                    
                    gpu_temp_str = f" {stats['gpu']['temperature']:.0f}°C" if stats['gpu']['temperature'] > 0 else ""
                    gpu_mem_str = f" {stats['gpu']['memory']:.0f}%" if stats['gpu']['memory'] > 0 else ""
                    
                    # Handle different GPU stat scenarios
                    if 'status' in stats['gpu'] and gpu_usage == 0:
                        # WSL2 or limited capability GPU - detected but no usage stats
                        status_msg = stats['gpu']['status']
                        bar_display = '─' * 10 + '▓' * 10  # Half-filled bar to indicate detection but no stats
                        print(f"{self.move_to(gpu_line)}GPU:    {bar_display} {self.YELLOW}  DETECTED{self.RESET} ({stats['gpu']['name']}) - {status_msg}{self.clear_line()}", end='')
                    elif 'method' in stats['gpu']:
                        # GPU with stats from alternative method
                        method_str = f" [{stats['gpu']['method']}]"
                        power_str = f" {stats['gpu']['power']:.0f}W" if stats['gpu'].get('power', 0) > 0 else ""
                        print(f"{self.move_to(gpu_line)}GPU:    {self.create_bar(gpu_usage)} {gpu_color}{gpu_usage:5.1f}%{self.RESET} {gpu_trend}{gpu_temp_str}{gpu_mem_str}{power_str} ({stats['gpu']['name']}){method_str}{self.clear_line()}", end='')
                    else:
                        # Normal GPU with full stats
                        power_str = f" {stats['gpu']['power']:.0f}W" if stats['gpu'].get('power', 0) > 0 else ""
                        print(f"{self.move_to(gpu_line)}GPU:    {self.create_bar(gpu_usage)} {gpu_color}{gpu_usage:5.1f}%{self.RESET} {gpu_trend}{gpu_temp_str}{gpu_mem_str}{power_str} ({stats['gpu']['name']}){self.clear_line()}", end='')
                    network_line = 7
                else:
                    # Check if we're in WSL2 to show appropriate message
                    wsl_info = self._detect_wsl_environment()
                    if wsl_info['is_wsl2']:
                        if wsl_info['gpu_passthrough']:
                            wsl_msg = "WSL2 - GPU passthrough enabled but no GPU detected"
                        else:
                            wsl_msg = "WSL2 - GPU passthrough not enabled"
                    elif wsl_info['is_wsl']:
                        wsl_msg = "WSL1 - GPU not supported"
                    else:
                        wsl_msg = "No GPU detected or drivers missing"
                    
                    print(f"{self.move_to(gpu_line)}GPU:    {'─' * 20} {self.RESET} N/A - {wsl_msg}{self.clear_line()}", end='')
                    network_line = 7
                
                # Network statistics (line determined by GPU presence)
                if self.config['display']['show_network']:
                    net = stats['network']
                    net_trend = self._get_trend(self.history['network']) if self.config['display']['show_trends'] else ""
                    print(f"{self.move_to(network_line)}Network: {self.CYAN}{net['bandwidth_mbps']:6.1f} Mbps{self.RESET} {net_trend} ↑{net['upload_mbps']:.1f} ↓{net['download_mbps']:.1f} Conn:{stats['connections']}{self.clear_line()}", end='')
                    agent_start_line = network_line + 2
                else:
                    print(f"{self.move_to(network_line)}{self.clear_line()}", end='')
                    agent_start_line = network_line + 1
                
                # Blank line before agents
                print(f"{self.move_to(agent_start_line - 1)}{self.clear_line()}", end='')
                
                # Agent/Container section header with health summary
                health_color = self.GREEN if healthy == total else (self.YELLOW if healthy > total * 0.5 else self.RED)
                print(f"{self.move_to(agent_start_line)}{icon} {display_type} ({health_color}{healthy}{self.RESET}/{total}) {'Name':<14} Status    RT{self.clear_line()}", end='')
                
                # Lines for agents/containers (6 items)
                for i in range(6):
                    line_num = agent_start_line + 1 + i
                    if i < len(agents):
                        print(f"{self.move_to(line_num)}{agents[i]}{self.clear_line()}", end='')
                    else:
                        print(f"{self.move_to(line_num)}{self.clear_line()}", end='')
                
                # Models section (if available)
                models_start = agent_start_line + 7
                if models and models[0] != "  Unable to retrieve models":
                    print(f"{self.move_to(models_start)}🤖 Ollama Models:{self.clear_line()}", end='')
                    for i in range(min(3, len(models))):
                        print(f"{self.move_to(models_start + 1 + i)}{models[i]}{self.clear_line()}", end='')
                    alert_line = models_start + 4
                else:
                    print(f"{self.move_to(models_start)}{self.clear_line()}", end='')
                    alert_line = models_start + 1
                
                # Clear remaining model lines
                for i in range(3):
                    if models_start + 1 + i < alert_line:
                        print(f"{self.move_to(models_start + 1 + i)}{self.clear_line()}", end='')
                
                # Enhanced alerts with multiple conditions
                alert_msg = self._generate_alert_message(stats, healthy, total)
                print(f"{self.move_to(alert_line)}🎯 Status: {alert_msg}{self.clear_line()}", end='')
                
                # Footer with timer controls and enhanced information
                footer_line = alert_line + 2
                config_info = "CFG" if hasattr(self, 'config') else "DEF"
                log_info = "LOG" if self.logger else "---"
                mode_str = "ADAPTIVE" if self.adaptive_mode else "MANUAL"
                rate_str = f"{self.manual_refresh_rate:.1f}s"
                print(f"{self.move_to(footer_line)}Controls: +/- (speed), A (adaptive), R (reset), Q (quit) | {mode_str} {rate_str} | {config_info} {log_info}{self.clear_line()}", end='')
                
                # Clear any remaining lines
                for i in range(footer_line + 1, 25):
                    print(f"{self.move_to(i)}{self.clear_line()}", end='')
                
                # Flush output
                sys.stdout.flush()
                
                # Handle keyboard input and sleep with proper refresh timing
                start_sleep = time.time()
                sleep_duration = self.current_refresh_rate
                
                while time.time() - start_sleep < sleep_duration:
                    key = self._get_keyboard_input() 
                    if key:
                        if not self._handle_keyboard_input(key):
                            raise KeyboardInterrupt
                    time.sleep(0.1)  # Small sleep to prevent high CPU usage
                
        except KeyboardInterrupt:
            # Clean exit with session cleanup
            self.cleanup()
    
    def _generate_alert_message(self, stats: Dict, healthy: int, total: int) -> str:
        """Generate comprehensive alert message"""
        alerts = []
        
        # System alerts
        if stats['cpu_percent'] > self.config['thresholds']['cpu_critical']:
            alerts.append(f"{self.RED}HIGH CPU: {stats['cpu_percent']:.1f}%{self.RESET}")
        elif stats['cpu_percent'] > self.config['thresholds']['cpu_warning']:
            alerts.append(f"{self.YELLOW}CPU: {stats['cpu_percent']:.1f}%{self.RESET}")
        
        if stats['mem_percent'] > self.config['thresholds']['memory_critical']:
            alerts.append(f"{self.RED}HIGH MEM: {stats['mem_percent']:.1f}%{self.RESET}")
        elif stats['mem_percent'] > self.config['thresholds']['memory_warning']:
            alerts.append(f"{self.YELLOW}MEM: {stats['mem_percent']:.1f}%{self.RESET}")
        
        if stats['disk_percent'] > self.config['thresholds']['disk_critical']:
            alerts.append(f"{self.RED}LOW DISK: {stats['disk_percent']:.1f}%{self.RESET}")
        
        # Agent health alerts
        if total > 0:
            health_ratio = healthy / total
            if health_ratio < 0.5:
                alerts.append(f"{self.RED}AGENTS CRITICAL{self.RESET}")
            elif health_ratio < 0.8:
                alerts.append(f"{self.YELLOW}AGENTS WARNING{self.RESET}")
        
        # GPU alerts
        if stats['gpu']['available'] and stats['gpu']['usage'] > 85:
            alerts.append(f"{self.RED}HIGH GPU: {stats['gpu']['usage']:.1f}%{self.RESET}")
        elif stats['gpu']['available'] and stats['gpu']['usage'] > 70:
            alerts.append(f"{self.YELLOW}GPU: {stats['gpu']['usage']:.1f}%{self.RESET}")
        
        # Network alerts
        if stats['network']['bandwidth_mbps'] > 100:
            alerts.append(f"{self.YELLOW}HIGH NET: {stats['network']['bandwidth_mbps']:.0f}Mbps{self.RESET}")
        
        if alerts:
            return " | ".join(alerts)
        else:
            return f"{self.GREEN}✅ All systems operational{self.RESET}"
    
    def cleanup(self):
        """Clean up resources and exit gracefully"""
        try:
            # Restore terminal settings
            if self.old_settings and sys.stdin.isatty():
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
            self.session.close()
            if self.logger:
                self.logger.info("Enhanced monitor stopped")
        except:
            pass
        
        print(self.SHOW_CURSOR)
        print(f"{self.move_to(25)}\n{self.GREEN}Enhanced Monitor stopped gracefully.{self.RESET}{self.clear_line()}")


def main():
    """Enhanced monitor entry point"""
    # Check if terminal supports ANSI
    if not sys.stdout.isatty() and '--force' not in sys.argv:
        print("Error: This monitor requires an interactive terminal.")
        print("For testing purposes, use --force flag.")
        return 1
    
    # Parse command line arguments
    config_path = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        config_path = sys.argv[1]
    elif '--config' in sys.argv:
        config_index = sys.argv.index('--config')
        if config_index + 1 < len(sys.argv):
            config_path = sys.argv[config_index + 1]
    
    # Create and run enhanced monitor
    monitor = EnhancedMonitor(config_path)
    try:
        monitor.run()
        return 0
    except Exception as e:
        monitor.cleanup()
        print(f"Monitor error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())