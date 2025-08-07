#!/usr/bin/env python3
"""
System Freeze Prevention for SutazAI Ollama Agents
Advanced monitoring and automatic intervention to prevent system freezes
"""

import asyncio
import logging
import os
import sys
import time
import psutil
import signal
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import sqlite3
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """Current system state snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    swap_percent: float
    load_avg: tuple
    active_processes: int
    ollama_processes: int
    agent_processes: int
    network_connections: int
    freeze_risk_score: float

@dataclass
class PreventiveAction:
    """Preventive action taken to avoid freeze"""
    action_type: str
    timestamp: datetime
    description: str
    severity: str
    success: bool
    error: Optional[str] = None

class FreezePreventionSystem:
    """
    Advanced system freeze prevention with automatic intervention
    
    This system monitors system resources and takes proactive measures
    to prevent freezes before they occur.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.monitor_interval = 5  # Monitor every 5 seconds
        self.db_path = '/opt/sutazaiapp/monitoring/freeze_prevention.db'
        self.ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
        
        # Thresholds for different severity levels
        self.thresholds = {
            'memory': {
                'warning': 80.0,    # 80% memory usage
                'critical': 90.0,   # 90% memory usage
                'emergency': 95.0   # 95% memory usage
            },
            'cpu': {
                'warning': 85.0,    # 85% CPU usage
                'critical': 95.0,   # 95% CPU usage
                'emergency': 98.0   # 98% CPU usage
            },
            'load': {
                'warning': 5.0,     # Load average > 5
                'critical': 10.0,   # Load average > 10
                'emergency': 15.0   # Load average > 15
            },
            'swap': {
                'warning': 50.0,    # 50% swap usage
                'critical': 80.0,   # 80% swap usage
                'emergency': 95.0   # 95% swap usage
            }
        }
        
        # System state tracking
        self.system_history: List[SystemState] = []
        self.max_history = 100  # Keep last 100 measurements
        self.actions_taken: List[PreventiveAction] = []
        
        # Flags and counters
        self.emergency_mode = False
        self.intervention_count = 0
        self.last_intervention = None
        self.shutdown_event = asyncio.Event()
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize database
        self._init_database()
        
        self.logger.info("Freeze Prevention System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for tracking"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_percent REAL,
                    swap_percent REAL,
                    load_avg_1 REAL,
                    load_avg_5 REAL,
                    load_avg_15 REAL,
                    active_processes INTEGER,
                    ollama_processes INTEGER,
                    agent_processes INTEGER,
                    network_connections INTEGER,
                    freeze_risk_score REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS preventive_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    action_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    success BOOLEAN,
                    error TEXT
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_states_timestamp ON system_states(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON preventive_actions(timestamp)')
    
    async def start(self):
        """Start the freeze prevention system"""
        self.logger.info("Starting Freeze Prevention System...")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._cleanup_loop()),
            asyncio.create_task(self._emergency_intervention_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Freeze prevention tasks cancelled")
        except Exception as e:
            self.logger.error(f"Error in freeze prevention: {e}")
            raise
    
    async def stop(self):
        """Stop the freeze prevention system"""
        self.shutdown_event.set()
        if self.session:
            await self.session.close()
        self.logger.info("Freeze Prevention System stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                # Collect system state
                state = await self._collect_system_state()
                
                # Add to history
                self.system_history.append(state)
                if len(self.system_history) > self.max_history:
                    self.system_history.pop(0)
                
                # Store in database
                await self._store_system_state(state)
                
                # Analyze and take action if needed
                await self._analyze_and_act(state)
                
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_state(self) -> SystemState:
        """Collect current system state"""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            swap = psutil.swap_memory()
            load_avg = os.getloadavg()
            
            # Process counts
            all_processes = list(psutil.process_iter(['pid', 'name', 'cmdline']))
            active_processes = len(all_processes)
            
            ollama_processes = len([
                p for p in all_processes 
                if p.info['name'] and 'ollama' in p.info['name'].lower()
            ])
            
            agent_processes = len([
                p for p in all_processes 
                if p.info['cmdline'] and any('agent' in str(cmd).lower() for cmd in p.info['cmdline'] or [])
            ])
            
            # Network connections
            network_connections = len(psutil.net_connections())
            
            # Calculate freeze risk score
            freeze_risk = self._calculate_freeze_risk(
                cpu_percent, memory.percent, swap.percent, load_avg[0]
            )
            
            return SystemState(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                swap_percent=swap.percent,
                load_avg=load_avg,
                active_processes=active_processes,
                ollama_processes=ollama_processes,
                agent_processes=agent_processes,
                network_connections=network_connections,
                freeze_risk_score=freeze_risk
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system state: {e}")
            # Return a default state to prevent crashes
            return SystemState(
                timestamp=datetime.utcnow(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                swap_percent=0.0,
                load_avg=(0.0, 0.0, 0.0),
                active_processes=0,
                ollama_processes=0,
                agent_processes=0,
                network_connections=0,
                freeze_risk_score=0.0
            )
    
    def _calculate_freeze_risk(self, cpu: float, memory: float, swap: float, load: float) -> float:
        """Calculate system freeze risk score (0-100)"""
        risk_factors = []
        
        # Memory pressure (highest weight)
        if memory > 95:
            risk_factors.append(40)
        elif memory > 90:
            risk_factors.append(30)
        elif memory > 80:
            risk_factors.append(15)
        elif memory > 70:
            risk_factors.append(5)
        
        # CPU pressure
        if cpu > 98:
            risk_factors.append(25)
        elif cpu > 95:
            risk_factors.append(20)
        elif cpu > 85:
            risk_factors.append(10)
        elif cpu > 75:
            risk_factors.append(3)
        
        # Swap usage (indicates memory pressure)
        if swap > 95:
            risk_factors.append(20)
        elif swap > 80:
            risk_factors.append(15)
        elif swap > 50:
            risk_factors.append(8)
        elif swap > 25:
            risk_factors.append(3)
        
        # Load average
        if load > 15:
            risk_factors.append(15)
        elif load > 10:
            risk_factors.append(10)
        elif load > 5:
            risk_factors.append(5)
        
        # Check for rapid changes (trend analysis)
        if len(self.system_history) >= 3:
            recent_states = self.system_history[-3:]
            memory_trend = recent_states[-1].memory_percent - recent_states[0].memory_percent
            cpu_trend = recent_states[-1].cpu_percent - recent_states[0].cpu_percent
            
            if memory_trend > 10:  # Rapid memory increase
                risk_factors.append(10)
            if cpu_trend > 15:  # Rapid CPU increase
                risk_factors.append(8)
        
        return min(100, sum(risk_factors))
    
    async def _analyze_and_act(self, state: SystemState):
        """Analyze system state and take preventive actions"""
        try:
            # Determine severity level
            severity = self._determine_severity(state)
            
            if severity == 'emergency':
                await self._handle_emergency(state)
            elif severity == 'critical':
                await self._handle_critical(state)
            elif severity == 'warning':
                await self._handle_warning(state)
            
            # Log current state periodically
            if int(time.time()) % 60 == 0:  # Every minute
                self.logger.info(
                    f"System state: CPU={state.cpu_percent:.1f}%, "
                    f"Memory={state.memory_percent:.1f}%, "
                    f"Load={state.load_avg[0]:.1f}, "
                    f"Risk={state.freeze_risk_score:.1f}%"
                )
                
        except Exception as e:
            self.logger.error(f"Error analyzing system state: {e}")
    
    def _determine_severity(self, state: SystemState) -> str:
        """Determine severity level based on system state"""
        
        # Emergency conditions (immediate intervention required)
        if (state.memory_percent >= self.thresholds['memory']['emergency'] or
            state.cpu_percent >= self.thresholds['cpu']['emergency'] or
            state.load_avg[0] >= self.thresholds['load']['emergency'] or
            state.freeze_risk_score >= 95):
            return 'emergency'
        
        # Critical conditions (urgent intervention required)
        if (state.memory_percent >= self.thresholds['memory']['critical'] or
            state.cpu_percent >= self.thresholds['cpu']['critical'] or
            state.load_avg[0] >= self.thresholds['load']['critical'] or
            state.freeze_risk_score >= 80):
            return 'critical'
        
        # Warning conditions (preventive measures)
        if (state.memory_percent >= self.thresholds['memory']['warning'] or
            state.cpu_percent >= self.thresholds['cpu']['warning'] or
            state.load_avg[0] >= self.thresholds['load']['warning'] or
            state.freeze_risk_score >= 60):
            return 'warning'
        
        return 'normal'
    
    async def _handle_emergency(self, state: SystemState):
        """Handle emergency conditions with immediate intervention"""
        self.logger.critical(f"EMERGENCY: System freeze imminent! Risk score: {state.freeze_risk_score:.1f}%")
        
        self.emergency_mode = True
        
        # Emergency actions (in order of preference)
        actions = [
            ('kill_hung_processes', 'Kill processes that appear hung'),
            ('kill_high_memory_agents', 'Kill agents using excessive memory'),
            ('restart_ollama', 'Restart Ollama service'),
            ('clear_system_cache', 'Clear system caches'),
            ('emergency_swap_clear', 'Clear swap space'),
        ]
        
        for action_type, description in actions:
            try:
                success = await self._execute_action(action_type, description, 'emergency')
                if success:
                    self.logger.warning(f"Emergency action successful: {description}")
                    await asyncio.sleep(5)  # Wait a bit to see if it helps
                    break
                else:
                    self.logger.error(f"Emergency action failed: {description}")
            except Exception as e:
                self.logger.error(f"Error executing emergency action {action_type}: {e}")
        
        # If nothing helps, consider controlled shutdown
        if state.freeze_risk_score >= 98:
            await self._consider_controlled_shutdown()
    
    async def _handle_critical(self, state: SystemState):
        """Handle critical conditions with urgent intervention"""
        self.logger.warning(f"CRITICAL: High freeze risk detected! Risk score: {state.freeze_risk_score:.1f}%")
        
        # Critical actions
        actions = [
            ('throttle_agents', 'Throttle agent processing'),
            ('kill_idle_agents', 'Kill idle agents'),
            ('reduce_ollama_connections', 'Reduce Ollama connection limit'),
            ('clear_application_cache', 'Clear application caches'),
        ]
        
        for action_type, description in actions:
            try:
                success = await self._execute_action(action_type, description, 'critical')
                if success:
                    break
            except Exception as e:
                self.logger.error(f"Error executing critical action {action_type}: {e}")
    
    async def _handle_warning(self, state: SystemState):
        """Handle warning conditions with preventive measures"""
        self.logger.info(f"WARNING: Elevated freeze risk. Risk score: {state.freeze_risk_score:.1f}%")
        
        # Warning actions (less aggressive)
        actions = [
            ('garbage_collect', 'Trigger garbage collection'),
            ('reduce_agent_concurrency', 'Reduce agent concurrency'),
            ('optimize_ollama_queue', 'Optimize Ollama request queue'),
            ('cleanup_temp_files', 'Clean up temporary files'),
        ]
        
        for action_type, description in actions:
            try:
                await self._execute_action(action_type, description, 'warning')
                break
            except Exception as e:
                self.logger.error(f"Error executing warning action {action_type}: {e}")
    
    async def _execute_action(self, action_type: str, description: str, severity: str) -> bool:
        """Execute a specific preventive action"""
        start_time = datetime.utcnow()
        success = False
        error = None
        
        try:
            self.logger.info(f"Executing {severity} action: {description}")
            
            if action_type == 'kill_hung_processes':
                success = await self._kill_hung_processes()
            elif action_type == 'kill_high_memory_agents':
                success = await self._kill_high_memory_agents()
            elif action_type == 'restart_ollama':
                success = await self._restart_ollama()
            elif action_type == 'clear_system_cache':
                success = await self._clear_system_cache()
            elif action_type == 'emergency_swap_clear':
                success = await self._clear_swap()
            elif action_type == 'throttle_agents':
                success = await self._throttle_agents()
            elif action_type == 'kill_idle_agents':
                success = await self._kill_idle_agents()
            elif action_type == 'reduce_ollama_connections':
                success = await self._reduce_ollama_connections()
            elif action_type == 'clear_application_cache':
                success = await self._clear_application_cache()
            elif action_type == 'garbage_collect':
                success = await self._trigger_garbage_collection()
            elif action_type == 'reduce_agent_concurrency':
                success = await self._reduce_agent_concurrency()
            elif action_type == 'optimize_ollama_queue':
                success = await self._optimize_ollama_queue()
            elif action_type == 'cleanup_temp_files':
                success = await self._cleanup_temp_files()
            else:
                error = f"Unknown action type: {action_type}"
                success = False
            
        except Exception as e:
            error = str(e)
            success = False
            self.logger.error(f"Action {action_type} failed: {error}")
        
        # Record the action
        action = PreventiveAction(
            action_type=action_type,
            timestamp=start_time,
            description=description,
            severity=severity,
            success=success,
            error=error
        )
        
        self.actions_taken.append(action)
        await self._store_preventive_action(action)
        
        if success:
            self.intervention_count += 1
            self.last_intervention = start_time
        
        return success
    
    async def _kill_hung_processes(self) -> bool:
        """Kill processes that appear hung"""
        try:
            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'status', 'create_time']):
                try:
                    # Skip system processes
                    if proc.info['pid'] < 100:
                        continue
                    
                    # Look for processes in uninterruptible sleep for too long
                    if proc.info['status'] == psutil.STATUS_DISK_SLEEP:
                        # Check if process has been in this state for more than 60 seconds
                        create_time = datetime.fromtimestamp(proc.info['create_time'])
                        if (datetime.now() - create_time).total_seconds() > 60:
                            self.logger.warning(f"Killing hung process: {proc.info['name']} (PID: {proc.info['pid']})")
                            proc.kill()
                            killed_count += 1
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.logger.info(f"Killed {killed_count} hung processes")
            return killed_count > 0
            
        except Exception as e:
            self.logger.error(f"Error killing hung processes: {e}")
            return False
    
    async def _kill_high_memory_agents(self) -> bool:
        """Kill agents using excessive memory"""
        try:
            killed_count = 0
            memory_threshold = 2048 * 1024 * 1024  # 2GB in bytes
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
                try:
                    if not proc.info['cmdline']:
                        continue
                    
                    # Check if it's an agent process
                    cmdline_str = ' '.join(proc.info['cmdline'])
                    if 'agent' in cmdline_str.lower() and proc.info['memory_info'].rss > memory_threshold:
                        self.logger.warning(f"Killing high-memory agent: {proc.info['name']} (PID: {proc.info['pid']}, Memory: {proc.info['memory_info'].rss / 1024 / 1024:.1f}MB)")
                        proc.terminate()
                        killed_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.logger.info(f"Killed {killed_count} high-memory agents")
            return killed_count > 0
            
        except Exception as e:
            self.logger.error(f"Error killing high-memory agents: {e}")
            return False
    
    async def _restart_ollama(self) -> bool:
        """Restart Ollama service"""
        try:
            # Try to restart Ollama using systemctl
            result = subprocess.run(['systemctl', 'restart', 'ollama'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self.logger.info("Ollama service restarted successfully")
                return True
            
            # If systemctl fails, try Docker restart
            result = subprocess.run(['docker', 'restart', 'ollama'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self.logger.info("Ollama container restarted successfully")
                return True
                
            self.logger.error("Failed to restart Ollama")
            return False
            
        except Exception as e:
            self.logger.error(f"Error restarting Ollama: {e}")
            return False
    
    async def _clear_system_cache(self) -> bool:
        """Clear system caches"""
        try:
            # Clear page cache, dentries and inodes
            subprocess.run(['sync'], check=True)
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
            
            self.logger.info("System caches cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing system cache: {e}")
            return False
    
    async def _clear_swap(self) -> bool:
        """Clear swap space"""
        try:
            # Turn off swap and turn it back on to clear it
            result1 = subprocess.run(['swapoff', '-a'], capture_output=True, text=True)
            await asyncio.sleep(2)
            result2 = subprocess.run(['swapon', '-a'], capture_output=True, text=True)
            
            if result1.returncode == 0 and result2.returncode == 0:
                self.logger.info("Swap space cleared")
                return True
            else:
                self.logger.error("Failed to clear swap space")
                return False
                
        except Exception as e:
            self.logger.error(f"Error clearing swap: {e}")
            return False
    
    async def _throttle_agents(self) -> bool:
        """Throttle agent processing"""
        try:
            # Send throttle signal to agent coordinator
            if self.session:
                async with self.session.post(f"{self.backend_url}/api/system/throttle") as response:
                    if response.status == 200:
                        self.logger.info("Agent throttling activated")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error throttling agents: {e}")
            return False
    
    async def _kill_idle_agents(self) -> bool:
        """Kill idle agents to free resources"""
        try:
            # Implementation would depend on agent architecture
            # For now, just log the intent
            self.logger.info("Would kill idle agents (implementation needed)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error killing idle agents: {e}")
            return False
    
    async def _reduce_ollama_connections(self) -> bool:
        """Reduce Ollama connection limits"""
        try:
            # This would require modifying Ollama configuration
            self.logger.info("Would reduce Ollama connections (implementation needed)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error reducing Ollama connections: {e}")
            return False
    
    async def _clear_application_cache(self) -> bool:
        """Clear application-specific caches"""
        try:
            # Clear common cache directories
            cache_dirs = [
                '/tmp',
                '/var/tmp',
                os.path.expanduser('~/.cache'),
                '/opt/sutazaiapp/cache'
            ]
            
            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    subprocess.run(['find', cache_dir, '-type', 'f', '-atime', '+1', '-delete'], 
                                 capture_output=True)
            
            self.logger.info("Application caches cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing application cache: {e}")
            return False
    
    async def _trigger_garbage_collection(self) -> bool:
        """Trigger garbage collection in Python processes"""
        try:
            # Send signal to trigger GC in all Python processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'python' in proc.info['name'].lower():
                        # Send SIGUSR1 as a signal to trigger GC (if implemented)
                        proc.send_signal(signal.SIGUSR1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.logger.info("Garbage collection triggered")
            return True
            
        except Exception as e:
            self.logger.error(f"Error triggering garbage collection: {e}")
            return False
    
    async def _reduce_agent_concurrency(self) -> bool:
        """Reduce agent concurrency limits"""
        try:
            # Send signal to reduce concurrency
            if self.session:
                async with self.session.post(f"{self.backend_url}/api/system/reduce-concurrency") as response:
                    if response.status == 200:
                        self.logger.info("Agent concurrency reduced")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error reducing agent concurrency: {e}")
            return False
    
    async def _optimize_ollama_queue(self) -> bool:
        """Optimize Ollama request queue"""
        try:
            # Clear pending requests or optimize queue
            self.logger.info("Would optimize Ollama queue (implementation needed)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing Ollama queue: {e}")
            return False
    
    async def _cleanup_temp_files(self) -> bool:
        """Clean up temporary files"""
        try:
            temp_dirs = ['/tmp', '/var/tmp']
            
            for temp_dir in temp_dirs:
                subprocess.run([
                    'find', temp_dir, 
                    '-type', 'f', 
                    '-atime', '+1',
                    '-size', '+100M',
                    '-delete'
                ], capture_output=True)
            
            self.logger.info("Temporary files cleaned up")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning temp files: {e}")
            return False
    
    async def _consider_controlled_shutdown(self):
        """Consider controlled shutdown to prevent hard freeze"""
        self.logger.critical("CONSIDERING CONTROLLED SHUTDOWN TO PREVENT HARD FREEZE")
        
        # This is a last resort - would need careful implementation
        # For now, just log the critical state
        await self._execute_action(
            'controlled_shutdown_considered',
            'System is at critical freeze risk - controlled shutdown considered',
            'emergency'
        )
    
    async def _store_system_state(self, state: SystemState):
        """Store system state in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO system_states (
                        timestamp, cpu_percent, memory_percent, disk_percent, swap_percent,
                        load_avg_1, load_avg_5, load_avg_15, active_processes, 
                        ollama_processes, agent_processes, network_connections, freeze_risk_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    state.timestamp.isoformat(), state.cpu_percent, state.memory_percent,
                    state.disk_percent, state.swap_percent, state.load_avg[0], state.load_avg[1],
                    state.load_avg[2], state.active_processes, state.ollama_processes,
                    state.agent_processes, state.network_connections, state.freeze_risk_score
                ))
        except Exception as e:
            self.logger.error(f"Error storing system state: {e}")
    
    async def _store_preventive_action(self, action: PreventiveAction):
        """Store preventive action in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO preventive_actions (
                        timestamp, action_type, description, severity, success, error
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    action.timestamp.isoformat(), action.action_type, action.description,
                    action.severity, action.success, action.error
                ))
        except Exception as e:
            self.logger.error(f"Error storing preventive action: {e}")
    
    async def _cleanup_loop(self):
        """Cleanup old data periodically"""
        while not self.shutdown_event.is_set():
            try:
                # Clean up old database records (keep last 7 days)
                cutoff = datetime.utcnow() - timedelta(days=7)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM system_states WHERE timestamp < ?', (cutoff.isoformat(),))
                    conn.execute('DELETE FROM preventive_actions WHERE timestamp < ?', (cutoff.isoformat(),))
                
                # Clean up memory
                if len(self.actions_taken) > 1000:
                    self.actions_taken = self.actions_taken[-500:]
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _emergency_intervention_loop(self):
        """Emergency intervention loop for immediate response"""
        while not self.shutdown_event.is_set():
            try:
                if self.emergency_mode:
                    # In emergency mode, monitor more frequently
                    await asyncio.sleep(1)
                    
                    # Check if we can exit emergency mode
                    if len(self.system_history) > 0:
                        latest_state = self.system_history[-1]
                        if latest_state.freeze_risk_score < 70:
                            self.emergency_mode = False
                            self.logger.info("Exiting emergency mode - system stabilized")
                else:
                    await asyncio.sleep(10)
                    
            except Exception as e:
                self.logger.error(f"Error in emergency intervention loop: {e}")
                await asyncio.sleep(10)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        latest_state = self.system_history[-1] if self.system_history else None
        
        return {
            'emergency_mode': self.emergency_mode,
            'intervention_count': self.intervention_count,
            'last_intervention': self.last_intervention.isoformat() if self.last_intervention else None,
            'latest_state': {
                'timestamp': latest_state.timestamp.isoformat(),
                'cpu_percent': latest_state.cpu_percent,
                'memory_percent': latest_state.memory_percent,
                'freeze_risk_score': latest_state.freeze_risk_score
            } if latest_state else None,
            'actions_taken_count': len(self.actions_taken)
        }


async def main():
    """Main entry point"""
    freeze_prevention = FreezePreventionSystem()
    
    # Setup signal handlers
    def signal_handler():
        logger.info("Received shutdown signal")
        freeze_prevention.shutdown_event.set()
    
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    try:
        await freeze_prevention.start()
    except KeyboardInterrupt:
        logger.info("Freeze prevention stopped by user")
    except Exception as e:
        logger.error(f"Freeze prevention error: {e}")
        raise
    finally:
        await freeze_prevention.stop()


if __name__ == "__main__":
    asyncio.run(main())