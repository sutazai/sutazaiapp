"""
Authorization and Control Module (ACM)
Secure authorization, system control, and emergency shutdown capabilities
"""

import asyncio
import logging
import json
import time
import uuid
import hashlib
import hmac
import secrets
import subprocess
import os
import signal
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
import psutil

logger = logging.getLogger(__name__)

class PermissionLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    EMERGENCY = "emergency"

class SystemState(str, Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
    EMERGENCY_STOP = "emergency_stop"
    OFFLINE = "offline"

class ControlAction(str, Enum):
    START_SERVICE = "start_service"
    STOP_SERVICE = "stop_service"
    RESTART_SERVICE = "restart_service"
    UPDATE_CONFIG = "update_config"
    SHUTDOWN_SYSTEM = "shutdown_system"
    EMERGENCY_STOP = "emergency_stop"
    SELF_MODIFY = "self_modify"
    DATA_ACCESS = "data_access"

@dataclass
class AuthSession:
    """Authentication session"""
    session_id: str
    user_id: str
    user_email: str
    permission_level: PermissionLevel
    created_at: float
    expires_at: float
    last_activity: float
    mfa_verified: bool
    emergency_code_used: bool = False
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None

@dataclass
class PermissionGrant:
    """Permission grant record"""
    id: str
    user_id: str
    permission: str
    resource: str
    granted_by: str
    granted_at: float
    expires_at: Optional[float]
    conditions: Dict[str, Any]
    revoked: bool = False

@dataclass
class ControlCommand:
    """System control command"""
    id: str
    command: ControlAction
    parameters: Dict[str, Any]
    issued_by: str
    issued_at: float
    executed: bool = False
    executed_at: Optional[float] = None
    success: bool = False
    result: Dict[str, Any] = None
    emergency: bool = False

@dataclass
class ShutdownSequence:
    """System shutdown sequence"""
    id: str
    initiated_by: str
    initiated_at: float
    shutdown_type: str  # graceful, forced, emergency
    estimated_duration: float
    steps: List[Dict[str, Any]]
    current_step: int = 0
    completed: bool = False

class AuthorizationControlModule:
    """
    Authorization and Control Module (ACM)
    Manages all system authorization, control, and shutdown operations
    """
    
    # Hardcoded authorized super admin - only user who can shut down system
    SUPER_ADMIN = {
        "email": "chrissuta01@gmail.com",
        "name": "Chris Suta",
        "id": "chrissuta01",
        "permission_level": PermissionLevel.SUPER_ADMIN,
        "emergency_codes": []  # Will be generated
    }
    
    def __init__(self, data_dir: str = "/opt/sutazaiapp/data/acm"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # System state
        self.system_state = SystemState.INITIALIZING
        self.shutdown_in_progress = False
        self.emergency_stop_triggered = False
        
        # Security components
        self.encryption_key = self._generate_or_load_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        self.jwt_secret = self._generate_or_load_jwt_secret()
        
        # Session management
        self.active_sessions = {}  # session_id -> AuthSession
        self.permission_grants = {}  # grant_id -> PermissionGrant
        self.control_commands = {}  # command_id -> ControlCommand
        
        # Emergency systems
        self.emergency_codes = self._generate_emergency_codes()
        self.emergency_contacts = [self.SUPER_ADMIN["email"]]
        self.shutdown_callbacks = []
        
        # System monitoring
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "active_processes": 0,
            "last_check": time.time()
        }
        
        # Audit logging
        self.audit_log = []
        self.max_audit_entries = 10000
        
        # Initialize
        self._load_persistent_data()
        self._initialize_super_admin()
        self._start_monitoring()
        self.system_state = SystemState.RUNNING
        
        logger.info("✅ Authorization and Control Module initialized")
    
    def _generate_or_load_encryption_key(self) -> bytes:
        """Generate or load encryption key"""
        key_file = self.data_dir / "encryption.key"
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            return key
    
    def _generate_or_load_jwt_secret(self) -> str:
        """Generate or load JWT secret"""
        secret_file = self.data_dir / "jwt.secret"
        if secret_file.exists():
            with open(secret_file, 'r') as f:
                return f.read().strip()
        else:
            secret = secrets.token_urlsafe(64)
            with open(secret_file, 'w') as f:
                f.write(secret)
            os.chmod(secret_file, 0o600)
            return secret
    
    def _generate_emergency_codes(self) -> List[str]:
        """Generate emergency access codes"""
        codes = []
        for _ in range(5):  # Generate 5 emergency codes
            code = secrets.token_urlsafe(16)
            codes.append(code)
        
        # Store encrypted emergency codes
        encrypted_codes = [self.cipher.encrypt(code.encode()).decode() for code in codes]
        self.SUPER_ADMIN["emergency_codes"] = encrypted_codes
        
        return codes
    
    def _load_persistent_data(self):
        """Load persistent data from disk"""
        try:
            # Load permission grants
            grants_file = self.data_dir / "permissions.json"
            if grants_file.exists():
                with open(grants_file, 'r') as f:
                    data = json.load(f)
                    for grant_data in data.get("grants", []):
                        grant = PermissionGrant(**grant_data)
                        self.permission_grants[grant.id] = grant
            
            # Load audit log
            audit_file = self.data_dir / "audit.json"
            if audit_file.exists():
                with open(audit_file, 'r') as f:
                    data = json.load(f)
                    self.audit_log = data.get("entries", [])[-self.max_audit_entries:]
            
            logger.info("✅ Persistent ACM data loaded")
            
        except Exception as e:
            logger.error(f"Failed to load persistent data: {e}")
    
    def _initialize_super_admin(self):
        """Initialize super admin with all permissions"""
        try:
            # Grant all permissions to super admin
            permissions = [
                "system:shutdown",
                "system:restart", 
                "system:modify",
                "data:access:all",
                "config:modify:all",
                "users:manage",
                "emergency:access"
            ]
            
            for permission in permissions:
                grant_id = str(uuid.uuid4())
                grant = PermissionGrant(
                    id=grant_id,
                    user_id=self.SUPER_ADMIN["id"],
                    permission=permission,
                    resource="*",
                    granted_by="system",
                    granted_at=time.time(),
                    expires_at=None,  # Never expires
                    conditions={"hardcoded": True}
                )
                self.permission_grants[grant_id] = grant
            
            self._audit_log("INITIALIZATION", "Super admin permissions granted", {
                "user": self.SUPER_ADMIN["email"],
                "permissions": permissions
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize super admin: {e}")
    
    def _start_monitoring(self):
        """Start system monitoring"""
        def monitor_loop():
            while self.system_state != SystemState.OFFLINE:
                try:
                    self._update_system_metrics()
                    self._check_system_health()
                    self._cleanup_expired_sessions()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _update_system_metrics(self):
        """Update system metrics"""
        try:
            self.system_metrics.update({
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_processes": len(psutil.pids()),
                "last_check": time.time()
            })
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def _check_system_health(self):
        """Check system health and trigger alerts if needed"""
        try:
            metrics = self.system_metrics
            
            # Check for critical conditions
            if metrics["cpu_usage"] > 95:
                self._trigger_alert("HIGH_CPU", f"CPU usage: {metrics['cpu_usage']}%")
            
            if metrics["memory_usage"] > 95:
                self._trigger_alert("HIGH_MEMORY", f"Memory usage: {metrics['memory_usage']}%")
            
            if metrics["disk_usage"] > 95:
                self._trigger_alert("HIGH_DISK", f"Disk usage: {metrics['disk_usage']}%")
                
        except Exception as e:
            logger.error(f"Failed to check system health: {e}")
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger system alert"""
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": time.time(),
            "metrics": self.system_metrics.copy()
        }
        
        self._audit_log("ALERT", alert_type, alert_data)
        logger.warning(f"🚨 SYSTEM ALERT: {alert_type} - {message}")
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time > session.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            self._audit_log("SESSION_EXPIRED", "Session expired", {"session_id": session_id})
    
    async def authenticate_user(self, email: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user and create session"""
        try:
            # Check if user is the super admin
            if email != self.SUPER_ADMIN["email"]:
                return {
                    "success": False,
                    "error": "Only the authorized super admin can access this system",
                    "authorized_user": self.SUPER_ADMIN["email"]
                }
            
            # Verify credentials (simplified - in production would check password/MFA)
            auth_method = credentials.get("method", "password")
            
            if auth_method == "emergency_code":
                emergency_code = credentials.get("emergency_code")
                if not self._verify_emergency_code(emergency_code):
                    self._audit_log("AUTH_FAILED", "Invalid emergency code", {"email": email})
                    return {"success": False, "error": "Invalid emergency code"}
            
            # Create session
            session_id = str(uuid.uuid4())
            current_time = time.time()
            
            session = AuthSession(
                session_id=session_id,
                user_id=self.SUPER_ADMIN["id"],
                user_email=email,
                permission_level=PermissionLevel.SUPER_ADMIN,
                created_at=current_time,
                expires_at=current_time + 3600,  # 1 hour
                last_activity=current_time,
                mfa_verified=auth_method == "mfa",
                emergency_code_used=auth_method == "emergency_code",
                device_fingerprint=credentials.get("device_fingerprint"),
                ip_address=credentials.get("ip_address")
            )
            
            self.active_sessions[session_id] = session
            
            # Generate JWT token
            token_payload = {
                "session_id": session_id,
                "user_id": session.user_id,
                "email": email,
                "permission_level": session.permission_level.value,
                "exp": session.expires_at,
                "iat": current_time
            }
            
            token = jwt.encode(token_payload, self.jwt_secret, algorithm="HS256")
            
            self._audit_log("AUTH_SUCCESS", "User authenticated", {
                "email": email,
                "method": auth_method,
                "session_id": session_id
            })
            
            return {
                "success": True,
                "session_id": session_id,
                "token": token,
                "expires_at": session.expires_at,
                "user": {
                    "id": session.user_id,
                    "email": email,
                    "permission_level": session.permission_level.value
                }
            }
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self._audit_log("AUTH_ERROR", "Authentication error", {"email": email, "error": str(e)})
            return {"success": False, "error": str(e)}
    
    def _verify_emergency_code(self, code: str) -> bool:
        """Verify emergency access code"""
        try:
            encrypted_codes = self.SUPER_ADMIN["emergency_codes"]
            for encrypted_code in encrypted_codes:
                try:
                    decrypted_code = self.cipher.decrypt(encrypted_code.encode()).decode()
                    if hmac.compare_digest(code, decrypted_code):
                        return True
                except:
                    continue
            return False
        except Exception as e:
            logger.error(f"Failed to verify emergency code: {e}")
            return False
    
    async def check_permission(self, session_id: str, action: ControlAction, resource: str = "*") -> Dict[str, Any]:
        """Check if user has permission for an action"""
        try:
            # Validate session
            session = self.active_sessions.get(session_id)
            if not session:
                return {"authorized": False, "error": "Invalid session"}
            
            # Update last activity
            session.last_activity = time.time()
            
            # Check if session expired
            if time.time() > session.expires_at:
                del self.active_sessions[session_id]
                return {"authorized": False, "error": "Session expired"}
            
            # Super admin has all permissions
            if session.permission_level == PermissionLevel.SUPER_ADMIN:
                return {"authorized": True, "user": session.user_email}
            
            # Check specific permissions
            required_permission = self._get_required_permission(action, resource)
            has_permission = self._user_has_permission(session.user_id, required_permission, resource)
            
            result = {
                "authorized": has_permission,
                "user": session.user_email,
                "permission_checked": required_permission
            }
            
            if not has_permission:
                result["error"] = f"Insufficient permissions for {action.value}"
            
            return result
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return {"authorized": False, "error": str(e)}
    
    def _get_required_permission(self, action: ControlAction, resource: str) -> str:
        """Get required permission for an action"""
        permission_map = {
            ControlAction.START_SERVICE: "system:control",
            ControlAction.STOP_SERVICE: "system:control", 
            ControlAction.RESTART_SERVICE: "system:control",
            ControlAction.UPDATE_CONFIG: "config:modify",
            ControlAction.SHUTDOWN_SYSTEM: "system:shutdown",
            ControlAction.EMERGENCY_STOP: "emergency:access",
            ControlAction.SELF_MODIFY: "system:modify",
            ControlAction.DATA_ACCESS: "data:access"
        }
        return permission_map.get(action, "system:basic")
    
    def _user_has_permission(self, user_id: str, permission: str, resource: str) -> bool:
        """Check if user has specific permission"""
        for grant in self.permission_grants.values():
            if (grant.user_id == user_id and 
                not grant.revoked and
                (grant.expires_at is None or time.time() < grant.expires_at) and
                (grant.permission == permission or grant.permission.endswith(":*")) and
                (grant.resource == "*" or grant.resource == resource)):
                return True
        return False
    
    async def execute_control_command(self, session_id: str, action: ControlAction, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a system control command"""
        try:
            # Check authorization
            auth_result = await self.check_permission(session_id, action)
            if not auth_result.get("authorized"):
                return {
                    "success": False,
                    "error": auth_result.get("error", "Unauthorized"),
                    "required_user": self.SUPER_ADMIN["email"]
                }
            
            session = self.active_sessions[session_id]
            command_id = str(uuid.uuid4())
            
            # Create command record
            command = ControlCommand(
                id=command_id,
                command=action,
                parameters=parameters or {},
                issued_by=session.user_email,
                issued_at=time.time(),
                emergency=action == ControlAction.EMERGENCY_STOP
            )
            
            self.control_commands[command_id] = command
            
            # Execute command
            result = await self._execute_command(command)
            
            # Update command with result
            command.executed = True
            command.executed_at = time.time()
            command.success = result.get("success", False)
            command.result = result
            
            self._audit_log("COMMAND_EXECUTED", f"Control command: {action.value}", {
                "command_id": command_id,
                "user": session.user_email,
                "success": command.success,
                "result": result
            })
            
            return {
                "success": command.success,
                "command_id": command_id,
                "result": result,
                "executed_at": command.executed_at
            }
            
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_command(self, command: ControlCommand) -> Dict[str, Any]:
        """Execute the actual command"""
        try:
            action = command.command
            params = command.parameters
            
            if action == ControlAction.SHUTDOWN_SYSTEM:
                return await self._execute_shutdown(params.get("shutdown_type", "graceful"))
            
            elif action == ControlAction.EMERGENCY_STOP:
                return await self._execute_emergency_stop()
            
            elif action == ControlAction.START_SERVICE:
                service_name = params.get("service_name")
                return await self._start_service(service_name)
            
            elif action == ControlAction.STOP_SERVICE:
                service_name = params.get("service_name")
                return await self._stop_service(service_name)
            
            elif action == ControlAction.RESTART_SERVICE:
                service_name = params.get("service_name")
                return await self._restart_service(service_name)
            
            elif action == ControlAction.UPDATE_CONFIG:
                config_data = params.get("config")
                return await self._update_config(config_data)
            
            elif action == ControlAction.SELF_MODIFY:
                modification_data = params.get("modification")
                return await self._self_modify(modification_data)
            
            else:
                return {"success": False, "error": f"Unknown command: {action.value}"}
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_shutdown(self, shutdown_type: str = "graceful") -> Dict[str, Any]:
        """Execute system shutdown"""
        try:
            if self.shutdown_in_progress:
                return {"success": False, "error": "Shutdown already in progress"}
            
            self.shutdown_in_progress = True
            self.system_state = SystemState.SHUTTING_DOWN
            
            shutdown_id = str(uuid.uuid4())
            
            # Define shutdown sequence
            shutdown_steps = [
                {"name": "Save system state", "duration": 5},
                {"name": "Stop AI agents", "duration": 10},
                {"name": "Save knowledge graph", "duration": 15},
                {"name": "Stop backend services", "duration": 10},
                {"name": "Close database connections", "duration": 5},
                {"name": "Final cleanup", "duration": 5}
            ]
            
            if shutdown_type == "forced":
                shutdown_steps = [
                    {"name": "Force stop all processes", "duration": 2},
                    {"name": "Emergency save", "duration": 3}
                ]
            
            sequence = ShutdownSequence(
                id=shutdown_id,
                initiated_by=command.issued_by,
                initiated_at=time.time(),
                shutdown_type=shutdown_type,
                estimated_duration=sum(step["duration"] for step in shutdown_steps),
                steps=shutdown_steps
            )
            
            logger.info(f"🛑 SYSTEM SHUTDOWN INITIATED: {shutdown_type.upper()}")
            logger.info(f"Estimated duration: {sequence.estimated_duration} seconds")
            
            # Execute shutdown steps
            for i, step in enumerate(shutdown_steps):
                sequence.current_step = i
                logger.info(f"Executing step {i+1}/{len(shutdown_steps)}: {step['name']}")
                
                try:
                    await self._execute_shutdown_step(step)
                    await asyncio.sleep(1)  # Brief pause between steps
                except Exception as e:
                    logger.error(f"Shutdown step failed: {step['name']} - {e}")
                    if shutdown_type == "graceful":
                        # In graceful shutdown, try to continue
                        continue
                    else:
                        # In forced shutdown, stop immediately
                        break
            
            sequence.completed = True
            sequence.current_step = len(shutdown_steps)
            
            # Final system shutdown
            self.system_state = SystemState.OFFLINE
            
            # Save final audit log
            self._audit_log("SYSTEM_SHUTDOWN", f"System shutdown completed: {shutdown_type}", {
                "shutdown_id": shutdown_id,
                "duration": time.time() - sequence.initiated_at,
                "steps_completed": sequence.current_step
            })
            
            # Call shutdown callbacks
            for callback in self.shutdown_callbacks:
                try:
                    await callback()
                except:
                    pass
            
            logger.info("✅ SYSTEM SHUTDOWN COMPLETED")
            
            return {
                "success": True,
                "shutdown_id": shutdown_id,
                "shutdown_type": shutdown_type,
                "duration": time.time() - sequence.initiated_at,
                "message": "System shutdown completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Shutdown execution failed: {e}")
            self.shutdown_in_progress = False
            self.system_state = SystemState.RUNNING
            return {"success": False, "error": str(e)}
    
    async def _execute_shutdown_step(self, step: Dict[str, Any]):
        """Execute a single shutdown step"""
        step_name = step["name"]
        
        if "save system state" in step_name.lower():
            await self._save_system_state()
        elif "stop ai agents" in step_name.lower():
            await self._stop_ai_agents()
        elif "save knowledge graph" in step_name.lower():
            await self._save_knowledge_graph()
        elif "stop backend services" in step_name.lower():
            await self._stop_backend_services()
        elif "close database" in step_name.lower():
            await self._close_database_connections()
        elif "cleanup" in step_name.lower():
            await self._final_cleanup()
        elif "force stop" in step_name.lower():
            await self._force_stop_processes()
        elif "emergency save" in step_name.lower():
            await self._emergency_save()
        
        # Simulate step duration
        await asyncio.sleep(min(step.get("duration", 1), 5))
    
    async def _execute_emergency_stop(self) -> Dict[str, Any]:
        """Execute emergency stop"""
        try:
            self.emergency_stop_triggered = True
            self.system_state = SystemState.EMERGENCY_STOP
            
            logger.critical("🚨 EMERGENCY STOP TRIGGERED")
            
            # Immediate actions
            await self._emergency_save()
            await self._force_stop_processes()
            
            # Set system to offline
            self.system_state = SystemState.OFFLINE
            
            self._audit_log("EMERGENCY_STOP", "Emergency stop executed", {
                "timestamp": time.time(),
                "triggered_by": "super_admin"
            })
            
            return {
                "success": True,
                "message": "Emergency stop executed",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _save_system_state(self):
        """Save current system state"""
        try:
            state_data = {
                "system_state": self.system_state.value,
                "active_sessions": len(self.active_sessions),
                "system_metrics": self.system_metrics,
                "shutdown_time": time.time()
            }
            
            with open(self.data_dir / "final_state.json", 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
    
    async def _stop_ai_agents(self):
        """Stop AI agents gracefully"""
        try:
            # Signal AI agents to stop
            logger.info("Stopping AI agents...")
            # Implementation would stop specific agent processes
            
        except Exception as e:
            logger.error(f"Failed to stop AI agents: {e}")
    
    async def _save_knowledge_graph(self):
        """Save knowledge graph"""
        try:
            # Trigger knowledge graph save
            from .kg import knowledge_graph
            await knowledge_graph.save_knowledge_graph()
            
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
    
    async def _stop_backend_services(self):
        """Stop backend services"""
        try:
            logger.info("Stopping backend services...")
            # Implementation would stop FastAPI and other services
            
        except Exception as e:
            logger.error(f"Failed to stop backend services: {e}")
    
    async def _close_database_connections(self):
        """Close database connections"""
        try:
            logger.info("Closing database connections...")
            # Implementation would close DB connections
            
        except Exception as e:
            logger.error(f"Failed to close database connections: {e}")
    
    async def _final_cleanup(self):
        """Perform final cleanup"""
        try:
            # Save audit log
            await self._save_audit_log()
            
            # Save permission grants
            await self._save_permission_grants()
            
            logger.info("Final cleanup completed")
            
        except Exception as e:
            logger.error(f"Final cleanup failed: {e}")
    
    async def _force_stop_processes(self):
        """Force stop all processes"""
        try:
            logger.warning("Force stopping processes...")
            # Implementation would forcefully terminate processes
            
        except Exception as e:
            logger.error(f"Failed to force stop processes: {e}")
    
    async def _emergency_save(self):
        """Emergency data save"""
        try:
            # Quick save of critical data
            emergency_data = {
                "timestamp": time.time(),
                "system_state": self.system_state.value,
                "active_sessions": list(self.active_sessions.keys()),
                "recent_commands": [cmd.id for cmd in list(self.control_commands.values())[-10:]]
            }
            
            with open(self.data_dir / "emergency_save.json", 'w') as f:
                json.dump(emergency_data, f, default=str)
                
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")
    
    async def _start_service(self, service_name: str) -> Dict[str, Any]:
        """Start a service"""
        try:
            logger.info(f"Starting service: {service_name}")
            # Implementation would start specific service
            return {"success": True, "service": service_name, "status": "started"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _stop_service(self, service_name: str) -> Dict[str, Any]:
        """Stop a service"""
        try:
            logger.info(f"Stopping service: {service_name}")
            # Implementation would stop specific service
            return {"success": True, "service": service_name, "status": "stopped"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _restart_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a service"""
        try:
            logger.info(f"Restarting service: {service_name}")
            # Implementation would restart specific service
            return {"success": True, "service": service_name, "status": "restarted"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _update_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update system configuration"""
        try:
            logger.info("Updating system configuration")
            # Implementation would update configuration
            return {"success": True, "updated_keys": list(config_data.keys())}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _self_modify(self, modification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform self-modification"""
        try:
            logger.info("Performing self-modification")
            # Implementation would perform controlled self-modification
            return {"success": True, "modifications": modification_data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _audit_log(self, event_type: str, description: str, details: Dict[str, Any] = None):
        """Add entry to audit log"""
        try:
            entry = {
                "id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "event_type": event_type,
                "description": description,
                "details": details or {},
                "system_state": self.system_state.value
            }
            
            self.audit_log.append(entry)
            
            # Keep only recent entries
            if len(self.audit_log) > self.max_audit_entries:
                self.audit_log = self.audit_log[-self.max_audit_entries:]
            
            # Log critical events
            if event_type in ["EMERGENCY_STOP", "SYSTEM_SHUTDOWN", "AUTH_FAILED"]:
                logger.critical(f"AUDIT: {event_type} - {description}")
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            return {
                "system_state": self.system_state.value,
                "shutdown_in_progress": self.shutdown_in_progress,
                "emergency_stop_triggered": self.emergency_stop_triggered,
                "active_sessions": len(self.active_sessions),
                "system_metrics": self.system_metrics.copy(),
                "authorized_user": self.SUPER_ADMIN["email"],
                "uptime": time.time() - (self.audit_log[0]["timestamp"] if self.audit_log else time.time()),
                "recent_commands": len([cmd for cmd in self.control_commands.values() 
                                     if time.time() - cmd.issued_at < 3600]),
                "audit_entries": len(self.audit_log)
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    async def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        try:
            return self.audit_log[-limit:]
        except Exception as e:
            logger.error(f"Failed to get audit log: {e}")
            return []
    
    async def _save_audit_log(self):
        """Save audit log to disk"""
        try:
            audit_data = {
                "entries": self.audit_log,
                "saved_at": time.time()
            }
            with open(self.data_dir / "audit.json", 'w') as f:
                json.dump(audit_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save audit log: {e}")
    
    async def _save_permission_grants(self):
        """Save permission grants to disk"""
        try:
            grants_data = {
                "grants": [asdict(grant) for grant in self.permission_grants.values()],
                "saved_at": time.time()
            }
            with open(self.data_dir / "permissions.json", 'w') as f:
                json.dump(grants_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save permission grants: {e}")
    
    def register_shutdown_callback(self, callback: Callable):
        """Register callback to be called during shutdown"""
        self.shutdown_callbacks.append(callback)
    
    def cleanup(self):
        """Cleanup ACM resources"""
        try:
            # Save all persistent data
            asyncio.create_task(self._save_audit_log())
            asyncio.create_task(self._save_permission_grants())
            
            logger.info("✅ ACM cleanup completed")
            
        except Exception as e:
            logger.error(f"ACM cleanup failed: {e}")

# Global instance
authorization_control_module = AuthorizationControlModule()

# Convenience functions
async def authenticate_user(email: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
    """Authenticate user"""
    return await authorization_control_module.authenticate_user(email, credentials)

async def check_permission(session_id: str, action: ControlAction, resource: str = "*") -> Dict[str, Any]:
    """Check user permission"""
    return await authorization_control_module.check_permission(session_id, action, resource)

async def execute_control_command(session_id: str, action: ControlAction, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute control command"""
    return await authorization_control_module.execute_control_command(session_id, action, parameters)

async def shutdown_system(session_id: str, shutdown_type: str = "graceful") -> Dict[str, Any]:
    """Shutdown system"""
    return await execute_control_command(session_id, ControlAction.SHUTDOWN_SYSTEM, {"shutdown_type": shutdown_type})

async def emergency_stop(session_id: str) -> Dict[str, Any]:
    """Emergency stop"""
    return await execute_control_command(session_id, ControlAction.EMERGENCY_STOP)

async def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    return await authorization_control_module.get_system_status()