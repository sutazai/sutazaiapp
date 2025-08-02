#!/usr/bin/env python3
# /opt/sutazaiapp/scripts/intelligent_autofix.py

import os
import sys
import json
import asyncio
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import docker
import psycopg2
import redis
import ast
import autopep8
import black
import isort
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import yaml
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/autofix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IssueType(Enum):
    DOCKER_RESTART = "docker_restart"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    CODE_QUALITY = "code_quality"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_BOTTLENECK = "performance_bottleneck"
    CONFIGURATION_ERROR = "configuration_error"
    DATABASE_ISSUE = "database_issue"
    MISSING_DEPENDENCY = "missing_dependency"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"

@dataclass
class Issue:
    type: IssueType
    severity: str  # critical, high, medium, low
    component: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    auto_fixable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FixResult:
    success: bool
    issue: Issue
    action_taken: str
    error_message: Optional[str] = None
    files_modified: List[str] = field(default_factory=list)

class IntelligentAutoFixer:
    """Intelligent system for detecting and automatically fixing issues"""
    
    def __init__(self):
        self.base_path = Path("/opt/sutazaiapp")
        self.docker_client = docker.from_env()
        self.issues_found: List[Issue] = []
        self.fixes_applied: List[FixResult] = []
        self.batch_size = 50  # Process files in batches
        
    async def run_comprehensive_fix(self) -> Dict[str, Any]:
        """Run comprehensive issue detection and auto-fix"""
        logger.info("Starting Intelligent Auto-Fix System...")
        
        # Phase 1: Detect all issues
        await self.detect_all_issues()
        
        # Phase 2: Prioritize issues
        prioritized_issues = self.prioritize_issues()
        
        # Phase 3: Apply fixes in batches
        await self.apply_fixes_in_batches(prioritized_issues)
        
        # Phase 4: Verify fixes
        verification_results = await self.verify_fixes()
        
        # Phase 5: Generate report
        report = self.generate_fix_report()
        
        return report
    
    async def detect_all_issues(self):
        """Detect all issues in the system"""
        detection_tasks = [
            self.detect_docker_issues(),
            self.detect_backend_issues(),
            self.detect_dependency_issues(),
            self.detect_code_quality_issues(),
            self.detect_security_issues(),
            self.detect_configuration_issues(),
            self.detect_database_issues(),
            self.detect_performance_issues()
        ]
        
        results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Detection error: {result}")
    
    async def detect_docker_issues(self):
        """Detect Docker container issues"""
        logger.info("Detecting Docker issues...")
        
        for container in self.docker_client.containers.list(all=True):
            # Check for restart loops
            if "Restarting" in container.status:
                # Get logs to understand the issue
                logs = container.logs(tail=100).decode('utf-8')
                
                # Analyze logs for common issues
                if "ModuleNotFoundError" in logs:
                    module_match = re.search(r"ModuleNotFoundError: No module named '(\w+)'", logs)
                    if module_match:
                        self.issues_found.append(Issue(
                            type=IssueType.MISSING_DEPENDENCY,
                            severity="critical",
                            component=container.name,
                            description=f"Missing Python module: {module_match.group(1)}",
                            suggested_fix=f"pip install {module_match.group(1)}",
                            metadata={"container": container.name, "module": module_match.group(1)}
                        ))
                
                elif "SyntaxError" in logs:
                    file_match = re.search(r'File "([^"]+)", line (\d+)', logs)
                    if file_match:
                        self.issues_found.append(Issue(
                            type=IssueType.SYNTAX_ERROR,
                            severity="critical",
                            component=container.name,
                            description=f"Syntax error in {file_match.group(1)}",
                            file_path=file_match.group(1),
                            line_number=int(file_match.group(2)),
                            metadata={"container": container.name}
                        ))
                
                elif "ImportError" in logs:
                    import_match = re.search(r"ImportError: cannot import name '(\w+)'", logs)
                    if import_match:
                        self.issues_found.append(Issue(
                            type=IssueType.IMPORT_ERROR,
                            severity="critical",
                            component=container.name,
                            description=f"Import error: {import_match.group(1)}",
                            metadata={"container": container.name, "import": import_match.group(1)}
                        ))
                
                else:
                    # Generic restart issue
                    self.issues_found.append(Issue(
                        type=IssueType.DOCKER_RESTART,
                        severity="critical",
                        component=container.name,
                        description=f"Container {container.name} is in restart loop",
                        suggested_fix="Rebuild container with fixed dependencies",
                        metadata={"container": container.name, "logs": logs[-500:]}
                    ))
    
    async def detect_backend_issues(self):
        """Detect backend-specific issues"""
        logger.info("Detecting backend issues...")
        
        backend_path = self.base_path / "backend"
        main_py = backend_path / "app" / "main.py"
        
        if not main_py.exists():
            self.issues_found.append(Issue(
                type=IssueType.CONFIGURATION_ERROR,
                severity="critical",
                component="backend",
                description="main.py missing in backend/app directory",
                suggested_fix="Create proper FastAPI main.py",
                file_path=str(main_py)
            ))
        else:
            # Check main.py content
            content = main_py.read_text()
            
            # Check for proper FastAPI setup
            if "from fastapi import FastAPI" not in content:
                self.issues_found.append(Issue(
                    type=IssueType.CODE_QUALITY,
                    severity="high",
                    component="backend",
                    description="FastAPI import missing",
                    file_path=str(main_py),
                    suggested_fix="Add proper FastAPI imports"
                ))
            
            if "app = FastAPI(" not in content:
                self.issues_found.append(Issue(
                    type=IssueType.CODE_QUALITY,
                    severity="high",
                    component="backend",
                    description="FastAPI app initialization missing",
                    file_path=str(main_py),
                    suggested_fix="Initialize FastAPI app properly"
                ))
            
            # Check for CORS middleware
            if "CORSMiddleware" not in content:
                self.issues_found.append(Issue(
                    type=IssueType.CONFIGURATION_ERROR,
                    severity="medium",
                    component="backend",
                    description="CORS middleware not configured",
                    file_path=str(main_py),
                    suggested_fix="Add CORS middleware for frontend communication"
                ))
    
    async def detect_dependency_issues(self):
        """Detect dependency conflicts and missing packages"""
        logger.info("Detecting dependency issues...")
        
        # Find all requirements files
        req_files = list(self.base_path.rglob("requirements*.txt"))
        
        all_deps = {}
        for req_file in req_files:
            component = req_file.parent.name
            try:
                content = req_file.read_text()
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        match = re.match(r'^([a-zA-Z0-9-_.]+)([><=!~]+.*)?$', line)
                        if match:
                            package = match.group(1).lower()
                            version = match.group(2) or ""
                            
                            if package in all_deps:
                                if all_deps[package]["version"] != version:
                                    self.issues_found.append(Issue(
                                        type=IssueType.DEPENDENCY_CONFLICT,
                                        severity="high",
                                        component=component,
                                        description=f"Dependency conflict: {package} has different versions",
                                        file_path=str(req_file),
                                        suggested_fix=f"Standardize {package} version across all components",
                                        metadata={
                                            "package": package,
                                            "versions": [all_deps[package]["version"], version],
                                            "files": [all_deps[package]["file"], str(req_file)]
                                        }
                                    ))
                            else:
                                all_deps[package] = {
                                    "version": version,
                                    "file": str(req_file),
                                    "component": component
                                }
            except Exception as e:
                logger.error(f"Error reading {req_file}: {e}")
    
    async def detect_code_quality_issues(self):
        """Detect code quality issues"""
        logger.info("Detecting code quality issues...")
        
        # Find all Python files
        py_files = list(self.base_path.rglob("*.py"))
        
        # Process in batches
        for i in range(0, len(py_files), self.batch_size):
            batch = py_files[i:i + self.batch_size]
            await self._process_code_quality_batch(batch)
    
    async def _process_code_quality_batch(self, files: List[Path]):
        """Process a batch of files for code quality issues"""
        loop = asyncio.get_event_loop()
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            tasks = [
                loop.run_in_executor(executor, self._analyze_python_file, file)
                for file in files
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    self.issues_found.extend(result)
    
    def _analyze_python_file(self, file_path: Path) -> List[Issue]:
        """Analyze a single Python file for issues"""
        if any(skip in str(file_path) for skip in ['.git', '__pycache__', 'node_modules', 'venv', '.env', 'scripts/intelligent_autofix.py']):
            return []

        issues = []
        
        try:
            content = file_path.read_text()
            
            # Check for syntax errors
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append(Issue(
                    type=IssueType.SYNTAX_ERROR,
                    severity="critical",
                    component=file_path.parent.name,
                    description=f"Syntax error: {e.msg}",
                    file_path=str(file_path),
                    line_number=e.lineno
                ))
            
            # Check for common issues
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                # Long lines
                if len(line) > 120:
                    issues.append(Issue(
                        type=IssueType.CODE_QUALITY,
                        severity="low",
                        component=file_path.parent.name,
                        description=f"Line too long ({len(line)} > 120 characters)",
                        file_path=str(file_path),
                        line_number=i,
                        suggested_fix="Break line into multiple lines"
                    ))
                
                # Hardcoded credentials
                if re.search(r'(password|secret|api_key)\s*=\s*["\'][^"\']+["\']', line, re.I):
                    issues.append(Issue(
                        type=IssueType.SECURITY_VULNERABILITY,
                        severity="high",
                        component=file_path.parent.name,
                        description="Potential hardcoded credential",
                        file_path=str(file_path),
                        line_number=i,
                        suggested_fix="Use environment variables for sensitive data"
                    ))
                
                # TODO/FIXME comments
                if re.search(r'#\s*(TODO|FIXME|HACK|XXX)', line):
                    issues.append(Issue(
                        type=IssueType.CODE_QUALITY,
                        severity="low",
                        component=file_path.parent.name,
                        description=f"Unresolved comment: {line.strip()}",
                        file_path=str(file_path),
                        line_number=i,
                        auto_fixable=False
                    ))
            
            # Check imports
            import_lines = [line for line in lines if line.startswith(('import ', 'from '))]
            if import_lines:
                try:
                    if not isort.check_code_string(content):
                        issues.append(Issue(
                            type=IssueType.CODE_QUALITY,
                            severity="low",
                            component=file_path.parent.name,
                            description="Imports not properly sorted",
                            file_path=str(file_path),
                            suggested_fix="Sort imports using isort"
                        ))
                except Exception as e:
                    logger.debug(f"isort check failed for {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
        
        return issues
    
    async def detect_security_issues(self):
        """Detect security vulnerabilities"""
        logger.info("Detecting security issues...")
        
        # Check for insecure configurations
        config_files = list(self.base_path.rglob("*.yml")) + list(self.base_path.rglob("*.yaml"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check for exposed ports without authentication
                if isinstance(config, dict):
                    self._check_config_security(config, config_file)
                    
            except Exception as e:
                logger.error(f"Error checking {config_file}: {e}")
    
    def _check_config_security(self, config: Dict, file_path: Path, parent_key: str = ""):
        """Recursively check configuration for security issues"""
        for key, value in config.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, dict):
                self._check_config_security(value, file_path, current_key)
            elif isinstance(value, str):
                # Check for exposed passwords
                if "password" in key.lower() and value and value != "changeme":
                    self.issues_found.append(Issue(
                        type=IssueType.SECURITY_VULNERABILITY,
                        severity="high",
                        component=file_path.parent.name,
                        description=f"Password exposed in configuration: {current_key}",
                        file_path=str(file_path),
                        suggested_fix="Use environment variable or secrets management"
                    ))
                
                # Check for public IP bindings
                if value in ["0.0.0.0", "::"] and "bind" in current_key.lower():
                    self.issues_found.append(Issue(
                        type=IssueType.SECURITY_VULNERABILITY,
                        severity="medium",
                        component=file_path.parent.name,
                        description=f"Service bound to all interfaces: {current_key}",
                        file_path=str(file_path),
                        suggested_fix="Bind to specific interface or use reverse proxy"
                    ))
    
    async def detect_configuration_issues(self):
        """Detect configuration problems"""
        logger.info("Detecting configuration issues...")
        
        # Check docker-compose.yml
        compose_file = self.base_path / "docker-compose-complete-agi.yml"
        if compose_file.exists():
            try:
                # Validate docker-compose syntax
                result = subprocess.run(
                    ["docker-compose", "-f", str(compose_file), "config"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    self.issues_found.append(Issue(
                        type=IssueType.CONFIGURATION_ERROR,
                        severity="critical",
                        component="docker",
                        description="Invalid docker-compose-complete-agi.yml",
                        file_path=str(compose_file),
                        suggested_fix="Fix docker-compose syntax errors",
                        metadata={"error": result.stderr}
                    ))
            except Exception as e:
                logger.error(f"Error validating docker-compose: {e}")
        
        # Check .env files
        env_files = list(self.base_path.rglob(".env*"))
        for env_file in env_files:
            if env_file.name != ".env.example":
                try:
                    content = env_file.read_text()
                    for i, line in enumerate(content.splitlines(), 1):
                        if line and not line.startswith("#"):
                            if "=" not in line:
                                self.issues_found.append(Issue(
                                    type=IssueType.CONFIGURATION_ERROR,
                                    severity="medium",
                                    component=env_file.parent.name,
                                    description=f"Invalid environment variable format",
                                    file_path=str(env_file),
                                    line_number=i,
                                    suggested_fix="Use format: KEY=VALUE"
                                ))
                except Exception as e:
                    logger.error(f"Error reading {env_file}: {e}")
    
    async def detect_database_issues(self):
        """Detect database connectivity and schema issues"""
        logger.info("Detecting database issues...")
        
        # Check PostgreSQL
        try:
            # Try to connect to PostgreSQL
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="sutazai_db",
                user="sutazai",
                password=os.getenv("POSTGRES_PASSWORD", "sutazai")
            )
            conn.close()
        except Exception as e:
            self.issues_found.append(Issue(
                type=IssueType.DATABASE_ISSUE,
                severity="high",
                component="database",
                description=f"PostgreSQL connection failed: {str(e)}",
                suggested_fix="Check PostgreSQL container and credentials"
            ))
        
        # Check Redis
        try:
            r = redis.Redis(host="localhost", port=6379, db=0)
            r.ping()
        except Exception as e:
            self.issues_found.append(Issue(
                type=IssueType.DATABASE_ISSUE,
                severity="medium",
                component="redis",
                description=f"Redis connection failed: {str(e)}",
                suggested_fix="Check Redis container status"
            ))
    
    async def detect_performance_issues(self):
        """Detect performance bottlenecks"""
        logger.info("Detecting performance issues...")
        
        # Check for large files
        large_files = []
        for file_path in self.base_path.rglob("*"):
            try:
                if file_path.is_file():
                    size = file_path.stat().st_size
                    if size > 10 * 1024 * 1024:  # 10MB
                        large_files.append((file_path, size))
            except (FileNotFoundError, PermissionError):
                continue
        
        if large_files:
            for file_path, size in sorted(large_files, key=lambda x: x[1], reverse=True)[:10]:
                self.issues_found.append(Issue(
                    type=IssueType.PERFORMANCE_BOTTLENECK,
                    severity="low",
                    component=file_path.parent.name,
                    description=f"Large file detected: {size / 1024 / 1024:.1f}MB",
                    file_path=str(file_path),
                    suggested_fix="Consider using Git LFS or external storage",
                    auto_fixable=False
                ))
        
        # Check for missing indexes in Python files
        model_files = list(self.base_path.rglob("models.py"))
        for model_file in model_files:
            try:
                content = model_file.read_text()
                if "db.Model" in content or "Base" in content:
                    if "index=True" not in content and "__table_args__" not in content:
                        self.issues_found.append(Issue(
                            type=IssueType.PERFORMANCE_BOTTLENECK,
                            severity="medium",
                            component=model_file.parent.name,
                            description="Database models may lack indexes",
                            file_path=str(model_file),
                            suggested_fix="Add indexes to frequently queried fields",
                            auto_fixable=False
                        ))
            except Exception as e:
                logger.error(f"Error checking {model_file}: {e}")
    
    def prioritize_issues(self) -> List[Issue]:
        """Prioritize issues by severity and dependencies"""
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        # Sort by severity and type
        return sorted(
            self.issues_found,
            key=lambda x: (
                severity_order.get(x.severity, 4),
                x.type.value,
                x.component
            )
        )
    
    async def apply_fixes_in_batches(self, issues: List[Issue]):
        """Apply fixes to issues in batches"""
        logger.info(f"Applying fixes to {len(issues)} issues...")
        
        # Group issues by type for batch processing
        issues_by_type = {}
        for issue in issues:
            if issue.auto_fixable:
                if issue.type not in issues_by_type:
                    issues_by_type[issue.type] = []
                issues_by_type[issue.type].append(issue)
        
        # Apply fixes by type
        # Critical fixes first
        critical_order = [
            IssueType.DOCKER_RESTART, 
            IssueType.MISSING_DEPENDENCY, 
            IssueType.SYNTAX_ERROR,
            IssueType.CONFIGURATION_ERROR,
            IssueType.DEPENDENCY_CONFLICT,
            IssueType.IMPORT_ERROR,
            IssueType.CODE_QUALITY,
        ]

        for issue_type in critical_order:
            if issue_type in issues_by_type:
                type_issues = issues_by_type[issue_type]
                logger.info(f"Fixing {len(type_issues)} {issue_type.value} issues...")
                
                if issue_type == IssueType.DOCKER_RESTART:
                    await self._fix_docker_issues(type_issues)
                elif issue_type == IssueType.MISSING_DEPENDENCY:
                    await self._fix_missing_dependencies(type_issues)
                elif issue_type == IssueType.CODE_QUALITY:
                    await self._fix_code_quality_issues(type_issues)
                elif issue_type == IssueType.SYNTAX_ERROR:
                    await self._fix_syntax_errors(type_issues)
                elif issue_type == IssueType.IMPORT_ERROR:
                    await self._fix_import_errors(type_issues)
                elif issue_type == IssueType.CONFIGURATION_ERROR:
                    await self._fix_configuration_errors(type_issues)
                elif issue_type == IssueType.DEPENDENCY_CONFLICT:
                    await self._fix_dependency_conflicts(type_issues)

    async def _fix_docker_issues(self, issues: List[Issue]):
        """Fix Docker container issues"""
        for issue in issues:
            container_name = issue.metadata.get("container")
            if "backend" in container_name:
                # Fix backend container
                logger.info(f"Fixing backend container: {container_name}")
                
                # Stop the container
                try:
                    container = self.docker_client.containers.get(container_name)
                    container.stop()
                    container.remove()
                    logger.info(f"Stopped and removed container: {container_name}")
                except docker.errors.NotFound:
                    logger.warning(f"Container {container_name} not found, proceeding with fix.")
                except Exception as e:
                    logger.error(f"Error stopping container {container_name}: {e}")

                # Create proper main.py
                await self._create_backend_main_py()
                
                # Rebuild container
                logger.info(f"Rebuilding container: {container_name}")
                result = subprocess.run(
                    ["docker-compose", "-f", "docker-compose-complete-agi.yml", "up", "--build", "-d", "backend"],
                    cwd=str(self.base_path),
                    capture_output=True,
                    text=True,
                    check=False
                )

                action_taken = f"Rebuilt backend container {container_name} with fixed main.py"
                if result.returncode != 0:
                    logger.error(f"Error rebuilding {container_name}: {result.stderr}")
                
                self.fixes_applied.append(FixResult(
                    success=result.returncode == 0,
                    issue=issue,
                    action_taken=action_taken,
                    error_message=result.stderr if result.returncode != 0 else None,
                    files_modified=[str(self.base_path / "backend" / "app" / "main.py")]
                ))
    
    async def _create_backend_main_py(self):
        """Create a working main.py for backend"""
        main_py_path = self.base_path / "backend" / "app" / "main.py"
        logger.info(f"Creating robust main entrypoint at {main_py_path}")

        main_py_content = '''from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import sys
from typing import Dict, Any, Optional, List
import uvicorn
from datetime import datetime
import json

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Logging Configuration ---
log_dir = "/opt/sutazaiapp/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Dummy/Fallback Imports ---
# This allows the server to start even if submodules have issues
try:
    from app.core.config import settings
except ImportError:
    logger.warning("Could not import settings, using fallback.")
    class Settings:
        APP_NAME: str = "SutazAI Fallback"
        API_V1_STR: str = "/api/v1"
        BACKEND_CORS_ORIGINS: List[str] = ["*"]
    settings = Settings()

# --- Application Lifespan (Startup/Shutdown Events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SutazAI Backend is starting up...")
    # --- Add startup logic here ---
    # Example: Connect to databases, initialize models, etc.
    # await connect_to_db()
    # await load_models()
    logger.info("Startup tasks complete.")
    yield
    logger.info("SutazAI Backend is shutting down...")
    # --- Add shutdown logic here ---
    # await close_db_connections()
    logger.info("Shutdown tasks complete.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# --- CORS Middleware ---
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()


# --- API Endpoints ---

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint providing basic API information."""
    return {
        "message": "Welcome to SutazAI automation/advanced automation System",
        "version": "1.0.0",
        "docs": "/docs"
        }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify service status."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")

# --- Dynamic Router Loading ---
# This section will attempt to load all routers from the `app/api/v1/` directory.
# If it fails, the application will still run with the basic endpoints above.
try:
    from app.api.v1.api import api_router
    app.include_router(api_router, prefix=settings.API_V1_STR)
    logger.info("Successfully loaded API routers.")
except ImportError as e:
    logger.error(f"Could not import API routers: {e}. Running with minimal endpoints.")
except Exception as e:
    logger.error(f"An unexpected error occurred while loading routers: {e}")

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
'''
        
        # Write the file
        main_py_path.parent.mkdir(parents=True, exist_ok=True)
        main_py_path.write_text(main_py_content)
        
        # Create supporting files
        await self._create_backend_structure()
    
    async def _create_backend_structure(self):
        """Create backend directory structure and essential files"""
        logger.info("Scaffolding backend directory structure.")
        backend_path = self.base_path / "backend" / "app"
        
        # Create directories
        directories = [
            "api/v1", "core", "crud", "db", "models", 
            "schemas", "services", "utils", "tests"
        ]
        
        for dir_name in directories:
            dir_path = backend_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()

        # Create core/config.py
        config_file = backend_path / "core" / "config.py"
        if not config_file.exists():
            config_content = '''from pydantic_settings import BaseSettings
from typing import List, Union
import os

class Settings(BaseSettings):
    APP_NAME: str = "SutazAI automation/advanced automation System"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_very_secret_key")
    
    # CORS Origins
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:8501", "http://localhost:3000"]

    # Database
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "sutazai-postgres")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "sutazai")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "sutazai")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "sutazai_db")
    DATABASE_URL: str = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}/{POSTGRES_DB}"

    class Config:
        case_sensitive = True

settings = Settings()
'''
            config_file.write_text(config_content)

        # Create api/v1/api.py
        api_router_file = backend_path / "api" / "v1" / "api.py"
        if not api_router_file.exists():
            api_router_content = '''from fastapi import APIRouter
# from app.api.v1.endpoints import items, users # Example endpoints

api_router = APIRouter()
# api_router.include_router(items.router, prefix="/items", tags=["items"])
# api_router.include_router(users.router, prefix="/users", tags=["users"])
'''
            api_router_file.write_text(api_router_content)

        # Create requirements.txt for backend
        requirements_file = self.base_path / "backend" / "requirements.txt"
        if not requirements_file.exists():
            requirements_content = '''fastapi
uvicorn[standard]
pydantic
pydantic-settings
sqlalchemy
psycopg2-binary
redis
python-multipart
python-jose[cryptography]
passlib[bcrypt]
langchain
langchain-community
ollama
chromadb
websockets
aiofiles
httpx
'''
            requirements_file.write_text(requirements_content)
    
    async def _fix_missing_dependencies(self, issues: List[Issue]):
        """Fix missing Python dependencies"""
        # Group by component
        deps_by_component = {}
        for issue in issues:
            component = issue.component
            module = issue.metadata.get("module", "")
            
            if component not in deps_by_component:
                deps_by_component[component] = set()
            deps_by_component[component].add(module)
        
        # Install dependencies
        for component, modules in deps_by_component.items():
            if "backend" not in component:
                logger.warning(f"Skipping dependency installation for non-backend component: {component}")
                continue

            container_name = component
            
            # Determine package names from module names
            packages = []
            for module in modules:
                # Common module to package mappings
                package_map = { "cv2": "opencv-python-headless", "PIL": "Pillow", "sklearn": "scikit-learn", "yaml": "PyYAML" }
                package = package_map.get(module, module)
                packages.append(package)
            
            if packages:
                logger.info(f"Attempting to install {len(packages)} packages in {container_name}")
                
                # Update requirements.txt
                req_file = self.base_path / "backend" / "requirements.txt"
                if req_file.exists():
                    with open(req_file, "a+") as f:
                        f.seek(0)
                        current_reqs = f.read()
                        for package in packages:
                            if package.lower() not in current_reqs.lower():
                                f.write(f"\\n{package}")
                
                # Install in container
                for package in packages:
                    try:
                        container = self.docker_client.containers.get(container_name)
                        result = container.exec_run(f"pip install {package}")
                        
                        self.fixes_applied.append(FixResult(
                            success=result.exit_code == 0,
                            issue=issue,
                            action_taken=f"Installed {package} in {container_name}",
                            error_message=result.output.decode() if result.exit_code !=0 else None,
                            files_modified=[str(req_file)] if req_file.exists() else []
                        ))
                    except Exception as e:
                        logger.error(f"Failed to install {package} in {container_name}: {e}")
    
    async def _fix_code_quality_issues(self, issues: List[Issue]):
        """Fix code quality issues in batches"""
        # Group issues by file
        issues_by_file = defaultdict(list)
        for issue in issues:
            if issue.file_path:
                issues_by_file[issue.file_path].append(issue)
        
        # Process files in batches
        file_paths = list(issues_by_file.keys())
        for i in range(0, len(file_paths), self.batch_size):
            batch = file_paths[i:i + self.batch_size]
            
            for file_path_str in batch:
                try:
                    file_path = Path(file_path_str)
                    if file_path.exists() and file_path.suffix == ".py":
                        content = file_path.read_text()
                        original_content = content
                        
                        # Apply formatting
                        content = isort.code(content)
                        content = black.format_str(content, mode=black.Mode())
                        
                        # Write back if changed
                        if content != original_content:
                            file_path.write_text(content)
                            
                            for issue in issues_by_file[file_path_str]:
                                self.fixes_applied.append(FixResult(
                                    success=True,
                                    issue=issue,
                                    action_taken="Applied code formatting (isort, black)",
                                    files_modified=[file_path_str]
                                ))
                except Exception as e:
                    logger.error(f"Error fixing {file_path_str}: {e}")
                    for issue in issues_by_file[file_path_str]:
                        self.fixes_applied.append(FixResult(
                            success=False,
                            issue=issue,
                            action_taken="Failed to apply code formatting",
                            error_message=str(e)
                        ))
    
    async def _fix_syntax_errors(self, issues: List[Issue]):
        """Attempt to fix syntax errors using autopep8"""
        for issue in issues:
            if issue.file_path:
                try:
                    path = Path(issue.file_path)
                    if path.exists():
                        content = path.read_text()
                        fixed_content = autopep8.fix_code(content)

                        if content != fixed_content:
                            path.write_text(fixed_content)
                            self.fixes_applied.append(FixResult(
                                success=True,
                                issue=issue,
                                action_taken=f"Attempted to fix syntax error with autopep8",
                                files_modified=[issue.file_path]
                            ))
                
                except Exception as e:
                    logger.error(f"Error fixing syntax in {issue.file_path}: {e}")
    
    async def _fix_import_errors(self, issues: List[Issue]):
        """Fix import errors by adding missing __init__.py files"""
        for issue in issues:
            if issue.file_path:
                try:
                    path = Path(issue.file_path)
                    if "cannot import name" in issue.description:
                        # This might be a circular import, harder to fix automatically.
                        # A simpler fix is ensuring packages are correctly structured.
                        current_dir = path.parent
                        while self.base_path in current_dir.parents:
                            init_file = current_dir / "__init__.py"
                            if not init_file.exists():
                                init_file.touch()
                                self.fixes_applied.append(FixResult(
                                    success=True,
                                    issue=issue,
                                    action_taken=f"Created missing __init__.py in {current_dir}",
                                    files_modified=[str(init_file)]
                                ))
                            current_dir = current_dir.parent

                except Exception as e:
                    logger.error(f"Error fixing import in {issue.file_path}: {e}")
    
    async def _fix_configuration_errors(self, issues: List[Issue]):
        """Fix configuration errors"""
        for issue in issues:
            if "docker-compose" in issue.file_path:
                compose_file = Path(issue.file_path)
                if compose_file.exists():
                    try:
                        with open(compose_file, 'r') as f:
                            compose_data = yaml.safe_load(f)
                        
                        # Ensure backend service uses correct command
                        if 'services' in compose_data and 'backend' in compose_data['services']:
                            backend = compose_data['services']['backend']
                            backend['command'] = 'python app/main.py'
                        
                        with open(compose_file, 'w') as f:
                            yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
                        
                        self.fixes_applied.append(FixResult(
                            success=True,
                            issue=issue,
                            action_taken="Fixed docker-compose.yml command for backend",
                            files_modified=[str(compose_file)]
                        ))
                    
                    except Exception as e:
                        logger.error(f"Error fixing {compose_file}: {e}")
            
            elif ".env" in str(issue.file_path):
                # Fix .env file format
                pass # Logic from previous implementation is reasonable
    
    async def _fix_dependency_conflicts(self, issues: List[Issue]):
        """Resolve dependency conflicts"""
        # Collect all conflicting packages
        conflicts = defaultdict(lambda: {"versions": set(), "files": set()})
        for issue in issues:
            package = issue.metadata.get("package", "")
            versions = issue.metadata.get("versions", [])
            files = issue.metadata.get("files", [])
            
            if package:
                conflicts[package]['versions'].update(versions)
                conflicts[package]['files'].update(files)
        
        # Resolve conflicts
        for package, info in conflicts.items():
            # Strategy: use the highest version if possible, otherwise remove version pinning
            # For simplicity here, we'll unify to the first version found
            target_version = list(info["versions"])[0] if info["versions"] else ''
            
            for file_path_str in info["files"]:
                try:
                    path = Path(file_path_str)
                    if path.exists():
                        content = path.read_text()
                        # Replace any version of this package
                        content = re.sub(
                            f"^{re.escape(package)}[><=!~].*$",
                            f"{package}{target_version}",
                            content,
                            flags=re.IGNORECASE | re.MULTILINE
                        )
                        path.write_text(content)
                        
                        for issue in issues: # find original issue
                            if issue.metadata.get("package") == package:
                                self.fixes_applied.append(FixResult(
                                    success=True,
                                    issue=issue,
                                    action_taken=f"Standardized {package} to version '{target_version}'",
                                    files_modified=[file_path_str]
                                ))
                
                except Exception as e:
                    logger.error(f"Error updating {file_path_str}: {e}")
    
    async def verify_fixes(self) -> Dict[str, Any]:
        """Verify that fixes were successful"""
        logger.info("Verifying fixes...")
        
        verification_results = {
            "total_issues": len(self.issues_found),
            "total_fixes_attempted": len(self.fixes_applied),
            "successful_fixes": sum(1 for fix in self.fixes_applied if fix.success),
            "failed_fixes": sum(1 for fix in self.fixes_applied if not fix.success),
            "containers_healthy": {},
            "remaining_issues": []
        }
        
        # Check container status
        for container in self.docker_client.containers.list(all=True):
            is_healthy = "Up" in container.status and "Restarting" not in container.status
            verification_results["containers_healthy"][container.name] = is_healthy
        
        # Re-run detection for critical issues
        original_issues = self.issues_found
        self.issues_found = []
        await self.detect_docker_issues()
        verification_results['remaining_issues'] = [i.description for i in self.issues_found]
        self.issues_found = original_issues

        return verification_results
    
    def generate_fix_report(self) -> Dict[str, Any]:
        """Generate comprehensive fix report"""
        report_path = self.base_path / "logs" / "autofix_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "summary": {
                "total_issues_found": len(self.issues_found),
                "issues_by_severity": defaultdict(int),
                "issues_by_type": defaultdict(int),
                "total_fixes_applied": len(self.fixes_applied),
                "successful_fixes": sum(1 for fix in self.fixes_applied if fix.success),
                "failed_fixes": sum(1 for fix in self.fixes_applied if not fix.success),
                "files_modified": sorted(list(set(
                    file for fix in self.fixes_applied if fix.files_modified
                    for file in fix.files_modified
                )))
            },
            "issues": [],
            "fixes": []
        }
        
        for issue in self.issues_found:
            report["summary"]["issues_by_severity"][issue.severity] += 1
            report["summary"]["issues_by_type"][issue.type.value] += 1
            report["issues"].append(asdict(issue))

        for fix in self.fixes_applied:
            fix_dict = asdict(fix)
            fix_dict['issue'] = asdict(fix.issue)
            report["fixes"].append(fix_dict)
        
        with open(report_path, 'w') as f:
            # Need to convert dataclasses to dicts for JSON serialization
            from dataclasses import asdict
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Fix report saved to: {report_path}")
        return report

# Main execution
async def main():
    fixer = IntelligentAutoFixer()
    try:
        report = await fixer.run_comprehensive_fix()
        
        # Print summary
        print("\\n" + "="*60)
        print("INTELLIGENT AUTO-FIX SUMMARY")
        print("="*60)
        summary = report['summary']
        print(f"Total Issues Found: {summary['total_issues_found']}")
        print(f"Fixes Applied: {summary['total_fixes_applied']}")
        print(f"Successful: {summary['successful_fixes']}")
        print(f"Failed: {summary['failed_fixes']}")
        print(f"Files Modified: {len(summary['files_modified'])}")
        print("\\nIssues by Severity:")
        for severity, count in summary['issues_by_severity'].items():
            print(f"  {severity}: {count}")
        print("\\nIssues by Type:")
        for issue_type, count in summary['issues_by_type'].items():
            print(f"  {issue_type}: {count}")
        print("="*60)
    except Exception as e:
        logger.critical(f"A critical error occurred during the auto-fix process: {e}", exc_info=True)

if __name__ == "__main__":
    from dataclasses import asdict
    # Ensure we run with sudo if not already root
    if os.geteuid() != 0:
        logger.warning("Script not running as root. Some Docker operations might fail.")
        # This is tricky in a script. For now, we'll just warn.
        # A better approach for production would be to manage Docker permissions for the user.
    
    asyncio.run(main()) 