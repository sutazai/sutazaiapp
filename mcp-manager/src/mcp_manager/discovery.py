"""
Server Discovery Engine for MCP Management System

Automatically discovers MCP servers from configuration files,
validates their setup, and maintains an up-to-date registry.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import toml
import yaml
from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .models import ConnectionType, ServerConfig, ServerType


class ConfigFileWatcher(FileSystemEventHandler):
    """Watches for changes in configuration files"""
    
    def __init__(self, discovery_engine: 'ServerDiscoveryEngine') -> None:
        self.discovery_engine = discovery_engine
        self.watched_files: Set[Path] = set()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path in self.watched_files:
            logger.info(f"Configuration file changed: {file_path}")
            self.discovery_engine._schedule_discovery()
    
    def add_watched_file(self, file_path: Path) -> None:
        """Add file to watch list"""
        self.watched_files.add(file_path)
    
    def remove_watched_file(self, file_path: Path) -> None:
        """Remove file from watch list"""
        self.watched_files.discard(file_path)


class ServerDiscoveryEngine:
    """
    Discovers and validates MCP servers from various configuration sources.
    
    Supports discovery from:
    - .mcp.json files (official format)
    - pyproject.toml files (Python projects)
    - package.json files (Node.js projects)
    - Custom YAML configuration files
    - Directory scanning for executable scripts
    """
    
    def __init__(self, config_directories: List[Path]) -> None:
        self.config_directories = config_directories
        self.discovered_servers: Dict[str, ServerConfig] = {}
        self._last_discovery_time: Optional[float] = None
        
        # File watching
        self._observer = Observer()
        self._file_watcher = ConfigFileWatcher(self)
        self._watching = False
        
        # Discovery patterns
        self._config_patterns = {
            '*.mcp.json': self._parse_mcp_json,
            '.mcp.json': self._parse_mcp_json,
            'pyproject.toml': self._parse_pyproject_toml,
            'package.json': self._parse_package_json,
            'mcp-config.yaml': self._parse_yaml_config,
            'mcp-servers.yaml': self._parse_yaml_config,
        }
        
        # Server type detection patterns
        self._type_patterns = {
            ServerType.PYTHON: ['.py', 'python', 'python3', 'pip', 'poetry'],
            ServerType.NODE: ['.js', '.ts', 'node', 'npm', 'npx', 'yarn'],
            ServerType.GO: ['.go', 'go'],
            ServerType.RUST: ['.rs', 'cargo'],
            ServerType.SHELL: ['.sh', '.bash', '.zsh', 'bash', 'sh'],
        }
    
    async def discover_servers(self, force_refresh: bool = False) -> Dict[str, ServerConfig]:
        """
        Discover all MCP servers from configured sources.
        
        Args:
            force_refresh: Force rediscovery even if recently discovered
            
        Returns:
            Dictionary of discovered server configurations
        """
        import time
        current_time = time.time()
        
        # Skip if recently discovered and not forcing
        if not force_refresh and self._last_discovery_time:
            if current_time - self._last_discovery_time < 30:  # 30 second cache
                return self.discovered_servers.copy()
        
        logger.info("Starting MCP server discovery")
        discovered = {}
        
        try:
            # Discover from each configured directory
            for config_dir in self.config_directories:
                if not config_dir.exists():
                    logger.warning(f"Configuration directory does not exist: {config_dir}")
                    continue
                
                logger.debug(f"Scanning directory: {config_dir}")
                dir_servers = await self._discover_from_directory(config_dir)
                
                # Merge discovered servers, handling conflicts
                for name, config in dir_servers.items():
                    if name in discovered:
                        logger.warning(f"Duplicate server name '{name}' found, using first occurrence")
                    else:
                        discovered[name] = config
            
            # Validate discovered servers
            validated_servers = {}
            for name, config in discovered.items():
                try:
                    validation_result = await self._validate_server_config(config)
                    if validation_result[0]:
                        validated_servers[name] = config
                        logger.debug(f"Validated server: {name}")
                    else:
                        logger.error(f"Server validation failed for '{name}': {validation_result[1]}")
                except Exception as e:
                    logger.error(f"Error validating server '{name}': {e}")
            
            self.discovered_servers = validated_servers
            self._last_discovery_time = current_time
            
            logger.success(f"Discovered {len(validated_servers)} valid MCP servers")
            
            # Start file watching if not already watching
            if not self._watching and validated_servers:
                await self._start_file_watching()
            
            return self.discovered_servers.copy()
            
        except Exception as e:
            logger.error(f"Server discovery failed: {e}")
            return {}
    
    async def _discover_from_directory(self, directory: Path) -> Dict[str, ServerConfig]:
        """Discover servers from a single directory"""
        discovered = {}
        
        try:
            # Scan for configuration files matching patterns
            for pattern, parser in self._config_patterns.items():
                for config_file in directory.rglob(pattern):
                    try:
                        logger.debug(f"Parsing config file: {config_file}")
                        servers = await parser(config_file)
                        
                        for server_config in servers:
                            discovered[server_config.name] = server_config
                            
                        # Add to watched files
                        self._file_watcher.add_watched_file(config_file)
                        
                    except Exception as e:
                        logger.error(f"Failed to parse {config_file}: {e}")
            
            # Scan for executable scripts that might be MCP servers
            await self._discover_executable_scripts(directory, discovered)
            
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return discovered
    
    async def _parse_mcp_json(self, config_file: Path) -> List[ServerConfig]:
        """Parse official .mcp.json configuration files"""
        servers = []
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            mcp_servers = data.get('mcpServers', {})
            
            for server_name, server_data in mcp_servers.items():
                try:
                    # Extract basic information
                    command = server_data.get('command', '')
                    args = server_data.get('args', [])
                    connection_type = ConnectionType(server_data.get('type', 'stdio'))
                    
                    # Determine server type from command
                    server_type = self._detect_server_type(command, args)
                    
                    # Create server configuration
                    config = ServerConfig(
                        name=server_name,
                        description=server_data.get('description', f"MCP Server: {server_name}"),
                        command=command,
                        args=args,
                        connection_type=connection_type,
                        server_type=server_type,
                        working_directory=config_file.parent,
                        environment=server_data.get('env', {}),
                        enabled=server_data.get('enabled', True)
                    )
                    
                    # Add additional configuration if present
                    if 'host' in server_data:
                        config.host = server_data['host']
                    if 'port' in server_data:
                        config.port = server_data['port']
                    if 'timeout' in server_data:
                        config.request_timeout = server_data['timeout']
                    
                    servers.append(config)
                    
                except Exception as e:
                    logger.error(f"Failed to parse server '{server_name}' from {config_file}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to parse MCP JSON file {config_file}: {e}")
        
        return servers
    
    async def _parse_pyproject_toml(self, config_file: Path) -> List[ServerConfig]:
        """Parse Python pyproject.toml files for MCP server definitions"""
        servers = []
        
        try:
            with open(config_file, 'r') as f:
                data = toml.load(f)
            
            # Check for MCP server definitions
            project = data.get('project', {})
            scripts = project.get('scripts', {})
            
            # Look for MCP-related scripts
            for script_name, script_command in scripts.items():
                if 'mcp' in script_name.lower() or 'mcp' in script_command.lower():
                    config = ServerConfig(
                        name=f"{project.get('name', config_file.stem)}_{script_name}",
                        description=f"Python MCP Server: {script_name}",
                        command="python",
                        args=["-m", script_command.replace(":", ".")],
                        connection_type=ConnectionType.STDIO,
                        server_type=ServerType.PYTHON,
                        working_directory=config_file.parent
                    )
                    servers.append(config)
            
            # Check for MCP tool definitions
            tool_config = data.get('tool', {}).get('mcp', {})
            if tool_config:
                servers_config = tool_config.get('servers', {})
                for server_name, server_data in servers_config.items():
                    config = ServerConfig(
                        name=server_name,
                        description=server_data.get('description', f"MCP Server: {server_name}"),
                        command=server_data.get('command', 'python'),
                        args=server_data.get('args', []),
                        connection_type=ConnectionType(server_data.get('type', 'stdio')),
                        server_type=ServerType.PYTHON,
                        working_directory=config_file.parent
                    )
                    servers.append(config)
                    
        except Exception as e:
            logger.error(f"Failed to parse pyproject.toml file {config_file}: {e}")
        
        return servers
    
    async def _parse_package_json(self, config_file: Path) -> List[ServerConfig]:
        """Parse Node.js package.json files for MCP server definitions"""
        servers = []
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            # Check scripts for MCP-related commands
            scripts = data.get('scripts', {})
            for script_name, script_command in scripts.items():
                if 'mcp' in script_name.lower():
                    config = ServerConfig(
                        name=f"{data.get('name', config_file.stem)}_{script_name}",
                        description=f"Node.js MCP Server: {script_name}",
                        command="npm",
                        args=["run", script_name],
                        connection_type=ConnectionType.STDIO,
                        server_type=ServerType.NODE,
                        working_directory=config_file.parent
                    )
                    servers.append(config)
            
            # Check for MCP-specific configuration
            mcp_config = data.get('mcp', {})
            if mcp_config:
                servers_config = mcp_config.get('servers', {})
                for server_name, server_data in servers_config.items():
                    config = ServerConfig(
                        name=server_name,
                        description=server_data.get('description', f"MCP Server: {server_name}"),
                        command=server_data.get('command', 'node'),
                        args=server_data.get('args', []),
                        connection_type=ConnectionType(server_data.get('type', 'stdio')),
                        server_type=ServerType.NODE,
                        working_directory=config_file.parent
                    )
                    servers.append(config)
                    
        except Exception as e:
            logger.error(f"Failed to parse package.json file {config_file}: {e}")
        
        return servers
    
    async def _parse_yaml_config(self, config_file: Path) -> List[ServerConfig]:
        """Parse YAML configuration files"""
        servers = []
        
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data:
                return servers
            
            # Handle different YAML structures
            servers_data = data.get('servers', data.get('mcp_servers', {}))
            
            for server_name, server_config in servers_data.items():
                try:
                    config = ServerConfig(
                        name=server_name,
                        description=server_config.get('description', f"MCP Server: {server_name}"),
                        command=server_config.get('command', ''),
                        args=server_config.get('args', []),
                        connection_type=ConnectionType(server_config.get('type', 'stdio')),
                        server_type=self._detect_server_type(
                            server_config.get('command', ''), 
                            server_config.get('args', [])
                        ),
                        working_directory=config_file.parent,
                        environment=server_config.get('environment', {}),
                        enabled=server_config.get('enabled', True)
                    )
                    
                    # Add optional fields
                    for field in ['host', 'port', 'timeout', 'auto_restart']:
                        if field in server_config:
                            setattr(config, field, server_config[field])
                    
                    servers.append(config)
                    
                except Exception as e:
                    logger.error(f"Failed to parse server '{server_name}' from YAML: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to parse YAML file {config_file}: {e}")
        
        return servers
    
    async def _discover_executable_scripts(self, directory: Path, discovered: Dict[str, ServerConfig]) -> None:
        """Discover executable scripts that might be MCP servers"""
        try:
            # Look in scripts/mcp directories
            mcp_scripts_dir = directory / "scripts" / "mcp"
            if mcp_scripts_dir.exists():
                for script_file in mcp_scripts_dir.rglob("*"):
                    if script_file.is_file() and os.access(script_file, os.X_OK):
                        # Skip if already discovered through config files
                        script_name = script_file.stem
                        if script_name not in discovered:
                            await self._analyze_script_for_mcp(script_file, discovered)
            
            # Look for wrapper scripts
            wrappers_dir = directory / "scripts" / "mcp" / "wrappers"
            if wrappers_dir.exists():
                for wrapper_script in wrappers_dir.glob("*.sh"):
                    script_name = wrapper_script.stem
                    if script_name not in discovered:
                        config = await self._parse_wrapper_script(wrapper_script)
                        if config:
                            discovered[script_name] = config
                            
        except Exception as e:
            logger.error(f"Error discovering executable scripts in {directory}: {e}")
    
    async def _parse_wrapper_script(self, script_file: Path) -> Optional[ServerConfig]:
        """Parse wrapper scripts to extract MCP server configuration"""
        try:
            with open(script_file, 'r') as f:
                content = f.read()
            
            # Extract configuration from script comments or variables
            server_name = script_file.stem
            description = ""
            command = ""
            server_type = ServerType.SHELL
            
            # Look for configuration in comments
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('# Purpose:'):
                    description = line.replace('# Purpose:', '').strip()
                elif 'MCP_COMMAND=' in line:
                    command = line.split('=', 1)[1].strip('"\'')
                elif 'python' in line.lower() and 'mcp' in line.lower():
                    server_type = ServerType.PYTHON
                elif 'node' in line.lower() or 'npm' in line.lower():
                    server_type = ServerType.NODE
            
            if not description:
                description = f"MCP Server wrapper: {server_name}"
            
            if not command:
                command = str(script_file)
            
            config = ServerConfig(
                name=server_name,
                description=description,
                command=command,
                args=[],
                connection_type=ConnectionType.STDIO,
                server_type=server_type,
                working_directory=script_file.parent
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to parse wrapper script {script_file}: {e}")
            return None
    
    def _detect_server_type(self, command: str, args: List[str]) -> ServerType:
        """Detect server type from command and arguments"""
        command_lower = command.lower()
        args_str = ' '.join(args).lower()
        
        for server_type, patterns in self._type_patterns.items():
            for pattern in patterns:
                if pattern in command_lower or pattern in args_str:
                    return server_type
        
        return ServerType.UNKNOWN
    
    async def _validate_server_config(self, config: ServerConfig) -> Tuple[bool, Optional[str]]:
        """Validate a server configuration"""
        try:
            # Check if command exists and is executable
            if config.server_type != ServerType.HTTP:
                command_path = self._find_executable(config.command)
                if not command_path:
                    return False, f"Command not found: {config.command}"
            
            # Validate working directory
            if config.working_directory and not config.working_directory.exists():
                return False, f"Working directory does not exist: {config.working_directory}"
            
            # Validate connection type specific requirements
            if config.connection_type in [ConnectionType.HTTP, ConnectionType.WEBSOCKET]:
                if not config.port and not config.base_url:
                    return False, f"HTTP/WebSocket servers require port or base_url"
            
            # Validate timeout values
            if config.startup_timeout <= 0:
                return False, "startup_timeout must be positive"
            
            if config.request_timeout <= 0:
                return False, "request_timeout must be positive"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _find_executable(self, command: str) -> Optional[str]:
        """Find executable in PATH or return absolute path if exists"""
        # If it's already an absolute path, check if it exists
        if os.path.isabs(command):
            return command if os.path.isfile(command) and os.access(command, os.X_OK) else None
        
        # Search in PATH
        for path in os.environ.get("PATH", "").split(os.pathsep):
            if path:
                exe_path = os.path.join(path, command)
                if os.path.isfile(exe_path) and os.access(exe_path, os.X_OK):
                    return exe_path
        
        return None
    
    async def _analyze_script_for_mcp(self, script_file: Path, discovered: Dict[str, ServerConfig]) -> None:
        """Analyze a script to determine if it's an MCP server"""
        try:
            # Read first few lines to check for MCP indicators
            with open(script_file, 'r') as f:
                first_lines = ''.join(f.readlines()[:20]).lower()
            
            # Check for MCP indicators
            mcp_indicators = ['mcp', 'model context protocol', 'fastmcp', 'mcp.server']
            
            if any(indicator in first_lines for indicator in mcp_indicators):
                script_name = script_file.stem
                
                config = ServerConfig(
                    name=script_name,
                    description=f"Discovered MCP server: {script_name}",
                    command=str(script_file),
                    args=[],
                    connection_type=ConnectionType.STDIO,
                    server_type=self._detect_server_type(str(script_file), []),
                    working_directory=script_file.parent
                )
                
                discovered[script_name] = config
                logger.debug(f"Discovered MCP script: {script_file}")
                
        except Exception as e:
            logger.debug(f"Could not analyze script {script_file}: {e}")
    
    async def _start_file_watching(self) -> None:
        """Start watching configuration files for changes"""
        try:
            # Watch each configuration directory
            for config_dir in self.config_directories:
                if config_dir.exists():
                    self._observer.schedule(
                        self._file_watcher, 
                        str(config_dir), 
                        recursive=True
                    )
            
            self._observer.start()
            self._watching = True
            logger.info("Started configuration file watching")
            
        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
    
    def _schedule_discovery(self) -> None:
        """Schedule a discovery run (called by file watcher)"""
        # This would typically trigger a background task
        # For now, we just log the event
        logger.info("Configuration change detected, discovery scheduled")
    
    async def stop_watching(self) -> None:
        """Stop file watching"""
        if self._watching:
            self._observer.stop()
            self._observer.join()
            self._watching = False
            logger.info("Stopped configuration file watching")
    
    def get_discovered_servers(self) -> Dict[str, ServerConfig]:
        """Get currently discovered servers"""
        return self.discovered_servers.copy()
    
    def get_server_by_name(self, name: str) -> Optional[ServerConfig]:
        """Get a specific server configuration by name"""
        return self.discovered_servers.get(name)