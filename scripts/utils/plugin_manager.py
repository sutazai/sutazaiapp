#!/usr/bin/env python3
"""
Plugin Manager - Manages JARVIS plugins for extensible functionality
"""

import asyncio
import json
import logging
import importlib.util
import sys
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class PluginManager:
    """Manages plugins for extending JARVIS functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plugin_dir = Path(config.get('plugin_dir', '/opt/sutazaiapp/services/jarvis/plugins'))
        self.auto_load = config.get('auto_load', True)
        self.enabled_plugins = set(config.get('enabled_plugins', []))
        
        # Plugin registry
        self.plugins: Dict[str, Dict[str, Any]] = {}
        self.plugin_instances: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialize plugin manager"""
        try:
            # Create plugin directory if it doesn't exist
            self.plugin_dir.mkdir(parents=True, exist_ok=True)
            
            # Create default plugins
            await self._create_default_plugins()
            
            # Auto-load plugins if enabled
            if self.auto_load:
                await self._discover_plugins()
                
            logger.info(f"Plugin manager initialized with {len(self.plugins)} plugins")
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin manager: {e}")
            
    async def shutdown(self):
        """Shutdown plugin manager"""
        try:
            # Shutdown all active plugins
            for plugin_name, instance in self.plugin_instances.items():
                if hasattr(instance, 'shutdown'):
                    await instance.shutdown()
                    
            logger.info("Plugin manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during plugin manager shutdown: {e}")
            
    async def _create_default_plugins(self):
        """Create default built-in plugins"""
        default_plugins = {
            'system_info': {
                'name': 'System Information',
                'description': 'Provides system information and status',
                'version': '1.0.0',
                'author': 'SutazAI',
                'commands': ['system status', 'system info', 'health check'],
                'builtin': True
            },
            'agent_manager': {
                'name': 'Agent Manager',
                'description': 'Manages AI agents and their status',
                'version': '1.0.0',
                'author': 'SutazAI',
                'commands': ['list agents', 'agent status', 'activate agent', 'deactivate agent'],
                'builtin': True
            },
            'task_tracker': {
                'name': 'Task Tracker',
                'description': 'Tracks and manages tasks and their progress',
                'version': '1.0.0',
                'author': 'SutazAI',
                'commands': ['list tasks', 'task status', 'cancel task'],
                'builtin': True
            },
            'voice_commands': {
                'name': 'Voice Commands',
                'description': 'Enhanced voice command processing',
                'version': '1.0.0',
                'author': 'SutazAI',
                'commands': ['voice settings', 'set voice', 'voice calibration'],
                'builtin': True
            }
        }
        
        for plugin_id, plugin_info in default_plugins.items():
            self.plugins[plugin_id] = plugin_info
            
    async def _discover_plugins(self):
        """Discover plugins in the plugin directory"""
        try:
            if not self.plugin_dir.exists():
                return
                
            for plugin_file in self.plugin_dir.glob('*.py'):
                if plugin_file.name.startswith('_'):
                    continue
                    
                try:
                    await self._load_plugin_file(plugin_file)
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error discovering plugins: {e}")
            
    async def _load_plugin_file(self, plugin_file: Path):
        """Load a plugin from file"""
        plugin_name = plugin_file.stem
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)
            
            # Get plugin info
            if hasattr(module, 'PLUGIN_INFO'):
                plugin_info = module.PLUGIN_INFO
                plugin_info['file_path'] = str(plugin_file)
                plugin_info['builtin'] = False
                
                self.plugins[plugin_name] = plugin_info
                
                # Auto-enable if in enabled list
                if plugin_name in self.enabled_plugins:
                    await self.enable_plugin(plugin_name)
                    
                logger.info(f"Loaded plugin: {plugin_name}")
                
        except Exception as e:
            logger.error(f"Error loading plugin file {plugin_file}: {e}")
            
    async def enable_plugin(self, plugin_name: str):
        """Enable a plugin"""
        try:
            if plugin_name not in self.plugins:
                raise ValueError(f"Plugin {plugin_name} not found")
                
            plugin_info = self.plugins[plugin_name]
            
            # Skip if already enabled
            if plugin_name in self.plugin_instances:
                logger.info(f"Plugin {plugin_name} already enabled")
                return
                
            # Load plugin instance for file-based plugins
            if not plugin_info.get('builtin', False):
                plugin_file = Path(plugin_info['file_path'])
                module_name = plugin_file.stem
                
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    
                    # Look for plugin class
                    plugin_class = None
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            hasattr(attr, 'execute') and 
                            attr_name != 'BasePlugin'):
                            plugin_class = attr
                            break
                            
                    if plugin_class:
                        instance = plugin_class()
                        if hasattr(instance, 'initialize'):
                            await instance.initialize()
                        self.plugin_instances[plugin_name] = instance
                        
            else:
                # Built-in plugin - create instance
                self.plugin_instances[plugin_name] = BuiltinPlugin(plugin_name, plugin_info)
                
            self.enabled_plugins.add(plugin_name)
            logger.info(f"Enabled plugin: {plugin_name}")
            
        except Exception as e:
            logger.error(f"Failed to enable plugin {plugin_name}: {e}")
            raise
            
    async def disable_plugin(self, plugin_name: str):
        """Disable a plugin"""
        try:
            if plugin_name not in self.enabled_plugins:
                logger.info(f"Plugin {plugin_name} not enabled")
                return
                
            # Shutdown plugin instance
            if plugin_name in self.plugin_instances:
                instance = self.plugin_instances[plugin_name]
                if hasattr(instance, 'shutdown'):
                    await instance.shutdown()
                del self.plugin_instances[plugin_name]
                
            self.enabled_plugins.discard(plugin_name)
            logger.info(f"Disabled plugin: {plugin_name}")
            
        except Exception as e:
            logger.error(f"Failed to disable plugin {plugin_name}: {e}")
            raise
            
    async def execute_plugin(self, plugin_name: str, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a plugin command"""
        try:
            if plugin_name not in self.plugin_instances:
                raise ValueError(f"Plugin {plugin_name} not enabled")
                
            instance = self.plugin_instances[plugin_name]
            
            # Execute the plugin
            result = await instance.execute(command, context or {})
            
            return {
                'plugin': plugin_name,
                'command': command,
                'result': result,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Plugin execution failed: {e}")
            return {
                'plugin': plugin_name,
                'command': command,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
            
    def is_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled"""
        return plugin_name in self.enabled_plugins
        
    def list_plugins(self) -> List[str]:
        """List all available plugins"""
        return list(self.plugins.keys())
        
    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """Get detailed plugin information"""
        plugin_list = []
        for name, info in self.plugins.items():
            plugin_list.append({
                'name': name,
                'display_name': info.get('name', name),
                'description': info.get('description', ''),
                'version': info.get('version', '1.0.0'),
                'author': info.get('author', 'Unknown'),
                'enabled': name in self.enabled_plugins,
                'builtin': info.get('builtin', False),
                'commands': info.get('commands', [])
            })
        return plugin_list
        
    async def get_plugin_commands(self, plugin_name: str) -> List[str]:
        """Get available commands for a plugin"""
        if plugin_name not in self.plugins:
            return []
            
        return self.plugins[plugin_name].get('commands', [])
        
    async def create_plugin_template(self, plugin_name: str) -> str:
        """Create a template for a new plugin"""
        template = f'''#!/usr/bin/env python3
"""
{plugin_name.title()} Plugin for JARVIS
"""

import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Plugin metadata
PLUGIN_INFO = {{
    'name': '{plugin_name.title()}',
    'description': 'Description of what this plugin does',
    'version': '1.0.0',
    'author': 'Your Name',
    'commands': ['command1', 'command2']
}}

class {plugin_name.title().replace('_', '')}Plugin:
    """Main plugin class"""
    
    def __init__(self):
        self.initialized = False
        
    async def initialize(self):
        """Initialize the plugin"""
        self.initialized = True
        logger.info(f"{{PLUGIN_INFO['name']}} plugin initialized")
        
    async def shutdown(self):
        """Shutdown the plugin"""
        self.initialized = False
        logger.info(f"{{PLUGIN_INFO['name']}} plugin shutdown")
        
    async def execute(self, command: str, context: Dict[str, Any]) -> Any:
        """Execute a plugin command"""
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
            
        command_lower = command.lower()
        
        if 'command1' in command_lower:
            return await self._handle_command1(context)
        elif 'command2' in command_lower:
            return await self._handle_command2(context)
        else:
            return {{"error": f"Unknown command: {{command}}"}}
            
    async def _handle_command1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command1"""
        return {{"message": "Command1 executed successfully"}}
        
    async def _handle_command2(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command2"""
        return {{"message": "Command2 executed successfully"}}
'''
        
        # Save template file
        plugin_file = self.plugin_dir / f"{plugin_name}.py"
        with open(plugin_file, 'w') as f:
            f.write(template)
            
        logger.info(f"Created plugin template: {plugin_file}")
        return str(plugin_file)


class BuiltinPlugin:
    """Handler for built-in plugins"""
    
    def __init__(self, name: str, info: Dict[str, Any]):
        self.name = name
        self.info = info
        
    async def execute(self, command: str, context: Dict[str, Any]) -> Any:
        """Execute built-in plugin command"""
        command_lower = command.lower()
        
        if self.name == 'system_info':
            return await self._handle_system_info(command_lower, context)
        elif self.name == 'agent_manager':
            return await self._handle_agent_manager(command_lower, context)
        elif self.name == 'task_tracker':
            return await self._handle_task_tracker(command_lower, context)
        elif self.name == 'voice_commands':
            return await self._handle_voice_commands(command_lower, context)
        else:
            return {"error": f"Unknown built-in plugin: {self.name}"}
            
    async def _handle_system_info(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system info commands"""
        import psutil
        import platform
        
        if 'status' in command or 'info' in command:
            return {
                'system': platform.system(),
                'platform': platform.platform(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'uptime': psutil.boot_time()
            }
        elif 'health' in command:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            health_status = 'healthy'
            if cpu_percent > 90 or memory_percent > 90:
                health_status = 'critical'
            elif cpu_percent > 70 or memory_percent > 70:
                health_status = 'warning'
                
            return {
                'status': health_status,
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent
            }
            
        return {"error": "Unknown system info command"}
        
    async def _handle_agent_manager(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent manager commands"""
        if 'list' in command:
            # This would integrate with the actual agent coordinator
            return {
                'agents': [
                    {'name': 'agent1', 'status': 'active'},
                    {'name': 'agent2', 'status': 'inactive'}
                ]
            }
        elif 'status' in command:
            return {'total_agents': 69, 'active_agents': 42, 'inactive_agents': 27}
            
        return {"error": "Unknown agent manager command"}
        
    async def _handle_task_tracker(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task tracker commands"""
        if 'list' in command:
            return {
                'tasks': [
                    {'id': 1, 'name': 'Task 1', 'status': 'running'},
                    {'id': 2, 'name': 'Task 2', 'status': 'completed'}
                ]
            }
        elif 'status' in command:
            return {'total_tasks': 5, 'running_tasks': 2, 'completed_tasks': 3}
            
        return {"error": "Unknown task tracker command"}
        
    async def _handle_voice_commands(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle voice command settings"""
        if 'settings' in command:
            return {
                'voice_enabled': True,
                'language': 'en-US',
                'voice_rate': 200,
                'voice_volume': 0.9
            }
        elif 'set voice' in command:
            return {'message': 'Voice settings updated'}
        elif 'calibration' in command:
            return {'message': 'Voice calibration completed'}
            
        return {"error": "Unknown voice command"}