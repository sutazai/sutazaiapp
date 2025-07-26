#!/usr/bin/env python3
"""
SutazAI Tool Registry
Manages available tools and capabilities for AI agents
"""

from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import asyncio


class ToolRegistry:
    """Registry for managing AI agent tools and capabilities"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, List[str]] = {
            "system": [],
            "code": [],
            "data": [],
            "communication": [],
            "file": []
        }
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the tool registry with default tools"""
        self._register_default_tools()
        self.initialized = True
    
    def _register_default_tools(self) -> None:
        """Register default system tools"""
        # System tools
        self.register_tool(
            name="system_status",
            description="Get system status and health information",
            category="system",
            function=self._system_status_tool
        )
        
        self.register_tool(
            name="list_files",
            description="List files in a directory",
            category="file",
            function=self._list_files_tool
        )
        
        self.register_tool(
            name="read_file",
            description="Read contents of a file",
            category="file",
            function=self._read_file_tool
        )
        
        # Code tools
        self.register_tool(
            name="execute_code",
            description="Execute code in a sandboxed environment",
            category="code",
            function=self._execute_code_tool
        )
    
    def register_tool(self, name: str, description: str, category: str, function: Callable, **metadata) -> None:
        """Register a new tool"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "category": category,
            "function": function,
            "metadata": metadata,
            "registered_at": datetime.now().isoformat()
        }
        
        if category not in self.categories:
            self.categories[category] = []
        
        if name not in self.categories[category]:
            self.categories[category].append(name)
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all tools in a category"""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def list_all_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools"""
        return list(self.tools.values())
    
    def execute_tool(self, name: str, *args, **kwargs) -> Any:
        """Execute a tool by name"""
        tool = self.get_tool(name)
        if tool:
            try:
                return tool["function"](*args, **kwargs)
            except Exception as e:
                return {"error": f"Tool execution failed: {str(e)}"}
        else:
            return {"error": f"Tool '{name}' not found"}
    
    def _system_status_tool(self) -> Dict[str, Any]:
        """Default system status tool"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "tools_registered": len(self.tools)
        }
    
    def _list_files_tool(self, path: str = ".") -> Dict[str, Any]:
        """Default file listing tool"""
        import os
        try:
            files = os.listdir(path)
            return {"files": files, "path": path}
        except Exception as e:
            return {"error": str(e)}
    
    def _read_file_tool(self, filepath: str) -> Dict[str, Any]:
        """Default file reading tool"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            return {"content": content, "filepath": filepath}
        except Exception as e:
            return {"error": str(e)}
    
    def _execute_code_tool(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Default code execution tool (placeholder)"""
        return {
            "result": "Code execution would happen here",
            "code": code,
            "language": language,
            "note": "This is a placeholder implementation"
        }
    
    def cleanup(self) -> None:
        """Cleanup registry on shutdown"""
        self.tools.clear()
        self.categories.clear()
        self.initialized = False