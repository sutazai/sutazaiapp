#!/usr/bin/env python3
"""Extended Memory MCP Server"""

import json
import sys
import asyncio
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.types import Tool, Resource

class ExtendedMemoryServer:
    def __init__(self):
        self.server = Server("extended-memory")
        self.memory_store: Dict[str, Any] = {}
        
    async def handle_store(self, key: str, value: Any) -> Dict[str, Any]:
        """Store a value in memory"""
        self.memory_store[key] = value
        return {"status": "stored", "key": key}
    
    async def handle_retrieve(self, key: str) -> Dict[str, Any]:
        """Retrieve a value from memory"""
        if key in self.memory_store:
            return {"status": "found", "key": key, "value": self.memory_store[key]}
        return {"status": "not_found", "key": key}
    
    async def handle_list(self) -> Dict[str, Any]:
        """List all keys in memory"""
        return {"keys": list(self.memory_store.keys()), "count": len(self.memory_store)}
    
    async def handle_clear(self) -> Dict[str, Any]:
        """Clear all memory"""
        self.memory_store.clear()
        return {"status": "cleared"}
    
    async def run(self):
        """Run the MCP server"""
        # Register tools
        self.server.add_tool(Tool(
            name="store",
            description="Store a value in extended memory",
            parameters={"key": "string", "value": "any"}
        ))
        self.server.add_tool(Tool(
            name="retrieve",
            description="Retrieve a value from extended memory",
            parameters={"key": "string"}
        ))
        self.server.add_tool(Tool(
            name="list",
            description="List all keys in extended memory",
            parameters={}
        ))
        self.server.add_tool(Tool(
            name="clear",
            description="Clear all extended memory",
            parameters={}
        ))
        
        # Start server
        await self.server.run()

if __name__ == "__main__":
    server = ExtendedMemoryServer()
    asyncio.run(server.run())
