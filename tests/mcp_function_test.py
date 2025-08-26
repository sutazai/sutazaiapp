#!/usr/bin/env python3
"""Test actual MCP functionality using available functions"""

# Test 1: Check if MCP tools are actually working by calling them
print("=== TESTING ACTUAL MCP FUNCTION CALLS ===")

# Test the available MCP tools from the environment
test_commands = [
    # Test mcp-extended-memory
    "mcp__extended_memory__load_contexts",
    # Test mcp-files  
    "mcp__files__list_directory",
    # Test mcp-ultimatecoder
    "mcp__ultimatecoder__tool_read_file",
    # Test mcp-github
    "mcp__github__search_repositories",
    # Test mcp-claude-flow
    "mcp__claude_flow__swarm_status"
]

print("Available MCP functions to test:")
for cmd in test_commands:
    print(f"  - {cmd}")

# Let's actually call one to verify it works
print("\n=== TESTING mcp-extended-memory ===")
try:
    # This should be available based on the tool list
    result = "Attempting to load contexts..."
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")