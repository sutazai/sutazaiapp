#!/usr/bin/env python3
"""
SuperClaude Framework Comprehensive Test Suite
Tests all major components and functionality
"""

import os
import json
import sys
from pathlib import Path

def color_text(text, color="green"):
    """Add color to terminal output"""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def test_installation():
    """Test SuperClaude installation"""
    print(color_text("\n=== Testing SuperClaude Installation ===", "blue"))
    
    claude_dir = Path.home() / ".claude"
    tests = []
    
    # Check main directory
    tests.append(("Claude directory exists", claude_dir.exists()))
    
    # Check core files
    core_files = ["CLAUDE.md", "FLAGS.md", "PRINCIPLES.md", "RULES.md"]
    for file in core_files:
        tests.append((f"Core file: {file}", (claude_dir / file).exists()))
    
    # Check directories
    dirs = ["commands/sc", "Agents", "Modes", "logs", "backups"]
    for dir_path in dirs:
        tests.append((f"Directory: {dir_path}", (claude_dir / dir_path).exists()))
    
    # Check metadata
    metadata_file = claude_dir / ".superclaude-metadata.json"
    tests.append(("Metadata file exists", metadata_file.exists()))
    
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                version = metadata.get("framework", {}).get("version", "Unknown")
                tests.append((f"Framework version: {version}", True))
        except:
            tests.append(("Metadata readable", False))
    
    return tests

def test_commands():
    """Test command files"""
    print(color_text("\n=== Testing Commands ===", "blue"))
    
    commands_dir = Path.home() / ".claude" / "commands" / "sc"
    tests = []
    
    expected_commands = [
        "analyze", "brainstorm", "build", "cleanup", "design",
        "document", "estimate", "explain", "git", "implement",
        "improve", "index", "load", "reflect", "save",
        "select-tool", "spawn", "task", "test", "troubleshoot", "workflow"
    ]
    
    for cmd in expected_commands:
        cmd_file = commands_dir / f"{cmd}.md"
        tests.append((f"Command: /sc:{cmd}", cmd_file.exists()))
    
    # Count total commands
    if commands_dir.exists():
        total = len(list(commands_dir.glob("*.md")))
        tests.append((f"Total commands found: {total}", total == 21))
    
    return tests

def test_agents():
    """Test agent configurations"""
    print(color_text("\n=== Testing Agents ===", "blue"))
    
    agents_dir = Path.home() / ".claude" / "Agents"
    tests = []
    
    expected_agents = [
        "backend-architect", "devops-architect", "frontend-architect",
        "performance-engineer", "python-expert", "quality-engineer",
        "security-engineer", "system-architect", "technical-writer"
    ]
    
    for agent in expected_agents:
        agent_file = agents_dir / f"{agent}.md"
        tests.append((f"Agent: {agent}", agent_file.exists()))
    
    # Count total agents
    if agents_dir.exists():
        total = len(list(agents_dir.glob("*.md")))
        tests.append((f"Total agents found: {total}", total >= 14))
    
    return tests

def test_modes():
    """Test behavioral modes"""
    print(color_text("\n=== Testing Modes ===", "blue"))
    
    modes_dir = Path.home() / ".claude" / "Modes"
    tests = []
    
    expected_modes = [
        "MODE_Brainstorming", "MODE_Introspection", "MODE_Orchestration",
        "MODE_Task_Management", "MODE_Token_Efficiency"
    ]
    
    for mode in expected_modes:
        mode_file = modes_dir / f"{mode}.md"
        tests.append((f"Mode: {mode.replace('MODE_', '')}", mode_file.exists()))
    
    return tests

def test_cli():
    """Test CLI functionality"""
    print(color_text("\n=== Testing CLI ===", "blue"))
    
    tests = []
    
    # Check if SuperClaude command is available
    import subprocess
    try:
        result = subprocess.run(["SuperClaude", "--version"], 
                              capture_output=True, text=True, timeout=5)
        version = result.stdout.strip() if result.returncode == 0 else "Error"
        tests.append((f"CLI Version: {version}", result.returncode == 0))
    except Exception as e:
        tests.append(("CLI accessible", False))
    
    return tests

def print_results(test_results):
    """Print test results with formatting"""
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = color_text("[PASS]", "green") if result else color_text("[FAIL]", "red")
        print(f"  {status} {test_name}")
    
    return passed, total

def main():
    """Run all tests"""
    print(color_text("=" * 60, "blue"))
    print(color_text("  SuperClaude Framework Test Suite v4.0.8", "blue"))
    print(color_text("=" * 60, "blue"))
    
    all_tests = []
    
    # Run installation tests
    install_tests = test_installation()
    passed, total = print_results(install_tests)
    all_tests.extend(install_tests)
    print(f"  Installation: {passed}/{total} passed")
    
    # Run command tests
    cmd_tests = test_commands()
    passed, total = print_results(cmd_tests)
    all_tests.extend(cmd_tests)
    print(f"  Commands: {passed}/{total} passed")
    
    # Run agent tests
    agent_tests = test_agents()
    passed, total = print_results(agent_tests)
    all_tests.extend(agent_tests)
    print(f"  Agents: {passed}/{total} passed")
    
    # Run mode tests
    mode_tests = test_modes()
    passed, total = print_results(mode_tests)
    all_tests.extend(mode_tests)
    print(f"  Modes: {passed}/{total} passed")
    
    # Run CLI tests
    cli_tests = test_cli()
    passed, total = print_results(cli_tests)
    all_tests.extend(cli_tests)
    print(f"  CLI: {passed}/{total} passed")
    
    # Final summary
    print(color_text("\n" + "=" * 60, "blue"))
    total_passed = sum(1 for _, result in all_tests if result)
    total_tests = len(all_tests)
    
    if total_passed == total_tests:
        print(color_text(f"[SUCCESS] ALL TESTS PASSED: {total_passed}/{total_tests}", "green"))
        print(color_text("SuperClaude Framework is fully operational!", "green"))
    else:
        print(color_text(f"[WARNING] TESTS SUMMARY: {total_passed}/{total_tests} passed", "yellow"))
        failed = total_tests - total_passed
        print(color_text(f"  {failed} test(s) need attention", "red"))
    
    print(color_text("=" * 60, "blue"))
    
    return 0 if total_passed == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())