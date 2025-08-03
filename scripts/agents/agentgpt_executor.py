#!/usr/bin/env python3
"""
Purpose: Execute local tasks using Ollama TinyLlama model
Usage: python agentgpt_executor.py "task description" [--file FILE] [--action ACTION]
Requirements: Ollama installed, TinyLlama model downloaded
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

class LocalTaskExecutor:
    """Simple local task executor using Ollama"""
    
    def __init__(self):
        self.model = "tinyllama"
        self.ollama_cmd = "ollama"
        
    def query_llm(self, prompt: str) -> str:
        """Query TinyLlama via Ollama CLI"""
        try:
            cmd = [self.ollama_cmd, "run", self.model, prompt]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "Error: LLM query timed out"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def execute_command(self, command: str) -> Dict[str, str]:
        """Execute a shell command safely"""
        # Basic safety check - only allow simple commands
        allowed_commands = ["ls", "pwd", "date", "echo", "cat", "wc"]
        cmd_parts = command.split()
        
        if not cmd_parts or cmd_parts[0] not in allowed_commands:
            return {
                "status": "error",
                "output": f"Command '{cmd_parts[0] if cmd_parts else ''}' not allowed"
            }
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                "status": "success",
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {
                "status": "error",
                "output": str(e)
            }
    
    def process_file(self, file_path: str, action: str) -> Dict[str, str]:
        """Process a file with basic operations"""
        path = Path(file_path)
        
        if action == "read":
            if not path.exists():
                return {"status": "error", "output": "File not found"}
            try:
                content = path.read_text()
                return {"status": "success", "output": f"Read {len(content)} characters"}
            except Exception as e:
                return {"status": "error", "output": str(e)}
        
        elif action == "exists":
            exists = path.exists()
            return {"status": "success", "output": str(exists)}
        
        else:
            return {"status": "error", "output": f"Unknown action: {action}"}
    
    def execute_task(self, task_description: str) -> Dict[str, any]:
        """Execute a task based on description"""
        # Map common tasks to commands
        task_lower = task_description.lower()
        
        if "list files" in task_lower or "show files" in task_lower:
            return self.execute_command("ls -la")
        elif "current directory" in task_lower or "where am i" in task_lower:
            return self.execute_command("pwd")
        elif "current time" in task_lower or "what time" in task_lower:
            return self.execute_command("date")
        else:
            # Ask LLM for simple command suggestion
            prompt = f"Convert this task to a simple shell command (ls, pwd, date, echo only): {task_description}"
            suggestion = self.query_llm(prompt)
            return {"status": "info", "output": f"Task unclear. LLM suggests: {suggestion}"}

def main():
    parser = argparse.ArgumentParser(description="Local Task Executor")
    parser.add_argument("task", nargs="?", help="Task description")
    parser.add_argument("--file", help="File to process")
    parser.add_argument("--action", default="read", help="File action (read, exists)")
    
    args = parser.parse_args()
    executor = LocalTaskExecutor()
    
    if args.file:
        result = executor.process_file(args.file, args.action)
    elif args.task:
        result = executor.execute_task(args.task)
    else:
        result = {"status": "error", "output": "No task or file specified"}
    
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result.get("status") == "success" else 1)

if __name__ == "__main__":
    main()