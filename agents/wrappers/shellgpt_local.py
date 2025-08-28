#!/usr/bin/env python3
"""
ShellGPT Wrapper - CLI Assistant
"""

import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class ShellGPTLocal(BaseAgentWrapper):
    """ShellGPT CLI assistant wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="ShellGPT",
            agent_description="Command-line interface assistant",
            port=8000
        )
        self.setup_shellgpt_routes()
    
    def setup_shellgpt_routes(self):
        """Setup ShellGPT routes"""
        
        @self.app.post("/command/generate")
        async def generate_command(request: Dict[str, Any]):
            """Generate shell commands"""
            try:
                task = request.get("task")
                os_type = request.get("os", "linux")
                
                command_prompt = f"""Generate {os_type} shell command for: {task}
                
                Provide the command with explanation."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are ShellGPT, a CLI expert."},
                        {"role": "user", "content": command_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                command = response.choices[0]["message"]["content"]
                
                return {"success": True, "command": command}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/script/generate")
        async def generate_script(request: Dict[str, Any]):
            """Generate shell scripts"""
            try:
                purpose = request.get("purpose")
                shell = request.get("shell", "bash")
                
                script_prompt = f"""Generate a {shell} script for: {purpose}
                
                Include error handling and comments."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are ShellGPT generating scripts."},
                        {"role": "user", "content": script_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                script = response.choices[0]["message"]["content"]
                
                return {"success": True, "script": script}
                
            except Exception as e:
                return {"success": False, "error": str(e)}

def main():
    agent = ShellGPTLocal()
    agent.run()

if __name__ == "__main__":
    main()