#!/usr/bin/env python3
"""
GPT-Engineer Wrapper - Autonomous Code Generation
"""

import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class GPTEngineerLocal(BaseAgentWrapper):
    """GPT-Engineer code generation wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="GPT-Engineer",
            agent_description="Autonomous code generation and project creation",
            port=8000
        )
        self.setup_gpt_engineer_routes()
    
    def setup_gpt_engineer_routes(self):
        """Setup GPT-Engineer routes"""
        
        @self.app.post("/project/generate")
        async def generate_project(request: Dict[str, Any]):
            """Generate a complete project"""
            try:
                project_description = request.get("description")
                language = request.get("language", "python")
                framework = request.get("framework", "")
                
                generation_prompt = f"""Generate a complete {language} project:
                Description: {project_description}
                Framework: {framework if framework else 'vanilla'}
                
                Provide:
                1. Project structure
                2. Main implementation files
                3. Configuration files
                4. Basic tests
                5. README documentation"""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are GPT-Engineer, generating complete projects."},
                        {"role": "user", "content": generation_prompt}
                    ],
                    max_tokens=4000
                )
                
                response = await self.generate_completion(chat_request)
                project_code = response.choices[0]["message"]["content"]
                
                return {"success": True, "project": project_code}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/code/implement")
        async def implement_feature(request: Dict[str, Any]):
            """Implement a specific feature"""
            try:
                feature = request.get("feature")
                context = request.get("context", "")
                
                implementation_prompt = f"""Implement this feature: {feature}
                
                Context: {context}
                
                Provide complete, working implementation."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are GPT-Engineer implementing features."},
                        {"role": "user", "content": implementation_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                implementation = response.choices[0]["message"]["content"]
                
                return {"success": True, "implementation": implementation}
                
            except Exception as e:
                return {"success": False, "error": str(e)}

def main():
    agent = GPTEngineerLocal()
    agent.run()

if __name__ == "__main__":
    main()