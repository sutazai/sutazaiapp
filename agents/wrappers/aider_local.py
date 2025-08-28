#!/usr/bin/env python3
"""
Aider Wrapper - AI Pair Programming Assistant
"""

import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class AiderLocal(BaseAgentWrapper):
    """Aider AI pair programming wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="Aider",
            agent_description="AI pair programming and code assistance",
            port=8000
        )
        self.setup_aider_routes()
    
    def setup_aider_routes(self):
        """Setup Aider-specific routes"""
        
        @self.app.post("/code/review")
        async def review_code(request: Dict[str, Any]):
            """Review code and suggest improvements"""
            try:
                code = request.get("code", "")
                language = request.get("language", "python")
                
                review_prompt = f"""Review this {language} code and suggest improvements:
                
                ```{language}
                {code}
                ```
                
                Provide specific suggestions for:
                1. Code quality
                2. Performance optimizations
                3. Best practices
                4. Potential bugs"""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are Aider, an AI pair programmer."},
                        {"role": "user", "content": review_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                review = response.choices[0]["message"]["content"]
                
                return {"success": True, "review": review}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/code/refactor")
        async def refactor_code(request: Dict[str, Any]):
            """Refactor code for better quality"""
            try:
                code = request.get("code", "")
                goal = request.get("goal", "improve readability")
                
                refactor_prompt = f"""Refactor this code to {goal}:
                
                {code}
                
                Provide the refactored version with explanations."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are Aider, refactoring code."},
                        {"role": "user", "content": refactor_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                refactored = response.choices[0]["message"]["content"]
                
                return {"success": True, "refactored_code": refactored}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/code/complete")
        async def complete_code(request: Dict[str, Any]):
            """Complete partial code"""
            try:
                partial_code = request.get("code", "")
                context = request.get("context", "")
                
                completion_prompt = f"""Complete this code:
                
                Context: {context}
                
                {partial_code}
                
                Provide the completed implementation."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are Aider, completing code implementations."},
                        {"role": "user", "content": completion_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                completed = response.choices[0]["message"]["content"]
                
                return {"success": True, "completed_code": completed}
                
            except Exception as e:
                return {"success": False, "error": str(e)}

def main():
    agent = AiderLocal()
    agent.run()

if __name__ == "__main__":
    main()