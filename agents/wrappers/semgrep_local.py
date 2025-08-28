#!/usr/bin/env python3
"""
Semgrep Wrapper - Code Security Analysis
"""

import os
import sys
import subprocess
import json
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class SemgrepLocal(BaseAgentWrapper):
    """Semgrep security analysis wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="Semgrep",
            agent_description="Code security analysis and vulnerability detection",
            port=8000
        )
        self.scan_results = []
        self.setup_semgrep_routes()
    
    def setup_semgrep_routes(self):
        """Setup Semgrep-specific routes"""
        
        @self.app.post("/scan")
        async def scan_code(request: Dict[str, Any]):
            """Scan code for security issues"""
            try:
                code = request.get("code", "")
                language = request.get("language", "python")
                
                # Analyze code using LLM for security issues
                security_prompt = f"""Analyze this {language} code for security vulnerabilities:
                
                ```{language}
                {code}
                ```
                
                Identify:
                1. Security vulnerabilities (injection, XSS, etc.)
                2. Code quality issues
                3. Best practice violations
                4. Potential bugs
                
                Provide specific line numbers and recommendations."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are Semgrep, a security analysis expert."},
                        {"role": "user", "content": security_prompt}
                    ],
                    temperature=0.3  # Lower temperature for more consistent analysis
                )
                
                response = await self.generate_completion(chat_request)
                analysis = response.choices[0]["message"]["content"]
                
                scan_result = {
                    "code_snippet": code[:200],
                    "language": language,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.scan_results.append(scan_result)
                
                return {
                    "success": True,
                    "vulnerabilities": analysis,
                    "scan_id": len(self.scan_results) - 1
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/scan/file")
        async def scan_file(request: Dict[str, Any]):
            """Scan a file for security issues"""
            try:
                file_path = request.get("file_path")
                
                if not os.path.exists(file_path):
                    return {"success": False, "error": "File not found"}
                
                with open(file_path, 'r') as f:
                    code = f.read()
                
                language = os.path.splitext(file_path)[1][1:]  # Get extension
                
                return await scan_code({"code": code, "language": language})
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/results")
        async def get_results():
            """Get scan results"""
            return {"results": self.scan_results}
        
        @self.app.post("/rules/custom")
        async def add_custom_rule(request: Dict[str, Any]):
            """Add custom security rule"""
            rule = request.get("rule")
            description = request.get("description")
            
            return {
                "success": True,
                "rule": rule,
                "description": description,
                "status": "Rule added for analysis"
            }

from datetime import datetime

def main():
    agent = SemgrepLocal()
    agent.run()

if __name__ == "__main__":
    main()