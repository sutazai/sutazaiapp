#!/usr/bin/env python3
"""
Documind Wrapper - Document Processing Agent
"""

import os
import sys
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_agent_wrapper import BaseAgentWrapper, ChatRequest

class DocumindLocal(BaseAgentWrapper):
    """Documind document processing wrapper"""
    
    def __init__(self):
        super().__init__(
            agent_name="Documind",
            agent_description="Intelligent document processing and extraction",
            port=8000
        )
        self.documents = []
        self.setup_documind_routes()
    
    def setup_documind_routes(self):
        """Setup Documind routes"""
        
        @self.app.get("/capabilities")
        async def get_capabilities():
            """Return Documind agent capabilities"""
            return {
                "agent": "Documind",
                "version": "1.0.0",
                "capabilities": ["document_processing", "text_extraction", "document_summarization", "information_extraction"],
                "supported_formats": ["pdf", "docx", "txt", "md", "html"],
                "endpoints": ["/health", "/capabilities", "/chat", "/document/analyze", "/document/summarize"]
            }
        
        @self.app.post("/document/analyze")
        async def analyze_document(request: Dict[str, Any]):
            """Analyze document content"""
            try:
                content = request.get("content")
                doc_type = request.get("type", "general")
                
                analysis_prompt = f"""Analyze this {doc_type} document:
                
                {content[:2000]}...
                
                Extract:
                1. Key information
                2. Main topics
                3. Important entities
                4. Summary"""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are Documind analyzing documents."},
                        {"role": "user", "content": analysis_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                analysis = response.choices[0]["message"]["content"]
                
                doc_record = {
                    "type": doc_type,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
                self.documents.append(doc_record)
                
                return {"success": True, "analysis": analysis}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/document/extract")
        async def extract_data(request: Dict[str, Any]):
            """Extract structured data from documents"""
            try:
                content = request.get("content")
                fields = request.get("fields", [])
                
                extraction_prompt = f"""Extract these fields from the document:
                Fields: {fields}
                
                Document:
                {content[:2000]}...
                
                Return structured data."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are Documind extracting data."},
                        {"role": "user", "content": extraction_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                extracted_data = response.choices[0]["message"]["content"]
                
                return {"success": True, "extracted_data": extracted_data}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/document/summarize")
        async def summarize_document(request: Dict[str, Any]):
            """Summarize document content"""
            try:
                content = request.get("content")
                length = request.get("length", "medium")
                
                summary_prompt = f"""Summarize this document ({length} length):
                
                {content[:3000]}...
                
                Provide a clear, concise summary."""
                
                chat_request = ChatRequest(
                    messages=[
                        {"role": "system", "content": "You are Documind creating summaries."},
                        {"role": "user", "content": summary_prompt}
                    ]
                )
                
                response = await self.generate_completion(chat_request)
                summary = response.choices[0]["message"]["content"]
                
                return {"success": True, "summary": summary}
                
            except Exception as e:
                return {"success": False, "error": str(e)}

from datetime import datetime

def main():
    agent = DocumindLocal()
    agent.run()

if __name__ == "__main__":
    main()