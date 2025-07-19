#\!/usr/bin/env python3
"""
LlamaIndex Service for SutazAI
Provides document indexing and retrieval
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI LlamaIndex Service", version="1.0.0")

class DocumentRequest(BaseModel):
    content: str
    doc_type: str = "text"
    index_name: str = "default"

class QueryRequest(BaseModel):
    query: str
    index_name: str = "default"
    top_k: int = 5

class LlamaIndexService:
    """LlamaIndex document processing service"""
    
    def __init__(self):
        self.ollama_url = "http://ollama:11434"
        self.service_name = "LlamaIndex"
        self.indexes = {}
        
    async def index_document(self, content: str, doc_type: str = "text", index_name: str = "default") -> Dict[str, Any]:
        """Index a document for retrieval"""
        try:
            logger.info(f"Indexing document in {index_name}")
            
            # Simple document storage (in production, would use real LlamaIndex)
            if index_name not in self.indexes:
                self.indexes[index_name] = []
            
            doc_id = f"doc_{len(self.indexes[index_name])}"
            document = {
                "id": doc_id,
                "content": content,
                "type": doc_type,
                "indexed_at": datetime.now().isoformat()
            }
            
            self.indexes[index_name].append(document)
            
            return {
                "success": True,
                "document_id": doc_id,
                "index_name": index_name,
                "message": f"Document indexed successfully"
            }
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {
                "success": False,
                "error": f"Indexing failed: {str(e)}"
            }
    
    async def query_index(self, query: str, index_name: str = "default", top_k: int = 5) -> Dict[str, Any]:
        """Query an index for relevant documents"""
        try:
            logger.info(f"Querying index {index_name}: {query}")
            
            if index_name not in self.indexes:
                return {
                    "success": False,
                    "error": f"Index {index_name} not found"
                }
            
            # Simple text matching (in production, would use vector similarity)
            documents = self.indexes[index_name]
            relevant_docs = []
            
            for doc in documents:
                # Simple keyword matching
                if any(word.lower() in doc["content"].lower() for word in query.split()):
                    relevant_docs.append(doc)
            
            # Limit results
            relevant_docs = relevant_docs[:top_k]
            
            # Generate response using LLM
            if relevant_docs:
                context = "\n".join([doc["content"] for doc in relevant_docs])
                response = await self._generate_response(query, context)
            else:
                response = f"No relevant documents found for query: {query}"
            
            return {
                "success": True,
                "query": query,
                "response": response,
                "documents_found": len(relevant_docs),
                "relevant_documents": relevant_docs
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "success": False,
                "error": f"Query failed: {str(e)}"
            }
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Generate response using local LLM"""
        try:
            prompt = f"""Based on the following context, answer the query:

Context:
{context}

Query: {query}

Provide a clear, accurate answer based on the context provided."""

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Unable to generate response")
            else:
                return "LLM service unavailable"
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error generating response: {str(e)}"

# Initialize service
llamaindex_service = LlamaIndexService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "LlamaIndex",
        "indexes": list(llamaindex_service.indexes.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/index")
async def index_document(request: DocumentRequest):
    """Index a document"""
    try:
        result = await llamaindex_service.index_document(
            request.content,
            request.doc_type,
            request.index_name
        )
        
        return {
            "success": result.get("success", True),
            "result": result,
            "service": "LlamaIndex",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_index(request: QueryRequest):
    """Query an index"""
    try:
        result = await llamaindex_service.query_index(
            request.query,
            request.index_name,
            request.top_k
        )
        
        return {
            "success": result.get("success", True),
            "result": result,
            "service": "LlamaIndex",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexes")
async def list_indexes():
    """List available indexes"""
    return {
        "indexes": list(llamaindex_service.indexes.keys()),
        "service": "LlamaIndex"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "LlamaIndex Document Processing",
        "status": "online",
        "version": "1.0.0",
        "description": "Document indexing and retrieval for SutazAI"
    }
EOF < /dev/null
