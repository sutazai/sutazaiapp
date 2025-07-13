"""
Enhanced API Routes for SutazAI
Comprehensive endpoint implementations
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from typing import Dict, List, Any, Optional
import logging
import json
from pydantic import BaseModel

try:
    from backend.services.vector_database import VectorDatabaseService
    vector_db = VectorDatabaseService()
except ImportError:
    vector_db = None

try:
    from backend.services.agent_manager import AgentManager
    agent_manager = AgentManager()
except ImportError:
    agent_manager = None

try:
    from ai_agents.ollama_agent import OllamaAgent
except ImportError:
    OllamaAgent = None

logger = logging.getLogger(__name__)

# Pydantic models
class TaskRequest(BaseModel):
    task_type: str
    parameters: Dict[str, Any]
    agent_name: Optional[str] = None

class DocumentAnalysisRequest(BaseModel):
    content: str
    analysis_type: str = "summary"
    
class CodeGenerationRequest(BaseModel):
    description: str
    language: str = "python"
    framework: Optional[str] = None
    
class VectorSearchRequest(BaseModel):
    query: str
    collection: str = "documents"
    n_results: int = 5

# Create router
router = APIRouter()

# Agent Management Endpoints
@router.get("/agents")
async def list_agents():
    """List all available agents"""
    try:
        agents_status = await agent_manager.list_agents()
        return agents_status
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_name}/execute")
async def execute_agent_task(agent_name: str, task: TaskRequest):
    """Execute a task using a specific agent"""
    try:
        result = await agent_manager.execute_task(
            agent_name, 
            task.dict()
        )
        return result
    except Exception as e:
        logger.error(f"Failed to execute task with agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_name}/status")
async def get_agent_status(agent_name: str):
    """Get status of a specific agent"""
    try:
        status = await agent_manager.get_agent_status(agent_name)
        return status
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document Processing Endpoints
@router.post("/documents/analyze")
async def analyze_document(request: DocumentAnalysisRequest):
    """Analyze a document using AI"""
    try:
        # Use Ollama for document analysis
        ollama = OllamaAgent()
        
        prompt = f"""
        Please analyze the following document and provide a {request.analysis_type}:
        
        Document Content:
        {request.content}
        
        Analysis Type: {request.analysis_type}
        """
        
        result = await ollama.generate_text(prompt)
        
        return {
            "analysis": result,
            "analysis_type": request.analysis_type,
            "success": True
        }
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        content = await file.read()
        
        # Process based on file type
        if file.filename.endswith('.txt'):
            text_content = content.decode('utf-8')
        elif file.filename.endswith('.pdf'):
            # Add PDF processing here
            text_content = "PDF processing not implemented yet"
        else:
            text_content = content.decode('utf-8', errors='ignore')
        
        # Store in vector database
        await vector_db.add_documents(
            collection_name="documents",
            documents=[text_content],
            metadatas=[{
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content)
            }]
        )
        
        return {
            "message": "Document uploaded and processed",
            "filename": file.filename,
            "size": len(content),
            "success": True
        }
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Code Generation Endpoints
@router.post("/code/generate")
async def generate_code(request: CodeGenerationRequest):
    """Generate code using AI"""
    try:
        ollama = OllamaAgent()
        
        prompt = f"""
        Generate {request.language} code for the following description:
        
        Description: {request.description}
        Language: {request.language}
        Framework: {request.framework or 'None specified'}
        
        Please provide clean, well-commented code that follows best practices.
        """
        
        result = await ollama.generate_text(prompt, model="codellama")
        
        return {
            "code": result,
            "language": request.language,
            "framework": request.framework,
            "success": True
        }
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/code/analyze")
async def analyze_code(code: str = Form(...), language: str = Form("python")):
    """Analyze code for issues and improvements"""
    try:
        # Use Semgrep if available
        try:
            semgrep_agent = agent_manager.agents.get("semgrep")
            if semgrep_agent:
                semgrep_result = await semgrep_agent.scan_code(code)
            else:
                semgrep_result = {"message": "Semgrep not available"}
        except:
            semgrep_result = {"message": "Semgrep analysis failed"}
        
        # Use Ollama for code review
        ollama = OllamaAgent()
        prompt = f"""
        Please analyze the following {language} code for:
        1. Potential bugs or errors
        2. Performance improvements
        3. Security issues
        4. Code quality suggestions
        
        Code:
        {code}
        """
        
        ai_analysis = await ollama.generate_text(prompt, model="codellama")
        
        return {
            "ai_analysis": ai_analysis,
            "semgrep_analysis": semgrep_result,
            "language": language,
            "success": True
        }
    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vector Database Endpoints
@router.post("/vector/search")
async def search_vectors(request: VectorSearchRequest):
    """Search vector database for similar content"""
    try:
        results = await vector_db.search_similar(
            collection_name=request.collection,
            query=request.query,
            n_results=request.n_results
        )
        
        return {
            "results": results,
            "query": request.query,
            "collection": request.collection,
            "success": True
        }
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vector/collections")
async def list_vector_collections():
    """List all vector database collections"""
    try:
        collections = await vector_db.list_collections()
        return {
            "collections": collections,
            "success": True
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vector/health")
async def vector_database_health():
    """Check vector database health"""
    try:
        health = vector_db.health_check()
        return health
    except Exception as e:
        logger.error(f"Vector database health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Financial Analysis Endpoints
@router.post("/finance/analyze")
async def analyze_financial_data(data: Dict[str, Any]):
    """Analyze financial data"""
    try:
        ollama = OllamaAgent()
        
        prompt = f"""
        Analyze the following financial data and provide insights:
        
        Data: {json.dumps(data, indent=2)}
        
        Please provide:
        1. Key financial metrics
        2. Trends and patterns
        3. Recommendations
        4. Risk assessment
        """
        
        result = await ollama.generate_text(prompt)
        
        return {
            "analysis": result,
            "data_summary": data,
            "success": True
        }
    except Exception as e:
        logger.error(f"Financial analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Management Endpoints
@router.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get agent status
        agents_status = await agent_manager.list_agents()
        
        # Get vector DB status
        vector_status = vector_db.health_check()
        
        # Get Ollama status
        try:
            ollama = OllamaAgent()
            ollama_status = await ollama.health_check()
        except:
            ollama_status = {"status": "unhealthy"}
        
        return {
            "agents": agents_status,
            "vector_database": vector_status,
            "ollama": ollama_status,
            "system": "operational",
            "success": True
        }
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
