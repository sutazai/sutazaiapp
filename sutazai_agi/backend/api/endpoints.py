import logging
from fastapi import APIRouter, Depends, HTTPException, Body, Query, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import time
import uuid
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Literal
from typing import AsyncGenerator
import json
import os

# Import necessary components from the application
from sutazai_agi.agents.agent_manager import get_agent_manager, AgentManager
from sutazai_agi.backend.services.chat_service import get_chat_service, ChatService
from sutazai_agi.backend.services.document_service import get_document_service, DocumentService
from sutazai_agi.backend.services.code_service import get_code_service, CodeService
# from .dependencies import get_current_user # For authentication (to be implemented)

logger = logging.getLogger(__name__)
router = APIRouter()

# --- OpenAI Compatible Pydantic Models ---

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    # Add tool_calls, tool_call_id if needed later for tool usage

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The agent name to use.")
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None # Parameter not directly used by agent manager logic, but common
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress.")
    # Add other OpenAI parameters like temperature, top_p if they need mapping later
    # Add session_id for custom handling if needed outside messages
    session_id: Optional[str] = Field("default_session", description="Identifier for the chat session (custom field).")

# Response models for non-streaming
class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # Agent name used
    choices: List[ChatCompletionChoice]
    # usage: Optional[UsageInfo] = None # Add later if token counting implemented

# Response models for streaming
class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None

class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # Agent name used
    choices: List[ChatCompletionChunkChoice]
    # usage: Optional[UsageInfo] = None # Add later if token counting implemented

# --- Existing Models (Keep or Adapt if needed elsewhere) ---

class AgentInfo(BaseModel):
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    enabled: bool
    model: Optional[str] = None
    tools: List[str] = []
    # Add other relevant config details from agents.yaml if needed

class AvailableAgentsResponse(BaseModel):
    agents: List[AgentInfo]

# --- OpenAI Compatible Models Endpoint --- 

class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{uuid.uuid4().hex}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = False
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sutazai-agi"
    permission: List[ModelPermission] = Field(default_factory=lambda: [ModelPermission()])
    root: Optional[str] = None
    parent: Optional[str] = None

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# --- API Endpoints --- 

@router.get("/models", 
             response_model=ModelListResponse, 
             tags=["Models"], 
             summary="List available models (agents) compatible with OpenAI clients")
async def list_openai_models(agent_manager: AgentManager = Depends(get_agent_manager)):
    """Provides a list of available agents in the format expected by OpenAI's /v1/models endpoint."""
    agent_names = agent_manager.get_available_agent_names()
    model_data = [
        ModelInfo(id=name) for name in agent_names
    ]
    return ModelListResponse(data=model_data)

# Rename /chat to /chat/completions for standard OpenAI path
# Keep old /chat for backward compatibility? Or just switch? Let's switch for now.
@router.post("/chat/completions", 
             # response_model=ChatCompletionResponse, # Response determined by streaming
             tags=["Chat"], 
             summary="Send messages to an AI agent (OpenAI Compatible)")
async def handle_chat_completion(
    request: Request, 
    chat_input: ChatCompletionRequest, 
    agent_manager: AgentManager = Depends(get_agent_manager),
    chat_service: ChatService = Depends(get_chat_service)
    # current_user: dict = Depends(get_current_user) # Add auth later
):
    """Receives user messages in OpenAI format and routes to the appropriate AI agent.
    
    Handles streaming responses if requested.
    """
    logger.info(f"Received request for /chat/completions. Agent requested: {chat_input.model}, Stream: {chat_input.stream}, Session: {chat_input.session_id}")
    logger.debug(f"Chat Input Payload: {chat_input.model_dump()}") # Log the full payload for debugging
    
    # ChatService instance is now correctly injected by FastAPI
    
    # Extract query and session_id (could refine this based on ChatService needs)
    last_user_message = None
    for msg in reversed(chat_input.messages):
        if msg.role == 'user':
            last_user_message = msg.content
            break
            
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found in the request.")
    
    session_id = chat_input.session_id
    stream = chat_input.stream
    requested_agent_name = chat_input.model 
    messages = [msg.model_dump() for msg in chat_input.messages] # Pass the full message list as dicts

    # TODO: Clarify if chat_input.model should be interpreted as a direct request or if routing is always default
    # For now, ChatService handles this: if requested_agent_name is provided, it uses it, otherwise routes.

    try:
        # Call the unified ChatService method, passing the full messages list
        result_or_generator = await chat_service.process_message_or_stream(
            messages=messages, # Pass the list of message dicts
            session_id=session_id, 
            stream=stream,
            requested_agent_name=requested_agent_name # Pass the requested agent
        )

        if stream:
            # Ensure the result is an async generator
            if not isinstance(result_or_generator, AsyncGenerator):
                 logger.error("ChatService did not return an async generator for streaming request.")
                 raise HTTPException(status_code=500, detail="Internal server error: Streaming failed.")
                 
            # Wrap the generator from ChatService into a StreamingResponse
            async def stream_wrapper():
                stream_id = f"chatcmpl-{uuid.uuid4().hex}" # Generate ID here or get from service?
                try:
                    async for chunk_data in result_or_generator:
                        chunk_type = chunk_data.get("type")
                        # Reconstruct the OpenAI chunk format based on yielded data
                        if chunk_type == "content_delta":
                             chunk = ChatCompletionChunk(
                                id=stream_id,
                                model=chunk_data.get("agent_name", requested_agent_name), # Get agent name if available
                                choices=[ChatCompletionChunkChoice(delta=DeltaMessage(role="assistant", content=chunk_data.get("content")))]
                            )
                             yield f"data: {chunk.model_dump_json()}\\n\\n"
                        elif chunk_type == "finish":
                            finish_reason = chunk_data.get("reason", "stop")
                            chunk = ChatCompletionChunk(
                                id=stream_id,
                                model=chunk_data.get("agent_name", requested_agent_name),
                                choices=[ChatCompletionChunkChoice(delta=DeltaMessage(), finish_reason=finish_reason)]
                            )
                            yield f"data: {chunk.model_dump_json()}\\n\\n"
                            yield f"data: [DONE]\\n\\n"
                            break 
                        elif chunk_type == "error":
                            logger.error(f"Stream yielded error: {chunk_data.get('message')}")
                            # Optionally yield an error event?
                            # yield f"event: error\\ndata: {json.dumps(chunk_data)}\\n\\n"
                            break # Stop streaming on error
                        # Handle other potential chunk types (e.g., tool calls) if added later
                except Exception as e:
                     logger.error(f"Error during stream generation in endpoint: {e}", exc_info=True)
                     # Don't yield from here if error occurs during generation itself
                finally:
                    # Ensure DONE is sent if loop finishes without explicit finish chunk? Might cause issues.
                    # yield f"data: [DONE]\\n\\n" # Be careful with this
                    pass # Let the finish chunk handle DONE message
                    
            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

        else:
            # --- Non-Streaming Logic ---
            # Ensure the result is a dict
            if not isinstance(result_or_generator, dict):
                 logger.error("ChatService did not return a dict for non-streaming request.")
                 raise HTTPException(status_code=500, detail="Internal server error: Invalid response format.")

            if result_or_generator.get("status") == "success":
                response = ChatCompletionResponse(
                    model=result_or_generator.get("agent_name", requested_agent_name),
                    choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content=result_or_generator.get("output")))]
                    # Add usage info later if available
                )
                return response
            else:
                # Agent execution failed, return error details
                raise HTTPException(
                    status_code=500, 
                    detail=result_or_generator.get("message", "Agent execution failed.")
                )

    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTP exceptions directly
    except Exception as e:
        logger.error(f"Error processing chat completion request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/agents", 
            response_model=AvailableAgentsResponse, 
            tags=["Agents"], 
            summary="List available and enabled agents")
async def list_available_agents(agent_manager: AgentManager = Depends(get_agent_manager)):
    """Returns a list of agents currently enabled in the system configuration."""
    try:
        enabled_agents_config = agent_manager.list_enabled_agents()
        # Map the full config to the AgentInfo response model
        agent_list = [
            AgentInfo(
                name=cfg.get("name", "Unnamed Agent"),
                description=cfg.get("description"),
                type=cfg.get("type"),
                enabled=cfg.get("enabled", False),
                model=cfg.get("model"), # Or resolve default model
                tools=cfg.get("tools", [])
            ) 
            for cfg in enabled_agents_config
        ]
        return AvailableAgentsResponse(agents=agent_list)
    except Exception as e:
        logger.exception(f"Unexpected error listing available agents: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Add Endpoints for other services (Document Analysis, Code Gen, etc.) --- 
# Example placeholder:
# @router.post("/analyze_document", tags=["Documents"], summary="Analyze an uploaded document")
# async def analyze_document(...):
#     # ... implementation using a dedicated service/agent ...
#     pass

# @router.post("/generate_code", tags=["Code"], summary="Generate code based on a prompt")
# async def generate_code(...):
#     # ... implementation using GPT-Engineer agent via AgentManager ...
#     pass 

# --- Agent Management Endpoints (Example - Can be expanded) ---

@router.get("/agents", 
             response_model=AvailableAgentsResponse, 
             tags=["Agents"],
             summary="List all configured and enabled agents")
async def list_enabled_agents(agent_manager: AgentManager = Depends(get_agent_manager)):
    """Returns a list of currently enabled agents and their configurations."""
    agents = agent_manager.list_enabled_agents()
    # Convert agent configs to AgentInfo Pydantic models
    agent_info_list = [AgentInfo(**agent) for agent in agents]
    return AvailableAgentsResponse(agents=agent_info_list)

# --- Document Processing Endpoints --- 

class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    doc_id: Optional[str] = None
    num_chunks: Optional[int] = None

class DocumentAnalysisResponse(BaseModel):
    status: str
    message: Optional[str] = None
    analysis_type: Optional[str] = None
    result: Optional[str] = None # The summary or other analysis result

class DocumentQueryResponse(BaseModel):
    status: str
    message: Optional[str] = None
    answer: Optional[str] = None
    sources: Optional[List[str]] = None

@router.post("/documents/upload", 
              response_model=DocumentUploadResponse, 
              tags=["Documents"], 
              summary="Upload and index a document")
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, TXT)"), 
    doc_service: DocumentService = Depends(get_document_service)
):
    """Uploads a document, extracts text, chunks, embeds, and indexes it."""
    logger.info(f"Received file upload request: {file.filename}, content type: {file.content_type}")
    if not file.filename:
         raise HTTPException(status_code=400, detail="Filename cannot be empty.")
         
    # Check allowed extensions (optional but recommended)
    allowed_extensions = get_setting("document_processing.allowed_upload_extensions", [".pdf", ".docx", ".txt", ".md"])
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
         raise HTTPException(status_code=400, detail=f"File type '{file_ext}' not allowed. Allowed types: {allowed_extensions}")

    try:
        content = await file.read()
        result = await doc_service.upload_and_index_document(file_content=content, filename=file.filename)
        if result.get("status") == "error":
             raise HTTPException(status_code=500, detail=result.get("message", "Failed to process document."))
        return DocumentUploadResponse(**result)
    except HTTPException as http_exc:
         raise http_exc # Re-raise validation errors
    except Exception as e:
        logger.error(f"Error during document upload processing for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during upload: {e}")
    finally:
        await file.close()

@router.post("/documents/{doc_id}/analyze", 
               response_model=DocumentAnalysisResponse, 
               tags=["Documents"], 
               summary="Analyze an indexed document")
async def analyze_document_endpoint(
    doc_id: str,
    analysis_type: str = Query("summary", description="Type of analysis (e.g., 'summary')"),
    doc_service: DocumentService = Depends(get_document_service)
):
    """Performs analysis (e.g., summarization) on a previously indexed document."""
    logger.info(f"Received analysis request for doc_id: {doc_id}, type: {analysis_type}")
    try:
        result = await doc_service.analyze_document(doc_id=doc_id, analysis_type=analysis_type)
        if result.get("status") == "error":
            # Distinguish between not found and other errors
            if "No content found" in result.get("message", ""):
                 raise HTTPException(status_code=404, detail=result.get("message"))
            else:
                 raise HTTPException(status_code=500, detail=result.get("message", "Failed to analyze document."))
        return DocumentAnalysisResponse(**result)
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Error during document analysis for doc_id '{doc_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {e}")

class DocumentQueryRequest(BaseModel):
    query: str

@router.post("/documents/{doc_id}/query", 
               response_model=DocumentQueryResponse, 
               tags=["Documents"], 
               summary="Query an indexed document")
async def query_document_endpoint(
    doc_id: str,
    request_body: DocumentQueryRequest,
    doc_service: DocumentService = Depends(get_document_service)
):
    """Answers a query based on the content of a specific indexed document using RAG."""
    logger.info(f"Received query request for doc_id: {doc_id}")
    try:
        result = await doc_service.query_document(doc_id=doc_id, query=request_body.query)
        if result.get("status") == "error":
             raise HTTPException(status_code=500, detail=result.get("message", "Failed to query document."))
        return DocumentQueryResponse(**result)
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Error during document query for doc_id '{doc_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during query: {e}")

# Add other endpoints as needed (e.g., list documents, delete document)

# --- Code Tools Endpoints --- 

class CodeGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Detailed natural language prompt describing the desired software.")
    project_name: str = Field(..., description="Unique name for the project (used as directory name).")

class CodeGenerationResponse(BaseModel):
    status: str
    message: str
    output_path: Optional[str] = None
    log: Optional[str] = None

class CodeEditRequest(BaseModel):
    files: List[str] = Field(..., description="List of file paths (relative to workspace/repo) to edit.")
    instruction: str = Field(..., description="Natural language instruction for the edit.")
    repo_path: Optional[str] = Field(None, description="Optional path to the git repository root relative to workspace.")

class CodeEditResponse(BaseModel):
    status: str
    message: str
    output: Optional[str] = None # Aider output/diff
    log: Optional[str] = None

@router.post("/code/generate", 
              response_model=CodeGenerationResponse, 
              tags=["Code"], 
              summary="Generate a codebase using GPT-Engineer")
async def generate_codebase_endpoint(
    request_body: CodeGenerationRequest,
    code_service: CodeService = Depends(get_code_service)
):
    """Generates a complete codebase based on a prompt using GPT-Engineer via the CodeService."""
    logger.info(f"Received codebase generation request for project: {request_body.project_name}")
    try:
        result = await code_service.generate_codebase(prompt=request_body.prompt, project_name=request_body.project_name)
        if result.get("status") == "error":
             # Use 500 for internal errors, maybe 400 if input was bad (though service might handle that)
             raise HTTPException(status_code=500, detail=result.get("message", "Code generation failed."))
        # Ensure the response model matches the keys returned by the service
        return CodeGenerationResponse(**result)
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Error during codebase generation endpoint processing for '{request_body.project_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during code generation: {e}")

@router.post("/code/edit", 
              response_model=CodeEditResponse, 
              tags=["Code"], 
              summary="Edit code files using Aider")
async def edit_code_endpoint(
    request_body: CodeEditRequest,
    code_service: CodeService = Depends(get_code_service)
):
    """Edits specified code files based on instructions using Aider via the CodeService."""
    logger.info(f"Received code edit request for files: {request_body.files}")
    try:
        result = await code_service.edit_code_files(
            files=request_body.files, 
            instruction=request_body.instruction, 
            repo_path=request_body.repo_path
        )
        if result.get("status") == "error":
             raise HTTPException(status_code=500, detail=result.get("message", "Code editing failed."))
        # Ensure the response model matches the keys returned by the service
        return CodeEditResponse(**result)
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Error during code edit endpoint processing for files '{request_body.files}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during code editing: {e}") 