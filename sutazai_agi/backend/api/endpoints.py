import logging
from fastapi import APIRouter, Depends, HTTPException, Body, Query, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import time
import uuid
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Literal
from typing import AsyncGenerator
import json

# Import necessary components from the application
from sutazai_agi.agents.agent_manager import get_agent_manager, AgentManager
from sutazai_agi.backend.services.chat_service import get_chat_service, ChatService
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
    agent_manager: AgentManager = Depends(get_agent_manager)
    # current_user: dict = Depends(get_current_user) # Add auth later
):
    """Receives user messages in OpenAI format and routes to the appropriate AI agent.
    
    Handles streaming responses if requested.
    """
    # Use ChatService to handle routing and execution
    chat_service: ChatService = Depends(get_chat_service)() # Get ChatService instance

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
    requested_agent_name = chat_input.model # API 'model' field specifies the desired agent OR triggers routing if not specified/default?
    # TODO: Clarify if chat_input.model should be interpreted as a direct request or if routing is always default
    # For now, assume if chat_input.model is sent, it bypasses router.
    # If user selects "Auto" or similar in UI, UI should send null/empty model?
    # Let ChatService handle the logic of using requested_agent_name vs routing.

    try:
        # Call the unified ChatService method
        result_or_generator = await chat_service.process_message_or_stream(
            query=last_user_message, 
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