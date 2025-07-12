from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from uuid import uuid4
import json

from memory import vector_memory
from agents import agent_orchestrator
from models import model_manager
from api.auth import get_current_user, require_admin
from api.database import db_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Store active chat sessions
active_sessions = {}

@router.get("/sessions")
async def list_chat_sessions(
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """List chat sessions for the current user."""
    try:
        user_filter = user_id if user_id else current_user.get("username")
        
        sessions = await db_manager.get_chat_sessions(
            user_id=user_filter,
            limit=limit,
            offset=offset
        )
        
        return {
            "sessions": sessions,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions")
async def create_chat_session(
    session_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create a new chat session."""
    try:
        session_id = str(uuid4())
        session_name = session_data.get("name", f"Chat Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
        model_name = session_data.get("model", "llama2")
        agent_type = session_data.get("agent_type", "chat")
        
        session = {
            "id": session_id,
            "name": session_name,
            "user_id": current_user.get("username"),
            "model_used": model_name,
            "agent_type": agent_type,
            "created_at": datetime.utcnow(),
            "message_count": 0,
            "status": "active"
        }
        
        await db_manager.create_chat_session(session)
        
        await db_manager.log_system_event(
            "info", "chat", "Chat session created",
            {"user": current_user.get("username"), "session_id": session_id}
        )
        
        return {
            "session_id": session_id,
            "session": session,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get details about a specific chat session."""
    try:
        session = await db_manager.get_chat_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found")
        
        # Check if user owns this session or is admin
        if session.get("user_id") != current_user.get("username") and "admin" not in current_user.get("roles", []):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "session": session,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/messages")
async def get_chat_messages(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get messages from a chat session."""
    try:
        # Verify session access
        session = await db_manager.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found")
        
        if session.get("user_id") != current_user.get("username") and "admin" not in current_user.get("roles", []):
            raise HTTPException(status_code=403, detail="Access denied")
        
        messages = await db_manager.get_chat_messages(
            session_id=session_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "session_id": session_id,
            "messages": messages,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat messages for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/messages")
async def send_chat_message(
    session_id: str,
    message_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Send a message in a chat session."""
    try:
        # Verify session access
        session = await db_manager.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found")
        
        if session.get("user_id") != current_user.get("username"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        user_message = message_data.get("message", "")
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="Message content is required")
        
        # Store user message
        user_msg_id = str(uuid4())
        user_msg = {
            "id": user_msg_id,
            "session_id": session_id,
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow(),
            "tokens_used": len(user_message.split()),  # Simple token estimate
            "processing_time": 0
        }
        await db_manager.store_chat_message(user_msg)
        
        # Process message with AI
        start_time = datetime.utcnow()
        
        # Get context from vector memory if available
        context = await get_chat_context(user_message, session_id)
        
        # Generate AI response
        ai_response = await generate_ai_response(
            message=user_message,
            context=context,
            session=session
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Store AI response
        ai_msg_id = str(uuid4())
        ai_msg = {
            "id": ai_msg_id,
            "session_id": session_id,
            "role": "assistant",
            "content": ai_response.get("content", ""),
            "timestamp": datetime.utcnow(),
            "tokens_used": ai_response.get("tokens_used", 0),
            "processing_time": processing_time,
            "model_used": ai_response.get("model", session.get("model_used"))
        }
        await db_manager.store_chat_message(ai_msg)
        
        # Update session message count
        await db_manager.update_chat_session(session_id, {
            "message_count": session.get("message_count", 0) + 2,
            "last_activity": datetime.utcnow()
        })
        
        await db_manager.log_system_event(
            "info", "chat", "Message exchanged",
            {
                "user": current_user.get("username"),
                "session_id": session_id,
                "processing_time": processing_time
            }
        )
        
        return {
            "user_message": user_msg,
            "ai_response": ai_msg,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete a chat session and all its messages."""
    try:
        session = await db_manager.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found")
        
        if session.get("user_id") != current_user.get("username") and "admin" not in current_user.get("roles", []):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete all messages first
        await db_manager.delete_chat_messages(session_id)
        
        # Delete session
        await db_manager.delete_chat_session(session_id)
        
        await db_manager.log_system_event(
            "info", "chat", "Chat session deleted",
            {"user": current_user.get("username"), "session_id": session_id}
        )
        
        return {
            "session_id": session_id,
            "status": "deleted",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_conversations(
    search_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Search through chat conversations."""
    try:
        query = search_data.get("query", "")
        session_id = search_data.get("session_id")
        limit = search_data.get("limit", 20)
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Search query is required")
        
        # Search through messages
        results = await db_manager.search_chat_messages(
            query=query,
            user_id=current_user.get("username"),
            session_id=session_id,
            limit=limit
        )
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time chat
@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    try:
        await websocket.accept()
        
        # Store active connection
        active_sessions[session_id] = websocket
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Process message (similar to send_chat_message but via WebSocket)
                user_message = message_data.get("message", "")
                
                if user_message.strip():
                    # Get session info (simplified for WebSocket)
                    session = await db_manager.get_chat_session(session_id)
                    
                    if session:
                        # Get context and generate response
                        context = await get_chat_context(user_message, session_id)
                        ai_response = await generate_ai_response(
                            message=user_message,
                            context=context,
                            session=session
                        )
                        
                        # Send response back
                        response = {
                            "type": "message",
                            "role": "assistant",
                            "content": ai_response.get("content", ""),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        await websocket.send_text(json.dumps(response))
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        finally:
            # Clean up
            if session_id in active_sessions:
                del active_sessions[session_id]
                
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        if session_id in active_sessions:
            del active_sessions[session_id]

# Helper functions
async def get_chat_context(message: str, session_id: str) -> List[Dict[str, Any]]:
    """Get relevant context for the chat message."""
    try:
        # Search vector memory for relevant information
        context_results = await vector_memory.search(
            query=message,
            limit=3,
            similarity_threshold=0.7
        )
        
        # Get recent conversation history
        recent_messages = await db_manager.get_chat_messages(
            session_id=session_id,
            limit=5,
            offset=0
        )
        
        return {
            "knowledge_base": context_results,
            "conversation_history": recent_messages
        }
    except Exception as e:
        logger.warning(f"Error getting chat context: {e}")
        return {"knowledge_base": [], "conversation_history": []}

async def generate_ai_response(
    message: str,
    context: Dict[str, Any],
    session: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate AI response using the configured model and agents."""
    try:
        model_name = session.get("model_used", "llama2")
        agent_type = session.get("agent_type", "chat")
        
        # Build prompt with context
        prompt_parts = []
        
        # Add knowledge context if available
        if context.get("knowledge_base"):
            prompt_parts.append("Relevant information:")
            for item in context["knowledge_base"][:2]:  # Limit context
                prompt_parts.append(f"- {item.get('content', '')[:200]}...")
            prompt_parts.append("")
        
        # Add conversation history
        if context.get("conversation_history"):
            prompt_parts.append("Recent conversation:")
            for msg in context["conversation_history"][-3:]:  # Last 3 messages
                role = msg.get("role", "")
                content = msg.get("content", "")[:100]
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")
        
        # Add current message
        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Generate response using model manager
        if model_name in model_manager.loaded_models:
            result = await model_manager.generate_text(
                model_name=model_name,
                prompt=full_prompt,
                max_tokens=500,
                temperature=0.7
            )
            
            return {
                "content": result.get("text", "I'm sorry, I couldn't generate a response."),
                "tokens_used": result.get("tokens_used", 0),
                "model": model_name
            }
        else:
            # Fallback response
            return {
                "content": "I'm currently not able to process your request. Please try again later.",
                "tokens_used": 0,
                "model": "fallback"
            }
            
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return {
            "content": "I encountered an error while processing your request. Please try again.",
            "tokens_used": 0,
            "model": "error"
        }
