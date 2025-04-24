import logging
from typing import Dict, Any, Optional, AsyncGenerator, Union, List

from sutazai_agi.agents.agent_manager import get_agent_manager, AgentManager
from sutazai_agi.vector_store.vector_store_interface import VectorStoreInterface

logger = logging.getLogger(__name__)

class ChatService:
    """Handles the business logic for chat interactions, including routing."""
    def __init__(self, agent_manager: AgentManager, vector_store: VectorStoreInterface):
        self.agent_manager = agent_manager
        self.vector_store = vector_store
        # TODO: Consider initializing default agent or managing agent sessions here

    async def process_message(self, agent_name: str, messages: List[Dict[str, str]], session_id: Optional[str] = None, stream: bool = False) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Processes an incoming chat message using the specified agent."""
        if not messages:
            raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

        # Extract the latest user message as the primary input for non-LangChain agents
        # LangChain agents usually process the whole message history
        last_user_message = next((msg["content"] for msg in reversed(messages) if msg.get("role") == "user"), None)
        if last_user_message is None:
             raise HTTPException(status_code=400, detail="No user message found in the messages list.")

        # TODO: Enhance agent selection logic. Maybe allow default or route based on query?
        # TODO: Improve context/memory handling. Fetch relevant history from vector store based on session_id/user_id before calling agent.
        # TODO: Pass message history correctly, especially for LangChain agents. Non-LangChain agents might just need the last message (`last_user_message`).
        # TODO: Ensure the tools passed to the agent_manager align with the selected agent's capabilities defined in agents.yaml.

        logger.info(f"Processing message for agent '{agent_name}'. Streaming: {stream}")

        if stream:
            # TODO: Implement streaming logic if agent manager supports it
            # Currently, agent manager's execute_task is non-streaming. Need a stream_task method.
            async def not_implemented_stream():
                 yield {"type": "error", "message": "Streaming not yet implemented for this service."}
            logger.warning("Streaming requested but not implemented in ChatService.")
            return not_implemented_stream()
            # Example for future streaming:
            # return self.agent_manager.stream_task(agent_name=agent_name, user_input=last_user_message, session_id=session_id) # Assuming stream_task exists
        else:
            # Non-streaming execution
            result = await self.agent_manager.execute_task(
                agent_name=agent_name,
                user_input=last_user_message, # Pass last message as primary input
                session_id=session_id
                # TODO: Pass full message history if needed by agent type (e.g., inside agent_config or context?)
            )
            
            # TODO: Process the result - e.g., add AI response back to conversation history in vector store?
            
            if result.get("status") == "error":
                 # Don't raise HTTPException here, let the API layer handle it based on the status
                 logger.error(f"Agent execution failed: {result.get('message')}")
                 # Return the error dict for the API layer
                 return result
                 # raise HTTPException(status_code=500, detail=result.get("message", "Agent execution failed"))

            return result # Return the successful result dictionary


# --- Global Chat Service Instance --- 
_chat_service: Optional[ChatService] = None

def get_chat_service() -> ChatService:
    """Returns a singleton instance of the ChatService."""
    global _chat_service
    if _chat_service is None:
        try:
            agent_manager = get_agent_manager() # Ensure agent manager is initialized
            _chat_service = ChatService(agent_manager)
            logger.info("ChatService initialized.")
        except Exception as e:
             logger.critical(f"Failed to initialize ChatService: {e}", exc_info=True)
             _chat_service = None
             raise # Re-raise critical error
             
    if _chat_service is None:
         raise RuntimeError("ChatService initialization failed previously.")
         
    return _chat_service 