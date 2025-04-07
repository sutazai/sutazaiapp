import logging
from typing import Dict, Any, Optional, AsyncGenerator, Union

from sutazai_agi.agents.agent_manager import get_agent_manager, AgentManager

logger = logging.getLogger(__name__)

# --- Define the Router Agent Name (must match config/agents.yaml) ---
ROUTER_AGENT_NAME = "RouterAgent"
# Define allowed specialist agents the router can choose
ALLOWED_SPECIALIST_AGENTS = {"ToolAgent", "SimpleChatAgent"}

class ChatService:
    """Handles the business logic for chat interactions, including routing."""
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        # Check if router and target agents exist
        if ROUTER_AGENT_NAME not in self.agent_manager.available_agents:
            logger.critical(f"Router agent '{ROUTER_AGENT_NAME}' not found or not enabled! Auto-routing disabled.")
            # Fallback or raise error - For now, log critical and routing will fail
        for agent_name in ALLOWED_SPECIALIST_AGENTS:
            if agent_name not in self.agent_manager.available_agents:
                logger.warning(f"Specialist agent '{agent_name}' for router not found or not enabled.")

    async def process_message_or_stream(
        self, 
        query: str, 
        session_id: str, 
        stream: bool = False, 
        requested_agent_name: Optional[str] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Processes a user message, potentially routing it first, handles streaming."""
        
        if requested_agent_name:
            # User explicitly requested an agent, bypass router
            target_agent_name = requested_agent_name
            logger.info(f"User explicitly selected agent: {target_agent_name}")
        else:
            # --- Use Router Agent --- 
            logger.info(f"Routing query for session '{session_id}'. Query: '{query[:50]}...'")
            router_task_input = {"query": query, "session_id": f"{session_id}_router"} # Separate router session?
            
            try:
                # Execute the Router Agent (non-streaming)
                router_result = await self.agent_manager.execute_task(ROUTER_AGENT_NAME, router_task_input)
                
                if router_result.get("status") == "success":
                    chosen_agent = router_result.get("output", "").strip()
                    if chosen_agent in ALLOWED_SPECIALIST_AGENTS:
                        target_agent_name = chosen_agent
                        logger.info(f"Router chose agent: {target_agent_name}")
                    else:
                        logger.warning(f"RouterAgent returned invalid agent name: '{chosen_agent}'. Falling back.")
                        target_agent_name = next(iter(ALLOWED_SPECIALIST_AGENTS)) # Fallback to first specialist
                else:
                    logger.error(f"RouterAgent failed: {router_result.get('message')}. Falling back.")
                    target_agent_name = next(iter(ALLOWED_SPECIALIST_AGENTS)) # Fallback

            except Exception as route_err:
                logger.error(f"Error during routing: {route_err}", exc_info=True)
                target_agent_name = next(iter(ALLOWED_SPECIALIST_AGENTS)) # Fallback
        
        # --- Prepare Input for Specialist Agent --- 
        specialist_task_input = {
            "query": query,
            "session_id": session_id, # Use original session for specialist
            "stream": stream
            # Add any other necessary parameters
        }
        
        # --- Execute or Stream with Chosen Specialist Agent --- 
        logger.info(f"Dispatching task to agent '{target_agent_name}'. Streaming: {stream}")
        if stream:
            # Call the streaming method of AgentManager
            # Ensure the chosen agent type supports streaming (currently checked in stream_langchain_task_manager)
            try:
                # Using stream_langchain_task_manager as example, adapt if AgentManager gets a generic stream_task
                agent_config = self.agent_manager.available_agents.get(target_agent_name)
                if not agent_config or agent_config.get("type") != "langchain": # Example check
                    logger.error(f"Streaming not supported for chosen agent '{target_agent_name}'")
                    # Yield an error chunk
                    async def error_stream(): 
                        yield {"type": "error", "message": f"Streaming not supported for agent '{target_agent_name}'."}
                        # Need to yield StopAsyncIteration? Check generator protocol
                    return error_stream()
                
                return self.agent_manager.stream_langchain_task_manager(target_agent_name, specialist_task_input)
            except Exception as stream_err:
                 logger.error(f"Failed to initiate stream for '{target_agent_name}': {stream_err}", exc_info=True)
                 async def error_stream(): 
                    yield {"type": "error", "message": f"Failed to start stream: {stream_err}"}
                 return error_stream()
        else:
            # Call the non-streaming method
            result = await self.agent_manager.execute_task(target_agent_name, specialist_task_input)
            # Add agent name to result for clarity
            result["agent_name"] = target_agent_name 
            return result


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