import logging
from typing import Optional, Dict, Any, List

# Import Agent Manager
from sutazai_agi.agents.agent_manager import get_agent_manager, AgentManager

# Integration functions are no longer imported/called directly here
# try:
#     from sutazai_agi.agents.integrations.gpt_engineer import run_gpt_engineer
# except ImportError:
#     logger.warning("gpt_engineer integration not found. Generate endpoint will fail.")
#     async def run_gpt_engineer(*args, **kwargs): # Mock function
#         return {"status": "error", "message": "GPT-Engineer integration not available."}
# 
# try:
#     from sutazai_agi.agents.integrations.aider import run_aider
# except ImportError:
#     logger.warning("aider integration not found. Edit endpoint will fail.")
#     async def run_aider(*args, **kwargs): # Mock function
#         return {"status": "error", "message": "Aider integration not available."}

logger = logging.getLogger(__name__)

# Define which agents are responsible for code generation/editing
# These should match names in agents.yaml and be configured with appropriate tools
CODE_GENERATION_AGENT = "AutoGen Multi-Agent" # Or potentially "ToolAgent"
CODE_EDITING_AGENT = "AutoGen Multi-Agent" # Or potentially "ToolAgent"

# Default agent names (can be overridden via config or request)
DEFAULT_GENERATOR_AGENT = "AutoGen Multi-Agent" # Example: Use AutoGen for generation
DEFAULT_EDITOR_AGENT = "Aider Agent"          # Example: Use Aider for editing

class CodeService:
    """Handles code generation and editing tasks by dispatching to appropriate agents."""

    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        logger.info("CodeService initialized with AgentManager.")
        # Verify required agents exist
        if CODE_GENERATION_AGENT not in self.agent_manager.available_agents:
            logger.warning(f"Code generation agent '{CODE_GENERATION_AGENT}' not found/enabled. Code generation may fail.")
        if CODE_EDITING_AGENT not in self.agent_manager.available_agents:
            logger.warning(f"Code editing agent '{CODE_EDITING_AGENT}' not found/enabled. Code editing may fail.")

    async def generate_codebase(self, prompt: str, target_path: Optional[str] = None, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Generates a codebase based on a prompt using an appropriate agent."""
        selected_agent = agent_name or DEFAULT_GENERATOR_AGENT
        logger.info(f"Generating codebase using agent '{selected_agent}' for prompt: {prompt[:50]}...")

        # Prepare input for the agent manager
        # The specific format might depend on the agent (e.g., GPT-Engineer might need prompt in a specific key)
        # For now, pass the prompt as the main user_input
        task_input = prompt
        # TODO: Add target_path to agent_config or context if the agent needs it passed explicitly

        result = await self.agent_manager.execute_task(
            agent_name=selected_agent,
            user_input=task_input
        )

        # TODO: Process result - e.g., confirm output path exists, maybe run basic tests/linting?
        return result

    async def edit_code_files(self, files: List[str], instruction: str, repo_path: Optional[str] = None, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Edits specified code files based on instructions using an appropriate agent."""
        selected_agent = agent_name or DEFAULT_EDITOR_AGENT
        logger.info(f"Editing files {files} using agent '{selected_agent}' with instruction: {instruction[:50]}...")

        # Prepare input for the agent manager
        # Aider agent might expect files and instruction in specific format
        # We might need to pass a structured dict instead of just the instruction string
        # For now, pass instruction as main input, files/repo_path need context handling
        task_input = instruction
        # TODO: Need a way to pass `files` and `repo_path` to the Aider agent via agent_manager
        # This might involve modifying execute_task signature or passing a context dict.

        result = await self.agent_manager.execute_task(
            agent_name=selected_agent,
            user_input=task_input
        )

        # TODO: Process result - e.g., return diff, confirm changes were committed by Aider?
        return result

# --- Dependency Injection (Singleton Pattern) ---

_code_service: Optional[CodeService] = None

def get_code_service() -> CodeService:
    """Provides a singleton instance of the CodeService."""
    global _code_service
    if _code_service is None:
        try:
            agent_manager = get_agent_manager() # Ensure AgentManager is initialized first
            _code_service = CodeService(agent_manager)
            logger.info("CodeService singleton initialized.")
        except Exception as e:
             logger.critical(f"Failed to initialize CodeService: {e}", exc_info=True)
             _code_service = None
             raise # Re-raise critical error
             
    if _code_service is None:
         raise RuntimeError("CodeService initialization failed previously.")
         
    return _code_service 