import logging # Import logging
from typing import Optional # Add Optional, Any


from backend.core.database import get_db
from backend.models.user_model import User

from jose import JWTError, jwt

from backend.core.config import get_settings, settings
settings = get_settings()

from backend.schemas import TokenPayload # Import TokenPayload
from backend.core import security # Import security module for ALGORITHM

# OAuth2 scheme setup

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import ValidationError

# Attempt direct import of settings
from backend.crud import user_crud # Import user_crud
from backend.models.base_models import User # Import User model from base_models

# Explicitly import sqlmodel.Session for type hinting
from sqlmodel import Session as SQLModelSession

# Import ModelManager
from backend.ai_agents.model_manager import ModelManager

# --- AI Agent and Core Service Imports ---
# Use absolute paths assuming 'backend' is accessible from the root
# Adjust these paths if the structure differs
from backend.ai_agents.agent_manager import AgentManager
from backend.ai_agents.communication.agent_communication import AgentCommunication
from backend.ai_agents.interaction.interaction_manager import InteractionManager
from backend.ai_agents.performance.metrics import PerformanceMetrics
from backend.ai_agents.workflows.workflow_engine import WorkflowEngine
from backend.ai_agents.memory.memory_manager import MemoryManager
from backend.ai_agents.memory.shared_memory import SharedMemoryManager
from backend.ai_agents.health.health_check import HealthCheck

# Assuming these core components exist at these paths
from backend.core.llm.llm_service import LLMService
from backend.core.tools.tool_registry import ToolRegistry
from backend.core.vector_store.vector_store import VectorStore # Assuming VectorStore class
from backend.core.state.app_state import AppState

logger = logging.getLogger(__name__) # Define logger

# --- Global Instances for Dependency Injection ---
# These should be initialized during application startup (e.g., in main.py)
# Use None as default and check for initialization in getters
llm_service_instance: Optional[LLMService] = None
tool_registry_instance: Optional[ToolRegistry] = None
vector_store_instance: Optional[VectorStore] = None
app_state_instance: Optional[AppState] = None
agent_communication_instance: Optional[AgentCommunication] = None
interaction_manager_instance: Optional[InteractionManager] = None
performance_metrics_instance: Optional[PerformanceMetrics] = None
workflow_engine_instance: Optional[WorkflowEngine] = None
memory_manager_instance: Optional[MemoryManager] = None
shared_memory_manager_instance: Optional[SharedMemoryManager] = None
health_check_instance: Optional[HealthCheck] = None
agent_manager_instance: Optional[AgentManager] = None
model_manager_instance: Optional[ModelManager] = None

# Optional: Lock for thread-safe lazy initialization if needed, but prefer startup initialization
_agent_manager_lock = threading.Lock()

# --- Initialization Function (Called from Startup) ---
# Consolidate initialization logic here or keep it in main.py
def initialize_dependencies(
    llm_service: LLMService,
    tool_registry: ToolRegistry,
    vector_store: VectorStore,
    app_state: AppState,
    agent_communication: AgentCommunication,
    interaction_manager: InteractionManager,
    performance_metrics: PerformanceMetrics,
    workflow_engine: WorkflowEngine,
    memory_manager: MemoryManager,
    shared_memory_manager: SharedMemoryManager,
    health_check: HealthCheck,
    agent_manager: AgentManager, # Add AgentManager here
    model_manager: ModelManager, # Add ModelManager here
) -> None:
    """Sets the global instances used by dependency injectors. Call this during app startup."""
    global llm_service_instance, tool_registry_instance, vector_store_instance, app_state_instance
    global agent_communication_instance, interaction_manager_instance, performance_metrics_instance
    global workflow_engine_instance, memory_manager_instance, shared_memory_manager_instance
    global health_check_instance, agent_manager_instance, model_manager_instance

    logger.info("Initializing global dependencies...")
    llm_service_instance = llm_service
    tool_registry_instance = tool_registry
    vector_store_instance = vector_store
    app_state_instance = app_state
    agent_communication_instance = agent_communication
    interaction_manager_instance = interaction_manager
    performance_metrics_instance = performance_metrics
    workflow_engine_instance = workflow_engine
    memory_manager_instance = memory_manager
    shared_memory_manager_instance = shared_memory_manager
    health_check_instance = health_check
    agent_manager_instance = agent_manager
    model_manager_instance = model_manager
    logger.info("Global dependencies initialized.")

# --- Dependency Injectors --- 

# Core Service Injectors
def get_llm_service() -> LLMService:
    if llm_service_instance is None:
        logger.error("LLMService dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM Service not available")
    return llm_service_instance

def get_tool_registry() -> ToolRegistry:
    if tool_registry_instance is None:
        logger.error("ToolRegistry dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Tool Registry not available")
    return tool_registry_instance

def get_vector_store() -> VectorStore:
    if vector_store_instance is None:
        logger.error("VectorStore dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector Store not available")
    return vector_store_instance

def get_app_state() -> AppState:
    if app_state_instance is None:
        logger.error("AppState dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Application State not available")
    return app_state_instance

# Agent Component Injectors
def get_agent_communication() -> AgentCommunication:
    if agent_communication_instance is None:
        logger.error("AgentCommunication dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent Communication Service not available")
    return agent_communication_instance

def get_interaction_manager() -> InteractionManager:
    if interaction_manager_instance is None:
        logger.error("InteractionManager dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Interaction Manager not available")
    return interaction_manager_instance

def get_performance_metrics() -> PerformanceMetrics:
    if performance_metrics_instance is None:
        logger.error("PerformanceMetrics dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Performance Metrics Service not available")
    return performance_metrics_instance

def get_workflow_engine() -> WorkflowEngine:
    if workflow_engine_instance is None:
        logger.error("WorkflowEngine dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Workflow Engine not available")
    return workflow_engine_instance

def get_memory_manager() -> MemoryManager:
    if memory_manager_instance is None:
        logger.error("MemoryManager dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memory Manager not available")
    return memory_manager_instance

def get_shared_memory_manager() -> SharedMemoryManager:
    if shared_memory_manager_instance is None:
        logger.error("SharedMemoryManager dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Shared Memory Manager not available")
    return shared_memory_manager_instance

def get_health_check() -> HealthCheck:
    if health_check_instance is None:
        logger.error("HealthCheck dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Health Check Service not available")
    return health_check_instance

def get_agent_manager() -> AgentManager:
    # Prefer initialization at startup, remove locking and lazy init if possible
    if agent_manager_instance is None:
        logger.critical("AgentManager dependency requested but not initialized.")
        # Optionally attempt lazy init here ONLY if startup init is not feasible
        # with _agent_manager_lock:
        #     if agent_manager_instance is None: 
        #         # ... try to initialize ...
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent Manager not available")
    return agent_manager_instance

def get_model_manager() -> ModelManager:
    """Dependency injector for the ModelManager singleton."""
    if model_manager_instance is None:
        logger.critical("ModelManager dependency requested but not initialized.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model Manager not available")
    return model_manager_instance

# --- Authentication Dependencies ---

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)

async def get_current_user(
    db: SQLModelSession = Depends(get_db), # Use SQLModelSession hint
    token: str = Depends(reusable_oauth2)
) -> User:
    """Dependency to get the current authenticated user."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError) as e:
        logger.error(f"Error decoding token or validating payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )

    # user_crud functions expect SQLModelSession, which db should be.
    # Use correct keyword arg 'id' for get_user_by_id
    user = user_crud.get_user_by_id(db=db, user_id=token_data.sub) # Pass db and use correct user_id param
    if not user:
        logger.warning(f"User not found for ID: {token_data.sub}") # Log warning
        raise HTTPException(status_code=404, detail="User not found")
    return user # Return type is User, matches hint

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to get the current active authenticated user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_current_active_superuser(
    current_user: User = Depends(get_current_active_user), # Depend on active user
) -> User:
    """Dependency to get the current active superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403, detail="The user doesn't have enough privileges" # Use 403 Forbidden
        )
    return current_user

# REMOVE Placeholder functions from original file
# def initialize_core_dependencies(): ...
# def initialize_agent_dependencies(): ...
# def get_interaction_manager(): ... # Now defined above
# def get_monitoring_system(): ... # Replace with specific monitoring injectors if needed 