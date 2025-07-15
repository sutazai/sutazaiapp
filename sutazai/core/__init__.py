"""
Sutazai - Advanced Self-Improving AGI System
Core module initialization
"""

__version__ = "1.0.0"
__author__ = "Chris Suta"
__email__ = "os.getenv("ADMIN_EMAIL", "admin@localhost")"

# Core system components - import modules first, then classes
from . import cgm
from . import kg  
from . import acm
from . import sutazai_core
from . import secure_storage

# Import main classes
from .cgm import CodeGenerationModule
from .kg import KnowledgeGraph
from .acm import AuthorizationControlModule
from .sutazai_core import SutazaiCore
from .secure_storage import SecureStorageSystem

# Import global instances for easy access
from .cgm import code_generation_module
from .kg import knowledge_graph
from .acm import authorization_control_module
from .sutazai_core import sutazai_core
from .secure_storage import secure_storage_system

__all__ = [
    "CodeGenerationModule",
    "KnowledgeGraph", 
    "AuthorizationControlModule",
    "SutazaiCore",
    "SecureStorageSystem",
    "code_generation_module",
    "knowledge_graph", 
    "authorization_control_module",
    "sutazai_core",
    "secure_storage_system"
]