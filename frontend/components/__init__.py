"""Frontend Components"""

# Import components with graceful fallback for missing dependencies
try:
    from .voice_assistant import VoiceAssistant
    VOICE_AVAILABLE = True
except ImportError as e:
    print(f"Voice assistant not available: {e}")
    VoiceAssistant = None
    VOICE_AVAILABLE = False

try:
    from .chat_interface import ChatInterface
except ImportError as e:
    print(f"Chat interface not available: {e}")
    ChatInterface = None

try:
    from .system_monitor import SystemMonitor
except ImportError as e:
    print(f"System monitor not available: {e}")
    SystemMonitor = None

__all__ = ["VoiceAssistant", "ChatInterface", "SystemMonitor", "VOICE_AVAILABLE"]