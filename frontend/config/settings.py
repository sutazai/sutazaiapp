"""
Frontend Configuration Settings
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings"""
    
    # Application Info
    APP_NAME = "JARVIS - SutazAI Assistant"
    APP_VERSION = "5.0.0"
    
    # Backend API
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:10200")
    API_TIMEOUT = 30
    
    # Voice Settings
    WAKE_WORD = "jarvis"
    SPEECH_RECOGNITION_ENGINE = "google"  # google, sphinx, whisper
    TTS_ENGINE = "pyttsx3"  # pyttsx3, gTTS
    VOICE_LANGUAGE = "en-US"
    SPEAKING_RATE = 175  # Words per minute
    
    # UI Settings
    THEME = "dark"
    SHOW_SYSTEM_METRICS = True
    ENABLE_VOICE_COMMANDS = True
    ENABLE_TYPING_ANIMATION = True
    
    # Session Settings
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_CHAT_HISTORY = 100
    
    # Audio Settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    AUDIO_FORMAT = "wav"
    SILENCE_THRESHOLD = 500
    
    # AI Agent Settings
    DEFAULT_AGENT = "jarvis"
    AVAILABLE_AGENTS = [
        "jarvis",
        "letta",
        "autogpt",
        "crewai",
        "baby-agi"
    ]
    
    # System Monitoring
    METRICS_UPDATE_INTERVAL = 5  # seconds
    SHOW_CPU_USAGE = True
    SHOW_MEMORY_USAGE = True
    SHOW_NETWORK_USAGE = True
    SHOW_DOCKER_STATS = True
    
    # Features
    ENABLE_FILE_UPLOAD = True
    ENABLE_CODE_EXECUTION = False  # Security: disabled by default
    ENABLE_WEB_SEARCH = True
    ENABLE_DOCUMENT_ANALYSIS = True
    
    # Styling
    PRIMARY_COLOR = "#00D4FF"  # Jarvis blue
    SECONDARY_COLOR = "#FF6B6B"
    SUCCESS_COLOR = "#4CAF50"
    WARNING_COLOR = "#FFC107"
    ERROR_COLOR = "#F44336"

settings = Settings()