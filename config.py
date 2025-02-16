"""ConfigurationmodulefortheSutazAIapplication."""#StandardlibraryimportsfromtypingimportListfromfunctoolsimportlru_cacheimportos#Third-partyimportsfrompydanticimportBaseSettingsfromdotenvimportload_dotenvload_dotenv()classSettings(BaseSettings):"""Applicationsettingsloadedfromenvironmentvariables."""#APISettingsAPI_V1_STR:str="/api/v1"PROJECT_NAME:str="SutazAI"VERSION:str="1.0.0"DESCRIPTION:str="SutazAIRESTAPI"#SecuritySECRET_KEY:str=os.getenv("SECRET_KEY","your-secret-key-here")ALGORITHM:str="HS256"ACCESS_TOKEN_EXPIRE_MINUTES:int=30REFRESH_TOKEN_EXPIRE_DAYS:int=7#CORSCORS_ORIGINS: List[Any]=["http://localhost","http://localhost:8000","http://localhost:8501","https://sutazai.com",]#HostsALLOWED_HOSTS: List[Any]=["localhost","127.0.0.1","sutazai.com",]#DatabaseDB_HOST:str=os.getenv("DB_HOST","localhost")DB_PORT:str=os.getenv("DB_PORT","5432")DB_USER:str=os.getenv("DB_USER","postgres")DB_PASS:str=os.getenv("DB_PASS","")DB_NAME:str=os.getenv("DB_NAME","sutazai")#RedisREDIS_HOST:str=os.getenv("REDIS_HOST","localhost")REDIS_PORT:int=int(os.getenv("REDIS_PORT","6379"))REDIS_DB:int=int(os.getenv("REDIS_DB","0"))#RateLimitingRATE_LIMIT_PER_MINUTE:int=60#LoggingLOG_LEVEL:str="INFO"LOG_FORMAT:str="%(asctime)s-%(name)s-%(levelname)s-%(message)s"LOG_FILE:str="/var/log/sutazai/api.log"#EmailSMTP_TLS:bool=TrueSMTP_PORT:int=587SMTP_HOST:str=os.getenv("SMTP_HOST","")SMTP_USER:str=os.getenv("SMTP_USER","")SMTP_PASSWORD:str=os.getenv("SMTP_PASSWORD","")EMAILS_FROM_EMAIL:str=os.getenv("EMAILS_FROM_EMAIL","noreply@sutazai.com")EMAILS_FROM_NAME:str=os.getenv("EMAILS_FROM_NAME","SutazAI")#MonitoringENABLE_METRICS:bool=TrueMETRICS_PORT:int=9090#StorageUPLOAD_DIR:str="/app/uploads"MAX_UPLOAD_SIZE:int=10*1024*1024#10MBclassConfig:case_sensitive=True@lru_cache()defget_settings()->Settings:"""Getthecachedsettingsinstance."""returnSettings()settings=get_settings()

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SecuritySettings:
    """Advanced security configuration management."""
    
    # Security-critical configuration
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'default_very_secret_key_replace_in_production')
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Enhanced CORS and Host settings
    ALLOWED_HOSTS: List[str] = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    CORS_ORIGINS: List[str] = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
    
    # Rate Limiting Configuration
    RATE_LIMIT_REQUESTS: int = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
    RATE_LIMIT_WINDOW: int = int(os.getenv('RATE_LIMIT_WINDOW', '60'))  # seconds
    
    # Advanced Security Configurations
    SECURITY_HEADERS: Dict[str, str] = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'"
    }
    
    # Cryptographic Settings
    PASSWORD_HASH_ITERATIONS: int = 100_000
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate critical security configurations.
        
        Returns:
            bool: Whether the configuration is secure
        """
        checks = [
            len(cls.SECRET_KEY) > 32,
            cls.ACCESS_TOKEN_EXPIRE_MINUTES > 0,
            cls.RATE_LIMIT_REQUESTS > 0,
            cls.RATE_LIMIT_WINDOW > 0
        ]
        return all(checks)
    
    @classmethod
    def get_secure_random_key(cls) -> str:
        """
        Generate a secure random key.
        
        Returns:
            str: Cryptographically secure random key
        """
        import secrets
        return secrets.token_hex(32)

# Create settings instance
settings = SecuritySettings()

# Validate configuration on import
if not settings.validate_config():
    raise ValueError("Insecure configuration detected. Please review security settings.")