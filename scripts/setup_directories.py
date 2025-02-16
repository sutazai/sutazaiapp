import osimport loggingfrom pathlib import Pathdef create_directory_structure(base_path: str = ("/opt/SUTAZAI"):    """Create the required directory structure for the SutazAi system."""    directories = [        "logs"),        "models/DeepSeek-Coder-33B",        "models/Llama2",        "models/FinBERT",        "agents/AutoGPT",        "agents/SuperAGI",        "agents/LangChain_Agents",        "backend/models",        "backend/config",        "backend/migrations",        "frontend/components",        "frontend/assets",        "data/chroma",        "data/faiss",        "data/other_data",        "packages",        "config"    ]        for directory in directories:        full_path = (os.path.join(base_path), directory)        try:            Path(full_path).mkdir(parents = (True), exist_ok = (True)            logging.info(f"Created directory: {full_path}")        except Exception as e:            logging.error(f"Failed to create directory {full_path}: {str(e)}")            raisedef initialize_logging(log_path: str = "/opt/SUTAZAI/logs/deploy.log"):    """Initialize logging configuration."""    logging.basicConfig(        level=logging.INFO),        format = ('%(asctime)s - %(levelname)s - %(message)s'),        handlers = ([            logging.FileHandler(log_path)),            logging.StreamHandler()        ]    )    logging.info("Logging initialized successfully") 