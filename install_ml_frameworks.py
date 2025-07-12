#!/usr/bin/env python3
"""
Installation script for ML frameworks in SutazAI system.
This script handles the installation and setup of various ML/NLP frameworks.
"""

import subprocess
import sys
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    try:
        logger.info(f"Running: {description or command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description or command}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description or command}")
        logger.error(f"Error: {e.stderr}")
        return False

def install_pytorch():
    """Install PyTorch with appropriate configuration."""
    logger.info("Installing PyTorch...")
    
    # Detect if CUDA is available
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        has_cuda = result.returncode == 0
    except:
        has_cuda = False
    
    if has_cuda:
        logger.info("CUDA detected, installing PyTorch with CUDA support")
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        logger.info("No CUDA detected, installing CPU-only PyTorch")
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(command, "PyTorch installation")

def install_tensorflow():
    """Install TensorFlow."""
    logger.info("Installing TensorFlow...")
    return run_command("pip install tensorflow>=2.13.0", "TensorFlow installation")

def install_transformers():
    """Install Hugging Face Transformers and related packages."""
    logger.info("Installing Transformers ecosystem...")
    packages = [
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "datasets>=2.13.0",
        "evaluate>=0.4.0",
        "sentence-transformers>=2.2.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    return True

def install_spacy():
    """Install spaCy and download language models."""
    logger.info("Installing spaCy...")
    
    if not run_command("pip install spacy>=3.6.0", "spaCy installation"):
        return False
    
    # Download English model
    logger.info("Downloading spaCy English model...")
    if not run_command("python -m spacy download en_core_web_sm", "spaCy English model"):
        logger.warning("Failed to download spaCy English model - will try later")
    
    return True

def install_nltk():
    """Install NLTK and download data."""
    logger.info("Installing NLTK...")
    
    if not run_command("pip install nltk>=3.8.0", "NLTK installation"):
        return False
    
    # Download NLTK data
    logger.info("Downloading NLTK data...")
    nltk_data_script = '''
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download essential NLTK data
downloads = ['punkt', 'vader_lexicon', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
for item in downloads:
    try:
        nltk.download(item, quiet=True)
        print(f"Downloaded {item}")
    except Exception as e:
        print(f"Failed to download {item}: {e}")
'''
    
    # Write script to temp file and execute
    script_path = "/tmp/download_nltk.py"
    with open(script_path, 'w') as f:
        f.write(nltk_data_script)
    
    run_command(f"python {script_path}", "NLTK data download")
    os.remove(script_path)
    
    return True

def install_onnx():
    """Install ONNX and ONNX Runtime."""
    logger.info("Installing ONNX...")
    packages = ["onnx>=1.14.0", "onnxruntime>=1.15.0"]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    return True

def install_additional_nlp():
    """Install additional NLP libraries."""
    logger.info("Installing additional NLP libraries...")
    packages = [
        "gensim>=4.3.0",
        "textblob>=0.17.0",
        "flair>=0.12.0",
        "langdetect>=1.0.9"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            logger.warning(f"Failed to install {package} - continuing...")
    
    return True

def install_scientific_computing():
    """Install scientific computing libraries."""
    logger.info("Installing scientific computing libraries...")
    packages = [
        "numpy>=1.24.0",
        "scipy>=1.10.0", 
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    return True

def install_monitoring():
    """Install monitoring and utility packages."""
    logger.info("Installing monitoring tools...")
    packages = [
        "psutil>=5.9.0",
        "tqdm>=4.65.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    return True

def verify_installations():
    """Verify that key packages are installed correctly."""
    logger.info("Verifying installations...")
    
    test_script = '''
import sys
import importlib

packages_to_test = [
    'torch', 'tensorflow', 'transformers', 'spacy', 'nltk', 
    'onnx', 'onnxruntime', 'numpy', 'scipy', 'sklearn', 'pandas'
]

failed_imports = []
for package in packages_to_test:
    try:
        importlib.import_module(package)
        print(f"✓ {package}")
    except ImportError as e:
        print(f"✗ {package}: {e}")
        failed_imports.append(package)

if failed_imports:
    print(f"\\nFailed to import: {', '.join(failed_imports)}")
    sys.exit(1)
else:
    print("\\nAll packages imported successfully!")
'''
    
    script_path = "/tmp/verify_ml.py"
    with open(script_path, 'w') as f:
        f.write(test_script)
    
    success = run_command(f"python {script_path}", "Package verification")
    os.remove(script_path)
    
    return success

def main():
    """Main installation process."""
    logger.info("Starting ML frameworks installation for SutazAI...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    
    # Upgrade pip first
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Installation steps
    steps = [
        ("Scientific Computing Libraries", install_scientific_computing),
        ("PyTorch", install_pytorch),
        ("TensorFlow", install_tensorflow),
        ("Transformers Ecosystem", install_transformers),
        ("spaCy", install_spacy),
        ("NLTK", install_nltk),
        ("ONNX", install_onnx),
        ("Additional NLP Libraries", install_additional_nlp),
        ("Monitoring Tools", install_monitoring)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"Step: {step_name}")
        logger.info(f"{'='*50}")
        
        if not step_func():
            failed_steps.append(step_name)
            logger.error(f"Failed: {step_name}")
        else:
            logger.info(f"Completed: {step_name}")
    
    # Verification
    logger.info(f"\n{'='*50}")
    logger.info("Verification")
    logger.info(f"{'='*50}")
    
    if verify_installations():
        logger.info("✓ Installation verification successful!")
    else:
        failed_steps.append("Verification")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Installation Summary")
    logger.info(f"{'='*50}")
    
    if failed_steps:
        logger.error(f"Failed steps: {', '.join(failed_steps)}")
        logger.error("Some installations failed. Check logs above for details.")
        sys.exit(1)
    else:
        logger.info("🎉 All ML frameworks installed successfully!")
        logger.info("SutazAI is now ready for advanced ML/NLP operations!")

if __name__ == "__main__":
    main()