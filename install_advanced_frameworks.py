#!/usr/bin/env python3
"""
Advanced AI Framework Installation Script for SutazAI v3+
This script installs all advanced ML/AI frameworks including computer vision,
specialized neural networks, and advanced NLP tools.
"""

import subprocess
import sys
import logging
import os
import platform
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description="", ignore_errors=False):
    """Run a shell command and handle errors."""
    try:
        logger.info(f"Running: {description or command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description or command}")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            logger.warning(f"Warning (ignored): {description or command}")
            logger.warning(f"Error: {e.stderr}")
            return False
        else:
            logger.error(f"Failed: {description or command}")
            logger.error(f"Error: {e.stderr}")
            return False

def detect_system_info():
    """Detect system information for optimized installation."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python_version": sys.version_info[:2],
        "has_cuda": False,
        "has_apt": False,
        "has_yum": False,
        "has_brew": False
    }
    
    # Check for CUDA
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
        info["has_cuda"] = result.returncode == 0
    except:
        info["has_cuda"] = False
    
    # Check package managers
    for manager in ["apt", "yum", "brew"]:
        try:
            result = subprocess.run(f"which {manager}", shell=True, capture_output=True)
            info[f"has_{manager}"] = result.returncode == 0
        except:
            info[f"has_{manager}"] = False
    
    return info

def install_system_dependencies(system_info):
    """Install system dependencies based on the platform."""
    logger.info("Installing system dependencies...")
    
    if system_info["has_apt"]:  # Ubuntu/Debian
        commands = [
            "apt update",
            "apt install -y build-essential cmake",
            "apt install -y libopencv-dev python3-opencv",
            "apt install -y libfann-dev libfann2",
            "apt install -y pkg-config",
            "apt install -y libjpeg-dev libtiff5-dev libpng-dev",
            "apt install -y libavcodec-dev libavformat-dev libswscale-dev",
            "apt install -y libgtk2.0-dev libcanberra-gtk-module",
            "apt install -y libxvidcore-dev libx264-dev",
            "apt install -y libatlas-base-dev gfortran",
            "apt install -y libboost-all-dev",
            "apt install -y libdlib-dev"
        ]
        
        for cmd in commands:
            run_command(f"sudo {cmd}", f"Installing system package: {cmd}", ignore_errors=True)
    
    elif system_info["has_yum"]:  # CentOS/RHEL/Fedora
        commands = [
            "yum groupinstall -y 'Development Tools'",
            "yum install -y cmake opencv-devel",
            "yum install -y fann-devel",
            "yum install -y boost-devel"
        ]
        
        for cmd in commands:
            run_command(f"sudo {cmd}", f"Installing system package: {cmd}", ignore_errors=True)
    
    elif system_info["has_brew"]:  # macOS
        commands = [
            "brew install cmake",
            "brew install opencv",
            "brew install fann",
            "brew install boost"
        ]
        
        for cmd in commands:
            run_command(cmd, f"Installing with Homebrew: {cmd}", ignore_errors=True)

def install_computer_vision():
    """Install computer vision frameworks."""
    logger.info("Installing computer vision frameworks...")
    
    packages = [
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0",
        "pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "imageio>=2.31.0",
        "face-recognition>=1.3.0",
        "mediapipe>=0.10.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}", ignore_errors=True)
    
    # Try to install dlib (can be problematic)
    logger.info("Installing dlib (may take several minutes)...")
    run_command("pip install dlib>=19.24.0", "Installing dlib", ignore_errors=True)

def install_neural_networks():
    """Install specialized neural network libraries."""
    logger.info("Installing neural network frameworks...")
    
    packages = [
        "fann2>=1.0.7",
        "chainer>=7.8.1"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}", ignore_errors=True)
    
    # CuPy for GPU acceleration (if CUDA available)
    system_info = detect_system_info()
    if system_info["has_cuda"]:
        logger.info("CUDA detected, installing CuPy with GPU support...")
        run_command("pip install cupy>=12.0.0", "Installing CuPy with CUDA", ignore_errors=True)
    else:
        logger.info("No CUDA detected, skipping CuPy installation")

def install_advanced_nlp():
    """Install advanced NLP frameworks."""
    logger.info("Installing advanced NLP frameworks...")
    
    packages = [
        "allennlp>=2.10.0",
        "allennlp-models>=2.10.0",
        "polyglot>=16.7.4",
        "textstat>=0.7.3",
        "jellyfish>=0.11.0",
        "python-Levenshtein>=0.21.0",
        "rapidfuzz>=3.1.0",
        "langdetect>=1.0.9",
        "langid.py>=1.1.6",
        "newspaper3k>=0.2.8",
        "readability>=0.3.1",
        "yake>=0.4.8"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}", ignore_errors=True)
    
    # Download Polyglot models
    logger.info("Downloading Polyglot language models...")
    polyglot_download_script = '''
import polyglot
from polyglot.downloader import downloader

# Download essential language packs
languages = ["en", "es", "fr", "de", "zh", "ja", "ar", "hi", "pt", "ru"]
for lang in languages:
    try:
        downloader.download(f"embeddings2.{lang}")
        downloader.download(f"ner2.{lang}")
        print(f"Downloaded {lang} models")
    except Exception as e:
        print(f"Failed to download {lang}: {e}")
'''
    
    script_path = "/tmp/download_polyglot.py"
    with open(script_path, 'w') as f:
        f.write(polyglot_download_script)
    
    run_command(f"python {script_path}", "Downloading Polyglot models", ignore_errors=True)
    try:
        os.remove(script_path)
    except:
        pass

def install_audio_processing():
    """Install audio processing libraries."""
    logger.info("Installing audio processing libraries...")
    
    packages = [
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "speechrecognition>=3.10.0",
        "pydub>=0.25.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}", ignore_errors=True)

def install_specialized_ml():
    """Install specialized ML libraries."""
    logger.info("Installing specialized ML libraries...")
    
    packages = [
        "statsmodels>=0.14.0",
        "pyod>=1.0.9",
        "torch-geometric>=2.3.0",
        "networkx>=3.1.0",
        "optuna>=3.2.0",
        "plotly>=5.15.0",
        "seaborn>=0.12.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}", ignore_errors=True)

def install_model_optimization():
    """Install model optimization tools."""
    logger.info("Installing model optimization tools...")
    
    packages = [
        "onnxmltools>=1.11.0",
        "numba>=0.57.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}", ignore_errors=True)

def install_distributed_computing():
    """Install distributed computing frameworks."""
    logger.info("Installing distributed computing frameworks...")
    
    packages = [
        "ray>=2.5.0",
        "dask[complete]>=2023.6.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}", ignore_errors=True)

def install_monitoring_tools():
    """Install monitoring and visualization tools."""
    logger.info("Installing monitoring tools...")
    
    packages = [
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "mlflow>=2.4.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}", ignore_errors=True)

def verify_advanced_installations():
    """Verify advanced framework installations."""
    logger.info("Verifying advanced installations...")
    
    test_script = '''
import sys
import importlib

# Test core packages
packages_to_test = [
    ("cv2", "OpenCV"),
    ("face_recognition", "Face Recognition"),
    ("chainer", "Chainer"),
    ("allennlp", "AllenNLP"),
    ("polyglot", "Polyglot"),
    ("librosa", "Librosa"),
    ("ray", "Ray"),
    ("wandb", "Weights & Biases"),
    ("optuna", "Optuna")
]

successful = []
failed = []

for package, name in packages_to_test:
    try:
        importlib.import_module(package)
        print(f"✓ {name}")
        successful.append(name)
    except ImportError as e:
        print(f"✗ {name}: {e}")
        failed.append(name)

print(f"\\nSummary:")
print(f"Successful: {len(successful)}/{len(packages_to_test)}")
print(f"Failed: {failed}")

if len(successful) >= len(packages_to_test) * 0.7:
    print("\\n🎉 Advanced frameworks installation mostly successful!")
    sys.exit(0)
else:
    print("\\n⚠️ Many advanced frameworks failed to install")
    sys.exit(1)
'''
    
    script_path = "/tmp/verify_advanced.py"
    with open(script_path, 'w') as f:
        f.write(test_script)
    
    success = run_command(f"python {script_path}", "Advanced package verification")
    try:
        os.remove(script_path)
    except:
        pass
    
    return success

def main():
    """Main installation process for advanced frameworks."""
    logger.info("Starting Advanced AI Frameworks Installation for SutazAI v3+...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    
    # Detect system
    system_info = detect_system_info()
    logger.info(f"Detected system: {system_info}")
    
    # Upgrade pip first
    run_command("pip install --upgrade pip setuptools wheel", "Upgrading pip and tools")
    
    # Installation steps
    steps = [
        ("System Dependencies", lambda: install_system_dependencies(system_info)),
        ("Computer Vision", install_computer_vision),
        ("Neural Networks", install_neural_networks),
        ("Advanced NLP", install_advanced_nlp),
        ("Audio Processing", install_audio_processing),
        ("Specialized ML", install_specialized_ml),
        ("Model Optimization", install_model_optimization),
        ("Distributed Computing", install_distributed_computing),
        ("Monitoring Tools", install_monitoring_tools)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Step: {step_name}")
        logger.info(f"{'='*60}")
        
        try:
            step_func()
            logger.info(f"Completed: {step_name}")
        except Exception as e:
            failed_steps.append(step_name)
            logger.error(f"Failed: {step_name} - {e}")
    
    # Verification
    logger.info(f"\n{'='*60}")
    logger.info("Verification")
    logger.info(f"{'='*60}")
    
    if not verify_advanced_installations():
        failed_steps.append("Verification")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Advanced Installation Summary")
    logger.info(f"{'='*60}")
    
    if failed_steps:
        logger.warning(f"Some steps failed: {', '.join(failed_steps)}")
        logger.warning("Some advanced features may not be available.")
        logger.info("SutazAI will still function with reduced capabilities.")
    else:
        logger.info("🚀 All advanced frameworks installed successfully!")
        logger.info("SutazAI v3+ is now ready for advanced AI operations!")
    
    # Post-installation notes
    logger.info(f"\n{'='*60}")
    logger.info("Post-Installation Notes")
    logger.info(f"{'='*60}")
    logger.info("• Some frameworks may require additional configuration")
    logger.info("• GPU frameworks require CUDA/ROCm for acceleration")
    logger.info("• Computer vision models may need additional downloads")
    logger.info("• Check framework documentation for advanced features")
    logger.info("• Test individual frameworks before production use")

if __name__ == "__main__":
    main()