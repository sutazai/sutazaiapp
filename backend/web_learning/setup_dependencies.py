#!/usr/bin/env python3
"""
Setup Dependencies for SutazAI V7 Self-Supervised Learning Pipeline
Automatically installs required packages and sets up the environment
"""

import sys
import subprocess
import os
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a system command with error handling - SECURE VERSION"""
    print(f"Running: {command}")
    if description:
        print(f"Description: {description}")
    
    try:
        # Convert string commands to secure list format
        if isinstance(command, str):
            # Parse command string into safe list - basic shlex splitting
            import shlex
            command_list = shlex.split(command)
        else:
            command_list = command
            
        result = subprocess.run(command_list, check=True, capture_output=True, text=True)
        print(f"✓ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported")
        print("Please upgrade to Python 3.8 or higher")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_pip_packages():
    """Install Python packages from requirements.txt"""
    print("\nInstalling Python packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install packages
    if not run_command(f"{sys.executable} -m pip install -r {requirements_file}", "Installing packages from requirements.txt"):
        return False
    
    return True

def install_nltk_data():
    """Install NLTK data packages"""
    print("\nInstalling NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        datasets = [
            'punkt',
            'stopwords', 
            'wordnet',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words',
            'vader_lexicon',
            'omw-1.4'
        ]
        
        for dataset in datasets:
            try:
                nltk.download(dataset, quiet=True)
                print(f"✓ Downloaded NLTK dataset: {dataset}")
            except Exception as e:
                print(f"✗ Failed to download {dataset}: {e}")
        
        return True
        
    except ImportError:
        print("✗ NLTK not installed")
        return False

def install_spacy_models():
    """Install spaCy language models"""
    print("\nInstalling spaCy models...")
    
    try:
        import spacy
        
        # Install English model
        if not run_command(f"{sys.executable} -m spacy download en_core_web_sm", "Installing English model"):
            print("⚠ Warning: spaCy English model installation failed")
        
        return True
        
    except ImportError:
        print("✗ spaCy not installed")
        return False

def setup_browser_drivers():
    """Setup browser drivers for web automation"""
    print("\nSetting up browser drivers...")
    
    system = platform.system().lower()
    
    if system == "linux":
        # Install Chrome/Chromium
        commands = [
            "wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -",
            "echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' | sudo tee /etc/apt/sources.list.d/google-chrome.list",
            "sudo apt-get update",
            "sudo apt-get install -y google-chrome-stable"
        ]
        
        for cmd in commands:
            if not run_command(cmd, "Installing Chrome"):
                print("⚠ Warning: Chrome installation failed")
                break
        
        # Install ChromeDriver
        if not run_command("sudo apt-get install -y chromium-chromedriver", "Installing ChromeDriver"):
            print("⚠ Warning: ChromeDriver installation failed")
    
    elif system == "darwin":  # macOS
        # Install Chrome via Homebrew
        commands = [
            "brew install --cask google-chrome",
            "brew install chromedriver"
        ]
        
        for cmd in commands:
            if not run_command(cmd, "Installing Chrome/ChromeDriver"):
                print("⚠ Warning: Chrome/ChromeDriver installation failed")
                break
    
    else:
        print("⚠ Warning: Automatic browser driver setup not supported on this platform")
        print("Please manually install Chrome and ChromeDriver")
    
    return True

def setup_system_dependencies():
    """Setup system-level dependencies"""
    print("\nSetting up system dependencies...")
    
    system = platform.system().lower()
    
    if system == "linux":
        # Update package lists
        run_command("sudo apt-get update", "Updating package lists")
        
        # Install system dependencies
        dependencies = [
            "build-essential",
            "python3-dev",
            "python3-pip",
            "libxml2-dev",
            "libxslt1-dev",
            "zlib1g-dev",
            "libjpeg-dev",
            "libpng-dev",
            "libssl-dev",
            "libffi-dev",
            "libcurl4-openssl-dev",
            "curl",
            "wget",
            "git"
        ]
        
        for dep in dependencies:
            if not run_command(f"sudo apt-get install -y {dep}", f"Installing {dep}"):
                print(f"⚠ Warning: {dep} installation failed")
    
    elif system == "darwin":  # macOS
        # Install Xcode command line tools
        run_command("xcode-select --install", "Installing Xcode command line tools")
        
        # Install Homebrew dependencies
        dependencies = [
            "python3",
            "libxml2",
            "libxslt",
            "openssl",
            "curl",
            "wget",
            "git"
        ]
        
        for dep in dependencies:
            if not run_command(f"brew install {dep}", f"Installing {dep}"):
                print(f"⚠ Warning: {dep} installation failed")
    
    else:
        print("⚠ Warning: Automatic system dependency setup not supported on this platform")
    
    return True

def create_virtual_environment():
    """Create a virtual environment for the project"""
    print("\nCreating virtual environment...")
    
    venv_path = Path(__file__).parent / "venv"
    
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
    
    if not run_command(f"{sys.executable} -m venv {venv_path}", "Creating virtual environment"):
        return False
    
    # Activate virtual environment and install packages
    if platform.system().lower() == "windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_executable = venv_path / "Scripts" / "pip.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        pip_executable = venv_path / "bin" / "pip"
    
    if not run_command(f"{pip_executable} install --upgrade pip", "Upgrading pip in virtual environment"):
        return False
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not run_command(f"{pip_executable} install -r {requirements_file}", "Installing packages in virtual environment"):
        return False
    
    print("✓ Virtual environment created and configured")
    print(f"To activate: source {activate_script}")
    
    return True

def test_installation():
    """Test if packages are installed correctly"""
    print("\nTesting installation...")
    
    # Test core packages
    test_packages = [
        "aiohttp",
        "beautifulsoup4",
        "numpy",
        "torch",
        "nltk",
        "selenium",
        "qdrant_client",
        "sentence_transformers"
    ]
    
    failed_packages = []
    
    for package in test_packages:
        try:
            __import__(package)
            print(f"✓ {package} imported successfully")
        except ImportError:
            print(f"✗ {package} import failed")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠ Warning: {len(failed_packages)} packages failed to import:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        return False
    
    print("\n✓ All core packages imported successfully")
    return True

def main():
    """Main setup function"""
    print("SutazAI V7 Self-Supervised Learning Pipeline - Dependency Setup")
    print("=" * 70)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("System Dependencies", setup_system_dependencies),
        ("Python Packages", install_pip_packages),
        ("NLTK Data", install_nltk_data),
        ("spaCy Models", install_spacy_models),
        ("Browser Drivers", setup_browser_drivers),
        ("Installation Test", test_installation)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            print(f"✗ {step_name} failed with exception: {e}")
            failed_steps.append(step_name)
    
    print("\n" + "=" * 70)
    print("SETUP SUMMARY")
    print("=" * 70)
    
    if failed_steps:
        print(f"⚠ {len(failed_steps)} steps failed:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nSome features may not work properly.")
        print("Please check the error messages above and install missing dependencies manually.")
    else:
        print("🎉 All setup steps completed successfully!")
        print("✓ Self-supervised learning pipeline is ready to use")
    
    print("\nNext steps:")
    print("1. Run the test suite: python3 simple_test.py")
    print("2. Check the README.md for usage examples")
    print("3. Start using the learning pipeline!")
    
    return len(failed_steps) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)