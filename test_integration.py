#!/usr/bin/env python3
"""
SutazAI System Integration Test
Quick test to verify system integration and readiness
"""

import os
import sys
import json
from pathlib import Path

def test_file_structure():
    """Test that all required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "docker-compose.yml",
        ".env",
        "deploy.sh",
        "README.md",
        "backend/enhanced_main.py",
        "backend/sutazai_core.py",
        "frontend/streamlit_app.py",
        "docker/backend.Dockerfile",
        "docker/streamlit.Dockerfile",
        "nginx/nginx.conf"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All required files present")
    return True

def test_env_configuration():
    """Test environment configuration"""
    print("🔍 Testing environment configuration...")
    
    if not Path(".env").exists():
        print("  ❌ .env file missing")
        return False
    
    with open(".env", "r") as f:
        env_content = f.read()
    
    required_vars = [
        "SUTAZAI_VERSION",
        "POSTGRES_DB",
        "POSTGRES_USER", 
        "DATABASE_URL",
        "DEFAULT_MODEL",
        "SECRET_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in env_content:
            missing_vars.append(var)
        else:
            print(f"  ✅ {var} defined")
    
    if missing_vars:
        print(f"  ❌ Missing environment variables: {missing_vars}")
        return False
    
    print("  ✅ Environment configuration complete")
    return True

def test_docker_compose():
    """Test Docker Compose configuration"""
    print("🔍 Testing Docker Compose configuration...")
    
    try:
        import yaml
    except ImportError:
        print("  ⚠️ PyYAML not available, skipping detailed config test")
        return True
    
    try:
        with open("docker-compose.yml", "r") as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config.get("services", {})
        
        # Check for essential services
        essential_services = [
            "sutazai-backend",
            "sutazai-streamlit", 
            "postgres",
            "redis",
            "qdrant",
            "ollama",
            "nginx"
        ]
        
        missing_services = []
        for service in essential_services:
            if service not in services:
                missing_services.append(service)
            else:
                print(f"  ✅ Service: {service}")
        
        if missing_services:
            print(f"  ❌ Missing services: {missing_services}")
            return False
        
        print(f"  ✅ Docker Compose configuration valid ({len(services)} services)")
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading Docker Compose config: {e}")
        return False

def test_code_syntax():
    """Test Python code syntax"""
    print("🔍 Testing Python code syntax...")
    
    python_files = [
        "backend/enhanced_main.py",
        "backend/sutazai_core.py",
        "frontend/streamlit_app.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        if Path(file_path).exists():
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                compile(content, file_path, 'exec')
                print(f"  ✅ {file_path} syntax valid")
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
                print(f"  ❌ {file_path} syntax error: {e}")
        else:
            print(f"  ⚠️ {file_path} not found")
    
    if syntax_errors:
        print(f"  ❌ Syntax errors found: {len(syntax_errors)}")
        return False
    
    print("  ✅ All Python files have valid syntax")
    return True

def test_system_readiness():
    """Test overall system readiness"""
    print("🔍 Testing system readiness...")
    
    # Check deployment script
    if Path("deploy.sh").exists() and os.access("deploy.sh", os.X_OK):
        print("  ✅ Deployment script executable")
    else:
        print("  ❌ Deployment script not executable")
        return False
    
    # Check directory structure
    required_dirs = ["backend", "frontend", "docker", "nginx"]
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            print(f"  ✅ Directory: {dir_name}")
        else:
            print(f"  ❌ Missing directory: {dir_name}")
            return False
    
    print("  ✅ System ready for deployment")
    return True

def main():
    """Run all integration tests"""
    print("🚀 SutazAI System Integration Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_env_configuration, 
        test_docker_compose,
        test_code_syntax,
        test_system_readiness
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System ready for deployment.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())