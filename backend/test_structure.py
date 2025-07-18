#!/usr/bin/env python3
"""
SutazAI Backend Structure Test
Tests the backend structure without requiring external dependencies
"""

import sys
import os
from pathlib import Path

def test_file_structure():
    """Test if all required files exist"""
    base_dir = Path(__file__).parent
    
    required_files = [
        "main_complete.py",
        "Dockerfile",
        "requirements.txt",
        ".env",
        "core/config.py",
        "core/database.py",
        "core/cache.py",
        "core/security.py",
        "core/monitoring.py",
        "core/logging_config.py",
        "services/agent_orchestrator.py",
        "services/model_manager.py",
        "services/vector_store.py",
        "services/document_processor.py",
        "services/code_generator.py",
        "services/web_automation.py",
        "services/financial_analyzer.py",
        "services/workflow_engine.py",
        "services/backup_manager.py",
        "api/v1/__init__.py",
        "api/v1/health.py",
        "api/v1/agents.py",
        "api/v1/models.py",
        "api/v1/documents.py",
        "api/v1/chat.py",
        "api/v1/workflows.py",
        "api/v1/admin.py"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print("=== SutazAI Backend Structure Test ===")
    print(f"✓ Found {len(existing_files)} required files")
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} files:")
        for file in missing_files:
            print(f"   - {file}")
    else:
        print("✓ All required files present")
    
    return len(missing_files) == 0

def test_import_structure():
    """Test Python import structure (syntax only)"""
    print("\n=== Testing Python Import Structure ===")
    
    try:
        # Test if Python can parse our files
        import ast
        
        python_files = [
            "main_complete.py",
            "core/config.py",
            "core/database.py",
            "core/cache.py",
            "core/security.py",
            "core/monitoring.py",
            "core/logging_config.py",
            "services/agent_orchestrator.py",
            "services/model_manager.py",
            "services/vector_store.py",
            "api/v1/health.py",
            "api/v1/agents.py",
            "api/v1/models.py",
            "api/v1/chat.py"
        ]
        
        base_dir = Path(__file__).parent
        syntax_errors = []
        
        for file_path in python_files:
            full_path = base_dir / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    ast.parse(content)
                    print(f"✓ {file_path} - syntax OK")
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path}: {e}")
                    print(f"❌ {file_path} - syntax error: {e}")
        
        if syntax_errors:
            print(f"\n❌ Found {len(syntax_errors)} syntax errors")
            return False
        else:
            print(f"\n✓ All {len(python_files)} Python files have valid syntax")
            return True
            
    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        return False

def test_docker_files():
    """Test Docker configuration files"""
    print("\n=== Testing Docker Configuration ===")
    
    base_dir = Path(__file__).parent
    
    # Check Dockerfile
    dockerfile = base_dir / "Dockerfile"
    if dockerfile.exists():
        print("✓ Dockerfile exists")
        
        # Check if it contains required instructions
        with open(dockerfile, 'r') as f:
            content = f.read()
        
        required_instructions = ["FROM", "WORKDIR", "COPY", "RUN", "EXPOSE", "CMD"]
        missing_instructions = []
        
        for instruction in required_instructions:
            if instruction not in content:
                missing_instructions.append(instruction)
        
        if missing_instructions:
            print(f"❌ Dockerfile missing instructions: {missing_instructions}")
            return False
        else:
            print("✓ Dockerfile contains all required instructions")
    else:
        print("❌ Dockerfile not found")
        return False
    
    # Check docker-compose files
    docker_compose_files = [
        base_dir.parent / "docker-compose.yml",
        base_dir.parent / "docker-compose-complete.yml"
    ]
    
    for compose_file in docker_compose_files:
        if compose_file.exists():
            print(f"✓ {compose_file.name} exists")
        else:
            print(f"❌ {compose_file.name} not found")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing SutazAI Backend Structure\n")
    
    results = []
    
    # Test file structure
    results.append(test_file_structure())
    
    # Test Python syntax
    results.append(test_import_structure())
    
    # Test Docker files
    results.append(test_docker_files())
    
    # Summary
    print("\n" + "="*50)
    if all(results):
        print("🎉 All tests passed! Backend structure is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start services: docker-compose up")
        print("3. Run backend: python main_complete.py")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)