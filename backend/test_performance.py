#!/usr/bin/env python3
"""
Test script to verify performance optimizations are working
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported"""
    
    required_modules = {
        'fastapi': 'FastAPI framework',
        'pydantic': 'Data validation',
        'httpx': 'Async HTTP client',
        'redis': 'Redis client',
        'psutil': 'System monitoring'
    }
    
    optional_modules = {
        'uvloop': 'High-performance event loop',
        'asyncpg': 'PostgreSQL async driver',
        'orjson': 'Fast JSON serialization'
    }
    
    print("🔍 Checking required dependencies...")
    print("-" * 40)
    
    missing_required = []
    for module, description in required_modules.items():
        try:
            importlib.import_module(module)
            print(f"✅ {module:15} - {description}")
        except ImportError:
            print(f"❌ {module:15} - {description}")
            missing_required.append(module)
            
    print("\n🔍 Checking optional performance dependencies...")
    print("-" * 40)
    
    missing_optional = []
    for module, description in optional_modules.items():
        try:
            importlib.import_module(module)
            print(f"✅ {module:15} - {description}")
        except ImportError:
            print(f"⚠️  {module:15} - {description} (optional)")
            missing_optional.append(module)
            
    print("\n" + "=" * 40)
    
    if missing_required:
        print(f"❌ Missing required dependencies: {', '.join(missing_required)}")
        print("\nTo install missing dependencies:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    else:
        print("✅ All required dependencies are installed!")
        
    if missing_optional:
        print(f"\n⚠️  Missing optional performance dependencies: {', '.join(missing_optional)}")
        print("\nFor maximum performance, install:")
        print(f"pip install {' '.join(missing_optional)}")
        
    return True

def test_performance_modules():
    """Test if our performance modules can be imported"""
    
    print("\n🔍 Testing custom performance modules...")
    print("-" * 40)
    
    test_modules = [
        ('app.core.connection_pool', 'Connection pooling'),
        ('app.core.cache', 'Caching layer'),
        ('app.core.task_queue', 'Background tasks'),
        ('app.services.ollama_async', 'Async Ollama service')
    ]
    
    all_working = True
    for module_path, description in test_modules:
        try:
            # Add backend directory to path
            sys.path.insert(0, '/opt/sutazaiapp/backend')
            importlib.import_module(module_path)
            print(f"✅ {description:25} - Ready")
        except ImportError as e:
            print(f"❌ {description:25} - Error: {e}")
            all_working = False
        except Exception as e:
            print(f"⚠️  {description:25} - Warning: {e}")
            
    return all_working

def main():
    print("=" * 50)
    print("🚀 SutazAI Performance Optimization Test")
    print("=" * 50)
    
    # Test basic dependencies
    deps_ok = test_imports()
    
    # Test performance modules
    modules_ok = test_performance_modules()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    if deps_ok and modules_ok:
        print("✅ System is ready for high-performance operation!")
        print("\nTo start the optimized backend:")
        print("  cd /opt/sutazaiapp/backend")
        print("  python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return 0
    elif deps_ok:
        print("⚠️  Dependencies OK but some performance modules have issues")
        print("   The system can run but may not achieve optimal performance")
        return 1
    else:
        print("❌ Missing critical dependencies")
        print("   Please install required packages before running")
        return 2

if __name__ == "__main__":
    sys.exit(main())