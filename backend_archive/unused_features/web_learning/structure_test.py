#!/usr/bin/env python3
"""
Structure Test for SutazAI V7 Self-Supervised Learning Pipeline
Tests code structure and class definitions without external dependencies
"""

import os
import sys
import importlib.util
import inspect
from pathlib import Path

def test_file_structure():
    """Test that all required files exist"""
    print("Testing file structure...")
    
    current_dir = Path(__file__).parent
    required_files = [
        "__init__.py",
        "web_scraper.py",
        "content_processor.py", 
        "knowledge_extractor.py",
        "learning_pipeline.py",
        "web_automation.py",
        "test_learning_pipeline.py",
        "simple_test.py",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not (current_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files exist")
    return True

def test_module_structure(module_name, expected_classes, expected_functions):
    """Test module structure"""
    print(f"\nTesting {module_name} structure...")
    
    try:
        # Load module
        spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        
        # Check if module can be loaded (syntax check)
        try:
            spec.loader.exec_module(module)
            print(f"✓ {module_name} syntax is valid")
        except Exception as e:
            print(f"✗ {module_name} syntax error: {e}")
            return False
        
        # Check for expected classes
        for class_name in expected_classes:
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                if inspect.isclass(cls):
                    print(f"✓ Class {class_name} found")
                else:
                    print(f"✗ {class_name} is not a class")
                    return False
            else:
                print(f"✗ Class {class_name} not found")
                return False
        
        # Check for expected functions
        for func_name in expected_functions:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if inspect.isfunction(func):
                    print(f"✓ Function {func_name} found")
                else:
                    print(f"✗ {func_name} is not a function")
                    return False
            else:
                print(f"✗ Function {func_name} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing {module_name}: {e}")
        return False

def test_class_methods(module_name, class_name, expected_methods):
    """Test class methods"""
    print(f"\nTesting {class_name} methods...")
    
    try:
        # Load module
        spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get class
        if not hasattr(module, class_name):
            print(f"✗ Class {class_name} not found")
            return False
        
        cls = getattr(module, class_name)
        
        # Check methods
        for method_name in expected_methods:
            if hasattr(cls, method_name):
                method = getattr(cls, method_name)
                if callable(method):
                    print(f"✓ Method {method_name} found")
                else:
                    print(f"✗ {method_name} is not callable")
                    return False
            else:
                print(f"✗ Method {method_name} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing {class_name}: {e}")
        return False

def test_web_scraper_structure():
    """Test web scraper structure"""
    expected_classes = [
        "WebScraper",
        "ScrapingConfig", 
        "RobotsChecker",
        "RateLimiter",
        "ContentValidator"
    ]
    
    expected_functions = [
        "create_web_scraper"
    ]
    
    result = test_module_structure("web_scraper", expected_classes, expected_functions)
    
    if result:
        # Test WebScraper methods
        scraper_methods = [
            "__init__",
            "start_session",
            "close_session",
            "fetch_url",
            "scrape_urls",
            "crawl_website",
            "extract_links",
            "get_statistics"
        ]
        
        result = test_class_methods("web_scraper", "WebScraper", scraper_methods)
    
    return result

def test_content_processor_structure():
    """Test content processor structure"""
    expected_classes = [
        "ContentProcessor",
        "ProcessedContent",
        "ContentType",
        "HTMLProcessor",
        "TextAnalyzer"
    ]
    
    expected_functions = [
        "create_content_processor"
    ]
    
    result = test_module_structure("content_processor", expected_classes, expected_functions)
    
    if result:
        # Test ContentProcessor methods
        processor_methods = [
            "__init__",
            "detect_content_type",
            "process_content",
            "get_statistics"
        ]
        
        result = test_class_methods("content_processor", "ContentProcessor", processor_methods)
    
    return result

def test_knowledge_extractor_structure():
    """Test knowledge extractor structure"""
    expected_classes = [
        "KnowledgeExtractor",
        "KnowledgeUnit",
        "KnowledgeType",
        "KnowledgeGraph",
        "FactualExtractor",
        "ConceptualExtractor",
        "RelationalExtractor",
        "NumericalExtractor"
    ]
    
    expected_functions = [
        "create_knowledge_extractor"
    ]
    
    result = test_module_structure("knowledge_extractor", expected_classes, expected_functions)
    
    if result:
        # Test KnowledgeExtractor methods
        extractor_methods = [
            "__init__",
            "initialize_neural_engine",
            "extract_knowledge",
            "search_knowledge",
            "get_statistics"
        ]
        
        result = test_class_methods("knowledge_extractor", "KnowledgeExtractor", extractor_methods)
    
    return result

def test_learning_pipeline_structure():
    """Test learning pipeline structure"""
    expected_classes = [
        "SelfSupervisedLearningPipeline",
        "LearningConfig",
        "LearningTask",
        "LearningSession",
        "LearningMode",
        "TopicDiscoverer",
        "KnowledgeValidator"
    ]
    
    expected_functions = [
        "create_learning_pipeline"
    ]
    
    result = test_module_structure("learning_pipeline", expected_classes, expected_functions)
    
    if result:
        # Test SelfSupervisedLearningPipeline methods
        pipeline_methods = [
            "__init__",
            "initialize",
            "start_learning_session",
            "get_overall_statistics",
            "shutdown"
        ]
        
        result = test_class_methods("learning_pipeline", "SelfSupervisedLearningPipeline", pipeline_methods)
    
    return result

def test_web_automation_structure():
    """Test web automation structure"""
    expected_classes = [
        "WebAutomation",
        "AutomationConfig",
        "BrowserManager"
    ]
    
    expected_functions = [
        "create_web_automation"
    ]
    
    result = test_module_structure("web_automation", expected_classes, expected_functions)
    
    if result:
        # Test WebAutomation methods
        automation_methods = [
            "__init__",
            "initialize",
            "automate_page_interaction",
            "extract_dynamic_content",
            "get_statistics",
            "shutdown"
        ]
        
        result = test_class_methods("web_automation", "WebAutomation", automation_methods)
    
    return result

def test_init_file():
    """Test __init__.py file"""
    print("\nTesting __init__.py...")
    
    try:
        with open("__init__.py", "r") as f:
            content = f.read()
        
        # Check for imports
        required_imports = [
            "WebScraper",
            "ScrapingConfig",
            "ContentProcessor",
            "ContentType",
            "KnowledgeExtractor",
            "SelfSupervisedLearningPipeline",
            "WebAutomation"
        ]
        
        for import_name in required_imports:
            if import_name in content:
                print(f"✓ {import_name} imported in __init__.py")
            else:
                print(f"✗ {import_name} not imported in __init__.py")
                return False
        
        # Check for __all__ definition
        if "__all__" in content:
            print("✓ __all__ defined in __init__.py")
        else:
            print("✗ __all__ not defined in __init__.py")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing __init__.py: {e}")
        return False

def test_docstrings():
    """Test that modules have proper docstrings"""
    print("\nTesting docstrings...")
    
    modules = [
        "web_scraper.py",
        "content_processor.py",
        "knowledge_extractor.py",
        "learning_pipeline.py",
        "web_automation.py"
    ]
    
    for module_file in modules:
        try:
            with open(module_file, "r") as f:
                content = f.read()
            
            # Check for module docstring
            if '"""' in content and content.strip().startswith('#!/usr/bin/env python3'):
                lines = content.split('\n')
                docstring_found = False
                for i, line in enumerate(lines):
                    if line.strip().startswith('"""') and i < 10:  # Docstring should be near the top
                        docstring_found = True
                        break
                
                if docstring_found:
                    print(f"✓ {module_file} has docstring")
                else:
                    print(f"✗ {module_file} missing docstring")
                    return False
            else:
                print(f"✗ {module_file} missing docstring")
                return False
                
        except Exception as e:
            print(f"✗ Error checking {module_file}: {e}")
            return False
    
    return True

def test_readme():
    """Test README.md file"""
    print("\nTesting README.md...")
    
    try:
        with open("README.md", "r") as f:
            content = f.read()
        
        # Check for required sections
        required_sections = [
            "# SutazAI V7 Self-Supervised Learning Pipeline",
            "## Overview",
            "## Architecture", 
            "## Components",
            "## Installation",
            "## Usage",
            "## Configuration",
            "## Testing",
            "## Performance",
            "## Security"
        ]
        
        for section in required_sections:
            if section in content:
                print(f"✓ README contains {section}")
            else:
                print(f"✗ README missing {section}")
                return False
        
        # Check minimum length
        if len(content) > 10000:  # At least 10KB of documentation
            print("✓ README has comprehensive documentation")
        else:
            print("✗ README documentation is too brief")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing README.md: {e}")
        return False

def main():
    """Run all structure tests"""
    print("SutazAI V7 Self-Supervised Learning Pipeline - Structure Tests")
    print("=" * 70)
    
    # Change to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tests = [
        ("File Structure", test_file_structure),
        ("Web Scraper Structure", test_web_scraper_structure),
        ("Content Processor Structure", test_content_processor_structure),
        ("Knowledge Extractor Structure", test_knowledge_extractor_structure),
        ("Learning Pipeline Structure", test_learning_pipeline_structure),
        ("Web Automation Structure", test_web_automation_structure),
        ("Init File", test_init_file),
        ("Docstrings", test_docstrings),
        ("README Documentation", test_readme)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("STRUCTURE TEST SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All structure tests PASSED!")
        print("✓ Self-supervised learning pipeline has correct structure")
        print("✓ All required classes and methods are present")
        print("✓ Code follows proper Python conventions")
        print("✓ Documentation is comprehensive")
    else:
        print(f"\n❌ {failed} structure tests FAILED")
        print("Please check the error messages above")
    
    print("\nStructure validated for:")
    print("• File organization and naming")
    print("• Class and method definitions")
    print("• Module imports and exports")
    print("• Documentation and docstrings")
    print("• README completeness")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)