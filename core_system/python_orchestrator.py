"""
SutazAI Python Orchestrator
This module orchestrates and manages Python-related operations in the SutazAI system.
"""

import importlib
import inspect
import os
import sys
from pathlib import Path


class PythonOrchestrator:
    """Manages and orchestrates Python modules, imports, and execution"""
    
    def __init__(self):
        self.loaded_modules = {}
        self.execution_history = []
        self.base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    
    def import_module(self, module_name):
        """Dynamically import a module and keep track of it"""
        try:
            if module_name in self.loaded_modules:
                return self.loaded_modules[module_name]
            
            module = importlib.import_module(module_name)
            self.loaded_modules[module_name] = module
            return module
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")
            return None
    
    def execute_function(self, module_name, function_name, *args, **kwargs):
        """Execute a function from a dynamically imported module"""
        module = self.import_module(module_name)
        if not module:
            return None
        
        if not hasattr(module, function_name):
            print(f"Function {function_name} not found in module {module_name}")
            return None
        
        function = getattr(module, function_name)
        result = function(*args, **kwargs)
        
        # Record execution for tracking
        self.execution_history.append({
            'module': module_name,
            'function': function_name,
            'args': args,
            'kwargs': kwargs,
            'timestamp': importlib.import_module('datetime').datetime.now()
        })
        
        return result
    
    def get_function_signature(self, module_name, function_name):
        """Get the signature of a function from a module"""
        module = self.import_module(module_name)
        if not module or not hasattr(module, function_name):
            return None
        
        function = getattr(module, function_name)
        return inspect.signature(function)
    
    def list_available_modules(self):
        """List all available Python modules in the system"""
        return list(self.loaded_modules.keys())
