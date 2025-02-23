#!/usr/bin/env python3
"""
SutazAI Comprehensive Documentation Management System

Advanced framework for:
- Automatic documentation discovery
- Centralized documentation management
- Version tracking
- Cross-referencing
- Automated documentation generation
"""

import importlib
import inspect
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List

import markdown
import pdoc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='/opt/sutazai_project/SutazAI/logs/documentation_manager.log'
)
logger = logging.getLogger('SutazAI.DocumentationManager')

class DocumentationManager:
    """
    Comprehensive documentation management system for SutazAI
    
    Provides advanced capabilities for:
    - Automatic documentation discovery
    - Centralized documentation organization
    - Version tracking
    - Cross-referencing
    - Automated documentation generation
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        docs_dir: str = '/opt/sutazai_project/SutazAI/docs'
    ):
        """
        Initialize documentation management system
        
        Args:
            base_dir (str): Base project directory
            docs_dir (str): Centralized documentation directory
        """
        self.base_dir = base_dir
        self.docs_dir = docs_dir
        
        # Ensure docs directory exists
        os.makedirs(self.docs_dir, exist_ok=True)
        
        # Documentation tracking
        self.documentation_index: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'documents': {},
            'categories': {}
        }
    
    def discover_documentation(self) -> Dict[str, List[str]]:
        """
        Discover documentation across the project
        
        Returns:
            Dictionary of documentation files by category
        """
        documentation_map = {
            'markdown': [],
            'python_docstrings': [],
            'system_docs': [],
            'architecture_docs': [],
            'security_docs': []
        }
        
        # Markdown documentation discovery
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.md'):
                    full_path = os.path.join(root, file)
                    
                    # Categorize markdown files
                    if 'architecture' in file.lower():
                        documentation_map['architecture_docs'].append(full_path)
                    elif 'security' in file.lower():
                        documentation_map['security_docs'].append(full_path)
                    elif 'system' in file.lower():
                        documentation_map['system_docs'].append(full_path)
                    else:
                        documentation_map['markdown'].append(full_path)
        
        # Python docstring discovery
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    try:
                        module_name = os.path.relpath(full_path, self.base_dir).replace('/', '.')[:-3]
                        module = importlib.import_module(module_name)
                        
                        # Check for docstrings
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) or inspect.isfunction(obj):
                                if obj.__module__ == module_name and obj.__doc__:
                                    documentation_map['python_docstrings'].append(full_path)
                                    break
                    except Exception as e:
                        logger.warning(f"Could not process module {full_path}: {e}")
        
        return documentation_map
    
    def generate_documentation_index(self, documentation_map: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Generate a comprehensive documentation index
        
        Args:
            documentation_map (Dict): Discovered documentation files
        
        Returns:
            Comprehensive documentation index
        """
        index = {
            'total_documents': 0,
            'categories': {}
        }
        
        for category, files in documentation_map.items():
            index['categories'][category] = {
                'count': len(files),
                'files': [os.path.basename(f) for f in files]
            }
            index['total_documents'] += len(files)
        
        return index
    
    def centralize_documentation(self, documentation_map: Dict[str, List[str]]):
        """
        Centralize documentation in the docs directory
        
        Args:
            documentation_map (Dict): Discovered documentation files
        """
        for category, files in documentation_map.items():
            category_dir = os.path.join(self.docs_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            for file_path in files:
                try:
                    # Copy file to centralized docs directory
                    shutil.copy2(file_path, category_dir)
                    logger.info(f"Copied {file_path} to {category_dir}")
                except Exception as e:
                    logger.error(f"Failed to copy {file_path}: {e}")
    
    def generate_markdown_documentation(self, documentation_map: Dict[str, List[str]]):
        """
        Generate consolidated markdown documentation
        
        Args:
            documentation_map (Dict): Discovered documentation files
        """
        consolidated_docs = {}
        
        for category, files in documentation_map.items():
            consolidated_content = []
            
            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        consolidated_content.append(f"## {os.path.basename(file_path)}\n\n{content}\n\n")
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
            
            if consolidated_content:
                consolidated_docs[category] = '\n'.join(consolidated_content)
        
        # Write consolidated documentation
        for category, content in consolidated_docs.items():
            output_path = os.path.join(self.docs_dir, f'{category}_consolidated.md')
            with open(output_path, 'w') as f:
                f.write(content)
            logger.info(f"Generated consolidated documentation: {output_path}")
    
    def generate_html_documentation(self, documentation_map: Dict[str, List[str]]):
        """
        Generate HTML documentation using pdoc
        
        Args:
            documentation_map (Dict): Discovered documentation files
        """
        html_dir = os.path.join(self.docs_dir, 'html')
        os.makedirs(html_dir, exist_ok=True)
        
        # Generate Python module documentation
        python_modules = [
            os.path.relpath(f, self.base_dir).replace('/', '.')[:-3] 
            for f in documentation_map.get('python_docstrings', [])
        ]
        
        try:
            pdoc.render.whole_modules(
                modules=python_modules, 
                output_directory=html_dir
            )
            logger.info(f"Generated HTML documentation in {html_dir}")
        except Exception as e:
            logger.error(f"HTML documentation generation failed: {e}")
    
    def run_documentation_management(self):
        """
        Execute comprehensive documentation management workflow
        """
        try:
            # Discover documentation
            documentation_map = self.discover_documentation()
            
            # Generate documentation index
            documentation_index = self.generate_documentation_index(documentation_map)
            
            # Centralize documentation
            self.centralize_documentation(documentation_map)
            
            # Generate consolidated markdown
            self.generate_markdown_documentation(documentation_map)
            
            # Generate HTML documentation
            self.generate_html_documentation(documentation_map)
            
            # Persist documentation index
            index_path = os.path.join(self.docs_dir, 'documentation_index.json')
            with open(index_path, 'w') as f:
                json.dump(documentation_index, f, indent=2)
            
            logger.info("Documentation management completed successfully")
        
        except Exception as e:
            logger.error(f"Documentation management failed: {e}")

def main():
    """
    Main execution for documentation management
    """
    try:
        doc_manager = DocumentationManager()
        doc_manager.run_documentation_management()
    
    except Exception as e:
        print(f"Documentation management failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()