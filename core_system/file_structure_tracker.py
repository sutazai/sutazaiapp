#!/usr/bin/env python3
"""
SutazAI Ultra-Comprehensive Project Structure Tracker

Provides an advanced, multi-dimensional, and hardcoded 
project structure documentation system.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml


class UltraComprehensiveStructureTracker:
    """
    Ultra-Comprehensive Project Structure Tracking System
    
    Features:
    - Hardcoded project structure template
    - Comprehensive metadata generation
    - Multiple output formats
    - Automated documentation
    """
    
    # Hardcoded Project Structure Template
    PROJECT_STRUCTURE_TEMPLATE = {
        "ai_agents": {
            "supreme_ai": "Supreme AI (non-root orchestrator)",
            "auto_gpt": "Autonomous GPT Agent",
            "superagi": "SuperAGI Agent Framework",
            "langchain_agents": "LangChain-based Agents",
            "tabbyml": "TabbyML Integration",
            "semgrep": "Code Analysis Agents",
            "gpt_engineer": "GPT-based Code Generation",
            "aider": "AI Collaborative Coding Assistant"
        },
        "model_management": {
            "GPT4All": "Open-source Language Model",
            "DeepSeek-R1": "Research-grade Language Model",
            "DeepSeek-Coder": "Code Generation Model",
            "Llama2": "Meta's Language Model",
            "Molmo": "Diagram Recognition Model"
        },
        "backend": {
            "main.py": "Application Entry Point",
            "api_routes.py": "API Endpoint Definitions",
            "services": "Business Logic Implementations",
            "config": "Backend Configuration",
            "tests": "Backend Test Suite"
        },
        "web_ui": {
            "package.json": "Node.js Dependencies",
            "node_modules": "Installed NPM Packages",
            "src": "Frontend Source Code",
            "public": "Static Assets",
            "build_or_dist": "Compiled Frontend"
        },
        "scripts": {
            "deploy.sh": "Main Online Deployment Script",
            "setup_repos.sh": "Manual Repository Synchronization",
            "test_pipeline.py": "Comprehensive Testing Pipeline"
        },
        "packages": {
            "wheels": "Pinned Python Wheel Packages",
            "node": "Cached Node.js Modules"
        },
        "logs": {
            "deploy.log": "Deployment Logs",
            "pipeline.log": "CI/CD Pipeline Logs", 
            "online_calls.log": "External API Call Logs"
        },
        "doc_data": {
            "pdfs": "PDF Document Storage",
            "diagrams": "Project Diagrams and Visualizations"
        },
        "root_files": [
            "requirements.txt",
            "venv",
            "README.md"
        ]
    }
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        log_dir: str = '/opt/sutazai_project/SutazAI/logs'
    ):
        """
        Initialize Ultra-Comprehensive Structure Tracker
        
        Args:
            base_dir (str): Base project directory
            log_dir (str): Logging directory
        """
        self.base_dir = base_dir
        self.log_dir = log_dir
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=os.path.join(log_dir, 'structure_tracker.log')
        )
        self.logger = logging.getLogger('SutazAI.UltraStructureTracker')
    
    def generate_comprehensive_structure(self) -> Dict[str, Any]:
        """
        Generate comprehensive project structure with real-time metadata
        
        Returns:
            Detailed project structure dictionary
        """
        structure = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "base_directory": self.base_dir,
                "total_components": 0,
                "total_files": 0,
                "total_directories": 0
            },
            "structure": self.PROJECT_STRUCTURE_TEMPLATE
        }
        
        # Validate and enrich structure with real-time data
        self._validate_and_enrich_structure(structure)
        
        return structure
    
    def _validate_and_enrich_structure(self, structure: Dict[str, Any]):
        """
        Validate and enrich project structure with real-time metadata
        
        Args:
            structure (Dict): Project structure dictionary to enrich
        """
        for category, contents in structure['structure'].items():
            # Handle both dictionary and list contents
            if isinstance(contents, dict):
                for component, description in list(contents.items()):
                    full_path = os.path.join(self.base_dir, category, component)
                    
                    if os.path.exists(full_path):
                        structure['metadata']['total_components'] += 1
                        
                        if os.path.isdir(full_path):
                            structure['metadata']['total_directories'] += 1
                            contents[component] = {
                                "description": description,
                                "path": full_path,
                                "contents": os.listdir(full_path)
                            }
                        else:
                            structure['metadata']['total_files'] += 1
                            contents[component] = {
                                "description": description,
                                "path": full_path,
                                "size": os.path.getsize(full_path)
                            }
            elif isinstance(contents, list):
                # Handle root_files list
                for component in contents:
                    full_path = os.path.join(self.base_dir, component)
                    
                    if os.path.exists(full_path):
                        structure['metadata']['total_components'] += 1
                        
                        if os.path.isdir(full_path):
                            structure['metadata']['total_directories'] += 1
                        else:
                            structure['metadata']['total_files'] += 1
    
    def generate_markdown_structure(self) -> str:
        """
        Generate a comprehensive markdown representation of project structure
        
        Returns:
            Markdown-formatted project structure
        """
        structure = self.generate_comprehensive_structure()
        
        markdown_lines = [
            "# 🌐 SutazAI Project Structure\n",
            f"## Generated: {structure['metadata']['generated_at']}\n",
            f"### Project Overview",
            f"- **Total Components**: {structure['metadata']['total_components']}",
            f"- **Total Directories**: {structure['metadata']['total_directories']}",
            f"- **Total Files**: {structure['metadata']['total_files']}\n",
            "## Detailed Structure\n"
        ]
        
        def _format_structure(contents: Any, indent: int = 0) -> List[str]:
            """
            Recursively format structure into markdown
            
            Args:
                contents (Any): Structure contents to format
                indent (int): Current indentation level
            
            Returns:
                List of markdown lines
            """
            lines = []
            
            # Handle dictionary contents
            if isinstance(contents, dict):
                for name, details in contents.items():
                    if isinstance(details, dict):
                        # Directory or file with additional details
                        description = details.get('description', '')
                        lines.append(f"{'  ' * indent}- 📁 **{name}/** {description}")
                        
                        # Recursively add contents if it's a directory
                        if 'contents' in details:
                            for item in details['contents']:
                                lines.append(f"{'  ' * (indent + 1)}- 📄 {item}")
                    else:
                        # Simple description
                        lines.append(f"{'  ' * indent}- 📄 **{name}**: {details}")
            
            # Handle list contents
            elif isinstance(contents, list):
                for item in contents:
                    lines.append(f"{'  ' * indent}- 📄 {item}")
            
            return lines
        
        # Generate markdown for each category
        for category, contents in structure['structure'].items():
            markdown_lines.append(f"### {category.upper().replace('_', ' ')}")
            markdown_lines.extend(_format_structure(contents))
            markdown_lines.append("")
        
        return "\n".join(markdown_lines)
    
    def update_structure_files(self):
        """
        Update comprehensive structure documentation in multiple formats
        """
        # JSON Structure
        json_path = os.path.join(self.base_dir, 'DIRECTORY_STRUCTURE.json')
        with open(json_path, 'w') as f:
            json.dump(self.generate_comprehensive_structure(), f, indent=2)
        
        # Markdown Structure
        md_path = os.path.join(self.base_dir, 'DIRECTORY_STRUCTURE.md')
        with open(md_path, 'w') as f:
            f.write(self.generate_markdown_structure())
        
        # YAML Structure (for additional flexibility)
        yaml_path = os.path.join(self.base_dir, 'DIRECTORY_STRUCTURE.yml')
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(self.generate_comprehensive_structure(), f, indent=2)
        
        self.logger.info("Project structure documentation updated successfully")
    
    def update_readme_structure(self):
        """
        Update README.md with comprehensive project structure
        """
        try:
            readme_path = os.path.join(self.base_dir, 'README.md')
            
            # Read existing README
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            # Generate markdown structure
            structure_markdown = self.generate_markdown_structure()
            
            # Replace structure section
            import re
            updated_content = re.sub(
                r'## 🏗️ Project Structure\n.*?(?=\n## |\Z)', 
                f"## 🏗️ Project Structure\n\n{structure_markdown}", 
                readme_content, 
                flags=re.DOTALL
            )
            
            # Write updated README
            with open(readme_path, 'w') as f:
                f.write(updated_content)
            
            self.logger.info("README project structure updated successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to update README structure: {e}")

def main():
    """
    Main execution for ultra-comprehensive structure tracking
    """
    # Configure logging at the start of the script
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s: %(message)s',
        filename='/opt/sutazai_project/SutazAI/logs/structure_tracker_debug.log',
        filemode='a'
    )
    logger = logging.getLogger('SutazAI.StructureTrackerMain')
    
    try:
        # Detailed logging for debugging
        logger.info("=" * 50)
        logger.info("Starting Project Structure Tracking")
        logger.info("=" * 50)
        
        # Print system and environment details
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Current User: {os.getlogin()}")
        logger.info(f"Current Working Directory: {os.getcwd()}")
        
        # Ensure base directory exists
        base_dir = '/opt/sutazai_project/SutazAI'
        if not os.path.exists(base_dir):
            logger.critical(f"Base directory does not exist: {base_dir}")
            print(f"Critical Error: Base directory {base_dir} does not exist")
            sys.exit(1)
        
        # Change to base directory
        os.chdir(base_dir)
        
        # Create tracker instance
        tracker = UltraComprehensiveStructureTracker(base_dir)
        
        # Update structure files with detailed logging
        logger.info("Starting structure files update")
        try:
            tracker.update_structure_files()
            logger.info("Structure files updated successfully")
        except Exception as file_update_error:
            logger.error(f"Structure files update failed: {file_update_error}", exc_info=True)
        
        # Update README with detailed logging
        logger.info("Starting README structure update")
        try:
            tracker.update_readme_structure()
            logger.info("README structure updated successfully")
        except Exception as readme_update_error:
            logger.error(f"README structure update failed: {readme_update_error}", exc_info=True)
        
        logger.info("Project structure documentation update completed")
        print("Project structure documentation updated comprehensively.")
    
    except Exception as e:
        # Comprehensive error logging
        logger.critical(f"Unexpected error in structure tracking: {e}", exc_info=True)
        print(f"Critical Error: {e}")
        
        # Additional system diagnostics
        import traceback
        logger.critical("Full Traceback:")
        logger.critical(traceback.format_exc())
        
        sys.exit(1)

if __name__ == '__main__':
    main() 