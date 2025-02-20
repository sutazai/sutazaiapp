#!/usr/bin/env python3
"""
SutazAI Advanced Project Indexing and Cross-Referencing System

Comprehensive framework for:
- Autonomous file and directory indexing
- Semantic cross-referencing
- Dependency mapping
- Intelligent linking between components
"""

import os
import sys
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import multiprocessing
import threading
import time

import networkx as nx
import yaml

class AdvancedProjectIndexer:
    """
    Ultra-Comprehensive Project Indexing and Cross-Referencing System
    
    Provides intelligent, autonomous indexing of project components
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        index_dir: str = '/opt/sutazai_project/SutazAI/logs/indexes'
    ):
        """
        Initialize advanced project indexer
        
        Args:
            base_dir (str): Base project directory
            index_dir (str): Directory for storing indexes
        """
        self.base_dir = base_dir
        self.index_dir = index_dir
        
        # Ensure index directory exists
        os.makedirs(index_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=os.path.join(index_dir, 'project_indexer.log')
        )
        self.logger = logging.getLogger('SutazAI.AdvancedProjectIndexer')
    
    def generate_file_index(self) -> Dict[str, Any]:
        """
        Generate a comprehensive index of all project files
        
        Returns:
            Dictionary of file metadata and cross-references
        """
        file_index = {
            'timestamp': datetime.now().isoformat(),
            'files': {},
            'file_types': {},
            'directory_structure': {}
        }
        
        # Traverse project directories
        for root, dirs, files in os.walk(self.base_dir):
            # Track directory structure
            relative_path = os.path.relpath(root, self.base_dir)
            file_index['directory_structure'][relative_path] = {
                'subdirectories': dirs,
                'files': files
            }
            
            # Index individual files
            for file in files:
                full_path = os.path.join(root, file)
                
                try:
                    # Generate file metadata
                    file_metadata = self._generate_file_metadata(full_path)
                    
                    # Store file metadata
                    file_index['files'][full_path] = file_metadata
                    
                    # Track file types
                    file_ext = os.path.splitext(file)[1]
                    file_index['file_types'][file_ext] = file_index['file_types'].get(file_ext, 0) + 1
                
                except Exception as e:
                    self.logger.warning(f"Could not index file {full_path}: {e}")
        
        # Persist index
        index_path = os.path.join(
            self.index_dir, 
            f'file_index_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(index_path, 'w') as f:
            json.dump(file_index, f, indent=2)
        
        self.logger.info(f"Comprehensive file index generated: {index_path}")
        
        return file_index
    
    def _generate_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Generate detailed metadata for a single file
        
        Args:
            file_path (str): Full path to the file
        
        Returns:
            Dictionary of file metadata
        """
        # Basic file metadata
        stat = os.stat(file_path)
        
        # Generate file hash for unique identification
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        metadata = {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1],
            'size_bytes': stat.st_size,
            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'hash': file_hash,
            'relative_path': os.path.relpath(file_path, self.base_dir)
        }
        
        # Additional metadata for specific file types
        if file_path.endswith('.py'):
            metadata.update(self._analyze_python_file(file_path))
        
        return metadata
    
    def _analyze_python_file(self, file_path: str) -> Dict[str, Any]:
        """
        Perform detailed analysis of Python files
        
        Args:
            file_path (str): Path to Python file
        
        Returns:
            Dictionary of Python-specific metadata
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Basic code analysis
            lines = content.split('\n')
            
            # Count classes, functions, imports
            import_count = sum(1 for line in lines if line.startswith(('import ', 'from ')))
            class_count = sum(1 for line in lines if line.startswith('class '))
            function_count = sum(1 for line in lines if line.startswith('def '))
            
            return {
                'python_metadata': {
                    'total_lines': len(lines),
                    'import_count': import_count,
                    'class_count': class_count,
                    'function_count': function_count
                }
            }
        
        except Exception as e:
            self.logger.warning(f"Python file analysis failed for {file_path}: {e}")
            return {}
    
    def generate_cross_reference_graph(self, file_index: Dict[str, Any]) -> nx.DiGraph:
        """
        Generate a cross-reference graph between project files
        
        Args:
            file_index (Dict): Comprehensive file index
        
        Returns:
            NetworkX Directed Graph of file cross-references
        """
        cross_reference_graph = nx.DiGraph()
        
        # Add nodes for all files
        for file_path in file_index['files']:
            cross_reference_graph.add_node(file_path)
        
        # Identify cross-references (placeholder - can be expanded)
        for file_path, metadata in file_index['files'].items():
            if metadata.get('python_metadata'):
                # Example: Look for import statements (would require more sophisticated parsing)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Simple import detection (can be made more robust)
                    for other_file_path in file_index['files']:
                        module_name = os.path.splitext(os.path.basename(other_file_path))[0]
                        if module_name in content:
                            cross_reference_graph.add_edge(file_path, other_file_path)
                
                except Exception as e:
                    self.logger.warning(f"Cross-reference generation failed for {file_path}: {e}")
        
        return cross_reference_graph
    
    def run_autonomous_indexing(self):
        """
        Execute comprehensive autonomous project indexing workflow
        """
        try:
            # Generate comprehensive file index
            file_index = self.generate_file_index()
            
            # Generate cross-reference graph
            cross_reference_graph = self.generate_cross_reference_graph(file_index)
            
            # Persist cross-reference graph
            graph_path = os.path.join(
                self.index_dir, 
                f'cross_reference_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(graph_path, 'w') as f:
                json.dump({
                    'nodes': list(cross_reference_graph.nodes()),
                    'edges': list(cross_reference_graph.edges())
                }, f, indent=2)
            
            self.logger.info("Autonomous project indexing completed successfully")
        
        except Exception as e:
            self.logger.error(f"Autonomous project indexing failed: {e}")

    def start_periodic_indexing(self, interval_hours=24):
        """
        Start periodic background indexing
        
        Args:
            interval_hours (int): Hours between each indexing run. Defaults to 24 (daily)
        """
        def periodic_indexing():
            while True:
                try:
                    self.logger.info(f"Starting periodic project indexing")
                    self.run_autonomous_indexing()
                    
                    # Sleep for the specified interval
                    time.sleep(interval_hours * 3600)
                
                except Exception as e:
                    self.logger.error(f"Periodic indexing failed: {e}")
                    # Wait a bit before retrying to prevent rapid error loops
                    time.sleep(3600)  # Wait 1 hour before retrying
        
        # Start the periodic indexing in a daemon thread
        indexing_thread = threading.Thread(target=periodic_indexing, daemon=True)
        indexing_thread.start()
        self.logger.info("Periodic background indexing started")

def run_indexing_process():
    """
    Wrapper function to run indexing in a separate process
    """
    try:
        indexer = AdvancedProjectIndexer()
        indexer.run_autonomous_indexing()
    except Exception as e:
        print(f"Background indexing process failed: {e}")
        sys.exit(1)

def main():
    """
    Main execution for advanced project indexing
    Run indexing in the background
    """
    try:
        # Create indexer instance
        indexer = AdvancedProjectIndexer()
        
        # Start periodic background indexing (every 24 hours)
        indexer.start_periodic_indexing()
        
        # Optional: Keep the main process running
        # This prevents the script from exiting immediately
        while True:
            time.sleep(3600)  # Sleep for an hour
    
    except Exception as e:
        print(f"Failed to start background indexing process: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()