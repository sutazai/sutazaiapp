#!/usr/bin/env python3
import os
import sys
import ast
import logging
import chardet
from typing import List, Dict, Optional, Tuple

class FileRepairTool:
    def __init__(self, base_path: str = '/media/ai/SutazAI_Storage/SutazAI/v1/consolidated'):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler('/var/log/sutazai/file_repair.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.base_path = base_path
        self.encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']
        
        # Repair statistics
        self.repair_stats = {
            'total_files': 0,
            'parsed_files': 0,
            'repaired_files': 0,
            'failed_files': []
        }
    
    def detect_encoding(self, file_path: str) -> Optional[str]:
        """
        Detect the encoding of a file using chardet.
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            Optional[str]: Detected encoding or None
        """
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                return result['encoding']
        except Exception as e:
            self.logger.error(f"Encoding detection failed for {file_path}: {e}")
            return None
    
    def parse_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Attempt to parse a Python file with multiple encoding strategies.
        
        Args:
            file_path (str): Path to the Python file
        
        Returns:
            Tuple[bool, Optional[str]]: (Parsed successfully, Repaired content)
        """
        self.repair_stats['total_files'] += 1
        
        # Try detecting encoding first
        detected_encoding = self.detect_encoding(file_path)
        
        # Prioritize detected encoding
        encodings_to_try = [detected_encoding] if detected_encoding else []
        encodings_to_try.extend([enc for enc in self.encodings if enc != detected_encoding])
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                
                # Attempt to parse the content
                ast.parse(content)
                
                self.repair_stats['parsed_files'] += 1
                return True, None
            
            except (SyntaxError, UnicodeDecodeError) as e:
                # Attempt basic syntax repair
                try:
                    repaired_content = self._repair_syntax(content)
                    
                    # Verify repaired content
                    ast.parse(repaired_content)
                    
                    # Save repaired file
                    self._save_repaired_file(file_path, repaired_content)
                    
                    self.logger.info(f"Repaired file: {file_path}")
                    self.repair_stats['repaired_files'] += 1
                    return True, repaired_content
                
                except Exception as repair_error:
                    self.logger.warning(f"Could not repair {file_path}: {repair_error}")
                    continue
        
        # If all attempts fail
        self.repair_stats['failed_files'].append(file_path)
        self.logger.error(f"Failed to parse file: {file_path}")
        return False, None
    
    def _repair_syntax(self, content: str) -> str:
        """
        Attempt to repair basic syntax issues.
        
        Args:
            content (str): File content
        
        Returns:
            str: Repaired content
        """
        # Remove invalid escape sequences
        content = content.replace('\\(', '(').replace('\\)', ')')
        
        # Add missing parentheses
        content = content.replace('{)', '}').replace('(]', '[')
        
        # Fix print statements for Python 3
        content = content.replace('print ', 'print(')
        
        # Remove any remaining problematic escape sequences
        content = content.encode('unicode_escape').decode()
        
        return content
    
    def _save_repaired_file(self, original_path: str, repaired_content: str):
        """
        Save the repaired file, creating a backup of the original.
        
        Args:
            original_path (str): Path to the original file
            repaired_content (str): Repaired file content
        """
        backup_path = f"{original_path}.bak"
        os.rename(original_path, backup_path)
        
        with open(original_path, 'w', encoding='utf-8') as file:
            file.write(repaired_content)
    
    def repair_files(self, directory: Optional[str] = None) -> Dict:
        """
        Recursively repair Python files in a directory.
        
        Args:
            directory (Optional[str]): Directory to repair. Uses base_path if None.
        
        Returns:
            Dict: Repair statistics
        """
        if not directory:
            directory = self.base_path
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.parse_file(file_path)
        
        return self.repair_stats
    
    def generate_report(self):
        """Generate a comprehensive repair report."""
        report_path = '/var/log/sutazai/file_repair_report.json'
        
        report = {
            'total_files': self.repair_stats['total_files'],
            'parsed_files': self.repair_stats['parsed_files'],
            'repaired_files': self.repair_stats['repaired_files'],
            'failed_files': self.repair_stats['failed_files']
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path

def main():
    repair_tool = FileRepairTool()
    
    # Repair files
    repair_stats = repair_tool.repair_files()
    
    # Generate report
    report_path = repair_tool.generate_report()
    
    print("File Repair Complete.")
    print(f"Total Files: {repair_stats['total_files']}")
    print(f"Parsed Files: {repair_stats['parsed_files']}")
    print(f"Repaired Files: {repair_stats['repaired_files']}")
    print(f"Failed Files: {len(repair_stats['failed_files'])}")
    print(f"Detailed report saved to: {report_path}")

if __name__ == '__main__':
    main()