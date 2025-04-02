#!/usr/bin/env python3
import os
import shutil

def clean_pycache(directory):
    """Recursively remove all __pycache__ directories and .pyc/.pyo files"""
    for root, dirs, files in os.walk(directory):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            print(f"Removing: {pycache_path}")
            shutil.rmtree(pycache_path)
            dirs.remove('__pycache__')  # Don't recurse into deleted directory
        
        # Remove .pyc and .pyo files
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                pyc_file = os.path.join(root, file)
                print(f"Removing: {pyc_file}")
                os.unlink(pyc_file)

if __name__ == "__main__":
    project_dir = "/opt/sutazaiapp"
    clean_pycache(project_dir)
    print("Python cache cleanup complete!") 