#!/usr/bin/env python3
import os
import re


    def fix_fstring_errors(file_path):
    """
    Fix f-string formatting errors in Python files.
    Replaces 'ff"' and 'fff"' with 'f"'.
    """
        try:
        with open(file_path) as f:
        content = f.read()
        
        # Replace 'ff"' and 'fff"' with 'f"'
        pattern = (
        r"(logger\.|self\.logger\.)"
        r'(info|error|warning|debug|exception)\(f{2,3}"'
    )
    modified_content = re.sub(
    pattern,
    r'\1\2(f"',
    content,
)

    if modified_content != content:
    with open(file_path, "w") as f:
    f.write(modified_content)
    print(f"Fixed f-string errors in {file_path}")
    return True
    return False
    except Exception as e:
    print(f"Error processing {file_path}: {e}")
    return False
    
    
        def process_directory(directory):
        """
        Process all Python files in a directory and its subdirectories.
        """
        fixed_files = 0
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                        if fix_fstring_errors(file_path):
                        fixed_files += 1
                        
                        print(f"Total files fixed: {fixed_files}")
                        
                        
                            def main():
                            project_root = "/opt/sutazaiapp"
                            process_directory(project_root)
                            
                            
                                if __name__ == "__main__":
                                main()
                                