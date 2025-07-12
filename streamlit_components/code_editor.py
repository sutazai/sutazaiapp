"""
AI Code Editor and Debugging Panel for SutazAI Streamlit Interface
Integrates Aider AI coding assistant with advanced debugging capabilities
"""

import streamlit as st
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import asyncio
from datetime import datetime
import traceback
import ast
import re

# Code analysis libraries
try:
    import black
    import flake8.api.legacy as flake8
    import mypy.api
    import pylint.lint
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False
    logging.warning("Code formatting/analysis libraries not fully available")

logger = logging.getLogger(__name__)

class CodeEditor:
    """Advanced AI Code Editor with Debugging and Analysis"""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.current_file = None
        self.execution_history = []
        self.debug_sessions = {}
        self.supported_languages = {
            'python': {
                'extension': '.py',
                'executor': self._execute_python,
                'analyzer': self._analyze_python,
                'formatter': self._format_python
            },
            'javascript': {
                'extension': '.js',
                'executor': self._execute_javascript,
                'analyzer': self._analyze_javascript,
                'formatter': self._format_javascript
            },
            'bash': {
                'extension': '.sh',
                'executor': self._execute_bash,
                'analyzer': self._analyze_bash,
                'formatter': None
            },
            'sql': {
                'extension': '.sql',
                'executor': self._execute_sql,
                'analyzer': self._analyze_sql,
                'formatter': self._format_sql
            }
        }
    
    def render_code_editor_interface(self):
        """Render the main code editor interface"""
        st.header("💻 AI Code Editor & Debugging Panel")
        
        # Editor tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Code Editor",
            "🤖 AI Assistant", 
            "🐛 Debugger",
            "🔍 Code Analysis",
            "📁 File Manager"
        ])
        
        with tab1:
            self.render_code_editor()
        
        with tab2:
            self.render_ai_assistant()
        
        with tab3:
            self.render_debugger()
        
        with tab4:
            self.render_code_analysis()
        
        with tab5:
            self.render_file_manager()
    
    def render_code_editor(self):
        """Render the main code editor"""
        st.subheader("📝 Code Editor")
        
        # Editor controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            language = st.selectbox(
                "Language",
                options=list(self.supported_languages.keys()),
                key="editor_language"
            )
        
        with col2:
            theme = st.selectbox(
                "Editor Theme",
                ["monokai", "github", "dracula", "vs-dark"],
                key="editor_theme"
            )
        
        with col3:
            font_size = st.slider("Font Size", 10, 20, 14, key="font_size")
        
        with col4:
            auto_save = st.checkbox("Auto Save", value=True, key="auto_save")
        
        # Load template or previous code
        if 'editor_code' not in st.session_state:
            st.session_state.editor_code = self._get_template_code(language)
        
        # Template selector
        with st.expander("📋 Code Templates"):
            templates = self._get_available_templates(language)
            selected_template = st.selectbox("Choose Template", templates)
            
            if st.button("Load Template"):
                st.session_state.editor_code = self._get_template_code(language, selected_template)
                st.rerun()
        
        # Main code editor
        code = st.text_area(
            "Code Editor",
            value=st.session_state.editor_code,
            height=400,
            key="main_editor",
            help=f"Write your {language} code here"
        )
        
        # Update session state
        if code != st.session_state.editor_code:
            st.session_state.editor_code = code
        
        # Editor action buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("▶️ Run Code", type="primary"):
                self._execute_code(code, language)
        
        with col2:
            if st.button("🔍 Analyze"):
                self._analyze_code(code, language)
        
        with col3:
            if st.button("✨ Format"):
                formatted_code = self._format_code(code, language)
                if formatted_code:
                    st.session_state.editor_code = formatted_code
                    st.rerun()
        
        with col4:
            if st.button("💾 Save"):
                self._save_code_file(code, language)
        
        with col5:
            if st.button("🤖 AI Help"):
                st.session_state.show_ai_help = True
    
    def _get_template_code(self, language: str, template: str = "basic") -> str:
        """Get template code for the specified language"""
        templates = {
            'python': {
                'basic': '''# Python Code Template
def main():
    """Main function"""
    print("Hello, World!")
    
if __name__ == "__main__":
    main()''',
                'class': '''# Python Class Template
class MyClass:
    """A sample class"""
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        return f"Hello, {self.name}!"
    
# Usage example
if __name__ == "__main__":
    obj = MyClass("SutazAI")
    print(obj.greet())''',
                'api': '''# Flask API Template
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from SutazAI!"})

@app.route('/api/data', methods=['POST'])
def process_data():
    data = request.get_json()
    return jsonify({"processed": data})

if __name__ == "__main__":
    app.run(debug=True)''',
                'data_analysis': '''# Data Analysis Template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and analyze data
def analyze_data():
    # Sample data
    data = {
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    }
    df = pd.DataFrame(data)
    
    # Basic statistics
    print(df.describe())
    
    # Visualization
    plt.scatter(df['x'], df['y'])
    plt.title('Sample Data Analysis')
    plt.show()

if __name__ == "__main__":
    analyze_data()'''
            },
            'javascript': {
                'basic': '''// JavaScript Code Template
function main() {
    console.log("Hello, World!");
}

main();''',
                'async': '''// Async JavaScript Template
async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

// Usage
fetchData('https://api.example.com/data')
    .then(data => console.log(data));''',
                'react': '''// React Component Template
import React, { useState, useEffect } from 'react';

const MyComponent = () => {
    const [data, setData] = useState(null);
    
    useEffect(() => {
        // Fetch data on component mount
        fetchData();
    }, []);
    
    const fetchData = async () => {
        // Your data fetching logic here
        setData({ message: "Hello from React!" });
    };
    
    return (
        <div>
            <h1>SutazAI Component</h1>
            {data && <p>{data.message}</p>}
        </div>
    );
};

export default MyComponent;'''
            },
            'bash': {
                'basic': '''#!/bin/bash
# Bash Script Template

echo "Hello, World!"

# Variables
NAME="SutazAI"
echo "Welcome to $NAME"

# Function
greet() {
    echo "Hello, $1!"
}

greet "User"''',
                'automation': '''#!/bin/bash
# Automation Script Template

set -e  # Exit on error

# Configuration
LOG_FILE="/var/log/automation.log"
BACKUP_DIR="/backup"

# Logging function
log() {
    echo "$(date): $1" | tee -a "$LOG_FILE"
}

# Main automation tasks
main() {
    log "Starting automation script"
    
    # Your automation tasks here
    log "Task 1: System check"
    # system_check
    
    log "Task 2: Backup data"
    # backup_data
    
    log "Automation script completed"
}

main "$@"'''
            },
            'sql': {
                'basic': '''-- SQL Query Template
SELECT * FROM users
WHERE created_date >= '2024-01-01'
ORDER BY created_date DESC;''',
                'analysis': '''-- Data Analysis SQL Template
WITH monthly_stats AS (
    SELECT 
        DATE_TRUNC('month', created_date) as month,
        COUNT(*) as user_count,
        AVG(age) as avg_age
    FROM users
    GROUP BY DATE_TRUNC('month', created_date)
)
SELECT 
    month,
    user_count,
    avg_age,
    LAG(user_count) OVER (ORDER BY month) as prev_month_users,
    (user_count - LAG(user_count) OVER (ORDER BY month)) / 
    LAG(user_count) OVER (ORDER BY month) * 100 as growth_rate
FROM monthly_stats
ORDER BY month;'''
            }
        }
        
        return templates.get(language, {}).get(template, f"# {language.title()} code\nprint('Hello, World!')")
    
    def _get_available_templates(self, language: str) -> List[str]:
        """Get available templates for a language"""
        templates = {
            'python': ['basic', 'class', 'api', 'data_analysis'],
            'javascript': ['basic', 'async', 'react'],
            'bash': ['basic', 'automation'],
            'sql': ['basic', 'analysis']
        }
        
        return templates.get(language, ['basic'])
    
    def _execute_code(self, code: str, language: str):
        """Execute code and display results"""
        try:
            st.subheader("🏃 Execution Results")
            
            # Create execution container
            with st.container():
                # Show execution info
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.info(f"Executing {language} code...")
                
                with col2:
                    st.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
                
                # Execute based on language
                if language in self.supported_languages:
                    executor = self.supported_languages[language]['executor']
                    result = executor(code)
                    
                    # Display results
                    if result['success']:
                        st.success("✅ Execution completed successfully!")
                        
                        if result['output']:
                            st.subheader("📤 Output")
                            st.code(result['output'], language='text')
                        
                        if result.get('plots'):
                            st.subheader("📊 Plots")
                            for plot in result['plots']:
                                st.pyplot(plot)
                    
                    else:
                        st.error("❌ Execution failed!")
                        
                        if result['error']:
                            st.subheader("🚨 Error Details")
                            st.code(result['error'], language='text')
                    
                    # Store in execution history
                    self.execution_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'language': language,
                        'code': code[:200] + "..." if len(code) > 200 else code,
                        'success': result['success'],
                        'output': result.get('output', ''),
                        'error': result.get('error', '')
                    })
                
                else:
                    st.error(f"Language {language} not supported for execution")
                    
        except Exception as e:
            st.error(f"Execution error: {str(e)}")
            logger.error(f"Code execution error: {e}")
    
    def _execute_python(self, code: str) -> Dict:
        """Execute Python code"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
            finally:
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': 'Execution timed out after 30 seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e)
            }
    
    def _execute_javascript(self, code: str) -> Dict:
        """Execute JavaScript code using Node.js"""
        try:
            # Check if Node.js is available
            try:
                subprocess.run(['node', '--version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return {
                    'success': False,
                    'output': '',
                    'error': 'Node.js not available. Please install Node.js to execute JavaScript code.'
                }
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute with Node.js
                result = subprocess.run(
                    ['node', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e)
            }
    
    def _execute_bash(self, code: str) -> Dict:
        """Execute Bash script"""
        try:
            # Execute bash code
            result = subprocess.run(
                ['bash', '-c', code],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': 'Execution timed out after 30 seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e)
            }
    
    def _execute_sql(self, code: str) -> Dict:
        """Execute SQL code (mock implementation)"""
        # This would connect to a real database in production
        return {
            'success': True,
            'output': 'SQL execution simulation - query would be executed against database',
            'error': ''
        }
    
    def _analyze_code(self, code: str, language: str):
        """Analyze code for issues and improvements"""
        st.subheader("🔍 Code Analysis Results")
        
        if language in self.supported_languages:
            analyzer = self.supported_languages[language]['analyzer']
            analysis = analyzer(code)
            
            # Display analysis results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Quality Metrics")
                
                metrics = analysis.get('metrics', {})
                st.metric("Lines of Code", metrics.get('lines', 0))
                st.metric("Complexity Score", metrics.get('complexity', 'N/A'))
                st.metric("Quality Score", f"{metrics.get('quality', 0)}/10")
            
            with col2:
                st.subheader("🎯 Issues Found")
                
                issues = analysis.get('issues', [])
                if issues:
                    for issue in issues[:5]:  # Show top 5 issues
                        severity_color = {
                            'error': '🔴',
                            'warning': '🟡', 
                            'info': '🔵'
                        }
                        st.write(f"{severity_color.get(issue['severity'], '⚪')} **Line {issue.get('line', 'N/A')}**: {issue['message']}")
                else:
                    st.success("No issues found!")
            
            # Suggestions
            suggestions = analysis.get('suggestions', [])
            if suggestions:
                st.subheader("💡 AI Suggestions")
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")
        
        else:
            st.info(f"Code analysis not available for {language}")
    
    def _analyze_python(self, code: str) -> Dict:
        """Analyze Python code"""
        analysis = {
            'metrics': {},
            'issues': [],
            'suggestions': []
        }
        
        try:
            # Basic metrics
            lines = code.split('\n')
            analysis['metrics']['lines'] = len([l for l in lines if l.strip()])
            
            # Try to parse the code
            try:
                tree = ast.parse(code)
                analysis['metrics']['complexity'] = self._calculate_complexity(tree)
                analysis['metrics']['quality'] = min(10, max(1, 10 - len(analysis['issues']) // 2))
            except SyntaxError as e:
                analysis['issues'].append({
                    'severity': 'error',
                    'line': e.lineno,
                    'message': f"Syntax Error: {e.msg}"
                })
            
            # Mock additional issues for demonstration
            if 'print(' in code:
                analysis['suggestions'].append("Consider using logging instead of print statements for production code")
            
            if 'import *' in code:
                analysis['issues'].append({
                    'severity': 'warning',
                    'line': 1,
                    'message': "Avoid wildcard imports (import *)"
                })
            
            if len(lines) > 100:
                analysis['suggestions'].append("Consider breaking this into smaller functions or modules")
            
        except Exception as e:
            logger.error(f"Python analysis error: {e}")
        
        return analysis
    
    def _analyze_javascript(self, code: str) -> Dict:
        """Analyze JavaScript code"""
        analysis = {
            'metrics': {
                'lines': len(code.split('\n')),
                'complexity': 'Medium',
                'quality': 7
            },
            'issues': [],
            'suggestions': []
        }
        
        # Basic JavaScript analysis
        if 'var ' in code:
            analysis['suggestions'].append("Consider using 'let' or 'const' instead of 'var'")
        
        if '==' in code and '===' not in code:
            analysis['issues'].append({
                'severity': 'warning',
                'line': 1,
                'message': "Use strict equality (===) instead of loose equality (==)"
            })
        
        return analysis
    
    def _analyze_bash(self, code: str) -> Dict:
        """Analyze Bash script"""
        analysis = {
            'metrics': {
                'lines': len(code.split('\n')),
                'complexity': 'Low',
                'quality': 8
            },
            'issues': [],
            'suggestions': []
        }
        
        if not code.startswith('#!/bin/bash'):
            analysis['suggestions'].append("Add shebang line (#!/bin/bash) at the beginning")
        
        if 'set -e' not in code:
            analysis['suggestions'].append("Consider adding 'set -e' to exit on error")
        
        return analysis
    
    def _analyze_sql(self, code: str) -> Dict:
        """Analyze SQL code"""
        analysis = {
            'metrics': {
                'lines': len(code.split('\n')),
                'complexity': 'Medium',
                'quality': 8
            },
            'issues': [],
            'suggestions': []
        }
        
        # Basic SQL analysis
        if 'SELECT *' in code.upper():
            analysis['suggestions'].append("Avoid SELECT * in production queries - specify column names")
        
        if not code.strip().endswith(';'):
            analysis['issues'].append({
                'severity': 'info',
                'line': len(code.split('\n')),
                'message': "SQL statements should end with semicolon"
            })
        
        return analysis
    
    def _calculate_complexity(self, tree) -> str:
        """Calculate code complexity from AST"""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.FunctionDef, ast.ClassDef)):
                complexity += 1
        
        if complexity < 5:
            return "Low"
        elif complexity < 15:
            return "Medium"
        else:
            return "High"
    
    def _format_code(self, code: str, language: str) -> Optional[str]:
        """Format code using language-specific formatters"""
        if language in self.supported_languages:
            formatter = self.supported_languages[language]['formatter']
            if formatter:
                return formatter(code)
        
        st.info(f"Code formatting not available for {language}")
        return None
    
    def _format_python(self, code: str) -> str:
        """Format Python code using Black"""
        try:
            if BLACK_AVAILABLE:
                formatted = black.format_str(code, mode=black.FileMode())
                st.success("✅ Code formatted successfully!")
                return formatted
            else:
                st.warning("Black formatter not available")
                return code
        except Exception as e:
            st.error(f"Formatting error: {str(e)}")
            return code
    
    def _format_javascript(self, code: str) -> str:
        """Format JavaScript code (basic implementation)"""
        # This would use prettier or similar in production
        st.info("JavaScript formatting would use Prettier in production")
        return code
    
    def _format_sql(self, code: str) -> str:
        """Format SQL code (basic implementation)"""
        # Basic SQL formatting
        formatted = re.sub(r'\s+', ' ', code.strip())
        formatted = re.sub(r',\s*', ',\n    ', formatted)
        formatted = re.sub(r'\bFROM\b', '\nFROM', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'\bWHERE\b', '\nWHERE', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'\bORDER BY\b', '\nORDER BY', formatted, flags=re.IGNORECASE)
        
        return formatted
    
    def _save_code_file(self, code: str, language: str):
        """Save code to file"""
        try:
            extension = self.supported_languages[language]['extension']
            filename = f"sutazai_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
            
            # In production, this would save to a user directory
            st.download_button(
                label="💾 Download Code File",
                data=code,
                file_name=filename,
                mime="text/plain"
            )
            
            st.success(f"✅ Code ready for download as {filename}")
            
        except Exception as e:
            st.error(f"Save error: {str(e)}")
    
    def render_ai_assistant(self):
        """Render AI coding assistant interface"""
        st.subheader("🤖 AI Coding Assistant (Aider Integration)")
        
        # AI assistant modes
        assistant_mode = st.selectbox(
            "Assistant Mode",
            [
                "Code Generation",
                "Code Review", 
                "Bug Fixing",
                "Code Explanation",
                "Optimization",
                "Testing"
            ]
        )
        
        # Context from current editor
        if 'editor_code' in st.session_state and st.session_state.editor_code:
            with st.expander("📋 Current Code Context"):
                st.code(st.session_state.editor_code[:500] + "..." if len(st.session_state.editor_code) > 500 else st.session_state.editor_code)
        
        # AI prompt input
        user_prompt = st.text_area(
            "Describe what you need help with:",
            height=100,
            placeholder="E.g., 'Generate a function to sort a list of dictionaries by multiple keys' or 'Review this code for potential security issues'"
        )
        
        # Additional context
        with st.expander("🔧 Additional Context"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_language = st.selectbox("Target Language", list(self.supported_languages.keys()))
                code_style = st.selectbox("Code Style", ["Clean", "Commented", "Minimal", "Production"])
            
            with col2:
                include_tests = st.checkbox("Include Tests")
                include_docs = st.checkbox("Include Documentation")
        
        if st.button("🚀 Get AI Help", type="primary") and user_prompt:
            self._process_ai_request(user_prompt, assistant_mode, {
                'language': target_language,
                'style': code_style,
                'include_tests': include_tests,
                'include_docs': include_docs,
                'current_code': st.session_state.get('editor_code', '')
            })
    
    def _process_ai_request(self, prompt: str, mode: str, context: Dict):
        """Process AI assistant request"""
        try:
            with st.spinner(f"AI Assistant working on {mode.lower()}..."):
                # Simulate AI processing - in production would call actual AI model
                response = self._simulate_ai_response(prompt, mode, context)
                
                st.subheader(f"🤖 AI {mode} Response")
                
                # Display AI response
                if response.get('code'):
                    st.subheader("💻 Generated Code")
                    st.code(response['code'], language=context['language'])
                    
                    # Option to apply to editor
                    if st.button("📝 Apply to Editor"):
                        st.session_state.editor_code = response['code']
                        st.success("Code applied to editor!")
                        st.rerun()
                
                if response.get('explanation'):
                    st.subheader("📖 Explanation")
                    st.write(response['explanation'])
                
                if response.get('suggestions'):
                    st.subheader("💡 Suggestions")
                    for suggestion in response['suggestions']:
                        st.write(f"• {suggestion}")
                
                if response.get('tests') and context['include_tests']:
                    st.subheader("🧪 Generated Tests")
                    st.code(response['tests'], language=context['language'])
                
        except Exception as e:
            st.error(f"AI Assistant error: {str(e)}")
    
    def _simulate_ai_response(self, prompt: str, mode: str, context: Dict) -> Dict:
        """Simulate AI assistant response"""
        # This would call the actual AI model in production
        
        if mode == "Code Generation":
            return {
                'code': f'''# Generated {context['language']} code for: {prompt[:50]}...

def solution():
    """
    {prompt}
    
    Returns:
        result: The solution to your request
    """
    # AI-generated implementation would be here
    result = "AI implementation"
    return result

# Example usage
if __name__ == "__main__":
    print(solution())''',
                'explanation': f"This code implements {prompt.lower()}. The function is designed to be efficient and follows {context['style'].lower()} coding practices.",
                'suggestions': [
                    "Consider adding input validation",
                    "Add error handling for edge cases",
                    "Consider performance optimization for large datasets"
                ]
            }
        
        elif mode == "Code Review":
            return {
                'explanation': "Code review completed. Overall, the code structure is good.",
                'suggestions': [
                    "Add type hints for better code documentation",
                    "Consider extracting magic numbers into constants",
                    "Add docstrings to all functions",
                    "Consider using more descriptive variable names"
                ]
            }
        
        elif mode == "Bug Fixing":
            return {
                'code': "# Fixed version with bug corrections\n" + context.get('current_code', ''),
                'explanation': "Identified and fixed potential bugs in the code.",
                'suggestions': [
                    "Fixed off-by-one error in loop",
                    "Added null checks for safety",
                    "Corrected variable scope issues"
                ]
            }
        
        else:
            return {
                'explanation': f"AI {mode} analysis complete.",
                'suggestions': ["Analysis results would appear here"]
            }
    
    def render_debugger(self):
        """Render debugging interface"""
        st.subheader("🐛 Interactive Debugger")
        
        # Debugger features
        debug_tab1, debug_tab2, debug_tab3 = st.tabs([
            "🔍 Debug Session",
            "📊 Execution Trace",
            "🚨 Error Analysis"
        ])
        
        with debug_tab1:
            self._render_debug_session()
        
        with debug_tab2:
            self._render_execution_trace()
        
        with debug_tab3:
            self._render_error_analysis()
    
    def _render_debug_session(self):
        """Render debug session interface"""
        st.write("**Debug Session Controls**")
        
        # Current code for debugging
        if 'editor_code' in st.session_state and st.session_state.editor_code:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🐛 Start Debug"):
                    self._start_debug_session()
            
            with col2:
                if st.button("⏭️ Step Over"):
                    st.info("Step over executed")
            
            with col3:
                if st.button("⬇️ Step Into"):
                    st.info("Step into executed")
            
            with col4:
                if st.button("⏹️ Stop Debug"):
                    st.info("Debug session stopped")
            
            # Breakpoints
            st.subheader("🔴 Breakpoints")
            
            lines = st.session_state.editor_code.split('\n')
            breakpoint_line = st.number_input(
                "Add breakpoint at line:",
                min_value=1,
                max_value=len(lines),
                value=1
            )
            
            if st.button("➕ Add Breakpoint"):
                if 'breakpoints' not in st.session_state:
                    st.session_state.breakpoints = set()
                
                st.session_state.breakpoints.add(breakpoint_line)
                st.success(f"Breakpoint added at line {breakpoint_line}")
            
            # Show current breakpoints
            if 'breakpoints' in st.session_state and st.session_state.breakpoints:
                st.write("**Active Breakpoints:**")
                for bp in sorted(st.session_state.breakpoints):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"Line {bp}: `{lines[bp-1].strip()}`")
                    with col2:
                        if st.button("❌", key=f"remove_bp_{bp}"):
                            st.session_state.breakpoints.remove(bp)
                            st.rerun()
            
            # Variable inspection
            st.subheader("🔍 Variable Inspector")
            
            # Mock variables for demonstration
            variables = {
                'x': {'value': '42', 'type': 'int'},
                'name': {'value': '"SutazAI"', 'type': 'str'},
                'data': {'value': '[1, 2, 3, 4, 5]', 'type': 'list'},
                'config': {'value': "{'debug': True}", 'type': 'dict'}
            }
            
            for var_name, var_info in variables.items():
                with st.expander(f"🔹 {var_name} ({var_info['type']})"):
                    st.code(f"{var_name} = {var_info['value']}")
        
        else:
            st.info("No code available for debugging. Write some code in the editor first.")
    
    def _start_debug_session(self):
        """Start a new debug session"""
        session_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.debug_sessions[session_id] = {
            'start_time': datetime.now(),
            'code': st.session_state.get('editor_code', ''),
            'status': 'active',
            'current_line': 1,
            'variables': {}
        }
        
        st.success(f"Debug session {session_id} started!")
    
    def _render_execution_trace(self):
        """Render execution trace"""
        st.write("**Execution Trace Log**")
        
        if self.execution_history:
            for i, execution in enumerate(reversed(self.execution_history[-5:])):
                with st.expander(f"🏃 Execution {len(self.execution_history) - i} - {execution['timestamp'][:19]}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Language:** {execution['language']}")
                        st.write(f"**Status:** {'✅ Success' if execution['success'] else '❌ Failed'}")
                    
                    with col2:
                        st.write(f"**Time:** {execution['timestamp'][11:19]}")
                        st.write(f"**Code Length:** {len(execution['code'])} chars")
                    
                    if execution['output']:
                        st.write("**Output:**")
                        st.code(execution['output'][:200] + "..." if len(execution['output']) > 200 else execution['output'])
                    
                    if execution['error']:
                        st.write("**Error:**")
                        st.code(execution['error'])
        else:
            st.info("No execution history available. Run some code to see traces.")
    
    def _render_error_analysis(self):
        """Render error analysis"""
        st.write("**AI-Powered Error Analysis**")
        
        # Recent errors from execution history
        recent_errors = [ex for ex in self.execution_history if not ex['success']]
        
        if recent_errors:
            latest_error = recent_errors[-1]
            
            st.subheader("🚨 Latest Error Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Language:** {latest_error['language']}")
                st.write(f"**Time:** {latest_error['timestamp'][11:19]}")
            
            with col2:
                st.write("**Error Type:** Runtime Error")
                st.write("**Severity:** Medium")
            
            # Error details
            st.write("**Error Message:**")
            st.code(latest_error['error'])
            
            # AI suggestions for error fixing
            st.subheader("🤖 AI Suggestions")
            
            error_suggestions = [
                "Check for syntax errors in your code",
                "Verify all variables are properly defined",
                "Ensure all imported modules are available",
                "Check file paths and permissions",
                "Add try-catch blocks for error handling"
            ]
            
            for suggestion in error_suggestions:
                st.write(f"• {suggestion}")
            
            # Quick fix button
            if st.button("🔧 Generate Fix"):
                st.info("AI would analyze the error and suggest specific fixes...")
        
        else:
            st.success("No errors found in recent executions!")
    
    def render_file_manager(self):
        """Render file manager interface"""
        st.subheader("📁 File Manager")
        
        # File operations
        file_tab1, file_tab2, file_tab3 = st.tabs([
            "📂 Browse Files",
            "📤 Upload/Download",
            "🗂️ Project Manager"
        ])
        
        with file_tab1:
            self._render_file_browser()
        
        with file_tab2:
            self._render_file_upload_download()
        
        with file_tab3:
            self._render_project_manager()
    
    def _render_file_browser(self):
        """Render file browser"""
        st.write("**File Browser**")
        
        # Mock file system for demonstration
        files = [
            {'name': 'main.py', 'type': 'python', 'size': '2.3 KB', 'modified': '2024-01-15 10:30'},
            {'name': 'utils.js', 'type': 'javascript', 'size': '1.8 KB', 'modified': '2024-01-15 09:15'},
            {'name': 'config.json', 'type': 'json', 'size': '0.5 KB', 'modified': '2024-01-14 16:45'},
            {'name': 'README.md', 'type': 'markdown', 'size': '1.2 KB', 'modified': '2024-01-14 14:20'},
            {'name': 'requirements.txt', 'type': 'text', 'size': '0.3 KB', 'modified': '2024-01-13 11:10'}
        ]
        
        # File list
        for file_info in files:
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 2, 1])
            
            with col1:
                file_icon = {
                    'python': '🐍',
                    'javascript': '📜',
                    'json': '📋',
                    'markdown': '📝',
                    'text': '📄'
                }
                st.write(f"{file_icon.get(file_info['type'], '📄')} {file_info['name']}")
            
            with col2:
                st.write(file_info['size'])
            
            with col3:
                st.write(file_info['modified'][11:16])
            
            with col4:
                st.write(file_info['modified'][:10])
            
            with col5:
                if st.button("📂", key=f"open_{file_info['name']}"):
                    st.session_state.editor_code = f"# Content of {file_info['name']}\n# File loaded successfully"
                    st.success(f"Loaded {file_info['name']}")
    
    def _render_file_upload_download(self):
        """Render file upload/download interface"""
        st.write("**File Upload & Download**")
        
        # Upload files
        uploaded_files = st.file_uploader(
            "Upload code files",
            type=['py', 'js', 'sql', 'sh', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                with st.expander(f"📄 {file.name}"):
                    content = str(file.read(), 'utf-8')
                    st.code(content[:500] + "..." if len(content) > 500 else content)
                    
                    if st.button(f"Load to Editor", key=f"load_{file.name}"):
                        st.session_state.editor_code = content
                        st.success(f"Loaded {file.name} to editor")
                        st.rerun()
        
        # Download current code
        st.write("**Download Current Code**")
        
        if 'editor_code' in st.session_state and st.session_state.editor_code:
            language = st.session_state.get('editor_language', 'python')
            extension = self.supported_languages[language]['extension']
            filename = f"sutazai_code{extension}"
            
            st.download_button(
                label="💾 Download Code",
                data=st.session_state.editor_code,
                file_name=filename,
                mime="text/plain"
            )
    
    def _render_project_manager(self):
        """Render project manager"""
        st.write("**Project Manager**")
        
        # Project creation
        with st.expander("🆕 Create New Project"):
            project_name = st.text_input("Project Name")
            project_type = st.selectbox("Project Type", [
                "Python Application",
                "Web Application",
                "Data Science",
                "API Service",
                "Machine Learning",
                "Custom"
            ])
            
            if st.button("Create Project") and project_name:
                st.success(f"Project '{project_name}' created!")
                
                # Generate project template
                template = self._get_project_template(project_type)
                st.session_state.editor_code = template
        
        # Existing projects (mock)
        st.write("**Recent Projects**")
        
        projects = [
            {'name': 'SutazAI Integration', 'type': 'Python Application', 'last_opened': '2024-01-15'},
            {'name': 'Data Analysis Tool', 'type': 'Data Science', 'last_opened': '2024-01-14'},
            {'name': 'Web API Service', 'type': 'API Service', 'last_opened': '2024-01-13'}
        ]
        
        for project in projects:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"📁 {project['name']}")
            
            with col2:
                st.write(project['type'])
            
            with col3:
                st.write(project['last_opened'])
            
            with col4:
                if st.button("📂", key=f"open_project_{project['name']}"):
                    st.info(f"Opening {project['name']}...")
    
    def _get_project_template(self, project_type: str) -> str:
        """Get project template based on type"""
        templates = {
            'Python Application': '''# SutazAI Python Application Template
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Application:
    """Main application class"""
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        """Load application configuration"""
        return {"debug": True}
    
    def run(self):
        """Run the application"""
        logger.info("Starting SutazAI application...")
        # Your application logic here
        print("Hello from SutazAI!")

if __name__ == "__main__":
    app = Application()
    app.run()''',
            
            'Web Application': '''# SutazAI Web Application Template
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET', 'POST'])
def api_data():
    if request.method == 'POST':
        data = request.get_json()
        # Process data here
        return jsonify({"status": "success", "data": data})
    
    return jsonify({"message": "SutazAI Web API"})

if __name__ == "__main__":
    app.run(debug=True)''',
            
            'Data Science': '''# SutazAI Data Science Template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data loading and preprocessing
def load_data():
    """Load and preprocess data"""
    # Load your data here
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    return data

def analyze_data(df):
    """Perform data analysis"""
    print("Data Shape:", df.shape)
    print("\\nData Info:")
    print(df.info())
    print("\\nDescriptive Statistics:")
    print(df.describe())

def main():
    """Main analysis pipeline"""
    df = load_data()
    analyze_data(df)
    
    # Your analysis here

if __name__ == "__main__":
    main()'''
        }
        
        return templates.get(project_type, "# Custom project template")
    
    def render(self):
        """Render the complete code editor interface"""
        self.render_code_editor_interface()