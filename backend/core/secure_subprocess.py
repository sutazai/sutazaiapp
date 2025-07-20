#!/usr/bin/env python3
"""
Secure Subprocess Management for SutazAI
Provides secure alternatives to shell=True subprocess calls
"""

import subprocess
import shlex
import logging
from typing import List, Union, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class SecureSubprocess:
    """Secure subprocess execution utility"""
    
    # Allowlist of safe executables
    SAFE_EXECUTABLES = {
        'docker', 'docker-compose', 'git', 'pip', 'python', 'python3',
        'grep', 'find', 'ls', 'cat', 'head', 'tail', 'wc', 'sort',
        'openssl', 'curl', 'wget', 'systemctl', 'journalctl',
        'apt-get', 'brew', 'yum', 'dnf', 'pacman',
        'ollama', 'streamlit', 'uvicorn', 'gunicorn'
    }
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = {
        '&&', '||', ';', '|', '>', '<', '`', '$(',
        'rm -rf', 'sudo rm', 'chmod 777', 'chmod +x',
        'eval', 'exec', '/bin/sh', '/bin/bash', 'bash -c', 'sh -c'
    }
    
    @staticmethod
    def is_safe_command(command: Union[str, List[str]]) -> bool:
        """Check if command is safe to execute"""
        if isinstance(command, str):
            # Check for dangerous patterns
            for pattern in SecureSubprocess.DANGEROUS_PATTERNS:
                if pattern in command.lower():
                    logger.warning(f"Dangerous pattern detected: {pattern}")
                    return False
            
            # Parse command
            try:
                command_list = shlex.split(command)
            except ValueError:
                logger.warning("Invalid command syntax")
                return False
        else:
            command_list = command
        
        if not command_list:
            return False
        
        # Check if executable is in allowlist
        executable = Path(command_list[0]).name
        if executable not in SecureSubprocess.SAFE_EXECUTABLES:
            logger.warning(f"Executable not in allowlist: {executable}")
            return False
        
        return True
    
    @staticmethod
    def run_secure(
        command: Union[str, List[str]],
        cwd: str = None,
        timeout: int = 30,
        capture_output: bool = True,
        check: bool = False,
        env: Dict[str, str] = None
    ) -> subprocess.CompletedProcess:
        """Run command securely without shell=True"""
        
        # Convert string to list if needed
        if isinstance(command, str):
            command_list = shlex.split(command)
        else:
            command_list = command.copy()
        
        # Security validation
        if not SecureSubprocess.is_safe_command(command_list):
            raise SecurityError(f"Command blocked for security: {command_list[0]}")
        
        logger.info(f"Executing secure command: {' '.join(command_list)}")
        
        try:
            result = subprocess.run(
                command_list,
                cwd=cwd,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                check=check,
                env=env
            )
            
            logger.info(f"Command completed with return code: {result.returncode}")
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with return code {e.returncode}")
            if capture_output:
                logger.error(f"STDERR: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing command: {e}")
            raise
    
    @staticmethod
    def run_docker_command(
        docker_args: List[str],
        timeout: int = 60,
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run Docker command securely"""
        command = ['docker'] + docker_args
        return SecureSubprocess.run_secure(
            command,
            timeout=timeout,
            capture_output=capture_output
        )
    
    @staticmethod
    def run_docker_compose_command(
        compose_args: List[str],
        timeout: int = 120,
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run docker-compose command securely"""
        command = ['docker', 'compose'] + compose_args
        return SecureSubprocess.run_secure(
            command,
            timeout=timeout,
            capture_output=capture_output
        )
    
    @staticmethod
    def run_git_command(
        git_args: List[str],
        cwd: str = None,
        timeout: int = 30,
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run git command securely"""
        command = ['git'] + git_args
        return SecureSubprocess.run_secure(
            command,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output
        )
    
    @staticmethod
    def run_python_command(
        python_args: List[str],
        timeout: int = 60,
        capture_output: bool = True,
        python_executable: str = 'python3'
    ) -> subprocess.CompletedProcess:
        """Run Python command securely"""
        command = [python_executable] + python_args
        return SecureSubprocess.run_secure(
            command,
            timeout=timeout,
            capture_output=capture_output
        )

class SecurityError(Exception):
    """Raised when a security violation is detected"""
    pass

def validate_path_safety(path: Union[str, Path]) -> bool:
    """Validate that a path is safe to use"""
    path = Path(path).resolve()
    
    # Block access to sensitive directories
    dangerous_paths = {
        Path('/etc'),
        Path('/proc'),
        Path('/sys'),
        Path('/dev'),
        Path('/root'),
        Path('/boot')
    }
    
    for dangerous in dangerous_paths:
        try:
            if path.is_relative_to(dangerous):
                return False
        except ValueError:
            continue
    
    return True

# Legacy compatibility functions with security warnings
def run_command_secure(command: Union[str, List[str]], **kwargs) -> Tuple[bool, str, str]:
    """Legacy function - prefer SecureSubprocess.run_secure()"""
    logger.warning("Using legacy run_command_secure - consider upgrading to SecureSubprocess")
    
    try:
        result = SecureSubprocess.run_secure(
            command,
            capture_output=True,
            **kwargs
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)