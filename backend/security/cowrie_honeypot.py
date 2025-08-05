"""
Cowrie SSH Honeypot Integration for SutazAI System
Advanced SSH honeypot for detecting brute force attacks and capturing attacker behavior
"""

import asyncio
import logging
import json
import subprocess
import os
import shutil
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import tarfile
import requests

# Import honeypot infrastructure
from security.honeypot_infrastructure import BaseHoneypot, HoneypotType, honeypot_orchestrator

logger = logging.getLogger(__name__)

class CowrieManager:
    """Manages Cowrie SSH honeypot deployment and configuration"""
    
    def __init__(self, install_dir: str = "/opt/sutazaiapp/backend/security/cowrie"):
        self.install_dir = Path(install_dir)
        self.config_dir = self.install_dir / "etc"
        self.log_dir = self.install_dir / "var" / "log" / "cowrie"
        self.data_dir = self.install_dir / "var" / "lib" / "cowrie"
        self.is_installed = False
        self.is_running = False
        self.process = None
        
    async def install(self) -> bool:
        """Install Cowrie honeypot"""
        try:
            logger.info("Installing Cowrie SSH honeypot...")
            
            # Create directories
            self.install_dir.mkdir(parents=True, exist_ok=True)
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already installed
            if (self.install_dir / "bin" / "cowrie").exists():
                self.is_installed = True
                logger.info("Cowrie already installed")
                return True
            
            # Install dependencies
            await self._install_dependencies()
            
            # Download and install Cowrie
            await self._download_cowrie()
            
            # Configure Cowrie
            await self._configure_cowrie()
            
            # Create startup script
            await self._create_startup_script()
            
            self.is_installed = True
            logger.info("Cowrie installation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cowrie installation failed: {e}")
            return False
    
    async def _install_dependencies(self):
        """Install Cowrie dependencies"""
        dependencies = [
            "python3-pip",
            "python3-dev",
            "python3-venv",
            "libssl-dev",
            "libffi-dev",
            "build-essential",
            "libpython3-dev",
            "python3-minimal",
            "authbind"
        ]
        
        # Install system dependencies
        try:
            cmd = ["apt-get", "update"]
            await self._run_command(cmd)
            
            cmd = ["apt-get", "install", "-y"] + dependencies
            await self._run_command(cmd)
            
        except Exception as e:
            logger.warning(f"Could not install system dependencies: {e}")
    
    async def _download_cowrie(self):
        """Download and extract Cowrie"""
        cowrie_url = "https://github.com/cowrie/cowrie/archive/refs/heads/master.tar.gz"
        
        # Download
        logger.info("Downloading Cowrie...")
        response = requests.get(cowrie_url, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = self.install_dir / "cowrie.tar.gz"
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract
        logger.info("Extracting Cowrie...")
        with tarfile.open(temp_file, 'r:gz') as tar:
            tar.extractall(self.install_dir)
        
        # Move extracted files to correct location
        extracted_dir = self.install_dir / "cowrie-master"
        if extracted_dir.exists():
            for item in extracted_dir.iterdir():
                shutil.move(str(item), str(self.install_dir))
            extracted_dir.rmdir()
        
        # Clean up
        temp_file.unlink()
        
        # Create virtual environment and install Python dependencies
        await self._setup_python_environment()
    
    async def _setup_python_environment(self):
        """Set up Python virtual environment for Cowrie"""
        venv_dir = self.install_dir / "cowrie-env"
        
        # Create virtual environment
        cmd = ["python3", "-m", "venv", str(venv_dir)]
        await self._run_command(cmd, cwd=self.install_dir)
        
        # Install requirements
        pip_path = venv_dir / "bin" / "pip"
        requirements_file = self.install_dir / "requirements.txt"
        
        if requirements_file.exists():
            cmd = [str(pip_path), "install", "-r", str(requirements_file)]
            await self._run_command(cmd, cwd=self.install_dir)
        
        # Install additional dependencies
        additional_deps = [
            "twisted[tls]",
            "cryptography",
            "configparser",
            "pyopenssl",
            "pyparsing",
            "packaging",
            "appdirs",
            "pyasn1_modules",
            "attrs",
            "bcrypt",
            "constantly",
            "hyperlink",
            "incremental",
            "pyasn1",
            "pycparser",
            "pynacl",
            "six",
            "zope.interface"
        ]
        
        for dep in additional_deps:
            try:
                cmd = [str(pip_path), "install", dep]
                await self._run_command(cmd, cwd=self.install_dir)
            except Exception as e:
                logger.warning(f"Could not install {dep}: {e}")
    
    async def _configure_cowrie(self):
        """Configure Cowrie honeypot"""
        config_file = self.config_dir / "cowrie.cfg"
        
        config_content = f"""[honeypot]
# Hostname for the honeypot
hostname = sutazai-server

# Banner to show when connecting
ssh_version_string = SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5

# Login timeout
login_timeout = 120

# Authentication timeout
authentication_timeout = 120

# Maximum authentication attempts
max_auth_attempts = 3

# Enable interaction logging
interact_enabled = true

# Fake filesystem
filesystem_file = share/cowrie/fs.pickle

# Enable sftp
sftp_enabled = true

# Listen endpoints
[ssh]
# SSH version to present
version = SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5

# Listen addresses and ports
listen_endpoints = tcp:2222:interface=0.0.0.0

# Host key files
rsa_public_key = etc/ssh_host_rsa_key.pub
rsa_private_key = etc/ssh_host_rsa_key
dsa_public_key = etc/ssh_host_dsa_key.pub
dsa_private_key = etc/ssh_host_dsa_key

[telnet]
# Telnet is disabled by default
enabled = false

# Authentication
[backend_pool]
# Authentication backend
pool_only = true

# Users and passwords that will be accepted
[backend_pool:userdb]
# Accept any username/password combination
enabled = true

# Output plugins
[output_jsonlog]
# JSON logging
logfile = var/log/cowrie/cowrie.json

[output_mysql]
# MySQL output (disabled)
enabled = false

[output_sqlite3]
# SQLite output
enabled = true
database = var/lib/cowrie/cowrie.db

# File download/upload settings
[artifact]
download_dir = var/lib/cowrie/downloads

# Shell settings
[shell]
# Shell to present
filesystem = share/cowrie/fs.pickle

# Process list
processes_file = share/cowrie/cmdoutput.json

# Enable file systems
[backend_pool:filesystem]
enabled = true

# File system configuration
[fs]
home_path = /home
guest_home_path = /home
fake_home_paths = /root, /home/admin, /home/user

# Logging
[output_localsyslog]
enabled = false

[output_file]
enabled = true
logfile = var/log/cowrie/cowrie.log

# Integration with our system
[output_webhook]
enabled = true
url = http://localhost:8000/api/v1/honeypot/cowrie-webhook
headers = {{"Content-Type": "application/json", "Authorization": "Bearer honeypot-token"}}
"""
        
        # Write config file
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Generate SSH host keys
        await self._generate_ssh_keys()
        
        # Create filesystem pickle if it doesn't exist
        await self._create_filesystem()
        
        logger.info("Cowrie configuration completed")
    
    async def _generate_ssh_keys(self):
        """Generate SSH host keys for Cowrie"""
        key_types = [
            ("rsa", "ssh_host_rsa_key", 2048),
            ("dsa", "ssh_host_dsa_key", 1024),
            ("ecdsa", "ssh_host_ecdsa_key", 256),
            ("ed25519", "ssh_host_ed25519_key", 256),
        ]
        
        for key_type, key_name, key_size in key_types:
            private_key_path = self.config_dir / key_name
            public_key_path = self.config_dir / f"{key_name}.pub"
            
            if not private_key_path.exists():
                try:
                    if key_type == "ed25519":
                        cmd = ["ssh-keygen", "-t", key_type, "-f", str(private_key_path), "-N", ""]
                    else:
                        cmd = ["ssh-keygen", "-t", key_type, "-b", str(key_size), "-f", str(private_key_path), "-N", ""]
                    
                    await self._run_command(cmd)
                    logger.info(f"Generated SSH {key_type} key")
                    
                except Exception as e:
                    logger.warning(f"Could not generate {key_type} key: {e}")
    
    async def _create_filesystem(self):
        """Create fake filesystem for Cowrie"""
        fs_dir = self.install_dir / "share" / "cowrie"
        fs_dir.mkdir(parents=True, exist_ok=True)
        
        fs_pickle = fs_dir / "fs.pickle"
        if not fs_pickle.exists():
            # Create a basic filesystem structure
            try:
                # Use Cowrie's createfs utility if available
                createfs_script = self.install_dir / "bin" / "createfs"
                if createfs_script.exists():
                    cmd = ["python3", str(createfs_script)]
                    await self._run_command(cmd, cwd=self.install_dir)
                else:
                    # Create a minimal filesystem pickle manually
                    import pickle
                    
                    # Basic filesystem structure
                    fs_structure = {
                        '/': {'type': 'dir', 'mode': 0o755, 'uid': 0, 'gid': 0},
                        '/bin': {'type': 'dir', 'mode': 0o755, 'uid': 0, 'gid': 0},
                        '/etc': {'type': 'dir', 'mode': 0o755, 'uid': 0, 'gid': 0},
                        '/home': {'type': 'dir', 'mode': 0o755, 'uid': 0, 'gid': 0},
                        '/root': {'type': 'dir', 'mode': 0o700, 'uid': 0, 'gid': 0},
                        '/tmp': {'type': 'dir', 'mode': 0o1777, 'uid': 0, 'gid': 0},
                        '/var': {'type': 'dir', 'mode': 0o755, 'uid': 0, 'gid': 0},
                        '/usr': {'type': 'dir', 'mode': 0o755, 'uid': 0, 'gid': 0},
                        '/bin/bash': {'type': 'file', 'mode': 0o755, 'uid': 0, 'gid': 0, 'size': 1234567},
                        '/bin/ls': {'type': 'file', 'mode': 0o755, 'uid': 0, 'gid': 0, 'size': 12345},
                        '/etc/passwd': {'type': 'file', 'mode': 0o644, 'uid': 0, 'gid': 0, 'size': 1024},
                    }
                    
                    with open(fs_pickle, 'wb') as f:
                        pickle.dump(fs_structure, f)
                        
                logger.info("Created filesystem structure")
                
            except Exception as e:
                logger.warning(f"Could not create filesystem: {e}")
        
        # Create command output file
        cmdoutput_file = fs_dir / "cmdoutput.json"
        if not cmdoutput_file.exists():
            cmdoutput = {
                "ls": {
                    "": "bin  boot  dev  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var\n"
                },
                "ps": {
                    "": "  PID TTY          TIME CMD\n  123 pts/0    00:00:00 bash\n  456 pts/0    00:00:00 ps\n"
                },
                "whoami": {
                    "": "root\n"
                },
                "uname": {
                    "-a": "Linux sutazai-server 5.4.0-74-generic #83-Ubuntu SMP Sat May 8 02:35:39 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux\n"
                }
            }
            
            with open(cmdoutput_file, 'w') as f:
                json.dump(cmdoutput, f, indent=2)
    
    async def _create_startup_script(self):
        """Create Cowrie startup script"""
        startup_script = self.install_dir / "start_cowrie.sh"
        venv_python = self.install_dir / "cowrie-env" / "bin" / "python"
        cowrie_script = self.install_dir / "bin" / "cowrie"
        
        script_content = f"""#!/bin/bash
cd {self.install_dir}
export PYTHONPATH={self.install_dir}:$PYTHONPATH
{venv_python} {cowrie_script} start
"""
        
        with open(startup_script, 'w') as f:
            f.write(script_content)
        
        # Make executable
        startup_script.chmod(0o755)
        
        # Create stop script
        stop_script = self.install_dir / "stop_cowrie.sh"
        stop_content = f"""#!/bin/bash
cd {self.install_dir}
export PYTHONPATH={self.install_dir}:$PYTHONPATH
{venv_python} {cowrie_script} stop
"""
        
        with open(stop_script, 'w') as f:
            f.write(stop_content)
        
        stop_script.chmod(0o755)
    
    async def start(self) -> bool:
        """Start Cowrie honeypot"""
        if not self.is_installed:
            if not await self.install():
                return False
        
        if self.is_running:
            logger.info("Cowrie is already running")
            return True
        
        try:
            logger.info("Starting Cowrie honeypot...")
            
            # Use startup script
            startup_script = self.install_dir / "start_cowrie.sh"
            
            # Start Cowrie process
            self.process = await asyncio.create_subprocess_exec(
                str(startup_script),
                cwd=self.install_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait a moment for startup
            await asyncio.sleep(5)
            
            # Check if process is still running
            if self.process.returncode is None:
                self.is_running = True
                logger.info("Cowrie started successfully")
                
                # Start log monitoring
                asyncio.create_task(self._monitor_logs())
                
                return True
            else:
                stdout, stderr = await self.process.communicate()
                logger.error(f"Cowrie failed to start: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start Cowrie: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop Cowrie honeypot"""
        if not self.is_running:
            return True
        
        try:
            logger.info("Stopping Cowrie honeypot...")
            
            # Use stop script
            stop_script = self.install_dir / "stop_cowrie.sh"
            
            process = await asyncio.create_subprocess_exec(
                str(stop_script),
                cwd=self.install_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if self.process:
                try:
                    self.process.terminate()
                    await asyncio.wait_for(self.process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            
            self.is_running = False
            logger.info("Cowrie stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Cowrie: {e}")
            return False
    
    async def _monitor_logs(self):
        """Monitor Cowrie logs and integrate with honeypot infrastructure"""
        json_log_file = self.log_dir / "cowrie.json"
        
        # Wait for log file to be created
        while not json_log_file.exists() and self.is_running:
            await asyncio.sleep(1)
        
        if not self.is_running:
            return
        
        try:
            # Monitor log file for new entries
            with open(json_log_file, 'r') as f:
                # Seek to end of file
                f.seek(0, 2)
                
                while self.is_running:
                    line = f.readline()
                    if line:
                        try:
                            log_entry = json.loads(line.strip())
                            await self._process_log_entry(log_entry)
                        except json.JSONDecodeError:
                            continue
                    else:
                        await asyncio.sleep(0.1)
                        
        except Exception as e:
            logger.error(f"Log monitoring error: {e}")
    
    async def _process_log_entry(self, log_entry: Dict[str, Any]):
        """Process Cowrie log entry and integrate with honeypot system"""
        try:
            event_type = log_entry.get('eventid', 'unknown')
            source_ip = log_entry.get('src_ip', '0.0.0.0')
            
            # Map Cowrie events to our honeypot system
            if event_type == 'cowrie.session.connect':
                await self._handle_connection_event(log_entry)
            elif event_type == 'cowrie.login.success':
                await self._handle_login_success(log_entry)
            elif event_type == 'cowrie.login.failed':
                await self._handle_login_failed(log_entry)
            elif event_type == 'cowrie.command.input':
                await self._handle_command_input(log_entry)
            elif event_type == 'cowrie.session.file_download':
                await self._handle_file_download(log_entry)
            elif event_type == 'cowrie.session.file_upload':
                await self._handle_file_upload(log_entry)
            
        except Exception as e:
            logger.error(f"Error processing Cowrie log entry: {e}")
    
    async def _handle_connection_event(self, log_entry: Dict[str, Any]):
        """Handle SSH connection event"""
        # This integrates with our honeypot infrastructure
        # For now, just log it
        logger.info(f"Cowrie SSH connection from {log_entry.get('src_ip')}:{log_entry.get('src_port')}")
    
    async def _handle_login_success(self, log_entry: Dict[str, Any]):
        """Handle successful login attempt"""
        logger.warning(f"Cowrie successful login: {log_entry.get('username')}@{log_entry.get('src_ip')}")
    
    async def _handle_login_failed(self, log_entry: Dict[str, Any]):
        """Handle failed login attempt"""
        logger.info(f"Cowrie failed login: {log_entry.get('username')}@{log_entry.get('src_ip')}")
    
    async def _handle_command_input(self, log_entry: Dict[str, Any]):
        """Handle command execution"""
        command = log_entry.get('input', '')
        logger.info(f"Cowrie command from {log_entry.get('src_ip')}: {command}")
    
    async def _handle_file_download(self, log_entry: Dict[str, Any]):
        """Handle file download attempt"""
        logger.warning(f"Cowrie file download from {log_entry.get('src_ip')}: {log_entry.get('url')}")
    
    async def _handle_file_upload(self, log_entry: Dict[str, Any]):
        """Handle file upload attempt"""
        logger.warning(f"Cowrie file upload from {log_entry.get('src_ip')}: {log_entry.get('filename')}")
    
    async def _run_command(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300):
        """Run system command"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"Command failed: {' '.join(cmd)} - {error_msg}")
            
            return stdout.decode()
            
        except asyncio.TimeoutError:
            if process:
                process.kill()
                await process.wait()
            raise Exception(f"Command timed out: {' '.join(cmd)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Cowrie status"""
        return {
            "installed": self.is_installed,
            "running": self.is_running,
            "install_dir": str(self.install_dir),
            "log_dir": str(self.log_dir),
            "config_dir": str(self.config_dir),
            "process_id": self.process.pid if self.process else None
        }

class CowrieIntegration:
    """Integration between Cowrie and our honeypot infrastructure"""
    
    def __init__(self):
        self.cowrie_manager = CowrieManager()
        self.integration_active = False
        
    async def deploy(self) -> bool:
        """Deploy Cowrie honeypot"""
        try:
            logger.info("Deploying Cowrie SSH honeypot...")
            
            # Install and start Cowrie
            if await self.cowrie_manager.start():
                self.integration_active = True
                logger.info("Cowrie honeypot deployed successfully")
                return True
            else:
                logger.error("Failed to deploy Cowrie honeypot")
                return False
                
        except Exception as e:
            logger.error(f"Cowrie deployment failed: {e}")
            return False
    
    async def undeploy(self) -> bool:
        """Undeploy Cowrie honeypot"""
        try:
            if await self.cowrie_manager.stop():
                self.integration_active = False
                logger.info("Cowrie honeypot undeployed successfully")
                return True
            else:
                logger.error("Failed to undeploy Cowrie honeypot")
                return False
                
        except Exception as e:
            logger.error(f"Cowrie undeployment failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "integration_active": self.integration_active,
            "cowrie_status": self.cowrie_manager.get_status()
        }

# Global Cowrie integration instance
cowrie_integration = CowrieIntegration()