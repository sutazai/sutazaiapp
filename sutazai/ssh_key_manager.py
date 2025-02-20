#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import json
import base64
import paramiko
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SutazAISSHKeyManager:
    def __init__(self, config_path="/opt/sutazai_project/SutazAI/config"):
        # Logging Setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config_path, 'ssh_key_management.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Configuration Paths
        self.config_path = config_path
        self.ssh_keys_dir = os.path.join(config_path, 'ssh_keys')
        self.key_metadata_file = os.path.join(config_path, 'ssh_key_metadata.json')

        # Ensure directories exist
        os.makedirs(self.ssh_keys_dir, exist_ok=True)

        # Default SSH Configuration
        self.default_servers = {
            'code_server': {
                'hostname': '192.168.100.136',
                'username': 'root',
                'port': 22
            },
            'deploy_server': {
                'hostname': '192.168.100.178',
                'username': 'root', 
                'port': 22
            }
        }

    def generate_ssh_key(self, server_name, key_type='rsa', key_size=4096):
        """
        Generate SSH key pair for a specific server
        
        Args:
            server_name (str): Name of the server
            key_type (str): Type of SSH key (default: rsa)
            key_size (int): Key size in bits (default: 4096)
        
        Returns:
            dict: Generated key metadata
        """
        try:
            # Generate key pair
            private_key_path = os.path.join(self.ssh_keys_dir, f'{server_name}_id_{key_type}')
            public_key_path = f'{private_key_path}.pub'

            subprocess.run([
                'ssh-keygen', 
                '-t', key_type, 
                '-b', str(key_size), 
                '-f', private_key_path, 
                '-N', ''  # No passphrase
            ], check=True)

            # Read public key
            with open(public_key_path, 'r') as f:
                public_key = f.read().strip()

            # Prepare key metadata
            key_metadata = {
                'server_name': server_name,
                'key_type': key_type,
                'key_size': key_size,
                'private_key_path': private_key_path,
                'public_key_path': public_key_path,
                'created_at': datetime.now().isoformat()
            }

            # Update metadata file
            self._update_key_metadata(key_metadata)

            self.logger.info(f"SSH key generated for {server_name}")
            return key_metadata

        except Exception as e:
            self.logger.error(f"Error generating SSH key for {server_name}: {e}")
            raise

    def distribute_ssh_key(self, server_name, username=None, password=None):
        """
        Distribute SSH public key to target server
        
        Args:
            server_name (str): Name of the server
            username (str, optional): Username for SSH connection
            password (str, optional): Password for SSH connection
        """
        try:
            # Retrieve server details
            server_details = self.default_servers.get(server_name, {})
            hostname = server_details.get('hostname')
            port = server_details.get('port', 22)
            username = username or server_details.get('username')

            if not hostname or not username:
                raise ValueError(f"Invalid server configuration for {server_name}")

            # Retrieve key metadata
            key_metadata = self._get_key_metadata(server_name)
            public_key_path = key_metadata['public_key_path']

            # Use paramiko for key distribution
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect using password or key
            if password:
                client.connect(hostname, port=port, username=username, password=password)
            else:
                client.connect(hostname, port=port, username=username)

            # Read public key
            with open(public_key_path, 'r') as f:
                public_key = f.read()

            # Append public key to authorized_keys
            client.exec_command(f'mkdir -p ~/.ssh && chmod 700 ~/.ssh')
            client.exec_command(f'echo "{public_key}" >> ~/.ssh/authorized_keys')
            client.exec_command('chmod 600 ~/.ssh/authorized_keys')

            self.logger.info(f"SSH key distributed to {server_name}")

        except Exception as e:
            self.logger.error(f"Error distributing SSH key to {server_name}: {e}")
            raise

    def _update_key_metadata(self, key_metadata):
        """
        Update SSH key metadata file
        
        Args:
            key_metadata (dict): Metadata for SSH key
        """
        try:
            # Read existing metadata
            if os.path.exists(self.key_metadata_file):
                with open(self.key_metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Update metadata
            metadata[key_metadata['server_name']] = key_metadata

            # Write updated metadata
            with open(self.key_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            self.logger.error(f"Error updating key metadata: {e}")
            raise

    def _get_key_metadata(self, server_name):
        """
        Retrieve SSH key metadata for a server
        
        Args:
            server_name (str): Name of the server
        
        Returns:
            dict: SSH key metadata
        """
        try:
            with open(self.key_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return metadata.get(server_name)

        except FileNotFoundError:
            self.logger.warning(f"No metadata found for {server_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving key metadata: {e}")
            raise

    def rotate_ssh_keys(self, server_name):
        """
        Rotate SSH keys for a specific server
        
        Args:
            server_name (str): Name of the server
        """
        try:
            # Remove existing keys
            old_metadata = self._get_key_metadata(server_name)
            if old_metadata:
                os.remove(old_metadata['private_key_path'])
                os.remove(old_metadata['public_key_path'])

            # Generate new keys
            new_metadata = self.generate_ssh_key(server_name)

            # Distribute new keys
            self.distribute_ssh_key(server_name)

            self.logger.info(f"SSH keys rotated for {server_name}")

        except Exception as e:
            self.logger.error(f"Error rotating SSH keys for {server_name}: {e}")
            raise

def main():
    # Example usage
    ssh_manager = SutazAISSHKeyManager()

    # Generate and distribute keys for both servers
    for server_name in ['code_server', 'deploy_server']:
        try:
            # Generate SSH keys
            ssh_manager.generate_ssh_key(server_name)

            # Distribute keys (requires manual password input or pre-configured authentication)
            # Uncomment and provide necessary authentication details
            # ssh_manager.distribute_ssh_key(server_name, username='root', password='your_password')

            # Rotate keys periodically
            # ssh_manager.rotate_ssh_keys(server_name)

        except Exception as e:
            print(f"Error processing {server_name}: {e}")

if __name__ == "__main__":
    main() 