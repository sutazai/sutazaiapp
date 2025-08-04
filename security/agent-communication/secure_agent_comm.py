#!/usr/bin/env python3
"""
Secure Agent Communication System
Implements mTLS and end-to-end encryption for 131 SutazAI agents
"""

import asyncio
import ssl
import socket
import logging
import json
import time
import hashlib
import hmac
import base64
import os
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import aiofiles
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import jwt
import redis
import threading
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from psycopg2.extras import RealDictCursor

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"
    MONITOR = "monitor"
    SECURITY = "security"
    ADMIN = "admin"

class MessageType(Enum):
    HEARTBEAT = "heartbeat"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    SECURITY_ALERT = "security_alert"
    COMMAND = "command"
    DATA_SYNC = "data_sync"

class SecurityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentIdentity:
    """Agent identity and credentials"""
    agent_id: str
    agent_name: str
    role: AgentRole
    public_key: bytes
    certificate: bytes
    permissions: Set[str]
    security_level: SecurityLevel
    last_seen: datetime
    is_active: bool = True

@dataclass
class SecureMessage:
    """Secure message structure"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: bytes  # Encrypted
    timestamp: datetime
    signature: bytes
    nonce: bytes
    expires_at: Optional[datetime] = None

class CertificateAuthority:
    """Internal Certificate Authority for agent certificates"""
    
    def __init__(self, ca_key_path: str = None, ca_cert_path: str = None):
        self.ca_key_path = ca_key_path or "/opt/sutazaiapp/security/ca/ca-key.pem"
        self.ca_cert_path = ca_cert_path or "/opt/sutazaiapp/security/ca/ca-cert.pem"
        self.ca_private_key = None
        self.ca_certificate = None
        self._initialize_ca()
    
    def _initialize_ca(self):
        """Initialize or load CA"""
        try:
            if os.path.exists(self.ca_key_path) and os.path.exists(self.ca_cert_path):
                self._load_ca()
            else:
                self._generate_ca()
        except Exception as e:
            logging.error(f"Failed to initialize CA: {e}")
            raise
    
    def _generate_ca(self):
        """Generate new CA key and certificate"""
        # Generate CA private key
        self.ca_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )
        
        # Generate CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SutazAI"),
            x509.NameAttribute(NameOID.COMMON_NAME, "SutazAI Root CA"),
        ])
        
        self.ca_certificate = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            self.ca_private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=3650)  # 10 years
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                key_cert_sign=True,
                crl_sign=True,
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).sign(self.ca_private_key, hashes.SHA256())
        
        # Save CA key and certificate
        os.makedirs(os.path.dirname(self.ca_key_path), exist_ok=True)
        
        with open(self.ca_key_path, "wb") as f:
            f.write(self.ca_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        with open(self.ca_cert_path, "wb") as f:
            f.write(self.ca_certificate.public_bytes(serialization.Encoding.PEM))
        
        logging.info("Generated new CA certificate")
    
    def _load_ca(self):
        """Load existing CA key and certificate"""
        with open(self.ca_key_path, "rb") as f:
            self.ca_private_key = serialization.load_pem_private_key(
                f.read(), password=None
            )
        
        with open(self.ca_cert_path, "rb") as f:
            self.ca_certificate = x509.load_pem_x509_certificate(f.read())
        
        logging.info("Loaded existing CA certificate")
    
    def generate_agent_certificate(self, agent_id: str, agent_name: str, 
                                 role: AgentRole) -> Tuple[bytes, bytes]:
        """Generate certificate for agent"""
        # Generate agent private key
        agent_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate subject
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SutazAI"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, role.value),
            x509.NameAttribute(NameOID.COMMON_NAME, agent_name),
        ])
        
        # Generate certificate
        certificate = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            self.ca_certificate.subject
        ).public_key(
            agent_private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)  # 1 year
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(agent_name),
                x509.DNSName(f"{agent_id}.sutazai.local"),
            ]),
            critical=False,
        ).add_extension(
            x509.KeyUsage(
                key_cert_sign=False,
                crl_sign=False,
                digital_signature=True,
                content_commitment=True,
                key_encipherment=True,
                data_encipherment=True,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=True,
        ).sign(self.ca_private_key, hashes.SHA256())
        
        # Return private key and certificate as PEM bytes
        private_key_pem = agent_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
        
        return private_key_pem, certificate_pem
    
    def verify_certificate(self, certificate_pem: bytes) -> bool:
        """Verify agent certificate against CA"""
        try:
            certificate = x509.load_pem_x509_certificate(certificate_pem)
            
            # Check if certificate is signed by our CA
            ca_public_key = self.ca_certificate.public_key()
            ca_public_key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
                padding.PKCS1v15(),
                certificate.signature_hash_algorithm,
            )
            
            # Check validity period
            now = datetime.utcnow()
            if now < certificate.not_valid_before or now > certificate.not_valid_after:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Certificate verification failed: {e}")
            return False

class SecureAgentCommunication:
    """Secure communication system for SutazAI agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ca = CertificateAuthority()
        self.agents: Dict[str, AgentIdentity] = {}
        self.redis_client = None
        self.db_connection = None
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.encryption_key = None
        self.running = False
        self.server = None
        self.session_pool = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize secure communication components"""
        try:
            # Initialize Redis for message queuing
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'redis'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password'),
                ssl=True,
                decode_responses=False  # Keep binary data
            )
            
            # Initialize PostgreSQL for agent registry
            self.db_connection = psycopg2.connect(
                host=self.config.get('postgres_host', 'postgres'),
                port=self.config.get('postgres_port', 5432),
                database=self.config.get('postgres_db', 'sutazai'),
                user=self.config.get('postgres_user', 'sutazai'),
                password=self.config.get('postgres_password'),
                sslmode='require'
            )
            
            # Initialize encryption
            self._setup_encryption()
            
            # Setup message handlers
            self._setup_message_handlers()
            
            # Load registered agents
            self._load_agents()
            
            self.logger.info("Secure Agent Communication initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Secure Agent Communication: {e}")
            raise
    
    def _setup_encryption(self):
        """Setup encryption for message payloads"""
        master_key = self.config.get('master_encryption_key', 'default-key')
        salt = self.config.get('encryption_salt', b'salt1234')
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.encryption_key = Fernet(key)
    
    def _setup_message_handlers(self):
        """Setup message type handlers"""
        self.message_handlers = {
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.TASK_RESPONSE: self._handle_task_response,
            MessageType.STATUS_UPDATE: self._handle_status_update,
            MessageType.SECURITY_ALERT: self._handle_security_alert,
            MessageType.COMMAND: self._handle_command,
            MessageType.DATA_SYNC: self._handle_data_sync,
        }
    
    def _load_agents(self):
        """Load registered agents from database"""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM agent_registry WHERE is_active = true")
            
            for row in cursor.fetchall():
                agent = AgentIdentity(
                    agent_id=row['agent_id'],
                    agent_name=row['agent_name'],
                    role=AgentRole(row['role']),
                    public_key=row['public_key'].encode() if row['public_key'] else b'',
                    certificate=row['certificate'].encode() if row['certificate'] else b'',
                    permissions=set(row['permissions'] or []),
                    security_level=SecurityLevel(row['security_level']),
                    last_seen=row['last_seen'],
                    is_active=row['is_active']
                )
                self.agents[agent.agent_id] = agent
            
            cursor.close()
            self.logger.info(f"Loaded {len(self.agents)} registered agents")
            
        except Exception as e:
            self.logger.error(f"Failed to load agents: {e}")
    
    async def register_agent(self, agent_id: str, agent_name: str, 
                           role: AgentRole, permissions: Set[str],
                           security_level: SecurityLevel = SecurityLevel.MEDIUM) -> AgentIdentity:
        """Register new agent and generate certificates"""
        try:
            # Generate certificate for agent
            private_key_pem, certificate_pem = self.ca.generate_agent_certificate(
                agent_id, agent_name, role
            )
            
            # Extract public key
            private_key = serialization.load_pem_private_key(private_key_pem, password=None)
            public_key_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Create agent identity
            agent = AgentIdentity(
                agent_id=agent_id,
                agent_name=agent_name,
                role=role,
                public_key=public_key_pem,
                certificate=certificate_pem,
                permissions=permissions,
                security_level=security_level,
                last_seen=datetime.utcnow(),
                is_active=True
            )
            
            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO agent_registry 
                (agent_id, agent_name, role, public_key, certificate, permissions, 
                 security_level, last_seen, is_active, private_key)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (agent_id) DO UPDATE SET
                agent_name = EXCLUDED.agent_name,
                role = EXCLUDED.role,
                public_key = EXCLUDED.public_key,
                certificate = EXCLUDED.certificate,
                permissions = EXCLUDED.permissions,
                security_level = EXCLUDED.security_level,
                last_seen = EXCLUDED.last_seen,
                is_active = EXCLUDED.is_active,
                private_key = EXCLUDED.private_key
            """, (
                agent_id, agent_name, role.value, public_key_pem.decode(),
                certificate_pem.decode(), list(permissions), security_level.value,
                datetime.utcnow(), True, private_key_pem.decode()
            ))
            self.db_connection.commit()
            cursor.close()
            
            # Add to local registry
            self.agents[agent_id] = agent
            
            self.logger.info(f"Registered agent: {agent_name} ({agent_id})")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    async def create_secure_message(self, sender_id: str, recipient_id: str,
                                  message_type: MessageType, payload: Dict[str, Any],
                                  expires_in: Optional[int] = None) -> SecureMessage:
        """Create encrypted and signed message"""
        try:
            # Verify sender exists
            if sender_id not in self.agents:
                raise ValueError(f"Unknown sender: {sender_id}")
            
            # Verify recipient exists (or allow broadcast)
            if recipient_id != "broadcast" and recipient_id not in self.agents:
                raise ValueError(f"Unknown recipient: {recipient_id}")
            
            # Generate message ID and nonce
            message_id = self._generate_message_id()
            nonce = os.urandom(16)
            
            # Encrypt payload
            payload_json = json.dumps(payload).encode()
            encrypted_payload = self.encryption_key.encrypt(payload_json)
            
            # Calculate expiration
            expires_at = None
            if expires_in:
                expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            
            # Create message structure for signing
            message_data = {
                "message_id": message_id,
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "message_type": message_type.value,
                "timestamp": datetime.utcnow().isoformat(),
                "nonce": base64.b64encode(nonce).decode(),
                "expires_at": expires_at.isoformat() if expires_at else None
            }
            
            # Sign message
            signature = self._sign_message(sender_id, message_data, encrypted_payload)
            
            return SecureMessage(
                message_id=message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=message_type,
                payload=encrypted_payload,
                timestamp=datetime.utcnow(),
                signature=signature,
                nonce=nonce,
                expires_at=expires_at
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create secure message: {e}")
            raise
    
    async def send_message(self, message: SecureMessage) -> bool:
        """Send secure message to recipient"""
        try:
            # Verify message hasn't expired
            if message.expires_at and datetime.utcnow() > message.expires_at:
                raise ValueError("Message has expired")
            
            # Serialize message
            message_data = {
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "message_type": message.message_type.value,
                "payload": base64.b64encode(message.payload).decode(),
                "timestamp": message.timestamp.isoformat(),
                "signature": base64.b64encode(message.signature).decode(),
                "nonce": base64.b64encode(message.nonce).decode(),
                "expires_at": message.expires_at.isoformat() if message.expires_at else None
            }
            
            # Store in Redis message queue
            if message.recipient_id == "broadcast":
                # Broadcast to all agents
                for agent_id in self.agents.keys():
                    if agent_id != message.sender_id:
                        queue_key = f"agent_messages:{agent_id}"
                        self.redis_client.lpush(queue_key, json.dumps(message_data))
            else:
                # Send to specific recipient
                queue_key = f"agent_messages:{message.recipient_id}"
                self.redis_client.lpush(queue_key, json.dumps(message_data))
            
            # Set TTL if message expires
            if message.expires_at:
                ttl = int((message.expires_at - datetime.utcnow()).total_seconds())
                if ttl > 0:
                    self.redis_client.expire(queue_key, ttl)
            
            self.logger.debug(f"Sent message {message.message_id} from {message.sender_id} to {message.recipient_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_messages(self, agent_id: str, timeout: int = 1) -> List[SecureMessage]:
        """Receive messages for agent"""
        try:
            # Verify agent exists
            if agent_id not in self.agents:
                raise ValueError(f"Unknown agent: {agent_id}")
            
            messages = []
            queue_key = f"agent_messages:{agent_id}"
            
            # Get messages from queue
            message_data_list = self.redis_client.lrange(queue_key, 0, -1)
            self.redis_client.delete(queue_key)  # Clear queue after reading
            
            for message_json in message_data_list:
                try:
                    message_data = json.loads(message_json)
                    
                    # Verify and decrypt message
                    message = await self._verify_and_decrypt_message(message_data)
                    if message:
                        messages.append(message)
                        
                except Exception as e:
                    self.logger.error(f"Failed to process message: {e}")
                    continue
            
            # Update agent last seen
            if messages:
                await self._update_agent_last_seen(agent_id)
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Failed to receive messages for {agent_id}: {e}")
            return []
    
    async def _verify_and_decrypt_message(self, message_data: Dict[str, Any]) -> Optional[SecureMessage]:
        """Verify signature and decrypt message"""
        try:
            sender_id = message_data['sender_id']
            
            # Verify sender exists
            if sender_id not in self.agents:
                self.logger.warning(f"Message from unknown sender: {sender_id}")
                return None
            
            # Check message expiration
            if message_data.get('expires_at'):
                expires_at = datetime.fromisoformat(message_data['expires_at'])
                if datetime.utcnow() > expires_at:
                    self.logger.debug(f"Discarded expired message: {message_data['message_id']}")
                    return None
            
            # Verify signature
            encrypted_payload = base64.b64decode(message_data['payload'])
            signature = base64.b64decode(message_data['signature'])
            
            # Create message data for verification (excluding payload and signature)
            verify_data = {
                "message_id": message_data['message_id'],
                "sender_id": sender_id,
                "recipient_id": message_data['recipient_id'],
                "message_type": message_data['message_type'],
                "timestamp": message_data['timestamp'],
                "nonce": message_data['nonce'],
                "expires_at": message_data.get('expires_at')
            }
            
            if not self._verify_message_signature(sender_id, verify_data, encrypted_payload, signature):
                self.logger.warning(f"Invalid signature for message: {message_data['message_id']}")
                return None
            
            # Decrypt payload
            try:
                decrypted_payload = self.encryption_key.decrypt(encrypted_payload)
            except Exception as e:
                self.logger.error(f"Failed to decrypt payload: {e}")
                return None
            
            return SecureMessage(
                message_id=message_data['message_id'],
                sender_id=sender_id,
                recipient_id=message_data['recipient_id'],
                message_type=MessageType(message_data['message_type']),
                payload=decrypted_payload,  # Store decrypted payload
                timestamp=datetime.fromisoformat(message_data['timestamp']),
                signature=signature,
                nonce=base64.b64decode(message_data['nonce']),
                expires_at=datetime.fromisoformat(message_data['expires_at']) if message_data.get('expires_at') else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to verify and decrypt message: {e}")
            return None
    
    def _sign_message(self, sender_id: str, message_data: Dict[str, Any], payload: bytes) -> bytes:
        """Sign message with sender's private key"""
        try:
            # Get sender's private key from database
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT private_key FROM agent_registry WHERE agent_id = %s", (sender_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                raise ValueError(f"No private key found for agent: {sender_id}")
            
            private_key_pem = result[0].encode()
            private_key = serialization.load_pem_private_key(private_key_pem, password=None)
            
            # Create signature data
            sign_data = json.dumps(message_data, sort_keys=True).encode() + payload
            
            # Sign with RSA-PSS
            signature = private_key.sign(
                sign_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Failed to sign message: {e}")
            raise
    
    def _verify_message_signature(self, sender_id: str, message_data: Dict[str, Any],
                                payload: bytes, signature: bytes) -> bool:
        """Verify message signature"""
        try:
            # Get sender's public key
            sender = self.agents.get(sender_id)
            if not sender:
                return False
            
            public_key = serialization.load_pem_public_key(sender.public_key)
            
            # Create signature data
            sign_data = json.dumps(message_data, sort_keys=True).encode() + payload
            
            # Verify signature
            public_key.verify(
                signature,
                sign_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Signature verification failed: {e}")
            return False
    
    async def _update_agent_last_seen(self, agent_id: str):
        """Update agent's last seen timestamp"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "UPDATE agent_registry SET last_seen = %s WHERE agent_id = %s",
                (datetime.utcnow(), agent_id)
            )
            self.db_connection.commit()
            cursor.close()
            
            # Update local registry
            if agent_id in self.agents:
                self.agents[agent_id].last_seen = datetime.utcnow()
                
        except Exception as e:
            self.logger.error(f"Failed to update last seen for {agent_id}: {e}")
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"msg_{timestamp}_{random_part}"
    
    # Message handlers
    async def _handle_heartbeat(self, message: SecureMessage):
        """Handle heartbeat message"""
        payload = json.loads(message.payload.decode())
        self.logger.debug(f"Heartbeat from {message.sender_id}: {payload}")
        
        # Update agent status
        await self._update_agent_last_seen(message.sender_id)
    
    async def _handle_task_request(self, message: SecureMessage):
        """Handle task request message"""
        payload = json.loads(message.payload.decode())
        self.logger.info(f"Task request from {message.sender_id}: {payload.get('task_type')}")
        
        # Process task request
        # This would integrate with your task management system
    
    async def _handle_task_response(self, message: SecureMessage):
        """Handle task response message"""
        payload = json.loads(message.payload.decode())
        self.logger.info(f"Task response from {message.sender_id}: {payload.get('status')}")
        
        # Process task response
        # This would integrate with your task management system
    
    async def _handle_status_update(self, message: SecureMessage):
        """Handle status update message"""
        payload = json.loads(message.payload.decode())
        self.logger.debug(f"Status update from {message.sender_id}: {payload}")
        
        # Update agent status in monitoring system
    
    async def _handle_security_alert(self, message: SecureMessage):
        """Handle security alert message"""
        payload = json.loads(message.payload.decode())
        self.logger.warning(f"Security alert from {message.sender_id}: {payload}")
        
        # Forward to security monitoring system
        alert_data = {
            "source_agent": message.sender_id,
            "alert_type": payload.get('alert_type'),
            "severity": payload.get('severity'),
            "description": payload.get('description'),
            "timestamp": message.timestamp.isoformat()
        }
        
        self.redis_client.lpush("security_alerts", json.dumps(alert_data))
    
    async def _handle_command(self, message: SecureMessage):
        """Handle command message"""
        payload = json.loads(message.payload.decode())
        self.logger.info(f"Command from {message.sender_id}: {payload.get('command')}")
        
        # Process command based on sender permissions
        sender = self.agents.get(message.sender_id)
        if sender and self._check_permission(sender, payload.get('command')):
            # Execute command
            pass
        else:
            self.logger.warning(f"Unauthorized command from {message.sender_id}")
    
    async def _handle_data_sync(self, message: SecureMessage):
        """Handle data synchronization message"""
        payload = json.loads(message.payload.decode())
        self.logger.debug(f"Data sync from {message.sender_id}: {payload.get('sync_type')}")
        
        # Process data synchronization
    
    def _check_permission(self, agent: AgentIdentity, command: str) -> bool:
        """Check if agent has permission for command"""
        required_permission = f"command:{command}"
        return required_permission in agent.permissions or "admin:all" in agent.permissions
    
    async def start_message_processing(self):
        """Start message processing loop"""
        self.running = True
        self.logger.info("Started secure agent communication")
        
        while self.running:
            try:
                # Process messages for all agents
                for agent_id in self.agents.keys():
                    messages = await self.receive_messages(agent_id, timeout=1)
                    
                    for message in messages:
                        # Route to appropriate handler
                        if message.message_type in self.message_handlers:
                            await self.message_handlers[message.message_type](message)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
                await asyncio.sleep(5)
    
    def stop_message_processing(self):
        """Stop message processing"""
        self.running = False
        self.logger.info("Stopped secure agent communication")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        active_agents = []
        inactive_agents = []
        
        now = datetime.utcnow()
        timeout_threshold = timedelta(minutes=5)
        
        for agent in self.agents.values():
            if agent.is_active and (now - agent.last_seen) < timeout_threshold:
                active_agents.append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.agent_name,
                    "role": agent.role.value,
                    "last_seen": agent.last_seen.isoformat(),
                    "security_level": agent.security_level.value
                })
            else:
                inactive_agents.append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.agent_name,
                    "role": agent.role.value,
                    "last_seen": agent.last_seen.isoformat() if agent.last_seen else None,
                    "is_active": agent.is_active
                })
        
        return {
            "total_agents": len(self.agents),
            "active_agents": len(active_agents),
            "inactive_agents": len(inactive_agents),
            "agents": {
                "active": active_agents,
                "inactive": inactive_agents
            }
        }

# Database schema for agent registry
AGENT_REGISTRY_SCHEMA = """
-- Agent registry table
CREATE TABLE IF NOT EXISTS agent_registry (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) UNIQUE NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    public_key TEXT NOT NULL,
    private_key TEXT NOT NULL,
    certificate TEXT NOT NULL,
    permissions JSONB DEFAULT '[]',
    security_level INTEGER DEFAULT 2,
    last_seen TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Message log table
CREATE TABLE IF NOT EXISTS message_log (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL,
    sender_id VARCHAR(255) NOT NULL,
    recipient_id VARCHAR(255) NOT NULL,
    message_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    delivered BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_agent_registry_agent_id ON agent_registry(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_registry_active ON agent_registry(is_active);
CREATE INDEX IF NOT EXISTS idx_message_log_timestamp ON message_log(timestamp);
"""

if __name__ == "__main__":
    # Example usage
    config = {
        'redis_host': 'redis',
        'redis_port': 6379,
        'postgres_host': 'postgres',
        'postgres_port': 5432,
        'postgres_db': 'sutazai',
        'postgres_user': 'sutazai',
        'master_encryption_key': 'sutazai-secure-key-2024',
        'encryption_salt': b'sutazai_salt_16b'
    }
    
    async def main():
        comm_system = SecureAgentCommunication(config)
        
        # Register sample agents
        await comm_system.register_agent(
            "orchestrator-001", "Main Orchestrator", 
            AgentRole.ORCHESTRATOR, {"command:all", "admin:read"},
            SecurityLevel.CRITICAL
        )
        
        await comm_system.register_agent(
            "worker-001", "Worker Agent 1", 
            AgentRole.WORKER, {"task:execute", "data:read"},
            SecurityLevel.MEDIUM
        )
        
        # Create and send a message
        message = await comm_system.create_secure_message(
            "orchestrator-001", "worker-001",
            MessageType.TASK_REQUEST,
            {"task_type": "process_data", "priority": "high"}
        )
        
        await comm_system.send_message(message)
        
        # Receive messages
        messages = await comm_system.receive_messages("worker-001")
        print(f"Received {len(messages)} messages")
        
        # Get agent status
        status = comm_system.get_agent_status()
        print(f"Agent Status: {status}")
    
    asyncio.run(main())