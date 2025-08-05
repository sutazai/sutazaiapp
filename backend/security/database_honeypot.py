"""
Advanced Database Honeypot for SQL Injection and Unauthorized Access Detection
Comprehensive database honeypot supporting multiple database protocols and attack vectors
"""

import asyncio
import logging
import json
import struct
import hashlib
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import socket
import re

# Import honeypot infrastructure
from security.honeypot_infrastructure import BaseHoneypot, HoneypotType, HoneypotEvent

logger = logging.getLogger(__name__)

class DatabaseProtocolHandler:
    """Base class for database protocol handlers"""
    
    def __init__(self, database, intelligence_engine):
        self.database = database
        self.intelligence_engine = intelligence_engine
        self.logger = logging.getLogger(f"db_honeypot.{self.__class__.__name__}")
        
    async def handle_connection(self, reader: asyncio.StreamReader, 
                              writer: asyncio.StreamWriter, honeypot_id: str, port: int):
        """Handle database connection - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _detect_sql_injection(self, query: str) -> Tuple[bool, List[str], str]:
        """Detect SQL injection patterns in query"""
        indicators = []
        severity = "low"
        
        if not query:
            return False, indicators, severity
        
        query_lower = query.lower()
        
        # SQL injection patterns
        sql_patterns = [
            (r'union\s+select', 'union_based', 'high'),
            (r'or\s+1\s*=\s*1', 'boolean_based', 'high'),
            (r'and\s+1\s*=\s*1', 'boolean_based', 'medium'),
            (r'drop\s+table', 'destructive', 'critical'),
            (r'drop\s+database', 'destructive', 'critical'),
            (r'delete\s+from', 'destructive', 'high'),
            (r'truncate\s+table', 'destructive', 'high'),
            (r'exec\s+xp_', 'command_execution', 'critical'),
            (r'sp_executesql', 'command_execution', 'critical'),
            (r'information_schema', 'information_gathering', 'medium'),
            (r'sys\.tables', 'information_gathering', 'medium'),
            (r'sys\.columns', 'information_gathering', 'medium'),
            (r'sysobjects', 'information_gathering', 'medium'),
            (r'admin[\'\"]\s*--', 'authentication_bypass', 'high'),
            (r'\'\s*or\s*[\'\"]\s*1[\'\"]\s*=\s*[\'\"]\s*1', 'authentication_bypass', 'high'),
            (r'waitfor\s+delay', 'time_based', 'medium'),
            (r'benchmark\s*\(', 'time_based', 'medium'),
            (r'pg_sleep\s*\(', 'time_based', 'medium'),
            (r'sleep\s*\(', 'time_based', 'medium'),
            (r'load_file\s*\(', 'file_access', 'high'),
            (r'into\s+outfile', 'file_access', 'high'),
            (r'into\s+dumpfile', 'file_access', 'high'),
            (r'char\s*\(\d+\)', 'obfuscation', 'medium'),
            (r'hex\s*\(', 'obfuscation', 'medium'),
            (r'unhex\s*\(', 'obfuscation', 'medium'),
            (r'substring\s*\(', 'data_extraction', 'medium'),
            (r'ascii\s*\(', 'data_extraction', 'medium'),
            (r'length\s*\(', 'data_extraction', 'low'),
            (r'count\s*\(', 'data_enumeration', 'low'),
            (r'group_concat\s*\(', 'data_extraction', 'medium'),
            (r'concat\s*\(', 'data_extraction', 'low'),
            (r'version\s*\(\)', 'fingerprinting', 'low'),
            (r'user\s*\(\)', 'fingerprinting', 'low'),
            (r'current_user', 'fingerprinting', 'low'),
            (r'database\s*\(\)', 'fingerprinting', 'low'),
        ]
        
        max_severity_level = 0
        severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        for pattern, indicator, sev in sql_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                indicators.append(indicator)
                if severity_levels[sev] > max_severity_level:
                    max_severity_level = severity_levels[sev]
                    severity = sev
        
        # Check for encoded/obfuscated content
        if any(enc in query for enc in ['%', '\\x', '\\u', '+', 'char(']):
            indicators.append('obfuscation')
            if max_severity_level < 2:
                severity = 'medium'
        
        # Check for multiple statements
        if ';' in query and any(cmd in query_lower for cmd in ['select', 'insert', 'update', 'delete']):
            indicators.append('multiple_statements')
            if max_severity_level < 3:
                severity = 'high'
        
        return len(indicators) > 0, indicators, severity

class MySQLHoneypot(DatabaseProtocolHandler):
    """MySQL protocol honeypot"""
    
    def __init__(self, database, intelligence_engine):
        super().__init__(database, intelligence_engine)
        self.server_version = "5.7.33-0ubuntu0.18.04.1"
        self.salt = self._generate_salt()
        
    def _generate_salt(self) -> bytes:
        """Generate random salt for authentication"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=20)).encode()
    
    async def handle_connection(self, reader: asyncio.StreamReader, 
                              writer: asyncio.StreamWriter, honeypot_id: str, port: int):
        """Handle MySQL connection"""
        client_addr = writer.get_extra_info('peername')
        if not client_addr:
            return
            
        source_ip, source_port = client_addr[0], client_addr[1]
        
        try:
            # Send handshake packet
            handshake = self._create_handshake_packet()
            writer.write(handshake)
            await writer.drain()
            
            await self._log_interaction(
                honeypot_id, source_ip, source_port, port,
                "mysql_connection_attempt",
                "MySQL handshake sent",
                "medium"
            )
            
            # Read client authentication
            auth_data = await reader.read(4096)
            if auth_data:
                await self._handle_authentication(
                    auth_data, writer, honeypot_id, source_ip, source_port, port
                )
                
                # Handle subsequent packets (queries)
                while True:
                    try:
                        packet_data = await asyncio.wait_for(reader.read(4096), timeout=30)
                        if not packet_data:
                            break
                            
                        await self._handle_mysql_packet(
                            packet_data, writer, honeypot_id, source_ip, source_port, port
                        )
                        
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        self.logger.debug(f"Error handling MySQL packet: {e}")
                        break
            
        except Exception as e:
            self.logger.debug(f"MySQL connection error from {source_ip}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    def _create_handshake_packet(self) -> bytes:
        """Create MySQL handshake packet"""
        # Simplified MySQL handshake packet
        packet_data = (
            b'\x0a' +  # Protocol version
            self.server_version.encode() + b'\x00' +  # Server version
            struct.pack('<I', random.randint(1, 1000000)) +  # Connection ID
            self.salt[:8] + b'\x00' +  # Auth plugin data part 1
            struct.pack('<H', 0xffff) +  # Capabilities (lower 16 bits)
            b'\x08' +  # Character set
            struct.pack('<H', 0x0002) +  # Status flags
            struct.pack('<H', 0x0000) +  # Capabilities (upper 16 bits)
            b'\x15' +  # Auth plugin data length
            b'\x00' * 10 +  # Reserved
            self.salt[8:] + b'\x00'  # Auth plugin data part 2
        )
        
        # Add packet header (length + sequence number)
        packet_length = len(packet_data)
        header = struct.pack('<I', packet_length)[:-1] + b'\x00'  # 3 bytes length + 1 byte seq
        
        return header + packet_data
    
    async def _handle_authentication(self, auth_data: bytes, writer: asyncio.StreamWriter,
                                   honeypot_id: str, source_ip: str, source_port: int, port: int):
        """Handle MySQL authentication"""
        try:
            # Parse authentication packet (simplified)
            if len(auth_data) > 36:
                # Skip packet header and capability flags
                pos = 32
                
                # Extract username
                username_end = auth_data.find(b'\x00', pos)
                if username_end != -1:
                    username = auth_data[pos:username_end].decode('utf-8', errors='ignore')
                    
                    # Log authentication attempt
                    await self._log_interaction(
                        honeypot_id, source_ip, source_port, port,
                        "mysql_auth_attempt",
                        f"Username: {username}",
                        "high",
                        credentials={"username": username, "password": "hidden"}
                    )
                    
                    # Send authentication error
                    error_packet = self._create_error_packet(
                        1045, "28000", f"Access denied for user '{username}'@'{source_ip}'"
                    )
                    writer.write(error_packet)
                    await writer.drain()
                    
        except Exception as e:
            self.logger.debug(f"Error parsing MySQL auth: {e}")
    
    async def _handle_mysql_packet(self, packet_data: bytes, writer: asyncio.StreamWriter,
                                 honeypot_id: str, source_ip: str, source_port: int, port: int):
        """Handle MySQL query packet"""
        try:
            if len(packet_data) < 5:
                return
            
            # Skip packet header
            command = packet_data[4]
            
            if command == 0x03:  # COM_QUERY
                query = packet_data[5:].decode('utf-8', errors='ignore')
                
                # Detect SQL injection
                is_injection, indicators, severity = self._detect_sql_injection(query)
                
                await self._log_interaction(
                    honeypot_id, source_ip, source_port, port,
                    "mysql_query",
                    f"Query: {query[:200]}",
                    severity if is_injection else "medium",
                    attack_vector="sql_injection" if is_injection else None,
                    threat_indicators=indicators
                )
                
                # Send error response
                error_packet = self._create_error_packet(
                    1142, "42000", "SELECT command denied to user"
                )
                writer.write(error_packet)
                await writer.drain()
                
        except Exception as e:
            self.logger.debug(f"Error handling MySQL packet: {e}")
    
    def _create_error_packet(self, error_code: int, sql_state: str, message: str) -> bytes:
        """Create MySQL error packet"""
        error_data = (
            b'\xff' +  # Error packet marker
            struct.pack('<H', error_code) +  # Error code
            b'#' + sql_state.encode() +  # SQL state marker + state
            message.encode('utf-8')  # Error message
        )
        
        # Add packet header
        packet_length = len(error_data)
        header = struct.pack('<I', packet_length)[:-1] + b'\x01'  # 3 bytes length + seq
        
        return header + error_data
    
    async def _log_interaction(self, honeypot_id: str, source_ip: str, source_port: int,
                             dest_port: int, event_type: str, payload: str, severity: str,
                             **kwargs):
        """Log database interaction"""
        event_id = f"{honeypot_id}_{int(datetime.utcnow().timestamp())}_{random.randint(1000, 9999)}"
        
        event = HoneypotEvent(
            id=event_id,
            timestamp=datetime.utcnow(),
            honeypot_id=honeypot_id,
            honeypot_type=HoneypotType.DATABASE.value,
            source_ip=source_ip,
            source_port=source_port,
            destination_port=dest_port,
            event_type=event_type,
            payload=payload,
            severity=severity,
            **kwargs
        )
        
        # Analyze for threats
        analysis = await self.intelligence_engine.analyze_event(event)
        event.threat_indicators = analysis['indicators']
        
        # Store event
        self.database.store_event(event)
        
        self.logger.info(f"MySQL interaction: {event_type} from {source_ip}")

class PostgreSQLHoneypot(DatabaseProtocolHandler):
    """PostgreSQL protocol honeypot"""
    
    def __init__(self, database, intelligence_engine):
        super().__init__(database, intelligence_engine)
        self.server_version = "PostgreSQL 12.9 on x86_64-pc-linux-gnu"
        
    async def handle_connection(self, reader: asyncio.StreamReader, 
                              writer: asyncio.StreamWriter, honeypot_id: str, port: int):
        """Handle PostgreSQL connection"""
        client_addr = writer.get_extra_info('peername')
        if not client_addr:
            return
            
        source_ip, source_port = client_addr[0], client_addr[1]
        
        try:
            # Read startup message
            startup_data = await reader.read(4096)
            if startup_data:
                await self._handle_startup_message(
                    startup_data, writer, honeypot_id, source_ip, source_port, port
                )
                
                # Handle authentication and queries
                while True:
                    try:
                        message = await asyncio.wait_for(reader.read(4096), timeout=30)
                        if not message:
                            break
                            
                        await self._handle_postgresql_message(
                            message, writer, honeypot_id, source_ip, source_port, port
                        )
                        
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        self.logger.debug(f"Error handling PostgreSQL message: {e}")
                        break
            
        except Exception as e:
            self.logger.debug(f"PostgreSQL connection error from {source_ip}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _handle_startup_message(self, startup_data: bytes, writer: asyncio.StreamWriter,
                                    honeypot_id: str, source_ip: str, source_port: int, port: int):
        """Handle PostgreSQL startup message"""
        try:
            if len(startup_data) >= 8:
                # Parse startup message
                length = struct.unpack('>I', startup_data[:4])[0]
                protocol = struct.unpack('>I', startup_data[4:8])[0]
                
                # Extract parameters (simplified)
                params = {}
                pos = 8
                while pos < len(startup_data) - 1:
                    try:
                        key_end = startup_data.find(b'\x00', pos)
                        if key_end == -1:
                            break
                        key = startup_data[pos:key_end].decode('utf-8', errors='ignore')
                        
                        value_start = key_end + 1
                        value_end = startup_data.find(b'\x00', value_start)
                        if value_end == -1:
                            break
                        value = startup_data[value_start:value_end].decode('utf-8', errors='ignore')
                        
                        params[key] = value
                        pos = value_end + 1
                        
                    except Exception:
                        break
                
                user = params.get('user', 'unknown')
                database = params.get('database', 'unknown')
                
                await self._log_interaction(
                    honeypot_id, source_ip, source_port, port,
                    "postgresql_connection_attempt",
                    f"User: {user}, Database: {database}",
                    "medium",
                    credentials={"username": user, "database": database}
                )
                
                # Send authentication request (password required)
                auth_request = self._create_auth_request()
                writer.write(auth_request)
                await writer.drain()
                
        except Exception as e:
            self.logger.debug(f"Error handling PostgreSQL startup: {e}")
    
    async def _handle_postgresql_message(self, message: bytes, writer: asyncio.StreamWriter,
                                       honeypot_id: str, source_ip: str, source_port: int, port: int):
        """Handle PostgreSQL protocol message"""
        try:
            if len(message) < 5:
                return
            
            msg_type = chr(message[0])
            msg_length = struct.unpack('>I', message[1:5])[0]
            msg_data = message[5:5+msg_length-4]
            
            if msg_type == 'p':  # Password message
                password = msg_data.rstrip(b'\x00').decode('utf-8', errors='ignore')
                
                await self._log_interaction(
                    honeypot_id, source_ip, source_port, port,
                    "postgresql_auth_attempt",
                    f"Password attempt",
                    "high",
                    credentials={"password": password}
                )
                
                # Send authentication failure
                error_response = self._create_error_response(
                    "28P01", "password authentication failed"
                )
                writer.write(error_response)
                await writer.drain()
                
            elif msg_type == 'Q':  # Simple query
                query = msg_data.rstrip(b'\x00').decode('utf-8', errors='ignore')
                
                # Detect SQL injection
                is_injection, indicators, severity = self._detect_sql_injection(query)
                
                await self._log_interaction(
                    honeypot_id, source_ip, source_port, port,
                    "postgresql_query",
                    f"Query: {query[:200]}",
                    severity if is_injection else "medium",
                    attack_vector="sql_injection" if is_injection else None,
                    threat_indicators=indicators
                )
                
                # Send error response
                error_response = self._create_error_response(
                    "42501", "permission denied"
                )
                writer.write(error_response)
                await writer.drain()
                
        except Exception as e:
            self.logger.debug(f"Error handling PostgreSQL message: {e}")
    
    def _create_auth_request(self) -> bytes:
        """Create PostgreSQL authentication request"""
        # Authentication MD5 password request
        salt = random.randbytes(4)
        auth_data = struct.pack('>I', 5) + salt  # MD5 auth + salt
        message = b'R' + struct.pack('>I', 12) + auth_data
        return message
    
    def _create_error_response(self, code: str, message: str) -> bytes:
        """Create PostgreSQL error response"""
        error_data = (
            b'S' + b'ERROR\x00' +
            b'C' + code.encode() + b'\x00' +
            b'M' + message.encode() + b'\x00' +
            b'\x00'
        )
        
        length = len(error_data) + 4
        response = b'E' + struct.pack('>I', length) + error_data
        return response
    
    async def _log_interaction(self, honeypot_id: str, source_ip: str, source_port: int,
                             dest_port: int, event_type: str, payload: str, severity: str,
                             **kwargs):
        """Log database interaction"""
        event_id = f"{honeypot_id}_{int(datetime.utcnow().timestamp())}_{random.randint(1000, 9999)}"
        
        event = HoneypotEvent(
            id=event_id,
            timestamp=datetime.utcnow(),
            honeypot_id=honeypot_id,
            honeypot_type=HoneypotType.DATABASE.value,
            source_ip=source_ip,
            source_port=source_port,
            destination_port=dest_port,
            event_type=event_type,
            payload=payload,
            severity=severity,
            **kwargs
        )
        
        # Analyze for threats
        analysis = await self.intelligence_engine.analyze_event(event)
        event.threat_indicators = analysis['indicators']
        
        # Store event
        self.database.store_event(event)
        
        self.logger.info(f"PostgreSQL interaction: {event_type} from {source_ip}")

class RedisHoneypot(DatabaseProtocolHandler):
    """Redis protocol honeypot"""
    
    def __init__(self, database, intelligence_engine):
        super().__init__(database, intelligence_engine)
        self.server_version = "6.2.6"
        
    async def handle_connection(self, reader: asyncio.StreamReader, 
                              writer: asyncio.StreamWriter, honeypot_id: str, port: int):
        """Handle Redis connection"""
        client_addr = writer.get_extra_info('peername')
        if not client_addr:
            return
            
        source_ip, source_port = client_addr[0], client_addr[1]
        
        try:
            await self._log_interaction(
                honeypot_id, source_ip, source_port, port,
                "redis_connection_attempt",
                "Redis connection established",
                "medium"
            )
            
            # Handle Redis commands
            while True:
                try:
                    data = await asyncio.wait_for(reader.read(4096), timeout=30)
                    if not data:
                        break
                        
                    await self._handle_redis_command(
                        data, writer, honeypot_id, source_ip, source_port, port
                    )
                    
                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    self.logger.debug(f"Error handling Redis command: {e}")
                    break
            
        except Exception as e:
            self.logger.debug(f"Redis connection error from {source_ip}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _handle_redis_command(self, data: bytes, writer: asyncio.StreamWriter,
                                  honeypot_id: str, source_ip: str, source_port: int, port: int):
        """Handle Redis command"""
        try:
            command_str = data.decode('utf-8', errors='ignore').strip()
            
            # Parse Redis protocol (simplified)
            if command_str.startswith('*'):
                # RESP protocol
                lines = command_str.split('\r\n')
                if len(lines) >= 3:
                    command = lines[2].upper() if lines[2] else ''
                else:
                    command = 'UNKNOWN'
            else:
                # Simple command
                command = command_str.split()[0].upper() if command_str else 'UNKNOWN'
            
            # Detect malicious Redis commands
            severity = "low"
            attack_vector = None
            indicators = []
            
            dangerous_commands = [
                'FLUSHALL', 'FLUSHDB', 'CONFIG', 'EVAL', 'SCRIPT',
                'SHUTDOWN', 'DEBUG', 'MIGRATE', 'RESTORE', 'DUMP'
            ]
            
            if command in dangerous_commands:
                severity = "high"
                attack_vector = "redis_exploitation"
                indicators = ["dangerous_redis_command"]
            elif command in ['AUTH', 'SELECT']:
                severity = "medium"
                attack_vector = "redis_access_attempt"
            elif 'LUA' in command_str.upper() or 'SCRIPT' in command_str.upper():
                severity = "high"
                attack_vector = "redis_script_injection"
                indicators = ["script_injection"]
            
            await self._log_interaction(
                honeypot_id, source_ip, source_port, port,
                "redis_command",
                f"Command: {command_str[:200]}",
                severity,
                attack_vector=attack_vector,
                threat_indicators=indicators
            )
            
            # Send Redis error response
            if command == 'AUTH':
                response = b"-ERR invalid password\r\n"
            else:
                response = b"-ERR NOAUTH Authentication required\r\n"
            
            writer.write(response)
            await writer.drain()
            
        except Exception as e:
            self.logger.debug(f"Error handling Redis command: {e}")
    
    async def _log_interaction(self, honeypot_id: str, source_ip: str, source_port: int,
                             dest_port: int, event_type: str, payload: str, severity: str,
                             **kwargs):
        """Log database interaction"""
        event_id = f"{honeypot_id}_{int(datetime.utcnow().timestamp())}_{random.randint(1000, 9999)}"
        
        event = HoneypotEvent(
            id=event_id,
            timestamp=datetime.utcnow(),
            honeypot_id=honeypot_id,
            honeypot_type=HoneypotType.DATABASE.value,
            source_ip=source_ip,
            source_port=source_port,
            destination_port=dest_port,
            event_type=event_type,
            payload=payload,
            severity=severity,
            **kwargs
        )
        
        # Analyze for threats
        analysis = await self.intelligence_engine.analyze_event(event)
        event.threat_indicators = analysis['indicators']
        
        # Store event
        self.database.store_event(event)
        
        self.logger.info(f"Redis interaction: {event_type} from {source_ip}")

class DatabaseHoneypotServer:
    """Multi-protocol database honeypot server"""
    
    def __init__(self, honeypot_id: str, port: int, database, intelligence_engine, 
                 protocol: str = "mysql"):
        self.honeypot_id = honeypot_id
        self.port = port
        self.database = database
        self.intelligence_engine = intelligence_engine
        self.protocol = protocol.lower()
        self.server = None
        self.is_running = False
        
        # Initialize protocol handler
        if self.protocol == "mysql":
            self.handler = MySQLHoneypot(database, intelligence_engine)
        elif self.protocol == "postgresql":
            self.handler = PostgreSQLHoneypot(database, intelligence_engine)
        elif self.protocol == "redis":
            self.handler = RedisHoneypot(database, intelligence_engine)
        else:
            raise ValueError(f"Unsupported database protocol: {protocol}")
    
    async def start(self):
        """Start database honeypot server"""
        try:
            self.server = await asyncio.start_server(
                self._handle_connection,
                '0.0.0.0',
                self.port
            )
            
            self.is_running = True
            logger.info(f"{self.protocol.upper()} database honeypot started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start {self.protocol} honeypot: {e}")
            raise
    
    async def stop(self):
        """Stop database honeypot server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        self.is_running = False
        logger.info(f"{self.protocol.upper()} database honeypot stopped")
    
    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming database connection"""
        try:
            await self.handler.handle_connection(reader, writer, self.honeypot_id, self.port)
        except Exception as e:
            logger.debug(f"Error handling {self.protocol} connection: {e}")

class DatabaseHoneypotManager:
    """Manager for database honeypots"""
    
    def __init__(self, database, intelligence_engine):
        self.database = database
        self.intelligence_engine = intelligence_engine
        self.honeypots = {}
        
    async def deploy_mysql_honeypot(self, port: int = 3306) -> bool:
        """Deploy MySQL honeypot"""
        try:
            honeypot_id = f"db_mysql_{port}"
            
            honeypot = DatabaseHoneypotServer(
                honeypot_id, port, self.database, self.intelligence_engine, "mysql"
            )
            
            await honeypot.start()
            self.honeypots[honeypot_id] = honeypot
            
            logger.info(f"MySQL honeypot deployed on port {port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy MySQL honeypot: {e}")
            return False
    
    async def deploy_postgresql_honeypot(self, port: int = 5432) -> bool:
        """Deploy PostgreSQL honeypot"""
        try:
            honeypot_id = f"db_postgresql_{port}"
            
            honeypot = DatabaseHoneypotServer(
                honeypot_id, port, self.database, self.intelligence_engine, "postgresql"
            )
            
            await honeypot.start()
            self.honeypots[honeypot_id] = honeypot
            
            logger.info(f"PostgreSQL honeypot deployed on port {port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy PostgreSQL honeypot: {e}")
            return False
    
    async def deploy_redis_honeypot(self, port: int = 6379) -> bool:
        """Deploy Redis honeypot"""
        try:
            honeypot_id = f"db_redis_{port}"
            
            honeypot = DatabaseHoneypotServer(
                honeypot_id, port, self.database, self.intelligence_engine, "redis"
            )
            
            await honeypot.start()
            self.honeypots[honeypot_id] = honeypot
            
            logger.info(f"Redis honeypot deployed on port {port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy Redis honeypot: {e}")
            return False
    
    async def deploy_all_database_honeypots(self) -> Dict[str, bool]:
        """Deploy all database honeypots"""
        results = {}
        
        # Deploy on non-standard ports to avoid conflicts
        database_configs = [
            ("mysql", 13306),
            ("postgresql", 15432),
            ("redis", 16379)
        ]
        
        for db_type, port in database_configs:
            try:
                if db_type == "mysql":
                    results[db_type] = await self.deploy_mysql_honeypot(port)
                elif db_type == "postgresql":
                    results[db_type] = await self.deploy_postgresql_honeypot(port)
                elif db_type == "redis":
                    results[db_type] = await self.deploy_redis_honeypot(port)
            except Exception as e:
                logger.error(f"Failed to deploy {db_type} honeypot: {e}")
                results[db_type] = False
        
        return results
    
    async def stop_all(self):
        """Stop all database honeypots"""
        for honeypot in self.honeypots.values():
            try:
                await honeypot.stop()
            except Exception as e:
                logger.error(f"Error stopping database honeypot: {e}")
        
        self.honeypots.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all database honeypots"""
        return {
            "active_honeypots": len(self.honeypots),
            "honeypots": {
                honeypot_id: {
                    "protocol": honeypot.protocol,
                    "port": honeypot.port,
                    "running": honeypot.is_running
                }
                for honeypot_id, honeypot in self.honeypots.items()
            }
        }

# Global database honeypot manager instance
database_honeypot_manager = None