#!/usr/bin/env python3
"""
MCP Download Manager

Safe download handling with comprehensive validation, integrity checking, and
security measures for MCP server packages. Provides robust download operations
with retry logic, progress monitoring, and safety mechanisms.

Author: Claude AI Assistant (python-architect.md)
Created: 2025-08-15 11:30:00 UTC
Version: 1.0.0
"""

import asyncio
import aiohttp
import hashlib
import tempfile
import shutil
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from config import get_config, MCPAutomationConfig


class DownloadState(Enum):
    """Download states for tracking."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationResult(Enum):
    """Validation result types."""
    VALID = "valid"
    INVALID_CHECKSUM = "invalid_checksum"
    INVALID_SIZE = "invalid_size"
    INVALID_FORMAT = "invalid_format"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class DownloadProgress:
    """Download progress tracking."""
    total_bytes: int
    downloaded_bytes: int
    percentage: float
    speed_bps: float
    eta_seconds: float
    state: DownloadState
    
    @property
    def speed_mbps(self) -> float:
        """Get speed in MB/s."""
        return self.speed_bps / (1024 * 1024) if self.speed_bps > 0 else 0.0


@dataclass
class DownloadResult:
    """Download operation result."""
    success: bool
    file_path: Optional[Path]
    checksum: Optional[str]
    size_bytes: int
    download_time_seconds: float
    validation_result: ValidationResult
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DownloadManager:
    """
    Comprehensive download manager for MCP packages.
    
    Provides safe, validated downloads with progress monitoring, integrity
    checking, and security measures following organizational standards.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """
        Initialize download manager.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.logger = self._setup_logging()
        
        # Initialize session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Download tracking
        self.active_downloads: Dict[str, DownloadProgress] = {}
        self.download_history: List[Dict[str, Any]] = []
        
        # Security patterns
        self.suspicious_patterns = [
            r'\.\./',  # Path traversal
            r'[<>:"\\|?*]',  # Invalid filename characters
            r'^[.\-]',  # Hidden files or files starting with dash
            r'(^|/)\.{1,2}($|/)',  # Current/parent directory references
        ]
        
        self.logger.info("Download manager initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for download manager."""
        logger = logging.getLogger(f"{__name__}.DownloadManager")
        logger.setLevel(getattr(logging, self.config.log_level.value))
        
        if not logger.handlers:
            # Create logs directory
            self.config.paths.logs_root.mkdir(parents=True, exist_ok=True)
            
            # File handler for detailed logs
            log_file = self.config.paths.logs_root / f"download_manager_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_session()
    
    async def _initialize_session(self):
        """Initialize HTTP session with security settings."""
        try:
            timeout = aiohttp.ClientTimeout(
                total=self.config.performance.download_timeout_seconds,
                connect=30,
                sock_read=60
            )
            
            connector = aiohttp.TCPConnector(
                limit=self.config.performance.max_concurrent_downloads,
                limit_per_host=2,
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=True
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': 'SutazAI-MCP-Automation/1.0.0',
                    'Accept': 'application/octet-stream, application/json',
                    'Accept-Encoding': 'gzip, deflate'
                }
            )
            
            self.logger.debug("HTTP session initialized with security settings")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize HTTP session: {e}")
            raise
    
    async def _cleanup_session(self):
        """Cleanup HTTP session."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                # Wait for proper cleanup
                await asyncio.sleep(0.1)
            
            self.logger.debug("HTTP session cleaned up")
        
        except Exception as e:
            self.logger.warning(f"Error during session cleanup: {e}")
    
    async def download_package(self, server_name: str, package_name: str,
                             version: Optional[str] = None,
                             progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> DownloadResult:
        """
        Download MCP package with validation and safety checks.
        
        Args:
            server_name: Name of the MCP server
            package_name: NPM package name
            version: Specific version (latest if None)
            progress_callback: Optional progress callback function
            
        Returns:
            DownloadResult with success status and details
        """
        download_id = f"{server_name}_{package_name}_{version or 'latest'}_{int(datetime.now().timestamp())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting download: {package_name}@{version or 'latest'}")
            
            # Initialize session if not already done
            if not self.session:
                await self._initialize_session()
            
            # Get package metadata
            package_info = await self._get_package_info(package_name, version)
            if not package_info:
                raise ValueError(f"Package not found: {package_name}")
            
            # Validate package security
            security_check = await self._validate_package_security(package_info)
            if security_check != ValidationResult.VALID:
                raise SecurityError(f"Security validation failed: {security_check.value}")
            
            # Download package
            download_url = package_info['dist']['tarball']
            expected_checksum = package_info['dist'].get('shasum', '')
            expected_size = package_info['dist'].get('unpackedSize', 0)
            
            # Create progress tracker
            progress = DownloadProgress(
                total_bytes=expected_size,
                downloaded_bytes=0,
                percentage=0.0,
                speed_bps=0.0,
                eta_seconds=0.0,
                state=DownloadState.DOWNLOADING
            )
            
            self.active_downloads[download_id] = progress
            
            # Download to temporary file
            temp_file = await self._download_file(
                download_url, 
                expected_size,
                download_id,
                progress_callback
            )
            
            # Update progress state
            progress.state = DownloadState.VALIDATING
            if progress_callback:
                progress_callback(progress)
            
            # Validate downloaded file
            validation_result = await self._validate_download(
                temp_file, 
                expected_checksum, 
                expected_size
            )
            
            if validation_result != ValidationResult.VALID:
                temp_file.unlink(missing_ok=True)
                raise ValidationError(f"Download validation failed: {validation_result.value}")
            
            # Calculate final checksum
            final_checksum = await self._calculate_checksum(temp_file)
            final_size = temp_file.stat().st_size
            
            # Move to final location
            final_path = self._get_download_path(server_name, package_name, version or package_info['version'])
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temp_file, final_path)
            
            # Update progress state
            progress.state = DownloadState.COMPLETED
            progress.percentage = 100.0
            if progress_callback:
                progress_callback(progress)
            
            # Calculate download time
            download_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Record success
            result = DownloadResult(
                success=True,
                file_path=final_path,
                checksum=final_checksum,
                size_bytes=final_size,
                download_time_seconds=download_time,
                validation_result=ValidationResult.VALID,
                metadata={
                    'package_info': package_info,
                    'download_url': download_url,
                    'server_name': server_name
                }
            )
            
            # Record in history
            await self._record_download(download_id, result, package_info)
            
            self.logger.info(f"Successfully downloaded {package_name}@{package_info['version']} in {download_time:.2f}s")
            
            return result
        
        except Exception as e:
            # Cleanup on failure
            if download_id in self.active_downloads:
                self.active_downloads[download_id].state = DownloadState.FAILED
            
            download_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = DownloadResult(
                success=False,
                file_path=None,
                checksum=None,
                size_bytes=0,
                download_time_seconds=download_time,
                validation_result=ValidationResult.SECURITY_VIOLATION if isinstance(e, SecurityError) else ValidationResult.INVALID_FORMAT,
                error_message=str(e),
                metadata={'server_name': server_name, 'package_name': package_name}
            )
            
            self.logger.error(f"Failed to download {package_name}: {e}")
            
            return result
        
        finally:
            # Cleanup tracking
            if download_id in self.active_downloads:
                del self.active_downloads[download_id]
    
    async def _get_package_info(self, package_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get package information from NPM registry."""
        try:
            # Validate package name
            if not self._is_valid_package_name(package_name):
                raise ValueError(f"Invalid package name: {package_name}")
            
            # Check if registry is allowed
            registry_url = "https://registry.npmjs.org"
            if registry_url not in self.config.security.allowed_registries:
                raise SecurityError(f"Registry not allowed: {registry_url}")
            
            # Build URL
            if version:
                url = f"{registry_url}/{package_name}/{version}"
            else:
                url = f"{registry_url}/{package_name}/latest"
            
            # Request package info
            async with self.session.get(url) as response:
                if response.status == 200:
                    package_info = await response.json()
                    
                    # Additional security validation
                    if not self._validate_package_metadata(package_info):
                        raise SecurityError("Package metadata validation failed")
                    
                    return package_info
                elif response.status == 404:
                    return None
                else:
                    raise RuntimeError(f"NPM registry error: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Failed to get package info for {package_name}: {e}")
            if isinstance(e, (SecurityError, ValueError, RuntimeError)):
                raise
            return None
    
    async def _download_file(self, url: str, expected_size: int, 
                           download_id: str,
                           progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> Path:
        """Download file with progress tracking."""
        try:
            # Create temporary file
            temp_file = Path(tempfile.mktemp(suffix='.mcp_download'))
            
            async with self.session.get(url) as response:
                response.raise_for_status()
                
                # Validate content length
                content_length = int(response.headers.get('Content-Length', 0))
                if content_length > self.config.security.max_download_size_mb * 1024 * 1024:
                    raise SecurityError(f"File too large: {content_length} bytes")
                
                # Initialize progress tracking
                progress = self.active_downloads[download_id]
                progress.total_bytes = content_length or expected_size
                
                downloaded = 0
                chunk_size = 8192
                start_time = datetime.now()
                
                with open(temp_file, 'wb') as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed > 0:
                            speed = downloaded / elapsed
                            eta = (progress.total_bytes - downloaded) / speed if speed > 0 else 0
                            
                            progress.downloaded_bytes = downloaded
                            progress.percentage = (downloaded / progress.total_bytes * 100) if progress.total_bytes > 0 else 0
                            progress.speed_bps = speed
                            progress.eta_seconds = eta
                            
                            if progress_callback:
                                progress_callback(progress)
                        
                        # Check for cancellation
                        if progress.state == DownloadState.CANCELLED:
                            temp_file.unlink(missing_ok=True)
                            raise asyncio.CancelledError("Download cancelled")
            
            return temp_file
        
        except Exception as e:
            # Cleanup temp file on error
            if 'temp_file' in locals():
                temp_file.unlink(missing_ok=True)
            raise
    
    async def _validate_download(self, file_path: Path, expected_checksum: str, 
                                expected_size: int) -> ValidationResult:
        """Validate downloaded file."""
        try:
            # Check file exists
            if not file_path.exists():
                return ValidationResult.INVALID_FORMAT
            
            # Check file size
            actual_size = file_path.stat().st_size
            if expected_size > 0 and abs(actual_size - expected_size) > 1024:  # Allow 1KB tolerance
                self.logger.warning(f"Size mismatch: expected {expected_size}, got {actual_size}")
                return ValidationResult.INVALID_SIZE
            
            # Check size limits
            if actual_size > self.config.security.max_download_size_mb * 1024 * 1024:
                return ValidationResult.SECURITY_VIOLATION
            
            # Verify checksum if provided
            if expected_checksum and self.config.security.verify_checksums:
                actual_checksum = await self._calculate_checksum(file_path)
                if actual_checksum != expected_checksum:
                    self.logger.error(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
                    return ValidationResult.INVALID_CHECKSUM
            
            # Basic file format validation (check if it's a valid gzipped tarball)
            if not await self._validate_tarball_format(file_path):
                return ValidationResult.INVALID_FORMAT
            
            return ValidationResult.VALID
        
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResult.INVALID_FORMAT
    
    async def _validate_package_security(self, package_info: Dict[str, Any]) -> ValidationResult:
        """Validate package for security issues."""
        try:
            # Check package name
            package_name = package_info.get('name', '')
            if not self._is_valid_package_name(package_name):
                return ValidationResult.SECURITY_VIOLATION
            
            # Check for suspicious patterns in name
            for pattern in self.suspicious_patterns:
                if re.search(pattern, package_name):
                    self.logger.warning(f"Suspicious pattern in package name: {pattern}")
                    return ValidationResult.SECURITY_VIOLATION
            
            # Validate required fields
            required_fields = ['name', 'version', 'dist']
            for field in required_fields:
                if field not in package_info:
                    return ValidationResult.INVALID_FORMAT
            
            # Validate dist information
            dist = package_info.get('dist', {})
            if 'tarball' not in dist:
                return ValidationResult.INVALID_FORMAT
            
            # Check tarball URL
            tarball_url = dist['tarball']
            if not self._is_valid_download_url(tarball_url):
                return ValidationResult.SECURITY_VIOLATION
            
            return ValidationResult.VALID
        
        except Exception as e:
            self.logger.error(f"Security validation error: {e}")
            return ValidationResult.SECURITY_VIOLATION
    
    def _is_valid_package_name(self, name: str) -> bool:
        """Validate NPM package name."""
        # NPM package name rules
        if not name or len(name) > 214:
            return False
        
        if name.startswith('.') or name.startswith('_'):
            return False
        
        # Check for valid characters
        valid_chars = re.compile(r'^[a-z0-9@/_-]+$')
        if not valid_chars.match(name.lower()):
            return False
        
        # Check for scoped packages (should start with @)
        if name.startswith('@'):
            parts = name.split('/')
            if len(parts) != 2:
                return False
        
        return True
    
    def _is_valid_download_url(self, url: str) -> bool:
        """Validate download URL for security."""
        # Must be HTTPS
        if not url.startswith('https://'):
            return False
        
        # Must be from allowed registries
        for registry in self.config.security.allowed_registries:
            if url.startswith(registry):
                return True
        
        return False
    
    def _validate_package_metadata(self, package_info: Dict[str, Any]) -> bool:
        """Validate package metadata for suspicious content."""
        try:
            # Check for required fields
            if 'name' not in package_info or 'version' not in package_info:
                return False
            
            # Validate version format
            version = package_info.get('version', '')
            version_pattern = re.compile(r'^\d+\.\d+\.\d+.*$')
            if not version_pattern.match(version):
                return False
            
            return True
        
        except Exception:
            return False
    
    async def _validate_tarball_format(self, file_path: Path) -> bool:
        """Basic validation that file is a gzipped tarball."""
        try:
            # Check file signature
            with open(file_path, 'rb') as f:
                header = f.read(3)
                # Check for gzip magic number
                if header[:2] != b'\\x1f\\x8b':
                    return False
            
            return True
        
        except Exception:
            return False
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA1 checksum (NPM uses SHA1)."""
        try:
            hash_sha1 = hashlib.sha1()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha1.update(chunk)
            
            return hash_sha1.hexdigest()
        
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def _get_download_path(self, server_name: str, package_name: str, version: str) -> Path:
        """Get final download path for package."""
        downloads_dir = self.config.paths.automation_root / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', package_name)
        safe_version = re.sub(r'[^a-zA-Z0-9._-]', '_', version)
        
        filename = f"{safe_name}-{safe_version}.tgz"
        return downloads_dir / filename
    
    async def _record_download(self, download_id: str, result: DownloadResult, 
                              package_info: Dict[str, Any]):
        """Record download in history."""
        try:
            record = {
                'download_id': download_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'package_name': package_info.get('name'),
                'version': package_info.get('version'),
                'success': result.success,
                'file_path': str(result.file_path) if result.file_path else None,
                'size_bytes': result.size_bytes,
                'download_time_seconds': result.download_time_seconds,
                'validation_result': result.validation_result.value,
                'error_message': result.error_message
            }
            
            self.download_history.append(record)
            
            # Keep only last 100 records
            if len(self.download_history) > 100:
                self.download_history = self.download_history[-100:]
            
            # Save to file
            history_file = self.config.paths.automation_root / "download_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.download_history, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to record download history: {e}")
    
    def get_download_progress(self, download_id: str) -> Optional[DownloadProgress]:
        """Get current download progress."""
        return self.active_downloads.get(download_id)
    
    def get_active_downloads(self) -> Dict[str, DownloadProgress]:
        """Get all active downloads."""
        return self.active_downloads.copy()
    
    def cancel_download(self, download_id: str) -> bool:
        """Cancel an active download."""
        if download_id in self.active_downloads:
            self.active_downloads[download_id].state = DownloadState.CANCELLED
            return True
        return False


class SecurityError(Exception):
    """Security validation error."""
    pass


class ValidationError(Exception):
    """Download validation error."""
    pass


if __name__ == "__main__":
    # Download manager testing
    async def test_download():
        config = get_config()
        
        async with DownloadManager(config) as dm:
            # Test download progress callback
            def progress_callback(progress: DownloadProgress):
                logger.info(f"Progress: {progress.percentage:.1f}% ({progress.speed_mbps:.2f} MB/s, ETA: {progress.eta_seconds:.0f}s)")
            
            # Test downloading a small MCP package
            result = await dm.download_package(
                server_name="files",
                package_name="@modelcontextprotocol/server-filesystem",
                progress_callback=progress_callback
            )
            
            logger.info(f"Download result: {result.success}")
            if result.success:
                logger.info(f"  File: {result.file_path}")
                logger.info(f"  Size: {result.size_bytes} bytes")
                logger.info(f"  Time: {result.download_time_seconds:.2f}s")
                logger.info(f"  Checksum: {result.checksum}")
            else:
                logger.error(f"  Error: {result.error_message}")
    
    # Run test
    asyncio.run(test_download())