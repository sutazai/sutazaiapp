"""
Data Versioning System
=====================

Comprehensive data versioning system that tracks changes, maintains history,
and enables time-travel queries across all data assets in the SutazAI system.
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid


class VersioningStrategy(Enum):
    """Data versioning strategies"""
    SNAPSHOT = "snapshot"          # Full data snapshots
    DELTA = "delta"               # Store only changes
    HYBRID = "hybrid"             # Combination of snapshots and deltas
    IMMUTABLE = "immutable"       # Append-only with immutable records


class ChangeType(Enum):
    """Types of data changes"""
    INSERT = "insert"
    UPDATE = "update" 
    DELETE = "delete"
    SCHEMA_CHANGE = "schema_change"
    BULK_LOAD = "bulk_load"
    MIGRATION = "migration"


@dataclass
class DataVersion:
    """Represents a version of a data asset"""
    version_id: str
    asset_id: str
    version_number: str
    
    # Version metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    change_type: ChangeType = ChangeType.UPDATE
    description: Optional[str] = None
    
    # Data content
    data_hash: Optional[str] = None
    data_size_bytes: Optional[int] = None
    record_count: Optional[int] = None
    
    # Change tracking
    changes_summary: Dict[str, Any] = field(default_factory=dict)
    affected_columns: List[str] = field(default_factory=list)
    affected_records: Optional[int] = None
    
    # Parent version tracking
    parent_version_id: Optional[str] = None
    branch_name: str = "main"
    
    # Storage information
    storage_location: Optional[str] = None
    compression_type: Optional[str] = None
    encryption_enabled: bool = False
    
    # Quality and validation
    quality_score: Optional[float] = None
    validation_status: str = "pending"  # pending, passed, failed
    validation_errors: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate version ID if not provided"""
        if not self.version_id:
            self.version_id = str(uuid.uuid4())


@dataclass
class VersionDiff:
    """Represents differences between two versions"""
    from_version_id: str
    to_version_id: str
    
    # Change statistics
    records_added: int = 0
    records_modified: int = 0
    records_deleted: int = 0
    columns_added: List[str] = field(default_factory=list)
    columns_removed: List[str] = field(default_factory=list)
    columns_modified: List[str] = field(default_factory=list)
    
    # Detailed changes
    added_records: List[Dict[str, Any]] = field(default_factory=list)
    modified_records: List[Dict[str, Any]] = field(default_factory=list)
    deleted_records: List[Dict[str, Any]] = field(default_factory=list)
    
    # Schema changes
    schema_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Diff metadata
    diff_timestamp: datetime = field(default_factory=datetime.utcnow)
    diff_size_bytes: Optional[int] = None


@dataclass
class VersionBranch:
    """Represents a version branch"""
    branch_name: str
    asset_id: str
    
    # Branch metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    description: Optional[str] = None
    
    # Branch state
    head_version_id: Optional[str] = None
    parent_branch: Optional[str] = None
    is_active: bool = True
    is_protected: bool = False
    
    # Branch statistics
    version_count: int = 0
    last_commit_at: Optional[datetime] = None
    
    # Merge information
    merged_to: Optional[str] = None
    merged_at: Optional[datetime] = None
    merge_commit_id: Optional[str] = None


class DataVersioning:
    """
    Comprehensive data versioning system that provides version control
    capabilities for data assets including branching, merging, and time-travel.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("data_versioning")
        
        # Version storage
        self.versions: Dict[str, DataVersion] = {}  # version_id -> version
        self.asset_versions: Dict[str, List[str]] = {}  # asset_id -> [version_ids]
        self.branches: Dict[str, Dict[str, VersionBranch]] = {}  # asset_id -> {branch_name -> branch}
        
        # Configuration
        self.default_strategy = VersioningStrategy(self.config.get('default_strategy', 'hybrid'))
        self.max_versions_per_asset = self.config.get('max_versions_per_asset', 100)
        self.auto_cleanup_enabled = self.config.get('auto_cleanup_enabled', True)
        self.retention_days = self.config.get('retention_days', 365)
        
        # Storage configuration
        self.storage_config = self.config.get('storage', {})
        self.compression_enabled = self.storage_config.get('compression_enabled', True)
        self.encryption_enabled = self.storage_config.get('encryption_enabled', False)
        
        # Statistics
        self.stats = {
            "total_versions": 0,
            "total_branches": 0,
            "assets_versioned": 0,
            "storage_used_bytes": 0,
            "time_travel_queries": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the versioning system"""
        try:
            self.logger.info("Initializing data versioning system")
            
            # Start background cleanup if enabled
            if self.auto_cleanup_enabled:
                await self._start_background_cleanup()
            
            self.logger.info(f"Data versioning initialized with strategy: {self.default_strategy.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data versioning: {e}")
            return False
    
    async def create_version(self, asset_id: str, data_content: Any,
                           change_type: ChangeType = ChangeType.UPDATE,
                           description: Optional[str] = None,
                           created_by: Optional[str] = None,
                           branch_name: str = "main") -> Optional[DataVersion]:
        """Create a new version of a data asset"""
        try:
            # Get or create branch
            branch = await self._get_or_create_branch(asset_id, branch_name, created_by)
            
            # Generate version number
            version_number = await self._generate_version_number(asset_id, branch_name)
            
            # Calculate data hash
            data_hash = self._calculate_data_hash(data_content)
            
            # Create version
            version = DataVersion(
                version_id="",  # Will be auto-generated
                asset_id=asset_id,
                version_number=version_number,
                created_by=created_by,
                change_type=change_type,
                description=description,
                data_hash=data_hash,
                branch_name=branch_name,
                parent_version_id=branch.head_version_id
            )
            
            # Calculate data statistics
            if isinstance(data_content, (list, dict)):
                version.data_size_bytes = len(json.dumps(data_content).encode())
                if isinstance(data_content, list):
                    version.record_count = len(data_content)
                else:
                    version.record_count = 1
            elif isinstance(data_content, str):
                version.data_size_bytes = len(data_content.encode())
                version.record_count = len(data_content.split('\n'))
            
            # Store version data
            await self._store_version_data(version, data_content)
            
            # Register version
            self.versions[version.version_id] = version
            
            if asset_id not in self.asset_versions:
                self.asset_versions[asset_id] = []
            self.asset_versions[asset_id].append(version.version_id)
            
            # Update branch
            branch.head_version_id = version.version_id
            branch.version_count += 1
            branch.last_commit_at = version.created_at
            
            # Update statistics
            self.stats["total_versions"] += 1
            if asset_id not in [v.asset_id for v in self.versions.values() if v.version_id != version.version_id]:
                self.stats["assets_versioned"] += 1
            if version.data_size_bytes:
                self.stats["storage_used_bytes"] += version.data_size_bytes
            
            self.logger.info(f"Created version {version.version_number} for asset {asset_id}")
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to create version for asset {asset_id}: {e}")
            return None
    
    async def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get a specific version by ID"""
        return self.versions.get(version_id)
    
    async def get_latest_version(self, asset_id: str, branch_name: str = "main") -> Optional[DataVersion]:
        """Get the latest version of an asset on a specific branch"""
        try:
            branches = self.branches.get(asset_id, {})
            branch = branches.get(branch_name)
            
            if branch and branch.head_version_id:
                return self.versions.get(branch.head_version_id)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest version for asset {asset_id}: {e}")
            return None
    
    async def get_version_history(self, asset_id: str, branch_name: str = "main",
                                limit: Optional[int] = None) -> List[DataVersion]:
        """Get version history for an asset"""
        try:
            versions = []
            
            # Get all versions for the asset on the specified branch
            asset_version_ids = self.asset_versions.get(asset_id, [])
            
            for version_id in asset_version_ids:
                version = self.versions.get(version_id)
                if version and version.branch_name == branch_name:
                    versions.append(version)
            
            # Sort by creation time (newest first)
            versions.sort(key=lambda x: x.created_at, reverse=True)
            
            # Apply limit
            if limit:
                versions = versions[:limit]
            
            return versions
            
        except Exception as e:
            self.logger.error(f"Error getting version history for asset {asset_id}: {e}")
            # Return empty list on error - version history unavailable
            return []  # Valid empty list: Version history retrieval error
    
    async def get_version_at_time(self, asset_id: str, timestamp: datetime,
                                branch_name: str = "main") -> Optional[DataVersion]:
        """Get the version that was current at a specific timestamp (time-travel query)"""
        try:
            self.stats["time_travel_queries"] += 1
            
            # Get all versions for the asset on the branch
            versions = await self.get_version_history(asset_id, branch_name)
            
            # Find the latest version created before or at the timestamp
            for version in versions:
                if version.created_at <= timestamp:
                    return version
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting version at time for asset {asset_id}: {e}")
            return None
    
    async def get_version_data(self, version_id: str) -> Optional[Any]:
        """Retrieve the actual data content for a version"""
        try:
            version = self.versions.get(version_id)
            if not version:
                return None
            
            # Load data from storage
            data = await self._load_version_data(version)
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data for version {version_id}: {e}")
            return None
    
    async def compare_versions(self, from_version_id: str, to_version_id: str) -> Optional[VersionDiff]:
        """Compare two versions and return the differences"""
        try:
            from_version = self.versions.get(from_version_id)
            to_version = self.versions.get(to_version_id)
            
            if not from_version or not to_version:
                return None
            
            # Load data for both versions
            from_data = await self.get_version_data(from_version_id)
            to_data = await self.get_version_data(to_version_id)
            
            if from_data is None or to_data is None:
                return None
            
            # Calculate differences
            diff = await self._calculate_version_diff(from_data, to_data, from_version_id, to_version_id)
            
            return diff
            
        except Exception as e:
            self.logger.error(f"Error comparing versions {from_version_id} and {to_version_id}: {e}")
            return None
    
    async def create_branch(self, asset_id: str, branch_name: str,
                          source_branch: str = "main",
                          created_by: Optional[str] = None,
                          description: Optional[str] = None) -> Optional[VersionBranch]:
        """Create a new branch from an existing branch"""
        try:
            # Check if branch already exists
            if asset_id in self.branches and branch_name in self.branches[asset_id]:
                self.logger.warning(f"Branch {branch_name} already exists for asset {asset_id}")
                return None
            
            # Get source branch
            source_branch_obj = None
            if asset_id in self.branches and source_branch in self.branches[asset_id]:
                source_branch_obj = self.branches[asset_id][source_branch]
            
            # Create new branch
            branch = VersionBranch(
                branch_name=branch_name,
                asset_id=asset_id,
                created_by=created_by,
                description=description,
                parent_branch=source_branch,
                head_version_id=source_branch_obj.head_version_id if source_branch_obj else None
            )
            
            # Store branch
            if asset_id not in self.branches:
                self.branches[asset_id] = {}
            self.branches[asset_id][branch_name] = branch
            
            self.stats["total_branches"] += 1
            
            self.logger.info(f"Created branch {branch_name} for asset {asset_id}")
            return branch
            
        except Exception as e:
            self.logger.error(f"Error creating branch {branch_name} for asset {asset_id}: {e}")
            return None
    
    async def merge_branch(self, asset_id: str, source_branch: str, target_branch: str,
                         merged_by: Optional[str] = None,
                         merge_strategy: str = "auto") -> bool:
        """Merge one branch into another"""
        try:
            # Get branches
            branches = self.branches.get(asset_id, {})
            source = branches.get(source_branch)
            target = branches.get(target_branch)
            
            if not source or not target:
                self.logger.error(f"Source or target branch not found for merge")
                return False
            
            # Get latest versions from both branches
            source_version = await self.get_latest_version(asset_id, source_branch)
            target_version = await self.get_latest_version(asset_id, target_branch)
            
            if not source_version:
                self.logger.warning(f"No versions found in source branch {source_branch}")
                return False
            
            # Simple merge strategy: create a new version in target branch with source data
            source_data = await self.get_version_data(source_version.version_id)
            if source_data is None:
                return False
            
            # Create merge commit in target branch
            merge_version = await self.create_version(
                asset_id=asset_id,
                data_content=source_data,
                change_type=ChangeType.UPDATE,
                description=f"Merge {source_branch} into {target_branch}",
                created_by=merged_by,
                branch_name=target_branch
            )
            
            if merge_version:
                # Mark source branch as merged
                source.merged_to = target_branch
                source.merged_at = datetime.utcnow()
                source.merge_commit_id = merge_version.version_id
                source.is_active = False
                
                self.logger.info(f"Successfully merged {source_branch} into {target_branch}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error merging branches: {e}")
            return False
    
    async def delete_version(self, version_id: str) -> bool:
        """Delete a specific version (use with caution)"""
        try:
            version = self.versions.get(version_id)
            if not version:
                return False
            
            # Remove from storage
            await self._delete_version_data(version)
            
            # Remove from indexes
            if version.asset_id in self.asset_versions:
                if version_id in self.asset_versions[version.asset_id]:
                    self.asset_versions[version.asset_id].remove(version_id)
            
            # Remove from versions
            del self.versions[version_id]
            
            # Update statistics
            self.stats["total_versions"] -= 1
            if version.data_size_bytes:
                self.stats["storage_used_bytes"] -= version.data_size_bytes
            
            self.logger.info(f"Deleted version {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting version {version_id}: {e}")
            return False
    
    async def cleanup_old_versions(self, asset_id: str, keep_count: int = 10) -> int:
        """Clean up old versions, keeping only the most recent ones"""
        try:
            versions = await self.get_version_history(asset_id)
            
            if len(versions) <= keep_count:
                return 0
            
            # Delete old versions
            versions_to_delete = versions[keep_count:]
            deleted_count = 0
            
            for version in versions_to_delete:
                success = await self.delete_version(version.version_id)
                if success:
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old versions for asset {asset_id}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up versions for asset {asset_id}: {e}")
            return 0
    
    async def _get_or_create_branch(self, asset_id: str, branch_name: str,
                                  created_by: Optional[str] = None) -> VersionBranch:
        """Get existing branch or create if it doesn't exist"""
        if asset_id not in self.branches:
            self.branches[asset_id] = {}
        
        if branch_name not in self.branches[asset_id]:
            branch = VersionBranch(
                branch_name=branch_name,
                asset_id=asset_id,
                created_by=created_by
            )
            self.branches[asset_id][branch_name] = branch
            self.stats["total_branches"] += 1
        
        return self.branches[asset_id][branch_name]
    
    async def _generate_version_number(self, asset_id: str, branch_name: str) -> str:
        """Generate the next version number for an asset"""
        try:
            versions = await self.get_version_history(asset_id, branch_name)
            
            if not versions:
                return "1.0.0"
            
            # Extract version numbers and find the highest
            version_numbers = []
            for version in versions:
                parts = version.version_number.split('.')
                if len(parts) == 3:
                    try:
                        major, minor, patch = map(int, parts)
                        version_numbers.append((major, minor, patch))
                    except ValueError:
                        continue
            
            if not version_numbers:
                return "1.0.0"
            
            # Get highest version and increment patch
            highest = max(version_numbers)
            new_patch = highest[2] + 1
            
            return f"{highest[0]}.{highest[1]}.{new_patch}"
            
        except Exception as e:
            self.logger.error(f"Error generating version number: {e}")
            return f"1.0.{len(await self.get_version_history(asset_id, branch_name))}"
    
    def _calculate_data_hash(self, data_content: Any) -> str:
        """Calculate hash of data content for integrity verification"""
        try:
            if isinstance(data_content, (dict, list)):
                content_str = json.dumps(data_content, sort_keys=True)
            else:
                content_str = str(data_content)
            
            return hashlib.sha256(content_str.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error calculating data hash: {e}")
            return ""
    
    async def _store_version_data(self, version: DataVersion, data_content: Any):
        """Store version data to persistent storage"""
        try:
            # In production, this would store to a distributed storage system
            # For now, we'll simulate storage
            storage_path = f"/data/versions/{version.asset_id}/{version.version_id}"
            version.storage_location = storage_path
            
            # Apply compression if enabled
            if self.compression_enabled:
                version.compression_type = "gzip"
            
            # Apply encryption if enabled
            if self.encryption_enabled:
                version.encryption_enabled = True
            
            self.logger.debug(f"Stored version data at {storage_path}")
            
        except Exception as e:
            self.logger.error(f"Error storing version data: {e}")
    
    async def _load_version_data(self, version: DataVersion) -> Optional[Any]:
        """Load version data from storage"""
        try:
            # In production, this would load from the actual storage system
            # For now, return a placeholder
            return {"status": "simulated_data", "version_id": version.version_id}
            
        except Exception as e:
            self.logger.error(f"Error loading version data: {e}")
            return None
    
    async def _delete_version_data(self, version: DataVersion):
        """Delete version data from storage"""
        try:
            # In production, this would delete from the actual storage system
            self.logger.debug(f"Deleted version data for {version.version_id}")
            
        except Exception as e:
            self.logger.error(f"Error deleting version data: {e}")
    
    async def _calculate_version_diff(self, from_data: Any, to_data: Any,
                                    from_version_id: str, to_version_id: str) -> VersionDiff:
        """Calculate differences between two versions"""
        
        diff = VersionDiff(
            from_version_id=from_version_id,
            to_version_id=to_version_id
        )
        
        try:
            # Simple diff calculation for different data types
            if isinstance(from_data, list) and isinstance(to_data, list):
                # List comparison
                from_set = set(json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item) 
                             for item in from_data)
                to_set = set(json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item) 
                           for item in to_data)
                
                added = to_set - from_set
                removed = from_set - to_set
                
                diff.records_added = len(added)
                diff.records_deleted = len(removed)
                
                # Sample of changes (limited for performance)
                if added:
                    diff.added_records = list(added)[:10]
                if removed:
                    diff.deleted_records = list(removed)[:10]
            
            elif isinstance(from_data, dict) and isinstance(to_data, dict):
                # Dictionary comparison
                from_keys = set(from_data.keys())
                to_keys = set(to_data.keys())
                
                diff.columns_added = list(to_keys - from_keys)
                diff.columns_removed = list(from_keys - to_keys)
                
                # Check for modified values
                common_keys = from_keys & to_keys
                modified_keys = []
                for key in common_keys:
                    if from_data[key] != to_data[key]:
                        modified_keys.append(key)
                
                diff.columns_modified = modified_keys
                diff.records_modified = len(modified_keys)
            
            # Calculate diff size (approximate)
            diff_content = {
                "added": diff.added_records,
                "modified": diff.modified_records,
                "deleted": diff.deleted_records,
                "schema_changes": diff.schema_changes
            }
            diff.diff_size_bytes = len(json.dumps(diff_content).encode())
            
        except Exception as e:
            self.logger.error(f"Error calculating version diff: {e}")
        
        return diff
    
    async def _start_background_cleanup(self):
        """Start background cleanup process"""
        self.logger.info("Started background version cleanup process")
        
        async def cleanup_task():
            while True:
                try:
                    # Run cleanup every hour
                    await asyncio.sleep(3600)
                    
                    # Clean up old versions based on retention policy
                    cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
                    
                    versions_to_delete = []
                    for version in self.versions.values():
                        if version.created_at < cutoff_date:
                            versions_to_delete.append(version.version_id)
                    
                    # Delete old versions
                    for version_id in versions_to_delete:
                        await self.delete_version(version_id)
                    
                    if versions_to_delete:
                        self.logger.info(f"Cleaned up {len(versions_to_delete)} old versions")
                
                except Exception as e:
                    self.logger.error(f"Error in background cleanup: {e}")
        
        asyncio.create_task(cleanup_task())
    
    def get_versioning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive versioning statistics"""
        stats = {
            "overview": self.stats.copy(),
            "versions_by_asset": {},
            "branches_by_asset": {},
            "storage_by_asset": {},
            "version_activity": {
                "recent_versions": [],
                "most_versioned_assets": [],
                "branch_activity": []
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Analyze versions
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        asset_version_counts = {}
        asset_storage_usage = {}
        
        for version in self.versions.values():
            asset_id = version.asset_id
            
            # Count versions per asset
            asset_version_counts[asset_id] = asset_version_counts.get(asset_id, 0) + 1
            
            # Calculate storage per asset
            storage_bytes = version.data_size_bytes or 0
            asset_storage_usage[asset_id] = asset_storage_usage.get(asset_id, 0) + storage_bytes
            
            # Recent versions
            if version.created_at >= recent_cutoff:
                stats["version_activity"]["recent_versions"].append({
                    "version_id": version.version_id,
                    "asset_id": version.asset_id,
                    "version_number": version.version_number,
                    "change_type": version.change_type.value,
                    "created_at": version.created_at.isoformat(),
                    "created_by": version.created_by
                })
        
        stats["versions_by_asset"] = asset_version_counts
        stats["storage_by_asset"] = asset_storage_usage
        
        # Most versioned assets
        most_versioned = sorted(asset_version_counts.items(), key=lambda x: x[1], reverse=True)
        stats["version_activity"]["most_versioned_assets"] = [
            {"asset_id": asset_id, "version_count": count}
            for asset_id, count in most_versioned[:10]
        ]
        
        # Branch statistics
        for asset_id, branches in self.branches.items():
            stats["branches_by_asset"][asset_id] = {}
            for branch_name, branch in branches.items():
                stats["branches_by_asset"][asset_id][branch_name] = {
                    "version_count": branch.version_count,
                    "is_active": branch.is_active,
                    "last_commit": branch.last_commit_at.isoformat() if branch.last_commit_at else None
                }
                
                # Branch activity
                if branch.last_commit_at and branch.last_commit_at >= recent_cutoff:
                    stats["version_activity"]["branch_activity"].append({
                        "asset_id": asset_id,
                        "branch_name": branch_name,
                        "last_commit": branch.last_commit_at.isoformat(),
                        "version_count": branch.version_count
                    })
        
        # Sort recent activity
        stats["version_activity"]["recent_versions"].sort(
            key=lambda x: x["created_at"], reverse=True
        )
        stats["version_activity"]["recent_versions"] = stats["version_activity"]["recent_versions"][:20]
        
        return stats
    
    async def shutdown(self):
        """Shutdown the versioning system"""
        try:
            self.logger.info("Shutting down data versioning system")
            # Any cleanup tasks would go here
            self.logger.info("Data versioning system shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during versioning system shutdown: {e}")