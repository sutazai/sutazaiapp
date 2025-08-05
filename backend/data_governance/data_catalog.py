"""
Data Catalog
============

Comprehensive data catalog system that provides discovery, metadata management,
and searchable inventory of all data assets in the SutazAI system.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import uuid
import hashlib

from .data_classifier import DataClassification, DataType


class AssetType(Enum):
    """Types of data assets"""
    TABLE = "table"
    VIEW = "view"
    FILE = "file"
    API = "api"
    STREAM = "stream"
    MODEL = "model"
    REPORT = "report"
    DASHBOARD = "dashboard"


class AssetStatus(Enum):
    """Status of data assets"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    UNDER_DEVELOPMENT = "under_development"
    DECOMMISSIONED = "decommissioned"


@dataclass
class DataSchema:
    """Schema definition for a data asset"""
    fields: List[Dict[str, Any]] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataAssetMetadata:
    """Comprehensive metadata for a data asset"""
    id: str
    name: str
    description: str
    asset_type: AssetType
    
    # Classification and governance
    classification: DataClassification = DataClassification.INTERNAL
    data_types: Set[DataType] = field(default_factory=set)
    sensitivity_level: str = "medium"
    
    # Location and access
    source_system: str = ""
    database_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    file_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    
    # Schema and structure
    schema: Optional[DataSchema] = None
    sample_data: Optional[List[Dict[str, Any]]] = None
    
    # Ownership and stewardship
    owner: Optional[str] = None
    steward: Optional[str] = None
    business_contact: Optional[str] = None
    technical_contact: Optional[str] = None
    
    # Lifecycle and status
    status: AssetStatus = AssetStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    
    # Usage and quality
    access_count: int = 0
    quality_score: Optional[float] = None
    availability_percentage: Optional[float] = None
    
    # Business context
    business_purpose: Optional[str] = None
    business_rules: List[str] = field(default_factory=list)
    data_lineage_upstream: List[str] = field(default_factory=list)
    data_lineage_downstream: List[str] = field(default_factory=list)
    
    # Technical details
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    update_frequency: Optional[str] = None
    retention_policy: Optional[str] = None
    
    # Tags and categorization
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Additional metadata
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate ID if not provided"""
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique asset ID"""
        components = [
            self.source_system,
            self.database_name or "",
            self.schema_name or "",
            self.table_name or self.name
        ]
        content = "|".join(str(c) for c in components)
        return hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class SearchQuery:
    """Search query for data catalog"""
    text: Optional[str] = None
    asset_types: Optional[List[AssetType]] = None
    classifications: Optional[List[DataClassification]] = None
    data_types: Optional[List[DataType]] = None
    owners: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    source_systems: Optional[List[str]] = None
    status: Optional[List[AssetStatus]] = None
    
    # Quality filters
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None
    
    # Time filters
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None
    
    # Sorting and pagination
    sort_by: str = "name"
    sort_order: str = "asc"  # asc or desc
    limit: int = 50
    offset: int = 0


@dataclass
class SearchResult:
    """Search result from data catalog"""
    assets: List[DataAssetMetadata]
    total_count: int
    query: SearchQuery
    search_time_ms: int
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)


class DataCatalog:
    """
    Comprehensive data catalog that provides discovery, metadata management,
    and search capabilities for all data assets in the SutazAI system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("data_catalog")
        
        # Asset storage
        self.assets: Dict[str, DataAssetMetadata] = {}
        
        # Search indexes (simplified - would use Elasticsearch in production)
        self.text_index: Dict[str, Set[str]] = {}  # term -> asset_ids
        self.facet_indexes: Dict[str, Dict[str, Set[str]]] = {
            "asset_type": {},
            "classification": {},
            "data_type": {},
            "owner": {},
            "source_system": {},
            "status": {},
            "tags": {},
            "categories": {}
        }
        
        # Configuration
        self.auto_discovery_enabled = self.config.get('auto_discovery_enabled', True)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        self.max_sample_records = self.config.get('max_sample_records', 5)
        
        # Statistics
        self.stats = {
            "total_assets": 0,
            "searches_performed": 0,
            "assets_accessed": 0,
            "auto_discovered_assets": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the data catalog"""
        try:
            self.logger.info("Initializing data catalog")
            
            # Start auto-discovery if enabled
            if self.auto_discovery_enabled:
                await self._start_auto_discovery()
            
            self.logger.info("Data catalog initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data catalog: {e}")
            return False
    
    async def register_asset(self, asset: DataAssetMetadata) -> bool:
        """Register a new data asset in the catalog"""
        try:
            # Update timestamps
            asset.updated_at = datetime.utcnow()
            
            # Store asset
            self.assets[asset.id] = asset
            
            # Update search indexes
            await self._update_search_indexes(asset)
            
            # Update statistics
            self.stats["total_assets"] = len(self.assets)
            
            self.logger.info(f"Registered data asset: {asset.name} ({asset.id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register asset {asset.id}: {e}")
            return False
    
    async def update_asset(self, asset_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing data asset"""
        try:
            asset = self.assets.get(asset_id)
            if not asset:
                self.logger.warning(f"Asset not found for update: {asset_id}")
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(asset, key):
                    setattr(asset, key, value)
            
            asset.updated_at = datetime.utcnow()
            
            # Update search indexes
            await self._update_search_indexes(asset)
            
            self.logger.info(f"Updated data asset: {asset_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update asset {asset_id}: {e}")
            return False
    
    async def delete_asset(self, asset_id: str) -> bool:
        """Remove a data asset from the catalog"""
        try:
            asset = self.assets.get(asset_id)
            if not asset:
                return False
            
            # Remove from indexes
            await self._remove_from_search_indexes(asset)
            
            # Remove from storage
            del self.assets[asset_id]
            
            # Update statistics
            self.stats["total_assets"] = len(self.assets)
            
            self.logger.info(f"Deleted data asset: {asset_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete asset {asset_id}: {e}")
            return False
    
    async def get_asset(self, asset_id: str) -> Optional[DataAssetMetadata]:
        """Get a specific data asset by ID"""
        asset = self.assets.get(asset_id)
        if asset:
            # Update access statistics
            asset.access_count += 1
            asset.last_accessed = datetime.utcnow()
            self.stats["assets_accessed"] += 1
        return asset
    
    async def search_assets(self, query: SearchQuery) -> SearchResult:
        """Search for data assets based on query criteria"""
        start_time = datetime.utcnow()
        
        try:
            # Find matching assets
            matching_assets = await self._execute_search(query)
            
            # Apply sorting
            matching_assets = self._sort_assets(matching_assets, query.sort_by, query.sort_order)
            
            # Apply pagination
            total_count = len(matching_assets)
            paginated_assets = matching_assets[query.offset:query.offset + query.limit]
            
            # Calculate facets
            facets = self._calculate_facets(matching_assets)
            
            # Calculate search time
            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = SearchResult(
                assets=paginated_assets,
                total_count=total_count,
                query=query,
                search_time_ms=int(search_time),
                facets=facets
            )
            
            self.stats["searches_performed"] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in asset search: {e}")
            return SearchResult(
                assets=[],
                total_count=0,
                query=query,
                search_time_ms=0
            )
    
    async def _execute_search(self, query: SearchQuery) -> List[DataAssetMetadata]:
        """Execute search query and return matching assets"""
        candidate_assets = set(self.assets.keys())
        
        # Text search
        if query.text:
            text_matches = self._search_text(query.text)
            candidate_assets = candidate_assets.intersection(text_matches)
        
        # Filter by asset types
        if query.asset_types:
            type_matches = set()
            for asset_type in query.asset_types:
                type_matches.update(self.facet_indexes["asset_type"].get(asset_type.value, set()))
            candidate_assets = candidate_assets.intersection(type_matches)
        
        # Filter by classifications
        if query.classifications:
            classification_matches = set()
            for classification in query.classifications:
                classification_matches.update(self.facet_indexes["classification"].get(classification.value, set()))
            candidate_assets = candidate_assets.intersection(classification_matches)
        
        # Filter by data types
        if query.data_types:
            data_type_matches = set()
            for data_type in query.data_types:
                data_type_matches.update(self.facet_indexes["data_type"].get(data_type.value, set()))
            candidate_assets = candidate_assets.intersection(data_type_matches)
        
        # Filter by owners
        if query.owners:
            owner_matches = set()
            for owner in query.owners:
                owner_matches.update(self.facet_indexes["owner"].get(owner, set()))
            candidate_assets = candidate_assets.intersection(owner_matches)
        
        # Filter by tags
        if query.tags:
            tag_matches = set()
            for tag in query.tags:
                tag_matches.update(self.facet_indexes["tags"].get(tag, set()))
            candidate_assets = candidate_assets.intersection(tag_matches)
        
        # Filter by categories
        if query.categories:
            category_matches = set()
            for category in query.categories:
                category_matches.update(self.facet_indexes["categories"].get(category, set()))
            candidate_assets = candidate_assets.intersection(category_matches)
        
        # Filter by source systems
        if query.source_systems:
            system_matches = set()
            for system in query.source_systems:
                system_matches.update(self.facet_indexes["source_system"].get(system, set()))
            candidate_assets = candidate_assets.intersection(system_matches)
        
        # Filter by status
        if query.status:
            status_matches = set()
            for status in query.status:
                status_matches.update(self.facet_indexes["status"].get(status.value, set()))
            candidate_assets = candidate_assets.intersection(status_matches)
        
        # Apply additional filters
        filtered_assets = []
        for asset_id in candidate_assets:
            asset = self.assets[asset_id]
            
            # Quality score filter
            if query.min_quality_score is not None:
                if asset.quality_score is None or asset.quality_score < query.min_quality_score:
                    continue
            
            if query.max_quality_score is not None:
                if asset.quality_score is None or asset.quality_score > query.max_quality_score:
                    continue
            
            # Time filters
            if query.created_after and asset.created_at < query.created_after:
                continue
            
            if query.created_before and asset.created_at > query.created_before:
                continue
            
            if query.updated_after and asset.updated_at < query.updated_after:
                continue
            
            if query.updated_before and asset.updated_at > query.updated_before:
                continue
            
            filtered_assets.append(asset)
        
        return filtered_assets
    
    def _search_text(self, text: str) -> Set[str]:
        """Search for text across asset metadata"""
        matching_assets = set()
        search_terms = text.lower().split()
        
        for term in search_terms:
            # Exact matches
            if term in self.text_index:
                matching_assets.update(self.text_index[term])
            
            # Partial matches
            for indexed_term, asset_ids in self.text_index.items():
                if term in indexed_term or indexed_term in term:
                    matching_assets.update(asset_ids)
        
        return matching_assets
    
    def _sort_assets(self, assets: List[DataAssetMetadata], sort_by: str, sort_order: str) -> List[DataAssetMetadata]:
        """Sort assets based on specified criteria"""
        reverse = sort_order.lower() == "desc"
        
        if sort_by == "name":
            return sorted(assets, key=lambda x: x.name.lower(), reverse=reverse)
        elif sort_by == "created_at":
            return sorted(assets, key=lambda x: x.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            return sorted(assets, key=lambda x: x.updated_at, reverse=reverse)
        elif sort_by == "access_count":
            return sorted(assets, key=lambda x: x.access_count, reverse=reverse)
        elif sort_by == "quality_score":
            return sorted(assets, key=lambda x: x.quality_score or 0, reverse=reverse)
        else:
            return assets
    
    def _calculate_facets(self, assets: List[DataAssetMetadata]) -> Dict[str, Dict[str, int]]:
        """Calculate facet counts for search results"""
        facets = {
            "asset_type": {},
            "classification": {},
            "data_type": {},
            "owner": {},
            "source_system": {},
            "status": {},
            "tags": {},
            "categories": {}
        }
        
        for asset in assets:
            # Asset type
            asset_type = asset.asset_type.value
            facets["asset_type"][asset_type] = facets["asset_type"].get(asset_type, 0) + 1
            
            # Classification
            classification = asset.classification.value
            facets["classification"][classification] = facets["classification"].get(classification, 0) + 1
            
            # Data types
            for data_type in asset.data_types:
                dt_value = data_type.value
                facets["data_type"][dt_value] = facets["data_type"].get(dt_value, 0) + 1
            
            # Owner
            if asset.owner:
                facets["owner"][asset.owner] = facets["owner"].get(asset.owner, 0) + 1
            
            # Source system
            if asset.source_system:
                facets["source_system"][asset.source_system] = facets["source_system"].get(asset.source_system, 0) + 1
            
            # Status
            status = asset.status.value
            facets["status"][status] = facets["status"].get(status, 0) + 1
            
            # Tags
            for tag in asset.tags:
                facets["tags"][tag] = facets["tags"].get(tag, 0) + 1
            
            # Categories
            for category in asset.categories:
                facets["categories"][category] = facets["categories"].get(category, 0) + 1
        
        return facets
    
    async def _update_search_indexes(self, asset: DataAssetMetadata):
        """Update search indexes for an asset"""
        
        # Text index
        searchable_text = [
            asset.name,
            asset.description,
            asset.business_purpose or "",
            asset.source_system,
            asset.database_name or "",
            asset.schema_name or "",
            asset.table_name or ""
        ]
        
        # Add all text content to index
        for text in searchable_text:
            if text:
                words = text.lower().split()
                for word in words:
                    if word not in self.text_index:
                        self.text_index[word] = set()
                    self.text_index[word].add(asset.id)
        
        # Facet indexes
        # Asset type
        asset_type = asset.asset_type.value
        if asset_type not in self.facet_indexes["asset_type"]:
            self.facet_indexes["asset_type"][asset_type] = set()
        self.facet_indexes["asset_type"][asset_type].add(asset.id)
        
        # Classification
        classification = asset.classification.value
        if classification not in self.facet_indexes["classification"]:
            self.facet_indexes["classification"][classification] = set()
        self.facet_indexes["classification"][classification].add(asset.id)
        
        # Data types
        for data_type in asset.data_types:
            dt_value = data_type.value
            if dt_value not in self.facet_indexes["data_type"]:
                self.facet_indexes["data_type"][dt_value] = set()
            self.facet_indexes["data_type"][dt_value].add(asset.id)
        
        # Owner
        if asset.owner:
            if asset.owner not in self.facet_indexes["owner"]:
                self.facet_indexes["owner"][asset.owner] = set()
            self.facet_indexes["owner"][asset.owner].add(asset.id)
        
        # Source system
        if asset.source_system:
            if asset.source_system not in self.facet_indexes["source_system"]:
                self.facet_indexes["source_system"][asset.source_system] = set()
            self.facet_indexes["source_system"][asset.source_system].add(asset.id)
        
        # Status
        status = asset.status.value
        if status not in self.facet_indexes["status"]:
            self.facet_indexes["status"][status] = set()
        self.facet_indexes["status"][status].add(asset.id)
        
        # Tags
        for tag in asset.tags:
            if tag not in self.facet_indexes["tags"]:
                self.facet_indexes["tags"][tag] = set()
            self.facet_indexes["tags"][tag].add(asset.id)
        
        # Categories
        for category in asset.categories:
            if category not in self.facet_indexes["categories"]:
                self.facet_indexes["categories"][category] = set()
            self.facet_indexes["categories"][category].add(asset.id)
    
    async def _remove_from_search_indexes(self, asset: DataAssetMetadata):
        """Remove asset from all search indexes"""
        
        # Remove from text index
        for term_assets in self.text_index.values():
            term_assets.discard(asset.id)
        
        # Remove from facet indexes
        for facet_type, facet_index in self.facet_indexes.items():
            for facet_value, asset_ids in facet_index.items():
                asset_ids.discard(asset.id)
    
    async def _start_auto_discovery(self):
        """Start automatic asset discovery process"""
        self.logger.info("Starting automatic asset discovery")
        
        # This would implement discovery from various sources:
        # - Database schemas
        # - File systems
        # - API endpoints
        # - Configuration files
        
        # For now, just log that it's enabled
        self.logger.info("Auto-discovery enabled - will discover assets from registered sources")
    
    async def discover_database_assets(self, connection_config: Dict[str, Any]) -> List[str]:
        """Discover assets from a database"""
        discovered_assets = []
        
        try:
            # This would connect to the database and discover tables/views
            # For now, create a sample asset
            
            database_name = connection_config.get('database', 'unknown_db')
            
            # Sample table discovery
            sample_asset = DataAssetMetadata(
                id="",  # Will be auto-generated
                name=f"sample_table_{database_name}",
                description=f"Auto-discovered table from {database_name}",
                asset_type=AssetType.TABLE,
                source_system="postgresql",
                database_name=database_name,
                schema_name="public",
                table_name="sample_table",
                status=AssetStatus.ACTIVE,
                tags=["auto-discovered", "database"],
                categories=["operational_data"]
            )
            
            success = await self.register_asset(sample_asset)
            if success:
                discovered_assets.append(sample_asset.id)
                self.stats["auto_discovered_assets"] += 1
            
            self.logger.info(f"Discovered {len(discovered_assets)} assets from database {database_name}")
            
        except Exception as e:
            self.logger.error(f"Error discovering database assets: {e}")
        
        return discovered_assets
    
    async def discover_file_assets(self, file_path: str) -> List[str]:
        """Discover assets from file system"""
        discovered_assets = []
        
        try:
            # This would scan the file system for data files
            # For now, create a sample file asset
            
            sample_asset = DataAssetMetadata(
                id="",  # Will be auto-generated
                name=f"sample_file",
                description=f"Auto-discovered file from {file_path}",
                asset_type=AssetType.FILE,
                source_system="filesystem",
                file_path=file_path,
                status=AssetStatus.ACTIVE,
                tags=["auto-discovered", "file"],
                categories=["raw_data"]
            )
            
            success = await self.register_asset(sample_asset)
            if success:
                discovered_assets.append(sample_asset.id)
                self.stats["auto_discovered_assets"] += 1
            
            self.logger.info(f"Discovered {len(discovered_assets)} assets from path {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error discovering file assets: {e}")
        
        return discovered_assets
    
    def get_catalog_statistics(self) -> Dict[str, Any]:
        """Get comprehensive catalog statistics"""
        stats = {
            "overview": self.stats.copy(),
            "assets_by_type": {},
            "assets_by_classification": {},
            "assets_by_status": {},
            "assets_by_source_system": {},
            "top_accessed_assets": [],
            "recent_assets": [],
            "quality_distribution": {
                "high_quality": 0,  # > 0.8
                "medium_quality": 0,  # 0.5 - 0.8
                "low_quality": 0,  # < 0.5
                "unknown_quality": 0  # None
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Analyze assets
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        
        for asset in self.assets.values():
            # Count by type
            asset_type = asset.asset_type.value
            stats["assets_by_type"][asset_type] = stats["assets_by_type"].get(asset_type, 0) + 1
            
            # Count by classification
            classification = asset.classification.value
            stats["assets_by_classification"][classification] = stats["assets_by_classification"].get(classification, 0) + 1
            
            # Count by status
            status = asset.status.value
            stats["assets_by_status"][status] = stats["assets_by_status"].get(status, 0) + 1
            
            # Count by source system
            system = asset.source_system or "unknown"
            stats["assets_by_source_system"][system] = stats["assets_by_source_system"].get(system, 0) + 1
            
            # Quality distribution
            if asset.quality_score is None:
                stats["quality_distribution"]["unknown_quality"] += 1
            elif asset.quality_score > 0.8:
                stats["quality_distribution"]["high_quality"] += 1
            elif asset.quality_score >= 0.5:
                stats["quality_distribution"]["medium_quality"] += 1
            else:
                stats["quality_distribution"]["low_quality"] += 1
            
            # Recent assets
            if asset.created_at >= recent_cutoff:
                stats["recent_assets"].append({
                    "id": asset.id,
                    "name": asset.name,
                    "type": asset.asset_type.value,
                    "created_at": asset.created_at.isoformat()
                })
        
        # Top accessed assets
        sorted_assets = sorted(self.assets.values(), key=lambda x: x.access_count, reverse=True)
        stats["top_accessed_assets"] = [
            {
                "id": asset.id,
                "name": asset.name,
                "type": asset.asset_type.value,
                "access_count": asset.access_count,
                "last_accessed": asset.last_accessed.isoformat() if asset.last_accessed else None
            }
            for asset in sorted_assets[:10]
        ]
        
        # Sort recent assets by date
        stats["recent_assets"].sort(key=lambda x: x["created_at"], reverse=True)
        stats["recent_assets"] = stats["recent_assets"][:20]  # Top 20
        
        return stats
    
    async def get_asset_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get personalized asset recommendations for a user"""
        recommendations = []
        
        try:
            # This would implement a recommendation algorithm based on:
            # - User's previous access patterns
            # - Assets similar to ones they've used
            # - Popular assets in their domain
            # - Recently updated high-quality assets
            
            # For now, return top-quality, recently updated assets
            candidate_assets = [
                asset for asset in self.assets.values()
                if asset.status == AssetStatus.ACTIVE and
                asset.quality_score and asset.quality_score > self.quality_threshold
            ]
            
            # Sort by combination of quality and recency
            scored_assets = []
            for asset in candidate_assets:
                # Simple scoring: quality * recency_factor
                days_since_update = (datetime.utcnow() - asset.updated_at).days
                recency_factor = max(0.1, 1.0 - (days_since_update / 365))  # Decay over a year
                score = asset.quality_score * recency_factor
                
                scored_assets.append((score, asset))
            
            # Sort by score and take top N
            scored_assets.sort(key=lambda x: x[0], reverse=True)
            
            for score, asset in scored_assets[:limit]:
                recommendations.append({
                    "asset_id": asset.id,
                    "name": asset.name,
                    "type": asset.asset_type.value,
                    "description": asset.description,
                    "quality_score": asset.quality_score,
                    "recommendation_score": score,
                    "reason": "High quality and recently updated"
                })
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations for user {user_id}: {e}")
        
        return recommendations
    
    async def shutdown(self):
        """Shutdown the data catalog"""
        try:
            self.logger.info("Shutting down data catalog")
            # Any cleanup tasks would go here
            self.logger.info("Data catalog shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during data catalog shutdown: {e}")
    
    # Convenience methods
    
    async def quick_add_table(self, database: str, schema: str, table: str,
                            description: str = "", owner: str = "") -> str:
        """Quickly add a database table to the catalog"""
        asset = DataAssetMetadata(
            id="",  # Will be auto-generated
            name=f"{schema}.{table}",
            description=description or f"Table {table} in schema {schema}",
            asset_type=AssetType.TABLE,
            source_system="postgresql",  # Default assumption
            database_name=database,
            schema_name=schema,
            table_name=table,
            owner=owner,
            status=AssetStatus.ACTIVE,
            tags=["database", "table"],
            categories=["structured_data"]
        )
        
        success = await self.register_asset(asset)
        return asset.id if success else ""
    
    async def quick_add_api(self, endpoint: str, service: str,
                          description: str = "", owner: str = "") -> str:
        """Quickly add an API endpoint to the catalog"""
        asset = DataAssetMetadata(
            id="",  # Will be auto-generated
            name=endpoint.split('/')[-1] or "api_endpoint",
            description=description or f"API endpoint {endpoint}",
            asset_type=AssetType.API,
            source_system=service,
            api_endpoint=endpoint,
            owner=owner,
            status=AssetStatus.ACTIVE,
            tags=["api", "endpoint"],
            categories=["api_data"]
        )
        
        success = await self.register_asset(asset)
        return asset.id if success else ""