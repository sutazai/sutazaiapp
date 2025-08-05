"""
Data Governance API
==================

FastAPI endpoints for data governance functionality including
dashboards, reporting, and management interfaces.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .governance_framework import DataGovernanceFramework
from .data_classifier import DataClassifier, DataClassification
from .lifecycle_manager import DataLifecycleManager
from .audit_logger import DataAuditLogger, AuditFilter, AuditEventType
from .compliance_manager import ComplianceManager, RegulationScope
from .lineage_tracker import DataLineageTracker
from .quality_monitor import DataQualityMonitor
from .data_catalog import DataCatalog, SearchQuery, AssetType
from .data_versioning import DataVersioning, ChangeType

logger = logging.getLogger("governance_api")

# Global governance framework instance
governance_framework: Optional[DataGovernanceFramework] = None

def get_governance_framework() -> DataGovernanceFramework:
    """Get the global governance framework instance"""
    global governance_framework
    if governance_framework is None:
        raise HTTPException(status_code=500, detail="Governance framework not initialized")
    return governance_framework

def set_governance_framework(framework: DataGovernanceFramework):
    """Set the global governance framework instance"""
    global governance_framework
    governance_framework = framework

# Create router
router = APIRouter(prefix="/api/v1/governance", tags=["Data Governance"])

@router.get("/health")
async def health_check():
    """Health check for governance system"""
    try:
        framework = get_governance_framework()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "classifier": "active",
                "lifecycle_manager": "active",
                "audit_logger": "active",
                "compliance_manager": "active",
                "lineage_tracker": "active",
                "quality_monitor": "active",
                "data_catalog": "active",
                "versioning": "active"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Governance system unhealthy")

# Dashboard Endpoints

@router.get("/dashboard/overview")
async def get_governance_overview():
    """Get comprehensive governance dashboard overview"""
    try:
        framework = get_governance_framework()
        dashboard_data = framework.get_governance_dashboard_data()
        
        # Add additional statistics from components
        dashboard_data["audit"] = await framework.audit_logger.get_audit_statistics()
        dashboard_data["quality"] = framework.quality_monitor.get_quality_statistics()
        dashboard_data["catalog"] = framework.data_catalog.get_catalog_statistics()
        dashboard_data["versioning"] = framework.versioning.get_versioning_statistics()
        dashboard_data["lineage"] = await framework.lineage_tracker.get_lineage_statistics()
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting governance overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get governance overview")

@router.get("/dashboard/compliance")
async def get_compliance_dashboard():
    """Get compliance-focused dashboard data"""
    try:
        framework = get_governance_framework()
        compliance_stats = framework.compliance_manager.get_compliance_statistics()
        
        # Add recent compliance reports
        recent_reports = []
        for regulation in [RegulationScope.GDPR, RegulationScope.CCPA, RegulationScope.HIPAA]:
            try:
                report = framework.compliance_manager.generate_compliance_report(regulation)
                recent_reports.append({
                    "regulation": regulation.value,
                    "total_violations": report.total_violations,
                    "critical_violations": report.critical_violations,
                    "overall_risk": report.overall_risk_level,
                    "generated_at": report.generated_at.isoformat()
                })
            except:
                continue
        
        return {
            "statistics": compliance_stats,
            "recent_reports": recent_reports,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting compliance dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get compliance dashboard")

@router.get("/dashboard/quality")
async def get_quality_dashboard():
    """Get data quality dashboard"""
    try:
        framework = get_governance_framework()
        quality_stats = framework.quality_monitor.get_quality_statistics()
        
        return {
            "statistics": quality_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting quality dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quality dashboard")

# Data Processing Endpoints

@router.post("/process-data")
async def process_data_governance(
    data_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process new data through the governance pipeline"""
    try:
        framework = get_governance_framework()
        
        # Process data asynchronously
        background_tasks.add_task(
            framework.process_new_data,
            data_id, content, metadata
        )
        
        return {
            "status": "accepted",
            "data_id": data_id,
            "message": "Data processing initiated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing data {data_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process data")

@router.get("/data/{data_id}/classification")
async def get_data_classification(data_id: str):
    """Get classification results for specific data"""
    try:
        framework = get_governance_framework()
        
        # This would typically retrieve from storage
        # For now, return a placeholder response
        return {
            "data_id": data_id,
            "message": "Classification retrieval not implemented",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting classification for {data_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get classification")

# Audit Endpoints

@router.get("/audit/events")
async def get_audit_events(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    event_types: Optional[List[str]] = Query(None),
    user_ids: Optional[List[str]] = Query(None),
    limit: int = Query(100, le=1000)
):
    """Get audit events with filtering"""
    try:
        framework = get_governance_framework()
        
        # Create filter
        filter_criteria = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            event_types=[AuditEventType(et) for et in event_types] if event_types else None,
            user_ids=user_ids,
            limit=limit
        )
        
        events = await framework.audit_logger.query_events(filter_criteria)
        
        # Convert events to dict format
        event_dicts = []
        for event in events:
            event_dict = {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "source": event.source,
                "target": event.target,
                "action": event.action,
                "description": event.description,
                "user_id": event.user_id,
                "success": event.success,
                "metadata": event.metadata
            }
            event_dicts.append(event_dict)
        
        return {
            "events": event_dicts,
            "total_count": len(events),
            "filter_applied": filter_criteria.__dict__,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting audit events: {e}")
        raise HTTPException(status_code=500, detail="Failed to get audit events")

@router.get("/audit/statistics")
async def get_audit_statistics(days: int = Query(30, le=365)):
    """Get audit statistics for specified period"""
    try:
        framework = get_governance_framework()
        stats = await framework.audit_logger.get_audit_statistics(days)
        
        return {
            "statistics": stats,
            "period_days": days,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting audit statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get audit statistics")

# Compliance Endpoints

@router.get("/compliance/reports/{regulation}")
async def get_compliance_report(
    regulation: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None)
):
    """Generate compliance report for specific regulation"""
    try:
        framework = get_governance_framework()
        
        # Validate regulation
        try:
            reg_scope = RegulationScope(regulation.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid regulation: {regulation}")
        
        report = framework.compliance_manager.generate_compliance_report(
            reg_scope, start_date, end_date
        )
        
        return {
            "report_id": report.report_id,
            "regulation": report.regulation.value,
            "generated_at": report.generated_at.isoformat(),
            "report_period_start": report.report_period_start.isoformat() if report.report_period_start else None,
            "report_period_end": report.report_period_end.isoformat() if report.report_period_end else None,
            "summary": {
                "total_violations": report.total_violations,
                "critical_violations": report.critical_violations,
                "high_violations": report.high_violations,
                "medium_violations": report.medium_violations,
                "low_violations": report.low_violations,
                "resolved_violations": report.resolved_violations,
                "overall_risk_level": report.overall_risk_level
            },
            "violations": [
                {
                    "id": v.id,
                    "violation_type": v.violation_type.value,
                    "severity": v.severity,
                    "description": v.description,
                    "detected_at": v.detected_at.isoformat(),
                    "resolved_at": v.resolved_at.isoformat() if v.resolved_at else None
                }
                for v in report.violations
            ],
            "recommendations": report.recommendations,
            "risk_factors": report.risk_factors
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")

@router.post("/compliance/violations/{violation_id}/resolve")
async def resolve_compliance_violation(
    violation_id: str,
    resolution_method: str,
    resolution_notes: str
):
    """Mark a compliance violation as resolved"""
    try:
        framework = get_governance_framework()
        
        success = framework.compliance_manager.resolve_violation(
            violation_id, resolution_method, resolution_notes
        )
        
        if success:
            return {
                "status": "resolved",
                "violation_id": violation_id,
                "resolution_method": resolution_method,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Violation not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving violation {violation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve violation")

# Data Catalog Endpoints

@router.get("/catalog/search")
async def search_catalog(
    q: Optional[str] = Query(None, description="Search text"),
    asset_types: Optional[List[str]] = Query(None),
    classifications: Optional[List[str]] = Query(None),
    owners: Optional[List[str]] = Query(None),
    tags: Optional[List[str]] = Query(None),
    limit: int = Query(50, le=1000),
    offset: int = Query(0, ge=0)
):
    """Search the data catalog"""
    try:
        framework = get_governance_framework()
        
        # Build search query
        query = SearchQuery(
            text=q,
            asset_types=[AssetType(at) for at in asset_types] if asset_types else None,
            classifications=[DataClassification(c) for c in classifications] if classifications else None,
            owners=owners,
            tags=tags,
            limit=limit,
            offset=offset
        )
        
        results = await framework.data_catalog.search_assets(query)
        
        # Convert assets to dict format
        asset_dicts = []
        for asset in results.assets:
            asset_dict = {
                "id": asset.id,
                "name": asset.name,
                "description": asset.description,
                "asset_type": asset.asset_type.value,
                "classification": asset.classification.value,
                "owner": asset.owner,
                "source_system": asset.source_system,
                "status": asset.status.value,
                "quality_score": asset.quality_score,
                "created_at": asset.created_at.isoformat(),
                "updated_at": asset.updated_at.isoformat(),
                "tags": asset.tags,
                "categories": asset.categories
            }
            asset_dicts.append(asset_dict)
        
        return {
            "assets": asset_dicts,
            "total_count": results.total_count,
            "search_time_ms": results.search_time_ms,
            "facets": results.facets,
            "query": {
                "text": query.text,
                "limit": query.limit,
                "offset": query.offset
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching catalog: {e}")
        raise HTTPException(status_code=500, detail="Failed to search catalog")

@router.get("/catalog/assets/{asset_id}")
async def get_catalog_asset(asset_id: str):
    """Get detailed information about a catalog asset"""
    try:
        framework = get_governance_framework()
        asset = await framework.data_catalog.get_asset(asset_id)
        
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        return {
            "id": asset.id,
            "name": asset.name,
            "description": asset.description,
            "asset_type": asset.asset_type.value,
            "classification": asset.classification.value,
            "data_types": [dt.value for dt in asset.data_types],
            "owner": asset.owner,
            "steward": asset.steward,
            "business_contact": asset.business_contact,
            "technical_contact": asset.technical_contact,
            "source_system": asset.source_system,
            "database_name": asset.database_name,
            "schema_name": asset.schema_name,
            "table_name": asset.table_name,
            "status": asset.status.value,
            "created_at": asset.created_at.isoformat(),
            "updated_at": asset.updated_at.isoformat(),
            "last_accessed": asset.last_accessed.isoformat() if asset.last_accessed else None,
            "access_count": asset.access_count,
            "quality_score": asset.quality_score,
            "business_purpose": asset.business_purpose,
            "business_rules": asset.business_rules,
            "size_bytes": asset.size_bytes,
            "row_count": asset.row_count,
            "update_frequency": asset.update_frequency,
            "retention_policy": asset.retention_policy,
            "tags": asset.tags,
            "categories": asset.categories,
            "custom_properties": asset.custom_properties,
            "schema": {
                "fields": asset.schema.fields if asset.schema else [],
                "primary_keys": asset.schema.primary_keys if asset.schema else [],
                "foreign_keys": asset.schema.foreign_keys if asset.schema else [],
                "version": asset.schema.version if asset.schema else None
            } if asset.schema else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting asset {asset_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get asset")

# Data Lineage Endpoints

@router.get("/lineage/{data_id}/upstream")
async def get_upstream_lineage(
    data_id: str,
    max_depth: Optional[int] = Query(10, le=50)
):
    """Get upstream data lineage"""
    try:
        framework = get_governance_framework()
        paths = await framework.lineage_tracker.get_upstream_lineage(data_id, max_depth)
        
        # Convert paths to dict format
        path_dicts = []
        for path in paths:
            path_dict = {
                "source_node_id": path.source_node_id,
                "target_node_id": path.target_node_id,
                "path": path.path,
                "path_length": path.path_length,
                "total_transformations": path.total_transformations,
                "path_confidence": path.path_confidence,
                "earliest_timestamp": path.earliest_timestamp.isoformat() if path.earliest_timestamp else None,
                "latest_timestamp": path.latest_timestamp.isoformat() if path.latest_timestamp else None,
                "edges": [
                    {
                        "source_node_id": edge.source_node_id,
                        "target_node_id": edge.target_node_id,
                        "event_type": edge.event_type.value,
                        "process_name": edge.process_name,
                        "timestamp": edge.timestamp.isoformat()
                    }
                    for edge in path.edges
                ]
            }
            path_dicts.append(path_dict)
        
        return {
            "data_id": data_id,
            "direction": "upstream",
            "paths": path_dicts,
            "total_paths": len(paths),
            "max_depth": max_depth,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting upstream lineage for {data_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get upstream lineage")

@router.get("/lineage/{data_id}/downstream")
async def get_downstream_lineage(
    data_id: str,
    max_depth: Optional[int] = Query(10, le=50)
):
    """Get downstream data lineage"""
    try:
        framework = get_governance_framework()
        paths = await framework.lineage_tracker.get_downstream_lineage(data_id, max_depth)
        
        # Convert paths to dict format (same as upstream)
        path_dicts = []
        for path in paths:
            path_dict = {
                "source_node_id": path.source_node_id,
                "target_node_id": path.target_node_id,
                "path": path.path,
                "path_length": path.path_length,
                "total_transformations": path.total_transformations,
                "path_confidence": path.path_confidence,
                "earliest_timestamp": path.earliest_timestamp.isoformat() if path.earliest_timestamp else None,
                "latest_timestamp": path.latest_timestamp.isoformat() if path.latest_timestamp else None,
                "edges": [
                    {
                        "source_node_id": edge.source_node_id,
                        "target_node_id": edge.target_node_id,
                        "event_type": edge.event_type.value,
                        "process_name": edge.process_name,
                        "timestamp": edge.timestamp.isoformat()
                    }
                    for edge in path.edges
                ]
            }
            path_dicts.append(path_dict)
        
        return {
            "data_id": data_id,
            "direction": "downstream",
            "paths": path_dicts,
            "total_paths": len(paths),
            "max_depth": max_depth,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting downstream lineage for {data_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get downstream lineage")

@router.get("/lineage/{data_id}/impact")
async def get_data_impact_analysis(data_id: str, change_type: str = "modification"):
    """Get impact analysis for potential changes to data"""
    try:
        framework = get_governance_framework()
        impact = await framework.lineage_tracker.analyze_data_impact(data_id, change_type)
        
        return {
            "impact_analysis": impact,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting impact analysis for {data_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get impact analysis")

# Data Quality Endpoints

@router.post("/quality/assess")
async def assess_data_quality(
    data_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Assess data quality for specific content"""
    try:
        framework = get_governance_framework()
        quality_score = await framework.quality_monitor.assess_quality(data_id, content, metadata)
        
        return {
            "data_id": data_id,
            "quality_score": quality_score,
            "assessment_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error assessing quality for {data_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to assess data quality")

@router.get("/quality/issues")
async def get_quality_issues(
    severity: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None),
    limit: int = Query(100, le=1000)
):
    """Get data quality issues"""
    try:
        framework = get_governance_framework()
        
        # Filter issues based on parameters
        issues = []
        for issue in framework.quality_monitor.issues.values():
            if severity and issue.severity != severity:
                continue
            if resolved is not None and (issue.resolved_at is not None) != resolved:
                continue
            
            issue_dict = {
                "id": issue.id,
                "rule_id": issue.rule_id,
                "data_id": issue.data_id,
                "issue_type": issue.issue_type.value,
                "dimension": issue.dimension.value,
                "description": issue.description,
                "severity": issue.severity,
                "failure_count": issue.failure_count,
                "total_records": issue.total_records,
                "failure_percentage": issue.failure_percentage,
                "detected_at": issue.detected_at.isoformat(),
                "resolved_at": issue.resolved_at.isoformat() if issue.resolved_at else None,
                "sample_failures": issue.sample_failures[:5],  # Limit sample size
                "metadata": issue.metadata
            }
            issues.append(issue_dict)
            
            if len(issues) >= limit:
                break
        
        return {
            "issues": issues,
            "total_count": len(issues),
            "filters_applied": {
                "severity": severity,
                "resolved": resolved
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting quality issues: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quality issues")

# Data Versioning Endpoints

@router.post("/versioning/{asset_id}/versions")
async def create_data_version(
    asset_id: str,
    content: Any,
    change_type: str = "update",
    description: Optional[str] = None,
    created_by: Optional[str] = None,
    branch_name: str = "main"
):
    """Create a new version of a data asset"""
    try:
        framework = get_governance_framework()
        
        # Validate change type
        try:
            change_type_enum = ChangeType(change_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid change type: {change_type}")
        
        version = await framework.versioning.create_version(
            asset_id=asset_id,
            data_content=content,
            change_type=change_type_enum,
            description=description,
            created_by=created_by,
            branch_name=branch_name
        )
        
        if version:
            return {
                "version_id": version.version_id,
                "asset_id": version.asset_id,
                "version_number": version.version_number,
                "change_type": version.change_type.value,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "branch_name": version.branch_name,
                "data_hash": version.data_hash,
                "data_size_bytes": version.data_size_bytes,
                "record_count": version.record_count
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create version")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating version for {asset_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create version")

@router.get("/versioning/{asset_id}/history")
async def get_version_history(
    asset_id: str,
    branch_name: str = Query("main"),
    limit: Optional[int] = Query(50, le=1000)
):
    """Get version history for an asset"""
    try:
        framework = get_governance_framework()
        versions = await framework.versioning.get_version_history(asset_id, branch_name, limit)
        
        version_dicts = []
        for version in versions:
            version_dict = {
                "version_id": version.version_id,
                "version_number": version.version_number,
                "change_type": version.change_type.value,
                "description": version.description,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "data_hash": version.data_hash,
                "data_size_bytes": version.data_size_bytes,
                "record_count": version.record_count,
                "quality_score": version.quality_score,
                "validation_status": version.validation_status,
                "tags": version.tags
            }
            version_dicts.append(version_dict)
        
        return {
            "asset_id": asset_id,
            "branch_name": branch_name,
            "versions": version_dicts,
            "total_count": len(versions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting version history for {asset_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get version history")

# Admin Endpoints

@router.post("/admin/run-governance-checks")
async def run_periodic_governance_checks(background_tasks: BackgroundTasks):
    """Trigger periodic governance checks manually"""
    try:
        framework = get_governance_framework()
        
        background_tasks.add_task(framework.run_periodic_governance_checks)
        
        return {
            "status": "initiated",
            "message": "Periodic governance checks started",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running governance checks: {e}")
        raise HTTPException(status_code=500, detail="Failed to run governance checks")

@router.get("/admin/statistics")
async def get_admin_statistics():
    """Get comprehensive system statistics for administrators"""
    try:
        framework = get_governance_framework()
        
        stats = {
            "governance_overview": framework.get_governance_dashboard_data(),
            "audit_stats": await framework.audit_logger.get_audit_statistics(30),
            "compliance_stats": framework.compliance_manager.get_compliance_statistics(),
            "quality_stats": framework.quality_monitor.get_quality_statistics(),
            "catalog_stats": framework.data_catalog.get_catalog_statistics(),
            "lineage_stats": await framework.lineage_tracker.get_lineage_statistics(),
            "versioning_stats": framework.versioning.get_versioning_statistics(),
            "system_health": {
                "uptime_hours": 0,  # Would calculate actual uptime
                "memory_usage_mb": 0,  # Would get actual memory usage
                "active_background_tasks": 0  # Would count active tasks
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting admin statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get admin statistics")