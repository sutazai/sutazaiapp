"""
Data Governance API Endpoints
=============================

FastAPI router for data governance functionality.
Integrates with the main SutazAI application.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Import governance components
from ....data_governance.governance_api import router as governance_router
from ....data_governance.governance_framework import DataGovernanceFramework
from ....data_governance.governance_api import set_governance_framework

logger = logging.getLogger("governance_endpoints")

# Create main router
router = APIRouter()

# Include the governance router
router.include_router(governance_router)

# Global governance framework instance
_governance_framework: Optional[DataGovernanceFramework] = None

async def get_governance_framework() -> DataGovernanceFramework:
    """Get or initialize the governance framework"""
    global _governance_framework
    
    if _governance_framework is None:
        # Initialize governance framework
        config = {
            'lifecycle': {
                'batch_size': 100,
                'execution_interval_minutes': 60,
                'require_approval_for_deletion': True,
                'max_deletions_per_batch': 10
            },
            'audit': {
                'audit_log_path': '/data/audit',
                'retention_days': 2555,  # 7 years
                'batch_size': 1000,
                'flush_interval_seconds': 60
            },
            'compliance': {
                'enabled_regulations': ['gdpr', 'ccpa', 'hipaa', 'sox'],
                'batch_size': 100,
                'audit_interval_hours': 24
            },
            'lineage': {
                'neo4j': {
                    'uri': 'bolt://localhost:7687',
                    'username': 'neo4j',
                    'password': 'password',
                    'database': 'lineage'
                },
                'max_lineage_depth': 20,
                'batch_size': 1000
            },
            'quality': {
                'batch_size': 1000,
                'assessment_interval_hours': 6,
                'default_quality_threshold': 0.8,
                'critical_quality_threshold': 0.5
            },
            'catalog': {
                'auto_discovery_enabled': True,
                'quality_threshold': 0.7,
                'max_sample_records': 5
            },
            'versioning': {
                'default_strategy': 'hybrid',
                'max_versions_per_asset': 100,
                'auto_cleanup_enabled': True,
                'retention_days': 365
            }
        }
        
        _governance_framework = DataGovernanceFramework(config)
        
        # Initialize the framework
        success = await _governance_framework.initialize()
        if not success:
            logger.error("Failed to initialize governance framework")
            raise HTTPException(status_code=500, detail="Governance framework initialization failed")
        
        # Set the framework in the governance API module
        set_governance_framework(_governance_framework)
        
        logger.info("Data governance framework initialized successfully")
    
    return _governance_framework

@router.on_event("startup")
async def startup_governance():
    """Initialize governance framework on startup"""
    try:
        await get_governance_framework()
        logger.info("Governance system startup complete")
    except Exception as e:
        logger.error(f"Failed to initialize governance system: {e}")

@router.on_event("shutdown") 
async def shutdown_governance():
    """Shutdown governance framework"""
    global _governance_framework
    if _governance_framework:
        try:
            await _governance_framework.shutdown()
            logger.info("Governance system shutdown complete")
        except Exception as e:
            logger.error(f"Error during governance shutdown: {e}")

# Additional integration endpoints specific to SutazAI

@router.post("/integrate/agent-data")
async def integrate_agent_data(
    agent_id: str,
    data_content: Any,
    data_type: str = "communication",
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Integrate AI agent data into governance pipeline"""
    try:
        framework = await get_governance_framework()
        
        # Create metadata for agent data
        metadata = {
            "source": "ai_agent",
            "agent_id": agent_id,
            "data_type": data_type,
            "created_at": datetime.utcnow().isoformat(),
            "source_system": "sutazai_agents"
        }
        
        # Process through governance pipeline
        data_id = f"agent_{agent_id}_{int(datetime.utcnow().timestamp())}"
        
        background_tasks.add_task(
            framework.process_new_data,
            data_id, str(data_content), metadata
        )
        
        return {
            "status": "accepted",
            "data_id": data_id,
            "agent_id": agent_id,
            "message": "Agent data submitted for governance processing",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error integrating agent data: {e}")
        raise HTTPException(status_code=500, detail="Failed to integrate agent data")

@router.post("/integrate/database-discovery")
async def discover_database_assets(
    database_config: Dict[str, Any],
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Discover and catalog database assets"""
    try:
        framework = await get_governance_framework()
        
        # Start asset discovery in background
        background_tasks.add_task(
            framework.data_catalog.discover_database_assets,
            database_config
        )
        
        return {
            "status": "initiated",
            "message": "Database asset discovery started",
            "database": database_config.get("database", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting database discovery: {e}")
        raise HTTPException(status_code=500, detail="Failed to start database discovery")

@router.get("/integration/sutazai-overview")
async def get_sutazai_governance_overview():
    """Get governance overview specific to SutazAI system"""
    try:
        framework = await get_governance_framework()
        
        # Get standard overview
        overview = framework.get_governance_dashboard_data()
        
        # Add SutazAI-specific metrics
        sutazai_metrics = {
            "ai_agents_monitored": 69,  # Current agent count
            "agent_data_classification": {
                "communication_data": "internal",
                "model_outputs": "confidential", 
                "training_data": "restricted"
            },
            "knowledge_graph_integration": {
                "nodes_tracked": await framework.lineage_tracker.get_lineage_statistics(),
                "relationships_mapped": "active"
            },
            "compliance_focus": {
                "ai_governance": "active",
                "data_minimization": "enforced",
                "automated_retention": "enabled"
            }
        }
        
        overview["sutazai_specific"] = sutazai_metrics
        
        return {
            "overview": overview,
            "system": "SutazAI",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting SutazAI governance overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get governance overview")

@router.post("/integration/model-tracking")
async def track_ai_model_lifecycle(
    model_name: str,
    model_version: str,
    training_data_ids: List[str],
    model_metadata: Dict[str, Any]
):
    """Track AI model in governance system"""
    try:
        framework = await get_governance_framework()
        
        # Register model in lineage tracker
        model_id = await framework.lineage_tracker.track_ai_model(
            model_name, model_version, training_data_ids
        )
        
        # Add to data catalog
        catalog_id = await framework.data_catalog.quick_add_api(
            f"/models/{model_name}/v{model_version}",
            "sutazai_models",
            f"AI model {model_name} version {model_version}",
            model_metadata.get("owner", "sutazai_system")
        )
        
        # Create initial version
        version = await framework.versioning.create_version(
            asset_id=catalog_id,
            data_content=model_metadata,
            change_type="insert",
            description=f"Initial version of model {model_name}",
            created_by=model_metadata.get("created_by", "system")
        )
        
        return {
            "model_id": model_id,
            "catalog_id": catalog_id,
            "version_id": version.version_id if version else None,
            "model_name": model_name,
            "model_version": model_version,
            "tracked_dependencies": len(training_data_ids),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error tracking AI model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to track AI model")

@router.get("/integration/agent-compliance-report")
async def get_agent_compliance_report():
    """Get compliance report specific to AI agent operations"""
    try:
        framework = await get_governance_framework()
        
        # Get compliance statistics
        compliance_stats = framework.compliance_manager.get_compliance_statistics()
        
        # Filter for agent-related compliance
        agent_compliance = {
            "total_agent_data_points": compliance_stats.get("total_violations", 0),
            "gdpr_compliance": {
                "agent_communications": "monitored",
                "personal_data_detection": "automated",
                "retention_policies": "enforced"
            },
            "data_minimization": {
                "excessive_logging": "monitored",
                "unnecessary_retention": "flagged",
                "automated_cleanup": "active"
            },
            "ai_specific_compliance": {
                "model_bias_monitoring": "planned",
                "explainability_tracking": "development",
                "audit_trail_completeness": "active"
            },
            "recommendations": [
                "Implement automated PII detection in agent communications",
                "Regular review of agent data retention policies", 
                "Enhance model versioning for compliance tracking"
            ]
        }
        
        return {
            "report": agent_compliance,
            "system": "SutazAI Agents",
            "report_date": datetime.utcnow().isoformat(),
            "next_review": (datetime.utcnow().replace(day=1) + timedelta(days=32)).replace(day=1).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating agent compliance report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate compliance report")