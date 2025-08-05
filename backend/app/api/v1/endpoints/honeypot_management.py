"""
Honeypot Management API Endpoints
Provides comprehensive API for managing and monitoring honeypot infrastructure
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
import json

# Import existing security infrastructure
try:
    from app.api.v1.security import get_current_user
    SECURITY_ENABLED = True
except ImportError:
    SECURITY_ENABLED = False
    
    async def get_current_user():
        return {"user_id": "admin", "scopes": ["admin"]}

# Import honeypot infrastructure
from security.honeypot_integration import unified_honeypot_manager

logger = logging.getLogger(__name__)

# Request/Response Models
class HoneypotDeploymentRequest(BaseModel):
    """Request model for honeypot deployment"""
    honeypot_types: List[str] = Field(
        default=["ssh", "web", "database", "ai_agent"],
        description="Types of honeypots to deploy"
    )
    ports_config: Optional[Dict[str, int]] = Field(
        default=None,
        description="Custom port configuration for honeypots"
    )
    enable_cowrie: bool = Field(
        default=True,
        description="Enable Cowrie SSH honeypot"
    )
    enable_https: bool = Field(
        default=True,
        description="Enable HTTPS honeypots"
    )
    security_integration: bool = Field(
        default=True,
        description="Enable integration with security system"
    )

class HoneypotStatusResponse(BaseModel):
    """Response model for honeypot status"""
    deployment_status: str
    total_honeypots: int
    active_honeypots: int
    security_integration: bool
    uptime: Optional[str] = None

class ThreatIntelligenceRequest(BaseModel):
    """Request model for threat intelligence queries"""
    time_period_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Time period in hours (1-168)"
    )
    severity_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter by severity levels"
    )
    attack_vector_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter by attack vectors"
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of events to analyze"
    )

class AttackerProfileResponse(BaseModel):
    """Response model for attacker profiles"""
    source_ip: str
    first_seen: str
    last_seen: str
    total_attempts: int
    threat_score: float
    attack_patterns: List[str]
    honeypots_targeted: List[str]

# Router setup
router = APIRouter()

# Helper functions
async def require_admin_access(current_user: Dict = Depends(get_current_user)):
    """Require admin access for honeypot management operations"""
    if SECURITY_ENABLED and current_user:
        if "admin" not in current_user.get("scopes", []):
            raise HTTPException(
                status_code=403, 
                detail="Admin access required for honeypot management"
            )
    return current_user

@router.post("/deploy", response_model=Dict[str, Any])
async def deploy_honeypot_infrastructure(
    request: HoneypotDeploymentRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_admin_access)
):
    """
    Deploy comprehensive honeypot infrastructure
    
    **Features:**
    - Multiple honeypot types (SSH, Web, Database, AI Agent)
    - Configurable ports and services
    - Security system integration
    - Background deployment process
    """
    try:
        logger.info(f"Honeypot deployment requested by user {current_user.get('user_id')}")
        
        # Initialize if not already done
        if not unified_honeypot_manager.orchestrator.database:
            await unified_honeypot_manager.initialize()
        
        # Start deployment in background
        background_tasks.add_task(
            run_deployment_background,
            request
        )
        
        return {
            "status": "deployment_initiated",
            "message": "Honeypot infrastructure deployment started",
            "requested_types": request.honeypot_types,
            "estimated_completion": (
                datetime.utcnow() + timedelta(minutes=5)
            ).isoformat(),
            "deployment_id": f"deploy_{int(datetime.utcnow().timestamp())}"
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate honeypot deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

async def run_deployment_background(request: HoneypotDeploymentRequest):
    """Run honeypot deployment in background"""
    try:
        logger.info("Starting background honeypot deployment...")
        
        # Deploy comprehensive infrastructure
        deployment_results = await unified_honeypot_manager.deploy_comprehensive_honeypot_infrastructure()
        
        logger.info(f"Background deployment completed: {deployment_results}")
        
    except Exception as e:
        logger.error(f"Background deployment failed: {e}")

@router.get("/status", response_model=Dict[str, Any])
async def get_honeypot_status(
    current_user: Dict = Depends(require_admin_access)
):
    """
    Get comprehensive honeypot infrastructure status
    
    **Returns:**
    - Overall deployment status
    - Individual component status
    - Recent activity metrics
    - Security integration status
    """
    try:
        status = await unified_honeypot_manager.get_comprehensive_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get honeypot status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/undeploy")
async def undeploy_honeypot_infrastructure(
    current_user: Dict = Depends(require_admin_access)
):
    """
    Undeploy all honeypot infrastructure
    
    **Warning:** This will stop all honeypots and disable threat detection
    """
    try:
        logger.info(f"Honeypot undeployment requested by user {current_user.get('user_id')}")
        
        await unified_honeypot_manager.undeploy_all()
        
        return {
            "status": "undeployed",
            "message": "Honeypot infrastructure undeployed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to undeploy honeypots: {e}")
        raise HTTPException(status_code=500, detail=f"Undeployment failed: {str(e)}")

@router.get("/events")
async def get_honeypot_events(
    limit: int = Query(default=100, le=1000, description="Maximum number of events"),
    hours: int = Query(default=24, le=168, description="Time period in hours"),
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    honeypot_type: Optional[str] = Query(default=None, description="Filter by honeypot type"),
    source_ip: Optional[str] = Query(default=None, description="Filter by source IP"),
    current_user: Dict = Depends(require_admin_access)
):
    """
    Get recent honeypot events with filtering options
    
    **Filters:**
    - Time period (hours)
    - Severity level
    - Honeypot type  
    - Source IP address
    """
    try:
        if not unified_honeypot_manager.orchestrator.database:
            raise HTTPException(status_code=503, detail="Honeypot infrastructure not available")
        
        # Get events from database
        events = unified_honeypot_manager.orchestrator.database.get_events(
            limit=limit, 
            hours=hours
        )
        
        # Apply filters
        filtered_events = []
        for event in events:
            # Severity filter
            if severity and event.severity != severity:
                continue
                
            # Honeypot type filter
            if honeypot_type and event.honeypot_type != honeypot_type:
                continue
                
            # Source IP filter
            if source_ip and event.source_ip != source_ip:
                continue
            
            # Convert to dict for JSON response
            event_dict = {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "honeypot_id": event.honeypot_id,
                "honeypot_type": event.honeypot_type,
                "source_ip": event.source_ip,
                "source_port": event.source_port,
                "destination_port": event.destination_port,
                "event_type": event.event_type,
                "severity": event.severity,
                "attack_vector": event.attack_vector,
                "payload_preview": event.payload[:200] if event.payload else "",
                "threat_indicators": event.threat_indicators,
                "user_agent": event.user_agent
            }
            
            if event.credentials:
                event_dict["credentials_attempted"] = {
                    "username": event.credentials.get("username", ""),
                    "password_length": len(event.credentials.get("password", ""))
                }
            
            filtered_events.append(event_dict)
        
        return {
            "events": filtered_events,
            "total": len(filtered_events),
            "filters_applied": {
                "time_period_hours": hours,
                "severity": severity,
                "honeypot_type": honeypot_type,
                "source_ip": source_ip
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get honeypot events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve events: {str(e)}")

@router.get("/attackers")
async def get_attacker_profiles(
    limit: int = Query(default=20, le=100, description="Maximum number of attackers"),
    min_attempts: int = Query(default=1, description="Minimum number of attempts"),
    sort_by: str = Query(default="threat_score", description="Sort by: threat_score, total_attempts, last_seen"),
    current_user: Dict = Depends(require_admin_access)
):
    """
    Get attacker profiles with behavior analysis
    
    **Features:**
    - Threat scoring
    - Attack pattern analysis
    - Temporal analysis
    - Targeting analysis
    """
    try:
        if not unified_honeypot_manager.orchestrator.database:
            raise HTTPException(status_code=503, detail="Honeypot infrastructure not available")
        
        # Get recent events to build attacker profiles
        events = unified_honeypot_manager.orchestrator.database.get_events(
            limit=5000, 
            hours=168  # Last week
        )
        
        # Build attacker profiles
        attacker_profiles = {}
        
        for event in events:
            ip = event.source_ip
            
            if ip not in attacker_profiles:
                attacker_profiles[ip] = {
                    "source_ip": ip,
                    "first_seen": event.timestamp,
                    "last_seen": event.timestamp,
                    "total_attempts": 0,
                    "honeypots_targeted": set(),
                    "attack_patterns": set(),
                    "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                    "user_agents": set(),
                    "credentials_tried": []
                }
            
            profile = attacker_profiles[ip]
            
            # Update profile
            profile["total_attempts"] += 1
            profile["last_seen"] = max(profile["last_seen"], event.timestamp)
            profile["first_seen"] = min(profile["first_seen"], event.timestamp)
            profile["honeypots_targeted"].add(event.honeypot_type)
            
            if event.attack_vector:
                profile["attack_patterns"].add(event.attack_vector)
            
            if event.severity in profile["severity_counts"]:
                profile["severity_counts"][event.severity] += 1
            
            if event.user_agent:
                profile["user_agents"].add(event.user_agent)
            
            if event.credentials:
                username = event.credentials.get("username", "")
                if username and username not in [c.get("username") for c in profile["credentials_tried"]]:
                    profile["credentials_tried"].append({"username": username})
        
        # Calculate threat scores and filter
        scored_attackers = []
        
        for ip, profile in attacker_profiles.items():
            if profile["total_attempts"] < min_attempts:
                continue
            
            # Calculate threat score (0-1)
            threat_score = 0.0
            
            # Base score from attempt count
            attempt_score = min(profile["total_attempts"] / 100.0, 0.4)
            threat_score += attempt_score
            
            # Severity score
            severity_score = (
                profile["severity_counts"]["critical"] * 0.25 +
                profile["severity_counts"]["high"] * 0.15 +
                profile["severity_counts"]["medium"] * 0.05 +
                profile["severity_counts"]["low"] * 0.01
            ) / max(profile["total_attempts"], 1)
            threat_score += min(severity_score, 0.3)
            
            # Diversity score
            diversity_score = (
                len(profile["honeypots_targeted"]) * 0.05 +
                len(profile["attack_patterns"]) * 0.05
            )
            threat_score += min(diversity_score, 0.2)
            
            # Persistence score
            time_span = (profile["last_seen"] - profile["first_seen"]).total_seconds()
            if time_span > 3600:  # More than 1 hour
                persistence_score = min(time_span / (7 * 24 * 3600), 0.1)  # Up to a week
                threat_score += persistence_score
            
            profile["threat_score"] = min(threat_score, 1.0)
            
            # Convert sets to lists for JSON serialization
            profile["honeypots_targeted"] = list(profile["honeypots_targeted"])
            profile["attack_patterns"] = list(profile["attack_patterns"])
            profile["user_agents"] = list(profile["user_agents"])
            
            # Format timestamps
            profile["first_seen"] = profile["first_seen"].isoformat()
            profile["last_seen"] = profile["last_seen"].isoformat()
            
            scored_attackers.append(profile)
        
        # Sort attackers
        if sort_by == "threat_score":
            scored_attackers.sort(key=lambda x: x["threat_score"], reverse=True)
        elif sort_by == "total_attempts":
            scored_attackers.sort(key=lambda x: x["total_attempts"], reverse=True)
        elif sort_by == "last_seen":
            scored_attackers.sort(key=lambda x: x["last_seen"], reverse=True)
        
        return {
            "attackers": scored_attackers[:limit],
            "total_analyzed": len(attacker_profiles),
            "filters_applied": {
                "min_attempts": min_attempts,
                "sort_by": sort_by
            },
            "analysis_period": "7 days",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get attacker profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze attackers: {str(e)}")

@router.get("/intelligence/report")
async def get_threat_intelligence_report(
    current_user: Dict = Depends(require_admin_access)
):
    """
    Generate comprehensive threat intelligence report
    
    **Includes:**
    - Attack trend analysis
    - Threat actor profiling
    - Attack vector distribution
    - Honeypot effectiveness metrics
    - Security recommendations
    """
    try:
        report = await unified_honeypot_manager.generate_threat_intelligence_report()
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate threat intelligence report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/analytics/dashboard")
async def get_analytics_dashboard_data(
    current_user: Dict = Depends(require_admin_access)
):
    """
    Get data for honeypot analytics dashboard
    
    **Dashboard Metrics:**
    - Real-time attack statistics
    - Geographic distribution
    - Attack timeline
    - Honeypot performance
    - Threat indicators
    """
    try:
        if not unified_honeypot_manager.orchestrator.database:
            raise HTTPException(status_code=503, detail="Honeypot infrastructure not available")
        
        # Get events for different time periods
        events_1h = unified_honeypot_manager.orchestrator.database.get_events(limit=1000, hours=1)
        events_24h = unified_honeypot_manager.orchestrator.database.get_events(limit=5000, hours=24)
        events_7d = unified_honeypot_manager.orchestrator.database.get_events(limit=10000, hours=168)
        
        # Calculate metrics
        dashboard_data = {
            "real_time_stats": {
                "events_last_hour": len(events_1h),
                "events_last_24h": len(events_24h),
                "events_last_7d": len(events_7d),
                "unique_attackers_24h": len(set(e.source_ip for e in events_24h)),
                "critical_events_24h": len([e for e in events_24h if e.severity == "critical"]),
                "high_events_24h": len([e for e in events_24h if e.severity == "high"])
            },
            
            "attack_distribution": {
                "by_honeypot_type": {},
                "by_severity": {},
                "by_attack_vector": {}
            },
            
            "timeline": {
                "hourly_events": [],
                "daily_events": []
            },
            
            "top_sources": [],
            
            "recent_attacks": []
        }
        
        # Attack distribution
        for event in events_24h:
            # By honeypot type
            hp_type = event.honeypot_type
            dashboard_data["attack_distribution"]["by_honeypot_type"][hp_type] = \
                dashboard_data["attack_distribution"]["by_honeypot_type"].get(hp_type, 0) + 1
            
            # By severity
            severity = event.severity
            dashboard_data["attack_distribution"]["by_severity"][severity] = \
                dashboard_data["attack_distribution"]["by_severity"].get(severity, 0) + 1
            
            # By attack vector
            if event.attack_vector:
                vector = event.attack_vector
                dashboard_data["attack_distribution"]["by_attack_vector"][vector] = \
                    dashboard_data["attack_distribution"]["by_attack_vector"].get(vector, 0) + 1
        
        # Timeline data (simplified - would need proper time bucketing in production)
        now = datetime.utcnow()
        hourly_buckets = {}
        
        for event in events_24h:
            hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_buckets[hour_key] = hourly_buckets.get(hour_key, 0) + 1
        
        dashboard_data["timeline"]["hourly_events"] = [
            {"time": hour.isoformat(), "count": count}
            for hour, count in sorted(hourly_buckets.items())
        ]
        
        # Top attacking IPs
        ip_counts = {}
        for event in events_24h:
            ip_counts[event.source_ip] = ip_counts.get(event.source_ip, 0) + 1
        
        dashboard_data["top_sources"] = [
            {"ip": ip, "attempts": count}
            for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Recent high-severity attacks
        recent_critical = [
            e for e in events_1h 
            if e.severity in ["critical", "high"]
        ][:10]
        
        dashboard_data["recent_attacks"] = [
            {
                "timestamp": event.timestamp.isoformat(),
                "source_ip": event.source_ip,
                "honeypot_type": event.honeypot_type,
                "event_type": event.event_type,
                "severity": event.severity,
                "attack_vector": event.attack_vector
            }
            for event in recent_critical
        ]
        
        dashboard_data["last_updated"] = datetime.utcnow().isoformat()
        
        return dashboard_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")

@router.post("/config/update")
async def update_honeypot_configuration(
    config_updates: Dict[str, Any],
    current_user: Dict = Depends(require_admin_access)
):
    """
    Update honeypot configuration settings
    
    **Configurable Settings:**
    - Alert thresholds
    - Logging levels
    - Security integration settings
    - Port configurations
    """
    try:
        logger.info(f"Configuration update requested by user {current_user.get('user_id')}")
        
        # Validate and apply configuration updates
        valid_configs = ["alert_thresholds", "logging_level", "security_integration"]
        applied_configs = {}
        
        for key, value in config_updates.items():
            if key in valid_configs:
                # Apply configuration (implementation depends on specific config)
                if key == "alert_thresholds" and isinstance(value, dict):
                    # Update alert thresholds
                    if unified_honeypot_manager.security_bridge:
                        unified_honeypot_manager.security_bridge.alert_threshold.update(value)
                        applied_configs[key] = value
                
                elif key == "logging_level":
                    # Update logging level
                    if value in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                        logging.getLogger("security.honeypot").setLevel(value)
                        applied_configs[key] = value
                
                elif key == "security_integration":
                    # Toggle security integration
                    if isinstance(value, bool):
                        if unified_honeypot_manager.security_bridge:
                            unified_honeypot_manager.security_bridge.is_active = value
                            applied_configs[key] = value
        
        return {
            "status": "configuration_updated",
            "applied_configs": applied_configs,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@router.get("/health")
async def get_honeypot_health():
    """
    Get honeypot infrastructure health status
    
    **Health Checks:**
    - Service availability
    - Database connectivity
    - Security integration status
    - Recent activity verification
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "alerts": []
        }
        
        # Check if infrastructure is deployed
        if not unified_honeypot_manager.is_deployed:
            health_status["status"] = "not_deployed"
            health_status["alerts"].append("Honeypot infrastructure is not deployed")
            return health_status
        
        # Check orchestrator
        if unified_honeypot_manager.orchestrator:
            if unified_honeypot_manager.orchestrator.is_running:
                health_status["components"]["orchestrator"] = "healthy"
            else:
                health_status["components"]["orchestrator"] = "unhealthy"
                health_status["status"] = "degraded"
                health_status["alerts"].append("Honeypot orchestrator is not running")
        
        # Check database connectivity
        try:
            if unified_honeypot_manager.orchestrator.database:
                # Try to get recent events
                recent_events = unified_honeypot_manager.orchestrator.database.get_events(limit=1, hours=1)
                health_status["components"]["database"] = "healthy"
            else:
                health_status["components"]["database"] = "unavailable"
                health_status["status"] = "unhealthy"
                health_status["alerts"].append("Database is not available")
        except Exception as e:
            health_status["components"]["database"] = "unhealthy"
            health_status["status"] = "degraded" if health_status["status"] == "healthy" else health_status["status"]
            health_status["alerts"].append(f"Database health check failed: {str(e)}")
        
        # Check security integration
        if unified_honeypot_manager.security_bridge:
            if unified_honeypot_manager.security_bridge.is_active:
                health_status["components"]["security_integration"] = "active"
            else:
                health_status["components"]["security_integration"] = "inactive"
        else:
            health_status["components"]["security_integration"] = "unavailable"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/capabilities")
async def get_honeypot_capabilities():
    """
    Get honeypot system capabilities and features
    
    **Returns:**
    - Available honeypot types
    - Supported attack detection methods
    - Integration capabilities
    - Configuration options
    """
    capabilities = {
        "honeypot_types": [
            {
                "type": "ssh",
                "description": "SSH honeypot using Cowrie for brute force detection",
                "protocols": ["SSH"],
                "default_ports": [22, 2222],
                "attack_detection": ["brute_force", "credential_harvesting", "command_injection"]
            },
            {
                "type": "web", 
                "description": "HTTP/HTTPS honeypot for web application attacks",
                "protocols": ["HTTP", "HTTPS"],
                "default_ports": [80, 8080, 443, 8443],
                "attack_detection": ["sql_injection", "xss", "path_traversal", "command_injection"]
            },
            {
                "type": "database",
                "description": "Database honeypots for SQL injection detection",
                "protocols": ["MySQL", "PostgreSQL", "Redis"],
                "default_ports": [3306, 5432, 6379],
                "attack_detection": ["sql_injection", "unauthorized_access", "data_exfiltration"]
            },
            {
                "type": "ai_agent",
                "description": "AI agent honeypots mimicking SutazAI services",
                "protocols": ["HTTP", "WebSocket"],
                "default_ports": [8000, 11434, 9000],
                "attack_detection": ["prompt_injection", "model_extraction", "ai_manipulation"]
            }
        ],
        
        "attack_detection_methods": [
            "pattern_matching",
            "behavioral_analysis", 
            "threat_intelligence",
            "machine_learning_classification",
            "signature_detection"
        ],
        
        "integrations": [
            {
                "name": "Security System Integration",
                "description": "Integration with existing security infrastructure",
                "enabled": SECURITY_ENABLED
            },
            {
                "name": "SIEM Integration", 
                "description": "Security Information and Event Management integration",
                "enabled": True
            },
            {
                "name": "Threat Intelligence Feeds",
                "description": "External threat intelligence integration",
                "enabled": True
            }
        ],
        
        "features": [
            "Real-time attack detection",
            "Attacker profiling and behavior analysis",
            "Threat intelligence generation",
            "Automated alerting",
            "Comprehensive logging",
            "Security orchestration integration",
            "Dashboard and analytics",
            "RESTful API management"
        ],
        
        "deployment_options": [
            "Standalone deployment",
            "Integrated with existing security systems",
            "Cloud-native deployment",
            "Containerized deployment",
            "High-availability deployment"
        ]
    }
    
    return capabilities