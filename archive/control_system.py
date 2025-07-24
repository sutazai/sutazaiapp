"""
Control System - Unified Backend for Chaos-to-Value Conversion
This system gives you control over the value extraction and intelligence processes
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import uvicorn

# Import our core systems
from chaos_engine import ChaosEngine, DataSource
from silent_operator import silent_operator
from value_extractor import value_extractor

app = FastAPI(
    title="SutazAI Control System",
    description="Unified backend for chaos-to-value conversion",
    version="1.0.0"
)

# CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instances
chaos_engine = ChaosEngine()
system_running = False
background_tasks = []

# Pydantic models for API
class DataSourceCreate(BaseModel):
    name: str
    url: str
    type: str  # 'api', 'scrape', 'feed', 'stream'
    frequency: int
    processor: str

class ExtractionRequest(BaseModel):
    data: Any
    source: str = "manual"
    extraction_intensity: str = "maximum"

class SystemConfig(BaseModel):
    chaos_threshold: float = 0.7
    extraction_intensity: str = "maximum"
    stealth_mode: bool = True
    value_priority: str = "monetary"

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "system": "SutazAI Control System",
        "status": "operational" if system_running else "stopped",
        "capabilities": [
            "chaos_to_value_conversion",
            "silent_intelligence_gathering", 
            "pattern_extraction",
            "opportunity_identification",
            "risk_assessment",
            "behavioral_analysis",
            "competitive_intelligence"
        ],
        "version": "1.0.0"
    }

@app.post("/system/start")
async def start_system(background_tasks: BackgroundTasks):
    """Start the complete chaos-to-value system"""
    global system_running, background_tasks
    
    if system_running:
        return {"status": "already_running"}
    
    try:
        # Start chaos engine
        chaos_tasks = await chaos_engine.start()
        background_tasks.extend(chaos_tasks)
        
        # Silent operator is already running (auto-started)
        
        system_running = True
        
        return {
            "status": "started",
            "components": [
                "chaos_engine",
                "silent_operator", 
                "value_extractor"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start system: {str(e)}")

@app.post("/system/stop")
async def stop_system():
    """Stop the system"""
    global system_running
    
    if not system_running:
        return {"status": "already_stopped"}
    
    try:
        await chaos_engine.stop()
        system_running = False
        
        return {
            "status": "stopped",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop system: {str(e)}")

@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get chaos engine status
        chaos_summary = await chaos_engine.get_intelligence_summary()
        
        # Get silent operator status
        silent_summary = await silent_operator.get_intelligence_summary()
        
        # Get value extractor status
        value_summary = await value_extractor.get_total_value_extracted()
        
        return {
            "system_running": system_running,
            "timestamp": datetime.now().isoformat(),
            "chaos_engine": chaos_summary,
            "silent_operator": silent_summary,
            "value_extractor": value_summary,
            "total_intelligence_packets": (
                chaos_summary.get("total_packets", 0) + 
                silent_summary.get("recent_packets", 0)
            ),
            "total_value_extracted": value_summary.get("cumulative_value", 0),
            "system_intelligence_level": silent_summary.get("intelligence_level", 1.0)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/data-sources/add")
async def add_data_source(source: DataSourceCreate):
    """Add a new data source for intelligence gathering"""
    try:
        data_source = DataSource(
            name=source.name,
            url=source.url,
            type=source.type,
            frequency=source.frequency,
            processor=source.processor
        )
        
        await chaos_engine.add_data_source(data_source)
        
        return {
            "status": "added",
            "source": source.name,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add data source: {str(e)}")

@app.get("/data-sources/list")
async def list_data_sources():
    """List all active data sources"""
    try:
        sources = []
        for source in chaos_engine.active_sources:
            sources.append({
                "name": source.name,
                "url": source.url,
                "type": source.type,
                "frequency": source.frequency,
                "value_score": source.value_score,
                "last_extraction": source.last_extraction.isoformat() if source.last_extraction else None
            })
        
        return {
            "active_sources": sources,
            "total_count": len(sources)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sources: {str(e)}")

@app.post("/extract/value")
async def extract_value_from_data(request: ExtractionRequest):
    """Extract value from provided data"""
    try:
        result = await value_extractor.extract_value(request.data, request.source)
        
        return {
            "extraction_id": hash(str(result.timestamp)),
            "source": result.source,
            "timestamp": result.timestamp.isoformat(),
            "value_metrics": {
                "total_value": result.value_metrics.total_value,
                "monetary_value": result.value_metrics.monetary_value,
                "strategic_value": result.value_metrics.strategic_value,
                "competitive_advantage": result.value_metrics.competitive_advantage,
                "opportunity_value": result.value_metrics.opportunity_value,
                "risk_mitigation": result.value_metrics.risk_mitigation,
                "time_value": result.value_metrics.time_value,
                "information_value": result.value_metrics.information_value
            },
            "confidence_score": result.confidence_score,
            "pattern_count": len(result.extracted_patterns),
            "actionable_items": result.actionable_items,
            "hidden_insights": result.hidden_insights,
            "next_actions": result.next_actions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract value: {str(e)}")

@app.get("/intelligence/summary")
async def get_intelligence_summary():
    """Get comprehensive intelligence summary"""
    try:
        # Get recent intelligence from chaos engine
        chaos_intel = await chaos_engine.get_intelligence_summary()
        
        # Get silent operator intelligence
        silent_intel = await silent_operator.get_intelligence_summary()
        
        # Combine and analyze
        summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "operational" if system_running else "stopped",
            "intelligence_sources": {
                "chaos_engine": {
                    "packets": chaos_intel.get("total_packets", 0),
                    "avg_value": chaos_intel.get("avg_value_score", 0),
                    "avg_confidence": chaos_intel.get("avg_confidence", 0)
                },
                "silent_operator": {
                    "packets": silent_intel.get("recent_packets", 0),
                    "stealth_level": silent_intel.get("stealth_level", 0),
                    "intelligence_level": silent_intel.get("intelligence_level", 1.0),
                    "total_value": silent_intel.get("total_value_accumulated", 0)
                }
            },
            "top_insights": chaos_intel.get("top_insights", []),
            "opportunity_count": chaos_intel.get("opportunity_count", 0),
            "risk_count": chaos_intel.get("risk_count", 0),
            "pattern_summary": chaos_intel.get("pattern_summary", {}),
            "recommendations": [
                "Continue intelligence gathering for pattern enhancement",
                "Monitor high-value data sources",
                "Leverage extracted patterns for competitive advantage"
            ]
        }
        
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get intelligence summary: {str(e)}")

@app.get("/patterns/latest")
async def get_latest_patterns():
    """Get latest extracted patterns"""
    try:
        # This would typically query the chaos engine's recent patterns
        # For now, return a structured response
        
        return {
            "timestamp": datetime.now().isoformat(),
            "pattern_count": len(chaos_engine.intelligence_buffer),
            "latest_patterns": [
                {
                    "id": i,
                    "source": packet.source,
                    "timestamp": packet.timestamp.isoformat(),
                    "value_score": packet.value_score,
                    "confidence": packet.confidence,
                    "insights_count": len(packet.actionable_insights)
                }
                for i, packet in enumerate(chaos_engine.intelligence_buffer[-10:])
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get patterns: {str(e)}")

@app.get("/opportunities/high-value")
async def get_high_value_opportunities():
    """Get high-value opportunities identified by the system"""
    try:
        opportunities = []
        
        # Extract high-value opportunities from intelligence buffer
        for packet in chaos_engine.intelligence_buffer:
            if packet.value_score > 0.7:  # High value threshold
                opportunities.append({
                    "source": packet.source,
                    "timestamp": packet.timestamp.isoformat(),
                    "value_score": packet.value_score,
                    "confidence": packet.confidence,
                    "insights": packet.actionable_insights[:3],  # Top 3
                    "silent_patterns": packet.silent_patterns
                })
        
        # Sort by value score
        opportunities.sort(key=lambda x: x["value_score"], reverse=True)
        
        return {
            "high_value_opportunities": opportunities[:20],  # Top 20
            "total_count": len(opportunities),
            "avg_value_score": sum(o["value_score"] for o in opportunities) / len(opportunities) if opportunities else 0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get opportunities: {str(e)}")

@app.post("/system/configure")
async def configure_system(config: SystemConfig):
    """Configure system parameters"""
    try:
        # Update chaos engine configuration
        chaos_engine.chaos_threshold = config.chaos_threshold
        chaos_engine.extraction_intensity = config.extraction_intensity
        
        # Update silent operator configuration
        if hasattr(silent_operator, 'visibility'):
            silent_operator.visibility = 0.0 if config.stealth_mode else 0.5
        
        return {
            "status": "configured",
            "configuration": config.dict(),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure system: {str(e)}")

@app.get("/analytics/performance")
async def get_performance_analytics():
    """Get system performance analytics"""
    try:
        # Calculate performance metrics
        total_packets = len(chaos_engine.intelligence_buffer)
        avg_value = sum(p.value_score for p in chaos_engine.intelligence_buffer) / total_packets if total_packets > 0 else 0
        avg_confidence = sum(p.confidence for p in chaos_engine.intelligence_buffer) / total_packets if total_packets > 0 else 0
        
        # Value extraction performance
        value_stats = await value_extractor.get_total_value_extracted()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "intelligence_performance": {
                "total_packets_processed": total_packets,
                "average_value_score": avg_value,
                "average_confidence": avg_confidence,
                "high_value_packet_ratio": len([p for p in chaos_engine.intelligence_buffer if p.value_score > 0.7]) / total_packets if total_packets > 0 else 0
            },
            "value_extraction_performance": value_stats,
            "system_efficiency": {
                "patterns_per_hour": total_packets / 24 if total_packets > 0 else 0,  # Assuming 24-hour operation
                "value_per_packet": avg_value,
                "confidence_trend": "stable"  # Would calculate actual trend
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance analytics: {str(e)}")

@app.get("/debug/system-state")
async def get_debug_info():
    """Get debug information about system state"""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "system_running": system_running,
            "chaos_engine": {
                "active_sources_count": len(chaos_engine.active_sources),
                "intelligence_buffer_size": len(chaos_engine.intelligence_buffer),
                "pattern_memory_size": len(chaos_engine.pattern_memory),
                "running": chaos_engine.running
            },
            "silent_operator": {
                "stealth_level": getattr(silent_operator, 'visibility', 'unknown'),
                "intelligence_level": getattr(silent_operator, 'intelligence_level', 'unknown'),
                "value_accumulator": getattr(silent_operator, 'value_accumulator', 'unknown')
            },
            "value_extractor": {
                "total_value_extracted": value_extractor.total_value_extracted,
                "extraction_history_size": len(value_extractor.extraction_history)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get debug info: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("SutazAI Control System starting...")
    print("Silent Operator: Already operational")
    print("Value Extractor: Ready")
    print("Chaos Engine: Ready to start")
    print("Control System: Online")

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "control_system:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production
        log_level="warning"  # Minimize logging for stealth
    )