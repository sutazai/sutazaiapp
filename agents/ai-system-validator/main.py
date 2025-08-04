#!/usr/bin/env python3
"""AI System Validator Agent - Validates system configurations and deployments"""

import asyncio
import os
import sys
import time
import psutil
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, generate_latest
import structlog
import uvicorn

# Configure structured logging
logger = structlog.get_logger()

# Prometheus metrics
validation_counter = Counter('system_validations_total', 'Total system validations performed')
validation_errors = Counter('system_validation_errors_total', 'Total validation errors')
system_health = Gauge('system_health_score', 'Overall system health score (0-100)')

app = FastAPI(title="AI System Validator")

class ValidationRequest(BaseModel):
    target: str
    validation_type: str = "full"
    timeout: int = 300

class ValidationResponse(BaseModel):
    status: str
    score: float
    issues: list
    recommendations: list

class SystemValidator:
    def __init__(self):
        self.logger = structlog.get_logger()
        
    async def validate_system(self, target: str, validation_type: str) -> dict:
        """Perform system validation"""
        self.logger.info("Starting system validation", target=target, type=validation_type)
        validation_counter.inc()
        
        issues = []
        recommendations = []
        score = 100.0
        
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # CPU validation
            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent}%")
                score -= 10
                recommendations.append("Consider scaling compute resources")
            
            # Memory validation
            if memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent}%")
                score -= 15
                recommendations.append("Increase memory allocation or optimize memory usage")
            
            # Disk validation
            if disk.percent > 90:
                issues.append(f"Low disk space: {disk.percent}% used")
                score -= 20
                recommendations.append("Clean up disk space or expand storage")
            
            # Service validation
            if validation_type in ["full", "services"]:
                service_issues = await self._validate_services(target)
                issues.extend(service_issues)
                score -= len(service_issues) * 5
            
            # Network validation
            if validation_type in ["full", "network"]:
                network_issues = await self._validate_network(target)
                issues.extend(network_issues)
                score -= len(network_issues) * 3
            
            # Update health metric
            system_health.set(max(0, score))
            
            return {
                "status": "healthy" if score >= 70 else "degraded" if score >= 40 else "critical",
                "score": max(0, score),
                "issues": issues,
                "recommendations": recommendations
            }
            
        except Exception as e:
            validation_errors.inc()
            self.logger.error("Validation failed", error=str(e))
            return {
                "status": "error",
                "score": 0,
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": ["Check system logs for details"]
            }
    
    async def _validate_services(self, target: str) -> list:
        """Validate running services"""
        issues = []
        
        # Check for critical processes
        critical_processes = ["dockerd", "containerd", "kubelet"]
        for proc_name in critical_processes:
            found = False
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and proc_name in proc.info['name']:
                    found = True
                    break
            if not found:
                issues.append(f"Critical process not running: {proc_name}")
        
        return issues
    
    async def _validate_network(self, target: str) -> list:
        """Validate network connectivity"""
        issues = []
        
        # Check network interfaces
        net_if = psutil.net_if_stats()
        down_interfaces = [iface for iface, stats in net_if.items() if not stats.isup]
        if down_interfaces:
            issues.append(f"Network interfaces down: {', '.join(down_interfaces)}")
        
        # Test connectivity
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8080", timeout=5.0)
                if response.status_code >= 500:
                    issues.append("Local service connectivity issues")
        except:
            pass  # Expected if no service on port 8080
        
        return issues

validator = SystemValidator()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/validate", response_model=ValidationResponse)
async def validate_system(request: ValidationRequest):
    """Perform system validation"""
    result = await validator.validate_system(request.target, request.validation_type)
    return ValidationResponse(**result)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/")
async def root():
    """Root endpoint with agent information"""
    return {
        "agent": "AI System Validator",
        "version": "1.0.0",
        "description": "Validates system configurations and deployments",
        "endpoints": ["/health", "/validate", "/metrics"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting AI System Validator", port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)