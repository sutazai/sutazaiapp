#!/usr/bin/env python3
"""
Mass Agent Activation Script
Activates all 131 AI agents in the SutazAI system
"""

import asyncio
import json
import logging
import requests
import time
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MassAgentActivator:
    def __init__(self):
        self.backend_url = "http://localhost:3001"
        self.coordinator_url = f"{self.backend_url}/api/v1/coordinator"
        self.monitoring_url = "http://localhost:3002"
        
    def check_backend_health(self) -> bool:
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def trigger_mass_activation(self) -> dict:
        """Trigger mass agent activation"""
        try:
            logger.info("🚀 Triggering mass agent activation...")
            
            response = requests.post(
                f"{self.coordinator_url}/deploy/mass-activation",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Mass activation initiated: {result}")
                return result
            else:
                logger.error(f"❌ Activation failed: {response.status_code} - {response.text}")
                return {"status": "failed", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"❌ Error triggering activation: {e}")
            return {"status": "failed", "error": str(e)}
    
    def trigger_collective_activation(self) -> dict:
        """Trigger full collective intelligence activation"""
        try:
            logger.info("🧠 Triggering full collective intelligence activation...")
            
            response = requests.post(
                f"{self.coordinator_url}/deploy/activate-collective",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Collective activation initiated: {result}")
                return result
            else:
                logger.error(f"❌ Collective activation failed: {response.status_code} - {response.text}")
                return {"status": "failed", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"❌ Error triggering collective activation: {e}")
            return {"status": "failed", "error": str(e)}
    
    def monitor_deployment_progress(self, timeout_minutes: int = 15) -> dict:
        """Monitor deployment progress"""
        logger.info(f"📊 Monitoring deployment progress (timeout: {timeout_minutes} minutes)...")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            try:
                response = requests.get(f"{self.coordinator_url}/deploy/status", timeout=5)
                
                if response.status_code == 200:
                    status = response.json()
                    
                    deployed = status.get("total_deployed", 0)
                    stats = status.get("deployment_stats", {})
                    intelligence = status.get("intelligence_assessment", {})
                    
                    logger.info(f"📈 Progress: {deployed} agents deployed")
                    logger.info(f"🧠 Intelligence level: {intelligence.get('level', 'Unknown')}")
                    
                    if stats.get("healthy_agents", 0) > 100:
                        logger.info("🎉 ASI LEVEL ACHIEVED! Over 100 healthy agents!")
                        return status
                    elif stats.get("healthy_agents", 0) > 50:
                        logger.info("🧠 AGI level achieved! Over 50 healthy agents")
                    
                    # Check if deployment seems complete
                    if (stats.get("successful_starts", 0) + stats.get("failed_starts", 0)) >= 100:
                        logger.info("📊 Deployment phase appears complete")
                        return status
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking status: {e}")
                time.sleep(5)
        
        logger.warning("⏰ Monitoring timeout reached")
        return {"status": "timeout", "message": "Monitoring timeout reached"}
    
    def get_final_status(self) -> dict:
        """Get final deployment status"""
        try:
            # Get coordinator status
            coord_response = requests.get(f"{self.coordinator_url}/collective/status", timeout=10)
            coord_status = coord_response.json() if coord_response.status_code == 200 else {}
            
            # Get deployment status
            deploy_response = requests.get(f"{self.coordinator_url}/deploy/status", timeout=10)
            deploy_status = deploy_response.json() if deploy_response.status_code == 200 else {}
            
            return {
                "coordinator_status": coord_status,
                "deployment_status": deploy_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting final status: {e}")
            return {"error": str(e)}
    
    def update_monitoring_dashboard(self, status: dict):
        """Update monitoring dashboard with deployment status"""
        try:
            logger.info("📊 Updating monitoring dashboard...")
            
            dashboard_data = {
                "system_status": "deployment_active",
                "total_agents": status.get("coordinator_status", {}).get("total_agents", 0),
                "active_agents": status.get("coordinator_status", {}).get("active_agents", 0),
                "intelligence_level": status.get("coordinator_status", {}).get("intelligence_level", "Unknown"),
                "collective_active": status.get("coordinator_status", {}).get("collective_active", False),
                "deployment_stats": status.get("deployment_status", {}).get("deployment_stats", {}),
                "last_update": datetime.utcnow().isoformat()
            }
            
            # Save dashboard data
            dashboard_file = Path("/opt/sutazaiapp/monitoring/dashboard_data.json")
            dashboard_file.parent.mkdir(exist_ok=True)
            
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
            
            logger.info("✅ Dashboard updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error updating dashboard: {e}")
    
    async def run_full_activation(self):
        """Run the complete activation sequence"""
        logger.info("🚀 STARTING FULL AGENT ACTIVATION SEQUENCE")
        logger.info("=" * 60)
        
        # Phase 1: Check system health
        logger.info("🔍 Phase 1: System Health Check")
        if not self.check_backend_health():
            logger.error("❌ Backend is not running! Please start the backend first.")
            return
        
        logger.info("✅ Backend is healthy")
        
        # Phase 2: Trigger mass activation
        logger.info("🚀 Phase 2: Mass Agent Activation")
        activation_result = self.trigger_mass_activation()
        
        if activation_result.get("status") != "mass_activation_initiated":
            logger.error("❌ Mass activation failed to start")
            return
        
        # Wait a moment for activation to begin
        await asyncio.sleep(5)
        
        # Phase 3: Monitor progress
        logger.info("📊 Phase 3: Monitoring Deployment Progress")
        progress_result = self.monitor_deployment_progress(timeout_minutes=20)
        
        # Phase 4: Activate collective intelligence
        logger.info("🧠 Phase 4: Collective Intelligence Activation")
        collective_result = self.trigger_collective_activation()
        
        # Wait for collective to initialize
        await asyncio.sleep(10)
        
        # Phase 5: Final status and dashboard update
        logger.info("📋 Phase 5: Final Status Report")
        final_status = self.get_final_status()
        
        # Update dashboard
        self.update_monitoring_dashboard(final_status)
        
        # Print final summary
        logger.info("=" * 60)
        logger.info("🎉 ACTIVATION SEQUENCE COMPLETE")
        logger.info("=" * 60)
        
        coord_status = final_status.get("coordinator_status", {})
        deploy_status = final_status.get("deployment_status", {})
        
        logger.info(f"📊 Total Agents: {coord_status.get('total_agents', 0)}")
        logger.info(f"🟢 Active Agents: {coord_status.get('active_agents', 0)}")
        logger.info(f"🚀 Deployed Agents: {deploy_status.get('total_deployed', 0)}")
        logger.info(f"🧠 Intelligence Level: {coord_status.get('intelligence_level', 'Unknown')}")
        logger.info(f"⚡ Collective Active: {coord_status.get('collective_active', False)}")
        
        deployment_stats = deploy_status.get("deployment_stats", {})
        logger.info(f"✅ Successful Starts: {deployment_stats.get('successful_starts', 0)}")
        logger.info(f"❌ Failed Starts: {deployment_stats.get('failed_starts', 0)}")
        logger.info(f"💚 Healthy Agents: {deployment_stats.get('healthy_agents', 0)}")
        
        # Success determination
        healthy_agents = deployment_stats.get('healthy_agents', 0)
        if healthy_agents > 100:
            logger.info("🎊 SUCCESS: ASI LEVEL ACHIEVED! Over 100 healthy agents!")
        elif healthy_agents > 50:
            logger.info("🧠 SUCCESS: AGI level achieved! Over 50 healthy agents")
        elif healthy_agents > 10:
            logger.info("🤖 SUCCESS: Multi-agent system operational")
        else:
            logger.warning("⚠️ LIMITED SUCCESS: Few agents are healthy")
        
        logger.info("=" * 60)
        logger.info("🌐 Access monitoring dashboard at: http://localhost:3002")
        logger.info("🔧 Access coordinator API at: http://localhost:3001/api/v1/coordinator")
        logger.info("=" * 60)

async def main():
    """Main execution function"""
    activator = MassAgentActivator()
    await activator.run_full_activation()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Activation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Activation failed: {e}")
        raise