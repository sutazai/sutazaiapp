"""
Honeypot Infrastructure Deployment Script
Automated deployment and management script for comprehensive honeypot infrastructure
"""

import asyncio
import logging
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/sutazaiapp/backend/logs/honeypot_deployment.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Import honeypot infrastructure
try:
    from security.honeypot_integration import unified_honeypot_manager
    from security.honeypot_infrastructure import honeypot_orchestrator
    
    # Import existing security infrastructure
    try:
        from app.core.security import security_manager
        SECURITY_INTEGRATION_AVAILABLE = True
        logger.info("Security system integration available")
    except ImportError:
        SECURITY_INTEGRATION_AVAILABLE = False
        logger.warning("Security system integration not available")
        
except ImportError as e:
    logger.error(f"Failed to import honeypot infrastructure: {e}")
    sys.exit(1)

class HoneypotDeploymentManager:
    """Manages honeypot infrastructure deployment and monitoring"""
    
    def __init__(self):
        self.deployment_start_time = None
        self.deployment_results = {}
        self.is_deployed = False
        
    async def deploy_full_infrastructure(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Deploy full honeypot infrastructure"""
        self.deployment_start_time = datetime.utcnow()
        
        logger.info("=" * 60)
        logger.info("🛡️  STARTING SUTAZAI HONEYPOT INFRASTRUCTURE DEPLOYMENT")
        logger.info("=" * 60)
        
        try:
            # Initialize unified manager
            logger.info("📋 Initializing honeypot management system...")
            await unified_honeypot_manager.initialize()
            logger.info("✅ Honeypot management system initialized")
            
            # Deploy comprehensive infrastructure
            logger.info("🚀 Deploying comprehensive honeypot infrastructure...")
            deployment_results = await unified_honeypot_manager.deploy_comprehensive_honeypot_infrastructure()
            
            self.deployment_results = deployment_results
            
            # Analyze deployment results
            success_count = self._count_successful_deployments(deployment_results)
            total_components = self._count_total_components(deployment_results)
            
            deployment_time = (datetime.utcnow() - self.deployment_start_time).total_seconds()
            
            if success_count >= 3:  # At least 3 components successful
                self.is_deployed = True
                logger.info("=" * 60)
                logger.info("🎯 HONEYPOT INFRASTRUCTURE DEPLOYED SUCCESSFULLY!")
                logger.info(f"   ✅ {success_count}/{total_components} components active")
                logger.info(f"   ⏱️  Deployment time: {deployment_time:.1f} seconds")
                logger.info("=" * 60)
                
                # Print component status
                await self._print_deployment_summary(deployment_results)
                
                # Print access information
                await self._print_access_information()
                
                # Start monitoring
                await self._start_deployment_monitoring()
                
            else:
                logger.error("=" * 60)
                logger.error("❌ HONEYPOT INFRASTRUCTURE DEPLOYMENT FAILED")
                logger.error(f"   Only {success_count}/{total_components} components deployed")
                logger.error("=" * 60)
                
                # Print failure details
                await self._print_failure_details(deployment_results)
            
            return {
                "success": self.is_deployed,
                "deployment_time": deployment_time,
                "components_deployed": success_count,
                "total_components": total_components,
                "results": deployment_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"💥 Deployment failed with exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _count_successful_deployments(self, results: Dict[str, Any]) -> int:
        """Count successful component deployments"""
        count = 0
        
        if results.get('orchestrator'):
            count += 1
            
        if results.get('cowrie_ssh'):
            count += 1
            
        web_honeypots = results.get('web_honeypots', {})
        count += sum(1 for result in web_honeypots.values() if result)
        
        db_honeypots = results.get('database_honeypots', {})
        count += sum(1 for result in db_honeypots.values() if result)
        
        ai_honeypots = results.get('ai_agent_honeypots', {})
        count += sum(1 for result in ai_honeypots.values() if result)
        
        if results.get('security_integration'):
            count += 1
            
        return count
    
    def _count_total_components(self, results: Dict[str, Any]) -> int:
        """Count total expected components"""
        return (
            1 +  # orchestrator
            1 +  # cowrie_ssh
            len(results.get('web_honeypots', {})) +
            len(results.get('database_honeypots', {})) +
            len(results.get('ai_agent_honeypots', {})) +
            1    # security_integration
        )
    
    async def _print_deployment_summary(self, results: Dict[str, Any]):
        """Print detailed deployment summary"""
        logger.info("\n📊 DEPLOYMENT SUMMARY:")
        logger.info("-" * 40)
        
        # Orchestrator
        if results.get('orchestrator'):
            logger.info("🎯 Honeypot Orchestrator: ACTIVE")
        else:
            logger.info("❌ Honeypot Orchestrator: FAILED")
        
        # Cowrie SSH
        if results.get('cowrie_ssh'):
            logger.info("🔐 Cowrie SSH Honeypot: ACTIVE (port 2222)")
        else:
            logger.info("❌ Cowrie SSH Honeypot: FAILED")
        
        # Web honeypots
        web_honeypots = results.get('web_honeypots', {})
        for hp_type, result in web_honeypots.items():
            if result:
                port = "8080" if "8080" in hp_type else "8443"
                protocol = "HTTPS" if "https" in hp_type else "HTTP"
                logger.info(f"🌐 {protocol} Web Honeypot: ACTIVE (port {port})")
            else:
                logger.info(f"❌ {hp_type.upper()} Web Honeypot: FAILED")
        
        # Database honeypots
        db_honeypots = results.get('database_honeypots', {})
        db_ports = {"mysql": 13306, "postgresql": 15432, "redis": 16379}
        for db_type, result in db_honeypots.items():
            if result:
                port = db_ports.get(db_type, "unknown")
                logger.info(f"🗄️  {db_type.upper()} Database Honeypot: ACTIVE (port {port})")
            else:
                logger.info(f"❌ {db_type.upper()} Database Honeypot: FAILED")
        
        # AI agent honeypots
        ai_honeypots = results.get('ai_agent_honeypots', {})
        ai_ports = {"port_10104": 10104, "port_8000": 8000, "port_8080": 8080, "port_9000": 9000}
        for hp_config, result in ai_honeypots.items():
            if result:
                port = ai_ports.get(hp_config, "unknown")
                logger.info(f"🤖 AI Agent Honeypot: ACTIVE (port {port})")
            else:
                logger.info(f"❌ AI Agent Honeypot ({hp_config}): FAILED")
        
        # Security integration
        if results.get('security_integration'):
            logger.info("🔗 Security System Integration: ACTIVE")
        else:
            logger.info("❌ Security System Integration: FAILED")
    
    async def _print_access_information(self):
        """Print access information for administrators"""
        logger.info("\n🔧 ADMINISTRATOR ACCESS INFORMATION:")
        logger.info("-" * 40)
        logger.info("Management API Endpoints:")
        logger.info("  • Status: GET /api/v1/honeypot/status")
        logger.info("  • Events: GET /api/v1/honeypot/events")
        logger.info("  • Analytics: GET /api/v1/honeypot/analytics/dashboard")
        logger.info("  • Threat Intel: GET /api/v1/honeypot/intelligence/report")
        logger.info("  • Health Check: GET /api/v1/honeypot/health")
        logger.info("\nDatabase Location:")
        logger.info("  • SQLite: /opt/sutazaiapp/backend/data/honeypot.db")
        logger.info("\nLog Files:")
        logger.info("  • Deployment: /opt/sutazaiapp/backend/logs/honeypot_deployment.log")
        logger.info("  • Runtime: /opt/sutazaiapp/backend/logs/")
    
    async def _print_failure_details(self, results: Dict[str, Any]):
        """Print failure details for troubleshooting"""
        logger.error("\n🔍 FAILURE ANALYSIS:")
        logger.error("-" * 40)
        
        failed_components = []
        
        if not results.get('orchestrator'):
            failed_components.append("Honeypot Orchestrator")
        
        if not results.get('cowrie_ssh'):
            failed_components.append("Cowrie SSH Honeypot")
        
        web_honeypots = results.get('web_honeypots', {})
        for hp_type, result in web_honeypots.items():
            if not result:
                failed_components.append(f"{hp_type.upper()} Web Honeypot")
        
        db_honeypots = results.get('database_honeypots', {})
        for db_type, result in db_honeypots.items():
            if not result:
                failed_components.append(f"{db_type.upper()} Database Honeypot")
        
        ai_honeypots = results.get('ai_agent_honeypots', {})
        for hp_config, result in ai_honeypots.items():
            if not result:
                failed_components.append(f"AI Agent Honeypot ({hp_config})")
        
        if not results.get('security_integration'):
            failed_components.append("Security System Integration")
        
        logger.error("Failed Components:")
        for component in failed_components:
            logger.error(f"  ❌ {component}")
        
        logger.error("\n🛠️  TROUBLESHOOTING SUGGESTIONS:")
        logger.error("1. Check port availability (ports may be in use)")
        logger.error("2. Verify permissions (deployment may require sudo)")
        logger.error("3. Check system dependencies")
        logger.error("4. Review logs for specific error messages")
        logger.error("5. Ensure sufficient system resources")
    
    async def _start_deployment_monitoring(self):
        """Start monitoring the deployed infrastructure"""
        logger.info("\n📈 Starting deployment monitoring...")
        
        try:
            # Get initial status
            status = await unified_honeypot_manager.get_comprehensive_status()
            
            logger.info("🎯 Initial Status Summary:")
            if status.get('components', {}).get('orchestrator', {}).get('infrastructure_status') == 'active':
                active_honeypots = status['components']['orchestrator'].get('active_honeypots', 0)
                logger.info(f"   Active Honeypots: {active_honeypots}")
            
            if status.get('recent_activity'):
                events_1h = status['recent_activity'].get('events_last_hour', 0)
                logger.info(f"   Events (last hour): {events_1h}")
            
            logger.info("✅ Monitoring system initialized")
            
        except Exception as e:
            logger.error(f"⚠️  Monitoring initialization failed: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        if not self.is_deployed:
            return {
                "deployed": False,
                "message": "Honeypot infrastructure not deployed"
            }
        
        try:
            status = await unified_honeypot_manager.get_comprehensive_status()
            status["deployment_results"] = self.deployment_results
            status["deployment_time"] = self.deployment_start_time.isoformat() if self.deployment_start_time else None
            return status
            
        except Exception as e:
            return {
                "deployed": True,
                "error": f"Status check failed: {str(e)}"
            }
    
    async def undeploy(self) -> Dict[str, Any]:
        """Undeploy honeypot infrastructure"""
        logger.info("🛑 Undeploying honeypot infrastructure...")
        
        try:
            await unified_honeypot_manager.undeploy_all()
            self.is_deployed = False
            
            logger.info("✅ Honeypot infrastructure undeployed successfully")
            
            return {
                "success": True,
                "message": "Infrastructure undeployed successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Undeployment failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Global deployment manager instance
deployment_manager = HoneypotDeploymentManager()

async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="SutazAI Honeypot Infrastructure Deployment")
    parser.add_argument(
        "action", 
        choices=["deploy", "status", "undeploy"],
        help="Action to perform"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path (JSON)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1
    
    try:
        if args.action == "deploy":
            result = await deployment_manager.deploy_full_infrastructure(config)
            
            if result.get("success"):
                logger.info("\n🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
                logger.info("The SutazAI honeypot infrastructure is now active and monitoring for threats.")
                return 0
            else:
                logger.error("\n💥 DEPLOYMENT FAILED!")
                logger.error("Check the logs above for specific failure details.")
                return 1
                
        elif args.action == "status":
            status = await deployment_manager.get_status()
            
            print("\n" + "=" * 50)
            print("SUTAZAI HONEYPOT INFRASTRUCTURE STATUS")
            print("=" * 50)
            print(json.dumps(status, indent=2, default=str))
            return 0
            
        elif args.action == "undeploy":
            result = await deployment_manager.undeploy()
            
            if result.get("success"):
                logger.info("✅ Infrastructure undeployed successfully")
                return 0
            else:
                logger.error(f"❌ Undeployment failed: {result.get('error')}")
                return 1
                
    except KeyboardInterrupt:
        logger.info("\n⚠️  Deployment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)