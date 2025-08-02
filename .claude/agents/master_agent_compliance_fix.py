#!/usr/bin/env python3
"""
Master Agent Compliance Fix Script

This is the main script that coordinates the complete agent compliance fix process:
1. Runs compliance check
2. Creates fix scripts for non-compliant agents
3. Executes all fixes
4. Verifies post-fix compliance
5. Generates final report
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class MasterComplianceFixer:
    """Coordinates the complete agent compliance fixing process"""
    
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/.claude/agents"):
        self.agents_dir = Path(agents_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_compliance_check(self) -> str:
        """Run initial compliance check"""
        print("🔍 Step 1: Running initial compliance check...")
        
        try:
            result = subprocess.run([
                sys.executable, 
                str(self.agents_dir / "agent_compliance_checker.py")
            ], capture_output=True, text=True, cwd=self.agents_dir)
            
            if result.returncode == 0:
                # Find the latest compliance report
                reports = list(self.agents_dir.glob("*compliance_report*.json"))
                if reports:
                    latest_report = max(reports, key=lambda x: x.stat().st_mtime)
                    print(f"  ✅ Compliance check complete: {latest_report}")
                    return str(latest_report)
                else:
                    print("  ❌ No compliance report generated")
                    return None
            else:
                print(f"  ❌ Compliance check failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"  ❌ Error running compliance check: {e}")
            return None
    
    def create_fixes(self, compliance_report: str) -> bool:
        """Create fix scripts for non-compliant agents"""
        print("🔧 Step 2: Creating fix scripts...")
        
        try:
            result = subprocess.run([
                sys.executable,
                str(self.agents_dir / "create_agent_fixes.py"),
                compliance_report
            ], capture_output=True, text=True, cwd=self.agents_dir)
            
            if result.returncode == 0:
                print("  ✅ Fix scripts created successfully")
                print(result.stdout)
                return True
            else:
                print(f"  ❌ Failed to create fix scripts: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error creating fixes: {e}")
            return False
    
    def run_all_fixes(self) -> bool:
        """Execute all fix scripts"""
        print("⚡ Step 3: Executing all fixes...")
        
        master_fix_script = self.agents_dir / "run_all_agent_fixes.sh"
        
        if not master_fix_script.exists():
            print("  ❌ Master fix script not found")
            return False
        
        try:
            result = subprocess.run([
                "bash", str(master_fix_script)
            ], capture_output=True, text=True, cwd=self.agents_dir)
            
            print(result.stdout)
            
            if result.returncode == 0:
                print("  ✅ All fixes executed successfully")
                return True
            else:
                print(f"  ⚠️  Some fixes may have failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error running fixes: {e}")
            return False
    
    def verify_post_fix_compliance(self) -> str:
        """Run post-fix compliance verification"""
        print("🔍 Step 4: Verifying post-fix compliance...")
        
        try:
            result = subprocess.run([
                sys.executable, 
                str(self.agents_dir / "agent_compliance_checker.py")
            ], capture_output=True, text=True, cwd=self.agents_dir)
            
            if result.returncode == 0:
                # Find the latest compliance report (post-fix)
                reports = list(self.agents_dir.glob("*compliance_report*.json"))
                if reports:
                    latest_report = max(reports, key=lambda x: x.stat().st_mtime)
                    print(f"  ✅ Post-fix compliance check complete: {latest_report}")
                    return str(latest_report)
                else:
                    print("  ❌ No post-fix compliance report generated")
                    return None
            else:
                print(f"  ❌ Post-fix compliance check failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"  ❌ Error running post-fix compliance check: {e}")
            return None
    
    def generate_final_report(self, pre_report: str, post_report: str) -> str:
        """Generate final compliance improvement report"""
        print("📊 Step 5: Generating final report...")
        
        try:
            # Load both reports
            with open(pre_report, 'r') as f:
                pre_data = json.load(f)
            
            with open(post_report, 'r') as f:
                post_data = json.load(f)
            
            # Calculate improvements
            pre_compliant = pre_data['compliant_agents']
            pre_total = pre_data['total_agents']
            post_compliant = post_data['compliant_agents']
            post_total = post_data['total_agents']
            
            improvement = post_compliant - pre_compliant
            pre_percentage = (pre_compliant / pre_total) * 100
            post_percentage = (post_compliant / post_total) * 100
            improvement_percentage = post_percentage - pre_percentage
            
            # Find which agents were fixed
            fixed_agents = []
            for agent_name, post_analysis in post_data['agents'].items():
                if post_analysis['compliant']:
                    pre_analysis = pre_data['agents'].get(agent_name, {})
                    if not pre_analysis.get('compliant', False):
                        fixed_agents.append(agent_name.replace('.md', ''))
            
            # Create final report
            final_report = {
                "timestamp": datetime.now().isoformat(),
                "process_summary": {
                    "pre_fix": {
                        "compliant_agents": pre_compliant,
                        "total_agents": pre_total,
                        "compliance_rate": f"{pre_percentage:.1f}%"
                    },
                    "post_fix": {
                        "compliant_agents": post_compliant,
                        "total_agents": post_total,
                        "compliance_rate": f"{post_percentage:.1f}%"
                    },
                    "improvement": {
                        "agents_fixed": improvement,
                        "percentage_improvement": f"{improvement_percentage:.1f}%",
                        "fixed_agents": fixed_agents
                    }
                },
                "reports": {
                    "pre_fix_report": pre_report,
                    "post_fix_report": post_report
                },
                "recommendations": self._generate_recommendations(post_data)
            }
            
            # Save final report
            final_report_file = f"final_compliance_report_{self.timestamp}.json"
            final_report_path = self.agents_dir / final_report_file
            
            with open(final_report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Final report saved: {final_report_path}")
            
            # Print summary
            self._print_final_summary(final_report)
            
            return str(final_report_path)
            
        except Exception as e:
            print(f"  ❌ Error generating final report: {e}")
            return None
    
    def _generate_recommendations(self, post_data: Dict) -> List[str]:
        """Generate recommendations based on post-fix status"""
        recommendations = []
        
        non_compliant_count = post_data['non_compliant_agents']
        
        if non_compliant_count == 0:
            recommendations.append("🎉 All agents are now compliant! Consider setting up automated compliance checking in CI/CD.")
        else:
            recommendations.extend([
                f"🔧 {non_compliant_count} agents still need manual attention",
                "📋 Review individual agent issues in the post-fix compliance report",
                "🔄 Consider running the fix process again for remaining issues",
                "🚀 Implement automated compliance checking in the deployment pipeline"
            ])
        
        recommendations.extend([
            "📝 Update agent documentation to include compliance requirements",
            "🎯 Train team members on CLAUDE.md rules and compliance process",
            "⚡ Set up monitoring to catch compliance issues early"
        ])
        
        return recommendations
    
    def _print_final_summary(self, final_report: Dict):
        """Print a formatted final summary"""
        summary = final_report['process_summary']
        
        print("\n" + "="*60)
        print("🎯 MASTER AGENT COMPLIANCE FIX SUMMARY")
        print("="*60)
        
        print(f"📊 BEFORE FIX:")
        print(f"   Compliant: {summary['pre_fix']['compliant_agents']}/{summary['pre_fix']['total_agents']} ({summary['pre_fix']['compliance_rate']})")
        
        print(f"📊 AFTER FIX:")
        print(f"   Compliant: {summary['post_fix']['compliant_agents']}/{summary['post_fix']['total_agents']} ({summary['post_fix']['compliance_rate']})")
        
        print(f"📈 IMPROVEMENT:")
        print(f"   Agents Fixed: {summary['improvement']['agents_fixed']}")
        print(f"   Improvement: {summary['improvement']['percentage_improvement']}")
        
        if summary['improvement']['fixed_agents']:
            print(f"✅ FIXED AGENTS ({len(summary['improvement']['fixed_agents'])}):")
            for agent in summary['improvement']['fixed_agents'][:10]:  # Show first 10
                print(f"   • {agent}")
            if len(summary['improvement']['fixed_agents']) > 10:
                print(f"   ... and {len(summary['improvement']['fixed_agents']) - 10} more")
        
        print(f"\n💡 NEXT STEPS:")
        for rec in final_report['recommendations']:
            print(f"   {rec}")
        
        print("\n" + "="*60)
    
    def run_complete_fix_process(self) -> bool:
        """Run the complete agent compliance fix process"""
        print("🚀 STARTING MASTER AGENT COMPLIANCE FIX PROCESS")
        print("="*60)
        
        # Step 1: Initial compliance check
        pre_report = self.run_compliance_check()
        if not pre_report:
            print("❌ Failed at step 1: Initial compliance check")
            return False
        
        # Step 2: Create fix scripts
        if not self.create_fixes(pre_report):
            print("❌ Failed at step 2: Creating fix scripts")
            return False
        
        # Step 3: Run all fixes
        if not self.run_all_fixes():
            print("⚠️  Step 3 completed with warnings: Some fixes may have failed")
        
        # Step 4: Post-fix compliance verification
        post_report = self.verify_post_fix_compliance()
        if not post_report:
            print("❌ Failed at step 4: Post-fix compliance verification")
            return False
        
        # Step 5: Generate final report
        final_report = self.generate_final_report(pre_report, post_report)
        if not final_report:
            print("❌ Failed at step 5: Final report generation")
            return False
        
        print("🎉 MASTER AGENT COMPLIANCE FIX PROCESS COMPLETED SUCCESSFULLY!")
        return True

def main():
    """Main function"""
    fixer = MasterComplianceFixer()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
Master Agent Compliance Fix Script

This script runs the complete agent compliance fixing process:

Usage:
  python3 master_agent_compliance_fix.py

Process Steps:
1. 🔍 Run initial compliance check
2. 🔧 Create fix scripts for non-compliant agents  
3. ⚡ Execute all fixes
4. 🔍 Verify post-fix compliance
5. 📊 Generate final improvement report

The script will create backups before making any changes and provide
detailed progress reporting throughout the process.
""")
        return
    
    success = fixer.run_complete_fix_process()
    
    if success:
        print("\n✅ All steps completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Process completed with errors. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()