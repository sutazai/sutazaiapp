#!/usr/bin/env python3
"""
SutazAI Deployment Validation
Final validation of the complete system deployment
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, capture_output=True):
    """Run shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def validate_docker_environment():
    """Validate Docker environment"""
    print("🐳 Validating Docker Environment...")
    
    # Check Docker
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print("  ❌ Docker not available")
        return False
    print(f"  ✅ Docker: {stdout.strip()}")
    
    # Check Docker Compose
    success, stdout, stderr = run_command("docker compose version")
    if not success:
        print("  ❌ Docker Compose not available")
        return False
    print(f"  ✅ Docker Compose: {stdout.strip()}")
    
    # Validate compose configuration
    success, stdout, stderr = run_command("docker compose config --quiet")
    if not success:
        print(f"  ❌ Docker Compose config invalid: {stderr}")
        return False
    print("  ✅ Docker Compose configuration valid")
    
    return True

def validate_system_architecture():
    """Validate complete system architecture"""
    print("🏗️ Validating System Architecture...")
    
    # Count services in docker-compose
    try:
        success, stdout, stderr = run_command("docker compose config --services")
        if success:
            services = stdout.strip().split('\n')
            service_count = len([s for s in services if s.strip()])
            print(f"  ✅ {service_count} services configured")
            
            # List key services
            key_services = [
                'sutazai-backend', 'sutazai-streamlit', 'postgres', 'redis',
                'qdrant', 'chromadb', 'ollama', 'nginx', 'prometheus', 'grafana'
            ]
            
            present_services = []
            missing_services = []
            
            for service in key_services:
                if service in services:
                    present_services.append(service)
                else:
                    missing_services.append(service)
            
            print(f"  ✅ Core services present: {len(present_services)}")
            if missing_services:
                print(f"  ⚠️  Missing services: {missing_services}")
            
        else:
            print(f"  ❌ Could not read compose services: {stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error validating architecture: {e}")
        return False
    
    return True

def validate_ai_components():
    """Validate AI components and agents"""
    print("🤖 Validating AI Components...")
    
    ai_services = [
        'autogpt', 'localagi', 'tabby', 'browser-use', 'skyvern',
        'documind', 'finrobot', 'gpt-engineer', 'aider', 'bigagi', 'agentzero'
    ]
    
    success, stdout, stderr = run_command("docker compose config --services")
    if success:
        services = stdout.strip().split('\n')
        
        present_ai = []
        missing_ai = []
        
        for ai_service in ai_services:
            if ai_service in services:
                present_ai.append(ai_service)
            else:
                missing_ai.append(ai_service)
        
        print(f"  ✅ AI agents configured: {len(present_ai)}")
        print(f"  📋 AI services: {', '.join(present_ai)}")
        
        if missing_ai:
            print(f"  ⚠️  Missing AI services: {missing_ai}")
        
        return len(present_ai) >= 8  # Require at least 8 AI services
    
    return False

def validate_infrastructure():
    """Validate infrastructure components"""
    print("🔧 Validating Infrastructure...")
    
    infra_components = {
        'databases': ['postgres', 'redis', 'qdrant', 'chromadb'],
        'models': ['ollama'],
        'monitoring': ['prometheus', 'grafana', 'node-exporter'],
        'proxy': ['nginx'],
        'core': ['sutazai-backend', 'sutazai-streamlit']
    }
    
    success, stdout, stderr = run_command("docker compose config --services")
    if success:
        services = stdout.strip().split('\n')
        
        all_valid = True
        for category, components in infra_components.items():
            present = [c for c in components if c in services]
            missing = [c for c in components if c not in services]
            
            print(f"  ✅ {category.title()}: {len(present)}/{len(components)} present")
            if missing:
                print(f"    ⚠️  Missing: {missing}")
                if category in ['core', 'databases']:  # Critical components
                    all_valid = False
        
        return all_valid
    
    return False

def validate_security_config():
    """Validate security configuration"""
    print("🔐 Validating Security Configuration...")
    
    # Check .env file
    if not Path('.env').exists():
        print("  ❌ .env file missing")
        return False
    
    with open('.env', 'r') as f:
        env_content = f.read()
    
    security_vars = [
        'SECRET_KEY', 'JWT_SECRET_KEY', 'POSTGRES_PASSWORD',
        'ENABLE_SECURITY', 'ENABLE_RATE_LIMITING'
    ]
    
    missing_security = []
    for var in security_vars:
        if var not in env_content:
            missing_security.append(var)
        else:
            print(f"  ✅ {var} configured")
    
    if missing_security:
        print(f"  ❌ Missing security variables: {missing_security}")
        return False
    
    # Check for default/weak passwords
    weak_patterns = ['password', 'admin', 'test', '123']
    for pattern in weak_patterns:
        if pattern in env_content.lower():
            print(f"  ⚠️  Potential weak password detected: {pattern}")
    
    print("  ✅ Security configuration complete")
    return True

def validate_deployment_readiness():
    """Validate deployment readiness"""
    print("🚀 Validating Deployment Readiness...")
    
    # Check deployment script
    deploy_script = Path('deploy.sh')
    if not deploy_script.exists():
        print("  ❌ deploy.sh missing")
        return False
    
    if not deploy_script.stat().st_mode & 0o111:  # Check if executable
        print("  ❌ deploy.sh not executable")
        return False
    
    print("  ✅ Deployment script ready")
    
    # Check required directories
    required_dirs = ['backend', 'frontend', 'docker', 'nginx']
    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).is_dir():
            missing_dirs.append(dir_name)
        else:
            print(f"  ✅ Directory: {dir_name}")
    
    if missing_dirs:
        print(f"  ❌ Missing directories: {missing_dirs}")
        return False
    
    # Check core files
    core_files = [
        'backend/enhanced_main.py',
        'backend/sutazai_core.py', 
        'frontend/streamlit_app.py',
        'docker-compose.yml'
    ]
    
    missing_files = []
    for file_path in core_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ File: {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ All deployment files ready")
    return True

def generate_deployment_report():
    """Generate final deployment report"""
    print("📊 Generating Deployment Report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": "SutazAI AGI/ASI Autonomous System",
        "version": "1.1.0",
        "status": "ready",
        "components": {}
    }
    
    # Count components
    success, stdout, stderr = run_command("docker compose config --services")
    if success:
        services = stdout.strip().split('\n')
        service_count = len([s for s in services if s.strip()])
        report["components"]["total_services"] = service_count
    
    # Count AI agents
    ai_services = [
        'autogpt', 'localagi', 'tabby', 'browser-use', 'skyvern',
        'documind', 'finrobot', 'gpt-engineer', 'aider', 'bigagi', 'agentzero'
    ]
    
    if success:
        services = stdout.strip().split('\n')
        ai_count = len([s for s in ai_services if s in services])
        report["components"]["ai_agents"] = ai_count
    
    # File sizes
    key_files = ['docker-compose.yml', '.env', 'deploy.sh', 'README.md']
    file_info = {}
    for file_path in key_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            file_info[file_path] = f"{size:,} bytes"
    
    report["file_sizes"] = file_info
    
    # Save report
    with open('deployment_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("  ✅ Report saved to deployment_report.json")
    return report

def main():
    """Run complete deployment validation"""
    print("🔍 SutazAI Deployment Validation")
    print("=" * 60)
    
    validations = [
        ("Docker Environment", validate_docker_environment),
        ("System Architecture", validate_system_architecture), 
        ("AI Components", validate_ai_components),
        ("Infrastructure", validate_infrastructure),
        ("Security Config", validate_security_config),
        ("Deployment Readiness", validate_deployment_readiness)
    ]
    
    passed = 0
    total = len(validations)
    
    for name, validation_func in validations:
        print(f"\n{name}:")
        try:
            if validation_func():
                passed += 1
                print(f"  ✅ {name} validation passed")
            else:
                print(f"  ❌ {name} validation failed")
        except Exception as e:
            print(f"  ❌ {name} validation error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📈 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 DEPLOYMENT VALIDATION SUCCESSFUL!")
        print("🚀 System ready for production deployment")
        
        # Generate report
        report = generate_deployment_report()
        print(f"\n📋 System Summary:")
        print(f"  • Total Services: {report['components'].get('total_services', 'N/A')}")
        print(f"  • AI Agents: {report['components'].get('ai_agents', 'N/A')}")
        print(f"  • Version: {report['version']}")
        
        return 0
    else:
        print("⚠️  DEPLOYMENT VALIDATION INCOMPLETE")
        print("Please review the failed checks above")
        return 1

if __name__ == "__main__":
    sys.exit(main())