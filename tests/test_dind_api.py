#!/usr/bin/env python3
"""
Test DinD MCP API Integration
Real test of the Docker-in-Docker MCP orchestrator
"""
import subprocess
import json
import sys

def test_dind_container():
    """Test if DinD container is running"""
    print("🔍 Testing DinD Container Status...")
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=sutazai-mcp-orchestrator", "--format", "{{.Status}}"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0 and "Up" in result.stdout:
            print(f"✅ DinD Container: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ DinD Container not running: {result.stdout}")
            return False
    except Exception as e:
        print(f"❌ Error checking DinD: {e}")
        return False

def test_dind_docker_api():
    """Test DinD internal Docker API"""
    print("\n🔍 Testing DinD Internal Docker API...")
    
    try:
        result = subprocess.run(
            ["docker", "exec", "sutazai-mcp-orchestrator", "docker", "version", "--format", "json"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            version_info = json.loads(result.stdout)
            print(f"✅ DinD Docker API: v{version_info['Server']['Version']}")
            return True
        else:
            print(f"❌ DinD Docker API failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing DinD API: {e}")
        return False

def test_mcp_containers():
    """Test MCP containers inside DinD"""
    print("\n🔍 Testing MCP Containers in DinD...")
    
    try:
        result = subprocess.run(
            ["docker", "exec", "sutazai-mcp-orchestrator", "docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            containers = result.stdout.strip().split('\n') if result.stdout.strip() else []
            print(f"📦 MCP Containers ({len(containers)}): {containers}")
            return containers
        else:
            print(f"❌ Failed to list MCP containers: {result.stderr}")
            return []
    except Exception as e:
        print(f"❌ Error listing MCP containers: {e}")
        return []

def test_dind_ports():
    """Test DinD exposed ports"""
    print("\n🔍 Testing DinD Port Mappings...")
    
    try:
        result = subprocess.run(
            ["docker", "port", "sutazai-mcp-orchestrator"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("🌐 DinD Port Mappings:")
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"  {line}")
            return True
        else:
            print(f"❌ Failed to get port mappings: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking ports: {e}")
        return False

def test_orchestrator_api():
    """Test DinD orchestrator API"""
    print("\n🔍 Testing DinD Orchestrator API...")
    
    try:
        result = subprocess.run(
            ["curl", "-s", "-m", "5", "http://localhost:18080/health"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0 and result.stdout:
            print(f"✅ Orchestrator API: {result.stdout}")
            return True
        else:
            print("⚠️ Orchestrator API not responding (expected - may not be implemented)")
            return False
    except Exception as e:
        print(f"⚠️ Orchestrator API test failed: {e}")
        return False

def main():
    """Run all DinD tests"""
    print("=" * 60)
    print("🐳 DIND MCP ORCHESTRATOR TEST")
    print("=" * 60)
    
    results = {
        "dind_container": test_dind_container(),
        "dind_api": test_dind_docker_api(),
        "port_mappings": test_dind_ports(),
        "orchestrator_api": test_orchestrator_api(),
    }
    
    # Test MCP containers
    mcp_containers = test_mcp_containers()
    results["mcp_containers"] = len(mcp_containers) > 0
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if results["dind_container"] and results["dind_api"]:
        print("🎯 CONCLUSION: DinD infrastructure is REAL and WORKING!")
        print("   Ready for MCP container deployment")
    else:
        print("⚠️ CONCLUSION: DinD infrastructure needs attention")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)