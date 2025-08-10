#!/usr/bin/env python3
"""
Test GPU detection methods for WSL2
"""
import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import subprocess
import os
import sys

def test_nvidia_smi_paths():
    """Test various nvidia-smi paths"""
    print("=== Testing nvidia-smi paths ===")
    
    paths = [
        '/mnt/c/Windows/System32/nvidia-smi.exe',
        '/mnt/c/Windows/system32/nvidia-smi.exe',
        '/mnt/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe',
        '/usr/lib/wsl/lib/nvidia-smi',
        'nvidia-smi.exe',
        'nvidia-smi'
    ]
    
    for path in paths:
        print(f"\nTesting: {path}")
        
        if path.startswith('/'):
            exists = os.path.exists(path)
            print(f"  Exists: {exists}")
            if not exists:
                continue
        
        try:
            # Test basic info query
            result = subprocess.run([path, '--query-gpu=name,driver_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                print(f"  SUCCESS - Basic info:")
                for line in result.stdout.strip().split('\n'):
                    print(f"    {line}")
                
                # Test XML query
                try:
                    xml_result = subprocess.run([path, '-q', '-x'], 
                                              capture_output=True, text=True, timeout=5)
                    if xml_result.returncode == 0:
                        print(f"  XML support: YES")
                    else:
                        print(f"  XML support: NO")
                except (IOError, OSError, FileNotFoundError) as e:
                    # TODO: Review this exception handling
                    logger.error(f"Unexpected exception: {e}", exc_info=True)
                    print(f"  XML support: ERROR")
                    
                # Test stats query
                try:
                    stats_result = subprocess.run([path, '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                                                capture_output=True, text=True, timeout=5)
                    if stats_result.returncode == 0:
                        print(f"  Stats query: SUCCESS")
                        for line in stats_result.stdout.strip().split('\n'):
                            print(f"    {line}")
                    else:
                        print(f"  Stats query: FAILED")
                except Exception as e:
                    print(f"  Stats query: ERROR - {e}")
                
            else:
                print(f"  FAILED - Return code: {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr.strip()}")
                    
        except Exception as e:
            print(f"  ERROR: {e}")

def test_wsl_environment():
    """Test WSL environment detection"""
    print("\n=== Testing WSL Environment ===")
    
    # Check /proc/version
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            print(f"Proc version: {version_info.strip()}")
            
            if 'microsoft' in version_info or 'wsl' in version_info:
                print("WSL detected: YES")
                if 'wsl2' in version_info or 'microsoft-standard-wsl2' in version_info:
                    print("WSL2 detected: YES")
                else:
                    print("WSL2 detected: NO (WSL1)")
            else:
                print("WSL detected: NO")
    
    # Check environment variables
    wsl_vars = ['WSL_DISTRO_NAME', 'WSLENV', 'WSLG_GPU', 'WSL2_GUI_APPS_ENABLED']
    for var in wsl_vars:
        value = os.environ.get(var)
        print(f"${var}: {value if value else 'Not set'}")
    
    # Check GPU passthrough indicators
    gpu_indicators = ['/dev/dxg', '/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm']
    for indicator in gpu_indicators:
        exists = os.path.exists(indicator)
        print(f"{indicator}: {'EXISTS' if exists else 'Not found'}")

def test_alternative_methods():
    """Test alternative GPU detection methods"""
    print("\n=== Testing Alternative Methods ===")
    
    # Test gpustat
    try:
        result = subprocess.run(['gpustat', '--json'], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            print("gpustat: AVAILABLE")
            print(f"  Output preview: {result.stdout[:100]}...")
        else:
            print("gpustat: NOT WORKING")
    except FileNotFoundError:
        print("gpustat: NOT INSTALLED")
        # Try to install it
        try:
            print("  Attempting to install gpustat...")
            install_result = subprocess.run(['pip', 'install', 'gpustat'], capture_output=True, text=True, timeout=30)
            if install_result.returncode == 0:
                print("  Installation successful")
                # Retry
                result = subprocess.run(['gpustat', '--json'], capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    print("  gpustat now working!")
                else:
                    print("  gpustat still not working after install")
            else:
                print(f"  Installation failed: {install_result.stderr}")
        except Exception as e:
            print(f"  Installation error: {e}")
    except Exception as e:
        print(f"gpustat: ERROR - {e}")
    
    # Test pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"pynvml: AVAILABLE - {device_count} device(s)")
        
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            print(f"  Device 0: {name}")
            print(f"  Driver: {driver_version}")
    except ImportError:
        print("pynvml: NOT INSTALLED")
        try:
            print("  Attempting to install nvidia-ml-py...")
            install_result = subprocess.run(['pip', 'install', 'nvidia-ml-py'], capture_output=True, text=True, timeout=30)
            if install_result.returncode == 0:
                print("  Installation successful")
            else:
                print(f"  Installation failed: {install_result.stderr}")
        except Exception as e:
            print(f"  Installation error: {e}")
    except Exception as e:
        print(f"pynvml: ERROR - {e}")

if __name__ == "__main__":
    print("GPU Detection Test for WSL2")
    print("=" * 50)
    
    test_wsl_environment()
    test_nvidia_smi_paths()
    test_alternative_methods()
    
    print("\n=== Test Complete ===")