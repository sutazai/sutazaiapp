#!/usr/bin/env python3
"""
Identify which GPU monitoring method is working correctly
"""
import subprocess
import os
import sys
import json

def test_method_1_windows_nvidia_smi():
    """Test Windows nvidia-smi.exe"""
    print("=== Method 1: Windows nvidia-smi.exe ===")
    paths = ['/mnt/c/Windows/System32/nvidia-smi.exe', '/mnt/c/Windows/system32/nvidia-smi.exe']
    
    for path in paths:
        if os.path.exists(path):
            print(f"Testing: {path}")
            try:
                # Test the exact query our monitor uses
                result = subprocess.run([
                    path, 
                    '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name,driver_version,gpu_uuid',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=3)
                
                if result.returncode == 0 and result.stdout.strip():
                    print(f"  SUCCESS - Raw output:")
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        print(f"    GPU {i}: {line}")
                        parts = [p.strip() for p in line.split(', ')]
                        if len(parts) >= 4:
                            usage = float(parts[0]) if parts[0] not in ['[Not Supported]', 'N/A', ''] else 0
                            mem_used = float(parts[1]) if parts[1] not in ['[Not Supported]', 'N/A', ''] else 0
                            mem_total = float(parts[2]) if parts[2] not in ['[Not Supported]', 'N/A', ''] else 1
                            temp = float(parts[3]) if parts[3] not in ['[Not Supported]', 'N/A', ''] else 0
                            power = float(parts[4]) if len(parts) > 4 and parts[4] not in ['[Not Supported]', 'N/A', ''] else 0
                            name = parts[5] if len(parts) > 5 else 'Unknown'
                            
                            print(f"    Parsed - Usage: {usage}%, Memory: {mem_used}/{mem_total}MB, Temp: {temp}°C, Power: {power}W")
                            print(f"    Name: {name}")
                            return True, f"Windows nvidia-smi: {path}"
                else:
                    print(f"  FAILED - Return code: {result.returncode}")
            except Exception as e:
                print(f"  ERROR: {e}")
    
    return False, "Windows nvidia-smi failed"

def test_method_2_wsl2_nvidia_smi():
    """Test WSL2 nvidia-smi"""
    print("\n=== Method 2: WSL2 nvidia-smi ===")
    path = '/usr/lib/wsl/lib/nvidia-smi'
    
    if os.path.exists(path):
        print(f"Testing: {path}")
        try:
            result = subprocess.run([
                path, 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,name,driver_version,gpu_uuid',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and result.stdout.strip():
                print(f"  SUCCESS - Raw output:")
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    print(f"    GPU {i}: {line}")
                    parts = [p.strip() for p in line.split(', ')]
                    if len(parts) >= 4:
                        usage = float(parts[0]) if parts[0] not in ['[Not Supported]', 'N/A', ''] else 0
                        mem_used = float(parts[1]) if parts[1] not in ['[Not Supported]', 'N/A', ''] else 0
                        mem_total = float(parts[2]) if parts[2] not in ['[Not Supported]', 'N/A', ''] else 1
                        temp = float(parts[3]) if parts[3] not in ['[Not Supported]', 'N/A', ''] else 0
                        
                        print(f"    Parsed - Usage: {usage}%, Memory: {mem_used}/{mem_total}MB, Temp: {temp}°C")
                        return True, f"WSL2 nvidia-smi: {path}"
            else:
                print(f"  FAILED - Return code: {result.returncode}")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"  NOT FOUND: {path}")
    
    return False, "WSL2 nvidia-smi failed"

def test_method_3_gpustat():
    """Test gpustat"""
    print("\n=== Method 3: gpustat ===")
    try:
        result = subprocess.run(['gpustat', '--json'], capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            print(f"  SUCCESS - Raw JSON output:")
            data = json.loads(result.stdout)
            print(f"    {json.dumps(data, indent=2)}")
            
            if 'gpus' in data and data['gpus']:
                gpu = data['gpus'][0]
                print(f"  Parsed first GPU:")
                for key, value in gpu.items():
                    print(f"    {key}: {value}")
                return True, "gpustat"
        else:
            print(f"  FAILED - Return code: {result.returncode}")
            if result.stderr:
                print(f"    Error: {result.stderr}")
    except FileNotFoundError:
        print("  NOT INSTALLED")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    return False, "gpustat failed"

def test_method_4_pynvml():
    """Test pynvml"""
    print("\n=== Method 4: pynvml ===")
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"  Device count: {device_count}")
        
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get stats
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                print(f"  Utilization: {gpu_util}%")
            except Exception as e:
                print(f"  Utilization error: {e}")
                gpu_util = 0
            
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / 1024 / 1024  # MB
                mem_total = mem_info.total / 1024 / 1024  # MB
                print(f"  Memory: {mem_used:.0f}/{mem_total:.0f}MB")
            except Exception as e:
                print(f"  Memory error: {e}")
                mem_used = mem_total = 0
            
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                print(f"  Temperature: {temp}°C")
            except Exception as e:
                print(f"  Temperature error: {e}")
                temp = 0
            
            try:
                name_raw = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name_raw, bytes):
                    name = name_raw.decode('utf-8')
                else:
                    name = str(name_raw)
                print(f"  Name: {name}")
            except Exception as e:
                print(f"  Name error: {e}")
                name = "Unknown"
            
            if gpu_util > 0 or temp > 0 or mem_used > 0:
                return True, "pynvml"
            else:
                return False, "pynvml - no meaningful stats"
                
    except ImportError:
        print("  NOT INSTALLED")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    return False, "pynvml failed"

def test_our_monitor_gpu_detection():
    """Test what our monitor actually detected"""
    print("\n=== Testing Our Monitor's Detection ===")
    
    # Import our monitor class
    sys.path.append('/opt/sutazaiapp/scripts/monitoring')
    try:
        from static_monitor import EnhancedMonitor
        
        # Create monitor instance
        monitor = EnhancedMonitor()
        
        print(f"GPU Available: {monitor.gpu_available}")
        print(f"GPU Driver Type: {monitor.gpu_driver_type}")
        print(f"GPU Info: {monitor.gpu_info}")
        
        if hasattr(monitor, 'nvidia_smi_path'):
            print(f"Working nvidia-smi path: {monitor.nvidia_smi_path}")
        
        # Get actual stats
        gpu_stats = monitor.get_gpu_stats()
        print(f"GPU Stats: {gpu_stats}")
        
        monitor.cleanup()
        
        return gpu_stats['available'], f"Monitor detected: {monitor.gpu_driver_type}"
        
    except Exception as e:
        print(f"Monitor test error: {e}")
        return False, f"Monitor test failed: {e}"

if __name__ == "__main__":
    print("GPU Method Identification Test")
    print("=" * 50)
    
    methods = []
    
    # Test each method
    success1, result1 = test_method_1_windows_nvidia_smi()
    if success1: methods.append(result1)
    
    success2, result2 = test_method_2_wsl2_nvidia_smi()
    if success2: methods.append(result2)
    
    success3, result3 = test_method_3_gpustat()
    if success3: methods.append(result3)
    
    success4, result4 = test_method_4_pynvml()
    if success4: methods.append(result4)
    
    # Test our monitor
    success5, result5 = test_our_monitor_gpu_detection()
    if success5: methods.append(result5)
    
    print("\n" + "=" * 50)
    print("SUMMARY OF WORKING METHODS:")
    print("=" * 50)
    
    if methods:
        for i, method in enumerate(methods, 1):
            print(f"{i}. {method}")
    else:
        print("No working methods found!")
    
    print("\nThe method showing temperature data (55°C) in the monitor is the one actually working.")