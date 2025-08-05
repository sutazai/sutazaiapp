# WSL2 Catastrophic Service Recovery Guide

## Current System Status
- **WSL Version**: WSL2 Ubuntu 24.04.2 LTS
- **Kernel**: 6.6.87.2-microsoft-standard-WSL2
- **Error**: Wsl/Service/E_UNEXPECTED
- **System State**: Partially functional but service communication compromised

## Immediate Assessment

### Root Cause Analysis
The `Wsl/Service/E_UNEXPECTED` error typically indicates:
1. WSL service corruption on Windows host
2. Registry corruption affecting WSL components
3. Windows Update interference with WSL services
4. Hyper-V or virtualization stack issues
5. Windows file system corruption affecting WSL metadata

## CRITICAL RECOVERY PROCEDURES

### Phase 1: Windows-Side Diagnostics (Run in Windows PowerShell as Administrator)

```powershell
# Check WSL service status
Get-Service LxssManager
Get-Service LxssManagerUser

# Check WSL installations
wsl --list --verbose
wsl --status

# Check Windows features
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform

# Check Hyper-V status
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
```

### Phase 2: Service Recovery (Windows PowerShell as Administrator)

```powershell
# Stop all WSL processes
wsl --shutdown

# Stop WSL services
Stop-Service LxssManager -Force
Stop-Service LxssManagerUser -Force

# Clear WSL service cache
Remove-Item -Path "$env:LOCALAPPDATA\Packages\MicrosoftCorporationII.WindowsSubsystemForLinux_*" -Recurse -ErrorAction SilentlyContinue

# Restart services
Start-Service LxssManager
Start-Service LxssManagerUser

# Test WSL functionality
wsl --list --verbose
```

### Phase 3: Registry Cleanup (Windows PowerShell as Administrator)

```powershell
# Backup current registry keys
reg export "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Lxss" "C:\temp\wsl_backup.reg"

# Remove corrupted WSL registry entries (DANGEROUS - backup first!)
# Only run if other methods fail
reg delete "HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Lxss" /f
```

### Phase 4: Component Reinstallation (Windows PowerShell as Administrator)

```powershell
# Disable and re-enable WSL features
Disable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart
Disable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart

# Restart required
Restart-Computer

# After restart, re-enable features
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart

# Update WSL
wsl --update --web-download
wsl --version
```

## EMERGENCY RECOVERY PROCEDURES

### Option 1: Distribution Export/Import (Preserve Data)

```powershell
# Export current distribution
wsl --export Ubuntu-24.04 C:\temp\ubuntu-backup.tar

# Unregister corrupted distribution
wsl --unregister Ubuntu-24.04

# Import distribution
wsl --import Ubuntu-24.04 C:\WSL\Ubuntu-24.04 C:\temp\ubuntu-backup.tar

# Set as default
wsl --set-default Ubuntu-24.04
```

### Option 2: Complete WSL Reset (Last Resort)

```powershell
# Full WSL reset
wsl --shutdown
wsl --unregister Ubuntu-24.04

# Reset WSL to factory defaults
wsl --install --distribution Ubuntu-24.04
```

## FROM WITHIN WSL: Self-Diagnostic Commands

```bash
# Check WSL interop functionality
/mnt/c/Windows/System32/cmd.exe /c "echo WSL interop test"

# Verify mount points
mount | grep -E "(9p|drvfs)"

# Check systemd status
systemctl status

# Verify network connectivity
ping -c 3 8.8.8.8
curl -I https://google.com

# Check disk space and permissions
df -h
ls -la /mnt/c/

# Test Windows path access
ls -la /mnt/c/Windows/System32/
```

## PREVENTIVE MEASURES

### 1. System Configuration Hardening

```powershell
# Set WSL to manual startup to prevent conflicts
Set-Service LxssManager -StartupType Manual
Set-Service LxssManagerUser -StartupType Manual

# Configure WSL settings
New-Item -Path "$env:USERPROFILE\.wslconfig" -ItemType File -Force
@"
[wsl2]
memory=8GB
processors=4
swap=2GB
swapFile=C:\\temp\\wsl2-swap.vhdx

[experimental]
autoMemoryReclaim=dropcache
"@ | Out-File -FilePath "$env:USERPROFILE\.wslconfig" -Encoding UTF8
```

### 2. Regular Maintenance Schedule

```bash
# Weekly maintenance script (run from WSL)
#!/bin/bash
# Save as /opt/sutazaiapp/scripts/wsl-maintenance.sh

echo "=== WSL Weekly Maintenance $(date) ===" | tee -a /var/log/wsl-maintenance.log

# Clean package cache
sudo apt clean
sudo apt autoremove -y

# Check disk usage
df -h | tee -a /var/log/wsl-maintenance.log

# Verify critical mount points
mount | grep -E "(9p|drvfs)" | tee -a /var/log/wsl-maintenance.log

# Test interop
if /mnt/c/Windows/System32/cmd.exe /c "echo test" >/dev/null 2>&1; then
    echo "✓ WSL interop functional" | tee -a /var/log/wsl-maintenance.log
else
    echo "✗ WSL interop failed" | tee -a /var/log/wsl-maintenance.log
fi

echo "=== Maintenance completed ===" | tee -a /var/log/wsl-maintenance.log
```

### 3. Backup Strategy

```powershell
# Automated backup script (Windows PowerShell)
$BackupPath = "C:\WSL-Backups\$(Get-Date -Format 'yyyyMMdd')"
New-Item -Path $BackupPath -ItemType Directory -Force

# Export all WSL distributions
wsl --list --quiet | ForEach-Object {
    if ($_ -ne "") {
        $DistroName = $_.Trim()
        Write-Host "Backing up $DistroName..."
        wsl --export $DistroName "$BackupPath\$DistroName.tar"
    }
}

# Backup WSL configuration
Copy-Item "$env:USERPROFILE\.wslconfig" "$BackupPath\" -ErrorAction SilentlyContinue
```

## TROUBLESHOOTING SPECIFIC ERROR PATTERNS

### Error: "The system cannot find the file specified"
```powershell
# Rebuild Windows image store
DISM /Online /Cleanup-Image /RestoreHealth
sfc /scannow
```

### Error: "Element not found"
```powershell
# Reset Windows Store components
Get-AppxPackage *WindowsSubsystemForLinux* | Remove-AppxPackage
Get-AppxPackage *TerminalBridge* | Remove-AppxPackage
wsl --install
```

### Error: "Access denied" or permission issues
```powershell
# Reset file permissions
icacls "C:\Users\%USERNAME%\AppData\Local\Packages\*WindowsSubsystemForLinux*" /reset /t
```

## MONITORING AND ALERTING

### Windows Event Log Monitoring
Monitor these Windows Event Logs for WSL issues:
- Application and Services Logs > Microsoft > Windows > WSL
- System Log for Hyper-V related errors
- Application Log for service failures

### Performance Counters
```powershell
# Monitor WSL performance
Get-Counter "\Process(wsl*)\Private Bytes"
Get-Counter "\Process(wsl*)\Handle Count"
```

## ESCALATION PROCEDURES

If all recovery methods fail:

1. **Create Windows System Restore Point**
2. **Collect WSL debug logs**: `wsl --debug-shell`
3. **Generate Windows system report**: `msinfo32 /report C:\temp\system-report.txt`
4. **Contact Microsoft Support** with specific error codes and system information

## SUCCESS VALIDATION

After recovery, verify these components:

```bash
# From WSL
echo "Testing WSL functionality..."

# 1. Basic operations
uname -a
df -h

# 2. Windows interop
/mnt/c/Windows/System32/cmd.exe /c "ver"

# 3. Network connectivity
ping -c 1 google.com

# 4. File system access
ls -la /mnt/c/Users/

# 5. Container functionality (if applicable)
docker version 2>/dev/null || echo "Docker not available"

echo "✓ WSL recovery validation complete"
```

This comprehensive recovery guide addresses the catastrophic WSL service failure with multiple recovery strategies, from least to most invasive, ensuring system restoration while preserving data integrity where possible.