# Windows PowerShell Script for Cursor Terminal Optimization
# Place this file in your Windows home directory and run it before using Cursor

Write-Host "Applying Cursor terminal optimization for Windows..." -ForegroundColor Green

# Kill any potentially stuck SSH processes (optional - uncomment if needed)
# Get-Process | Where-Object {$_.Name -like "*ssh*"} | Stop-Process -Force

# Increase PowerShell buffer size
$MaximumHistoryCount = 10000
$Host.UI.RawUI.BufferSize = New-Object System.Management.Automation.Host.Size(120, 3000)

# Set environment variables for better SSH terminal performance
$env:TERM = "xterm-256color"
$env:PYTHONUNBUFFERED = "1"

# Create a helper function for use with Cursor's terminal
function Reset-CursorTerminal {
    Write-Host "Resetting Cursor terminal connection..." -ForegroundColor Yellow
    
    # Set terminal variables
    $env:TERM = "xterm-256color"
    $env:PYTHONUNBUFFERED = "1"
    
    # Optionally kill any stuck SSH processes (uncomment if needed)
    # Get-Process | Where-Object {$_.Name -like "*ssh*"} | Stop-Process -Force
    
    Write-Host "Terminal reset complete. Try reconnecting to your SSH session." -ForegroundColor Green
}

# Create a function to safely run analysis tools
function Invoke-SafeAnalysis {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Command,
        
        [Parameter(Mandatory=$true)]
        [string]$OutputFile
    )
    
    Write-Host "Running $Command with output to $OutputFile" -ForegroundColor Cyan
    ssh root@192.168.100.28 "cd /opt/sutazaiapp && timeout 60s $Command > $OutputFile 2>&1" 
    Write-Host "Analysis complete. Check $OutputFile for results." -ForegroundColor Green
}

# Create some helpful shortcut functions
function Invoke-SafeSemgrep {
    param([string]$Target = ".")
    Invoke-SafeAnalysis -Command "semgrep --config-auto $Target" -OutputFile "logs/semgrep_report.txt"
}

function Invoke-SafePylint {
    param([string]$Target = "backend/")
    Invoke-SafeAnalysis -Command "pylint $Target" -OutputFile "logs/pylint_report.txt"
}

function Invoke-SafeMypy {
    param([string]$Target = ".")
    Invoke-SafeAnalysis -Command "mypy $Target" -OutputFile "logs/mypy_report.txt"
}

# Export the functions so they're available in your PowerShell session
Export-ModuleMember -Function Reset-CursorTerminal, Invoke-SafeAnalysis, Invoke-SafeSemgrep, Invoke-SafePylint, Invoke-SafeMypy

Write-Host "Windows optimization for Cursor completed!" -ForegroundColor Green
Write-Host "Use these commands in PowerShell to prevent terminal freezing:" -ForegroundColor Yellow
Write-Host "  Reset-CursorTerminal - Resets terminal variables" -ForegroundColor Cyan
Write-Host "  Invoke-SafeSemgrep - Safely runs Semgrep" -ForegroundColor Cyan
Write-Host "  Invoke-SafePylint - Safely runs Pylint" -ForegroundColor Cyan
Write-Host "  Invoke-SafeMypy - Safely runs Mypy" -ForegroundColor Cyan 