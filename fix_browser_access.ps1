# SutazAI Browser Access Fix Script
# Run this in PowerShell as Administrator

Write-Host "🚀 SutazAI Browser Access Fix" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "❌ ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "✅ Running as Administrator" -ForegroundColor Green

# Get WSL IP
Write-Host "🔍 Finding WSL IP address..." -ForegroundColor Yellow
$wslIP = ""
try {
    $wslInfo = wsl hostname -I
    $wslIP = $wslInfo.Trim()
    Write-Host "📍 WSL IP: $wslIP" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not auto-detect WSL IP, using container IP 172.17.0.3" -ForegroundColor Yellow
    $wslIP = "172.17.0.3"
}

# Remove existing port proxy rules
Write-Host "🧹 Cleaning existing port proxy rules..." -ForegroundColor Yellow
try {
    netsh interface portproxy delete v4tov4 listenport=5555 2>$null
    Write-Host "✅ Cleaned existing rules" -ForegroundColor Green
} catch {
    Write-Host "ℹ️  No existing rules to clean" -ForegroundColor Blue
}

# Add new port proxy rule
Write-Host "🔧 Setting up port forwarding..." -ForegroundColor Yellow
try {
    $result = netsh interface portproxy add v4tov4 listenport=5555 listenaddress=0.0.0.0 connectport=5555 connectaddress=$wslIP
    Write-Host "✅ Port forwarding configured: localhost:5555 -> $wslIP:5555" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to configure port forwarding" -ForegroundColor Red
    pause
    exit 1
}

# Check Windows Firewall
Write-Host "🔥 Checking Windows Firewall..." -ForegroundColor Yellow
try {
    $firewallRule = Get-NetFirewallRule -DisplayName "SutazAI Port 5555" -ErrorAction SilentlyContinue
    if (-not $firewallRule) {
        New-NetFirewallRule -DisplayName "SutazAI Port 5555" -Direction Inbound -LocalPort 5555 -Protocol TCP -Action Allow
        Write-Host "✅ Firewall rule created for port 5555" -ForegroundColor Green
    } else {
        Write-Host "✅ Firewall rule already exists" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Could not configure firewall rule (may need manual configuration)" -ForegroundColor Yellow
}

# Show current port proxy rules
Write-Host "📋 Current port proxy rules:" -ForegroundColor Yellow
netsh interface portproxy show all

Write-Host ""
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host "✅ Port forwarding: localhost:5555 -> $wslIP:5555" -ForegroundColor Green
Write-Host "✅ Firewall rule configured" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Open your browser and go to:" -ForegroundColor Cyan
Write-Host "   http://localhost:5555" -ForegroundColor White -BackgroundColor Blue
Write-Host ""
Write-Host "📱 Alternative access methods:" -ForegroundColor Yellow
Write-Host "   • API Docs: http://localhost:5555/docs" -ForegroundColor White
Write-Host "   • Health: http://localhost:5555/health" -ForegroundColor White
Write-Host "   • Direct IP: http://$wslIP:5555" -ForegroundColor White
Write-Host ""

# Test the connection
Write-Host "🧪 Testing connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5555/health" -TimeoutSec 5 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ SUCCESS! SutazAI is accessible at http://localhost:5555" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Connection test returned status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Connection test failed. Please check that SutazAI is running in WSL." -ForegroundColor Red
    Write-Host "   Run this in WSL: cd /opt/sutazaiapp && ./sutazai_simple.sh" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")