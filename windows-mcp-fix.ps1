# Windows MCP Server Fix Script
# Fixes MCP server connections for Windows environment
# Author: MCP Integration Specialist
# Date: 2025-08-25

Write-Host "=== Windows MCP Server Fix ===" -ForegroundColor Cyan
Write-Host "Starting at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# Configuration
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootPath = $scriptPath
$mcpConfigPath = Join-Path $rootPath ".mcp.json"
$claudeConfigPath = "$env:APPDATA\Claude\claude_desktop_config.json"

# Function to test command availability
function Test-CommandExists {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to test NPM package
function Test-NpmPackage {
    param($Package)
    Write-Host "  Testing $Package..." -NoNewline
    try {
        $result = & npx -y $Package --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host " ✓" -ForegroundColor Green
            return $true
        } else {
            Write-Host " ✗" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host " ✗" -ForegroundColor Red
        return $false
    }
}

# Kill existing MCP processes
Write-Host "`nStep 1: Cleaning up existing MCP processes..." -ForegroundColor Yellow
$processes = @(
    "node*mcp*",
    "claude-flow",
    "ruv-swarm",
    "mcp-server*"
)

foreach ($proc in $processes) {
    Get-Process | Where-Object { $_.ProcessName -like $proc } | Stop-Process -Force -ErrorAction SilentlyContinue
}
Write-Host "  Processes terminated" -ForegroundColor Green

# Test prerequisites
Write-Host "`nStep 2: Testing prerequisites..." -ForegroundColor Yellow
$prereqs = @{
    "node" = Test-CommandExists "node"
    "npm" = Test-CommandExists "npm"
    "npx" = Test-CommandExists "npx"
    "git" = Test-CommandExists "git"
}

foreach ($key in $prereqs.Keys) {
    $status = if ($prereqs[$key]) { "✓" } else { "✗" }
    $color = if ($prereqs[$key]) { "Green" } else { "Red" }
    Write-Host "  $key : $status" -ForegroundColor $color
}

# Test MCP packages
Write-Host "`nStep 3: Testing MCP packages..." -ForegroundColor Yellow
$packages = @(
    "@modelcontextprotocol/server-files",
    "@modelcontextprotocol/server-http-fetch",
    "@modelcontextprotocol/server-ddg",
    "claude-flow@alpha",
    "ruv-swarm@latest"
)

$availablePackages = @()
foreach ($pkg in $packages) {
    if (Test-NpmPackage $pkg) {
        $availablePackages += $pkg
    }
}

# Create Windows-compatible MCP configuration
Write-Host "`nStep 4: Creating Windows MCP configuration..." -ForegroundColor Yellow

$mcpConfig = @{
    mcpServers = @{
        "files" = @{
            command = "npx"
            args = @("-y", "@modelcontextprotocol/server-files", $rootPath)
            type = "stdio"
        }
        "http-fetch" = @{
            command = "npx"
            args = @("-y", "@modelcontextprotocol/server-http-fetch")
            type = "stdio"
        }
        "ddg-search" = @{
            command = "npx"
            args = @("-y", "@modelcontextprotocol/server-ddg")
            type = "stdio"
        }
        "claude-flow" = @{
            command = "npx"
            args = @("claude-flow@alpha", "mcp", "start", "--stdio")
            type = "stdio"
        }
        "ruv-swarm" = @{
            command = "npx"
            args = @("ruv-swarm@latest", "mcp", "start", "--stdio", "--stability")
            type = "stdio"
        }
    }
}

# Add sequential-thinking if installed globally
if (Test-Path "$env:APPDATA\npm\node_modules\mcp-sequential-thinking") {
    $mcpConfig.mcpServers["sequential-thinking"] = @{
        command = "node"
        args = @("$env:APPDATA\npm\node_modules\mcp-sequential-thinking\dist\index.js")
        type = "stdio"
    }
    Write-Host "  Added sequential-thinking (global)" -ForegroundColor Green
}

# Save MCP configuration
$mcpConfigJson = $mcpConfig | ConvertTo-Json -Depth 10
Set-Content -Path $mcpConfigPath -Value $mcpConfigJson -Encoding UTF8
Write-Host "  Saved to $mcpConfigPath" -ForegroundColor Green

# Update Claude Desktop configuration
Write-Host "`nStep 5: Updating Claude Desktop configuration..." -ForegroundColor Yellow

if (Test-Path $claudeConfigPath) {
    Copy-Item $claudeConfigPath "$claudeConfigPath.backup" -Force
    Write-Host "  Backed up existing config" -ForegroundColor Gray
}

# Create minimal Claude config with working servers
$claudeConfig = @{
    mcpServers = @{
        "files" = @{
            command = "npx"
            args = @("-y", "@modelcontextprotocol/server-files", $rootPath)
        }
        "http-fetch" = @{
            command = "npx"
            args = @("-y", "@modelcontextprotocol/server-http-fetch")
        }
    }
}

# Add sequential-thinking if available
if (Test-CommandExists "mcp-server-sequential-thinking") {
    $claudeConfig.mcpServers["sequential-thinking"] = @{
        command = "mcp-server-sequential-thinking"
    }
}

$claudeConfigJson = $claudeConfig | ConvertTo-Json -Depth 10
Set-Content -Path $claudeConfigPath -Value $claudeConfigJson -Encoding UTF8
Write-Host "  Updated Claude Desktop config" -ForegroundColor Green

# Test MCP servers
Write-Host "`nStep 6: Testing MCP server connections..." -ForegroundColor Yellow

function Test-McpServer {
    param($Name, $Command, $Args)
    
    Write-Host "  Testing $Name..." -NoNewline
    try {
        $proc = Start-Process -FilePath $Command -ArgumentList $Args -NoNewWindow -PassThru -RedirectStandardOutput "NUL" -RedirectStandardError "NUL"
        Start-Sleep -Seconds 2
        
        if (!$proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Write-Host " ✓ (responds)" -ForegroundColor Green
            return $true
        } else {
            Write-Host " ✗ (exited)" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host " ✗ (error)" -ForegroundColor Red
        return $false
    }
}

$workingServers = @()
foreach ($server in $mcpConfig.mcpServers.Keys) {
    $config = $mcpConfig.mcpServers[$server]
    if (Test-McpServer $server $config.command $config.args) {
        $workingServers += $server
    }
}

# Generate report
Write-Host "`n=== MCP Server Status Report ===" -ForegroundColor Cyan
Write-Host "Working Servers: $($workingServers.Count)" -ForegroundColor Green
foreach ($server in $workingServers) {
    Write-Host "  ✓ $server" -ForegroundColor Green
}

Write-Host "`nConfiguration Files:" -ForegroundColor Yellow
Write-Host "  MCP Config: $mcpConfigPath" -ForegroundColor Gray
Write-Host "  Claude Config: $claudeConfigPath" -ForegroundColor Gray

Write-Host "`n=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Restart Claude Desktop application" -ForegroundColor Yellow
Write-Host "2. Use the /mcp command in Claude to reconnect" -ForegroundColor Yellow
Write-Host "3. The following servers should be available:" -ForegroundColor Yellow
foreach ($server in $workingServers) {
    Write-Host "   - $server" -ForegroundColor Gray
}

Write-Host "`nFix completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray