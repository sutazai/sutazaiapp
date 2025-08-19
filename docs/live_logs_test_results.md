# Live Logs Testing Report - 2025-08-19 22:46:34

## Testing Environment
- Script: /opt/sutazaiapp/scripts/monitoring/live_logs.sh
- User: root
- Working Directory: /opt/sutazaiapp/scripts/mcp/wrappers

## Option 1: System Overview
- **Status**: TIMEOUT (may be interactive)
- **Duration**: 3s (timeout)
- **Notes**: Option appears to be interactive or long-running
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

```

## Option 2: Live Logs (All Services)
- **Status**: FAILED ❌
- **Exit Code**: 1
- **Duration**: 0s
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

```

## Option 3: Test API Endpoints
- **Status**: FAILED ❌
- **Exit Code**: 1
- **Duration**: 5s
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

```

## Option 4: Container Statistics
- **Status**: FAILED ❌
- **Exit Code**: 1
- **Duration**: 2s
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

```

## Option 5: Log Management
- **Status**: FAILED ❌
- **Exit Code**: 1
- **Duration**: 2s
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

```

## Option 6: Debug Controls
- **Status**: FAILED ❌
- **Exit Code**: 1
- **Duration**: 0s
- **Error**: 6. Set Log Level to ERROR
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

```

## Option 7: Database Repair
- **Status**: FAILED ❌
- **Exit Code**: 1
- **Duration**: 0s
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

```

## Option 8: System Repair
- **Status**: TIMEOUT (may be interactive)
- **Duration**: 15s (timeout)
- **Notes**: Option appears to be interactive or long-running
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

```

## Option 9: Restart All Services
- **Status**: FAILED ❌
- **Exit Code**: 1
- **Duration**: 7s
- **Error**: Error response from daemon: Cannot restart container 6f4628df479a9926d5c8cd0810ac5695ed4382c07d4242e351777dd53b11af51: unable to find user promtail: no matching entries in passwd file
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

```

## Option 10: Unified Live Logs (All in One)
- **Status**: TIMEOUT (may be interactive)
- **Duration**: 5s (timeout)
- **Notes**: Option appears to be interactive or long-running
- **Sample Output**:
```
[0;36m╔══════════════════════════════════════════════════════════════╗[0m
[0;36m║                   SUTAZAI MONITORING MENU                   ║[0m
[0;36m║                                                   [[0;31mnum[0m]     ║[0m
[0;36m╚══════════════════════════════════════════════════════════════╝[0m

