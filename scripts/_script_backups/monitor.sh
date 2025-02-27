#!/bin/bash
case $1 in
    start)
        ;;
    status)
        sutazai-cli system-status --detailed --encrypted
        ;;
    audit)
        ;;
    *)
        echo "Usage: $0 {start|status|audit}"
        exit 1
esac

# Updated paths
SUTAZAI_STATE_LOG="/var/log/sutazai/state_monitor.log"
SUTAZAI_OPTIMIZATION_LOG="/var/log/sutazai/optimization_monitor.log"

# Updated function names
monitor_sutazai_states() {
    python3 SutazAi/state_monitor.py >> $SUTAZAI_STATE_LOG 2>&1
}

# Allow Super AI agent to use terminal
if ! grep -q "super_ai" /etc/sutazai/terminal_access; then
    echo "super_ai" >> /etc/sutazai/terminal_access
fi 