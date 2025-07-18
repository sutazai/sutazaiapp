#!/bin/bash
# SutazAI System Management Script

case "$1" in
    start)
        echo "Starting SutazAI system..."
        ./deploy_production_final.sh
        ;;
    stop)
        echo "Stopping SutazAI system..."
        pkill -f intelligent_backend.py
        pkill -f intelligent_chat_app.py
        docker-compose -f docker-compose-agents-simple.yml down
        docker-compose down
        ;;
    restart)
        echo "Restarting SutazAI system..."
        $0 stop
        sleep 5
        $0 start
        ;;
    status)
        echo "SutazAI System Status:"
        curl -s http://localhost:8000/api/system/complete_status | jq .
        ;;
    monitor)
        echo "Starting system monitor..."
        ./monitor_production_system.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|monitor}"
        exit 1
        ;;
esac
