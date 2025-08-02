# SutazAI Quick Start Guide 🚀

Welcome to SutazAI! This guide will help you get started with your multi-agent AI system.

## 🎯 First Steps

### 1. Access the System

Open your web browser and navigate to:
- **Main Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

### 2. Explore the Dashboard

The frontend automatically discovers and displays all 85+ AI agents organized by category:
- 🤖 Task Automation
- 💻 Code Generation  
- 📊 Data Analysis
- 🧠 ML/AI
- 🏗️ Infrastructure
- 🔒 Security
- 🧩 Specialized

## 💡 Try These Examples

### Example 1: Code Generation
1. Go to the "Code Generation" tab
2. Find "GPT Engineer" or "Aider"
3. Click "Execute Task"
4. Enter: "Create a Python script that fetches weather data"
5. Watch the AI generate complete code!

### Example 2: Task Automation
1. Go to the "Task Automation" tab
2. Select "AutoGPT" or "CrewAI"
3. Enter a complex task like: "Research the top 5 AI trends for 2025 and create a summary"
4. The agent will break down the task and execute it

### Example 3: Data Analysis
1. Go to the "Data Analysis" tab
2. Choose "Data Pipeline Engineer"
3. Request: "Analyze system performance metrics and create a report"
4. Get insights about your system

## 🛠️ Essential Commands

### Check System Status
```bash
./scripts/show_system_dashboard.sh
```

### Manage Agents
```bash
./scripts/agent_manager.sh
```
Options:
- View all agents
- Start/stop agents
- View logs
- Export agent list

### Monitor Health
```bash
python3 ./scripts/health_monitor.py
```

### Verify System
```bash
./scripts/verify_complete_system.sh
```

## 🔧 Common Operations

### View Agent Logs
```bash
docker logs sutazai-[agent-name] -f
```

### Restart an Agent
```bash
docker restart sutazai-[agent-name]
```

### Check Resource Usage
```bash
docker stats
```

## 📊 Monitoring

### Grafana Dashboard
Access comprehensive system metrics at:
http://localhost:3000

Default credentials:
- Username: admin
- Password: admin

### Real-time Monitoring
```bash
watch -n 5 docker ps --format "table {{.Names}}\t{{.Status}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## 🔄 Workflows

### n8n Workflow Builder
Create automated workflows at:
http://localhost:5678

Example workflow:
1. Trigger: Webhook or schedule
2. Action: Call AI agent
3. Process: Transform data
4. Output: Send results

## ⚡ Performance Tips

1. **Resource Management**
   - Stop unused agents to free resources
   - Use `./scripts/agent_manager.sh` option 3

2. **Optimize Performance**
   - Run weekly: `sudo ./scripts/optimize_system.sh`
   - Monitor with: `./scripts/health_monitor.py`

3. **Batch Operations**
   - Group similar tasks for efficiency
   - Use workflow engines for automation

## 🚨 Troubleshooting

### Agent Not Responding
```bash
# Check status
docker ps | grep [agent-name]

# View logs
docker logs sutazai-[agent-name]

# Restart
docker restart sutazai-[agent-name]
```

### High Resource Usage
```bash
# Check usage
docker stats --no-stream

# Optimize system
sudo ./scripts/optimize_system.sh
```

### API Connection Issues
```bash
# Check backend
curl http://localhost:8000/health

# Restart if needed
docker restart sutazai-backend
```

## 📚 Learn More

- **Full Documentation**: `/opt/sutazaiapp/docs/`
- **API Reference**: http://localhost:8000/docs
- **Architecture Guide**: `/opt/sutazaiapp/docs/system/architecture/`

## 🎉 Next Steps

1. **Explore Different Agents**: Each agent has unique capabilities
2. **Create Workflows**: Combine agents for complex tasks
3. **Monitor Performance**: Use Grafana dashboards
4. **Customize**: Modify agent configurations as needed

## 💬 Getting Help

1. Check logs: `docker logs [container-name]`
2. Run diagnostics: `./scripts/verify_complete_system.sh`
3. Review documentation in `/opt/sutazaiapp/docs/`

---

**Enjoy your AI-powered automation journey with SutazAI!** 🚀✨