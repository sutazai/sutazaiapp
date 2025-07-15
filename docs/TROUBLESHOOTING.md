# SutazAI Troubleshooting Guide

## Quick Diagnostics

### System Health Check
```bash
# Run comprehensive system check
python3 scripts/system_health.py

# Quick health check
curl http://localhost:8000/health

# Check service status
systemctl status sutazai
```

### Log Analysis
```bash
# View recent logs
tail -f logs/sutazai.log

# Check error logs
grep "ERROR" logs/sutazai.log | tail -20

# View specific component logs
tail -f logs/ai_agents.log
tail -f logs/neural_network.log
tail -f logs/security.log
```

## Common Issues and Solutions

### Installation Issues

#### Python Dependencies
**Problem**: Module import errors during installation
```
ModuleNotFoundError: No module named 'package_name'
```

**Solution**:
```bash
# Update pip
python3 -m pip install --upgrade pip

# Install missing dependencies
pip install -r requirements.txt

# If still failing, install individually
pip install package_name

# Check Python path
echo $PYTHONPATH
export PYTHONPATH=/opt/sutazaiapp:$PYTHONPATH
```

#### Docker Issues
**Problem**: Docker permission denied
```
docker: Got permission denied while trying to connect to the Docker daemon socket
```

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart session or run
newgrp docker

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

#### Port Conflicts
**Problem**: Port 8000 already in use
```
Error: Address already in use
```

**Solution**:
```bash
# Check what's using the port
sudo netstat -tulpn | grep :8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 $(sudo lsof -t -i:8000)

# Or use a different port
export SUTAZAI_PORT=8001
python3 main.py --port 8001
```

### Runtime Issues

#### Application Won't Start
**Problem**: SutazAI fails to start

**Diagnostic Steps**:
```bash
# Check logs for errors
tail -20 logs/sutazai.log

# Verify configuration
python3 scripts/validate_config.py

# Test database connection
python3 scripts/test_database.py

# Check dependencies
python3 scripts/check_dependencies.py
```

**Common Solutions**:
```bash
# Reset database
rm data/sutazai.db
python3 scripts/init_db.py

# Clear cache
rm -rf cache/*

# Reset configuration
cp config/default.env .env
```

#### High Memory Usage
**Problem**: System using excessive memory

**Diagnostic**:
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check SutazAI processes
ps aux | grep sutazai
```

**Solutions**:
```bash
# Reduce model size
export AI_MODEL_SIZE=small

# Limit worker processes
export MAX_WORKERS=4

# Increase swap space
sudo swapon --show
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Slow Performance
**Problem**: System running slowly

**Diagnostic**:
```bash
# Check CPU usage
top
htop

# Check disk I/O
iotop

# Run performance analysis
python3 performance_optimization.py --analyze
```

**Solutions**:
```bash
# Optimize database
python3 scripts/optimize_database.py

# Clear old logs
find logs/ -name "*.log" -mtime +7 -delete

# Restart with optimizations
./restart.sh --optimized
```

### AI and ML Issues

#### Model Loading Failures
**Problem**: AI models fail to load
```
Error: Unable to load model 'model_name'
```

**Diagnostic**:
```bash
# Check model files
ls -la models/
du -sh models/*

# Test model loading
python3 scripts/test_models.py

# Check model registry
cat data/model_registry.json
```

**Solutions**:
```bash
# Re-download models
python3 scripts/download_models.py --force

# Use fallback models
export USE_FALLBACK_MODELS=true

# Check disk space
df -h
```

#### Neural Network Issues
**Problem**: Neural network not responding

**Diagnostic**:
```bash
# Check neural network status
curl http://localhost:8000/api/v1/neural/status

# View neural network logs
tail -f logs/neural_network.log

# Test neural connections
python3 scripts/test_neural_network.py
```

**Solutions**:
```bash
# Reset neural network
python3 scripts/reset_neural_network.py

# Restart neural services
systemctl restart sutazai-neural

# Rebuild neural connections
python3 scripts/rebuild_neural_network.py
```

#### Generation Quality Issues
**Problem**: Poor quality AI-generated content

**Solutions**:
```bash
# Retrain models
python3 scripts/retrain_models.py

# Adjust generation parameters
export GENERATION_TEMPERATURE=0.7
export MAX_GENERATION_LENGTH=2000

# Use different model
export DEFAULT_MODEL=code_llama
```

### Database Issues

#### Database Corruption
**Problem**: Database file corrupted
```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions**:
```bash
# Backup current database
cp data/sutazai.db data/sutazai.db.backup

# Attempt repair
sqlite3 data/sutazai.db ".recover" | sqlite3 data/sutazai_recovered.db

# If repair fails, restore from backup
cp backups/sutazai-latest.db data/sutazai.db

# Or reinitialize
rm data/sutazai.db
python3 scripts/init_db.py
```

#### Database Connection Issues
**Problem**: Cannot connect to database

**Diagnostic**:
```bash
# Test database connection
python3 scripts/test_database.py

# Check database file permissions
ls -la data/sutazai.db

# Verify database integrity
sqlite3 data/sutazai.db "PRAGMA integrity_check;"
```

**Solutions**:
```bash
# Fix permissions
chmod 644 data/sutazai.db
chown sutazai:sutazai data/sutazai.db

# Restart database service
systemctl restart sqlite

# Reset connection pool
python3 scripts/reset_db_pool.py
```

#### Slow Database Queries
**Problem**: Database queries taking too long

**Solutions**:
```bash
# Analyze query performance
python3 scripts/analyze_db_performance.py

# Optimize database
sqlite3 data/sutazai.db "VACUUM; ANALYZE;"

# Add indexes
python3 scripts/add_database_indexes.py

# Update statistics
sqlite3 data/sutazai.db "ANALYZE;"
```

### Network and API Issues

#### API Not Responding
**Problem**: API endpoints returning errors

**Diagnostic**:
```bash
# Test API health
curl -v http://localhost:8000/health

# Check API logs
tail -f logs/api.log

# Test specific endpoints
curl http://localhost:8000/api/v1/status
```

**Solutions**:
```bash
# Restart API service
systemctl restart sutazai-api

# Check network configuration
netstat -tulpn | grep :8000

# Verify firewall settings
sudo ufw status
```

#### SSL/TLS Issues
**Problem**: HTTPS connection problems

**Diagnostic**:
```bash
# Test SSL certificate
openssl s_client -connect localhost:443

# Check certificate expiry
openssl x509 -in certificate.crt -text -noout | grep "Not After"
```

**Solutions**:
```bash
# Renew SSL certificate
python3 scripts/renew_ssl.py

# Generate new certificate
python3 scripts/generate_ssl.py

# Update certificate configuration
vim config/ssl.conf
```

#### Rate Limiting Issues
**Problem**: Requests being rate limited

**Solutions**:
```bash
# Check rate limit configuration
grep "rate_limit" config/*.conf

# Adjust rate limits
export API_RATE_LIMIT=1000

# Whitelist IP addresses
python3 scripts/whitelist_ip.py --ip YOUR_IP
```

### Security Issues

#### Authentication Failures
**Problem**: Cannot authenticate users

**Diagnostic**:
```bash
# Check authentication logs
grep "auth" logs/security.log

# Test authentication system
python3 scripts/test_auth.py

# Verify user database
sqlite3 data/sutazai.db "SELECT * FROM users LIMIT 5;"
```

**Solutions**:
```bash
# Reset user authentication
python3 scripts/reset_auth.py

# Create new admin user
python3 scripts/create_user.py --email admin@example.com --admin

# Update authentication keys
python3 scripts/generate_auth_keys.py
```

#### Permission Denied Errors
**Problem**: Access denied to resources

**Solutions**:
```bash
# Check file permissions
ls -la /opt/sutazaiapp/

# Fix ownership
sudo chown -R sutazai:sutazai /opt/sutazaiapp/

# Fix permissions
chmod -R 755 /opt/sutazaiapp/
chmod 644 /opt/sutazaiapp/data/*
```

## Advanced Troubleshooting

### Debug Mode
```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start with debug logging
python3 main.py --debug

# Enable verbose logging
export VERBOSE_LOGGING=true
```

### Performance Profiling
```bash
# Profile application performance
python3 -m cProfile -o profile.stats main.py

# Analyze profile
python3 scripts/analyze_profile.py profile.stats

# Memory profiling
python3 -m memory_profiler main.py
```

### System Monitoring
```bash
# Monitor system resources
htop
iotop
nethogs

# Monitor specific processes
watch -n 1 'ps aux | grep sutazai'

# Monitor disk usage
watch -n 5 'df -h'
```

### Container Troubleshooting
```bash
# View container logs
docker logs sutazai

# Enter container for debugging
docker exec -it sutazai /bin/bash

# Check container resource usage
docker stats sutazai

# Restart container
docker restart sutazai
```

## Recovery Procedures

### System Recovery
```bash
# Full system recovery
./scripts/emergency_recovery.sh

# Restore from backup
./scripts/restore_backup.sh --date 2024-01-01

# Factory reset
./scripts/factory_reset.sh --confirm
```

### Data Recovery
```bash
# Recover corrupted database
python3 scripts/recover_database.py

# Restore user data
python3 scripts/restore_user_data.py --user EMAIL

# Rebuild indexes
python3 scripts/rebuild_indexes.py
```

## Preventive Measures

### Regular Maintenance
```bash
# Weekly maintenance script
./scripts/weekly_maintenance.sh

# Update system
./scripts/update_system.sh

# Clean logs and cache
./scripts/cleanup.sh
```

### Monitoring Setup
```bash
# Setup monitoring
python3 scripts/setup_monitoring.py

# Configure alerts
python3 scripts/configure_alerts.py

# Test alert system
python3 scripts/test_alerts.py
```

### Backup Strategy
```bash
# Automatic backups
crontab -e
# Add: 0 2 * * * /opt/sutazaiapp/scripts/backup.sh

# Test backup restoration
./scripts/test_backup_restore.sh

# Verify backup integrity
./scripts/verify_backups.sh
```

## Getting Help

### Self-Help Resources
1. Check this troubleshooting guide
2. Review system logs
3. Run diagnostic scripts
4. Search the documentation

### Community Support
- **GitHub Issues**: https://github.com/sutazai/sutazaiapp/issues
- **Discord Community**: https://discord.gg/sutazai
- **Stack Overflow**: Tag questions with `sutazai`

### Professional Support
- **Email**: support@sutazai.com
- **Enterprise Support**: enterprise@sutazai.com
- **Emergency Hotline**: Available for enterprise customers

### Bug Reports
When reporting bugs, include:
1. System specifications
2. Error messages
3. Log files
4. Steps to reproduce
5. Expected vs actual behavior

### Feature Requests
Submit feature requests through:
1. GitHub Issues (feature request template)
2. Community Discord
3. Email to features@sutazai.com

---

**Remember**: Most issues can be resolved by restarting the system, checking logs, and following the solutions in this guide.
