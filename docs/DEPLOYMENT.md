# Deployment Guide

This document outlines the deployment process for the SutazAI application.

## Prerequisites

1. Python 3.11 or higher
2. Node.js 18 or higher (for web UI)
3. PostgreSQL 14 or higher
4. Redis 6 or higher

## System Requirements

- CPU: 4+ cores
- RAM: 8GB minimum, 16GB recommended
- Storage: 50GB minimum
- OS: Ubuntu 20.04 LTS or higher

## Installation Steps

1. Clone the Repository:
   ```bash
   cd /opt
   sudo mkdir sutazaiapp
   sudo chown -R sutazaiapp_dev:sutazaiapp_dev sutazaiapp
   cd sutazaiapp
   git clone https://sutazaiapp:github_token@github.com/sutazai/sutazaiapp.git .
   ```

2. Set Up Python Virtual Environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. Install Node.js Dependencies:
   ```bash
   cd web_ui
   npm install
   ```

4. Configure the Application:
   - Copy example configuration files:
     ```bash
     cp backend/config/config.example.yml backend/config/config.yml
     cp scripts/config/deploy.example.toml scripts/config/deploy.toml
     ```
   - Edit configuration files with appropriate values

5. Set Up Database:
   ```bash
   # Create database and user
   sudo -u postgres psql
   CREATE DATABASE sutazai;
   CREATE USER sutazai WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE sutazai TO sutazai;
   ```

6. Set Up Redis:
   ```bash
   # Edit Redis configuration
   sudo cp redis.conf /etc/redis/redis.conf
   sudo systemctl restart redis
   ```

7. Set File Permissions:
   ```bash
   sudo chown -R sutazaiapp_dev:sutazaiapp_dev /opt/sutazaiapp
   chmod -R 750 /opt/sutazaiapp
   ```

## Running the Application

1. Start Backend Server:
   ```bash
   cd /opt/sutazaiapp
   source venv/bin/activate
   python -m backend.backend_main
   ```

2. Start Web UI Development Server:
   ```bash
   cd web_ui
   npm run dev
   ```

3. Start Web UI Production Server:
   ```bash
   cd web_ui
   npm run build
   npm run start
   ```

## Monitoring

1. Check Application Logs:
   ```bash
   tail -f /opt/sutazaiapp/logs/backend.log
   tail -f /opt/sutazaiapp/logs/web_ui.log
   ```

2. Check System Status:
   ```bash
   systemctl status sutazai-backend
   systemctl status sutazai-web
   ```

## Troubleshooting

1. Backend Issues:
   - Check logs in `/opt/sutazaiapp/logs/`
   - Verify database connection
   - Check Redis connection
   - Validate configuration files

2. Web UI Issues:
   - Check Node.js version
   - Verify npm dependencies
   - Check build output
   - Validate environment variables

3. Permission Issues:
   - Verify file ownership
   - Check directory permissions
   - Validate user access

## Backup and Recovery

1. Database Backup:
   ```bash
   pg_dump -U sutazai sutazai > backup.sql
   ```

2. File Backup:
   ```bash
   tar -czf sutazai_backup.tar.gz /opt/sutazaiapp
   ```

3. Recovery:
   ```bash
   # Restore database
   psql -U sutazai sutazai < backup.sql
   
   # Restore files
   tar -xzf sutazai_backup.tar.gz -C /opt/sutazaiapp
   ```

## Security Notes

1. File Permissions:
   - Keep files owned by `sutazaiapp_dev`
   - Maintain 750 permissions
   - Secure configuration files

2. Database Security:
   - Use strong passwords
   - Limit database access
   - Regular security updates

3. Network Security:
   - Configure firewalls
   - Use HTTPS
   - Implement rate limiting

## Maintenance

1. Regular Updates:
   ```bash
   # Update code
   git pull origin master
   
   # Update dependencies
   pip install -r requirements.txt
   cd web_ui && npm install
   ```

2. Log Rotation:
   - Configure logrotate
   - Monitor disk space
   - Archive old logs

3. Database Maintenance:
   - Regular backups
   - Index optimization
   - Query optimization 