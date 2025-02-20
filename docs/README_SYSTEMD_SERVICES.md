# SutazAI Systemd Services Management

## Available Services

1. `sutazai-file-structure-manager.service`
2. `sutazai-system-integration.service`
3. `sutazai-system-health-monitor.service`
4. `sutazai-project-optimizer.service`
5. `sutazai-auto-remediation.service`
6. `sutazai-master.service` (Starts all other services)

## Service Management Commands

### Start All Services

```bash
sudo systemctl start sutazai-master.service
```

### Stop All Services

```bash
sudo systemctl stop sutazai-master.service
```

### Restart All Services

```bash
sudo systemctl restart sutazai-master.service
```

### Check Status of All Services

```bash
sudo systemctl status sutazai-master.service
```

### Enable/Disable Services on Boot

```bash
# Enable services to start on boot
sudo systemctl enable sutazai-master.service

# Disable services from starting on boot
sudo systemctl disable sutazai-master.service
```

### Individual Service Management

You can manage individual services using the same commands, replacing `sutazai-master.service` with the specific service name.

Example:

```bash
sudo systemctl status sutazai-file-structure-manager.service
```

## Logs

Service logs are stored in:

- `/opt/sutazai_project/SutazAI/logs/`

You can view logs using:

```bash
journalctl -u sutazai-master.service
```

## Troubleshooting

1. If services fail to start, check logs for specific errors
2. Ensure all dependencies are installed
3. Verify Python script permissions
4. Run the setup script again: `/opt/sutazai_project/SutazAI/scripts/setup_systemd_services.sh`

## Reload Systemd Configuration

If you modify service files:

```bash
sudo systemctl daemon-reload
```
