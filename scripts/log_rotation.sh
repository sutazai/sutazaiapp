#!/bin/bash

# Setup log rotation
setup_log_rotation() {
    cat <<EOF > /etc/logrotate.d/sutazai
/var/log/sutazai/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 640 root adm
}
EOF
} 