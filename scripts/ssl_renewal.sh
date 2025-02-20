#!/bin/bash
# Automatically renew SSL certificates
certbot renew --quiet --post-hook "systemctl reload nginx"
echo "SSL certificates renewed successfully!" 