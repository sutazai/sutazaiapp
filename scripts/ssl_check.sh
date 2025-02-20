#!/bin/bash
# Check SSL certificate expiration
openssl x509 -enddate -noout -in /etc/ssl/certs/sutazai.crt
echo "SSL certificate check completed!" 