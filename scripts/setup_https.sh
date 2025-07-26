#!/bin/bash
# Setup HTTPS with Nginx for SutazAI
# This script configures a proper HTTPS setup using Nginx as a reverse proxy

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run this script as root or with sudo"
    exit 1
fi

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Install Nginx if not present
if ! command -v nginx &> /dev/null; then
    echo "Installing Nginx..."
    apt-get update
    apt-get install -y nginx
    systemctl enable nginx
    systemctl start nginx
fi

# Create directories for SSL certificates
mkdir -p /etc/nginx/ssl/sutazaiapp

# Check if Let's Encrypt is to be used
USE_LETSENCRYPT=false
DOMAIN=""

# Ask user for domain configuration
read -p "Do you want to use Let's Encrypt for a real certificate? (y/n): " use_le
if [ "$use_le" = "y" ] || [ "$use_le" = "Y" ]; then
    USE_LETSENCRYPT=true
    read -p "Enter your domain name (e.g., example.com): " DOMAIN
    
    # Install certbot for Let's Encrypt
    apt-get install -y certbot python3-certbot-nginx
    
    # Get certificate from Let's Encrypt
    echo "Obtaining SSL certificate from Let's Encrypt..."
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email admin@"$DOMAIN" --redirect
else
    echo "Using self-signed certificate..."
    
    # Generate a self-signed certificate
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
      -keyout /etc/nginx/ssl/sutazaiapp/privkey.pem \
      -out /etc/nginx/ssl/sutazaiapp/fullchain.pem \
      -subj "/C=US/ST=State/L=City/O=SutazAI/CN=localhost"
      
    chmod 600 /etc/nginx/ssl/sutazaiapp/privkey.pem /etc/nginx/ssl/sutazaiapp/fullchain.pem
fi

# Create Nginx configuration
CONFIG_FILE="/etc/nginx/sites-available/sutazaiapp"

if [ "$USE_LETSENCRYPT" = true ]; then
    # Using Let's Encrypt (certbot will handle most of the config)
    echo "Configuring Nginx to proxy to backend..."
    
    # Create a temporary file and move it into place
    cat > /tmp/sutazai_nginx.conf << EOF
server {
    listen 80;
    server_name $DOMAIN;
    
    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name $DOMAIN;
    
    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    
    # HSTS (uncomment when you're sure everything works)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
    mv /tmp/sutazai_nginx.conf "$CONFIG_FILE"
    
else
    # Using self-signed certificate
    cat > "$CONFIG_FILE" << EOF
server {
    listen 80;
    server_name _;
    
    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name _;
    
    ssl_certificate /etc/nginx/ssl/sutazaiapp/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/sutazaiapp/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    
    # Add security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
fi

# Enable the site
ln -sf "$CONFIG_FILE" /etc/nginx/sites-enabled/sutazaiapp

# Remove default site if it exists
if [ -f /etc/nginx/sites-enabled/default ]; then
    rm /etc/nginx/sites-enabled/default
fi

# Update environment variables for the application
if [ -f "$PROJECT_ROOT/.env" ]; then
    # Remove existing entries if they exist
    sed -i '/BEHIND_PROXY/d' "$PROJECT_ROOT/.env"
    sed -i '/ENFORCE_HTTPS/d' "$PROJECT_ROOT/.env"
    
    # Add environment variables
    echo "BEHIND_PROXY=true" >> "$PROJECT_ROOT/.env"
    echo "ENFORCE_HTTPS=true" >> "$PROJECT_ROOT/.env"
    
    echo "Updated environment variables in .env file"
else
    echo "BEHIND_PROXY=true" > "$PROJECT_ROOT/.env"
    echo "ENFORCE_HTTPS=true" >> "$PROJECT_ROOT/.env"
    echo "Created .env file with required environment variables"
fi

# Test Nginx configuration
echo "Testing Nginx configuration..."
nginx -t

if [ $? -eq 0 ]; then
    # Reload Nginx
    echo "Reloading Nginx..."
    systemctl reload nginx
    
    # Update services to use the new configuration
    echo "Restarting backend service to use the new configuration..."
    bash "$PROJECT_ROOT/scripts/stop_backend.sh"
    bash "$PROJECT_ROOT/scripts/start_backend.sh"
    
    echo "HTTPS setup complete!"
    if [ "$USE_LETSENCRYPT" = true ]; then
        echo "Your site is now available at https://$DOMAIN"
    else
        echo "Your site is now available with a self-signed certificate."
        echo "Note: Browsers will show a warning about the self-signed certificate."
        echo "For production use, consider using Let's Encrypt or a commercial certificate."
    fi
else
    echo "Nginx configuration test failed. Please check the error messages above."
    exit 1
fi 