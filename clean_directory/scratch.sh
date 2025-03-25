#!/bin/bash
# setup_tls_proxy.sh - Script to set up a local TLS proxy for LLM APIs

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (sudo)"
    exit 1
fi

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "Installing nginx..."
    apt-get update
    apt-get install -y nginx
fi

# Create config file
echo "Creating nginx configuration..."
cat > /etc/nginx/sites-available/llm-proxy.conf << 'EOF'
server {
    listen 8080;
    server_name localhost;

    # Connection pooling settings
    keepalive_timeout 65;
    keepalive_requests 100;

    # Logging
    access_log /var/log/nginx/llm-proxy-access.log;
    error_log /var/log/nginx/llm-proxy-error.log;

    # OpenAI API proxy
    location /openai/ {
        proxy_pass https://api.openai.com/;
        proxy_ssl_server_name on;
        proxy_ssl_protocols TLSv1.2 TLSv1.3;

        # Remove the /openai/ prefix when forwarding
        rewrite ^/openai/(.*)$ /$1 break;

        # Headers
        proxy_set_header Host api.openai.com;
        proxy_set_header X-Real-IP $remote_addr;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Anthropic API proxy
    location /anthropic/ {
        proxy_pass https://api.anthropic.com/;
        proxy_ssl_server_name on;
        proxy_ssl_protocols TLSv1.2 TLSv1.3;

        # Remove the /anthropic/ prefix when forwarding
        rewrite ^/anthropic/(.*)$ /$1 break;

        # Headers
        proxy_set_header Host api.anthropic.com;
        proxy_set_header X-Real-IP $remote_addr;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Google Gemini API proxy
    location /gemini/ {
        proxy_pass https://generativelanguage.googleapis.com/;
        proxy_ssl_server_name on;
        proxy_ssl_protocols TLSv1.2 TLSv1.3;

        # Remove the /gemini/ prefix when forwarding
        rewrite ^/gemini/(.*)$ /$1 break;

        # Headers
        proxy_set_header Host generativelanguage.googleapis.com;
        proxy_set_header X-Real-IP $remote_addr;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# Enable the site
ln -sf /etc/nginx/sites-available/llm-proxy.conf /etc/nginx/sites-enabled/

# Test the configuration
echo "Testing nginx configuration..."
nginx -t

# Restart nginx
echo "Restarting nginx..."
systemctl restart nginx

# Update kernel module configuration
echo "Updating LLM orchestrator configuration..."

# Function to check if the module is loaded
check_module() {
    if ! lsmod | grep -q llm_orchestrator; then
        echo "LLM orchestrator module is not loaded"
        return 1
    fi
    return 0
}

# Function to update provider configuration
update_provider() {
    local id=$1
    local prefix=$2
    echo "$id,127.0.0.1,8080" > /sys/class/llm_orchestrator/llm_orchestrator/provider_hosts

    # Verify it worked
    if [ $? -eq 0 ]; then
        echo "Successfully updated provider $id to use local proxy"
    else
        echo "Failed to update provider $id configuration"
    fi
}

# Update providers if module is loaded
if check_module; then
    # Update OpenAI configuration (provider 0)
    update_provider 0 "/openai"

    # Update Anthropic configuration (provider 1)
    update_provider 1 "/anthropic"

    # Update Google Gemini configuration (provider 2)
    update_provider 2 "/gemini"

    # Update path formats in orchestrator_main.c to include the prefix
    echo "Don't forget to update the path formats in your code to include the proxy prefix:"
    echo "OpenAI:     /openai/v1/chat/completions"
    echo "Anthropic:  /anthropic/v1/messages"
    echo "Gemini:     /gemini/v1/models/gemini-pro:generateContent?key=xxx"
else
    echo "Note: The LLM orchestrator module isn't currently loaded."
    echo "After loading the module, run these commands to update the configuration:"
    echo "echo '0,127.0.0.1,8080' > /sys/class/llm_orchestrator/llm_orchestrator/provider_hosts"
    echo "echo '1,127.0.0.1,8080' > /sys/class/llm_orchestrator/llm_orchestrator/provider_hosts"
    echo "echo '2,127.0.0.1,8080' > /sys/class/llm_orchestrator/llm_orchestrator/provider_hosts"
fi

echo "TLS proxy setup complete!"
echo "Your kernel module should now connect to localhost:8080 with paths:"
echo "  OpenAI:     /openai/v1/chat/completions"
echo "  Anthropic:  /anthropic/v1/messages"
echo "  Gemini:     /gemini/v1/models/gemini-pro:generateContent?key=xxx"

# Test connectivity to the proxy
echo "Testing connectivity to the proxy..."
if curl -s "http://localhost:8080/" > /dev/null; then
    echo "Successfully connected to the proxy"
else
    echo "Warning: Could not connect to the proxy. Check nginx status."
fi