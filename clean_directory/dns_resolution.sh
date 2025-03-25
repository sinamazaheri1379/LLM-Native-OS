#!/bin/bash
# dns_resolver.sh - Script to resolve API domain names and update the kernel module

# Path to the sysfs interface
SYSFS_PATH="/sys/class/llm_orchestrator/llm_orchestrator/provider_hosts"

# Check if the module is loaded and sysfs interface exists
if [ ! -e "$SYSFS_PATH" ]; then
    echo "Error: Module not loaded or sysfs interface not available at $SYSFS_PATH"
    echo "Make sure the module is loaded with 'sudo insmod llm_orchestrator.ko'"
    exit 1
fi

# Function to resolve domain name and update provider
resolve_and_update() {
    PROVIDER_ID=$1
    DOMAIN=$2
    DEFAULT_PORT=$3

    echo "Resolving $DOMAIN..."

    # Try to resolve using dig first (provides better control)
    if command -v dig &> /dev/null; then
        IP=$(dig +short $DOMAIN | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
    # Fallback to host command
    elif command -v host &> /dev/null; then
        IP=$(host $DOMAIN | grep "has address" | head -1 | awk '{print $4}')
    # Fallback to getent
    elif command -v getent &> /dev/null; then
        IP=$(getent hosts $DOMAIN | awk '{print $1}' | head -1)
    # Final fallback to nslookup
    else
        IP=$(nslookup $DOMAIN | grep -A 2 Name | grep Address | tail -1 | awk '{print $2}')
    fi

    if [ -z "$IP" ]; then
        echo "Failed to resolve $DOMAIN. Check your internet connection or DNS settings."
        return 1
    fi

    echo "Resolved $DOMAIN to $IP"
    echo "Updating provider $PROVIDER_ID..."

    # Update the provider configuration
    echo "$PROVIDER_ID,$IP,$DEFAULT_PORT" > $SYSFS_PATH

    if [ $? -eq 0 ]; then
        echo "Successfully updated provider $PROVIDER_ID ($DOMAIN) to use $IP:$DEFAULT_PORT"
    else
        echo "Failed to update provider $PROVIDER_ID"
        return 1
    fi

    return 0
}

# Resolve and update all providers
echo "Updating API endpoint IP addresses..."

# OpenAI
resolve_and_update 0 "api.openai.com" 443
OPENAI_STATUS=$?

# Anthropic
resolve_and_update 1 "api.anthropic.com" 443
ANTHROPIC_STATUS=$?

# Google Gemini
resolve_and_update 2 "generativelanguage.googleapis.com" 443
GEMINI_STATUS=$?

# Print summary
echo ""
echo "DNS Resolution Summary:"
echo "----------------------"
echo "OpenAI: $([ $OPENAI_STATUS -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo "Anthropic: $([ $ANTHROPIC_STATUS -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo "Google Gemini: $([ $GEMINI_STATUS -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo ""

# Check if all resolutions succeeded
if [ $OPENAI_STATUS -eq 0 ] && [ $ANTHROPIC_STATUS -eq 0 ] && [ $GEMINI_STATUS -eq 0 ]; then
    echo "All API endpoints resolved and updated successfully!"
    exit 0
else
    echo "Some API endpoints could not be resolved. Check your internet connection and DNS settings."
    exit 1
fi