#!/bin/sh
# Health check script for nginx container

# Check if nginx is running
if ! pgrep nginx > /dev/null; then
    echo "ERROR: nginx process not found"
    exit 1
fi

# Check if nginx is responding on port 80
if ! curl -f -s http://localhost:80/health > /dev/null; then
    echo "ERROR: nginx not responding on port 80"
    exit 1
fi

# Check nginx configuration syntax
if ! nginx -t > /dev/null 2>&1; then
    echo "ERROR: nginx configuration syntax error"
    exit 1
fi

echo "OK: nginx is healthy"
exit 0