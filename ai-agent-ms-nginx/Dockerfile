# Dockerfile for Nginx reverse proxy
# Simple HTTP setup with SSL/HTTPS readiness for future

FROM nginx:alpine

# Metadata labels
LABEL maintainer="AI Agent RAG Project"
LABEL description="Nginx reverse proxy for AI Agent RAG microservice"
LABEL version="1.0"

# Install curl for health checks and openssl for future SSL setup
RUN apk add --no-cache \
    curl \
    openssl \
    && rm -rf /var/cache/apk/*

# Create directories for SSL certificates (for future use)
RUN mkdir -p /etc/nginx/ssl \
    && mkdir -p /etc/nginx/conf.d

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf
COPY ai-agent-locations.conf /etc/nginx/conf.d/ai-agent-locations.conf

# Copy certs
# COPY certs/ /etc/nginx/ssl/

# Copy and set up health check script
COPY health-check.sh /usr/local/bin/health-check.sh
RUN chmod +x /usr/local/bin/health-check.sh

# Set proper permissions for SSL directory (ready for future certificates)
RUN chown -R nginx:nginx /etc/nginx/ssl

# Expose HTTP port (HTTPS port 443 ready for future)
EXPOSE 80
EXPOSE 443

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /usr/local/bin/health-check.sh

# Create volume for SSL certificates (ready for future use)
VOLUME ["/etc/nginx/ssl"]

# Start nginx in foreground mode
CMD ["nginx", "-g", "daemon off;"]