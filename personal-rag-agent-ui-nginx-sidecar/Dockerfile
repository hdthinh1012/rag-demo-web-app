# Dockerfile for Nginx sidecar container
# Serves as reverse proxy to Next.js Personal RAG Agent UI

FROM nginx:alpine

# Metadata labels
LABEL maintainer="Personal RAG Agent UI Project"
LABEL description="Nginx sidecar for Next.js Personal RAG Agent UI"
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
COPY ai-agent-ui-locations.conf /etc/nginx/conf.d/ai-agent-ui-locations.conf

# Copy certs
COPY certs/ /etc/nginx/ssl/

# Copy and set up health check script
COPY health-check.sh /usr/local/bin/health-check.sh
RUN chmod +x /usr/local/bin/health-check.sh

# Test nginx configuration during build
# RUN nginx -t

# Expose HTTP port
EXPOSE 80
EXPOSE 444

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /usr/local/bin/health-check.sh

# Create volume for SSL certificates (ready for future use)
VOLUME ["/etc/nginx/ssl"]

# Start nginx in foreground mode
CMD ["nginx", "-g", "daemon off;"]