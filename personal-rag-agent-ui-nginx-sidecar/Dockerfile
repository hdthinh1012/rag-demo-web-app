# Dockerfile for Nginx sidecar container
# Serves as reverse proxy to Next.js Personal RAG Agent UI

FROM nginx:alpine

# Metadata labels
LABEL maintainer="Personal RAG Agent UI Project"
LABEL description="Nginx sidecar for Next.js Personal RAG Agent UI"
LABEL version="1.0"

# Install curl for health checks
RUN apk add --no-cache curl && rm -rf /var/cache/apk/*

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy and set up health check script
COPY health-check.sh /usr/local/bin/health-check.sh
RUN chmod +x /usr/local/bin/health-check.sh

# Test nginx configuration during build
# RUN nginx -t

# Expose HTTP port
EXPOSE 80

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /usr/local/bin/health-check.sh

# Start nginx in foreground mode
CMD ["nginx", "-g", "daemon off;"]