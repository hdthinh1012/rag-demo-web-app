#!/bin/bash
# Simple build and run script for AI Agent Nginx container

set -e

# Configuration
CONTAINER_NAME="ai-agent-nginx"
IMAGE_NAME="ai-agent-nginx"
NETWORK_NAME="ai-agent-ms-network"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print colored messages
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create network if it doesn't exist
create_network() {
    if ! docker network ls | grep -q $NETWORK_NAME; then
        info "Creating Docker network: $NETWORK_NAME"
        docker network create $NETWORK_NAME
        success "Network created successfully"
    else
        info "Network $NETWORK_NAME already exists"
    fi
}

# Build Docker image
build_image() {
    info "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
    success "Image built successfully"
}

# Stop and remove existing container
cleanup_container() {
    if docker ps -a | grep -q $CONTAINER_NAME; then
        warning "Found existing container: $CONTAINER_NAME"
        info "Stopping and removing existing container..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        success "Existing container removed"
    fi
}

# Run the container
run_container() {
    info "Starting nginx container..."
    
    # Create ssl directory if it doesn't exist
    mkdir -p ssl
    
    docker run -d \
        --name $CONTAINER_NAME \
        --network $NETWORK_NAME \
        -p 80:80 \
        -p 443:443 \
        -v $(pwd)/ssl:/etc/nginx/ssl:ro \
        --restart unless-stopped \
        $IMAGE_NAME
    
    success "Container started successfully"
    info "Container name: $CONTAINER_NAME"
    info "HTTP URL: http://localhost"
    info "Network: $NETWORK_NAME"
}

# Show container status
show_status() {
    info "Container status:"
    docker ps -f name=$CONTAINER_NAME
    
    echo ""
    info "Container logs (last 10 lines):"
    docker logs --tail 10 $CONTAINER_NAME
}

# Main script
main() {
    info "Starting AI Agent Nginx setup..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    create_network
    build_image
    cleanup_container
    run_container
    
    echo ""
    success "Setup complete!"
    
    # Wait a moment for container to start
    sleep 3
    show_status
    
    echo ""
    info "Test the setup:"
    echo "  curl http://localhost/health"
    echo ""
    info "View logs:"
    echo "  docker logs -f $CONTAINER_NAME"
    echo ""
    info "Stop container:"
    echo "  docker stop $CONTAINER_NAME"
}

# Run main function
main "$@"