# Docker Deployment Guide

This guide explains how to run the AI Agent Microservice using Docker with the `continuumio/miniconda3` base image.

## Files Overview

- `environment_docker.yml` - Linux-compatible conda environment (removes Windows-specific packages)
- `Dockerfile` - Docker image definition using miniconda3 base
- `docker-compose.yml` - Docker Compose configuration for easy deployment
- `.dockerignore` - Excludes unnecessary files from Docker build context

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose installed
- Google Cloud service account JSON key file
- `.env` file with your configuration

### 2. Setup

1. **Place your service account key:**
   ```bash
   mkdir -p credentials
   cp /path/to/your-service-account-key.json ./credentials/service-account.json
   ```

2. **Create .env file:**
   ```bash
   PROJECT_ID=your-google-cloud-project-id
   LOCATION=asia-southeast1
   ```

### 3. Build and Run

#### Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

#### Using Docker directly

```bash
# Build the image
docker build -t ai-agent-rag .

# Run the container
docker run -d \
  --name ai-agent \
  -p 5000:5000 \
  -e PROJECT_ID=your-project-id \
  -e LOCATION=asia-southeast1 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account.json \
  -v $(pwd)/credentials:/app/credentials:ro \
  -v ai-agent-files:/app/files-db \
  -v ai-agent-vectordb:/app/vector_db_storage \
  ai-agent-rag
```

But please run the script in `ai-agent-ms-network` to get the `.env` file.

## Key Differences from Windows Environment

The `environment_docker.yml` removes these Windows-specific packages:

- `icc_rt` - Intel C++ Compiler Runtime
- `intel-openmp` - Intel OpenMP (Windows version)
- `ucrt` - Universal C Runtime
- `vc` - Visual C++
- `vc14_runtime` - Visual C++ Runtime
- `vs2015_runtime` - Visual Studio Runtime

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_ID` | Google Cloud Project ID | `personal-rag-ai-agent` |
| `LOCATION` | Google Cloud Region | `asia-southeast1` |
| `FLASK_ENV` | Flask Environment | `production` |
| `FLASK_DEBUG` | Flask Debug Mode | `0` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON | `/app/credentials/service-account.json` |

## Volumes

- `ai-agent-files` - Persistent storage for uploaded PDF files
- `ai-agent-vectordb` - Persistent storage for vector database and metadata

## Health Check

The container includes a health check that calls the `/health` endpoint every 30 seconds.

## Troubleshooting

### Check container logs
```bash
docker-compose logs ai-agent
```

### Access container shell
```bash
docker-compose exec ai-agent bash
```

### Verify conda environment
```bash
docker-compose exec ai-agent conda info --envs
```

### Test API endpoints
```bash
# Health check
curl http://localhost:5000/health

# List files
curl http://localhost:5000/files
```

## Production Considerations

1. **Security**: Never include service account keys in the Docker image
2. **Volumes**: Use named volumes or bind mounts for data persistence
3. **Networking**: Use Docker networks for service communication
4. **Monitoring**: Add proper logging and monitoring solutions
5. **Scaling**: Consider using Kubernetes for production deployments