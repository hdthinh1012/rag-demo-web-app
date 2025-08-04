# Nginx Sidecar for Personal RAG Agent UI

This directory contains the nginx sidecar configuration for the Personal RAG Agent UI Next.js application. The sidecar acts as a reverse proxy to the Next.js container.

## 🏗️ Architecture

```
Browser (localhost:81) → Nginx Sidecar → Next.js App (port 3000)
                ↓              ↓              ↓
              Port 81      Container      Container
                          Network:       Network:
                          ui-nginx-      personal-rag-
                          sidecar        agent-ui
```

## 📁 File Structure

```
personal-rag-agent-ui-nginx-sidecar/
├── Dockerfile           # Nginx sidecar Docker image
├── nginx.conf           # Nginx configuration with upstream
├── build-image.ps1      # Build Docker image script
├── run-sidecar.ps1      # Run sidecar container script
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Build the Nginx Sidecar Image

```powershell
# Navigate to sidecar directory
cd personal-rag-agent-ui-nginx-sidecar

# Build the image
./build-image.ps1
```

### 2. Run Both Containers

```powershell
# Navigate to network directory
cd ../personal-rag-agent-ui-network

# Setup complete stack with sidecar
./setup-with-sidecar.ps1
```

### 3. Access the Application

- **Via Nginx Sidecar**: http://localhost:81
- **Direct Next.js**: http://localhost:3000  
- **Sidecar Health Check**: http://localhost:81/health

## 🔧 Manual Setup

### Build Nginx Sidecar

```powershell
docker build -t personal-rag-agent-ui-nginx-sidecar .
```

### Create Network

```powershell
docker network create personal-rag-agent-ui-network
```

### Run Next.js Container

```powershell
docker run -d \
  --name personal-rag-agent-ui \
  --network personal-rag-agent-ui-network \
  -p 3000:3000 \
  hdthinh1012/personal-rag-agent-ui:latest
```

### Run Nginx Sidecar

```powershell
docker run -d \
  --name ui-nginx-sidecar \
  --network personal-rag-agent-ui-network \
  -p 81:80 \
  personal-rag-agent-ui-nginx-sidecar
```

## 📊 Container Communication

### Network Communication

```
┌─────────────────────────────────────────────────────────┐
│              personal-rag-agent-ui-network              │
├─────────────────────────┬───────────────────────────────┤
│  personal-rag-agent-ui  │      ui-nginx-sidecar         │
│  (Next.js App)          │      (Nginx Reverse Proxy)    │
│  Internal: :3000        │      Internal: :80            │
│  External: :3000        │      External: :81            │
└─────────────────────────┴───────────────────────────────┘
```

### Nginx Upstream Configuration

```nginx
upstream nextjs_backend {
    server personal-rag-agent-ui:3000 max_fails=3 fail_timeout=30s;
}
```

## 🧪 Testing the Setup

### Health Checks

```powershell
# Test nginx sidecar health
curl http://localhost:81/health

# Test Next.js through sidecar
curl http://localhost:81

# Test Next.js directly
curl http://localhost:3000
```

### Check Container Logs

```powershell
# Nginx sidecar logs
docker logs -f ui-nginx-sidecar

# Next.js app logs
docker logs -f personal-rag-agent-ui
```

### Network Inspection

```powershell
# Inspect the Docker network
docker network inspect personal-rag-agent-ui-network

# Check container connectivity
docker exec ui-nginx-sidecar nslookup personal-rag-agent-ui
```

## 🔍 Configuration Details

### Nginx Features

- **Upstream Configuration**: Load balancing to Next.js container
- **Health Checks**: Built-in health monitoring
- **CORS Support**: Cross-origin request handling
- **WebSocket Support**: For Next.js hot reload
- **Static File Caching**: Optimized static asset serving
- **Security Headers**: XSS, frame options, content type protection

### Performance Optimizations

- **Proxy Buffering**: Optimized for Next.js responses
- **Connection Keep-Alive**: Reduced connection overhead
- **Static Asset Caching**: 1-hour cache for static files
- **Gzip Compression**: Automatic response compression

### Timeouts & Limits

| Setting | Value | Purpose |
|---------|-------|---------|
| `client_max_body_size` | 50M | File upload support |
| `proxy_connect_timeout` | 30s | Backend connection |
| `proxy_read_timeout` | 60s | Response reading |
| `keepalive_timeout` | 65s | Connection reuse |

## 🛠️ Management Scripts

### Available Scripts

```powershell
# Build nginx sidecar image
./build-image.ps1

# Run sidecar container only
./run-sidecar.ps1

# Setup complete stack (Next.js + Sidecar)
cd ../personal-rag-agent-ui-network
./setup-with-sidecar.ps1

# Clean up everything
./clean-with-sidecar.ps1 -RemoveImages -RemoveNetwork
```

### Script Options

```powershell
# Custom network name
./setup-with-sidecar.ps1 -NetworkName "my-custom-network"

# Custom ports
./setup-with-sidecar.ps1 -NextjsPort 3001 -SidecarPort 82

# Selective cleanup
./clean-with-sidecar.ps1 -RemoveImages      # Remove images only
./clean-with-sidecar.ps1 -RemoveNetwork     # Remove network only
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Container Not Found Error

```
Error: No such container: personal-rag-agent-ui
```

**Solution**: Ensure Next.js container is running first
```powershell
docker ps -a | findstr personal-rag-agent-ui
```

#### 2. Network Connection Issues

```
nginx: [error] connect() failed (111: Connection refused)
```

**Solution**: Check both containers are on the same network
```powershell
docker network inspect personal-rag-agent-ui-network
```

#### 3. Port Already in Use

```
Error: port is already allocated
```

**Solution**: Use different port or stop conflicting container
```powershell
./run-sidecar.ps1 -Port 82
```

### Debug Commands

```powershell
# Check nginx configuration
docker exec ui-nginx-sidecar nginx -t

# Test internal connectivity
docker exec ui-nginx-sidecar wget -qO- http://personal-rag-agent-ui:3000

# Check container status
docker ps --filter name=personal-rag-agent-ui --filter name=ui-nginx-sidecar
```

## 📋 Comparison: Sidecar vs Reverse Proxy

| Aspect | Nginx Sidecar | Nginx Reverse Proxy |
|--------|---------------|---------------------|
| **Containers** | 2 containers (App + Sidecar) | 2+ containers (Proxy + Apps) |
| **Network** | Same network | Same network |
| **Use Case** | Single app proxy | Multiple app routing |
| **Complexity** | Medium | Lower |
| **Scalability** | App + proxy scale together | Independent scaling |

## 🎯 When to Use Nginx Sidecar

**✅ Use Sidecar When:**
- Kubernetes deployments
- Need app-specific proxy configuration
- Service mesh architecture
- Container-per-service model

**❌ Don't Use Sidecar When:**
- Simple single-app deployment
- Multiple apps sharing one proxy
- Development environment
- Resource constraints

For your Personal RAG Agent, the sidecar approach works well for learning container networking patterns, but a simple reverse proxy setup might be more practical for production use.