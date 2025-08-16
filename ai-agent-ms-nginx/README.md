# Nginx Reverse Proxy for AI Agent RAG Microservice

This directory contains Nginx configuration for load balancing and reverse proxying the AI Agent RAG microservice with special handling for form-data file uploads.

## Features

- **Load Balancing**: Distributes requests across multiple AI Agent backend instances
- **File Upload Optimization**: Handles large PDF uploads (up to 20MB) to `/generate-response`
- **CORS Support**: Configured for frontend integration
- **Health Monitoring**: Proper health check routing
- **Security Headers**: Basic security headers included
- **Extended Timeouts**: Optimized for AI processing workloads
- **Docker Network Integration**: Works with Docker's internal DNS resolution

## Configuration Highlights

### Upstream Backend
```nginx
upstream ai_agent_backend {
    server ai-agent-ms:5000 max_fails=3 fail_timeout=30s;
    # Uses Docker container name for internal network routing
}
```

### File Upload Handling
- **Max file size**: 20MB (slightly higher than Flask's 16MB limit)
- **Extended timeouts**: 5 minutes for upload and processing
- **Disabled buffering**: For streaming large files
- **Preserved headers**: Maintains `Content-Type` for multipart/form-data

### Endpoints Configured
- `POST /generate-response` - Form-data file upload with extended timeouts
- `GET /health` - Health check with quick timeouts
- `GET /files` - File listing
- `POST /storage/rebuild` - Vector database rebuild (10-minute timeout)
- `GET /storage/info` - Storage information
- `/debug/*` - Debug endpoints (can be disabled in production)

## Usage

### 1. Prerequisites

Make sure your AI Agent container is running in the `ai-agent-ms-network`:

```bash
# Create the network if it doesn't exist
docker network create ai-agent-ms-network

# Verify your ai-agent-ms container is running in this network
docker network inspect ai-agent-ms-network
```

### 2. Build and Run Nginx Container

#### Option A: Using the Build Script (Recommended)

```bash
# Make the script executable (Linux/macOS)
chmod +x build-and-run.sh

# Run the automated setup
./build-and-run.sh
```

#### Option B: Manual Commands

```bash
# Create network if needed
docker network create ai-agent-ms-network

# Build the nginx image with custom configuration
docker build -t ai-agent-nginx .

# Run nginx container in the same network
docker run -d \
  --name ai-agent-nginx \
  --network ai-agent-ms-network \
  -p 80:80 \
  -p 443:443 \
  -v $(pwd)/ssl:/etc/nginx/ssl:ro \
  --restart unless-stopped \
  ai-agent-nginx

# Check nginx logs
docker logs ai-agent-nginx

# Follow nginx logs
docker logs -f ai-agent-nginx
```

### 3. Test the Setup

```bash
# Health check through nginx
curl http://localhost/health

# File upload through nginx
curl -X POST http://localhost/generate-response \
  -F "query=What is this document about?" \
  -F "files=@test.pdf"

# List files
curl http://localhost/files
```

### 4. Container Management

```bash
# Stop nginx
docker stop ai-agent-nginx

# Remove nginx container
docker rm ai-agent-nginx

# Restart nginx
docker restart ai-agent-nginx

# Update nginx configuration (after editing nginx.conf)
docker exec ai-agent-nginx nginx -s reload
```

### 5. SSL/HTTPS Setup (Future)

The Dockerfile is prepared for SSL/HTTPS setup. When you're ready to enable HTTPS:

#### Step 1: Generate or Obtain SSL Certificates

```bash
# Create SSL directory
mkdir -p ssl

# Option A: Self-signed certificates (for development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/server.key \
  -out ssl/server.crt \
  -subj "/CN=localhost"

# Option B: Let's Encrypt certificates (for production)
# Copy your certificates to the ssl directory
# cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/server.crt
# cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/server.key
```

#### Step 2: Enable HTTPS in nginx.conf

Uncomment the HTTPS server block in `nginx.conf`:

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;
    
    # ... rest of your location blocks
}
```

#### Step 3: Update Dockerfile

Uncomment the HTTPS port exposure in `Dockerfile`:

```dockerfile
EXPOSE 443  # Uncomment this line
```

#### Step 4: Rebuild and Deploy

```bash
# Rebuild the image
docker build -t ai-agent-nginx .

# Stop existing container
docker stop ai-agent-nginx && docker rm ai-agent-nginx

# Run with SSL support
docker run -d \
  --name ai-agent-nginx \
  --network ai-agent-ms-network \
  -p 80:80 \
  -p 443:443 \
  -v $(pwd)/ssl:/etc/nginx/ssl:ro \
  --restart unless-stopped \
  ai-agent-nginx
```

### 6. Production Deployment

1. **Update upstream servers** in `nginx.conf`:
   ```nginx
   upstream ai_agent_backend {
       server ai-agent-ms:5000;
       server ai-agent-ms-2:5000;  # For multiple instances
       server ai-agent-ms-3:5000;
   }
   ```

2. **Deploy with proper SSL certificates** following the SSL setup steps above.

## File Structure

```
ai-agent-ms-nginx/
├── Dockerfile           # Simple nginx Docker build with SSL readiness
├── nginx.conf           # Main nginx configuration with detailed comments
├── health-check.sh      # Health check script for container monitoring
├── build-and-run.sh     # Automated build and deployment script
├── env.template         # Environment template
└── README.md           # This file
```

## Configuration Details

### Upstream Configuration
- **Health checks**: Fails after 3 consecutive failures
- **Failover**: 30-second timeout before retry
- **Load balancing**: Round-robin (default)

### Timeouts
| Endpoint | Connect | Send | Read | Purpose |
|----------|---------|------|------|---------|
| `/generate-response` | 30s | 300s | 300s | File upload/processing |
| `/health` | 5s | 10s | 10s | Quick health checks |
| `/storage/rebuild` | 30s | 600s | 600s | Vector DB rebuild |
| Default | 30s | 120s | 120s | General requests |

### Security Features
- **CORS headers**: Configured for `http://localhost:3000`
- **Security headers**: XSS protection, content type sniffing prevention
- **Method restrictions**: Only POST allowed on `/generate-response`
- **Debug endpoints**: Can be disabled in production

## Monitoring

### Nginx Logs
```bash
# Access logs
docker exec ai-agent-nginx tail -f /var/log/nginx/access.log

# Error logs
docker exec ai-agent-nginx tail -f /var/log/nginx/error.log

# View container logs
docker logs -f ai-agent-nginx
```

### Health Checks
```bash
# Check backend health through nginx
curl http://localhost/health

# Direct backend health (if accessible)
curl http://ai-agent-ms:5000/health
```

## Scaling

### Horizontal Scaling
```bash
# Run multiple AI Agent instances
docker run -d --name ai-agent-ms-2 --network ai-agent-ms-network [ai-agent-options]
docker run -d --name ai-agent-ms-3 --network ai-agent-ms-network [ai-agent-options]

# Update nginx.conf to include new instances
# Then reload nginx configuration
docker exec ai-agent-nginx nginx -s reload
```

### Load Testing
```bash
# Install Apache Bench
apt-get install apache2-utils

# Test health endpoint
ab -n 1000 -c 10 http://localhost/health

# Test file upload (prepare test.pdf first)
ab -n 10 -c 2 -p test.pdf -T 'multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' http://localhost/generate-response
```

## Troubleshooting

### Common Issues

1. **413 Request Entity Too Large**
   - Check `client_max_body_size` in nginx.conf
   - Ensure it's >= Flask's `MAX_CONTENT_LENGTH`

2. **504 Gateway Timeout**
   - Increase `proxy_read_timeout` for processing-heavy endpoints
   - Check backend service health

3. **502 Bad Gateway**
   - Verify backend service is running
   - Check upstream server configuration
   - Verify network connectivity between containers

4. **CORS Issues**
   - Update `Access-Control-Allow-Origin` headers
   - Check preflight OPTIONS handling

### Debug Commands
```bash
# Test nginx configuration
docker exec ai-agent-nginx nginx -t

# Reload nginx configuration
docker exec ai-agent-nginx nginx -s reload

# Check nginx status
docker exec ai-agent-nginx ps aux | grep nginx

# Inspect network connectivity
docker network inspect ai-agent-ms-network
```

# SSL Setup

This guide shows how to create self-signed certificates using your own Certificate Authority (CA) for Docker-based Nginx deployment.

## Step 1: Generate Your Own CA Certificate

Create a fake Certificate Authority (without real authority) in the standard CA directory:

```bash
# Create CA directory structure
sudo mkdir -p /etc/pki/CA/{private,certs,newcerts,crl}
sudo touch /etc/pki/CA/index.txt
echo 1000 | sudo tee /etc/pki/CA/serial

# Navigate to CA folder
cd /etc/pki/CA

# Generate CA private key (4096-bit RSA)
sudo openssl genrsa -out private/cakey.pem 4096

# Create CA certificate signing request using openssl.cnf
# Edit the default certificate information in openssl.cnf
sudo cp /etc/ssl/openssl.cnf /etc/ssl/openssl.cnf.bak
sudo sed -i 's/^countryName_default.*/countryName_default = US/' /etc/ssl/openssl.cnf
sudo sed -i 's/^stateOrProvinceName_default.*/stateOrProvinceName_default = CA/' /etc/ssl/openssl.cnf
sudo sed -i 's/^localityName_default.*/localityName_default = San Francisco/' /etc/ssl/openssl.cnf
sudo sed -i 's/^0.organizationName_default.*/0.organizationName_default = Personal Certificate Authority Company/' /etc/ssl/openssl.cnf
sudo sed -i 's/^organizationalUnitName_default.*/organizationalUnitName_default = IT Department/' /etc/ssl/openssl.cnf
sudo sed -i 's/^commonName_default.*/commonName_default = Personal CA/' /etc/ssl/openssl.cnf

# Generate CA certificate signing request (will use defaults from openssl.cnf)
sudo openssl req -new -key private/cakey.pem -out ca.csr -config /etc/ssl/openssl.cnf

# Self-sign the CA certificate (valid for 10 years)
sudo openssl x509 -req -in ca.csr -signkey private/cakey.pem -out cacert.pem -days 3650

# Set proper permissions
sudo chmod 600 private/cakey.pem
sudo chmod 644 cacert.pem
```

## Step 2: Create Server Certificate Signed by Your CA

Generate a certificate for your server (replace `YOUR_SERVER_IP` with your actual IP):

```bash
# Navigate to your project's certs directory
cd /home/hdthinh1012/projects/aws-project/rag-demo-web-app/ai-agent-ms-nginx/certs

# Generate server private key
openssl genrsa -out server.key 4096

# Create server certificate signing request (CSR)
# IMPORTANT: Use the SAME organizationName as your CA certificate
openssl req -new -key server.key -out server.csr \
  -subj "/C=US/ST=CA/L=San Francisco/O=Personal Certificate Authority Company/CN=YOUR_SERVER_IP"

# Create Subject Alternative Name (SAN) file for IP addresses
echo "subjectAltName=IP:YOUR_SERVER_IP" > san.txt

# Sign the server certificate using your CA
sudo openssl ca -in server.csr -out server.crt -days 365 -extfile san.txt

# Copy CA certificate to project directory (for client trust)
sudo cp /etc/pki/CA/cacert.pem ./ca.crt
```

**Important Notes:**
- The `organizationName` (O field) in server CSR **must match** the CA certificate's organizationName
- If they differ, OpenSSL will reject the signing with: "The organizationName field is different between CA certificate and the request"
- This is a security feature to prevent unauthorized certificate issuance
- **Bypass option:** You can override this requirement by using `policy = policy_anything` in the openssl.cnf `[ CA_default ]` section instead of the default `policy = policy_match`

## Step 3: Configure Docker Nginx with SSL

Update your project structure to include SSL certificates:

```bash
# Project structure after SSL setup
ai-agent-ms-nginx/
├── certs/
│   ├── server.key      # Server private key
│   ├── server.crt      # Server certificate (signed by CA)
│   ├── ca.crt          # CA certificate (for client trust)
│   └── san.txt         # Subject Alternative Name file
├── Dockerfile
├── nginx.conf
└── README.md
```

**Enable HTTPS in nginx.conf:**

Uncomment and configure the HTTPS server block:

```nginx
server {
    listen 443 ssl http2;
    server_name YOUR_SERVER_IP;
    
    ssl_certificate /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Your location blocks here...
}
```

**Build and deploy with SSL:**

```bash
# Build the Docker image
docker build -t ai-agent-nginx .

# Run container with SSL certificates mounted
docker run -d \
  --name ai-agent-nginx \
  --network ai-agent-ms-network \
  -p 80:80 \
  -p 443:443 \
  -v $(pwd)/certs:/etc/nginx/ssl:ro \
  --restart unless-stopped \
  ai-agent-nginx
```

**Test HTTPS connection:**

```bash
# Test with self-signed certificate (ignore certificate warnings)
curl -k https://YOUR_SERVER_IP/health

# For browsers: Import ca.crt into your browser's trusted certificates
# to avoid security warnings
```

## Troubleshooting SSL Issues

**Certificate Authority organizationName mismatch:**
```bash
# Check CA certificate organization
sudo openssl x509 -in /etc/pki/CA/cacert.pem -text -noout | grep "Subject:"

# Check server CSR organization  
openssl req -in server.csr -text -noout | grep "Subject:"

# Both must have the same "O=" (organizationName) field
```

**Certificate verification:**
```bash
# Verify server certificate against CA
openssl verify -CAfile ca.crt server.crt

# Check certificate details
openssl x509 -in server.crt -text -noout
```

For production use, replace self-signed certificates with certificates from a trusted Certificate Authority like Let's Encrypt.

