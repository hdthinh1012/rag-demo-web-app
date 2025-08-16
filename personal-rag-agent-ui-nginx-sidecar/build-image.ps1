# Build script for nginx sidecar container
# Builds Docker image for reverse proxy to Next.js app

Write-Host "Building nginx sidecar image for Personal RAG Agent UI..." -ForegroundColor Green

# Build the Docker image
docker build -t hdthinh1012/personal-rag-agent-ui-nginx-sidecar:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Nginx sidecar image built successfully!" -ForegroundColor Green
    Write-Host "Image name: personal-rag-agent-ui-nginx-sidecar" -ForegroundColor Cyan
    Write-Host "To run: docker run -d --name ui-nginx-sidecar --network personal-rag-agent-ui-network -p 81:80 personal-rag-agent-ui-nginx-sidecar" -ForegroundColor Yellow
} else {
    Write-Host "❌ Failed to build nginx sidecar image" -ForegroundColor Red
    exit $LASTEXITCODE
}
