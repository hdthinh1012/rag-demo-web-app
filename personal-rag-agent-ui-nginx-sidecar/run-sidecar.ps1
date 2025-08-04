# Run nginx sidecar container
# Starts the nginx reverse proxy for Next.js app

param(
    [string]$NetworkName = "personal-rag-agent-ui-network",
    [string]$ContainerName = "ui-nginx-sidecar",
    [int]$Port = 81
)

Write-Host "Starting nginx sidecar container..." -ForegroundColor Green

# Check if network exists
$networkExists = docker network ls --filter name=$NetworkName --format "{{.Name}}" | Select-String -Pattern "^$NetworkName$"
if (-not $networkExists) {
    Write-Host "Creating Docker network: $NetworkName" -ForegroundColor Yellow
    docker network create $NetworkName
}

# Stop and remove existing container if it exists
$existingContainer = docker ps -a --filter name=$ContainerName --format "{{.Names}}" | Select-String -Pattern "^$ContainerName$"
if ($existingContainer) {
    Write-Host "Stopping and removing existing container: $ContainerName" -ForegroundColor Yellow
    docker stop $ContainerName 2>$null
    docker rm $ContainerName 2>$null
}

# Run the nginx sidecar container
Write-Host "Running nginx sidecar on port $Port..." -ForegroundColor Cyan
docker run -d `
    --name $ContainerName `
    --network $NetworkName `
    -p "${Port}:80" `
    --restart unless-stopped `
    personal-rag-agent-ui-nginx-sidecar

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Nginx sidecar started successfully!" -ForegroundColor Green
    Write-Host "🌐 Access URL: http://localhost:$Port" -ForegroundColor Cyan
    Write-Host "🔍 Check logs: docker logs $ContainerName" -ForegroundColor Yellow
    Write-Host "🏥 Health check: http://localhost:$Port/health" -ForegroundColor Magenta
} else {
    Write-Host "❌ Failed to start nginx sidecar container" -ForegroundColor Red
    exit $LASTEXITCODE
}