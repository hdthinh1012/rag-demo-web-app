# Complete setup script for Next.js app with nginx sidecar
# Creates network and runs both containers

param(
    [string]$NetworkName = "personal-rag-agent-ui-network",
    [string]$NextjsContainer = "personal-rag-agent-ui",
    [string]$SidecarContainer = "ui-nginx-sidecar",
    [int]$NextjsPort = 3000,
    [int]$SidecarPort = 81
)

Write-Host "🚀 Setting up Personal RAG Agent UI with Nginx Sidecar..." -ForegroundColor Green

# Step 1: Create Docker network
Write-Host "📡 Creating Docker network: $NetworkName" -ForegroundColor Cyan
$networkExists = docker network ls --filter name=$NetworkName --format "{{.Name}}" | Select-String -Pattern "^$NetworkName$"
if (-not $networkExists) {
    docker network create $NetworkName
    Write-Host "✅ Network created: $NetworkName" -ForegroundColor Green
} else {
    Write-Host "ℹ️  Network already exists: $NetworkName" -ForegroundColor Yellow
}

# Step 2: Stop existing containers
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
docker stop $NextjsContainer $SidecarContainer 2>$null
docker rm $NextjsContainer $SidecarContainer 2>$null

# Step 3: Run Next.js container
Write-Host "🔄 Starting Next.js container..." -ForegroundColor Cyan
docker run -d `
    --name $NextjsContainer `
    --network $NetworkName `
    -p "${NextjsPort}:3000" `
    --restart unless-stopped `
    hdthinh1012/personal-rag-agent-ui:latest

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start Next.js container" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Step 4: Run nginx sidecar container
Write-Host "🔄 Starting nginx sidecar container..." -ForegroundColor Cyan
docker run -d `
    --name $SidecarContainer `
    --network $NetworkName `
    -p "${SidecarPort}:80" `
    --restart unless-stopped `
    personal-rag-agent-ui-nginx-sidecar

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start nginx sidecar container" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Step 5: Wait for containers to be ready
Write-Host "⏳ Waiting for containers to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Step 6: Show status and access information
Write-Host "`n🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray

Write-Host "📊 Container Status:" -ForegroundColor Cyan
docker ps --filter name=$NextjsContainer --filter name=$SidecarContainer --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Write-Host "`n🌐 Access URLs:" -ForegroundColor Cyan
Write-Host "  • Next.js Direct:  http://localhost:$NextjsPort" -ForegroundColor White
Write-Host "  • Via Nginx Sidecar: http://localhost:$SidecarPort" -ForegroundColor White
Write-Host "  • Sidecar Health:    http://localhost:$SidecarPort/health" -ForegroundColor White

Write-Host "`n🔍 Useful Commands:" -ForegroundColor Cyan
Write-Host "  • Check logs (Next.js): docker logs -f $NextjsContainer" -ForegroundColor White
Write-Host "  • Check logs (Sidecar): docker logs -f $SidecarContainer" -ForegroundColor White
Write-Host "  • Stop all: docker stop $NextjsContainer $SidecarContainer" -ForegroundColor White
Write-Host "  • Network inspect: docker network inspect $NetworkName" -ForegroundColor White

Write-Host "`n🧪 Test the Setup:" -ForegroundColor Cyan
Write-Host "  curl http://localhost:$SidecarPort" -ForegroundColor White
Write-Host "  curl http://localhost:$SidecarPort/health" -ForegroundColor White

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray