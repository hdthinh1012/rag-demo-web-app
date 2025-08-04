# Clean up script for Next.js app with nginx sidecar
# Stops containers, removes containers, and optionally removes network and images

param(
    [switch]$RemoveImages,
    [switch]$RemoveNetwork,
    [string]$NetworkName = "personal-rag-agent-ui-network",
    [string]$NextjsContainer = "personal-rag-agent-ui",
    [string]$SidecarContainer = "ui-nginx-sidecar"
)

Write-Host "🧹 Cleaning up Personal RAG Agent UI with Nginx Sidecar..." -ForegroundColor Yellow

# Step 1: Stop containers
Write-Host "🛑 Stopping containers..." -ForegroundColor Cyan
$containers = @($NextjsContainer, $SidecarContainer)
foreach ($container in $containers) {
    $exists = docker ps -a --filter name=$container --format "{{.Names}}" | Select-String -Pattern "^$container$"
    if ($exists) {
        Write-Host "  Stopping: $container" -ForegroundColor White
        docker stop $container 2>$null
    }
}

# Step 2: Remove containers
Write-Host "🗑️  Removing containers..." -ForegroundColor Cyan
foreach ($container in $containers) {
    $exists = docker ps -a --filter name=$container --format "{{.Names}}" | Select-String -Pattern "^$container$"
    if ($exists) {
        Write-Host "  Removing: $container" -ForegroundColor White
        docker rm $container 2>$null
    }
}

# Step 3: Remove images (if requested)
if ($RemoveImages) {
    Write-Host "🗑️  Removing images..." -ForegroundColor Cyan
    $images = @("hdthinh1012/personal-rag-agent-ui:latest", "personal-rag-agent-ui-nginx-sidecar")
    foreach ($image in $images) {
        $exists = docker images --filter reference=$image --format "{{.Repository}}:{{.Tag}}" | Select-String -Pattern $image
        if ($exists) {
            Write-Host "  Removing image: $image" -ForegroundColor White
            docker rmi $image 2>$null
        }
    }
}

# Step 4: Remove network (if requested)
if ($RemoveNetwork) {
    Write-Host "🗑️  Removing network..." -ForegroundColor Cyan
    $networkExists = docker network ls --filter name=$NetworkName --format "{{.Name}}" | Select-String -Pattern "^$NetworkName$"
    if ($networkExists) {
        Write-Host "  Removing network: $NetworkName" -ForegroundColor White
        docker network rm $NetworkName 2>$null
    }
}

# Step 5: Show cleanup summary
Write-Host "`n✅ Cleanup completed!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray

Write-Host "📊 Remaining containers:" -ForegroundColor Cyan
$remainingContainers = docker ps -a --filter name=$NextjsContainer --filter name=$SidecarContainer --format "{{.Names}}"
if ($remainingContainers) {
    docker ps -a --filter name=$NextjsContainer --filter name=$SidecarContainer --format "table {{.Names}}\t{{.Status}}"
} else {
    Write-Host "  No containers remaining" -ForegroundColor Green
}

if ($RemoveImages) {
    Write-Host "`n📊 Remaining images:" -ForegroundColor Cyan
    $remainingImages = docker images --filter reference="*personal-rag-agent-ui*" --format "{{.Repository}}:{{.Tag}}"
    if ($remainingImages) {
        docker images --filter reference="*personal-rag-agent-ui*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    } else {
        Write-Host "  No related images remaining" -ForegroundColor Green
    }
}

Write-Host "`n💡 Usage examples:" -ForegroundColor Cyan
Write-Host "  • Clean containers only:     ./clean-with-sidecar.ps1" -ForegroundColor White
Write-Host "  • Clean + remove images:     ./clean-with-sidecar.ps1 -RemoveImages" -ForegroundColor White
Write-Host "  • Clean + remove network:    ./clean-with-sidecar.ps1 -RemoveNetwork" -ForegroundColor White
Write-Host "  • Clean everything:          ./clean-with-sidecar.ps1 -RemoveImages -RemoveNetwork" -ForegroundColor White

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray