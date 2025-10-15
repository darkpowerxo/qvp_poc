# Docker helper scripts for QVP Platform (PowerShell)

function Build-QVP {
    Write-Host "Building QVP Docker image..." -ForegroundColor Green
    docker build -t qvp-platform:latest .
}

function Run-QVPDemo {
    Write-Host "Running QVP demo in Docker..." -ForegroundColor Green
    docker run --rm -v "${PWD}/data:/app/data" qvp-platform:latest
}

function Start-QVPServices {
    Write-Host "Starting all QVP services..." -ForegroundColor Green
    docker-compose up -d
}

function Stop-QVPServices {
    Write-Host "Stopping all QVP services..." -ForegroundColor Green
    docker-compose down
}

function Get-QVPLogs {
    param(
        [string]$Service = "qvp-app"
    )
    docker-compose logs -f $Service
}

function Enter-QVPContainer {
    param(
        [string]$Service = "qvp-app"
    )
    docker-compose exec $Service bash
}

function Rebuild-QVP {
    Write-Host "Rebuilding and restarting services..." -ForegroundColor Green
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
}

function Get-QVPStatus {
    docker-compose ps
}

# Export functions
Export-ModuleMember -Function Build-QVP, Run-QVPDemo, Start-QVPServices, Stop-QVPServices, Get-QVPLogs, Enter-QVPContainer, Rebuild-QVP, Get-QVPStatus

Write-Host @"
QVP Docker Helper Functions Loaded:
  Build-QVP              - Build the Docker image
  Run-QVPDemo            - Run demo script in Docker
  Start-QVPServices      - Start all services with docker-compose
  Stop-QVPServices       - Stop all services
  Get-QVPLogs [service]  - View logs (default: qvp-app)
  Enter-QVPContainer     - Execute bash in container
  Rebuild-QVP            - Rebuild and restart all services
  Get-QVPStatus          - Show service status
"@ -ForegroundColor Cyan
