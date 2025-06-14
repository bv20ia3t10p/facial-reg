# Federated Learning System Setup Script
# Sets up FL + HE + DP system with 1 server + 2 clients

param(
    [switch]$SkipDependencies,
    [switch]$SkipDataSetup,
    [switch]$StartServices,
    [string]$DataPath = "D:\data",
    [string]$ModelsPath = "D:\models",
    [string]$LogsPath = "D:\logs"
)

Write-Host "=== Federated Learning System Setup ===" -ForegroundColor Green
Write-Host "Setting up FL + HE + DP with 1 server + 2 clients" -ForegroundColor Yellow

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Warning "Running without administrator privileges. Some operations may fail."
}

# Function to create directories
function New-DirectoryIfNotExists {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Host "Created directory: $Path" -ForegroundColor Green
    } else {
        Write-Host "Directory exists: $Path" -ForegroundColor Gray
    }
}

# Function to check system requirements
function Test-SystemRequirements {
    Write-Host "`n--- Checking System Requirements ---" -ForegroundColor Cyan
    
    # Check available memory
    $memory = Get-WmiObject -Class Win32_ComputerSystem
    $totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 2)
    Write-Host "Total System Memory: $totalMemoryGB GB"
    
    if ($totalMemoryGB -lt 8) {
        Write-Warning "System has less than 8GB RAM. Federated learning may be slow."
    }
    
    # Check available disk space on D: drive
    if (Test-Path "D:\") {
        $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='D:'"
        $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
        Write-Host "D: Drive Free Space: $freeSpaceGB GB"
        
        if ($freeSpaceGB -lt 10) {
            Write-Warning "D: drive has less than 10GB free space. Consider cleaning up."
        }
    } else {
        Write-Warning "D: drive not found. Using current drive."
        $DataPath = ".\data"
        $ModelsPath = ".\models"
        $LogsPath = ".\logs"
    }
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Host "Docker: $dockerVersion" -ForegroundColor Green
    } catch {
        Write-Error "Docker not found. Please install Docker Desktop."
        exit 1
    }
    
    # Check Python
    try {
        $pythonVersion = python --version
        Write-Host "Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Warning "Python not found in PATH. Docker will handle Python dependencies."
    }
}

# Function to setup directory structure
function Initialize-DirectoryStructure {
    Write-Host "`n--- Setting up Directory Structure ---" -ForegroundColor Cyan
    
    # Main directories
    New-DirectoryIfNotExists $DataPath
    New-DirectoryIfNotExists $ModelsPath
    New-DirectoryIfNotExists $LogsPath
    New-DirectoryIfNotExists "$DataPath\cache"
    New-DirectoryIfNotExists "$DataPath\redis"
    New-DirectoryIfNotExists "$DataPath\prometheus"
    New-DirectoryIfNotExists "$DataPath\grafana"
    
    # Federated learning directories
    New-DirectoryIfNotExists "$DataPath\client1"
    New-DirectoryIfNotExists "$DataPath\client2"
    New-DirectoryIfNotExists "$DataPath\federated"
    
    # Model directories
    New-DirectoryIfNotExists "$ModelsPath\backup"
    New-DirectoryIfNotExists "$ModelsPath\federated"
    
    # Log directories
    New-DirectoryIfNotExists "$LogsPath\api"
    New-DirectoryIfNotExists "$LogsPath\federated"
    New-DirectoryIfNotExists "$LogsPath\clients"
    
    Write-Host "Directory structure created successfully!" -ForegroundColor Green
}

# Function to setup sample data for clients
function Initialize-ClientData {
    param([switch]$SkipDataSetup)
    
    if ($SkipDataSetup) {
        Write-Host "Skipping data setup as requested." -ForegroundColor Yellow
        return
    }
    
    Write-Host "`n--- Setting up Client Data ---" -ForegroundColor Cyan
    
    # Create sample data structure for clients
    $client1DataPath = "$DataPath\client1"
    $client2DataPath = "$DataPath\client2"
    
    # Create identity folders for client 1
    for ($i = 1; $i -le 5; $i++) {
        New-DirectoryIfNotExists "$client1DataPath\identity_$i"
    }
    
    # Create identity folders for client 2
    for ($i = 6; $i -le 10; $i++) {
        New-DirectoryIfNotExists "$client2DataPath\identity_$i"
    }
    
    # Create sample configuration files
    $client1Config = @{
        client_id = "client1"
        client_type = "mobile"
        data_path = "/app/data/client1"
        privacy_budget = 50.0
        local_epochs = 1
        batch_size = 8
    }
    
    $client2Config = @{
        client_id = "client2"
        client_type = "mobile"
        data_path = "/app/data/client2"
        privacy_budget = 50.0
        local_epochs = 1
        batch_size = 8
    }
    
    $client1Config | ConvertTo-Json | Out-File "$client1DataPath\config.json" -Encoding UTF8
    $client2Config | ConvertTo-Json | Out-File "$client2DataPath\config.json" -Encoding UTF8
    
    Write-Host "Client data structure created!" -ForegroundColor Green
    Write-Host "Note: Add actual biometric images to client directories for training." -ForegroundColor Yellow
}

# Function to install Python dependencies
function Install-Dependencies {
    param([switch]$SkipDependencies)
    
    if ($SkipDependencies) {
        Write-Host "Skipping dependency installation as requested." -ForegroundColor Yellow
        return
    }
    
    Write-Host "`n--- Installing Dependencies ---" -ForegroundColor Cyan
    
    try {
        # Check if requirements.txt exists
        if (Test-Path "requirements.txt") {
            Write-Host "Installing Python dependencies..."
            pip install -r requirements.txt
            Write-Host "Dependencies installed successfully!" -ForegroundColor Green
        } else {
            Write-Warning "requirements.txt not found. Skipping Python dependency installation."
        }
    } catch {
        Write-Warning "Failed to install Python dependencies. Docker will handle this."
    }
}

# Function to build Docker images
function Build-DockerImages {
    Write-Host "`n--- Building Docker Images ---" -ForegroundColor Cyan
    
    try {
        Write-Host "Building biometric system Docker image..."
        docker build -t biometric-system .
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Docker image built successfully!" -ForegroundColor Green
        } else {
            Write-Error "Failed to build Docker image."
            exit 1
        }
    } catch {
        Write-Error "Error building Docker image: $_"
        exit 1
    }
}

# Function to create monitoring configuration
function Initialize-MonitoringConfig {
    Write-Host "`n--- Setting up Monitoring Configuration ---" -ForegroundColor Cyan
    
    # Create monitoring directory
    New-DirectoryIfNotExists "monitoring"
    
    # Prometheus configuration
    $prometheusConfig = @"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'biometric-api'
    static_configs:
      - targets: ['biometric-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'federated-coordinator'
    static_configs:
      - targets: ['federated-coordinator:8001']
    metrics_path: '/health'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
"@
    
    $prometheusConfig | Out-File "monitoring\prometheus.yml" -Encoding UTF8
    
    # Create Grafana directories
    New-DirectoryIfNotExists "monitoring\grafana\dashboards"
    New-DirectoryIfNotExists "monitoring\grafana\datasources"
    
    # Grafana datasource configuration
    $grafanaDatasource = @"
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
"@
    
    $grafanaDatasource | Out-File "monitoring\grafana\datasources\prometheus.yml" -Encoding UTF8
    
    Write-Host "Monitoring configuration created!" -ForegroundColor Green
}

# Function to start services
function Start-FederatedServices {
    param([switch]$StartServices)
    
    if (-not $StartServices) {
        Write-Host "Use -StartServices flag to automatically start services." -ForegroundColor Yellow
        return
    }
    
    Write-Host "`n--- Starting Federated Learning Services ---" -ForegroundColor Cyan
    
    try {
        # Stop any existing services
        Write-Host "Stopping existing services..."
        docker-compose down 2>$null
        
        # Start services
        Write-Host "Starting federated learning system..."
        docker-compose up -d
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Services started successfully!" -ForegroundColor Green
            
            # Wait for services to be ready
            Write-Host "Waiting for services to initialize..."
            Start-Sleep -Seconds 30
            
            # Check service status
            Write-Host "`n--- Service Status ---" -ForegroundColor Cyan
            docker-compose ps
            
            # Display access URLs
            Write-Host "`n--- Access URLs ---" -ForegroundColor Green
            Write-Host "Biometric API: http://localhost:8000" -ForegroundColor White
            Write-Host "Federated Coordinator: http://localhost:8001" -ForegroundColor White
            Write-Host "Database Admin: http://localhost:8080" -ForegroundColor White
            Write-Host "Grafana Dashboard: http://localhost:3000 (admin/admin123)" -ForegroundColor White
            Write-Host "Prometheus: http://localhost:9090" -ForegroundColor White
            
        } else {
            Write-Error "Failed to start services."
        }
    } catch {
        Write-Error "Error starting services: $_"
    }
}

# Function to run system tests
function Test-FederatedSystem {
    Write-Host "`n--- Testing Federated System ---" -ForegroundColor Cyan
    
    # Test API health
    try {
        $apiHealth = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 10
        Write-Host "✓ Biometric API is healthy" -ForegroundColor Green
    } catch {
        Write-Host "✗ Biometric API is not responding" -ForegroundColor Red
    }
    
    # Test Federated Coordinator
    try {
        $coordinatorHealth = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -TimeoutSec 10
        Write-Host "✓ Federated Coordinator is healthy" -ForegroundColor Green
    } catch {
        Write-Host "✗ Federated Coordinator is not responding" -ForegroundColor Red
    }
    
    # Test Federated Status
    try {
        $federatedStatus = Invoke-RestMethod -Uri "http://localhost:8000/federated/status" -Method Get -TimeoutSec 10
        Write-Host "✓ Federated integration is working" -ForegroundColor Green
        Write-Host "  - Active clients: $($federatedStatus.active_clients)" -ForegroundColor Gray
        Write-Host "  - HE enabled: $($federatedStatus.he_enabled)" -ForegroundColor Gray
    } catch {
        Write-Host "✗ Federated integration is not working" -ForegroundColor Red
    }
}

# Function to display usage instructions
function Show-UsageInstructions {
    Write-Host "`n=== Federated Learning System Ready ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Add biometric training data to client directories:"
    Write-Host "   - $DataPath\client1\ (for identities 1-5)"
    Write-Host "   - $DataPath\client2\ (for identities 6-10)"
    Write-Host ""
    Write-Host "2. Start a federated learning round:"
    Write-Host "   curl -X POST http://localhost:8000/federated/trigger-round"
    Write-Host ""
    Write-Host "3. Monitor the system:"
    Write-Host "   - API Status: http://localhost:8000/health"
    Write-Host "   - Federated Status: http://localhost:8000/federated/status"
    Write-Host "   - Grafana Dashboard: http://localhost:3000"
    Write-Host ""
    Write-Host "4. View logs:"
    Write-Host "   docker-compose logs -f federated-coordinator"
    Write-Host "   docker-compose logs -f federated-client1"
    Write-Host "   docker-compose logs -f federated-client2"
    Write-Host ""
    Write-Host "5. Stop the system:"
    Write-Host "   docker-compose down"
    Write-Host ""
    Write-Host "System Features:" -ForegroundColor Cyan
    Write-Host "✓ Federated Learning (FL) - Distributed training across clients"
    Write-Host "✓ Homomorphic Encryption (HE) - Encrypted model updates"
    Write-Host "✓ Differential Privacy (DP) - Privacy-preserving training"
    Write-Host "✓ Hybrid Architecture - API + Federated service + Shared storage"
    Write-Host "✓ Memory Optimized - Designed for limited RAM systems"
    Write-Host "✓ Monitoring & Logging - Comprehensive system monitoring"
}

# Main execution
try {
    Test-SystemRequirements
    Initialize-DirectoryStructure
    Initialize-ClientData -SkipDataSetup:$SkipDataSetup
    Install-Dependencies -SkipDependencies:$SkipDependencies
    Build-DockerImages
    Initialize-MonitoringConfig
    Start-FederatedServices -StartServices:$StartServices
    
    if ($StartServices) {
        Start-Sleep -Seconds 10
        Test-FederatedSystem
    }
    
    Show-UsageInstructions
    
} catch {
    Write-Error "Setup failed: $_"
    exit 1
}

Write-Host "`nSetup completed successfully! 🎉" -ForegroundColor Green
 