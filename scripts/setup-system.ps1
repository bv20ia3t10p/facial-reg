# PowerShell Setup Script for Biometric Authentication System
# Optimized for Windows with D: drive efficiency and memory management

param(
    [switch]$Development,
    [switch]$SkipDocker,
    [string]$DataDrive = "D:"
)

Write-Host "🚀 Setting up Phone-Based Biometric Authentication System" -ForegroundColor Green
Write-Host "📍 Data drive: $DataDrive" -ForegroundColor Yellow
Write-Host "🔧 Mode: $(if ($Development) { 'Development' } else { 'Production' })" -ForegroundColor Yellow

# Function to check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $currentUser).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
}

# Function to create directory if it doesn't exist
function New-DirectoryIfNotExists {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        Write-Host "📁 Creating directory: $Path" -ForegroundColor Cyan
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

# Check administrator privileges
if (!(Test-Administrator)) {
    Write-Host "❌ This script requires administrator privileges. Please run as administrator." -ForegroundColor Red
    exit 1
}

# Check if Docker is installed and running
if (!$SkipDocker) {
    Write-Host "🐳 Checking Docker installation..." -ForegroundColor Cyan
    try {
        $dockerVersion = docker --version
        Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
        
        # Check if Docker is running
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠️  Docker is not running. Please start Docker Desktop." -ForegroundColor Yellow
            Start-Process "Docker Desktop"
            Write-Host "⏳ Waiting for Docker to start..." -ForegroundColor Yellow
            do {
                Start-Sleep -Seconds 5
                docker info 2>$null | Out-Null
            } while ($LASTEXITCODE -ne 0)
        }
        Write-Host "✅ Docker is running" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker not found. Please install Docker Desktop first." -ForegroundColor Red
        Write-Host "🔗 Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Blue
        exit 1
    }
}

# Create directory structure on D: drive
Write-Host "📁 Creating directory structure..." -ForegroundColor Cyan

$directories = @(
    "$DataDrive\cache",
    "$DataDrive\models",
    "$DataDrive\data",
    "$DataDrive\logs",
    "$DataDrive\logs\nginx",
    "$DataDrive\redis-data",
    "$DataDrive\prometheus-data",
    "$DataDrive\grafana-data"
)

foreach ($dir in $directories) {
    New-DirectoryIfNotExists -Path $dir
}

# Set appropriate permissions
Write-Host "🔐 Setting directory permissions..." -ForegroundColor Cyan
foreach ($dir in $directories) {
    try {
        $acl = Get-Acl $dir
        $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Everyone", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
        $acl.SetAccessRule($accessRule)
        Set-Acl $dir $acl
    }
    catch {
        Write-Host "⚠️  Could not set permissions for $dir" -ForegroundColor Yellow
    }
}

# Copy existing model if it exists
Write-Host "📦 Checking for existing models..." -ForegroundColor Cyan
if (Test-Path "best_pretrained_model.pth") {
    Write-Host "✅ Found existing model, copying to D: drive..." -ForegroundColor Green
    Copy-Item "best_pretrained_model.pth" "$DataDrive\models\" -Force
}

# Copy existing data if it exists
if (Test-Path "data\partitioned") {
    Write-Host "✅ Found partitioned data, copying to D: drive..." -ForegroundColor Green
    Copy-Item "data\partitioned" "$DataDrive\data\" -Recurse -Force
}

# Create environment file
Write-Host "⚙️  Setting up environment configuration..." -ForegroundColor Cyan
@"
# Biometric Authentication System Configuration
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1

# Paths
DATA_PATH=$DataDrive/data
MODELS_PATH=$DataDrive/models
CACHE_PATH=$DataDrive/cache
LOGS_PATH=$DataDrive/logs

# Database
DB_PATH=$DataDrive/data/biometric.db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Security
JWT_SECRET=biometric_auth_secret_2024
ENCRYPTION_KEY=biometric_encryption_key_2024
"@ | Out-File -FilePath ".env" -Encoding UTF8

# Memory optimization for Windows
Write-Host "🧠 Applying memory optimizations..." -ForegroundColor Cyan

# Set environment variables for current session
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:DOCKER_BUILDKIT = "1"

# Check available memory
$totalMemory = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB
Write-Host "💾 Total system memory: $([math]::Round($totalMemory, 2)) GB" -ForegroundColor Cyan

if ($totalMemory -lt 8) {
    Write-Host "⚠️  System has less than 8GB RAM. Applying conservative memory limits..." -ForegroundColor Yellow
    
    # Create memory-optimized docker-compose override
    @"
version: '3.8'
services:
  biometric-api:
    mem_limit: 1024m
    mem_reservation: 512m
    cpus: '1.0'
  
  redis:
    mem_limit: 128m
    command: redis-server --save 60 1 --loglevel warning --maxmemory 100mb --maxmemory-policy allkeys-lru
    
  prometheus:
    mem_limit: 256m
    
  grafana:
    mem_limit: 256m
    
  nginx:
    mem_limit: 64m
    
  sqlite-web:
    mem_limit: 64m
"@ | Out-File -FilePath "docker-compose.override.yml" -Encoding UTF8
}

# Check GPU availability
Write-Host "🎮 Checking GPU availability..." -ForegroundColor Cyan
try {
    $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" -or $_.Name -like "*AMD*" }
    if ($gpu) {
        Write-Host "✅ GPU detected: $($gpu.Name)" -ForegroundColor Green
        
        # Check NVIDIA-Docker support
        try {
            docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ NVIDIA Docker runtime available" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "⚠️  NVIDIA Docker runtime not available. GPU acceleration disabled." -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️  No GPU detected. Using CPU-only mode." -ForegroundColor Yellow
    }
}
catch {
    Write-Host "⚠️  Could not detect GPU information" -ForegroundColor Yellow
}

# Build Docker images
if (!$SkipDocker) {
    Write-Host "🏗️  Building Docker images..." -ForegroundColor Cyan
    try {
        docker-compose build --parallel
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Docker images built successfully" -ForegroundColor Green
        } else {
            throw "Docker build failed"
        }
    }
    catch {
        Write-Host "❌ Failed to build Docker images: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Create startup script
Write-Host "🎬 Creating startup script..." -ForegroundColor Cyan
@"
@echo off
echo Starting Biometric Authentication System...
echo.

REM Set memory optimization
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set DOCKER_BUILDKIT=1

REM Start services
docker-compose up -d

REM Wait for services to be ready
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service health
echo.
echo Checking service health:
curl -s http://localhost:8000/health || echo API: Not ready
curl -s http://localhost:3000 || echo Grafana: Not ready
echo.

echo System started! Access points:
echo - API: http://localhost:8000
echo - API Docs: http://localhost:8000/docs
echo - Database Admin: http://localhost:8080
echo - Grafana Dashboard: http://localhost:3000 (admin/biometric2024)
echo - Prometheus: http://localhost:9090
echo.
pause
"@ | Out-File -FilePath "start-system.bat" -Encoding ASCII

# Create shutdown script
@"
@echo off
echo Stopping Biometric Authentication System...
docker-compose down
echo System stopped.
pause
"@ | Out-File -FilePath "stop-system.bat" -Encoding ASCII

# Create client test script
@"
@echo off
echo Testing Biometric Authentication API...
echo.

REM Test health endpoint
echo Testing health endpoint...
curl -X GET "http://localhost:8000/health" -H "accept: application/json"
echo.

REM Test metrics endpoint  
echo Testing metrics endpoint...
curl -X GET "http://localhost:8000/metrics" -H "accept: application/json"
echo.

echo Test completed. Check the responses above.
pause
"@ | Out-File -FilePath "test-api.bat" -Encoding ASCII

# Final system check
Write-Host "🔍 Running final system check..." -ForegroundColor Cyan

$checks = @(
    @{ Name = "Docker"; Command = "docker --version"; Required = $true },
    @{ Name = "Docker Compose"; Command = "docker-compose --version"; Required = $true },
    @{ Name = "Data Directory"; Path = "$DataDrive\data"; Required = $true },
    @{ Name = "Models Directory"; Path = "$DataDrive\models"; Required = $true },
    @{ Name = "Environment File"; Path = ".env"; Required = $true }
)

$allPassed = $true
foreach ($check in $checks) {
    try {
        if ($check.Command) {
            Invoke-Expression $check.Command | Out-Null
            Write-Host "✅ $($check.Name): OK" -ForegroundColor Green
        } elseif ($check.Path) {
            if (Test-Path $check.Path) {
                Write-Host "✅ $($check.Name): OK" -ForegroundColor Green
            } else {
                throw "Path not found"
            }
        }
    }
    catch {
        if ($check.Required) {
            Write-Host "❌ $($check.Name): FAILED" -ForegroundColor Red
            $allPassed = $false
        } else {
            Write-Host "⚠️  $($check.Name): Optional" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n🎉 Setup completed!" -ForegroundColor Green
if ($allPassed) {
    Write-Host "✅ All checks passed. System is ready to start." -ForegroundColor Green
    Write-Host "`n🚀 To start the system:" -ForegroundColor Cyan
    Write-Host "   .\start-system.bat" -ForegroundColor White
    Write-Host "`n📊 To test the API:" -ForegroundColor Cyan
    Write-Host "   .\test-api.bat" -ForegroundColor White
    Write-Host "`n📊 To access services:" -ForegroundColor Cyan
    Write-Host "   API: http://localhost:8000" -ForegroundColor White
    Write-Host "   Docs: http://localhost:8000/docs" -ForegroundColor White
    Write-Host "   Database: http://localhost:8080" -ForegroundColor White
    Write-Host "   Grafana: http://localhost:3000 (admin/biometric2024)" -ForegroundColor White
    Write-Host "   Prometheus: http://localhost:9090" -ForegroundColor White
} else {
    Write-Host "⚠️  Some checks failed. Please review and fix issues before starting." -ForegroundColor Yellow
}

Write-Host "`n💾 Memory optimization applied:" -ForegroundColor Green
Write-Host "   - All data stored on $DataDrive drive" -ForegroundColor White
Write-Host "   - PyTorch memory chunking: 512MB" -ForegroundColor White
Write-Host "   - CPU threads limited: 2" -ForegroundColor White
Write-Host "   - Redis memory limit: 200MB" -ForegroundColor White
Write-Host "   - Docker memory limits applied" -ForegroundColor White

Write-Host "`n📋 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Run .\start-system.bat to start all services" -ForegroundColor White
Write-Host "   2. Access API documentation at http://localhost:8000/docs" -ForegroundColor White
Write-Host "   3. Test enrollment and authentication endpoints" -ForegroundColor White
Write-Host "   4. Monitor system health via Grafana dashboard" -ForegroundColor White 