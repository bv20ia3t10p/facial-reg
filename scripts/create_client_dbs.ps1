# Create client databases PowerShell script
$ErrorActionPreference = "Stop"

# Get the script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath

# Ensure we're in the project root
Set-Location $projectRoot

# Create the database directory if it doesn't exist
$databaseDir = Join-Path $projectRoot "database"
if (-not (Test-Path $databaseDir)) {
    New-Item -ItemType Directory -Path $databaseDir
}

# Run the Python script to create databases
Write-Host "Creating client databases..."
python scripts/create_client_dbs.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Client databases created successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to create client databases" -ForegroundColor Red
    exit 1
}

# List created databases
Write-Host ""
Write-Host "Created databases:"
Get-ChildItem -Path $databaseDir -Filter "*.db" | ForEach-Object {
    Write-Host "  - $($_.Name)" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Done!"