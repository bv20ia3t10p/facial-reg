# Test Federated Learning System
# Verifies FL + HE + DP functionality

param(
    [switch]$TriggerRound,
    [switch]$Verbose
)

Write-Host "=== Testing Federated Learning System ===" -ForegroundColor Green

# Function to test API endpoint
function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Name,
        [string]$Method = "GET",
        [hashtable]$Body = $null
    )
    
    try {
        if ($Body) {
            $response = Invoke-RestMethod -Uri $Url -Method $Method -Body ($Body | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 10
        } else {
            $response = Invoke-RestMethod -Uri $Url -Method $Method -TimeoutSec 10
        }
        
        Write-Host "✓ $Name" -ForegroundColor Green
        
        if ($Verbose) {
            Write-Host "  Response: $($response | ConvertTo-Json -Depth 2)" -ForegroundColor Gray
        }
        
        return $response
    } catch {
        Write-Host "✗ $Name - $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Test basic services
Write-Host "`n--- Testing Basic Services ---" -ForegroundColor Cyan

$apiHealth = Test-Endpoint -Url "http://localhost:8000/health" -Name "Biometric API Health"
$coordinatorHealth = Test-Endpoint -Url "http://localhost:8001/health" -Name "Federated Coordinator Health"

# Test federated integration
Write-Host "`n--- Testing Federated Integration ---" -ForegroundColor Cyan

$federatedStatus = Test-Endpoint -Url "http://localhost:8000/federated/status" -Name "Federated Status"

if ($federatedStatus) {
    Write-Host "  - Coordinator Healthy: $($federatedStatus.coordinator_healthy)" -ForegroundColor Gray
    Write-Host "  - Active Clients: $($federatedStatus.active_clients)" -ForegroundColor Gray
    Write-Host "  - HE Enabled: $($federatedStatus.he_enabled)" -ForegroundColor Gray
    Write-Host "  - Registered Clients: $($federatedStatus.registered_clients)" -ForegroundColor Gray
}

# Test client registration
Write-Host "`n--- Testing Client Registration ---" -ForegroundColor Cyan

$activeClients = Test-Endpoint -Url "http://localhost:8001/clients/active" -Name "Active Clients"

if ($activeClients) {
    Write-Host "  - Total Active Clients: $($activeClients.count)" -ForegroundColor Gray
    
    foreach ($client in $activeClients.active_clients) {
        Write-Host "  - Client: $($client.client_id) (Type: $($client.client_type), Privacy: $($client.privacy_spent))" -ForegroundColor Gray
    }
}

# Test model information
Write-Host "`n--- Testing Model Information ---" -ForegroundColor Cyan

$globalModel = Test-Endpoint -Url "http://localhost:8001/models/global" -Name "Global Model Info"

if ($globalModel) {
    Write-Host "  - Model Available: $($globalModel.model_available)" -ForegroundColor Gray
    Write-Host "  - Last Updated: $($globalModel.last_updated)" -ForegroundColor Gray
    Write-Host "  - Round: $($globalModel.round)" -ForegroundColor Gray
    Write-Host "  - Participants: $($globalModel.participants)" -ForegroundColor Gray
}

$modelHistory = Test-Endpoint -Url "http://localhost:8000/federated/model-history" -Name "Model History"

if ($modelHistory -and $modelHistory.model_history) {
    Write-Host "  - Total Model Versions: $($modelHistory.model_history.Count)" -ForegroundColor Gray
}

# Test current round
Write-Host "`n--- Testing Current Round ---" -ForegroundColor Cyan

$currentRound = Test-Endpoint -Url "http://localhost:8001/rounds/current" -Name "Current Round"

if ($currentRound -and $currentRound.round_id) {
    Write-Host "  - Round ID: $($currentRound.round_id)" -ForegroundColor Gray
    Write-Host "  - Started At: $($currentRound.started_at)" -ForegroundColor Gray
    Write-Host "  - Participants Submitted: $($currentRound.participants_submitted)" -ForegroundColor Gray
} else {
    Write-Host "  - No active round" -ForegroundColor Gray
}

# Trigger federated round if requested
if ($TriggerRound) {
    Write-Host "`n--- Triggering Federated Round ---" -ForegroundColor Cyan
    
    $roundResult = Test-Endpoint -Url "http://localhost:8000/federated/trigger-round" -Name "Trigger Federated Round" -Method "POST"
    
    if ($roundResult) {
        Write-Host "  - Round triggered successfully!" -ForegroundColor Green
        
        # Wait and check round status
        Write-Host "  - Waiting 10 seconds for round to start..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        
        $newRound = Test-Endpoint -Url "http://localhost:8001/rounds/current" -Name "New Round Status"
        
        if ($newRound -and $newRound.round_id) {
            Write-Host "  - New Round ID: $($newRound.round_id)" -ForegroundColor Green
            Write-Host "  - Participants: $($newRound.participants_submitted)" -ForegroundColor Gray
        }
    }
}

# Test privacy status for clients
Write-Host "`n--- Testing Privacy Status ---" -ForegroundColor Cyan

$client1Privacy = Test-Endpoint -Url "http://localhost:8001/privacy/status/client1" -Name "Client1 Privacy Status"
$client2Privacy = Test-Endpoint -Url "http://localhost:8001/privacy/status/client2" -Name "Client2 Privacy Status"

if ($client1Privacy) {
    Write-Host "  - Client1 Privacy Spent: $($client1Privacy.privacy_spent)/$($client1Privacy.privacy_budget)" -ForegroundColor Gray
    Write-Host "  - Client1 Can Participate: $($client1Privacy.can_participate)" -ForegroundColor Gray
}

if ($client2Privacy) {
    Write-Host "  - Client2 Privacy Spent: $($client2Privacy.privacy_spent)/$($client2Privacy.privacy_budget)" -ForegroundColor Gray
    Write-Host "  - Client2 Can Participate: $($client2Privacy.can_participate)" -ForegroundColor Gray
}

# Test monitoring endpoints
Write-Host "`n--- Testing Monitoring ---" -ForegroundColor Cyan

$apiMetrics = Test-Endpoint -Url "http://localhost:8000/metrics" -Name "API Metrics"

if ($apiMetrics) {
    Write-Host "  - Total Users: $($apiMetrics.database.total_users)" -ForegroundColor Gray
    Write-Host "  - Recent Attempts: $($apiMetrics.database.recent_attempts)" -ForegroundColor Gray
    Write-Host "  - System Memory Used: $([math]::Round($apiMetrics.memory.system_used * 100, 1))%" -ForegroundColor Gray
}

# Test database access
Write-Host "`n--- Testing Database Access ---" -ForegroundColor Cyan

try {
    $sqliteResponse = Invoke-WebRequest -Uri "http://localhost:8080" -TimeoutSec 5
    Write-Host "✓ SQLite Web Admin accessible" -ForegroundColor Green
} catch {
    Write-Host "✗ SQLite Web Admin not accessible" -ForegroundColor Red
}

# Test monitoring services
try {
    $prometheusResponse = Invoke-WebRequest -Uri "http://localhost:9090" -TimeoutSec 5
    Write-Host "✓ Prometheus accessible" -ForegroundColor Green
} catch {
    Write-Host "✗ Prometheus not accessible" -ForegroundColor Red
}

try {
    $grafanaResponse = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5
    Write-Host "✓ Grafana accessible" -ForegroundColor Green
} catch {
    Write-Host "✗ Grafana not accessible" -ForegroundColor Red
}

# Summary
Write-Host "`n=== Test Summary ===" -ForegroundColor Green

$services = @(
    @{Name="Biometric API"; Status=$apiHealth -ne $null},
    @{Name="Federated Coordinator"; Status=$coordinatorHealth -ne $null},
    @{Name="Federated Integration"; Status=$federatedStatus -ne $null},
    @{Name="Client Registration"; Status=$activeClients -ne $null}
)

$passedTests = ($services | Where-Object {$_.Status}).Count
$totalTests = $services.Count

Write-Host "Passed: $passedTests/$totalTests tests" -ForegroundColor $(if ($passedTests -eq $totalTests) {"Green"} else {"Yellow"})

foreach ($service in $services) {
    $status = if ($service.Status) {"✓"} else {"✗"}
    $color = if ($service.Status) {"Green"} else {"Red"}
    Write-Host "$status $($service.Name)" -ForegroundColor $color
}

if ($passedTests -eq $totalTests) {
    Write-Host "`nFederated Learning System is working correctly! 🎉" -ForegroundColor Green
} else {
    Write-Host "`nSome components are not working. Check the logs for details." -ForegroundColor Yellow
    Write-Host "Use: docker-compose logs -f [service-name]" -ForegroundColor Gray
}

Write-Host "`nUseful Commands:" -ForegroundColor Cyan
Write-Host "- View logs: docker-compose logs -f federated-coordinator" -ForegroundColor Gray
Write-Host "- Trigger round: .\scripts\test-federated-system.ps1 -TriggerRound" -ForegroundColor Gray
Write-Host "- Verbose output: .\scripts\test-federated-system.ps1 -Verbose" -ForegroundColor Gray
Write-Host "- Stop system: docker-compose down" -ForegroundColor Gray 