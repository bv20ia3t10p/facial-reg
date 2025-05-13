param (
    [ValidateSet("server", "client")]
    [string]$mode = "server",
    [string]$client_id = "",
    [string]$data_dir = ".\data\processed",
    [string]$model_dir = ".\models",
    [int]$batch_size = 32,
    [int]$epochs = 20,
    [int]$img_size = 112,
    [double]$lr = 0.001,
    [switch]$debug,
    [switch]$help,
    [switch]$install_deps,
    [switch]$skip_deps_check
)

# ======================================================
# Facial Recognition Model Training Script for Windows
# ======================================================

Write-Host "Facial Recognition Model Training - Windows PowerShell Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Show help if requested
if ($help) {
    Write-Host "Usage: .\train_model.ps1 [-mode <server|client>] [-client_id <id>] [-data_dir <data_path>] [-model_dir <model_path>] [-batch_size <size>] [-epochs <num>] [-img_size <size>] [-lr <rate>] [-debug] [-help] [-install_deps] [-skip_deps_check]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -mode            Training mode (server or client). Default: server"
    Write-Host "  -client_id       Client ID when in client mode"
    Write-Host "  -data_dir        Path to training data directory. Default: .\data\processed"
    Write-Host "  -model_dir       Directory to save trained models. Default: .\models"
    Write-Host "  -batch_size      Batch size for training. Default: 32"
    Write-Host "  -epochs          Number of training epochs. Default: 20"
    Write-Host "  -img_size        Image size for training. Default: 112"
    Write-Host "  -lr              Learning rate. Default: 0.001"
    Write-Host "  -debug           Enable debug mode"
    Write-Host "  -install_deps    Install required Python dependencies"
    Write-Host "  -skip_deps_check Skip dependency check (use after installation)"
    Write-Host "  -help            Show this help message"
    exit 0
}

# Validate client_id if in client mode
if ($mode -eq "client" -and [string]::IsNullOrEmpty($client_id)) {
    Write-Host "Error: client_id is required when in client mode" -ForegroundColor Red
    exit 1
}

# Create data directory if it doesn't exist
if (-not (Test-Path $data_dir)) {
    Write-Host "Data directory not found: $data_dir" -ForegroundColor Yellow
    Write-Host "Creating data directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $data_dir -Force | Out-Null
    Write-Host "NOTE: You will need to add training data to $data_dir" -ForegroundColor Yellow
}

# Create model directory if it doesn't exist
if (-not (Test-Path $model_dir)) {
    Write-Host "Creating model save directory: $model_dir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $model_dir -Force | Out-Null
}

# Display training configuration
Write-Host "Training Configuration:" -ForegroundColor Green
Write-Host "  Mode:             $mode"
if ($mode -eq "client") {
    Write-Host "  Client ID:        $client_id"
}
Write-Host "  Data Directory:   $data_dir"
Write-Host "  Model Directory:  $model_dir"
Write-Host "  Batch Size:       $batch_size"
Write-Host "  Epochs:           $epochs"
Write-Host "  Image Size:       $img_size"
Write-Host "  Learning Rate:    $lr"
if ($debug) {
    Write-Host "  Debug Mode:       Enabled"
}
Write-Host ""

# Validate Python installation
try {
    Write-Host "Checking Python installation..." -ForegroundColor Yellow
    python --version
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
        exit 1
    }
    
    # Check for training script
    $trainingScript = ".\src\scripts\training\train_model.py"
    if (-not (Test-Path $trainingScript)) {
        Write-Host "Error: Training script not found at $trainingScript" -ForegroundColor Red
        Write-Host "Please ensure the repository structure is correct." -ForegroundColor Red
        exit 1
    }
    
    # Direct installation of required packages if specified
    if ($install_deps) {
        Write-Host "Installing required Python packages..." -ForegroundColor Yellow
        python -m pip install tensorflow numpy pandas matplotlib scikit-learn
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Failed to install packages" -ForegroundColor Red
            exit 1
        }
        Write-Host "Packages installed successfully." -ForegroundColor Green
        $skip_deps_check = $true
    }
    
    # Check TensorFlow only if not skipping dependency check
    if (-not $skip_deps_check) {
        Write-Host "Checking if TensorFlow is installed..." -ForegroundColor Yellow
        $tfCheck = python -c "try: 
    import tensorflow as tf
    print(f'TensorFlow {tf.__version__} is installed')
    exit(0)
except ImportError as e: 
    print(f'TensorFlow import error: {e}')
    exit(1)
except Exception as e:
    print(f'Unexpected error: {e}')
    exit(2)" 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "TensorFlow is not installed or has issues. Please install it with:" -ForegroundColor Red
            Write-Host "   python -m pip install tensorflow" -ForegroundColor Yellow
            Write-Host "Or run this script with the -install_deps flag:" -ForegroundColor Yellow
            Write-Host "   .\train_model.ps1 -install_deps" -ForegroundColor Yellow
            exit 1
        } else {
            Write-Host $tfCheck -ForegroundColor Green
        }
    }
    
    Write-Host "Python environment is ready for training." -ForegroundColor Green
}
catch {
    Write-Host "Error checking Python installation: $_" -ForegroundColor Red
    exit 1
}

# Train the model using Python
Write-Host "Starting model training..." -ForegroundColor Green

try {
    # Build command with all parameters
    $pythonCmd = "python .\src\scripts\training\train_model.py --mode $mode"
    
    # Add client ID if in client mode
    if ($mode -eq "client" -and -not [string]::IsNullOrEmpty($client_id)) {
        $pythonCmd += " --client-id $client_id"
    }
    
    # Add common parameters
    $pythonCmd += " --data-dir `"$data_dir`" --model-dir `"$model_dir`" --epochs $epochs --batch-size $batch_size --img-size $img_size --lr $lr"
    
    # Add debug flag if specified
    if ($debug) {
        $pythonCmd += " --debug"
    }
    
    # Execute the command
    Write-Host "Executing: $pythonCmd" -ForegroundColor Yellow
    Invoke-Expression $pythonCmd
    
    # Check if training was successful
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Model training completed successfully!" -ForegroundColor Green
        Write-Host "Model saved to: $model_dir" -ForegroundColor Green
    }
    else {
        Write-Host "Error: Model training failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
catch {
    Write-Host "Error: An exception occurred during training:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} 