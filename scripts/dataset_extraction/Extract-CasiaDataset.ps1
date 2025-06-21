# ======================================================
# CASIA-WebFace Dataset Extraction and Preparation Script
# ======================================================

param (
    [string]$rec_file = "data\casia-webface\train.rec",
    [string]$idx_file = "data\casia-webface\train.idx",
    [string]$output_dir = "data\casia-extracted",
    [string]$partitioned_dir = "data\partitioned",
    [int]$max_images = $null,
    [int]$max_classes = 300,
    [int]$classes_per_split = 100,
    [switch]$skip_extraction,
    [switch]$skip_partitioning,
    [switch]$help
)

# Show help if requested
if ($help) {
    Write-Host "Usage: .\Extract-CasiaDataset.ps1 [-rec_file <path>] [-idx_file <path>] [-output_dir <path>] [-partitioned_dir <path>] [-max_images <num>] [-max_classes <num>] [-classes_per_split <num>] [-skip_extraction] [-skip_partitioning] [-help]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -rec_file           Path to the .rec file. Default: data\casia-webface\train.rec"
    Write-Host "  -idx_file           Path to the .idx file. Default: data\casia-webface\train.idx"
    Write-Host "  -output_dir         Directory to save extracted images. Default: data\casia-extracted"
    Write-Host "  -partitioned_dir    Directory to save partitioned dataset. Default: data\partitioned"
    Write-Host "  -max_images         Maximum number of images to extract (for testing)"
    Write-Host "  -max_classes        Maximum number of classes to extract (default: 300)"
    Write-Host "  -classes_per_split  Number of classes per split (default: 100)"
    Write-Host "  -skip_extraction    Skip extraction stage (if you already extracted images)"
    Write-Host "  -skip_partitioning  Skip partitioning stage"
    Write-Host "  -help               Show this help message"
    exit 0
}

Write-Host "CASIA-WebFace Dataset Extraction and Preparation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Target: $classes_per_split classes per split (Total: $($classes_per_split * 3) classes)" -ForegroundColor Cyan

# Check if rec_file exists
if (-not (Test-Path $rec_file) -and -not $skip_extraction) {
    Write-Host "Error: RecordIO file not found at $rec_file" -ForegroundColor Red
    Write-Host "Please place the train.rec file in the data\casia-webface directory or specify the correct path with -rec_file" -ForegroundColor Yellow
    exit 1
}

# Create necessary directories
if (-not (Test-Path (Split-Path -Path $output_dir -Parent))) {
    Write-Host "Creating parent directory for output..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path (Split-Path -Path $output_dir -Parent) -Force | Out-Null
}

if (-not (Test-Path $output_dir)) {
    Write-Host "Creating output directory: $output_dir..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $output_dir -Force | Out-Null
}

if (-not (Test-Path $partitioned_dir)) {
    Write-Host "Creating partitioned directory: $partitioned_dir..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $partitioned_dir -Force | Out-Null
}

# Check if dataset already exists in output directory
$folderCount = (Get-ChildItem -Path $output_dir -Directory -ErrorAction SilentlyContinue | Measure-Object).Count
if ($folderCount -gt 0 -and -not $skip_extraction) {
    Write-Host "Found existing data in $output_dir" -ForegroundColor Yellow
    Write-Host "NOTE: Running the extraction will CLEAN the output directory and REMOVE ALL existing data!" -ForegroundColor Red
    Write-Host "Would you like to use the existing data instead of extracting again? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    if ($response -eq "Y" -or $response -eq "y") {
        Write-Host "Using existing data in $output_dir" -ForegroundColor Green
        $skip_extraction = $true
    }
    else {
        Write-Host "Will extract data and REMOVE ALL existing content in $output_dir" -ForegroundColor Yellow
    }
}

# Check if PIL (Pillow) is installed
if (-not $skip_extraction) {
    try {
        python -c "from PIL import Image" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "PIL/Pillow is required for extracting images but not installed." -ForegroundColor Red
            Write-Host "Please install it with: pip install pillow" -ForegroundColor Yellow
            $installPillow = Read-Host "Install Pillow now? (Y/N)"
            if ($installPillow -eq "Y" -or $installPillow -eq "y") {
                python -m pip install pillow
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "Failed to install Pillow. Please install it manually." -ForegroundColor Red
                    exit 1
                }
            } else {
                exit 1
            }
        }
    }
    catch {
        Write-Host "Error checking PIL installation: $_" -ForegroundColor Red
        exit 1
    }
}

# Extract CASIA-WebFace dataset from RecordIO file
if (-not $skip_extraction) {
    # Check for the extraction script or create it
    $extractorScript = "extract_rec_simple.py"
    if (-not (Test-Path $extractorScript)) {
        Write-Host "Error: Extractor script not found at $extractorScript" -ForegroundColor Red
        Write-Host "Please ensure the repository contains the extraction script." -ForegroundColor Red
        exit 1
    }
    
    # Validate class count configuration
    $expected_total = $classes_per_split * 3
    if ($max_classes -ne $expected_total) {
        Write-Host "Warning: max_classes ($max_classes) doesn't match expected total ($expected_total = $classes_per_split * 3)" -ForegroundColor Yellow
        Write-Host "Adjusting max_classes to $expected_total to ensure $classes_per_split classes per split" -ForegroundColor Yellow
        $max_classes = $expected_total
    }
    
    # Run extraction
    Write-Host "Extracting CASIA-WebFace dataset from RecordIO files..." -ForegroundColor Green
    Write-Host "This may take some time depending on the dataset size." -ForegroundColor Yellow
    Write-Host "The output directory will be completely cleaned before extraction begins." -ForegroundColor Yellow
    Write-Host "Limiting to maximum of $max_classes classes ($classes_per_split per split)." -ForegroundColor Yellow
    
    $lstFile = $rec_file -replace "\.rec$", ".lst"
    if (Test-Path $lstFile) {
        Write-Host "Found LST file at $lstFile, will use for class information" -ForegroundColor Yellow
        $lstOption = "--lst_file `"$lstFile`""
    }
    else {
        $lstOption = ""
    }
    
    # Use the scanning method as it's more reliable
    $extractCmd = "python $extractorScript --rec_file `"$rec_file`" $lstOption --output_dir `"$output_dir`" --scan"
    
    if ($max_images) {
        $extractCmd += " --max_images $max_images"
    }
    
    if ($max_classes) {
        $extractCmd += " --max_classes $max_classes"
    }
    
    Write-Host "Running: $extractCmd" -ForegroundColor Yellow
    Invoke-Expression $extractCmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to extract dataset with exit code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
    
    # Verify extraction was successful
    $extractedFiles = Get-ChildItem -Path $output_dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object
    if ($extractedFiles.Count -eq 0) {
        Write-Host "Warning: No files were extracted to $output_dir" -ForegroundColor Red
        Write-Host "The RecordIO file format might not be compatible with our extractor" -ForegroundColor Red
        Write-Host "Please try using a different extraction method or check if the RecordIO file is valid" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Dataset extraction completed successfully! Extracted $($extractedFiles.Count) files." -ForegroundColor Green
}
else {
    Write-Host "Skipping extraction stage as requested." -ForegroundColor Yellow
    
    # Verify extracted data exists
    $extractedFiles = Get-ChildItem -Path $output_dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object
    if ($extractedFiles.Count -eq 0) {
        Write-Host "Warning: No files found in $output_dir. Make sure data is already extracted before skipping extraction." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Found $($extractedFiles.Count) files in the extraction directory." -ForegroundColor Green
}

# Partition dataset for federated learning
if (-not $skip_partitioning) {
    # Create partitioned directories
    $serverDir = Join-Path $partitioned_dir "server"
    $client1Dir = Join-Path $partitioned_dir "client1"
    $client2Dir = Join-Path $partitioned_dir "client2"
    
    # Clean partition directories
    if (Test-Path $partitioned_dir) {
        Write-Host "Cleaning partition directories..." -ForegroundColor Yellow
        Remove-Item -Path $partitioned_dir -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    New-Item -ItemType Directory -Path $serverDir -Force | Out-Null
    New-Item -ItemType Directory -Path $client1Dir -Force | Out-Null
    New-Item -ItemType Directory -Path $client2Dir -Force | Out-Null
    
    Write-Host "Partitioning dataset into server and client sets ($classes_per_split classes each)..." -ForegroundColor Green
    
    # Get all class directories
    $classDirectories = Get-ChildItem -Path $output_dir -Directory | Where-Object { 
        # Only include directories with at least 5 images to ensure enough for training
        $imageCount = (Get-ChildItem -Path $_.FullName -File).Count
        $imageCount -ge 5
    }
    
    $totalClasses = $classDirectories.Count
    Write-Host "Found $totalClasses valid classes (with at least 5 images each)" -ForegroundColor Yellow
    
    if ($totalClasses -eq 0) {
        Write-Host "Error: No valid classes found. Make sure extraction was successful." -ForegroundColor Red
        exit 1
    }
    
    # Check if we have enough classes for the desired split
    $requiredClasses = $classes_per_split * 3
    if ($totalClasses -lt $requiredClasses) {
        Write-Host "Warning: Only found $totalClasses classes, but need $requiredClasses for $classes_per_split per split" -ForegroundColor Yellow
        $classes_per_split = [math]::Floor($totalClasses / 3)
        Write-Host "Adjusting to $classes_per_split classes per split" -ForegroundColor Yellow
        
        if ($classes_per_split -eq 0) {
            Write-Host "Error: Not enough classes to create meaningful splits" -ForegroundColor Red
            exit 1
        }
    }
    
    # Use exactly the specified number of classes per split
    $serverClassCount = $classes_per_split
    $client1ClassCount = $classes_per_split
    $client2ClassCount = $classes_per_split
    
    Write-Host "Partitioning into: Server=$serverClassCount classes, Client1=$client1ClassCount classes, Client2=$client2ClassCount classes" -ForegroundColor Yellow
    
    # Shuffle classes randomly and take exactly the required number for each split
    $shuffledClasses = $classDirectories | Sort-Object { Get-Random }
    $serverClasses = $shuffledClasses[0..($serverClassCount-1)]
    $client1Classes = $shuffledClasses[$serverClassCount..($serverClassCount+$client1ClassCount-1)]
    $client2Classes = $shuffledClasses[($serverClassCount+$client1ClassCount)..($serverClassCount+$client1ClassCount+$client2ClassCount-1)]
    
    # Copy server classes
    Write-Host "Copying server classes..." -ForegroundColor Yellow
    foreach ($class in $serverClasses) {
        $targetDir = Join-Path $serverDir $class.Name
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        
        # Copy all files in this class
        Copy-Item -Path "$($class.FullName)\*" -Destination $targetDir -Recurse
    }
    
    # Copy client1 classes
    Write-Host "Copying client1 classes..." -ForegroundColor Yellow
    foreach ($class in $client1Classes) {
        $targetDir = Join-Path $client1Dir $class.Name
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        
        # Copy all files in this class
        Copy-Item -Path "$($class.FullName)\*" -Destination $targetDir -Recurse
    }
    
    # Copy client2 classes
    Write-Host "Copying client2 classes..." -ForegroundColor Yellow
    foreach ($class in $client2Classes) {
        $targetDir = Join-Path $client2Dir $class.Name
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        
        # Copy all files in this class
        Copy-Item -Path "$($class.FullName)\*" -Destination $targetDir -Recurse
    }
    
    # Count images in each partition
    $serverImages = (Get-ChildItem -Path $serverDir -Recurse -File).Count
    $client1Images = (Get-ChildItem -Path $client1Dir -Recurse -File).Count
    $client2Images = (Get-ChildItem -Path $client2Dir -Recurse -File).Count
    $totalImages = $serverImages + $client1Images + $client2Images
    
    Write-Host "Partitioning complete!" -ForegroundColor Green
    Write-Host "Server: $serverClassCount classes, $serverImages images" -ForegroundColor Green
    Write-Host "Client1: $client1ClassCount classes, $client1Images images" -ForegroundColor Green
    Write-Host "Client2: $client2ClassCount classes, $client2Images images" -ForegroundColor Green
    Write-Host "Total used: $($serverClassCount + $client1ClassCount + $client2ClassCount) classes, $totalImages images" -ForegroundColor Green
    if ($totalClasses -gt ($serverClassCount + $client1ClassCount + $client2ClassCount)) {
        $unusedClasses = $totalClasses - ($serverClassCount + $client1ClassCount + $client2ClassCount)
        Write-Host "Note: $unusedClasses classes were extracted but not used in partitioning" -ForegroundColor Yellow
    }
}
else {
    Write-Host "Skipping partitioning stage as requested." -ForegroundColor Yellow
}

# Display success message
Write-Host ""
Write-Host "CASIA-WebFace dataset has been processed successfully!" -ForegroundColor Green
Write-Host "  - Extracted data: $output_dir" -ForegroundColor Green
Write-Host "  - Partitioned data: $partitioned_dir" -ForegroundColor Green
Write-Host ""
Write-Host "IMPROVED EFFICIENCY:" -ForegroundColor Cyan
Write-Host "  - Now extracts exactly $($classes_per_split * 3) classes total ($classes_per_split per split)" -ForegroundColor Cyan
Write-Host "  - Stops extraction early once target classes are reached" -ForegroundColor Cyan
Write-Host "  - Each split gets exactly $classes_per_split classes" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run the training script with:" -ForegroundColor Cyan
Write-Host "    .\train_model.ps1 -mode server -data_dir '.\$partitioned_dir\server' -model_dir '.\models'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 