# Test the fixed mapping by authenticating with an image from identity 101
$url = "http://localhost:8000/api/auth/authenticate"

# Create a temporary image file path - we'll use an existing one from the data
$imagePath = "E:\Repos\facial-reg\data\partitioned\client1\101\client1_101_001.jpg"

# Check if the image exists locally
if (Test-Path $imagePath) {
    Write-Host "Testing with local image: $imagePath"
    
    # Use curl.exe to make the request with form data
    $result = curl.exe -X POST $url `
        -H "Content-Type: multipart/form-data" `
        -F "image=@$imagePath" `
        -F "device_info=test_device"
    
    Write-Host "API Response:"
    $result | ConvertFrom-Json | ConvertTo-Json -Depth 10
} else {
    Write-Host "Local image not found. Let's check what we have:"
    Get-ChildItem "E:\Repos\facial-reg\data\partitioned\client1\101\" -Filter "*.jpg" | Select-Object -First 3
} 