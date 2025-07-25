# This script sends a test request to the /generate-speech endpoint using Windows PowerShell 5.1 compatible method.
# It uploads the GeminiReport.pdf as the context document.

$pdfPath = "C:\Users\hdthi.THINH-VOSTRO\Projects\aws-projects\scripts\gemini_v2_5_report.pdf"

# Check if the PDF file exists
if (-not (Test-Path $pdfPath)) {
    Write-Host "Error: PDF file not found at '$pdfPath'"
    Write-Host "Please ensure the file exists or run the download command from the notebook."
    exit 1
}

Write-Host "Sending request to Flask server..."
Write-Host "Uploading file: $pdfPath"
Write-Host "File size: $((Get-Item $pdfPath).Length) bytes"

# Create multipart/form-data manually for Windows PowerShell 5.1 compatibility
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

# Read the PDF file as bytes
$fileBytes = [System.IO.File]::ReadAllBytes($pdfPath)
$fileName = [System.IO.Path]::GetFileName($pdfPath)
$fileEnc = [System.Text.Encoding]::GetEncoding('ISO-8859-1').GetString($fileBytes)

# Build the multipart body
$bodyLines = @(
    "--$boundary",
    "Content-Disposition: form-data; name=`"text`"$LF",
    "What is the newest version of Gemini, list all of different flavor?",
    "--$boundary",
    "Content-Disposition: form-data; name=`"language`"$LF",
    "en-US",
    "--$boundary",
    "Content-Disposition: form-data; name=`"documents`"; filename=`"$fileName`"",
    "Content-Type: application/pdf$LF",
    $fileEnc,
    "--$boundary--$LF"
) -join $LF

# Send the request
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:5001/generate-speech" -Method POST -Body $bodyLines -ContentType "multipart/form-data; boundary=$boundary"
    
    Write-Host "Response Status: $($response.StatusCode)"
    Write-Host "Content-Type: $($response.Headers['Content-Type'])"
    Write-Host "Response Length: $($response.Content.Length) bytes"
    
    # Save the audio response to a file
    $outputPath = "temp/api_response.wav"
    [System.IO.File]::WriteAllBytes($outputPath, $response.Content)
    Write-Host "Audio saved to: $outputPath"
    
} catch {
    Write-Host "Error occurred: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        Write-Host "HTTP Status: $($_.Exception.Response.StatusCode)"
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody"
    }
} 