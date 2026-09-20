[CmdletBinding()]
param(
    [string]$Destination = "models/gaze/mediapipe-face-landmarker/face_landmarker.task"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$source = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
$expectedSha256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
$expectedBytes = 3758596
$destinationPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Destination))
$destinationDirectory = Split-Path -Parent $destinationPath

New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null

if (Test-Path -LiteralPath $destinationPath) {
    $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destinationPath).Hash.ToLowerInvariant()
    if ($existingHash -eq $expectedSha256) {
        Write-Output "Verified existing Face Landmarker model: $destinationPath"
        exit 0
    }
    throw "Existing model checksum mismatch at $destinationPath. Remove it explicitly before retrying."
}

$temporaryPath = "$destinationPath.download"
try {
    Invoke-WebRequest -Uri $source -OutFile $temporaryPath
    $download = Get-Item -LiteralPath $temporaryPath
    if ($download.Length -ne $expectedBytes) {
        throw "Model size mismatch: expected $expectedBytes bytes, received $($download.Length)."
    }
    $actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $temporaryPath).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        throw "Model checksum mismatch: expected $expectedSha256, received $actualSha256."
    }
    Move-Item -LiteralPath $temporaryPath -Destination $destinationPath
    Write-Output "Downloaded and verified Face Landmarker model: $destinationPath"
}
finally {
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }
}

