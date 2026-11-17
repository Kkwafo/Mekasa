<#
.SYNOPSIS
    Construye Mekasa para Windows: carpeta portable, .zip y (si hay Inno Setup) instalador.

.DESCRIPTION
    Todo lo necesario para usar Mekasa sin internet queda dentro: Python, OpenCV, MediaPipe,
    el modelo de rostro y la configuración. Resultado en dist/:
      dist/Mekasa/                                   carpeta lista para usar (Mekasa.exe)
      dist/Mekasa-<versión>-windows-x64-portable.zip la misma carpeta comprimida
      dist/Mekasa-<versión>-setup.exe                instalador (requiere Inno Setup 6)

.EXAMPLE
    .\scripts\build-release.ps1
    .\scripts\build-release.ps1 -SinInstalador
#>
[CmdletBinding()]
param(
    [switch]$SinInstalador,
    # Omite la simulación del .exe (p. ej. si Smart App Control bloquea ejecutables sin firma).
    [switch]$SinPrueba,
    [string]$PyInstallerVersion = "6.22.3"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Find-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $winget = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($winget) { return $winget.FullName }
    throw "No se encontró 'uv'. Instálelo con: winget install --id astral-sh.uv -e"
}

function Find-Iscc {
    $command = Get-Command iscc -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($candidate in @(
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    return $null
}

$uv = Find-Uv
$version = (Select-String -Path pyproject.toml -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
Write-Host "Mekasa $version" -ForegroundColor Cyan

& $uv sync --locked
if ($LASTEXITCODE -ne 0) { throw "uv sync falló" }

powershell -ExecutionPolicy Bypass -File .\scripts\fetch-face-landmarker.ps1
if ($LASTEXITCODE -ne 0) { throw "No se pudo obtener el modelo de rostro" }

$commit = $null
$dirty = $false
try {
    $commit = (git rev-parse HEAD).Trim()
    $dirty = [bool](git status --porcelain)
}
catch { }
New-Item -ItemType Directory -Force -Path build | Out-Null
@{
    version    = $version
    commit     = $commit
    dirty      = $dirty
    built_utc  = (Get-Date).ToUniversalTime().ToString("o")
} | ConvertTo-Json | Set-Content -Encoding utf8 build\build-info.json

if (Test-Path dist\Mekasa) { Remove-Item -Recurse -Force dist\Mekasa }
& $uv run --locked --with "pyinstaller==$PyInstallerVersion" pyinstaller packaging\mekasa.spec `
    --noconfirm --clean --distpath dist --workpath build\pyinstaller
if ($LASTEXITCODE -ne 0) { throw "PyInstaller falló" }

# Prueba de humo: la simulación sin cámara recorre la cadena real y falla si hubo una
# selección insegura. Usa una carpeta de datos temporal para no tocar la del usuario.
if ($SinPrueba) {
    Write-Warning "Se omitió la simulación del ejecutable."
}
else {
    $smokeHome = Join-Path $env:TEMP "mekasa-smoke-$PID"
    $env:MEKASA_HOME = $smokeHome
    try {
        & dist\Mekasa\Mekasa-consola.exe simular | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "La simulación del ejecutable falló (código $LASTEXITCODE)" }
    }
    finally {
        Remove-Item Env:\MEKASA_HOME
        if (Test-Path $smokeHome) { Remove-Item -Recurse -Force $smokeHome }
    }
    Write-Host "Simulación del ejecutable: OK" -ForegroundColor Green
}

$zip = "dist\Mekasa-$version-windows-x64-portable.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path dist\Mekasa -DestinationPath $zip
Write-Host "Portable: $zip" -ForegroundColor Green

if (-not $SinInstalador) {
    $iscc = Find-Iscc
    if ($iscc) {
        & $iscc "/DMyAppVersion=$version" "/O$root\dist" packaging\mekasa.iss
        if ($LASTEXITCODE -ne 0) { throw "Inno Setup falló" }
        Write-Host "Instalador: dist\Mekasa-$version-setup.exe" -ForegroundColor Green
    }
    else {
        Write-Warning "Inno Setup 6 no está instalado: se omite el instalador (winget install JRSoftware.InnoSetup)."
    }
}
