<#
.SYNOPSIS
    Mekasa: instalar, calibrar y usar el tablero desde PowerShell.

.EXAMPLE
    .\scripts\mekasa.ps1 instalar      # una sola vez
    .\scripts\mekasa.ps1 camara        # ver qué cámara responde
    .\scripts\mekasa.ps1 calibrar      # ~2 min con la persona frente a la cámara
    .\scripts\mekasa.ps1 regiones      # ~2 min, mide el acierto con consignas
    .\scripts\mekasa.ps1 tablero       # tablero SÍ / NO / AGUA / AYUDA
    .\scripts\mekasa.ps1 tablero -Ojo LEFT -Dwell 1.2
    .\scripts\mekasa.ps1 tablero -Disposicion binary-horizontal
    .\scripts\mekasa.ps1 simular       # prueba de seguridad sin cámara
    .\scripts\mekasa.ps1 tests         # pruebas automáticas
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("instalar", "camara", "calibrar", "regiones", "tablero", "simular", "tests", "ayuda")]
    [string]$Accion = "ayuda",

    # Calibración: "rapido" (~2 min) o "completo" (~4,5 min, más tiempo por punto).
    [ValidateSet("rapido", "completo")]
    [string]$Perfil = "rapido",

    # Umbral personal de "ojo cerrado" (EAR). 0.09 corresponde a la sesión del 2026-11-16,
    # donde el párpado bajaba al mirar hacia abajo. El valor poblacional es 0.15.
    [double]$UmbralOjoCerrado = 0.09,

    [ValidateSet("BOTH", "LEFT", "RIGHT")]
    [string]$Ojo = "BOTH",

    [double]$Dwell = 0,

    [ValidateSet("", "quadrant", "binary-horizontal", "binary-vertical")]
    [string]$Disposicion = "",

    [switch]$Confirmar,
    [switch]$SinAjuste,
    [switch]$Ventana
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

$uv = Find-Uv

function Invoke-Py {
    param([string]$Module, [string[]]$Arguments = @())
    & $uv run --locked python -m $Module @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falló $Module (código $LASTEXITCODE)" }
}

function Write-PersonalConfig {
    $target = "work/calibration/experiment-001-personal.toml"
    New-Item -ItemType Directory -Force -Path "work/calibration" | Out-Null
    $text = Get-Content "config/experiment-001.toml" -Raw -Encoding utf8
    $value = $UmbralOjoCerrado.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $text = $text -replace "(?m)^closed_ear_threshold = .*$", "closed_ear_threshold = $value"
    [System.IO.File]::WriteAllText((Join-Path $root $target), $text, [System.Text.UTF8Encoding]::new($false))
    return $target
}

switch ($Accion) {
    "instalar" {
        & $uv python install 3.12
        & $uv sync --locked
        powershell -ExecutionPolicy Bypass -File .\scripts\fetch-face-landmarker.ps1
        Write-Host "Listo. Siguiente paso: .\scripts\mekasa.ps1 camara" -ForegroundColor Green
    }
    "camara" {
        Invoke-Py "mekasa_gaze.tracking.probe"
    }
    "calibrar" {
        $guided = if ($Perfil -eq "rapido") { "config/experiment-001-quick.toml" } else { "config/experiment-001.toml" }
        Write-Host "1/2 Validación guiada ($Perfil). Mire cada punto sin mover la cabeza; ESC cancela." -ForegroundColor Cyan
        Invoke-Py "mekasa_gaze.runners.guided_experiment" @("--config", $guided)
        Write-Host "2/2 Entrenando calibraciones (ambos ojos, izquierdo, derecho)..." -ForegroundColor Cyan
        $value = $UmbralOjoCerrado.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        foreach ($mode in "BOTH", "LEFT", "RIGHT") {
            $out = "work/calibration/personal-gaze-model-$($mode.ToLower()).json"
            & $uv run --locked python -m mekasa_gaze.tracking.calibration --eye-mode $mode `
                --reclassify-ear-threshold $value --output $out
            if ($LASTEXITCODE -ne 0) { Write-Warning "No se pudo calibrar el modo $mode (ver mensaje arriba)." }
        }
        Write-Host "Listo. Siguiente paso: .\scripts\mekasa.ps1 regiones  o  .\scripts\mekasa.ps1 tablero" -ForegroundColor Green
    }
    "regiones" {
        $personal = Write-PersonalConfig
        Invoke-Py "mekasa_gaze.runners.controlled_region_validation" @(
            "--calibration", "work/calibration/personal-gaze-model-$($Ojo.ToLower()).json",
            "--guided-config", $personal
        )
    }
    "tablero" {
        $arguments = @("--eye-mode", $Ojo)
        if ($Dwell -gt 0) { $arguments += @("--dwell", $Dwell.ToString([System.Globalization.CultureInfo]::InvariantCulture)) }
        if ($Disposicion) { $arguments += @("--layout", $Disposicion) }
        if ($Confirmar) { $arguments += "--confirm" }
        if ($SinAjuste) { $arguments += "--no-align" }
        if ($Ventana) { $arguments += "--no-fullscreen" }
        Write-Host "Teclas: 1-4 elegir · ESPACIO pausa · A reajustar · M cambiar ojo · G ver mirada · ESC salir" -ForegroundColor Cyan
        Invoke-Py "mekasa_gaze.runners.communication_board" $arguments
    }
    "simular" {
        Invoke-Py "mekasa_gaze.runners.communication_board" @("--simulate")
    }
    "tests" {
        & $uv run --locked python -m pytest -q
    }
    default {
        Get-Help $MyInvocation.MyCommand.Path -Examples
    }
}
