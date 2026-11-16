<#
.SYNOPSIS
    Mekasa: instalar, probar y usar el tablero desde PowerShell.

.DESCRIPTION
    El tablero calibra al empezar (~35 s): una calibración de otro día u otra postura no sirve.
    Durante el uso compensa la distancia a la cámara, se reajusta solo si la persona cambia
    de posición y aprende de cada selección.

.EXAMPLE
    .\scripts\mekasa.ps1 instalar      # una sola vez
    .\scripts\mekasa.ps1 camara        # ver qué cámara responde
    .\scripts\mekasa.ps1 prueba        # calibra (~35 s) + 8 consignas, mide el acierto real
    .\scripts\mekasa.ps1 tablero       # calibra (~35 s) y abre SÍ / NO / AGUA / AYUDA
    .\scripts\mekasa.ps1 tablero -SinCalibrar   # reutiliza la última calibración + ajuste de 4 puntos
    .\scripts\mekasa.ps1 tablero -Ojo LEFT -Dwell 1.2
    .\scripts\mekasa.ps1 simular       # prueba de seguridad sin cámara
    .\scripts\mekasa.ps1 tests         # pruebas automáticas
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("instalar", "camara", "prueba", "tablero", "simular", "tests", "ayuda")]
    [string]$Accion = "ayuda",

    [ValidateSet("BOTH", "LEFT", "RIGHT")]
    [string]$Ojo = "BOTH",

    # Segundos de mirada fija para elegir (0 = valor de config/communication.toml, 0.9 s).
    [double]$Dwell = 0,

    # Veces que se pide cada opción en la prueba (2 = 8 consignas, ~1 min).
    [ValidateRange(1, 5)]
    [int]$Repeticiones = 2,

    [switch]$Confirmar,
    [switch]$SinCalibrar,
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

function Get-BoardArguments {
    $arguments = @("--eye-mode", $Ojo)
    if ($Dwell -gt 0) { $arguments += @("--dwell", $Dwell.ToString([System.Globalization.CultureInfo]::InvariantCulture)) }
    if ($Confirmar) { $arguments += "--confirm" }
    if ($SinCalibrar) { $arguments += "--no-calibrate" }
    if ($Ventana) { $arguments += "--no-fullscreen" }
    return $arguments
}

$keys = "Teclas: 1-4 elegir · ESPACIO pausa · A reajustar · K recalibrar · M cambiar ojo · G ver mirada · C borrar · ESC salir"

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
    "prueba" {
        Write-Host "Calibración (~35 s): mire cada punto; en el punto central mueva despacio la cabeza." -ForegroundColor Cyan
        Write-Host "Después se pide mirar cada opción $Repeticiones veces. El resultado queda en pantalla y en work/communication." -ForegroundColor Cyan
        Write-Host $keys -ForegroundColor DarkCyan
        $arguments = (Get-BoardArguments) + @("--test", "$Repeticiones")
        Invoke-Py "mekasa_gaze.runners.communication_board" $arguments
    }
    "tablero" {
        if (-not $SinCalibrar) {
            Write-Host "Calibración (~35 s): mire cada punto; en el punto central mueva despacio la cabeza." -ForegroundColor Cyan
        }
        Write-Host $keys -ForegroundColor DarkCyan
        Invoke-Py "mekasa_gaze.runners.communication_board" (Get-BoardArguments)
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
