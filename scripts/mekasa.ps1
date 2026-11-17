﻿<#
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
    .\scripts\mekasa.ps1 tablero       # calibra (~35 s) y abre el tablero de conversación
    .\scripts\mekasa.ps1 tablero -Genero F      # frases en femenino ("Estoy cansada")
    .\scripts\mekasa.ps1 tablero -Simple        # tablero simple SÍ / NO / AGUA / AYUDA
    #   Por defecto: mirar 3 s elige, o 3 parpadeos (o guiños) y DESPUÉS mirar la opción 1 s;
    #   4 parpadeos empiezan/paran; 5 dicen el mensaje; 7 o más llaman a urgencias (y repiten).
    .\scripts\mekasa.ps1 tablero -Parpadeo      # solo parpadeos (nunca por tiempo)
    .\scripts\mekasa.ps1 tablero -SoloTiempo    # solo mirada (0,9 s), sin elegir por parpadeo
    .\scripts\mekasa.ps1 tablero -Dwell 2       # mirar 2 s en lugar de 3
    .\scripts\mekasa.ps1 tablero -Perfil lentes # calibraciones separadas por persona
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

    # Segundos de mirada fija para elegir (0 = valor de config/communication.toml: 3 s).
    [double]$Dwell = 0,

    # Veces que se pide cada opción en la prueba (2 = 8 consignas, ~1 min).
    [ValidateRange(1, 5)]
    [int]$Repeticiones = 2,

    # Género gramatical de las frases ("Estoy cansado" / "Estoy cansada").
    [ValidateSet("M", "F")]
    [string]$Genero = "",

    # Etiqueta corta por persona: cada una conserva sus calibraciones y sus reportes.
    [string]$Perfil = "",

    [switch]$Simple,
    # Solo parpadeos (nunca por tiempo) / solo mirada por tiempo. Sin ninguno: los dos.
    [switch]$Parpadeo,
    [switch]$SoloTiempo,
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
    if ($Genero) { $arguments += @("--gender", $Genero.ToLower()) }
    if ($Simple) { $arguments += @("--interface", "simple") }
    if ($Perfil) { $arguments += @("--profile", $Perfil) }
    if ($Parpadeo) { $arguments += @("--selection", "blink") }
    elseif ($SoloTiempo) { $arguments += @("--selection", "dwell") }
    if ($SinCalibrar) { $arguments += "--no-calibrate" }
    if ($Ventana) { $arguments += "--no-fullscreen" }
    return $arguments
}

$keys = "Parpadeos seguidos: 3 y luego mirar = elegir · 4 empezar/parar · 5 decir · 7+ urgencia. Teclas: 1-4 elegir · RETROCESO volver · H inicio · ESPACIO pausa · A reajustar · K recalibrar · M cambiar ojo · G ver mirada · C borrar · ESC salir"

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
