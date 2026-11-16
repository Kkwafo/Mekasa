# Estado de Mekasa y punto de continuación

Última actualización: 2026-11-16, rama `feature/v0-communication-loop` (creada desde `origin/development` @ `33d9499`).

## Qué funciona de extremo a extremo

1. **Captura y detección (hito 1).** `uv run mekasa-experiment-001 --no-display` corre en este equipo. Hay más detalles en `research/EVIDENCE.md`.
2. **Tablero de comunicación (hito 2, parcial).** `uv run mekasa-board` abre un tablero SÍ / NO / AGUA / AYUDA a pantalla completa.
   - Usa dwell ajustable (900 ms por defecto), progreso visible, histéresis, enfriamiento y rearme al salir de la opción.
   - Tiene confirmación opcional (`--confirm`), modo binario (`--layout binary-horizontal|binary-vertical`) y modos oculares `--eye-mode BOTH|LEFT|RIGHT`.
   - Usa TTS local (Windows SAPI). Si la voz falla, el mensaje queda visible.
   - Pausa con la barra espaciadora. El acceso directo por teclas 1-4 y clic funciona incluso sin calibración.
   - La captura corre en un hilo con un búfer de un solo frame y reconexión automática.
3. **Calibración por modo ocular.** `uv run mekasa-calibrate --eye-mode LEFT --output work/calibration/personal-gaze-model-left.json`. Hace lo mismo con `RIGHT` y `BOTH`. El tablero carga `personal-gaze-model-{both,left,right}.json` y, para `BOTH`, acepta el archivo histórico `personal-gaze-model.json`.
4. **Replay sin cámara.** `uv run mekasa-board --simulate` pasa datos sintéticos por la cadena real: validez → ubicación → estimador → suavizado → tablero → sesión → voz. Lo hace en los 3 modos oculares y sale con código 1 si hubo una emisión en un segmento inseguro.

## Pruebas realmente ejecutadas (2026-11-16)

| Prueba | Resultado |
| --- | --- |
| `uv run pytest` | 96 tests aprobados (57 previos y 39 nuevos) |
| `ruff check`, `ruff format --check`, `mypy` estricto | Sin errores (mypy por `python -m mypy`, porque Control de aplicaciones de Windows bloquea el lanzador `mypy.exe`) |
| `mekasa-experiment-001 --no-display` | 30,01 FPS, 282/901 `TRACKING_GOOD`. No había una persona estable delante: es prueba de captura, no de selección. |
| `mekasa-board --simulate` | 0 emisiones inseguras en BOTH, LEFT, RIGHT y sin calibración |
| Smoke de hardware sin ventana, 15 s (cámara real, MediaPipe, pipeline y sesión sin calibración) | 211 frames, 14,07 FPS, procesamiento p50 16,4 ms y p95 28,1 ms, captura→decisión p50 18,4 ms y p95 30,6 ms, 0 emisiones |

Bloqueos de cero emisión cubiertos por tests deterministas (`tests/test_communication.py` y `tests/test_board_pipeline.py`):

- cámara desconectada y frame congelado o vencido;
- tracking inválido y persona fuera de distancia;
- ojo requerido cerrado y modo `BOTH` con un solo ojo;
- pausa, cambio de modo y arranque sin calibración válida o con calibración de otro modo;
- mirada fija sobre la opción ya confirmada;
- caída de señal a mitad del dwell y confirmación pendiente vencida.

## Resultados por modo con baseline y modelo adaptado

**No hay todavía.** Se necesita una persona frente a la cámara que complete la validación guiada. El último resultado real es del 2026-09-21, en modo binocular y con distancia mayor:

- acierto por blanco fijo: 63 %;
- error mediano: 125,6 px;
- 4 regiones con corrección de sesión: 7/8.

Esto **no alcanza** para usar el sistema como CAA. Para uso real, el acceso directo sigue siendo la vía segura.

## Próximos pasos (en orden), comando exacto y archivos

1. **Primera medición guiada en este equipo (requiere persona y buena luz).**
   ```powershell
   uv run mekasa-guided-validation
   uv run mekasa-calibrate --eye-mode BOTH --output work/calibration/personal-gaze-model-both.json
   uv run mekasa-calibrate --eye-mode LEFT --output work/calibration/personal-gaze-model-left.json
   uv run mekasa-calibrate --eye-mode RIGHT --output work/calibration/personal-gaze-model-right.json
   uv run mekasa-controlled-region-validation
   uv run mekasa-board
   ```
   Hay que registrar el resultado en `research/COMPATIBILITY.md`. Con la luz actual la cámara bajó a 14 FPS: medir también con más luz.
2. **Validación con blancos instruidos sobre el tablero.** Agregar un runner que pida opciones al azar, sume pausas de mirada libre (falsos positivos por minuto) y compare dwell de 450, 900 y 1200 ms. Se puede reutilizar `CommunicationSession` y el protocolo de `runners/controlled_region_validation.py`.
3. **Calibración corta.** Comparar 3, 5 y 9 puntos no colineales (`config/experiment-001.toml`, `[targets.*]`). Recalibrar solo una región u ojo, y guardar el perfil asociado a persona, modo, cámara y pantalla.
4. **Hito 3: estimador neuronal.** Una MLP pequeña en CPU sobre las features, con máscara por ojo, o una corrección residual del Ridge. Requiere entrenamiento reproducible, promoción solo si mejora en blancos apartados, y rollback atómico. Hay que justificar la dependencia (numpy puro o PyTorch/ONNX; ver el ADR-006 histórico).
5. **Pendientes de UI.**
   - Texto Unicode en pantalla: hoy "SÍ" se ve como "SI" porque Hershey solo tiene ASCII. La voz sí pronuncia el acento.
   - Pictogramas con licencia verificada.
   - Barrido con pulsador.
   - Ajuste de posición y tamaño de opciones desde la UI.
6. **Distribución.**
   - Reporte de licencias de dependencias.
   - Revisión de licencia del modelo MediaPipe (redistribución bloqueada; ver `models/gaze/mediapipe-face-landmarker/metadata.json`).
   - Instalador para Windows.
   - Protocolo de piloto con consentimiento y apoyo profesional.

## Formulario corto de prueba física

Pegar en `research/COMPATIBILITY.md`:

```
Fecha / sistema / CPU:
Cámara (modelo, integrada/USB) / backend / resolución y FPS efectivos (mekasa-probe-cameras):
Luz (natural/artificial/contraluz) / lentes (sí/no) / distancia aproximada:
Modo ocular / dwell / layout:
Calibración completada (sí/no) y reporte:
Selecciones correctas / incorrectas / sin selección (con blancos instruidos):
Falsos positivos durante N minutos de mirada libre:
Observaciones de la persona y de quien acompaña:
```

## Entorno de esta sesión

- `uv` 0.13.8 instalado con winget (alcance de usuario) y CPython 3.12.16 instalado por uv.
- Modelo Face Landmarker descargado con `scripts/fetch-face-landmarker.ps1` (SHA-256 verificado, no versionado).
- No se hicieron operaciones remotas (push, PR ni publicación).
