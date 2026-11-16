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

## Sesión guiada con persona, 2026-11-16 (una persona, un equipo, una sesión)

Equipo: Windows 11 Pro 26300, Intel familia 6 modelo 151 (20 CPU lógicas), cámara índice 0 MSMF 640×480.

| Prueba | Resultado | Estado |
| --- | --- | --- |
| Validación guiada de 9 puntos (`mekasa-guided-validation`) | Completa, 7109 frames a 30,0 FPS, ancho de cara 0,37–0,42 | medido en Mekasa |
| Mirada hacia abajo con el umbral de ojo cerrado 0,15 | DOWN: 267/269 frames "ojo cerrado" (EAR mediano 0,12–0,14 contra 0,19 en el centro): el párpado sigue la mirada | medido en Mekasa |
| Reanálisis con umbral personal 0,09 (`--reclassify-ear-threshold 0.09`, elegido después de ver los datos) | Uso de ambos ojos: DOWN 0,82, BOTTOM_RIGHT 0,85, BOTTOM_LEFT 1,0 | reanálisis exploratorio; validar en otra sesión |
| Ridge, bloque 3 apartado, 9 blancos (configuración por defecto) | BOTH 40 % / 153 px; LEFT 23 % / 208 px; RIGHT 19 % / 203 px | medido en Mekasa |
| Mismo modelo, cuadrante 2×2 (frames de blancos diagonales, correlados, 4 posiciones) | BOTH 98,5 % (334 frames); LEFT 51 %; RIGHT 68 % | medido en Mekasa (nivel frame, no selección) |
| Validación controlada de 4 regiones (BOTH, dwell 1,2 s, corrección de sesión) | 7/8 correctos (87,5 %; IC 95 % Wilson ~53–98 %), 0 sin selección, respuesta ~1,22 s; error TOP_LEFT→BOTTOM_RIGHT; 20 % de frames fuera de distancia | medido en Mekasa |
| Tablero `mekasa-board` (BOTH, dwell 0,9 s, sin corrección de sesión), 30 s | 4 confirmaciones por mirada, todas en la fila superior; 279 ticks sin señal ocular, 129 sin tracking, 17 enfoques cancelados; captura→decisión p50 15 ms y p95 28 ms; 21,6 FPS procesados | medido en Mekasa |

Hallazgos:

1. **El umbral de ojo cerrado debe ser personal o depender de la mirada vertical.** Con el valor poblacional, mirar hacia abajo cuenta como "ojo cerrado".
2. **La fila inferior del tablero es casi inalcanzable.** Incluso con 0,09, un 15 % de frames sin señal ocular repartidos al azar alcanza para cortar un dwell de 27 frames contiguos (0,85^27 ≈ 1 %). Hace falta tolerar caídas breves de señal ocular, como un parpadeo, sin cancelar ni avanzar el dwell (decisión pendiente).
3. **La señal vertical del ojo derecho casi no varía** (iris_y 0,37 arriba contra 0,355 abajo); la del izquierdo sí. Hipótesis: los pares de párpado usados para normalizar no son simétricos entre ojos (`_LEFT_EYE` usa 387/373 y `_RIGHT_EYE` usa 158/153; los espejos serían 385/380 o 160/144). Hay que verificarlo antes de confiar en el modo RIGHT.
4. **La deriva de postura entre bloques** (pitch de 7° a 11°) degrada la precisión fina. La corrección de sesión todavía no se aplica en el tablero.

## Cambios tras la sesión del 2026-11-16 (precisión, duración, UI)

| Cambio | Motivo | Estado |
| --- | --- | --- |
| EAR de ambos ojos como dato de entrada de la calibración | En el bloque apartado: RIGHT 2×2 68 → 97 %, LEFT 51 → 76 %, BOTH 9 puntos 40 → 48 % | elegido sobre el mismo bloque; confirmar en otra sesión |
| `*_iris_v`: posición vertical del iris respecto de la línea entre comisuras | El párpado sigue a la mirada y aplana `iris_y` (ojo derecho casi plano) | hipótesis sin medir; requiere nueva validación guiada |
| Umbral personal de ojo cerrado guardado en la calibración y aplicado por el tablero | La mirada hacia abajo contaba como ojo cerrado | implementado y testeado |
| Ajuste rápido de 4 puntos al abrir el tablero (corrección afín de sesión) | La corrección de sesión llevó 4 regiones de 58 % a 87,5 % en septiembre; el tablero no la tenía | implementado, verificado en replay; falta medir con persona |
| Tolerancia de 0,2 s a caídas de señal ocular (congela el progreso, no avanza ni emite) | ~15 % de caídas sueltas hacían inalcanzable la fila inferior | implementado; las pérdidas de cámara, imagen y rostro siguen cancelando |
| Protocolo rápido `config/experiment-001-quick.toml` | Pedido de pruebas más cortas: ~1,7 min en lugar de ~3,6 min de medición | listo, sin probar con persona |
| UI: texto con acentos (Pillow y Segoe UI), indicador por ojo, aviso de distancia, pantalla de ajuste | SÍ se veía como SI; faltaba saber por qué no seleccionaba | verificado con capturas renderizadas |
| `scripts/mekasa.ps1` | Levantar y probar con un comando | `simular` y `tests` ejecutados |

## Resultados por modo con baseline y modelo adaptado

Baseline Ridge por modo: ver la sesión del 2026-11-16 arriba. **No hay modelo adaptado todavía** (hito 3). El resultado anterior es del 2026-09-21, en modo binocular y con distancia mayor:

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
