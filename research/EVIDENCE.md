# Tabla de evidencia (Mekasa)

Estados posibles de cada cifra:

- **publicado**: leído en la fuente indicada durante la sesión de revisión.
- **hipótesis de ingeniería**: valor elegido por Mekasa y pendiente de contrastar.
- **medido en Mekasa**: obtenido en una ejecución registrada en este repositorio.
- **pendiente**: la fuente no se pudo leer o no se verificó todavía. La cifra no se usa para decidir.

El error angular, el error en pantalla y el acierto en la tarea de comunicación se registran por separado. Los grados se convierten a unidades de pantalla solo si hay una geometría medida (distancia, tamaño y resolución). No hacemos esa conversión para ninguna fila de esta tabla.

Revisión: 2026-11-16. Fuentes de difícil acceso:

- `gazecapture.csail.mit.edu` rechazó la conexión.
- Los artículos de PMC devolvieron una verificación de navegador que no intentamos sortear.

| Fuente y enlace | Año | Población | Equipo | Modo ocular | Protocolo | Valor y unidad | Estado | Limitación | Decisión en Mekasa | Prueba propia |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [GazeCapture / iTracker, Krafka et al.](https://gazecapture.csail.mit.edu/) | 2016 | Pendiente | Teléfono y tableta | Ambos ojos + cara | Con y sin calibración | Pendiente: la página rechazó la conexión | pendiente | Usa móvil y tableta, no webcam de escritorio | Solo como referencia metodológica: separar error sin calibrar y calibrado y anotar la geometría | Ninguna todavía |
| [FAZE, Park et al.](https://research.nvidia.com/publication/2019-10_few-shot-adaptive-gaze-estimation) | 2019 | No indicada en la página | No indicado | Ambos ojos | Pocas muestras de calibración (≤ 9; mejoras desde 3) | Error 3,18 en GazeCapture, unidad no impresa en la página; 19 % de mejora relativa | publicado (falta confirmar la unidad en el PDF) | Dataset de móvil; red pre-entrenada grande | Comparar calibración de 3, 5 y 9 puntos con el mismo protocolo; distinguir posiciones únicas de fotogramas correlacionados | Pendiente: protocolo de 3/5/9 puntos (docs/STATUS.md, siguiente paso 3) |
| [On-Device Few-Shot Personalization, He et al.](https://research.google/pubs/on-device-few-shot-personalization-for-real-time-gaze-estimation/) | 2019 | No indicada | Cámara frontal de teléfono | Ambos ojos | 3 puntos de calibración | 24 % de mejora relativa sobre el mejor modelo sin calibrar; 20 veces menos FLOPS | publicado (la página no da error absoluto) | No hay error absoluto ni población en la página | Apoya la personalización con pocos puntos en CPU; justifica probar una cabeza pequeña sobre las features | Pendiente (hito 3, red neuronal) |
| [Person-Specific Gaze Estimation from Low-Quality Webcam Images](https://pmc.ncbi.nlm.nih.gov/articles/PMC10147084/) | Pendiente | Pendiente | Webcam | Pendiente | Pendiente | Pendiente: verificación de navegador | pendiente | No leída | Ninguna hasta leerla | Ninguna |
| [MPIIGaze, Zhang et al.](https://pubmed.ncbi.nlm.nih.gov/29990057/) / [ETH-XGaze](https://ait.ethz.ch/xgaze) | Pendiente | Pendiente | Portátil (MPIIGaze) | Pendiente | Pendiente | Pendiente | pendiente | La licencia de datos ETH-XGaze debe revisarse antes de descargar o entrenar | No se descarga ni se usa ningún dataset | Ninguna |
| [Majaranta et al., feedback y dwell](https://doi.org/10.1145/968363.968390) | 2004 | Pendiente | Eye tracker | Pendiente | Escritura ocular | 450 ms y 900 ms (valores citados por el plan, sin leer la fuente) | pendiente | Tarea distinta (escritura, no tablero de 4 opciones) | Dwell inicial 900 ms; compararlo con 450 ms y 1200 ms en Mekasa | Replay: dispara al tiempo configurado (±1 frame) para 450, 900 y 1200 ms (`test_dwell_hypotheses_fire_at_configured_time`). Falta la prueba guiada con personas. |
| [ASHA, práctica de CAA](https://www.asha.org/practice-portal/professional-issues/augmentative-and-alternative-communication/) | — | — | — | — | — | — | pendiente | No leída en esta sesión | Selección directa más acceso alternativo (teclado, puntero o pulsador) siempre disponible | Implementado: teclas 1-4 y clic funcionan sin calibración |
| [Karlsson et al., consenso en parálisis cerebral](https://pmc.ncbi.nlm.nih.gov/articles/PMC7874479/) | Pendiente | Parálisis cerebral | Pendiente | Pendiente | Consenso | Pendiente: verificación de navegador | pendiente | No leída | Pendiente | Ninguna |
| [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | — | — | CPU, `mediapipe==1.0.1` | Ambos ojos (malla completa) | Modo VIDEO | 478 landmarks (inspeccionado en el código) | medido en Mekasa | Si el rostro completo no se detecta, se pierden ambos ojos. El modo de un solo ojo depende de que la malla siga detectándose. | Insumo del estimador, no estimador de mirada | Pendiente: medir pérdida de malla con un ojo ocluido |

## Mediciones propias relevantes

| Medición | Valor | Estado | Fuente |
| --- | --- | --- | --- |
| Captura y detección, i5-8250U, persona presente | 901/901 frames `TRACKING_GOOD`, 29,88 FPS | medido en Mekasa (2026-09-20) | `research/experiments/001-eye-feature-extraction/results/2026-09-20-windows-i5-8250u-dynamic-uncontrolled-gaze.json` |
| Captura y detección en el equipo actual (20 CPU lógicas), sin persona estable delante | 282/901 frames `TRACKING_GOOD` (31,3 %), 30,01 FPS, features p50 3,2 ms | medido en Mekasa (2026-11-16) | `results/latest.json` (no versionado). **Es una prueba de captura, no de selección.** |
| Validación guiada de 9 puntos con modelo Ridge, distancia mayor | Acierto por blanco fijo 63,09 %, error mediano 125,6 px y p90 174,0 px (783 muestras) | medido en Mekasa (2026-09-21) | Historial git, commit `424a5d0` |
| 4 regiones con corrección de sesión | 7/8 ensayos puntuados correctos (87,5 %), 8 ensayos en 1 sesión | medido en Mekasa (2026-09-21) | Historial git, `research/experiments/003-region-selection/RESULTS.md` antes de `05edf42` |
| Tablero, replay sintético, 3 modos oculares | 0 emisiones en las condiciones inseguras; SÍ, AGUA y AYUDA comunicados | verificado en replay | `uv run mekasa-board --simulate` |
| Tablero, hardware real sin ventana, 15 s, sin calibración | 211 frames, 14,07 FPS efectivos, captura→decisión p50 18,4 ms y p95 30,6 ms, 0 emisiones | medido en Mekasa (2026-11-16) | `docs/STATUS.md` |
