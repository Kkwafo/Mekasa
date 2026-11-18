# Matriz de compatibilidad

Estados: **verificado en hardware**, **verificado en replay**, **pendiente**, **no compatible**.

Si una combinación funciona en hardware, eso no prueba que tenga buena precisión de mirada. La columna "Selección" cuenta solo si hubo blancos instruidos.

| Sistema | CPU | Cámara / backend | Resolución y FPS | Captura | Landmarks | Selección por mirada | Estado | Notas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Windows 11 (26200) | Intel i5-8250U | Integrada, MSMF índice 0 | 640×480 a 30 → 29,88 efectivos | Sí | 901/901 GOOD | 4 regiones: 7/8 correctos con corrección de sesión | verificado en hardware | Una sola sesión de operador |
| Windows 11 Pro (26300) | 20 CPU lógicas | Índice 0, MSMF | 640×480 a 30 → 30,01 efectivos (exp. 001); 14,07 efectivos en el tablero | Sí | 31 % GOOD, sin persona estable | No medida (sin calibración) | verificado en hardware (captura y pipeline) | La caída a 14 FPS en el tablero coincide con escena sin rostro y posible exposición automática. Hay que repetir con una persona y buena luz. |
| Windows 11 Pro (26300) | Intel fam. 6 mod. 151, 20 lógicas | Índice 0, MSMF | 640×480 a 30 → 30,0 efectivos (guiada) | Sí | 7109 frames, distancia 0,37–0,42 | 4 regiones: 7/8 (BOTH, corrección de sesión); tablero: solo la fila superior alcanzable | verificado en hardware | Una persona, una sesión. Umbral de ojo cerrado personal 0,09. Ver docs/STATUS.md |
| Cualquiera | — | Sintética | 30 FPS simulados | — | — | 3 modos oculares, 0 emisiones inseguras | verificado en replay | `uv run mekasa-board --simulate` |
| Windows | — | USB UVC externa | 640×480 y 1280×720, a 15 y 30 FPS | — | — | — | pendiente | Usar el formulario de prueba física de `docs/STATUS.md` |
| Linux (V4L2) | — | Integrada o USB | — | — | — | — | pendiente | El CI corre en ubuntu; la cámara real no está probada |
| macOS | — | Integrada | — | — | — | — | pendiente | |
| Cualquiera | — | Teléfono como webcam del sistema / cámara IP | — | — | — | — | pendiente | Se evaluará como extensión si hay hardware para probarlo |
| Cualquiera | — | Cámara solo infrarroja (tipo Windows Hello) | — | — | — | — | pendiente | El modelo se entrenó con imágenes RGB y los resultados son inciertos |
| Sistema no Windows | — | — | — | — | — | Voz | no compatible (por ahora) | El TTS local solo existe para Windows SAPI. En otros sistemas el mensaje queda en pantalla. |
