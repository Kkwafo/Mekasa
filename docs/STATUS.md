# Estado de Mekasa y punto de continuación

Última actualización: 2026-11-16, rama `feature/v0-communication-loop` (creada desde `origin/development` @ `33d9499`).

## Qué funciona de extremo a extremo

1. **Captura y detección (hito 1).** `uv run mekasa-experiment-001 --no-display` corre en este equipo. Hay más detalles en `research/EVIDENCE.md`.
2. **Tablero de comunicación (hito 2, parcial).** `uv run mekasa-board` abre un tablero SÍ / NO / AGUA / AYUDA a pantalla completa.
   - Por defecto es un tablero de conversación (menús para armar mensajes completos, URGENTE en un paso); `--interface simple` deja SÍ / NO / AGUA / AYUDA.
   - Empieza con una calibración de ~35 s en la misma sesión y muestra el acierto validado por modo ocular. `--test N` agrega una prueba con consignas sobre el tablero real.
   - Usa dwell ajustable (900 ms por defecto), progreso visible, histéresis, enfriamiento y rearme al salir de la opción.
   - Tiene confirmación opcional (`--confirm`) y modos oculares `--eye-mode BOTH|LEFT|RIGHT`. El modo binario se quitó.
   - Mientras se usa: compensa la distancia, avisa cambios de posición, se reajusta solo y aprende de cada selección por mirada.
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

## Análisis de la sesión de las 20:20 y rediseño (2026-11-16, noche)

Síntoma informado: los cuadrantes superiores funcionan; abajo-izquierda y abajo-derecha no.

### Qué pasó en esa sesión

1. **La calibración nueva nunca se entrenó.** `mekasa.ps1 calibrar` corrió la validación guiada, pero los tres `mekasa-calibrate` fallaron con "no usable GAZE_DISCRIMINATION observations":
   - la configuración pedía `left_iris_v` y `right_iris_v`;
   - el reporte guiado guarda una lista fija de features que no las incluía (`services/reporting.py`, `_PERSISTED_FEATURES`).

   El script solo mostró una advertencia, así que regiones y tablero usaron la calibración de las 16:52. **Fue un error mío de la ronda anterior**: probé la calibración solo con reportes viejos. Ya está corregido: el reporte guarda `*_iris_v`. El flujo nuevo ya no depende de ese paso.
2. **Una calibración de otra sesión no sirve.** Las dos validaciones guiadas de hoy son A (16:36) y B (20:20). Acierto 2×2 a nivel frame, Ridge sobre landmarks:

   | Entrenado → evaluado | Acierto 2×2 | Horizontal | Vertical |
   | --- | --- | --- | --- |
   | A → A (bloque 3 apartado) | 96–100 % | 97–100 % | 97–98 % |
   | B → B (bloque 3 apartado) | 40–77 % | 87–100 % | 50–90 % |
   | A → B | 26–47 % | 56–74 % | 50–82 % |
   | B → A | 15–36 % | 51–87 % | 18–52 % |
   | A → B con ajuste afín de 4 puntos | 70–72 % | 93–95 % | 75–83 % |
   | B, una sola ronda nueva de 9 puntos | 66–67 % | 89–90 % | 72–81 % |
   | A, una sola ronda nueva de 9 puntos | 100 % | 100 % | 100 % |

   El ajuste de 4 puntos no rescata una calibración vieja. Una calibración nueva sí.
3. **En B, los landmarks casi no traen señal vertical.**
   - `iris_y`, normalizado por el párpado, queda plano: 0,41 arriba y 0,42 abajo.
   - El EAR baja poco: de 0,37 a 0,31 (en A bajaba de 0,26 a 0,12).
   - La horizontal sigue bien.
   - En B, la persona estaba corrida en la imagen (centro x 0,56 contra 0,40) y con los ojos más abiertos. No sé si era otra persona o la misma con otra luz o lentes.

   Así se explica exactamente que "arriba funciona, abajo no".

### Qué se cambió (medido sobre A y B, validación cruzada entre rondas)

| Cambio | Evidencia | Estado |
| --- | --- | --- |
| Calibración dentro del tablero al inicio de cada uso (~35 s: 9 puntos, cabeza en movimiento mirando el centro, 9 puntos en otro orden) | Ver la tabla anterior | implementado; test de punta a punta con persona sintética |
| Segmento de movimiento de cabeza mirando un punto fijo | Con 2 rondas, sin y con el segmento: A 81 → 95 %, B 94 → 98 % (2×2, frame). El segmento se agrupa con su ronda, sin fuga entre folds | medido en Mekasa (reanálisis de A y B) |
| Umbral personal de ojo cerrado = 60 % del EAR al mirar la fila de abajo (entre 0,06 y 0,18), más descarte de frames cercanos a parpadeos | A: 0,082; B: 0,173. Con 0,15 fijo, A pierde toda la fila de abajo | medido en Mekasa (reanálisis) |
| Modelo con selección automática: landmarks, imagen del ojo (parches de 20×10 alineados a las comisuras, como WebGazer) o ambos; alpha por validación cruzada entre rondas | Landmarks, 2 rondas + cabeza: A BOTH 95 % (100 % por fijación); B BOTH 98 % (100 % por fijación). **La imagen del ojo no se pudo medir con datos reales**: los reportes no guardan imágenes. En el replay sintético sin señal vertical en landmarks, la imagen la recupera (100 % contra < 85 %) | landmarks medidos; imagen sin medir con persona |
| Tablero hasta el borde inferior; estado y mensaje arriba | Más ángulo vertical entre filas (centros de fila 0,34 y 0,77 en lugar de 0,28 y 0,66) | implementado |
| Compensación de distancia (escala alrededor de la cámara por ancho de rostro de calibración / actual) | Geometría proyectiva | hipótesis de ingeniería |
| Reajuste automático de 4 puntos si la persona queda > 2,5 s más cerca, más lejos, corrida o girada respecto de la calibración | El ajuste afín corrige bien cambios de escala y desplazamiento dentro de una sesión | hipótesis de ingeniería; verificado en replay |
| Aprendizaje implícito: cada selección por mirada suma una muestra liviana (decae, tope de 30, `C` en 5 s la retira) | Idea de calibración implícita de WebGazer con clics | hipótesis de ingeniería |
| `--test N`: consignas sobre el tablero real, solo por mirada, sin aprendizaje implícito | Medida prospectiva honesta del sistema completo | implementado |
| Se quitó la disposición binaria | Pedido explícito | hecho |

Limitaciones:

- Los porcentajes de calibración son validación cruzada dentro de la misma sesión, y el tipo de modelo y el alpha se eligieron sobre esas mismas rondas: son optimistas.
- La única medida honesta del tablero es `mekasa.ps1 prueba`.
- Son dos sesiones de un equipo y no está claro si fue una persona o dos.

## Tablero de conversación (2026-11-16, noche)

Pedido: comunicar más que SÍ / NO / AGUA / AYUDA, con necesidades reales, y armar un mensaje completo antes de decirlo.

| Parte | Detalle | Estado |
| --- | --- | --- |
| Contenido | Necesidades más citadas para pacientes que no pueden hablar: dolor y su ubicación, sed y boca seca, aspiración, posición, frío y calor, baño, miedo, familia, información sobre su estado, ocio y afecto. Fuentes en `research/EVIDENCE.md` | publicado (resúmenes de búsqueda, no leídos completos) |
| Navegación | Grilla 2×2 en todas las pantallas (la única con precisión medida), 3 opciones + VOLVER, submenús MÁS… | implementado |
| Mensaje | Frases con prefijo de categoría ("Me duele" + "la cabeza"), intensidad opcional, varias frases antes de DECIR, BORRAR ÚLTIMO | implementado; tests unitarios y de punta a punta |
| Urgente | Una sola selección desde Inicio: voz + alarma local; detalles también inmediatos | implementado; test de punta a punta |
| Vocabulario editable | `config/vocabulario.txt`, validado con número de línea | implementado |

Sin medir con persona:

- **Tiempo por mensaje.** "Me duele la cabeza, mucho." son 6 selecciones más DECIR. Estimado entre 12 y 20 s con dwell de 0,9 s: es una hipótesis.
- **Fatiga ocular.** Según la encuesta en ELA, es la causa principal de abandono del seguimiento ocular.
- **Disparos accidentales de URGENTE.** Están en un solo paso a propósito; si molestan, se puede pedir confirmación solo para URGENTE (no está implementado).

## Selección por parpadeo y empezar/parar con 4 parpadeos (2026-11-16, noche)

Pedido: poder empezar y parar cuando la persona quiere elegir, y una modalidad donde se elige parpadeando 2 veces, no por tiempo.

| Parte | Detalle | Estado |
| --- | --- | --- |
| Detector (`tracking/blinks.py`) | Cierre de 0,08–0,8 s; serie con huecos ≤ 0,8 s; reporta 2 o 4 al terminar; ignora 1, 3, más de 4, cierres largos y pérdida de rostro | implementado; 4 tests unitarios |
| Modo parpadeo en la sesión | Mirar 0,4 s arma la opción y 2 parpadeos la eligen dentro de los 3 s siguientes. Nunca elige por tiempo. La pérdida de cámara o rostro desarma; la de ojo no, porque es el propio parpadeo | implementado; 4 tests de sesión |
| Empezar y parar | 4 parpadeos alternan lectura y pausa, en los dos modos. El modo parpadeo empieza en espera, y los reajustes automáticos no reactivan la lectura si estaba en pausa | implementado |
| De punta a punta | Persona sintética: 4 parpadeos empiezan; 5 s mirando fijo no eligen nada; "Necesito agua." se arma con 5 dobles parpadeos; 4 parpadeos pausan | test aprobado |

Sin medir con persona:

- **Dobles parpadeos espontáneos.** Pueden ocurrir; por eso se exige la mirada estable previa.
- **Parpadeos voluntarios** que no bajen del umbral personal de ojo cerrado (parpadeos incompletos).
- **Cansancio** de parpadear seguido.

Sugerencia: medir 10 dobles parpadeos voluntarios y 5 minutos de reposo (falsas selecciones), y ajustar `blink_*`.

## Modo combinado y atajos por parpadeo (2026-11-16, noche)

Pedido: mirada y parpadeo a la vez. Mirar 3 s elige; 3 parpadeos eligen; 5 dicen el mensaje; 7 llaman a urgencias.

| Parte | Detalle | Estado |
| --- | --- | --- |
| `selection_mode = "combined"` (por defecto) | Mirar `combined_dwell_seconds` (3 s) elige; mirar 0,4 s arma la opción para elegirla con parpadeos | implementado; tests de sesión y de punta a punta |
| Gestos configurables | 3 elegir · 4 empezar/parar · 5 decir · 7 urgencia (`blink_*_count`, distintos entre 2 y 9) | implementado |
| Urgencia por parpadeo | Funciona en cualquier fase y pantalla, aun en pausa o calibrando | implementado; test de punta a punta |
| Temporizador congelado durante una serie | Evita elegir por tiempo en medio de 5 o 7 parpadeos | implementado; test unitario |
| Rearme | Por tiempo hay que apartar la mirada; por parpadeo se puede elegir lo que quedó en el mismo lugar | implementado; test unitario y de punta a punta |

Sin medir con persona:

- **Contar con precisión 3, 4 y 5.** Si confunde 4 con 5 se pausa en lugar de decir; los números se pueden separar más (por ejemplo 3 / 5 / 7 / 9).
- **El temporizador se congela 0,8 s** después de cada parpadeo espontáneo: elegir por tiempo puede tardar algo más.
- **Urgencias falsas** por series largas involuntarias.

## Guiños: calibración por ojo y elegir con 3 guiños (2026-11-16, noche)

| Parte | Detalle | Estado |
| --- | --- | --- |
| Paso de guiños en la calibración (`tracking/wink_calibration.py`) | "Cierre solo el ojo derecho / izquierdo, 3 veces", con conteo en vivo. Umbral por ojo = punto medio entre la apertura normal más baja (incluida la mirada hacia abajo) y la apertura al guiñar | implementado; test unitario |
| Capacidad por ojo | Un ojo se habilita si con esos umbrales se reconocen al menos 2 de 3 guiños con el otro ojo abierto. Si no logra aislarlo, queda deshabilitado y se dice en el resumen | implementado; test de punta a punta (ojo derecho sí, izquierdo no) |
| Detector | Cada cierre se clasifica por mayoría de cuadros (ambos, izquierdo o derecho); cada tipo arma su propia serie. Los guiños solo eligen, nunca pausan, dicen ni llaman a urgencias | implementado; tests unitarios |
| Perfil guardado | `work/calibration/session-wink-profile.json` (solo umbrales y conteos), reutilizado con `-SinCalibrar` | implementado |
| Corrección encontrada en la prueba | Si un gesto (5 = decir, 7 = urgencia) o una tecla cambiaba la pantalla, el temporizador de 3 s seguía sobre el mismo casillero y podía elegir en la pantalla nueva. Ahora se reinicia el foco y hay un enfriamiento (`CommunicationSession.interrupt`) | corregido; test |

Sin medir con persona:

- Si puede guiñar con cada ojo de forma confiable.
- Si al guiñar el otro ojo se entrecierra lo suficiente para que el guiño se lea como parpadeo de ambos ojos (entonces no cuenta).

## Primero parpadear, después mirar; urgencia con 7 o más (2026-11-17)

Comentario de la persona: "al pestañear o guiñar el ojo a veces se va para otro lado", y mirar fijo mientras se parpadea cuesta.

| Cambio | Detalle | Estado |
| --- | --- | --- |
| `blink_select_order = "blink_then_look"` (por defecto) | 3 parpadeos o guiños abren una ventana de 5 s (estado SELECTING, "Elija"). La opción mirada 1 s dentro de la ventana se elige. La mirada durante el gesto no importa. Otro parpadeo no cierra la ventana; perder la cara o pausar sí | implementado; tests de sesión y de punta a punta (mirando a otro lado durante el gesto) |
| Mirada congelada al reabrir | Se mantiene el último punto estable 0,3 s después de cada cierre | implementado |
| Urgencia con 7 **o más** | Se dispara en el séptimo parpadeo, sin esperar al final de la serie. Si sigue, se repite como mucho cada 2,5 s | implementado; tests unitario y de punta a punta (14 parpadeos = 2 llamadas) |
| Urgencia desactivada durante el paso de guiños | Ese paso pide cierres repetidos (y quien no aísla un ojo cierra los dos). Lo encontró el test de punta a punta | corregido; test |

Sin medir con persona: si 1 s de mirada después del gesto es cómodo (`intent_focus_seconds`) y si 0,3 s de congelamiento alcanzan.

## Qué es desarrollo propio y qué es de terceros

| Parte | De quién | Qué hace |
| --- | --- | --- |
| MediaPipe Face Landmarker (`face_landmarker.task` y la librería `mediapipe`) | **Google**. Redes neuronales pre-entrenadas, congeladas, no las entrenamos | Detecta la cara y devuelve 478 puntos (incluidos iris y párpados) y la pose de la cabeza en cada imagen. **Es la única red neuronal del sistema.** El modelo se descarga aparte; su licencia de redistribución está pendiente de revisar |
| OpenCV, NumPy, Pillow, psutil | Proyectos abiertos (Apache 2.0 / BSD / HPND) | Cámara y procesamiento de imagen; álgebra; texto con acentos; métricas del sistema |
| Voz en español | **Microsoft** (Windows SAPI, voz local) | Leer los mensajes en voz alta |
| Todo lo demás en `engines/gaze_engine/src` | **Desarrollo propio de Mekasa** | Ver la lista de abajo |

El desarrollo propio incluye:

- **Señal:** rasgos del ojo (apertura del ojo, iris normalizado, posición respecto de las comisuras), imágenes del ojo de 20×10 y reglas de calidad de señal.
- **Calibración en la sesión:** con selección automática del modelo y umbral personal de ojo cerrado.
- **Corrección durante el uso:** monitor de distancia, reajuste automático y aprendizaje por selección.
- **Interacción:** detector de parpadeos y sesión de selección con sus bloqueos de seguridad.
- **Comunicación:** tablero de conversación y vocabulario, interfaz, script y reportes.

**¿Hay una red neuronal propia para la calibración? No.** El modelo personal es una **regresión ridge**, un modelo lineal con regularización, entrenado en ~35 s con la propia persona. Sobre los puntos del rostro, las imágenes del ojo o ambos, se elige automáticamente.

- **Por qué no una red neuronal:** con ~700 imágenes muy parecidas de una sola persona y una sola sesión, una red sobreajusta. El modelo lineal con validación cruzada es lo que los datos sostienen (en las sesiones de hoy: 95–98 % 2×2 con puntos del rostro).
- **El hito 3 (red pequeña propia en CPU) sigue pendiente.** Tiene sentido como candidata más dentro de la misma selección, que solo se usa si gana en las rondas apartadas. Mejor todavía: una red pre-entrenada con muchas personas y personalizada con pocos puntos (FAZE, He et al.), lo que exige datasets con licencia compatible.

## Resultados por modo con baseline y modelo adaptado

Baseline Ridge por modo: ver la sesión del 2026-11-16 arriba. **No hay modelo adaptado todavía** (hito 3). El resultado anterior es del 2026-09-21, en modo binocular y con distancia mayor:

- acierto por blanco fijo: 63 %;
- error mediano: 125,6 px;
- 4 regiones con corrección de sesión: 7/8.

Esto **no alcanza** para usar el sistema como CAA. Para uso real, el acceso directo sigue siendo la vía segura.

## Próximos pasos (en orden), comando exacto y archivos

1. **Prueba con la persona (requiere buena luz y la cámara arriba de la pantalla).**
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\mekasa.ps1 prueba
   ```
   - Hay que anotar el acierto por opción (en pantalla y en `work/communication/board-session-latest.json` → `test`) y qué modelo eligió cada modo (`calibration.fits.*.kind`).
   - Si abajo sigue fallando con `kind = landmarks`, revisar la luz sobre los ojos. Si falla con `appearance` o `hybrid`, probar `-Ojo LEFT` o `-Ojo RIGHT` y subir la cámara o bajar la pantalla.
2. **Tablero de conversación con la persona.** Pedir 5 mensajes concretos (p. ej. "Me duele la espalda, bastante") y medir el tiempo y los errores por mensaje. Anotar qué falta en el vocabulario y editar `config/vocabulario.txt`.
3. **Falsos positivos y dwell.** Un minuto de mirada libre por perfil: `-Dwell 0.45`, `0.9` y `1.2`, contando mensajes no deseados.
3. **Calibración corta.** Comparar 3, 5 y 9 puntos no colineales (`config/experiment-001.toml`, `[targets.*]`). Recalibrar solo una región u ojo, y guardar el perfil asociado a persona, modo, cámara y pantalla.
4. **Hito 3: estimador neuronal.** Una MLP pequeña en CPU sobre las features, con máscara por ojo, o una corrección residual del Ridge. Requiere entrenamiento reproducible, promoción solo si mejora en blancos apartados, y rollback atómico. Hay que justificar la dependencia (numpy puro o PyTorch/ONNX; ver el ADR-006 histórico).
5. **Pendientes de UI.**
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
Modo ocular / dwell / modelo elegido (calibration.fits.*.kind):
Calibración completada (sí/no) y reporte:
Selecciones correctas / incorrectas / sin selección (con blancos instruidos):
Falsos positivos durante N minutos de mirada libre:
Observaciones de la persona y de quien acompaña:
```

## Entorno de esta sesión

- `uv` 0.13.8 instalado con winget (alcance de usuario) y CPython 3.12.16 instalado por uv.
- Modelo Face Landmarker descargado con `scripts/fetch-face-landmarker.ps1` (SHA-256 verificado, no versionado).
- No se hicieron operaciones remotas (push, PR ni publicación).
