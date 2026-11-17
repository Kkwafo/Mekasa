# Mekasa (眼傘) — Sistema de Comunicación Aumentativa y Alternativa por Mirada

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Architecture: Clean & SOLID](https://img.shields.io/badge/architecture-Clean%20%26%20SOLID-emerald.svg)]()
[![Type Checked: Mypy Strict](https://img.shields.io/badge/mypy-strict%20checked-blue.svg)](https://mypy.readthedocs.io/)
[![Linter: Ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://docs.astral.sh/ruff/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Mekasa** es un prototipo técnico abierto y de procesamiento local (*edge / on-device*) que investiga interacción de **Comunicación Aumentativa y Alternativa (CAA)** mediante seguimiento ocular (*eye-tracking*). Su objetivo es explorar una vía de comunicación accesible para personas con compromiso motor o del habla; no afirma restituir capacidades ni sustituye una evaluación profesional.

---

## Tabla de Contenidos
1. [Visión del Proyecto y Filosofía](#1-visión-del-proyecto-y-filosofía)
2. [Arquitectura de Software (Clean & SOLID)](#2-arquitectura-de-software-clean--solid)
3. [Estructura del Repositorio](#3-estructura-del-repositorio)
4. [Instalación y Configuración](#4-instalación-y-configuración)
5. [Guía de Ejecución de Pruebas y Linters](#5-guía-de-ejecución-de-pruebas-y-linters)
6. [Flujo de Trabajo Experimental (Runners)](#6-flujo-de-trabajo-experimental-runners)
7. [Fundamentos Matemáticos y Componentes del Núcleo](#7-fundamentos-matemáticos-y-componentes-del-núcleo)
8. [Roadmap de Integración IA / LLM (Mekasa Intelligence)](#8-roadmap-de-integración-ia--llm-mekasa-intelligence)
9. [Aviso Clínico, Ético y Seguridad](#9-aviso-clínico-ético-y-seguridad)
10. [Licencia](#10-licencia)

---

## 1. Visión del Proyecto y Filosofía

La pérdida súbita del lenguaje oral tras un ACV sitúa al paciente en un estado de vulnerabilidad extremo. Los sistemas comerciales de eye-tracking tradicionales suelen depender de hardware propietario de alto costo (sensores infrarrojos especializados de miles de dólares), arquitecturas cerradas o telemetría en la nube.

Mekasa nace bajo un conjunto de **principios irrenunciables**:

* **Hardware Convencional en Evaluación:** El V0 se prueba con cámaras RGB integradas o USB a 640x480 y 30 FPS. La compatibilidad y precisión dependen de cámara, iluminación, distancia, movimiento, lentes y características individuales.
* **Procesamiento 100% On-Device (Local-First):** Toda la inferencia de visión computacional, filtrado geométrico, regresión y síntesis de voz opera localmente en la máquina del usuario.
* **Privacidad Local por Diseño:** No se guardan ni se suben fotogramas, video o landmarks faciales crudos. Los modelos personales derivados y reportes técnicos sí se almacenan localmente para poder calibrar y medir el sistema.
* **Mitigación Rigurosa del Efecto Midas (*Midas Touch*):** La mirada es un sentido exploratorio primario; mirar un elemento no equivale a desear activarlo. La interacción requiere protocolos deterministas de permanencia (*dwell-time*), histéresis espacial y enfriamiento.
* **Arquitectura Desacoplada para Asistencia Cognitiva:** Una clara separación de responsabilidades entre el hardware de captura, los algoritmos de mirada, la interacción por regiones y la futura capa de predicción semántica mediante Modelos de Lenguaje Locales (SLM / LLM).

---

## 2. Arquitectura de Software (Clean & SOLID)

El sistema aplica **SOLID de forma pragmática**: dominio y contratos en `core`, visión en `tracking`, presentación en `ui`, infraestructura en `services` y cinco flujos ejecutables en `runners`. No se mantienen capas vacías ni módulos espejo.

```
+-----------------------------------------------------------------------+
|                             PRESENTATION                              |
|   (Mekasa UI / OpenCV Visualization / Audio Synthesizer Controller)   |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                             INTERACTION                               |
|   (DwellSelector, RegionProtocol, RegionCorrection, Anti-Midas Logic) |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                              ESTIMATION                               |
|   (CalibrationModel, Ridge Regression closed-form, Normalization)     |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                          TRACKING & HARDWARE                          |
|   (OpenCVCameraProvider, FaceFeatureExtractor, Geometry, Validity)    |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                            CORE & DOMAIN                              |
|   (Protocols, Interfaces, Configs, Domain Value Objects, Enums)      |
+-----------------------------------------------------------------------+
```

### Flujo de Datos End-to-End

```
Camera Hardware (BGR)
       │
       ▼
OpenCVCameraProvider [tracking/camera.py]
       │ (CameraFrame: BGR + Timestamp NS)
       ▼
FaceFeatureExtractor [tracking/landmarker.py] (MediaPipe Face Mesh 478 pts)
       │
       ├─► Geometry [tracking/geometry.py] (EAR, Vectorial Iris Proj, Euler Head Pose)
       ├─► Validity [tracking/validity.py] (Experimental signal-quality rules)
       └─► Placement [tracking/placement.py] (Distance & Bounding Box Assessment)
       │
       ▼ (FeatureObservation)
CalibrationModel [tracking/calibration.py] (Standardized Ridge Linear Mapping)
       │ (Continuous Normalized Gaze X, Y)
       ▼
PredictionSmoother [runners/live_evaluation.py] (EMA Jitter Suppression)
       │ (Smoothed Gaze X, Y)
       ▼
RegionPredictionCorrector [tracking/region_correction.py] (Affine Session Alignment)
       │
       ▼
DwellSelector [tracking/interaction.py] (Hysteresis, Dwell Timer, Refractory Cooldown)
       │
       ▼ (Confirmed Region Selection Event)
AudioInstructionController / SpeechBackend [services/speech.py] (Windows SAPI TTS)
```

---

## 3. Estructura del Repositorio

```text
Mekasa/
├── config/                                 # Archivos de configuración declarativa TOML
│   ├── experiment-001.toml                 # Configuración de validación visual guiada
│   ├── experiment-002.toml                 # Configuración de calibración personal Ridge
│   └── experiment-003.toml                 # Configuración de interacción por 4 cuadrantes
├── engines/
│   └── gaze_engine/                        # Motor principal de visión, mirada e interacción
│       ├── src/
│       │   └── mekasa_gaze/
│       │       ├── core/                   # CAPA DOMAIN / CORE: Contratos e interfaces base
│       │       │   ├── config.py           # Configuración inmutable mediante dataclasses
│       │       │   ├── domain.py           # Entidades de dominio, estados de mirada y enums
│       │       │   ├── interfaces.py       # Protocolos abstractos (CameraProvider, FeatureExtractor, etc.)
│       │       │   ├── protocol.py         # Máquina de estados del protocolo guiado de 9 puntos
│       │       │   └── region_protocol.py  # Protocolo controlado de selección de regiones
│       │       ├── tracking/               # CAPA HARDWARE & TRACKING: Sensores y visión computacional
│       │       │   ├── calibration.py      # Entrenamiento y evaluación de calibración Ridge
│       │       │   ├── camera.py           # Adquisición de video OpenCV (MSMF/DShow/V4L2)
│       │       │   ├── geometry.py         # Matemáticas oculares: EAR, proyección de iris, Euler
│       │       │   ├── interaction.py      # DwellSelector (Permanencia, histéresis y Midas Touch)
│       │       │   ├── landmarker.py       # Wrapper de MediaPipe FaceLandmarker en modo VIDEO
│       │       │   ├── placement.py        # Diagnóstico de distancia y centrado del usuario
│       │       │   ├── probe.py            # Enumeración y prueba de cámaras del sistema
│       │       │   ├── region_correction.py# Corrección afín de sesión para cuadrantes
│       │       │   └── validity.py         # Calificación clínica de fotogramas y parpadeos
│       │       ├── ui/                     # CAPA PRESENTATION: Renderizado gráfico interactivo
│       │       │   └── visualization.py    # Pantallas de carga, telemetría dividida y cuadrantes
│       │       ├── services/               # CAPA SERVICES: Infraestructura y servicios auxiliares
│       │       │   ├── metrics.py          # Métricas de rendimiento (FPS, latencias, RSS, CPU)
│       │       │   ├── reporting.py        # Compilación de reportes Schema 5 en JSON
│       │       │   ├── runtime.py          # Entorno del host, versiones y hash Git
│       │       │   └── speech.py           # Síntesis de voz local por Windows SAPI / PowerShell
│       │       ├── runners/                # CAPA APPLICATION: Orquestadores de experimentos (CLI)
│       │       │   ├── experiment001.py    # Prueba técnica de latencias y extracción
│       │       │   ├── guided_experiment.py# Validación guiada de 9 puntos interactiva
│       │       │   ├── live_evaluation.py  # Evaluación continua de mirada en tiempo real
│       │       │   ├── region_evaluation.py# Evaluación libre de 4 cuadrantes con dwell
│       │       │   └── controlled_region_validation.py # Validación empírica de regiones controlada
│       │       ├── __init__.py              # API pública mínima del paquete
│       │       └── __main__.py              # Entrada predeterminada del motor
│       └── tests/                          # Suite exhaustiva de pruebas unitarias (56 tests)
├── models/                                 # Modelos de aprendizaje automático (descargados localmente)
├── protocols/                              # Definiciones de transporte neutral (Protobuf futuro)
├── scripts/                                # Scripts de utilidad y descarga de modelos
│   └── fetch-face-landmarker.ps1           # Descarga verificada del modelo Face Landmarker
├── pyproject.toml                          # Configuración de dependencias, Ruff, Mypy y Pytest
└── README.md                               # Documentación principal del sistema
```

---

## 4. Instalación y Configuración

### Requisitos del Sistema
* **Sistema Operativo:** Windows 10/11 (recomendado para soporte directo de síntesis de voz SAPI en español), Linux o macOS.
* **Python:** Versión `3.12` (gestionada de forma óptima mediante `uv`).
* **Hardware:** Cámara web integrada o USB de al menos 30 FPS.
* **Administrador de Paquetes:** [`uv`](https://docs.astral.sh/uv/) (recomendado por velocidad y aislamiento reproducible).

### Instalación Paso a Paso

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/usuario/mekasa.git
   cd Mekasa
   ```

2. **Instalar el entorno reproducible con `uv`:**
   ```powershell
   uv python install 3.12
   uv sync --locked
   ```

3. **Descargar el modelo MediaPipe Face Landmarker:**
   El binario oficial de Google MediaPipe (`face_landmarker.task`, ~30 MB) se descarga explícitamente mediante el script provisto para no contaminar el historial Git:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\fetch-face-landmarker.ps1
   ```

4. **Identificar la cámara disponible:**
   Comprueba qué índice de cámara física responde en tu sistema:
   ```powershell
   uv run mekasa-probe-cameras
   ```

### Interfaz accesible por mirada

La pantalla de selección utiliza todo el ancho disponible y oculta la cámara y la telemetría técnica para evitar distracciones. Cada región tiene separación amplia, bordes redondeados, sombra suave y un crecimiento progresivo durante la fijación. El progreso de permanencia se representa simultáneamente dentro de la tarjeta, alrededor del punto de mirada y en una barra inferior. La calibración conserva su panel técnico porque allí sí es necesario diagnosticar distancia, tracking y calidad de señal.

---

## 5. Guía de Ejecución de Pruebas y Linters

El proyecto mantiene una disciplina estricta de calidad, cobertura y tipado estático:

### Ejecución de Pruebas Unitarias
Ejecuta la suite completa de pruebas unitarias, que valida geometría, protocolos, calibración, interacción, reportes y renderizado:
```powershell
uv run pytest
```

### Verificación de Estilo y Linteo con Ruff
Comprueba que el código cumpla con las normas PEP8 y límites de línea de 100 caracteres:
```powershell
uv run ruff check .
```

### Verificación Estática de Tipos con Mypy
Asegura consistencia de contratos con tipado estático estricto en todos los módulos de `gaze_engine`:
```powershell
uv run mypy engines/gaze_engine/src
```

---

## 6. Flujo de Trabajo Experimental (Runners)

El sistema provee una secuencia completa de comandos para calibrar, validar y operar la selección por mirada:

### Paso 1: Validación Visual Guiada (Experimento 001)
Presenta una secuencia determinista y aleatorizada de 9 puntos en pantalla, guiada por audio local en español y control de distancia facial:
```powershell
uv run mekasa-guided-validation
```
* **Controles:** `Esc` o `Q` para cancelar. La prueba se detiene automáticamente si el participante sale del área de captura.
* **Salida:** Reporte estructurado en formato JSON Schema 5 en `research/experiments/001-eye-feature-extraction/results/guided-validation-result.json`.

### Paso 2: Calibración Personal por Regresión Ridge (Experimento 002)
Entrena un modelo lineal regularizado utilizando los fotogramas del reporte guiado anterior, evaluando el desempeño sobre bloques retenidos (*held-out*):
```powershell
uv run mekasa-calibrate
```
* **Salida:** Artefacto JSON `work/calibration/personal-gaze-model.json` con los pesos ajustados y métricas de error angular y en píxeles.

### Paso 3: Evaluación en Vivo en Tiempo Real
Permite al usuario observar la estimación de su punto de mirada sobre los 9 objetivos para evaluar la precisión del modelo entrenado:
```powershell
uv run mekasa-live-evaluation
```

### Paso 4: Selección Libre por Permanencia en 4 Cuadrantes (Experimento 003)
Expone cuatro grandes cuadrantes en pantalla con retroalimentación visual de permanencia (*dwell time*) y períodos de enfriamiento (*cooldown*):
```powershell
uv run mekasa-region-evaluation
```

### Uso rápido con PowerShell

```powershell
.\scripts\mekasa.ps1 instalar               # una sola vez
.\scripts\mekasa.ps1 prueba                 # calibra (~35 s) + 8 consignas: mide el acierto real
.\scripts\mekasa.ps1 tablero                # calibra (~35 s) y abre el tablero
.\scripts\mekasa.ps1 tablero -SinCalibrar   # última calibración + ajuste de 4 puntos (~10 s)
```

Opciones: `-Ojo LEFT|RIGHT`, `-Dwell 1.2`, `-Confirmar`, `-Ventana`, `-Repeticiones 3` (prueba), `-Perfil <etiqueta>` (calibraciones separadas por persona).

Cada sesión queda en `work/communication/history/` y cada calibración en `work/calibration/[profiles/<etiqueta>/]history/`. Son archivos locales y no contienen imágenes ni mensajes.

### Tablero de conversación (armar mensajes completos con la mirada)

Por defecto el tablero ya no se limita a SÍ / NO / AGUA / AYUDA. Cada pantalla tiene 4 casilleros, la grilla 2×2 que tiene precisión medida. Con ellos la persona navega menús, arma un mensaje y lo dice.

| Pantalla | Casilleros |
| --- | --- |
| Inicio | **SÍ** y **NO** (se dicen al instante), **HABLAR** (o **MENSAJE (n)** si hay un mensaje armado), **URGENTE** |
| Hablar | **NECESITO** (comer y beber, mi cuerpo, mi entorno), **CÓMO ESTOY** (me duele, tengo, ánimo), **MÁS…** (preguntas y respuestas, personas, tiempo libre), **VOLVER** |
| Me duele → zona | después pregunta **UN POCO / BASTANTE / MUCHO** (o **SIGUIENTE**) |
| Mensaje | **DECIR** (dice todo el mensaje), **AGREGAR**, **BORRAR ÚLTIMO**, **INICIO** |
| Urgente | Al entrar dice "¡Necesito ayuda urgente! Vengan, por favor." y suena una alarma. Detalle: **NO PUEDO RESPIRAR**, **DOLOR MUY FUERTE**, **ME SIENTO MUY MAL** |

Ejemplo: HABLAR → CÓMO ESTOY → ME DUELE → CABEZA → MUCHO → AGREGAR → NECESITO → COMER Y BEBER → AGUA → DECIR. El tablero dice: "Me duele la cabeza, mucho. Necesito agua."

* **El vocabulario se edita sin programar** en [`config/vocabulario.txt`](config/vocabulario.txt). Cada línea es `ETIQUETA | texto | opciones`; la sangría arma los submenús. Hay como máximo 3 opciones por menú (el cuarto casillero es VOLVER) y `{o|a}` resuelve el género. Al abrir, el tablero valida el archivo y, si hay un error, dice en qué línea está.
* **Género:** `gender = "f"` en `config/communication.toml`, o `-Genero F` en el script.
* **Seguridad:** el casillero recién elegido no puede volver a dispararse hasta que la mirada salga de él. Así, una mirada fija nunca encadena menús.
* **Privacidad:** el reporte cuenta selecciones y mensajes dichos, pero no guarda su contenido.
* **El tablero simple** sigue disponible con `--interface simple` o `-Simple`.

### Cómo se elige: mirada, parpadeos o los dos

Por defecto (`selection_mode = "combined"`) funcionan **los dos a la vez**.

| Gesto | Qué hace |
| --- | --- |
| Mirar una opción **3 s** | La elige |
| **3 parpadeos** (o **3 guiños** con un ojo que la calibración reconoció) **y después mirar** la opción **1 s** | La elige. Mientras parpadea puede mirar a cualquier lado; tiene 5 s para mirar la opción ("Elija") |
| **4 parpadeos** | Empieza o para la lectura (también `ESPACIO`) |
| **5 parpadeos** | Dice ya el mensaje armado, desde cualquier pantalla |
| **7 o más parpadeos** | Llamada de urgencia **en el acto** (voz y alarma), desde cualquier pantalla, aun en pausa o calibrando. Si sigue parpadeando, **la repite** sin esperar (como mucho cada 2,5 s) |

```powershell
.\scripts\mekasa.ps1 tablero              # los dos (mirar 3 s o 3 parpadeos)
.\scripts\mekasa.ps1 tablero -Dwell 2     # mirar 2 s en lugar de 3
.\scripts\mekasa.ps1 tablero -Parpadeo    # solo parpadeos: mirar nunca elige
.\scripts\mekasa.ps1 tablero -SoloTiempo  # solo mirada (0,9 s)
```

* **Un parpadeo** es cerrar los ojos entre 0,08 y 0,8 s: los dos en "ambos ojos", el ojo activo en modo de un ojo.
* **Una serie** son parpadeos separados por menos de 0,8 s con los ojos abiertos. Se reconoce al terminar, así que 3, 4, 5 y 7 no se confunden mientras se cuentan.
* **Se ignoran:**
  * un parpadeo suelto;
  * cantidades sin función (2, 6, 8 o más);
  * cerrar los ojos más de 0,8 s, que además cancela la serie;
  * perder el rostro.
* **Guiños:** al final de la calibración (unos 14 s) el tablero pide cerrar **solo el ojo derecho** 3 veces y después **solo el izquierdo**.
  * Con eso aprende un umbral de cierre para cada ojo y comprueba si la persona puede cerrar uno manteniendo el otro abierto.
  * Los guiños de un ojo se activan solo si se reconocieron al menos 2 de los 3; el resultado aparece en la pantalla de calibración.
  * Como los parpadeos espontáneos son de los dos ojos, un guiño casi nunca ocurre sin querer.
  * Se desactiva con `calibrate_winks = false`.
* **Arriba aparece** "Parpadeos: n" o "Guiños ojo derecho: n" mientras se cuenta.
* **Mientras se cuenta una serie, el temporizador de mirada se congela.** Así, 5 o 7 parpadeos no terminan eligiendo por tiempo la opción que se estaba mirando.
* **Primero parpadear, después mirar.** Al parpadear o guiñar, la mirada estimada salta unos instantes, y mirar fijo mientras se parpadea es difícil. Por eso el orden por defecto es: el gesto abre una ventana de 5 s ("Elija"), y la opción que se mira 1 s dentro de esa ventana se elige. El orden anterior sigue disponible con `blink_select_order = "look_then_blink"`.
* **La mirada se congela 0,3 s al reabrir los ojos** (`blink_settle_seconds`), así un parpadeo no mueve el foco.
* **El suavizado no se reinicia al parpadear.** Descarta los cuadros en que el párpado se cierra rápido (como máximo 0,3 s) y usa la mediana de los últimos cuadros. En la sesión B, el acierto en el segundo posterior a un parpadeo pasó de 82,8 % a 100 % (ver `docs/STATUS.md`).
* **Rango de distancia:** el tablero acepta un rostro de 0,16 a 0,62 del ancho de la imagen (`min_face_width`, `max_face_width`) y compensa la distancia entre 0,65 y 1,55 veces. Solo se reajusta solo con cambios de más del 25 %.
* **Elegir por tiempo exige apartar la mirada** antes de volver a elegir el mismo casillero. Los parpadeos, que son deliberados, pueden elegir la opción nueva aunque esté en el mismo lugar.
* **Todo se ajusta** en `config/communication.toml`: `blink_select_count`, `blink_toggle_count`, `blink_speak_count`, `blink_urgent_count`, `combined_dwell_seconds` y `blink_*`. Los tiempos son hipótesis de ingeniería.

### Tablero de comunicación (opciones comunes)

```powershell
uv run mekasa-board                       # calibra y abre el tablero a pantalla completa
uv run mekasa-board --test 2              # calibra, prueba con consignas, luego tablero
uv run mekasa-board --no-calibrate        # reutiliza work/calibration/session-gaze-model-*.json
uv run mekasa-board --eye-mode LEFT       # BOTH | LEFT | RIGHT (lado anatómico)
uv run mekasa-board --simulate            # replay sin cámara de todos los bloqueos de seguridad
```

* **Calibración al inicio (~35 s).** Una calibración de otra sesión no sirve: en los datos del 2026-11-16 el acierto 2×2 cayó de 96–100 % dentro de la sesión a 26 % entre sesiones. La calibración muestra 9 puntos dentro del tablero, luego el punto central mientras la persona mueve despacio la cabeza, y repite los 9 puntos en otro orden. Entrena un modelo para cada modo ocular y muestra el acierto validado (arriba y abajo por separado). `R` la repite.
* **Qué aprende el modelo.** Por cada modo ocular compara tres variantes con validación cruzada entre rondas y se queda con la mejor:
  * puntos del rostro (iris, apertura del ojo, pose de la cabeza);
  * imagen del ojo: parches de 20×10 píxeles alineados a las comisuras, que captan el párpado y la esclera, donde está la señal vertical;
  * ambas combinadas.
* **Umbral personal de ojo cerrado.** Se calcula de la propia mirada hacia abajo, porque el párpado baja al mirar abajo.
* **Mientras se usa:**
  * compensa la distancia a la cámara con el ancho del rostro;
  * avisa si la persona se acercó, se alejó, se corrió o giró;
  * si el cambio dura más de 2,5 s, hace solo un ajuste rápido de 4 puntos;
  * aprende un poco de cada selección hecha con la mirada; `C`, dentro de los 5 s, la borra y la descarta.
* **Configuración:** `config/communication.toml`. Contiene dwell, enfriamiento, recuperación, edad máxima de frame, confirmación, frases, voz y las opciones de precisión (`calibrate_at_start`, `auto_realign`, `implicit_recalibration`, `distance_compensation`, `camera_position`).
* **Acceso directo:** las teclas `1`–`4` y el clic del mouse funcionan siempre, aun sin calibración.
* **Otros controles:** `RETROCESO` vuelve un nivel, `H` va al inicio, `ESPACIO` pausa, `A` hace un ajuste rápido, `K` recalibra, `ENTER` confirma la opción pendiente, `M` cambia el modo ocular, `G` muestra el punto de mirada, `C` borra el mensaje y `ESC` sale.
* **Parpadeos:** caídas de señal ocular de hasta 0,2 s congelan el progreso en lugar de cancelarlo; las pérdidas de cámara, imagen o rostro cancelan siempre.
* **Bloqueos de seguridad:** una cámara desconectada, un frame vencido, tracking inválido, el ojo requerido no visible, la pausa o un cambio de modo vacían el foco y la confirmación pendiente.
* **Si falla la voz:** el mensaje confirmado queda visible.
* **Privacidad:**
  * Las imágenes del ojo solo existen en memoria.
  * El modelo guardado contiene el promedio de esas imágenes, un ojo de 20×10 píxeles; es un archivo local y se borra borrando `work/calibration/session-gaze-model-*.json`.
  * Los reportes no guardan frames, parches, landmarks ni historial de mensajes.

Los runners de investigación (pasos 1–5) siguen disponibles para estudios. El uso diario pasa por el tablero.

Estado actual, mediciones y próximos pasos: [`docs/STATUS.md`](docs/STATUS.md). Evidencia y compatibilidad: [`research/EVIDENCE.md`](research/EVIDENCE.md) y [`research/COMPATIBILITY.md`](research/COMPATIBILITY.md).

### Paso 5: Validación Controlada por Regiones
Evalúa empíricamente la tasa de aciertos y latencias pidiendo al usuario fijar la mirada en regiones solicitadas aleatoriamente:
```powershell
uv run mekasa-controlled-region-validation
```

---

## 7. Fundamentos Matemáticos y Componentes del Núcleo

### Geometría Oculomotora e Invarianza de Escala
* **Eye Aspect Ratio (EAR):** Implementado siguiendo la formulación de Soukupová & Čech (2016):
  $$\text{EAR} = \frac{\|\mathbf{p}_2 - \mathbf{p}_6\| + \|\mathbf{p}_3 - \mathbf{p}_5\|}{2 \|\mathbf{p}_1 - \mathbf{p}_4\|}$$
  donde $\mathbf{p}_1 \dots \mathbf{p}_6$ son las coordenadas 2D de los párpados. Permite discriminar aperturas palpebrales normales de cierres oculares y parpadeos breves.
* **Proyección de Iris Normalizada:** Para lograr invarianza frente a movimientos anteroposteriores y escala facial, el centroide del iris se proyecta ortogonalmente sobre el vector director del ojo (desde el canto externo al interno) y el vector de apertura vertical palpebral:
  $$x_{\text{norm}} = \frac{(\mathbf{iris} - \mathbf{start}) \cdot (\mathbf{end} - \mathbf{start})}{\|\mathbf{end} - \mathbf{start}\|^2}$$
* **Pose Cefálica (Euler):** A partir de la matriz de transformación 4x4 provista por MediaPipe, se extraen los ángulos de orientación de la cabeza en grados (guiñada/*yaw*, cabeceo/*pitch*, balanceo/*roll*).

### Calificación Técnica de la Señal
Cada fotograma se clasifica de forma determinista mediante heurísticas experimentales; no son umbrales clínicos:
1. `TRACKING_LOST`: Si la cantidad de rostros detectados no es exactamente 1.
2. `TRACKING_UNCERTAIN`: Si la distancia o geometría de los landmarks es aberrante o inestable.
3. `GAZE_VALID`: Ambos ojos abiertos (EAR por encima del umbral) con iris en rango fisiológico.
4. `GAZE_PARTIAL`: Un ojo utilizable (se imputa la media del conjunto de entrenamiento en el ojo cerrado para no falsear la inferencia binocular).
5. `BLINK`: Secuencias cortas de cierre palpebral acotadas por fotogramas válidos ($\le 350\text{ ms}$).

### Calibración por Regresión Ridge (Tikhonov)
El mapeo de las características oculares y de cabeza hacia coordenadas de pantalla $(x, y)$ se formula como un problema de regresión multivariada regularizada:
$$\mathbf{W} = \left( \mathbf{X}_{\text{norm}}^T \mathbf{X}_{\text{norm}} + \alpha \mathbf{I} \right)^{-1} \mathbf{X}_{\text{norm}}^T \mathbf{Y}_{\text{centered}}$$
donde $\mathbf{X}_{\text{norm}}$ son los vectores de características estandarizados mediante Z-score y $\alpha$ es el hiperparámetro de penalización Ridge.

### Mitigación del Midas Touch en Interacción
Para prevenir selecciones involuntarias mientras el usuario explora visualmente la interfaz:
* **Permanencia Temporal (*Dwell Time*):** Requiere sostener la fijación continua en la región durante un lapso configurable (típicamente $700\text{ ms} - 1000\text{ ms}$).
* **Margen de Histéresis Espacial:** Expande dinámicamente los bordes del cuadrante activo ($\sim 5\% - 10\%$) para evitar fluctuaciones en los límites de cada botón.
* **Período Refractario (*Cooldown*):** Tras una selección confirmada, el sistema entra en una pausa de seguridad ($1.0\text{ s} - 1.5\text{ s}$) que ignora nuevas activaciones inmediatas.
* **Cancelación Inmediata:** Si el tracking se pierde o la validez de la mirada se degrada, el progreso de permanencia se resetea a cero de inmediato; nunca completa una selección.

---

## 8. Roadmap de Integración IA / LLM (Mekasa Intelligence)

Mekasa está estructurado arquitectónicamente para incorporar en sus próximas fases un **Motor de Inferencia de Lenguaje Local (SLM / LLM on-device)** que transforme la selección elemental de regiones en una experiencia comunicativa rica y de mínimo esfuerzo físico.

```text
[ Gaze Selection: 4 Cuadrantes ] ──► [ Intent Buffer ]
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │    Mekasa LLM Engine (GGUF / llama.cpp)       │
                    │   • Gemma-2B / Llama-3.2-1B / Qwen-1.5B       │
                    │   • Adaptación al contexto del paciente       │
                    │   • Árbol de decisiones dinámico              │
                    └───────────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
     [ Tablero Dinámico Predictivo ]                 [ Síntesis Neural Local ]
 (Ofrece las 4 opciones más probables)           (Audio natural expresivo offline)
```

### Líneas de Desarrollo en el Roadmap

1. **Modelos Pequeños Cuantizados (SLM On-Device):**
   * Integración de modelos de 1B a 3B parámetros (ej. Gemma-2B, Llama-3.2-1B o SmolLM2) ejecutados en cuantización de 4 bits (`Q4_K_M`) a través de un motor de bajo consumo en C++/Python (como `llama.cpp` o `MLC-LLM`).
   * Ejecución local sin enviar conversaciones a servicios externos. Rendimiento y latencia deberán medirse en el hardware objetivo antes de elegir un modelo.

2. **Minimización del Esfuerzo Motor por Intención (*Effort Per Intent*):**
   * En pacientes con fatiga ocular severa post-ACV, cada selección de mirada cuesta energía física.
   * El LLM re-ordenará dinámicamente los cuadrantes de comunicación según el contexto: hora del día (mañana: desayuno, medicación; noche: descanso, luz), historial reciente y estado fisiológico reportado.

3. **Expansión Semántica de Frases:**
   * A partir de secuencias mínimas de selección (ej. `[DOLOR]` $\rightarrow$ `[CABEZA]` $\rightarrow$ `[FUERTE]`), el motor generará oraciones completas y moduladas: *"Tengo un dolor de cabeza muy fuerte, ¿podrían alcanzarme mi medicación?"*.

4. **Síntesis Vocal Neural Ultra-Rápida:**
   * Sustitución paulatina de Windows SAPI por sintetizadores neuronales locales de peso pluma (ej. modelos VITS o Piper TTS en español), otorgando voz natural, prosodia adecuada e identidad al usuario sin telemetría.

---

## 9. Aviso Clínico, Ético y Seguridad

> [!IMPORTANT]
> **Aviso de Dispositivo No Médico:**  
> Mekasa es un prototipo experimental de investigación en tecnología asistiva. **No es un dispositivo médico homologado**, no cuenta con certificación de agencias sanitarias (FDA, CE, ANMAT) y **no está diseñado para el diagnóstico, tratamiento ni monitorización de ninguna patología clínica**.

* **Ámbito de Aplicación Estricto:** Mekasa V0 tiene como único propósito explorar la viabilidad de la comunicación aumentativa. **Está estrictamente prohibido utilizar Mekasa para controlar equipos de soporte vital, sistemas de llamado de emergencia críticos, sillas de ruedas motorizadas, cerraduras de seguridad o interfaces bancarias/financieras.**
* **Validación con Profesionales:** Ninguna métrica técnica de laboratorio (como la exactitud en píxeles o el porcentaje de fotogramas válidos) sustituye la evaluación clínica individualizada conducida por fonoaudiólogos/logopedas, terapeutas ocupacionales y médicos fisiatras (PM&R).

---

## 10. Licencia

El código fuente del proyecto Mekasa se distribuye bajo la licencia de código abierto **Apache License 2.0**. Consulta el archivo `LICENSE` para conocer los términos completos.

*Los modelos pre-entrenados de terceros (como MediaPipe Face Landmarker de Google LLC) están sujetos a sus propios términos y licencias de distribución.*
