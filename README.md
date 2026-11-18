# Mekasa

**Hablar con los ojos, con una cámara común y sin internet.**

> *Mekasa* viene del **twi**, una lengua de Ghana, y significa **«hablarás»**.

Mekasa es un programa gratuito y de código abierto para Windows. Permite que una persona que no puede hablar ni moverse se comunique **mirando la pantalla** o **parpadeando**: elige frases en un tablero y la computadora las dice en voz alta.

---

## Descargar e instalar

1. Entrá a **[Releases → última versión](https://github.com/Kkwafo/Mekasa/releases/latest)**.
2. Descargá **`Mekasa-<versión>-setup.exe`** y abrilo. No pide permisos de administrador.
3. Si aparece *«Windows protegió su PC»*, tocá **Más información → Ejecutar de todos modos**. Aparece porque el programa todavía no tiene firma digital.
4. En el menú Inicio queda **Mekasa (tablero)**.

¿Preferís no instalar nada? Descargá **`Mekasa-<versión>-windows-x64-portable.zip`**, descomprimilo y abrí `Mekasa.exe`.

**Se necesita:** Windows 10 u 11 de 64 bits y una cámara web (la de la notebook alcanza). **Internet: no.** Todo funciona y queda en la computadora.

> En PCs con el **Control inteligente de aplicaciones** de Windows 11 activado, el programa puede quedar bloqueado hasta que tenga firma digital.

---

## Cómo se usa

1. **Calibración (~35 s).** Al abrir, la persona mira unos puntos en la pantalla. Así Mekasa aprende cómo se mueven *sus* ojos ese día y en esa postura.
2. **Tablero.** Cada pantalla tiene 4 casilleros grandes: **SÍ**, **NO**, **HABLAR** y **URGENTE**. Desde HABLAR se arman mensajes completos, como *«Me duele la cabeza, mucho. Necesito agua.»*.
3. **Para elegir** un casillero hay dos formas:
   - mirarlo **3 segundos**, o
   - **parpadear 3 veces** y después mirarlo **1 segundo**.

| Parpadeos seguidos | Qué hace |
| --- | --- |
| 3 y después mirar | Elige esa opción |
| 4 | Pausa o reanuda |
| 5 | Dice el mensaje armado |
| **7 o más** | **Pide ayuda urgente** con voz y alarma, desde cualquier pantalla |

**Para la familia o el acompañante:** las teclas `1`–`4` y el mouse también eligen. `ESPACIO` pausa, `K` recalibra y `ESC` sale.

**Para mostrar o filmar cómo funciona:** abrí **Mekasa - demostración (con cámara)** desde el menú Inicio (o tocá `V` en el tablero). Aparece la imagen de la cámara, como un espejo, con los puntos de los ojos que sigue Mekasa: borde verde si ve bien el rostro, rojo si no. `V` cambia entre chica arriba a la derecha, grande al centro y apagada. La imagen solo se muestra: no se graba.

**Mekasa aprende con el uso.** Cada vez que la persona elige algo y no lo corrige, el modelo se ajusta un poco más a ella, y ese aprendizaje se guarda al cerrar.

### Cambiar las frases

Desde el menú Inicio, **Mekasa → datos y vocabulario** abre la carpeta de datos. Ahí se edita `config\vocabulario.txt` con el Bloc de notas, sin programar, para agregar personas, comidas o frases propias. Los cambios no se pierden al actualizar.

En esa misma carpeta (`%LOCALAPPDATA%\Mekasa`) quedan las calibraciones, los reportes de uso y los registros. Mekasa les reserva un espacio fijo, de 512 MB por defecto, y borra lo más viejo cuando se llena. **No guarda video, fotos ni el contenido de los mensajes.**

---

## La historia

Después de un ACV, una persona puede perder de golpe el habla y el movimiento, y quedar consciente pero sin forma de decir que tiene sed, que algo le duele o que necesita ayuda. Los sistemas comerciales que leen la mirada existen, pero suelen costar miles de dólares, usar cámaras especiales o enviar datos a la nube.

Mekasa empezó con una pregunta: **¿alcanza una cámara web común para que alguien vuelva a hablar?**

- Primero se midió qué tan bien se pueden seguir los ojos con una cámara común a 30 cuadros por segundo.
- Después vino la calibración personal y un tablero de 4 casilleros, la grilla con la que la precisión medida fue mejor.
- Las sesiones con una persona real marcaron cada cambio. La fila de abajo no se podía elegir, porque al mirar hacia abajo el párpado tapa el ojo. Una calibración de otro día no servía. Hubo que sumar **parpadeos y guiños**, porque mantener la mirada fija cansa.
- Así llegó la versión 1.0: un programa que se instala con dos clics, funciona sin internet y sigue aprendiendo de quien lo usa.

Cada decisión, con sus mediciones (lo que funcionó y lo que no), está en [`docs/STATUS.md`](docs/STATUS.md).

---

## Importante

Mekasa es un **prototipo de investigación**. **No es un dispositivo médico** y no reemplaza la evaluación de fonoaudiólogos, terapeutas ni médicos. No lo uses como único medio para pedir ayuda en una emergencia ni para controlar equipos de soporte vital.

---

## Para desarrolladores

- Documentación técnica completa (arquitectura, calibración, parámetros, experimentos): [`docs/TECNICO.md`](docs/TECNICO.md)
- Correr desde el código: `uv sync` y después `uv run mekasa` (tablero) o `uv run mekasa simular` (sin cámara)
- Generar el instalador: `.\scripts\build-release.ps1`
- Publicar una versión: subir `version` en `pyproject.toml` y llevarla a `main`; GitHub construye y publica el release solo.

## Licencia

Código bajo [Apache 2.0](LICENSE). El modelo de rostro de MediaPipe (Google) tiene sus propios términos.
