# QueAI-STT-CPU-LOCAL-MS

`QueAI-STT-CPU-LOCAL-MS` es un módulo descargable para [QueAI](https://queai.dev/) que permite convertir audio y video en texto de forma local, ejecutando transcripción en CPU mediante `faster-whisper`.

Está diseñado para integrarse como plugin dentro del ecosistema QueAI, manteniendo una arquitectura desacoplada, portable y fácil de desplegar con Docker.

---

## Características

- Transcripción local en CPU.
- Basado en `faster-whisper`.
- Cuantización `int8` por defecto para reducir consumo de RAM.
- Cola de tareas asincrónica.
- Progreso en vivo mediante SSE.
- Eventos por segmento durante la transcripción.
- Soporte para archivos:
  - `.wav`
  - `.mp3`
  - `.m4a`
  - `.ogg`
  - `.webm`
  - `.mp4`
  - `.mpeg`
  - `.mpga`
  - `.flac`
- Normalización automática con FFmpeg.
- Resultados descargables en:
  - `.txt`
  - `.json`
  - `.srt`
  - `.vtt`
- Grabación directa desde navegador.
- Interfaz moderna y pensada para usuarios no técnicos.

---

## Modelos disponibles

El módulo acepta modelos compatibles con `faster-whisper`, por ejemplo:

- `tiny`
- `base`
- `small`
- `medium`
- `large-v3`
- `distil-small.en`
- `distil-medium.en`
- `distil-large-v3`

Para CPU local equilibrado se recomienda empezar con:

```txt
base
```

Para mayor calidad, pero más consumo:

```txt
small
```

Para equipos modestos:

```txt
tiny
```

---

## Variables de entorno

```env
STT_DEFAULT_MODEL=base
STT_DEVICE=cpu
STT_COMPUTE_TYPE=int8
STT_WORKERS=1
STT_CPU_THREADS=4
STT_NUM_WORKERS=1
STT_MAX_UPLOAD_MB=500
```

---

## Endpoints principales

```txt
GET    /api/stt_local_cpu/health
GET    /api/stt_local_cpu/config
GET    /api/stt_local_cpu/models
POST   /api/stt_local_cpu/transcribe
GET    /api/stt_local_cpu/jobs/{job_id}
GET    /api/stt_local_cpu/jobs/{job_id}/events
GET    /api/stt_local_cpu/jobs/{job_id}/result.txt
GET    /api/stt_local_cpu/jobs/{job_id}/result.json
GET    /api/stt_local_cpu/jobs/{job_id}/result.srt
GET    /api/stt_local_cpu/jobs/{job_id}/result.vtt
DELETE /api/stt_local_cpu/jobs
```

---

## Arquitectura

1. El usuario sube un audio/video o graba desde el navegador.
2. El backend valida el archivo.
3. Se crea un job en cola.
4. El worker normaliza el audio a WAV mono 16 kHz.
5. `faster-whisper` transcribe localmente en CPU.
6. El progreso se transmite mediante SSE.
7. El resultado se guarda como TXT, JSON, SRT y VTT.
8. La UI permite visualizar y descargar la transcripción.

---

## Despliegue

```bash
docker compose up -d --build
```

---

## Licencia
MIT


---
