#!/usr/bin/env sh
set -eu

MODELS_DIR="/code/models"
PRELOADED_DIR="/opt/queai-stt-preloaded-models"
PREPARED_DIR="$MODELS_DIR/.prepared"

mkdir -p "$MODELS_DIR" "$PREPARED_DIR" /code/runtime/stt_jobs /code/logs

if [ -d "$PRELOADED_DIR" ]; then
    if [ ! -f "$PREPARED_DIR/tiny.json" ] && [ ! -f "$PREPARED_DIR/tiny.json.json" ]; then
        echo "[QueAI-STT] Inicializando modelo base preinstalado en el volumen..."
        cp -a "$PRELOADED_DIR"/. "$MODELS_DIR"/
        echo "[QueAI-STT] Modelo base listo."
    else
        echo "[QueAI-STT] Modelo base ya existe en el volumen."
    fi
fi

exec "$@"