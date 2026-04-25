FROM ghcr.io/queai-project/queai-stt-base:sha-5202ed5

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/code/models/.hf_cache
ENV TRANSFORMERS_CACHE=/code/models/.hf_cache
ENV XDG_CACHE_HOME=/code/models/.cache

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

# Guardamos una copia del modelo que viene dentro de la imagen base.
# Esto es importante porque el volumen /code/models oculta el contenido original de la imagen.
RUN mkdir -p /opt/queai-stt-preloaded-models \
    && cp -a /code/models/. /opt/queai-stt-preloaded-models/

COPY ./app /code/app
COPY ./frontend_dist /code/frontend_dist
COPY ./manifest.json /code/manifest.json
COPY ./README.md /code/README.md
COPY ./LICENSE /code/LICENSE
COPY ./entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && mkdir -p /code/runtime/stt_jobs /code/logs /code/models

ENTRYPOINT ["/entrypoint.sh"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]