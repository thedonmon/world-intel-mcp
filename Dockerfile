FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORLD_INTEL_CACHE_DB=/data/cache.db

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[http]" \
    && useradd --create-home --uid 10001 appuser \
    && mkdir /data \
    && chown appuser:appuser /data

USER appuser
EXPOSE 8000
VOLUME ["/data"]
ENTRYPOINT ["world-intel-mcp"]
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
