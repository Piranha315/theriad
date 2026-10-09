FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY theriad ./theriad
COPY --chown=10001:10001 router-assets /opt/theriad-router-assets
RUN pip install --no-cache-dir . && useradd -m -u 10001 gateway
USER gateway
ENV PORT=8080 THERIAD_GATEWAY_STORE=gcp THERIAD_DISABLE_LIVE_PRICING=1 THERIAD_GATEWAY_ROUTER_ASSETS=/opt/theriad-router-assets
CMD ["sh", "-c", "exec uvicorn theriad.gateway.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
