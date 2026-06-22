FROM python:3.13-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

FROM base AS runtime

ENV PYTHONUNBUFFERED=1

EXPOSE 4000

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:4000/health').raise_for_status()"

CMD ["uv", "run", "uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "4000"]
