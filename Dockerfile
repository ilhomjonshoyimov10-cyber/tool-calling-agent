FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agent ./agent
RUN pip install .

RUN useradd --create-home --uid 1000 agent && mkdir -p /app/data && chown -R agent /app
USER agent

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
