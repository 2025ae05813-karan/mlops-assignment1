# ---------------------------------------------------------------------------
# Heart Disease Risk Prediction API — container image (Assignment Task 6)
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps kept minimal; curl is used by the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source.
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Build the dataset and train the model at image-build time so the container
# is fully self-contained and reproducible (no external artifacts needed).
RUN python -m src.data_loader --sites cleveland \
    && python -m src.train

# Run as a non-root user for security.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-level health check hitting the API's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
