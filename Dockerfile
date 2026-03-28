# -----------------------------------------------------------------------
# Customer Support Triage — OpenEnv
# Base: Python 3.11 slim for small image size
# Port: 7860 (Hugging Face Spaces default)
# -----------------------------------------------------------------------

FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="Customer Support Triage — OpenEnv"
LABEL org.opencontainers.image.description="OpenEnv environment for AI agent customer support training"
LABEL org.opencontainers.image.version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY environment/ ./environment/
COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Environment variables with defaults
ENV PORT=7860
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

CMD ["python", "app.py"]
