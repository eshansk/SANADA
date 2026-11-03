# Multi-stage production Dockerfile for SANADA Machine Listening API
# Member 5: Data Engineering & Backend API (Week 4 Containerization)

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies for audio decoding (libsndfile & ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source modules
COPY backend/ ./backend/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Ensure static directories and checkpoint folders exist
RUN mkdir -p /app/backend/static/stems /app/models/checkpoints

# Pre-cache model checkpoints
RUN python scripts/download_checkpoints.py

EXPOSE 8000

# Health check to ensure service vitality
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Launch ASGI production server
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
