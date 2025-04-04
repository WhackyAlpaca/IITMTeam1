FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# Required for Google Cloud Run
ENV PORT=8080
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Create directories
RUN mkdir -p /app/models

# Copy application
COPY . .

# Expose port (must match $PORT)
EXPOSE 8080

# Command for Cloud Run
CMD exec uvicorn combined:app --host 0.0.0.0 --port ${PORT}
