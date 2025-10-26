# ML API Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for API server (not training dependencies)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt && \
    rm -rf /root/.cache/pip && \
    find /usr/local/lib/python3.11 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY ml_api_server.py .
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser && \
    chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 5250

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5250/health || exit 1

# Run the ML API server
CMD ["python", "ml_api_server.py"]
