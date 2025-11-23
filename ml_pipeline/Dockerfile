# ML API Dockerfile - RL Server
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal requirements for API server (not training dependencies)
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    find /usr/local/lib/python3.11 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Copy application code (new structure)
COPY server/ ./server/
COPY common/ ./common/
COPY models/rl/ ./models/rl/
COPY trained_models/rl/ ./trained_models/rl/
COPY checkpoints/v3-fixed-entropy/ ./checkpoints/v3-fixed-entropy/

# Verify RL models exist and are not LFS pointers
RUN ls -lh checkpoints/v3-fixed-entropy/checkpoint_ep350.pt && \
    ls -lh trained_models/rl/feature_scaler_v2.pkl && \
    if [ $(stat -c%s checkpoints/v3-fixed-entropy/checkpoint_ep350.pt 2>/dev/null || stat -f%z checkpoints/v3-fixed-entropy/checkpoint_ep350.pt 2>/dev/null || echo 0) -lt 1000000 ]; then \
        echo "ERROR: Model files appear to be LFS pointers, not actual files!"; \
        exit 1; \
    fi

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser && \
    chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 5250

# Health check - disabled in favor of docker-compose healthcheck
# which has proper timeouts for slow model loading
# HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
#     CMD curl -f http://localhost:5250/health || exit 1

# Run the ML API server (new location)
CMD ["python", "server/app.py"]
