#!/bin/bash
set -e

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    log_info "Shutting down services..."

    # Kill Python ML API if running
    if [ ! -z "$ML_API_PID" ]; then
        log_info "Stopping ML API (PID: $ML_API_PID)..."
        kill -TERM "$ML_API_PID" 2>/dev/null || true
        wait "$ML_API_PID" 2>/dev/null || true
    fi

    # Kill .NET API if running
    if [ ! -z "$DOTNET_PID" ]; then
        log_info "Stopping .NET API (PID: $DOTNET_PID)..."
        kill -TERM "$DOTNET_PID" 2>/dev/null || true
        wait "$DOTNET_PID" 2>/dev/null || true
    fi

    log_info "Shutdown complete"
    exit 0
}

# Trap SIGTERM and SIGINT
trap cleanup SIGTERM SIGINT

log_info "Starting CryptoArbitrage API services..."
log_info "Environment: ${ASPNETCORE_ENVIRONMENT:-Production}"

# Change to ML pipeline directory and start Python ML API in background
cd /app/ml_pipeline
log_info "Starting ML API server on port 5250..."
python3 ml_api_server.py > /tmp/ml_api.log 2>&1 &
ML_API_PID=$!
log_info "ML API started with PID: $ML_API_PID"

# Wait for ML API to be ready (max 30 seconds)
log_info "Waiting for ML API to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -f http://localhost:5250/health > /dev/null 2>&1; then
        log_info "✅ ML API is ready!"
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        log_error "ML API failed to start within 30 seconds"
        log_error "ML API logs:"
        cat /tmp/ml_api.log
        cleanup
        exit 1
    fi

    sleep 1
done

# Start .NET API in foreground
cd /app
log_info "Starting .NET API on port 8080..."
dotnet CryptoArbitrage.API.dll &
DOTNET_PID=$!
log_info ".NET API started with PID: $DOTNET_PID"

log_info "========================================="
log_info "Both services are running:"
log_info "  - ML API:   http://localhost:5250"
log_info "  - .NET API: http://localhost:8080"
log_info "========================================="

# Wait for .NET API to exit
wait "$DOTNET_PID"
DOTNET_EXIT_CODE=$?

log_warn ".NET API exited with code: $DOTNET_EXIT_CODE"
cleanup
exit $DOTNET_EXIT_CODE
