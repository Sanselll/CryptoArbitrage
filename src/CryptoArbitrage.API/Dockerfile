# Multi-stage Dockerfile for .NET 8 Backend with ML Pipeline
# NOTE: Build context must be the project root, not src/CryptoArbitrage.API
# Usage: docker build -f src/CryptoArbitrage.API/Dockerfile -t crypto-arbitrage-api .

# Stage 1: Build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy csproj and restore dependencies
COPY ["src/CryptoArbitrage.API/CryptoArbitrage.API.csproj", "./"]
RUN dotnet restore "CryptoArbitrage.API.csproj"

# Copy everything else and build
COPY ["src/CryptoArbitrage.API/", "./"]
RUN dotnet build "CryptoArbitrage.API.csproj" -c Release -o /app/build

# Stage 2: Publish
FROM build AS publish
RUN dotnet publish "CryptoArbitrage.API.csproj" -c Release -o /app/publish /p:UseAppHost=false

# Stage 3: Runtime
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy published .NET files
COPY --from=publish /app/publish .

# Copy ML pipeline files
COPY --chown=appuser:appuser ml_pipeline/requirements.txt /app/ml_pipeline/
COPY --chown=appuser:appuser ml_pipeline/ml_api_server.py /app/ml_pipeline/
COPY --chown=appuser:appuser ml_pipeline/src /app/ml_pipeline/src/
COPY --chown=appuser:appuser ml_pipeline/models /app/ml_pipeline/models/

# Install Python dependencies (use --break-system-packages since we're in a container)
RUN pip3 install --no-cache-dir --break-system-packages -r /app/ml_pipeline/requirements.txt

# Copy entrypoint script
COPY --chown=appuser:appuser src/CryptoArbitrage.API/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports (8080 for .NET API, 5250 only internal for ML API)
EXPOSE 8080

# Health check for .NET API
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV ASPNETCORE_URLS=http://+:8080
ENV ASPNETCORE_ENVIRONMENT=Production

# Use entrypoint script to start both services
ENTRYPOINT ["/app/entrypoint.sh"]
