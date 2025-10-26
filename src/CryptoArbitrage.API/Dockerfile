# Multi-stage Dockerfile for .NET 8 Backend
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

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy published .NET files
COPY --from=publish /app/publish .

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check for .NET API
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV ASPNETCORE_URLS=http://+:8080
ENV ASPNETCORE_ENVIRONMENT=Production

# Run the .NET application
ENTRYPOINT ["dotnet", "CryptoArbitrage.API.dll"]
