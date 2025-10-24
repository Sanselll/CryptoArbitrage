# Environment Configuration Guide

This document describes how to run the CryptoArbitrage platform in different environments (Demo vs Live).

## Overview

The platform supports two operational modes:

- **Demo Mode**: Uses exchange testnet/demo trading endpoints with paper trading (no real money)
- **Live Mode**: Uses live exchange endpoints with real trading (real money at risk)

The environment is configured at the **application level** (not per-user), meaning all users connecting to a running instance share the same environment mode.

## Architecture

### Configuration Files

The platform uses explicit, purpose-specific configuration files:

1. **appsettings.json** - Base configuration shared across all environments (logging, authentication, JWT, arbitrage config)
2. **appsettings.DevelopmentDemo.json** - Local development in Demo mode
3. **appsettings.DevelopmentLive.json** - Local development in Live mode
4. **appsettings.ProductionDemo.json** - Docker/Cloud Demo mode
5. **appsettings.ProductionLive.json** - Docker/Cloud Live mode

Each environment-specific file defines:
- `Environment.IsLive` - Boolean flag (false for Demo, true for Live)
- `Environment.Mode` - Display name ("Demo" or "Live")
- `ConnectionStrings.DefaultConnection` - Database connection string

### Database Separation

Each environment uses its own PostgreSQL database:
- **Demo**: `crypto_arbitrage_demo`
- **Live**: `crypto_arbitrage_live`

This ensures complete data isolation between environments.

### Exchange Connections

The platform automatically connects to appropriate exchange endpoints based on environment:

**Demo Mode:**
- Binance: Testnet endpoint
- Bybit: Demo Trading endpoint

**Live Mode:**
- Binance: Live production endpoint
- Bybit: Live production endpoint

## Running Different Environments

### Local Development (dotnet run)

The standard way to run the application locally for development.

#### Run Demo Mode

```bash
cd src/CryptoArbitrage.API

# Use DevelopmentDemo configuration
dotnet run --environment DevelopmentDemo
```

The application will:
- Load `appsettings.DevelopmentDemo.json`
- Connect to `crypto_arbitrage_demo` database
- Use exchange testnet/demo endpoints
- Run on default port (http://localhost:5000)

#### Run Live Mode

```bash
cd src/CryptoArbitrage.API

# Use DevelopmentLive configuration
dotnet run --environment DevelopmentLive
```

The application will:
- Load `appsettings.DevelopmentLive.json`
- Connect to `crypto_arbitrage_live` database
- Use exchange production endpoints
- Run on default port (http://localhost:5000)

### Docker Deployment (Recommended for Production)

Docker Compose is configured to run **both Live and Demo modes simultaneously** on different ports.

#### Docker Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Services                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                                           │
│  │  postgres    │  (Shared PostgreSQL Container)            │
│  │              │  • crypto_arbitrage_live                  │
│  │  Port: 5432  │  • crypto_arbitrage_demo                  │
│  └──────────────┘                                           │
│                                                              │
│  ┌──────────────┐           ┌──────────────┐               │
│  │ backend-live │           │ backend-demo │               │
│  │              │           │              │               │
│  │  Port: 8080  │           │  Port: 8081  │               │
│  │  Env:        │           │  Env:        │               │
│  │  Production  │           │  Production  │               │
│  │  Live        │           │  Demo        │               │
│  └──────────────┘           └──────────────┘               │
│                                                              │
│  ┌──────────────┐           ┌──────────────┐               │
│  │frontend-live │           │frontend-demo │               │
│  │              │           │              │               │
│  │  Port: 80    │           │  Port: 81    │               │
│  │  API: 8080   │           │  API: 8081   │               │
│  └──────────────┘           └──────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Start All Services (Both Live and Demo)

```bash
# From project root directory
docker compose up -d
```

This starts:
- 1 PostgreSQL container (with both databases)
- 2 backend containers (Live on 8080, Demo on 8081)
- 2 frontend containers (Live on 80, Demo on 81)

#### Start Only Live Mode

```bash
docker compose up -d postgres backend-live frontend-live
```

#### Start Only Demo Mode

```bash
docker compose up -d postgres backend-demo frontend-demo
```

#### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend-live
docker compose logs -f backend-demo
docker compose logs -f postgres
```

#### Stop Services

```bash
# Stop all services (preserves database data)
docker compose down

# Stop and remove all data (WARNING: deletes databases!)
docker compose down -v
```

#### Access the Applications

Once running, access the frontends at:

- **Live Mode**: http://localhost (frontend) and http://localhost:8080/api (backend API)
- **Demo Mode**: http://localhost:81 (frontend) and http://localhost:8081/api (backend API)

## Database Setup

### Local Development Databases

Before running locally, create the required databases:

```bash
# For Demo environment
psql -U postgres -c "CREATE DATABASE crypto_arbitrage_demo;"

# For Live environment
psql -U postgres -c "CREATE DATABASE crypto_arbitrage_live;"
```

### Docker Databases

Databases are **automatically created** by the `docker-init-db.sh` script when the PostgreSQL container first starts. No manual setup needed.

The script creates:
- `crypto_arbitrage_live`
- `crypto_arbitrage_demo`

### Automatic Migrations

The application automatically applies EF Core migrations on startup, so you don't need to manually run migrations. Each instance will create/update its own database schema.

## Environment Configuration

### Required Environment Variables

The following environment variables are required when running in Docker (ProductionLive or ProductionDemo mode):

```bash
# Database credentials
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here

# JWT configuration
JWT_SECRET_KEY=your_256_bit_secret_key_minimum_32_characters_long

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# API key encryption
ENCRYPTION_KEY=your_32_character_encryption_key
```

### .env File

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
# Edit .env with your actual values
```

The `.env` file is automatically loaded by Docker Compose.

### Configuration Hierarchy

Configuration is loaded in the following order (later overrides earlier):

1. `appsettings.json` (base configuration)
2. `appsettings.{Environment}.json` (environment-specific)
3. Environment variables (highest priority)

Example for Docker Live backend:
1. Loads `appsettings.json` (base: logging, JWT structure, ArbitrageConfig)
2. Loads `appsettings.ProductionLive.json` (database, environment flags)
3. Substitutes environment variables (JWT_SECRET_KEY, GOOGLE_CLIENT_ID, etc.)

## Frontend Configuration

The frontend automatically detects the environment from the backend API.

### Environment Badge

The Header component displays the current environment mode:
- Blue "Demo" badge when connected to demo instance
- Yellow "Live" badge when connected to live instance

The frontend fetches environment status from: `GET /api/environment/status`

### Connecting Frontend to Different Backends

In Docker, frontends are built with the correct API base URL:
- `frontend-live`: Built with `VITE_API_BASE_URL=http://localhost:8080/api`
- `frontend-demo`: Built with `VITE_API_BASE_URL=http://localhost:8081/api`

This is configured in `docker-compose.yml` as build arguments.

## Verification

### Check Current Environment

You can verify which environment is running:

#### Via API Endpoint

```bash
# Check Live instance
curl http://localhost:8080/api/environment/status

# Check Demo instance
curl http://localhost:8081/api/environment/status
```

Response:
```json
{
  "isLive": false,
  "mode": "Demo",
  "timestamp": "2025-10-19T15:30:00Z"
}
```

#### Via Startup Logs

When the application starts, it logs the environment mode:

```bash
# For Docker
docker compose logs backend-live | grep "Starting application"
docker compose logs backend-demo | grep "Starting application"
```

Output:
```
info: Startup[0]
      Starting application in Demo mode (IsLive: False)
```

#### Via Exchange Connector Logs

When connecting to exchanges, the logs show which environment is used:

```
info: CryptoArbitrage.API.Services.BinanceConnector[0]
      Connecting to Binance using Testnet environment (IsLive: False)

info: CryptoArbitrage.API.Services.BybitConnector[0]
      Connecting to Bybit using Demo environment (IsLive: False)
```

## Security Considerations

### Demo Mode
- Uses testnet credentials (API keys for testnet)
- No real funds at risk
- Suitable for testing and development
- Can be exposed more liberally

### Live Mode
- Uses production credentials with real funds
- **EXTREME CAUTION REQUIRED**
- Ensure proper security measures:
  - Use strong JWT secret keys
  - Enable HTTPS in production
  - Restrict network access
  - Enable proper authentication
  - Regular security audits
  - Never commit live credentials to version control

## Configuration Files Reference

### appsettings.json (Base)

Contains shared configuration:
- Logging levels
- Authentication (Google OAuth client ID, allowed users)
- JWT structure (issuer, audience, expiration)
- ArbitrageConfig (all arbitrage engine settings)

**Note**: Sensitive values (JWT secret, encryption key) are NOT in this file.

### appsettings.DevelopmentDemo.json

```json
{
  "Environment": {
    "IsLive": false,
    "Mode": "Demo"
  },
  "ConnectionStrings": {
    "DefaultConnection": "Host=localhost;Port=5432;Database=crypto_arbitrage_demo;Username=sansel"
  }
}
```

### appsettings.DevelopmentLive.json

```json
{
  "Environment": {
    "IsLive": true,
    "Mode": "Live"
  },
  "ConnectionStrings": {
    "DefaultConnection": "Host=localhost;Port=5432;Database=crypto_arbitrage_live;Username=sansel"
  }
}
```

### appsettings.ProductionDemo.json

```json
{
  "Environment": {
    "IsLive": false,
    "Mode": "Demo"
  },
  "ConnectionStrings": {
    "DefaultConnection": "Host=postgres;Port=5432;Database=crypto_arbitrage_demo;Username=${POSTGRES_USER};Password=${POSTGRES_PASSWORD}"
  },
  "Authentication": {
    "Google": {
      "ClientId": "${GOOGLE_CLIENT_ID}",
      "ClientSecret": "${GOOGLE_CLIENT_SECRET}"
    }
  },
  "Jwt": {
    "SecretKey": "${JWT_SECRET_KEY}"
  },
  "Encryption": {
    "Key": "${ENCRYPTION_KEY}"
  }
}
```

### appsettings.ProductionLive.json

```json
{
  "Environment": {
    "IsLive": true,
    "Mode": "Live"
  },
  "ConnectionStrings": {
    "DefaultConnection": "Host=postgres;Port=5432;Database=crypto_arbitrage_live;Username=${POSTGRES_USER};Password=${POSTGRES_PASSWORD}"
  },
  "Authentication": {
    "Google": {
      "ClientId": "${GOOGLE_CLIENT_ID}",
      "ClientSecret": "${GOOGLE_CLIENT_SECRET}"
    }
  },
  "Jwt": {
    "SecretKey": "${JWT_SECRET_KEY}"
  },
  "Encryption": {
    "Key": "${ENCRYPTION_KEY}"
  }
}
```

## Troubleshooting

### Application Won't Start

**Issue**: Application fails to start with database connection error

**Solution**: Ensure the database exists:

For local development:
```bash
# Create Demo database
psql -U postgres -c "CREATE DATABASE crypto_arbitrage_demo;"

# Create Live database
psql -U postgres -c "CREATE DATABASE crypto_arbitrage_live;"
```

For Docker:
```bash
# Check if postgres container is running
docker compose ps postgres

# View postgres logs
docker compose logs postgres

# Restart postgres container
docker compose restart postgres
```

### Wrong Environment Detected

**Issue**: Application shows wrong environment mode

**Solution**: Check environment variable priority:
1. Environment variables (highest priority)
2. appsettings.{Environment}.json
3. appsettings.json (lowest priority)

For Docker, verify `ASPNETCORE_ENVIRONMENT` in `docker-compose.yml`:
```yaml
backend-live:
  environment:
    - ASPNETCORE_ENVIRONMENT=ProductionLive  # Must be exactly "ProductionLive"

backend-demo:
  environment:
    - ASPNETCORE_ENVIRONMENT=ProductionDemo  # Must be exactly "ProductionDemo"
```

### Migrations Not Applied

**Issue**: Database tables are missing

**Solution**: Migrations apply automatically on startup. If they don't:

For local development:
```bash
cd src/CryptoArbitrage.API
dotnet ef database update
```

For Docker:
```bash
docker compose exec backend-live dotnet ef database update
docker compose exec backend-demo dotnet ef database update
```

### Exchange Connection Fails

**Issue**: Cannot connect to exchange

**Solution**: Verify you're using correct API keys for the environment:
- Demo mode requires testnet/demo API keys
- Live mode requires production API keys

Check logs for specific errors:
```bash
# Local
tail -f logs/application.log

# Docker
docker compose logs -f backend-live
docker compose logs -f backend-demo
```

### Port Already in Use

**Issue**: Docker fails to start with "port already allocated"

**Solution**: Check if ports are in use:
```bash
# Check ports 80, 81, 8080, 8081, 5432
lsof -i :80
lsof -i :81
lsof -i :8080
lsof -i :8081
lsof -i :5432

# Stop conflicting services or change ports in docker-compose.yml
```

### Database Data Inconsistency

**Issue**: Wrong data appears in environment

**Solution**: Verify correct database connection:

```bash
# For Docker, check which database each service connects to
docker compose exec backend-live env | grep ConnectionStrings
docker compose exec backend-demo env | grep ConnectionStrings

# Connect to postgres and list databases
docker compose exec postgres psql -U postgres -c "\l"

# Verify data in each database
docker compose exec postgres psql -U postgres -d crypto_arbitrage_live -c "SELECT COUNT(*) FROM \"Exchanges\";"
docker compose exec postgres psql -U postgres -d crypto_arbitrage_demo -c "SELECT COUNT(*) FROM \"Exchanges\";"
```

## Best Practices

1. **Always start with Demo mode** for testing new features
2. **Use separate credentials** for Demo and Live environments
3. **Never mix environments** - keep data completely isolated
4. **Monitor Live mode closely** - real money is at risk
5. **Test thoroughly in Demo** before deploying to Live
6. **Use environment-specific secrets** - never share credentials between environments
7. **Implement proper access controls** - restrict who can deploy to Live
8. **Set up alerts** for Live environment anomalies
9. **Regular backups** of Live database
10. **Document all changes** to Live configuration
11. **Use Docker for production** - ensures consistent deployment
12. **Run both modes simultaneously** - allows easy A/B testing and gradual rollout

## Migration from Old Per-User Demo Trading

The platform previously supported per-user demo trading flags. This has been replaced with application-level environment configuration.

### What Changed

- **Removed**: `UseDemoTrading` field from `UserExchangeApiKeys` table
- **Removed**: Demo trading checkbox in API key management UI
- **Removed**: Old configuration files (appsettings.Development.json, appsettings.Production.json, appsettings.Docker.json)
- **Added**: Explicit environment configuration files (DevelopmentDemo, DevelopmentLive, ProductionDemo, ProductionLive)
- **Added**: Separate databases per environment
- **Added**: Environment status badge in UI
- **Added**: Dual Docker deployment (Live and Demo simultaneously)

### Migration Steps

If you're upgrading from the old system:

1. Database migration runs automatically - removes `UseDemoTrading` column
2. Update any custom scripts to use new environment configuration
3. Create separate databases for demo and live
4. Update Docker Compose configuration to new structure
5. Users will need to add separate API keys for each environment instance

## Quick Reference Commands

### Local Development

```bash
# Demo mode
cd src/CryptoArbitrage.API
dotnet run --environment DevelopmentDemo

# Live mode
cd src/CryptoArbitrage.API
dotnet run --environment DevelopmentLive
```

### Docker

```bash
# Start all services
docker compose up -d

# Start only Live
docker compose up -d postgres backend-live frontend-live

# Start only Demo
docker compose up -d postgres backend-demo frontend-demo

# View logs
docker compose logs -f backend-live
docker compose logs -f backend-demo

# Check status
docker compose ps

# Stop services
docker compose down

# Stop and remove data
docker compose down -v
```

### Database

```bash
# Create local databases
psql -U postgres -c "CREATE DATABASE crypto_arbitrage_demo;"
psql -U postgres -c "CREATE DATABASE crypto_arbitrage_live;"

# Run migrations locally
cd src/CryptoArbitrage.API
dotnet ef database update

# Run migrations in Docker
docker compose exec backend-live dotnet ef database update
docker compose exec backend-demo dotnet ef database update

# Access Docker database
docker compose exec postgres psql -U postgres -d crypto_arbitrage_live
docker compose exec postgres psql -U postgres -d crypto_arbitrage_demo
```

## Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture overview
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Production deployment guide
- [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) - Development setup
- [API_REFERENCE.md](./API_REFERENCE.md) - API endpoint reference
- [../docker-compose.yml](../docker-compose.yml) - Docker service configuration
- [../.env.example](../.env.example) - Environment variables reference
