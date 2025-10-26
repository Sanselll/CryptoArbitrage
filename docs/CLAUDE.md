# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crypto Funding Arbitrage Platform - A real-time cryptocurrency funding rate arbitrage system that monitors perpetual futures funding rates across multiple exchanges (Binance, Bybit) and identifies profitable arbitrage opportunities. Built with .NET 8 backend and React + TypeScript frontend.

## Development Commands

### Backend (.NET 8)

Located in: `src/CryptoArbitrage.API/`

```bash
# Restore dependencies
cd src/CryptoArbitrage.API
dotnet restore

# Build
dotnet build

# Run (starts on http://localhost:5000 or https://localhost:5001)
dotnet run

# Run with watch mode (auto-reload on changes)
dotnet watch run

# Database migrations (using EF Core)
dotnet ef migrations add MigrationName
dotnet ef database update
```

### Frontend (React + TypeScript)

Located in: `client/`

```bash
# Install dependencies
cd client
npm install

# Development server (starts on http://localhost:5173)
npm run dev

# Build for production
npm run build

# Lint
npm run lint

# Preview production build
npm run preview
```

## Architecture Overview

### Backend Architecture

**Core Pattern**: Background service architecture with real-time SignalR broadcasting

- **ArbitrageEngineService** (Background Service): The heart of the system. Runs continuously in a loop every 5 seconds (configurable). Manages the entire arbitrage detection lifecycle:
  1. Fetches funding rates from all enabled exchanges via exchange connectors
  2. Stores rates in SQLite database
  3. Detects arbitrage opportunities by comparing rates across exchange pairs
  4. Broadcasts updates via SignalR to connected clients
  5. Updates positions and balances
  6. Optionally auto-executes trades (when AutoExecute=true)

- **Exchange Connector Pattern**: All exchange integrations implement `IExchangeConnector` interface. This allows seamless addition of new exchanges without modifying core logic. Connectors handle:
  - Authentication with exchange APIs
  - Fetching funding rates
  - Placing orders
  - Managing positions
  - Account balance queries

- **SignalR Hub** (`ArbitrageHub`): WebSocket-based real-time communication. Broadcasts events:
  - `ReceiveFundingRates`: Live funding rate updates
  - `ReceiveOpportunities`: New arbitrage opportunities
  - `ReceivePositions`: Position updates
  - `ReceiveBalances`: Account balance updates
  - `ReceivePnLUpdate`: P&L updates
  - `ReceiveAlert`: System alerts

- **Data Layer**: Entity Framework Core with SQLite. Database is auto-created on first run at `arbitrage.db`. Includes entities for Exchanges, FundingRates, Positions, Trades, ArbitrageOpportunities, and PerformanceMetrics.

### Frontend Architecture

**Core Pattern**: Real-time dashboard with centralized state management

- **State Management**: Zustand store (`stores/arbitrageStore.ts`) manages all application state including funding rates, positions, opportunities, balances, and connection status.

- **Real-time Connection**: SignalR client service (`services/signalRService.ts`) maintains WebSocket connection to backend. On message receipt, updates Zustand store which triggers React re-renders.

- **Component Structure**:
  - `Header.tsx`: Connection status, total P&L, today's P&L
  - `BalanceWidget.tsx`: Total balance, available balance, margin utilization
  - `FundingRateMonitor.tsx`: Live funding rates from all exchanges
  - `OpportunitiesList.tsx`: Detected arbitrage opportunities with profitability metrics
  - `PositionsGrid.tsx`: Open positions with real-time P&L tracking

- **Styling**: Tailwind CSS with custom Binance-inspired dark theme color palette.

## Configuration

### Backend Configuration (`src/CryptoArbitrage.API/Program.cs`)

The `ArbitrageConfig` object controls all engine behavior:

- `MinSpreadPercentage`: Minimum annualized spread percentage to trigger opportunity detection (default: 0.1%)
- `MaxPositionSizeUsd`: Maximum position size per trade
- `MinPositionSizeUsd`: Minimum position size per trade
- `MaxLeverage`: Maximum leverage allowed
- `MaxTotalExposure`: Maximum total exposure across all positions
- `AutoExecute`: Enable/disable automatic trade execution (DANGEROUS - test thoroughly first)
- `DataRefreshIntervalSeconds`: How often to fetch data and run analysis (default: 5 seconds)
- `WatchedSymbols`: List of symbols to monitor (e.g., ["BTCUSDT", "ETHUSDT"])
- `EnabledExchanges`: List of exchanges to use (e.g., ["Binance", "Bybit"])

### CORS Configuration

Frontend origins are configured in `Program.cs`. Default: `http://localhost:3000` and `http://localhost:5173`. Modify if running on different ports.

### Exchange API Keys

Exchange API credentials are stored in the SQLite database (not in code). To add keys:
1. Run the backend once to create the database
2. Exchanges are seeded with `IsEnabled=false`
3. Update exchange records via API endpoints (`PUT /api/exchange/{id}`) or directly in SQLite

**Security Note**: For production, migrate to Azure Key Vault, AWS Secrets Manager, or similar secure storage.

## Key Workflow Concepts

### Arbitrage Detection Algorithm

Located in `ArbitrageEngineService.DetectOpportunitiesAsync()`:

1. Collects funding rates from all enabled exchanges for all watched symbols
2. Finds symbols that exist on multiple exchanges (intersection)
3. For each symbol, compares funding rates between all exchange pairs
4. When annualized spread exceeds `MinSpreadPercentage`:
   - Identifies which exchange has lower funding rate (go LONG there)
   - Identifies which exchange has higher funding rate (go SHORT there)
   - Calculates estimated profit percentage (annualized spread)
   - Creates ArbitrageOpportunityDto
5. Returns opportunities sorted by profitability (highest spread first)

### Position Management

The strategy creates market-neutral hedged positions:
- LONG position on exchange with lower/negative funding rate (receives funding)
- SHORT position on exchange with higher funding rate (pays less or receives more)
- Net result: Collect funding rate differential every 8 hours while remaining hedged

## Adding New Exchanges

To integrate a new exchange:

1. **Install NuGet package** (if available, e.g., OKX.Net, KuCoin.Net)
2. **Create connector class** in `Services/` implementing `IExchangeConnector`:
   - Inherit interface and implement all methods
   - Handle authentication, API calls, error handling
   - Map exchange-specific models to DTOs
3. **Register in DI container** (`Program.cs`):
   ```csharp
   builder.Services.AddScoped<NewExchangeConnector>();
   ```
4. **Add to ArbitrageEngineService** switch statement in `InitializeExchangesAsync()`
5. **Seed database** with new exchange record
6. **Test thoroughly** before enabling in production

## Testing Strategy

While this codebase doesn't currently have unit tests, when adding tests consider:

- **Exchange Connectors**: Mock API responses, test error handling, rate limiting
- **ArbitrageEngine**: Test opportunity detection logic with mock funding rates
- **Risk Management**: Test position sizing, exposure limits, leverage constraints
- **Database**: Integration tests for EF Core queries

## Important Safety Considerations

This is a financial trading application. Exercise extreme caution:

- **Always test on testnet** before using real funds
- **Never commit API keys** to version control
- **Monitor AutoExecute carefully** - start with it disabled
- **Validate all calculations** - funding rates can change rapidly
- **Consider exchange risks** - withdrawals, API downtime, liquidation
- This is educational software - use at your own risk

## Database Schema Notes

- **Exchanges**: Stores exchange configuration (Name, ApiKey, ApiSecret, IsEnabled)
- **FundingRates**: Time-series data of historical funding rates per exchange per symbol
- **Positions**: Open and closed positions with P&L tracking
- **Trades**: Individual trade executions
- **ArbitrageOpportunities**: Log of detected opportunities (for backtesting/analysis)
- **PerformanceMetrics**: Daily aggregated performance statistics

Database is created automatically via migrations on startup. For production, use proper migrations (`dotnet ef migrations add`).

## Database Migrations - CRITICAL SAFETY RULES

**NEVER drop the database to fix migration issues in development or production. ALWAYS use migrations to alter the schema.**

### Safe Migration Workflow

1. **BEFORE creating ANY migration:**
   ```bash
   # ALWAYS backup the database first
   ./backup-db.sh
   ```

2. **Create a migration (adds/modifies tables or columns):**
   ```bash
   cd src/CryptoArbitrage.API
   dotnet ef migrations add DescriptiveMigrationName
   ```

3. **Review the generated migration file:**
   - Check `Migrations/YYYYMMDDHHMMSS_MigrationName.cs`
   - Verify it only contains ADD/ALTER operations, NO DROP TABL ES
   - If you see `DropTable()` or `DropColumn()` - **STOP and review carefully**

4. **Apply the migration:**
   ```bash
   dotnet ef database update
   ```

5. **If migration fails:**
   - **DO NOT** drop the database
   - **DO NOT** delete all migrations and start fresh
   - **DO** create a new migration to fix the issue
   - **DO** restore from backup if data is corrupted

### Fixing Schema Issues WITHOUT Losing Data

If you need to fix an entity configuration (like adding `ValueGeneratedOnAdd()`):

1. Update `ArbitrageDbContext.cs` with the correct configuration
2. Create a new migration: `dotnet ef migrations add FixEntityConfiguration`
3. The migration will generate ALTER TABLE statements
4. Apply it: `dotnet ef database update`

### When You MUST Drop the Database

Only drop the database if:
- It's a fresh development start with NO real data
- You have a verified backup
- User explicitly approves data loss

**Commands to drop database:**
```bash
# PostgreSQL
/opt/homebrew/opt/postgresql@16/bin/dropdb --force crypto_arbitrage
/opt/homebrew/opt/postgresql@16/bin/createdb crypto_arbitrage

# Then apply migrations
dotnet ef database update
```

### Database Backup and Restore

```bash
# Backup
./backup-db.sh

# Restore
./restore-db.sh backups/crypto_arbitrage_YYYYMMDD_HHMMSS.sql
```

## Docker Deployment

The project is containerized with Docker and ready for deployment to any cloud platform or local environment.

### Quick Start with Docker

```bash
# Local development deployment
./deploy-local.sh

# View logs
./deploy-local.sh logs

# Stop services
./deploy-local.sh down
```

### Docker Commands

```bash
# Build and start all services (PostgreSQL, Backend, Frontend)
docker-compose up -d

# Build with fresh images
docker-compose up -d --build

# View logs
docker-compose logs -f
docker-compose logs -f backend  # Specific service

# Check status
docker-compose ps

# Stop services (preserves database)
docker-compose down

# Stop and remove all data (WARNING: deletes database!)
docker-compose down -v

# Access database
docker-compose exec postgres psql -U postgres -d crypto_arbitrage
```

### Database Management

```bash
# Run migrations in Docker
docker-compose exec backend dotnet ef database update

# Create new migration
cd src/CryptoArbitrage.API
dotnet ef migrations add MigrationName

# Backup database
docker-compose exec postgres pg_dump -U postgres crypto_arbitrage > backup.sql

# Restore database
cat backup.sql | docker-compose exec -T postgres psql -U postgres crypto_arbitrage
```

### Production Deployment

```bash
# Deploy to production
./deploy-production.sh deploy

# Create database backup
./deploy-production.sh backup

# Restore from backup
./deploy-production.sh restore backup-20241018-120000.sql.gz

# Check service health
./deploy-production.sh health

# View production logs
./deploy-production.sh logs backend
```

### Environment Configuration

- **Development**: `.env` file (copy from `.env.example`)
- **Production**: `.env.production` file (copy from `.env.production.example`)

**IMPORTANT**: Never commit `.env` or `.env.production` files. Always change default passwords and secrets!

### Docker Image Structure

- **Backend**: Multi-stage .NET 8 build (SDK → Runtime)
- **Frontend**: Multi-stage React build (Node → Nginx)
- **Database**: PostgreSQL 16 Alpine with persistent volumes

### CI/CD

GitHub Actions workflows are configured for:
- **CI** (`.github/workflows/ci.yml`): Runs on every push/PR, builds and tests code
- **Docker Build** (`.github/workflows/docker-build-push.yml`): Builds and pushes images to GitHub Container Registry

Trigger manual deployment:
```bash
gh workflow run docker-build-push.yml
```

### Cloud Deployment

The Docker setup is cloud-agnostic and works with:
- **AWS**: ECS, Fargate, or EC2
- **Azure**: Container Instances or App Service
- **GCP**: Cloud Run or GKE
- **DigitalOcean**: Droplets or App Platform
- **Any VPS**: With Docker installed

See `DEPLOYMENT.md` for detailed deployment guides for each platform.

## Database

**Database**: PostgreSQL 16 (migrated from SQLite for production readiness)

**Connection**: Configured via environment variables in `appsettings.json` and `.env` files

**Persistence**: Docker volumes ensure data persists across container restarts and deployments

## Machine Learning (ML) Pipeline

### Architecture

The ML system uses a **Flask REST API microservice** architecture running on port 5250, separate from the C# backend on port 5052.

**Why Flask API instead of embedded Python.NET?**
- ✅ Simple deployment (no platform-specific Python DLL dependencies)
- ✅ Easy debugging (standard Flask logs)
- ✅ Scalable (can run on separate server/container)
- ✅ Language agnostic (any service can call HTTP API)

### ML Commands

**Location**: `ml_pipeline/` directory

#### Setup and Training

```bash
cd ml_pipeline

# Create virtual environment (first time only)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models (creates .pkl files in models/xgboost/)
./train.sh

# Output:
# - models/xgboost/profit_model.pkl
# - models/xgboost/success_model.pkl
# - models/xgboost/duration_model.pkl
# - models/xgboost/scaler.pkl
```

#### Start ML API Server

```bash
cd ml_pipeline
source venv/bin/activate

# Start Flask server (development)
python ml_api_server.py

# Server runs on http://localhost:5250

# Start with Gunicorn (production)
gunicorn -w 4 -b 0.0.0.0:5250 ml_api_server:app
```

#### Testing ML API

```bash
# Health check
curl http://localhost:5250/health

# Expected response:
# {"status": "healthy", "service": "ml-api", "version": "1.0.0"}

# Single prediction (requires opportunity JSON)
curl -X POST http://localhost:5250/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "fundProfit8h": -0.0242,
    "fundApr": -10.63,
    "volume24h": 1500000000,
    ...
  }'

# Batch predictions
curl -X POST http://localhost:5250/predict/batch \
  -H "Content-Type: application/json" \
  -d @opportunities.json
```

#### Model Validation

```bash
cd ml_pipeline
source venv/bin/activate

# Validate trained models against backend predictions
python validate_backend_predictions.py

# This will:
# 1. Load historical opportunities
# 2. Make predictions via ML API
# 3. Compare with backend opportunity data
# 4. Output validation report
```

#### Troubleshooting ML API

```bash
# Check if ML API is running
curl http://localhost:5250/health

# Find process using port 5250
lsof -i :5250

# Kill ML API process
pkill -f ml_api_server

# View ML API logs
tail -f /tmp/ml_api_server.log

# Check models exist
ls -la ml_pipeline/models/xgboost/
# Should show: profit_model.pkl, success_model.pkl, duration_model.pkl, scaler.pkl

# Retrain models if missing
cd ml_pipeline
source venv/bin/activate
./train.sh
```

### ML Integration with C# Backend

The C# backend automatically connects to ML API on startup:

```bash
# Backend checks ML API health on startup
cd src/CryptoArbitrage.API
dotnet run

# Look for log message:
# ✅ Python ML API is available at http://localhost:5250
# OR
# ⚠️ Python ML API is not available. ML predictions will be disabled.
```

**Backend configuration** (`appsettings.json`):
```json
{
  "MLApi": {
    "Host": "localhost",
    "Port": "5250"
  }
}
```

**Integration flow**:
1. OpportunityEnricher calls OpportunityMLScorer
2. OpportunityMLScorer calls PythonMLApiClient (HTTP client)
3. PythonMLApiClient sends POST request to `http://localhost:5250/predict`
4. Flask API runs prediction and returns scores
5. Backend enriches opportunities with ML predictions

### ML Model Files

**Location**: `ml_pipeline/models/xgboost/`

**Files** (gitignored):
- `profit_model.pkl` - Predicts expected profit percentage
- `success_model.pkl` - Predicts probability of profitable trade
- `duration_model.pkl` - Predicts optimal hold duration (hours)
- `scaler.pkl` - StandardScaler for feature normalization

**Training data**: `src/CryptoArbitrage.HistoricalCollector/data/training_data.csv`

### Historical Data Collection

```bash
# Navigate to historical collector
cd src/CryptoArbitrage.HistoricalCollector

# Backfill historical data (snapshots + simulations)
dotnet run -- backfill --start-date 2024-04-24 --end-date 2024-10-24

# Generate training data from snapshots
dotnet run -- simulate --output training_data.csv

# Live collection (runs continuously)
dotnet run -- live --interval 5

# Full pipeline (backfill + simulate)
dotnet run -- full --start-date 2024-04-24 --end-date 2024-10-24 --output training_data.csv
```

### ML Deployment

**Development**:
```bash
# Terminal 1: Start ML API
cd ml_pipeline && source venv/bin/activate && python ml_api_server.py

# Terminal 2: Start C# backend
cd src/CryptoArbitrage.API && dotnet run

# Terminal 3: Start frontend
cd client && npm run dev
```

**Production (systemd)**:
```bash
# Create /etc/systemd/system/ml-api.service
sudo systemctl enable ml-api
sudo systemctl start ml-api
sudo systemctl status ml-api

# View logs
sudo journalctl -u ml-api -f
```

**Production (Docker)**:
```bash
# Build ML API image
docker build -f Dockerfile.ml-api -t crypto-arbitrage-ml-api .

# Run ML API container
docker run -d -p 5250:5250 --name ml-api crypto-arbitrage-ml-api

# Or use docker-compose (includes backend + frontend + ML API)
docker-compose up -d
docker-compose logs -f ml-api
```

### ML Documentation

- **ML API Guide**: `ml_pipeline/ML_API_GUIDE.md` - Comprehensive API reference
- **ML Pipeline README**: `ml_pipeline/README.md` - Quick start guide
- **ML Implementation Guide**: `docs/ml-implementation-guide.md` - Full system design
- **Architecture**: `docs/ARCHITECTURE.md` - ML Services Architecture section
- **Deployment**: `docs/DEPLOYMENT_GUIDE.md` - ML API Service Deployment section

### Port Configuration

- **Frontend**: 5173 (Vite dev server)
- **Backend**: 5052 (C# ASP.NET Core)
- **ML API**: 5250 (Python Flask)
- **PostgreSQL**: 5432

### ML Performance

**Features**: 54 engineered features per opportunity
**Models**: XGBoost (gradient boosting)
**Inference time**: ~10-20ms per opportunity
**Batch efficiency**: Use `/predict/batch` for multiple opportunities

### Model Retraining

```bash
# When new training data is available
cd ml_pipeline
source venv/bin/activate

# Retrain models
./train.sh

# Restart ML API to load new models
# systemd:
sudo systemctl restart ml-api

# Docker:
docker restart ml-api

# Manual:
pkill -f ml_api_server
python ml_api_server.py
```

## Git Commit Guidelines

When creating git commits:

- **DO NOT** include Claude Code attribution or co-authorship information
- Use only the repository owner's name and credentials
- Keep commit messages concise and descriptive
- Follow the existing commit message style in the repository
- Focus on what changed and why, not who made the change
