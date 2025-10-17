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

Database is created automatically via `db.Database.EnsureCreated()` on startup. For production, use proper migrations (`dotnet ef migrations add`).
