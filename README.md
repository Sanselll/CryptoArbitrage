# Crypto Funding Arbitrage Platform

A professional, real-time cryptocurrency funding rate arbitrage platform built with .NET 8 and React + TypeScript. This application automatically monitors funding rates across multiple exchanges (Binance, Bybit) and identifies profitable arbitrage opportunities.

## Features

- **Real-time Funding Rate Monitoring**: Live tracking of perpetual futures funding rates across multiple exchanges
- **Arbitrage Detection Engine**: Automatically identifies profitable funding rate differentials
- **Binance-style Dashboard**: Professional dark-themed UI similar to Binance's interface
- **Position Management**: Track open positions, P&L, and funding fees in real-time
- **SignalR Integration**: WebSocket-based real-time updates with sub-second latency
- **SQLite Database**: Lightweight, local database for historical data and analytics
- **Multi-Exchange Support**: Currently supports Binance and Bybit (extensible architecture)

## Architecture

### Backend (.NET 8)
- **ASP.NET Core Web API**: RESTful API with SignalR hub
- **Entity Framework Core**: ORM with SQLite provider
- **Exchange Connectors**: Binance.Net and Bybit.Net libraries for exchange integration
- **Background Services**: ArbitrageEngineService runs continuously monitoring opportunities
- **Real-time Broadcasting**: SignalR pushes updates to connected clients

### Frontend (React + TypeScript)
- **React 18**: Modern React with hooks
- **TypeScript**: Full type safety
- **Tailwind CSS**: Utility-first CSS with custom Binance color scheme
- **Zustand**: Lightweight state management
- **SignalR Client**: Real-time connection to backend hub
- **Lucide React**: Icon library

## Prerequisites

- **.NET 8 SDK**: [Download here](https://dotnet.microsoft.com/download/dotnet/8.0)
- **Node.js** (v18+): [Download here](https://nodejs.org/)
- **Exchange API Keys** (optional for live trading):
  - Binance Futures API keys
  - Bybit API keys

## Installation

### 1. Clone the Repository

```bash
cd ~/Projects/CryptoArbitrage
```

### 2. Setup Backend

```bash
cd src/CryptoArbitrage.API

# Restore NuGet packages
dotnet restore

# Build the project
dotnet build
```

### 3. Configure Exchange API Keys

Exchange API keys are now configured via `appsettings.json` files:

```bash
cd src/CryptoArbitrage.API

# Copy the example configuration file
cp appsettings.Development.json.example appsettings.Development.json

# Edit appsettings.Development.json and add your API keys
# This file is excluded from Git for security
```

**Configuration Structure:**

`appsettings.Development.json`:
```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "ArbitrageConfig": {
    "Exchanges": [
      {
        "Name": "Binance",
        "ApiKey": "YOUR_BINANCE_API_KEY_HERE",
        "ApiSecret": "YOUR_BINANCE_API_SECRET_HERE",
        "IsEnabled": true
      },
      {
        "Name": "Bybit",
        "ApiKey": "YOUR_BYBIT_API_KEY_HERE",
        "ApiSecret": "YOUR_BYBIT_API_SECRET_HERE",
        "IsEnabled": false
      }
    ]
  }
}
```

**⚠️ Security Notes**:
- `appsettings.Development.json` is excluded from Git via `.gitignore`
- Never commit API keys to version control
- For production, use secure configuration management (Azure Key Vault, AWS Secrets Manager, etc.)

### 4. Setup Frontend

```bash
cd ../../client

# Install dependencies
npm install

# Start development server
npm run dev
```

## Running the Application

### Start Backend (Terminal 1)

```bash
cd ~/Projects/CryptoArbitrage/src/CryptoArbitrage.API
dotnet run
```

The API will start on `http://localhost:5000` (or `https://localhost:5001`)

### Start Frontend (Terminal 2)

```bash
cd ~/Projects/CryptoArbitrage/client
npm run dev
```

The UI will be available at `http://localhost:5173`

## Configuration

### Backend Configuration

Configuration is managed through `appsettings.json` files. Edit `/src/CryptoArbitrage.API/appsettings.json` for trading parameters:

```json
{
  "ArbitrageConfig": {
    "MinSpreadPercentage": 0.1,
    "MaxPositionSizeUsd": 10000,
    "MinPositionSizeUsd": 100,
    "MaxLeverage": 5,
    "MaxTotalExposure": 50000,
    "AutoExecute": false,
    "DataRefreshIntervalSeconds": 5,
    "AutoDiscoverSymbols": true,
    "MinDailyVolumeUsd": 10000000,
    "MaxSymbolCount": 50,
    "MinAbsFundingRate": 0.0001,
    "SymbolRefreshIntervalHours": 24,
    "WatchedSymbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
    "Exchanges": [
      {
        "Name": "Binance",
        "IsEnabled": true
      },
      {
        "Name": "Bybit",
        "IsEnabled": false
      }
    ]
  }
}
```

**Configuration Options:**
- `MinSpreadPercentage`: Minimum annualized spread to trigger opportunity (%)
- `MaxPositionSizeUsd`: Maximum position size per trade ($)
- `MinPositionSizeUsd`: Minimum position size per trade ($)
- `MaxLeverage`: Maximum leverage multiplier
- `MaxTotalExposure`: Maximum total exposure across all positions ($)
- `AutoExecute`: Enable/disable automatic trade execution (⚠️ dangerous)
- `DataRefreshIntervalSeconds`: Data refresh interval (seconds)
- `AutoDiscoverSymbols`: Automatically discover high-volume symbols
- `MinDailyVolumeUsd`: Minimum 24h volume for auto-discovered symbols ($)
- `MaxSymbolCount`: Maximum number of symbols to monitor
- `MinAbsFundingRate`: Minimum absolute funding rate threshold
- `SymbolRefreshIntervalHours`: How often to refresh symbol list (hours)

**API Keys** are configured separately in `appsettings.Development.json` (see step 3 above).

### Frontend Configuration

Edit `/client/src/services/signalRService.ts` to change the API endpoint:

```typescript
async connect(url: string = 'http://localhost:5000/arbitragehub')
```

## API Endpoints

### Positions
- `GET /api/position` - List all positions
- `GET /api/position/{id}` - Get position details
- `GET /api/position?status=Open` - Filter by status

### Opportunities
- `GET /api/opportunity` - List arbitrage opportunities
- `GET /api/opportunity/active` - Get currently active opportunities

### SignalR Hub
- **Endpoint**: `/arbitragehub`
- **Events**:
  - `ReceiveFundingRates` - Real-time funding rate updates
  - `ReceivePositions` - Position updates
  - `ReceiveOpportunities` - New arbitrage opportunities
  - `ReceiveBalances` - Account balance updates
  - `ReceivePnLUpdate` - P&L updates
  - `ReceiveAlert` - System alerts

## Strategy Explanation

### Funding Rate Arbitrage

Perpetual futures contracts use a funding rate mechanism to keep the contract price close to the spot price:

1. **Positive Funding Rate**: Longs pay shorts (when futures > spot)
2. **Negative Funding Rate**: Shorts pay longs (when spot > futures)

**The Arbitrage Opportunity**: When two exchanges have significantly different funding rates for the same contract, you can:

1. Go **LONG** on the exchange with the lower (or more negative) funding rate
2. Go **SHORT** on the exchange with the higher funding rate
3. Collect the funding rate differential every 8 hours
4. Remain market-neutral (hedged position)

### Risk Management

- **Position Sizing**: Automated calculation based on account balance and risk limits
- **Liquidation Protection**: Monitor liquidation prices and close positions if necessary
- **Correlation Monitoring**: Ensure positions remain balanced across exchanges
- **Stop-Loss**: Automatic position closure if losses exceed thresholds

## Dashboard Components

### 1. Header
- Connection status indicator
- Total P&L display
- Today's P&L display

### 2. Balance Widget
- Total balance across all exchanges
- Available balance
- Margin utilization percentage
- Unrealized P&L

### 3. Arbitrage Opportunities
- Real-time list of detected opportunities
- Spread percentage and annualized returns
- Long/Short exchange pairs
- Automatic sorting by profitability

### 4. Funding Rate Monitor
- Live funding rates from all exchanges
- Current rate and annualized percentage
- Next funding time countdown

### 5. Positions Grid
- Open positions with real-time P&L
- Entry price, size, and leverage
- Net funding fees (received - paid)
- Position status

## Development

### Backend Project Structure

```
src/CryptoArbitrage.API/
├── Controllers/          # API controllers
├── Data/
│   ├── Entities/        # Database entities
│   └── ArbitrageDbContext.cs
├── Services/
│   ├── BinanceConnector.cs
│   ├── BybitConnector.cs
│   ├── IExchangeConnector.cs
│   └── ArbitrageEngineService.cs
├── Hubs/
│   └── ArbitrageHub.cs  # SignalR hub
├── Models/              # DTOs
├── Config/              # Configuration classes
└── Program.cs           # Application entry point
```

### Frontend Project Structure

```
client/src/
├── components/          # React components
│   ├── Header.tsx
│   ├── BalanceWidget.tsx
│   ├── FundingRateMonitor.tsx
│   ├── OpportunitiesList.tsx
│   └── PositionsGrid.tsx
├── services/
│   └── signalRService.ts
├── stores/
│   └── arbitrageStore.ts
├── types/
│   └── index.ts         # TypeScript interfaces
├── App.tsx
└── main.tsx
```

## Database Schema

The application uses SQLite with the following main tables:

- **FundingRates**: Historical funding rate data (includes exchange name as string)
- **Positions**: Open and closed positions
- **Executions**: Trade execution records
- **PerformanceMetrics**: Daily performance statistics

**Note**: Exchange configuration is now managed via `appsettings.json` files, not in the database.

## Safety & Disclaimers

**⚠️ IMPORTANT**: This software is for educational purposes only.

- **Test thoroughly** on testnet/paper trading before using real funds
- **Start with small positions** to validate the strategy
- **Monitor constantly** when running with live API keys
- **Market risk**: Funding rates can change rapidly
- **Exchange risk**: Withdrawals, delays, or exchange issues can affect profitability
- **Execution risk**: Slippage and fees can impact returns
- **Not financial advice**: Use at your own risk

## Extending the Platform

### Adding New Exchanges

1. Create a new connector class implementing `IExchangeConnector`
2. Install the appropriate NuGet package (if available)
3. Add the connector to the service provider in `Program.cs`
4. Update the `ArbitrageEngineService` to instantiate the new connector

### Adding New Strategies

The modular design allows for additional strategies:

- Price arbitrage between spot markets
- Triangular arbitrage
- Options volatility arbitrage
- Cross-exchange market making

## Troubleshooting

### Backend won't start
- Ensure .NET 8 SDK is installed: `dotnet --version`
- Check port 5000 is not in use
- Review logs for specific errors

### Frontend won't connect
- Verify backend is running
- Check CORS settings in `Program.cs`
- Inspect browser console for errors
- Ensure SignalR endpoint URL is correct

### No funding rate data
- Verify exchange API connectivity
- Check if exchanges are enabled in database
- Review backend logs for API errors

## Performance Optimization

- **Database**: For high-frequency trading, consider PostgreSQL instead of SQLite
- **Caching**: Implement Redis for distributed caching
- **Scaling**: Deploy multiple instances with load balancing
- **WebSocket**: Optimize SignalR settings for lower latency

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## Support

For questions and support, please open an issue on GitHub.

---

**Built with**  by the crypto trading community
