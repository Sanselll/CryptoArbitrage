# Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Backend Architecture](#backend-architecture)
3. [Frontend Architecture](#frontend-architecture)
4. [Data Flow](#data-flow)
5. [Database Design](#database-design)
6. [Real-time Communication](#real-time-communication)
7. [Security Considerations](#security-considerations)

## System Overview

The Crypto Funding Arbitrage Platform is a full-stack application designed to monitor perpetual futures funding rates across multiple cryptocurrency exchanges and identify profitable arbitrage opportunities in real-time.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Dashboard│  │ Positions│  │  Funding │  │Opportunities│ │
│  │ Widgets  │  │   Grid   │  │   Rates  │  │    List    │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘  │
│       │             │              │              │          │
│       └─────────────┴──────────────┴──────────────┘          │
│                          │                                   │
│                   ┌──────▼──────┐                           │
│                   │   Zustand   │                           │
│                   │    Store    │                           │
│                   └──────┬──────┘                           │
│                          │                                   │
│                   ┌──────▼──────┐                           │
│                   │  SignalR    │                           │
│                   │   Client    │                           │
│                   └──────┬──────┘                           │
└──────────────────────────┼───────────────────────────────────┘
                           │ WebSocket
                           │
┌──────────────────────────┼───────────────────────────────────┐
│                   ┌──────▼──────┐                           │
│                   │  SignalR    │                           │
│                   │     Hub     │                           │
│                   └──────┬──────┘                           │
│                          │                                   │
│         Backend (.NET 8 Web API)                            │
│                          │                                   │
│   ┌──────────────────────┼────────────────────────┐         │
│   │                      │                        │         │
│   │  ┌───────────────────▼────────┐              │         │
│   │  │ ArbitrageEngineService     │              │         │
│   │  │  (Background Service)       │              │         │
│   │  └───┬──────────────┬─────────┘              │         │
│   │      │              │                         │         │
│   │  ┌───▼────┐    ┌────▼────┐    ┌─────────┐  │         │
│   │  │Binance │    │ Bybit   │    │  Risk   │  │         │
│   │  │Connector   │Connector│    │ Manager │  │         │
│   │  └───┬────┘    └────┬────┘    └────┬────┘  │         │
│   │      │              │               │        │         │
│   └──────┼──────────────┼───────────────┼────────┘         │
│          │              │               │                   │
│   ┌──────▼──────────────▼───────────────▼────────┐         │
│   │         Entity Framework Core                 │         │
│   │              (ORM Layer)                      │         │
│   └──────────────────┬────────────────────────────┘         │
│                      │                                       │
│   ┌──────────────────▼────────────────────────────┐         │
│   │            SQLite Database                     │         │
│   │  (Exchanges, Positions, FundingRates, etc.)   │         │
│   └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                     │              │
          ┌──────────┘              └──────────┐
          │                                    │
    ┌─────▼─────┐                       ┌──────▼──────┐
    │  Binance  │                       │    Bybit    │
    │    API    │                       │     API     │
    └───────────┘                       └─────────────┘
```

## Backend Architecture

### Technology Stack
- **Framework**: ASP.NET Core 8.0
- **Database**: SQLite with Entity Framework Core 9.0
- **Real-time**: SignalR for WebSocket communication
- **Exchange Integration**: Binance.Net 10.8.0, Bybit.Net 5.10.0

### Layer Architecture

#### 1. Presentation Layer
**Location**: `/Controllers`, `/Hubs`

**Responsibilities**:
- HTTP API endpoints for CRUD operations
- SignalR hub for real-time broadcasting
- Request validation and response formatting

**Key Components**:
- `ExchangeController`: Manage exchange configurations
- `PositionController`: Query and manage positions
- `OpportunityController`: Access arbitrage opportunities
- `ArbitrageHub`: Real-time WebSocket communication

#### 2. Business Logic Layer
**Location**: `/Services`

**Responsibilities**:
- Arbitrage detection algorithms
- Position management logic
- Risk calculation and validation
- Exchange API integration

**Key Components**:

##### ArbitrageEngineService
- **Type**: Hosted Background Service
- **Lifecycle**: Runs continuously from application start
- **Responsibilities**:
  - Initialize exchange connections
  - Poll funding rates every N seconds
  - Detect arbitrage opportunities
  - Broadcast updates via SignalR
  - Log opportunities to database

**Main Loop**:
```csharp
while (!stoppingToken.IsCancellationRequested)
{
    1. Fetch funding rates from all exchanges
    2. Save to database
    3. Analyze for arbitrage opportunities
    4. Calculate spreads and APR
    5. Broadcast via SignalR
    6. Update positions and balances
    7. Wait for refresh interval
}
```

##### Exchange Connectors
- **Pattern**: Interface-based abstraction (`IExchangeConnector`)
- **Implementations**: `BinanceConnector`, `BybitConnector`
- **Responsibilities**:
  - Authenticate with exchange APIs
  - Fetch funding rates
  - Get account balances
  - Place and manage orders
  - Subscribe to real-time updates

**Interface Design**:
```csharp
public interface IExchangeConnector
{
    string ExchangeName { get; }
    Task<bool> ConnectAsync(string apiKey, string apiSecret);
    Task DisconnectAsync();
    Task<List<FundingRateDto>> GetFundingRatesAsync(List<string> symbols);
    Task<AccountBalanceDto> GetAccountBalanceAsync();
    Task<string> PlaceMarketOrderAsync(string symbol, PositionSide side, decimal quantity, decimal leverage);
    Task<bool> ClosePositionAsync(string symbol);
    Task<List<PositionDto>> GetOpenPositionsAsync();
    Task SubscribeToFundingRatesAsync(Action<FundingRateDto> onUpdate);
}
```

#### 3. Data Access Layer
**Location**: `/Data`

**Responsibilities**:
- Database context management
- Entity configurations
- Data persistence and retrieval

**Key Components**:
- `ArbitrageDbContext`: EF Core DbContext
- Entity models with relationships
- Fluent API configurations for precision and constraints

#### 4. Configuration Layer
**Location**: `/Config`

**Responsibilities**:
- Application settings
- Strategy parameters
- Risk management thresholds

**ArbitrageConfig Properties**:
- `MinSpreadPercentage`: Minimum spread to consider
- `MaxPositionSizeUsd`: Maximum position size
- `MaxLeverage`: Maximum allowed leverage
- `MaxTotalExposure`: Portfolio-wide exposure limit
- `WatchedSymbols`: List of symbols to monitor
- `AutoExecute`: Enable/disable automatic execution

### Dependency Injection

All services registered in `Program.cs`:

```csharp
// Configuration
builder.Services.AddSingleton<ArbitrageConfig>(arbitrageConfig);

// Database
builder.Services.AddDbContext<ArbitrageDbContext>(options =>
    options.UseSqlite(connectionString));

// SignalR
builder.Services.AddSignalR();

// Services
builder.Services.AddScoped<BinanceConnector>();
builder.Services.AddScoped<BybitConnector>();
builder.Services.AddHostedService<ArbitrageEngineService>();

// CORS
builder.Services.AddCors(/* configuration */);
```

## Frontend Architecture

### Technology Stack
- **Framework**: React 18 with TypeScript 5
- **Build Tool**: Vite 5
- **Styling**: Tailwind CSS 3
- **State Management**: Zustand 4
- **Real-time**: @microsoft/signalr 8
- **Icons**: lucide-react 0.4

### Component Architecture

#### Component Hierarchy

```
App
├── Header
│   ├── Connection Status
│   ├── Total P&L Display
│   └── Today P&L Display
├── BalanceWidget (4 cards)
│   ├── Total Balance
│   ├── Available Balance
│   ├── Margin Used
│   └── Unrealized P&L
├── Main Layout (Grid)
│   ├── OpportunitiesList
│   │   └── Opportunity Cards
│   └── FundingRateMonitor
│       └── Funding Rate Table
└── PositionsGrid
    └── Positions Table
```

#### State Management

**Zustand Store** (`arbitrageStore.ts`):

```typescript
interface ArbitrageState {
  // Data
  fundingRates: FundingRate[];
  positions: Position[];
  opportunities: ArbitrageOpportunity[];
  balances: AccountBalance[];
  totalPnL: number;
  todayPnL: number;
  isConnected: boolean;

  // Actions
  setFundingRates: (rates: FundingRate[]) => void;
  setPositions: (positions: Position[]) => void;
  setOpportunities: (opportunities: ArbitrageOpportunity[]) => void;
  setBalances: (balances: AccountBalance[]) => void;
  setPnL: (totalPnL: number, todayPnL: number) => void;
  setConnected: (connected: boolean) => void;
}
```

**Benefits of Zustand**:
- Minimal boilerplate
- No context providers needed
- TypeScript friendly
- DevTools support
- 1KB bundle size

#### Real-time Service

**SignalR Service** (`signalRService.ts`):

**Architecture**:
- Singleton pattern
- Automatic reconnection
- Event-based callbacks
- Unsubscribe support

**Connection Lifecycle**:
1. `connect()` - Establish WebSocket connection
2. `setupEventHandlers()` - Register server event listeners
3. Auto-reconnect on disconnect
4. `disconnect()` - Clean shutdown

**Event Handling**:
```typescript
// Register callback
const unsubscribe = signalRService.onFundingRates((data) => {
  setFundingRates(data);
});

// Clean up
useEffect(() => {
  // Setup
  return () => {
    unsubscribe(); // Cleanup
  };
}, []);
```

## Data Flow

### Funding Rate Update Flow

1. **Exchange API Call** (every 5 seconds)
   - `ArbitrageEngineService` calls exchange connectors
   - Connectors fetch data from Binance/Bybit APIs

2. **Database Persistence**
   - Raw funding rates saved to `FundingRates` table
   - Historical data retained for analysis

3. **Opportunity Detection**
   - Compare rates across exchanges for same symbol
   - Calculate spread = |rate1 - rate2|
   - Filter by minimum threshold
   - Create `ArbitrageOpportunity` records

4. **SignalR Broadcast**
   - Hub broadcasts to all connected clients
   - Events: `ReceiveFundingRates`, `ReceiveOpportunities`

5. **Frontend Update**
   - SignalR client receives event
   - Callback updates Zustand store
   - React components re-render automatically

### Position Management Flow

1. **Manual Position Open** (via API)
   - POST to exchange API
   - Receive order confirmation
   - Create `Position` record
   - Create initial `Trade` record

2. **Real-time Position Updates**
   - Background service polls open positions
   - Calculate current unrealized P&L
   - Update position records
   - Broadcast via SignalR

3. **Funding Fee Collection**
   - Occurs every 8 hours on exchanges
   - Accumulated in `TotalFundingFeeReceived`/`Paid`
   - Net funding displayed on UI

4. **Position Close**
   - Close order placed on exchange
   - Final P&L calculated
   - Position status → Closed
   - `PerformanceMetrics` updated

## Database Design

### Entity Relationship Diagram

```
┌─────────────┐
│  Exchange   │
│ ─────────── │
│ PK Id       │◄────┐
│    Name     │     │
│    ApiKey   │     │
│    ApiSecret│     │
│    IsEnabled│     │
└─────────────┘     │
                    │
      ┌─────────────┼─────────────┐
      │             │             │
      │             │             │
┌─────▼──────┐ ┌────▼────┐  ┌────▼──────────────┐
│ FundingRate│ │Position │  │ArbitrageOpportunity│
│ ────────── │ │──────── │  │ ─────────────────  │
│ PK Id      │ │ PK Id   │  │ PK Id              │
│ FK ExchangeId│ FK ExchangeId FK LongExchangeId│
│    Symbol  │ │   Symbol│  │ FK ShortExchangeId │
│    Rate    │ │   Side  │  │    Symbol          │
│    ...     │ │   Entry │  │    LongFundingRate │
└────────────┘ │   Exit  │  │    ShortFundingRate│
               │   PnL   │  │    SpreadRate      │
               │   ...   │  │    Status          │
               └────┬────┘  └────────────────────┘
                    │
              ┌─────▼────┐
              │  Trade   │
              │ ──────── │
              │ PK Id    │
              │ FK PositionId
              │    Type  │
              │    Price │
              │    Qty   │
              └──────────┘

┌──────────────────┐
│PerformanceMetric │
│ ──────────────── │
│ PK Id            │
│    Date          │
│    TotalPnL      │
│    WinRate       │
│    MaxDrawdown   │
│    ...           │
└──────────────────┘
```

### Table Specifications

#### Exchange
- **Purpose**: Store exchange configurations and credentials
- **Key Fields**:
  - `Name` (UNIQUE): Exchange identifier
  - `ApiKey`, `ApiSecret`: Encrypted credentials
  - `IsEnabled`: Toggle without deleting

#### FundingRate
- **Purpose**: Historical funding rate data
- **Indexes**: `(ExchangeId, Symbol, RecordedAt)`
- **Precision**: `decimal(18, 8)` for rate accuracy
- **Frequency**: Recorded every 5 seconds

#### Position
- **Purpose**: Track trading positions
- **Enums**: `PositionSide` (Long/Short), `PositionStatus` (Open/Closed/Liquidated)
- **Calculated Fields**:
  - `UnrealizedPnL` = (CurrentPrice - EntryPrice) × Quantity × Direction
  - `NetFundingFee` = Received - Paid

#### ArbitrageOpportunity
- **Purpose**: Log detected opportunities
- **Two Foreign Keys**: `LongExchangeId`, `ShortExchangeId`
- **Cascade Behavior**: RESTRICT (prevent accidental deletion)
- **TTL**: Opportunities expire after 10 minutes

#### Trade
- **Purpose**: Individual order executions
- **Relationship**: Many-to-One with Position
- **Types**: Entry, Exit, PartialExit, Liquidation

#### PerformanceMetric
- **Purpose**: Daily performance aggregation
- **Unique Constraint**: One record per date
- **Metrics**: Win rate, Sharpe ratio, max drawdown, etc.

## Real-time Communication

### SignalR Hub

**Endpoint**: `/arbitragehub`

**Connection Setup**:
```typescript
const connection = new HubConnectionBuilder()
  .withUrl('http://localhost:5000/arbitragehub')
  .withAutomaticReconnect()
  .configureLogging(LogLevel.Information)
  .build();
```

**Server-to-Client Events**:

| Event Name | Payload Type | Frequency | Purpose |
|------------|--------------|-----------|---------|
| `ReceiveFundingRates` | `FundingRateDto[]` | 5s | Funding rate updates |
| `ReceivePositions` | `PositionDto[]` | 5s | Position updates |
| `ReceiveOpportunities` | `ArbitrageOpportunityDto[]` | 5s | New opportunities |
| `ReceiveBalances` | `AccountBalanceDto[]` | 5s | Balance updates |
| `ReceivePnLUpdate` | `{ totalPnL, todayPnL }` | 5s | P&L updates |
| `ReceiveAlert` | `{ message, severity }` | Ad-hoc | System alerts |

**Broadcasting Pattern**:
```csharp
await _hubContext.Clients.All.SendAsync("ReceiveFundingRates", data);
```

**Client-to-Server** (Future):
- Execute opportunity
- Modify position
- Update configuration

## Security Considerations

### API Key Management

**Current Implementation**:
- Stored in SQLite database
- Plain text (⚠️ DEVELOPMENT ONLY)

**Production Recommendations**:
1. **Encryption at Rest**:
   ```csharp
   services.AddDataProtection()
       .PersistKeysToFileSystem(new DirectoryInfo(@"./keys/"))
       .SetApplicationName("CryptoArbitrage");
   ```

2. **Secret Management**:
   - Azure Key Vault
   - AWS Secrets Manager
   - HashiCorp Vault
   - Environment variables

3. **Access Control**:
   - API key permissions (read-only vs trading)
   - IP whitelist on exchanges
   - Separate keys for each environment

### CORS Configuration

**Current**: Allows `localhost:3000` and `localhost:5173`

**Production**:
```csharp
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins("https://yourdomain.com")
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
});
```

### Rate Limiting

**Future Enhancement**:
```csharp
builder.Services.AddRateLimiter(options =>
{
    options.AddFixedWindowLimiter("api", opt =>
    {
        opt.Window = TimeSpan.FromMinutes(1);
        opt.PermitLimit = 60;
    });
});
```

### Authentication & Authorization

**Future Enhancement**:
- JWT authentication for API
- User roles (Admin, Viewer, Trader)
- Per-user API configurations
- Audit logging

## Performance Considerations

### Database Optimization

**Indexes**:
- Composite indexes on frequently queried columns
- Covering indexes for read-heavy queries

**Query Optimization**:
- `.Include()` for eager loading related entities
- `.AsNoTracking()` for read-only queries
- Pagination for large result sets

### Caching Strategy

**In-Memory Cache**:
```csharp
services.AddMemoryCache();

// Cache funding rates for 5 seconds
_cache.Set("funding_rates", rates, TimeSpan.FromSeconds(5));
```

**Distributed Cache** (Production):
```csharp
services.AddStackExchangeRedisCache(options =>
{
    options.Configuration = "localhost:6379";
});
```

### SignalR Scaling

**Single Server**: Current configuration works fine

**Multi-Server** (Future):
```csharp
services.AddSignalR()
    .AddStackExchangeRedis("localhost:6379");
```

### Exchange API Rate Limits

**Binance Futures**:
- 2400 requests per minute (weight-based)
- WebSocket: 10 connections per IP

**Bybit**:
- 120 requests per minute per endpoint
- WebSocket: 500 subscriptions per connection

**Implementation**:
- Respect exchange rate limits
- Use WebSocket streams when available
- Implement exponential backoff on errors

## Deployment Architecture

### Development
- Backend: `dotnet run` (localhost:5000)
- Frontend: `npm run dev` (localhost:5173)
- Database: SQLite file

### Production Options

#### Option 1: Single VPS
```
┌────────────────────────────────────┐
│           VPS (Ubuntu)              │
│  ┌──────────────────────────────┐  │
│  │  Nginx (Reverse Proxy)       │  │
│  │  Port 80/443 → :5000/:3000  │  │
│  └──────────────────────────────┘  │
│  ┌──────────────┐  ┌────────────┐  │
│  │  .NET 8 API  │  │   React    │  │
│  │    (systemd) │  │   (static) │  │
│  └──────────────┘  └────────────┘  │
│  ┌──────────────────────────────┐  │
│  │     SQLite Database          │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
```

#### Option 2: Cloud Services
```
┌──────────────────────────────────────┐
│         Azure App Service             │
│    ┌──────────────────────────┐      │
│    │     .NET 8 Web App       │      │
│    └────────────┬─────────────┘      │
└─────────────────┼────────────────────┘
                  │
┌─────────────────▼────────────────────┐
│      Azure SQL Database              │
│      (or PostgreSQL)                 │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│       Vercel / Netlify               │
│    ┌──────────────────────────┐      │
│    │      React Static         │      │
│    └──────────────────────────┘      │
└──────────────────────────────────────┘
```

#### Option 3: Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build: ./src/CryptoArbitrage.API
    ports:
      - "5000:8080"
    environment:
      - ConnectionStrings__DefaultConnection=...
    volumes:
      - ./data:/app/data

  frontend:
    build: ./client
    ports:
      - "80:80"
    depends_on:
      - backend
```

## Monitoring & Observability

### Logging

**Serilog Integration**:
```csharp
builder.Host.UseSerilog((context, config) =>
{
    config.WriteTo.Console()
          .WriteTo.File("logs/app-.txt", rollingInterval: RollingInterval.Day)
          .WriteTo.Seq("http://localhost:5341");
});
```

### Metrics

**Application Insights**:
```csharp
builder.Services.AddApplicationInsightsTelemetry();
```

**Custom Metrics**:
- Opportunities detected per minute
- Average spread percentage
- Position P&L distribution
- API response times

### Health Checks

```csharp
builder.Services.AddHealthChecks()
    .AddDbContextCheck<ArbitrageDbContext>()
    .AddCheck<ExchangeConnectivityCheck>("exchanges");

app.MapHealthChecks("/health");
```

## Testing Strategy

### Unit Tests
- Exchange connector logic
- Arbitrage detection algorithms
- Risk calculations

### Integration Tests
- Database operations
- API endpoints
- SignalR hub communication

### End-to-End Tests
- Full user workflows
- Real-time data propagation
- Error scenarios

### Test Framework
```bash
dotnet new xunit -n CryptoArbitrage.Tests
```

## Extensibility

### Adding New Exchanges

1. Create connector class:
```csharp
public class KuCoinConnector : IExchangeConnector
{
    // Implement interface
}
```

2. Register in DI:
```csharp
builder.Services.AddScoped<KuCoinConnector>();
```

3. Update engine initialization

### Adding New Strategies

```csharp
public interface IArbitrageStrategy
{
    Task<List<Opportunity>> DetectAsync(MarketData data);
    Task<bool> ExecuteAsync(Opportunity opportunity);
}

public class SpotFuturesArbitrage : IArbitrageStrategy { }
public class TriangularArbitrage : IArbitrageStrategy { }
```

### Plugin Architecture

Future: Load strategies from separate assemblies
```csharp
var strategies = Directory.GetFiles("plugins", "*.dll")
    .SelectMany(Assembly.LoadFrom)
    .SelectMany(a => a.GetTypes())
    .Where(t => typeof(IArbitrageStrategy).IsAssignableFrom(t));
```

---

This architecture supports the core requirements while maintaining flexibility for future enhancements. The modular design, clean separation of concerns, and adherence to SOLID principles make the codebase maintainable and testable.
