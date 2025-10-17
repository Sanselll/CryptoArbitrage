# Development Guide

Complete guide for developers who want to understand, modify, or extend the Crypto Funding Arbitrage Platform.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Backend Development](#backend-development)
4. [Frontend Development](#frontend-development)
5. [Database Management](#database-management)
6. [Testing](#testing)
7. [Debugging](#debugging)
8. [Code Style & Standards](#code-style--standards)
9. [Adding New Features](#adding-new-features)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)

---

## Development Environment Setup

### Required Tools

**Backend**:
- .NET 8 SDK
- Visual Studio 2022, Rider, or VS Code
- SQLite Browser (optional)
- Postman or similar API testing tool

**Frontend**:
- Node.js 18+
- VS Code with extensions:
  - ESLint
  - Prettier
  - Tailwind CSS IntelliSense
  - TypeScript Vue Plugin (Volar)

**Database**:
- SQLite (included with .NET)
- DB Browser for SQLite (GUI tool)

**Version Control**:
- Git
- GitHub account

### VS Code Extensions

```json
{
  "recommendations": [
    "ms-dotnettools.csharp",
    "ms-dotnettools.csdevkit",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "bradlc.vscode-tailwindcss",
    "Vue.volar",
    "formulahendry.auto-rename-tag",
    "christian-kohler.path-intellisense"
  ]
}
```

### Initial Setup

```bash
# Clone repository
git clone <repo-url>
cd CryptoArbitrage

# Backend setup
cd src/CryptoArbitrage.API
dotnet restore
dotnet build

# Frontend setup
cd ../../client
npm install

# Run both
# Terminal 1:
cd src/CryptoArbitrage.API && dotnet watch run

# Terminal 2:
cd client && npm run dev
```

---

## Project Structure

### Backend Structure

```
src/CryptoArbitrage.API/
├── Controllers/              # API endpoints
│   ├── ExchangeController.cs
│   ├── PositionController.cs
│   └── OpportunityController.cs
├── Data/
│   ├── Entities/            # Database models
│   │   ├── Exchange.cs
│   │   ├── Position.cs
│   │   ├── FundingRate.cs
│   │   ├── Trade.cs
│   │   ├── ArbitrageOpportunity.cs
│   │   └── PerformanceMetric.cs
│   └── ArbitrageDbContext.cs
├── Services/                # Business logic
│   ├── IExchangeConnector.cs
│   ├── BinanceConnector.cs
│   ├── BybitConnector.cs
│   └── ArbitrageEngineService.cs
├── Hubs/                    # SignalR
│   └── ArbitrageHub.cs
├── Models/                  # DTOs
│   ├── FundingRateDto.cs
│   ├── PositionDto.cs
│   ├── ArbitrageOpportunityDto.cs
│   └── AccountBalanceDto.cs
├── Config/                  # Configuration
│   └── ArbitrageConfig.cs
├── Program.cs               # Entry point
├── appsettings.json
└── CryptoArbitrage.API.csproj
```

### Frontend Structure

```
client/
├── src/
│   ├── components/          # React components
│   │   ├── Header.tsx
│   │   ├── BalanceWidget.tsx
│   │   ├── FundingRateMonitor.tsx
│   │   ├── OpportunitiesList.tsx
│   │   └── PositionsGrid.tsx
│   ├── services/            # API clients
│   │   └── signalRService.ts
│   ├── stores/              # State management
│   │   └── arbitrageStore.ts
│   ├── types/               # TypeScript types
│   │   └── index.ts
│   ├── hooks/               # Custom React hooks
│   ├── App.tsx              # Main app component
│   ├── main.tsx             # Entry point
│   └── index.css            # Global styles
├── public/
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

---

## Backend Development

### Adding a New Entity

**Step 1: Create Entity Class**

```csharp
// Data/Entities/Alert.cs
namespace CryptoArbitrage.API.Data.Entities;

public class Alert
{
    public int Id { get; set; }
    public string Type { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public string Severity { get; set; } = string.Empty;
    public bool IsRead { get; set; } = false;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
```

**Step 2: Add to DbContext**

```csharp
// Data/ArbitrageDbContext.cs
public DbSet<Alert> Alerts { get; set; }

protected override void OnModelCreating(ModelBuilder modelBuilder)
{
    // ... existing configuration

    modelBuilder.Entity<Alert>(entity =>
    {
        entity.HasKey(e => e.Id);
        entity.HasIndex(e => e.CreatedAt);
        entity.Property(e => e.Type).IsRequired().HasMaxLength(50);
        entity.Property(e => e.Message).IsRequired();
    });
}
```

**Step 3: Create Migration**

```bash
dotnet ef migrations add AddAlertEntity
dotnet ef database update
```

**Step 4: Create DTO**

```csharp
// Models/AlertDto.cs
namespace CryptoArbitrage.API.Models;

public class AlertDto
{
    public int Id { get; set; }
    public string Type { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public string Severity { get; set; } = string.Empty;
    public bool IsRead { get; set; }
    public DateTime CreatedAt { get; set; }
}
```

**Step 5: Create Controller**

```csharp
// Controllers/AlertController.cs
[ApiController]
[Route("api/[controller]")]
public class AlertController : ControllerBase
{
    private readonly ArbitrageDbContext _context;

    public AlertController(ArbitrageDbContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<List<AlertDto>>> GetAlerts(
        [FromQuery] bool unreadOnly = false)
    {
        var query = _context.Alerts.AsQueryable();

        if (unreadOnly)
            query = query.Where(a => !a.IsRead);

        var alerts = await query
            .OrderByDescending(a => a.CreatedAt)
            .Take(50)
            .Select(a => new AlertDto
            {
                Id = a.Id,
                Type = a.Type,
                Message = a.Message,
                Severity = a.Severity,
                IsRead = a.IsRead,
                CreatedAt = a.CreatedAt
            })
            .ToListAsync();

        return alerts;
    }

    [HttpPost("{id}/mark-read")]
    public async Task<IActionResult> MarkAsRead(int id)
    {
        var alert = await _context.Alerts.FindAsync(id);
        if (alert == null)
            return NotFound();

        alert.IsRead = true;
        await _context.SaveChangesAsync();

        return NoContent();
    }
}
```

### Adding a New Exchange Connector

**Step 1: Install NuGet Package**

```bash
dotnet add package Kraken.Net
```

**Step 2: Implement IExchangeConnector**

```csharp
// Services/KrakenConnector.cs
using Kraken.Net.Clients;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Services;

public class KrakenConnector : IExchangeConnector
{
    private readonly ILogger<KrakenConnector> _logger;
    private KrakenRestClient? _restClient;

    public string ExchangeName => "Kraken";

    public KrakenConnector(ILogger<KrakenConnector> logger)
    {
        _logger = logger;
    }

    public async Task<bool> ConnectAsync(string apiKey, string apiSecret)
    {
        try
        {
            _restClient = new KrakenRestClient(options =>
            {
                options.ApiCredentials = new ApiCredentials(apiKey, apiSecret);
            });

            // Test connection
            var accountInfo = await _restClient.SpotApi.Account.GetBalancesAsync();

            if (accountInfo.Success)
            {
                _logger.LogInformation("Successfully connected to Kraken");
                return true;
            }

            _logger.LogError("Failed to connect to Kraken: {Error}", accountInfo.Error);
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error connecting to Kraken");
            return false;
        }
    }

    public async Task<List<FundingRateDto>> GetFundingRatesAsync(List<string> symbols)
    {
        // Implement Kraken-specific logic
        // Note: Kraken may have different API structure
        return new List<FundingRateDto>();
    }

    // Implement other interface methods...
}
```

**Step 3: Register in DI Container**

```csharp
// Program.cs
builder.Services.AddScoped<KrakenConnector>();
```

**Step 4: Update ArbitrageEngineService**

```csharp
// Services/ArbitrageEngineService.cs
IExchangeConnector? connector = exchange.Name.ToLower() switch
{
    "binance" => new BinanceConnector(...),
    "bybit" => new BybitConnector(...),
    "kraken" => new KrakenConnector(...),
    _ => null
};
```

### Creating a Background Service

```csharp
// Services/PerformanceTrackerService.cs
namespace CryptoArbitrage.API.Services;

public class PerformanceTrackerService : BackgroundService
{
    private readonly ILogger<PerformanceTrackerService> _logger;
    private readonly IServiceProvider _serviceProvider;

    public PerformanceTrackerService(
        ILogger<PerformanceTrackerService> logger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Performance Tracker Service starting");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await CalculateDailyMetricsAsync();
                await Task.Delay(TimeSpan.FromHours(1), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in performance tracker");
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken);
            }
        }
    }

    private async Task CalculateDailyMetricsAsync()
    {
        using var scope = _serviceProvider.CreateScope();
        var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

        var today = DateTime.UtcNow.Date;
        var positions = await dbContext.Positions
            .Where(p => p.ClosedAt >= today && p.ClosedAt < today.AddDays(1))
            .ToListAsync();

        var metric = new PerformanceMetric
        {
            Date = today,
            TotalPnL = positions.Sum(p => p.RealizedPnL),
            TotalTrades = positions.Count,
            WinningTrades = positions.Count(p => p.RealizedPnL > 0),
            LosingTrades = positions.Count(p => p.RealizedPnL < 0),
            // ... calculate other metrics
        };

        dbContext.PerformanceMetrics.Add(metric);
        await dbContext.SaveChangesAsync();
    }
}

// Register in Program.cs
builder.Services.AddHostedService<PerformanceTrackerService>();
```

---

## Frontend Development

### Creating a New Component

**Step 1: Create Component File**

```tsx
// src/components/AlertsList.tsx
import { Bell, CheckCircle } from 'lucide-react';
import { useState, useEffect } from 'react';

interface Alert {
  id: number;
  type: string;
  message: string;
  severity: string;
  isRead: boolean;
  createdAt: string;
}

export const AlertsList = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    fetchAlerts();
  }, []);

  const fetchAlerts = async () => {
    const response = await fetch('http://localhost:5000/api/alert');
    const data = await response.json();
    setAlerts(data);
  };

  const markAsRead = async (id: number) => {
    await fetch(`http://localhost:5000/api/alert/${id}/mark-read`, {
      method: 'POST',
    });
    fetchAlerts();
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error': return 'text-binance-red';
      case 'warning': return 'text-binance-yellow';
      default: return 'text-binance-green';
    }
  };

  return (
    <div className="bg-binance-bg-secondary rounded-lg p-4">
      <div className="flex items-center gap-2 mb-4">
        <Bell className="w-5 h-5 text-binance-yellow" />
        <h2 className="text-lg font-semibold">Alerts</h2>
        <span className="ml-auto px-2 py-1 bg-binance-red/20 text-binance-red text-xs rounded">
          {alerts.filter(a => !a.isRead).length} Unread
        </span>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {alerts.map((alert) => (
          <div
            key={alert.id}
            className={`p-3 rounded border ${
              alert.isRead
                ? 'border-binance-border opacity-50'
                : 'border-binance-yellow/50'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className={`font-medium ${getSeverityColor(alert.severity)}`}>
                    {alert.type}
                  </span>
                  <span className="text-xs text-binance-text-secondary">
                    {new Date(alert.createdAt).toLocaleString()}
                  </span>
                </div>
                <p className="text-sm mt-1">{alert.message}</p>
              </div>
              {!alert.isRead && (
                <button
                  onClick={() => markAsRead(alert.id)}
                  className="text-binance-green hover:text-binance-green/80"
                >
                  <CheckCircle className="w-5 h-5" />
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

**Step 2: Add to App**

```tsx
// src/App.tsx
import { AlertsList } from './components/AlertsList';

function App() {
  return (
    <div className="h-screen flex flex-col bg-binance-bg">
      <Header />
      <main className="flex-1 overflow-hidden p-4">
        {/* ... existing components ... */}
        <AlertsList />
      </main>
    </div>
  );
}
```

### Adding to Zustand Store

```typescript
// src/stores/arbitrageStore.ts
import { create } from 'zustand';

interface Alert {
  id: number;
  type: string;
  message: string;
  severity: string;
  isRead: boolean;
  createdAt: string;
}

interface ArbitrageState {
  // ... existing state
  alerts: Alert[];
  unreadCount: number;

  // ... existing actions
  setAlerts: (alerts: Alert[]) => void;
  addAlert: (alert: Alert) => void;
  markAlertAsRead: (id: number) => void;
}

export const useArbitrageStore = create<ArbitrageState>((set) => ({
  // ... existing state
  alerts: [],
  unreadCount: 0,

  // ... existing actions
  setAlerts: (alerts) => set({
    alerts,
    unreadCount: alerts.filter(a => !a.isRead).length
  }),

  addAlert: (alert) => set((state) => ({
    alerts: [alert, ...state.alerts],
    unreadCount: alert.isRead ? state.unreadCount : state.unreadCount + 1
  })),

  markAlertAsRead: (id) => set((state) => ({
    alerts: state.alerts.map(a =>
      a.id === id ? { ...a, isRead: true } : a
    ),
    unreadCount: state.unreadCount - 1
  })),
}));
```

### Creating Custom Hooks

```typescript
// src/hooks/useAlerts.ts
import { useEffect } from 'react';
import { useArbitrageStore } from '../stores/arbitrageStore';

export const useAlerts = () => {
  const { alerts, setAlerts, addAlert } = useArbitrageStore();

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 30000); // Poll every 30s
    return () => clearInterval(interval);
  }, []);

  const fetchAlerts = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/alert');
      const data = await response.json();
      setAlerts(data);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    }
  };

  const markAsRead = async (id: number) => {
    try {
      await fetch(`http://localhost:5000/api/alert/${id}/mark-read`, {
        method: 'POST',
      });
      useArbitrageStore.getState().markAlertAsRead(id);
    } catch (error) {
      console.error('Failed to mark alert as read:', error);
    }
  };

  return { alerts, markAsRead };
};
```

---

## Database Management

### Creating Migrations

```bash
# Add migration
dotnet ef migrations add MigrationName

# Update database
dotnet ef database update

# Rollback to previous migration
dotnet ef database update PreviousMigrationName

# Remove last migration (if not applied)
dotnet ef migrations remove

# Generate SQL script
dotnet ef migrations script > migration.sql
```

### Seeding Data

```csharp
// Data/DbSeeder.cs
public static class DbSeeder
{
    public static async Task SeedAsync(ArbitrageDbContext context)
    {
        // Check if already seeded
        if (await context.Exchanges.AnyAsync())
            return;

        var exchanges = new List<Exchange>
        {
            new() { Name = "Binance", IsEnabled = false },
            new() { Name = "Bybit", IsEnabled = false },
            new() { Name = "OKX", IsEnabled = false }
        };

        context.Exchanges.AddRange(exchanges);
        await context.SaveChangesAsync();
    }
}

// In Program.cs
using (var scope = app.Services.CreateScope())
{
    var context = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();
    await DbSeeder.SeedAsync(context);
}
```

### Database Queries

**Efficient queries**:

```csharp
// Bad: Multiple database calls
var exchange = await context.Exchanges.FindAsync(id);
var positions = await context.Positions
    .Where(p => p.ExchangeId == exchange.Id).ToListAsync();

// Good: Single query with Include
var exchange = await context.Exchanges
    .Include(e => e.Positions)
    .FirstOrDefaultAsync(e => e.Id == id);

// Read-only queries
var positions = await context.Positions
    .AsNoTracking()
    .Where(p => p.Status == PositionStatus.Open)
    .ToListAsync();

// Projections
var positionSummaries = await context.Positions
    .Select(p => new { p.Symbol, p.UnrealizedPnL })
    .ToListAsync();
```

---

## Testing

### Unit Tests

**Install packages**:

```bash
dotnet new xunit -n CryptoArbitrage.Tests
dotnet add package Moq
dotnet add package FluentAssertions
```

**Example test**:

```csharp
// Tests/Services/ArbitrageEngineServiceTests.cs
public class ArbitrageEngineServiceTests
{
    [Fact]
    public async Task DetectOpportunities_ReturnsValidOpportunities()
    {
        // Arrange
        var fundingRates = new Dictionary<string, List<FundingRateDto>>
        {
            ["Binance"] = new()
            {
                new FundingRateDto
                {
                    Symbol = "BTCUSDT",
                    Rate = -0.0001m,
                    AnnualizedRate = -0.1095m
                }
            },
            ["Bybit"] = new()
            {
                new FundingRateDto
                {
                    Symbol = "BTCUSDT",
                    Rate = 0.0005m,
                    AnnualizedRate = 0.5475m
                }
            }
        };

        var service = new ArbitrageEngineService(...);

        // Act
        var opportunities = await service.DetectOpportunitiesAsync(fundingRates);

        // Assert
        opportunities.Should().HaveCount(1);
        opportunities[0].Symbol.Should().Be("BTCUSDT");
        opportunities[0].SpreadRate.Should().Be(0.0006m);
    }
}
```

### Integration Tests

```csharp
// Tests/Controllers/ExchangeControllerTests.cs
public class ExchangeControllerIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ExchangeControllerIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task GetExchanges_ReturnsSuccessStatusCode()
    {
        // Act
        var response = await _client.GetAsync("/api/exchange");

        // Assert
        response.EnsureSuccessStatusCode();
        var content = await response.Content.ReadAsStringAsync();
        content.Should().Contain("Binance");
    }
}
```

### Frontend Tests (with Vitest)

```bash
npm install -D vitest @testing-library/react @testing-library/jest-dom
```

```typescript
// src/components/__tests__/Header.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Header } from '../Header';

describe('Header', () => {
  it('renders connection status', () => {
    render(<Header />);
    expect(screen.getByText(/connected/i)).toBeInTheDocument();
  });

  it('displays total P&L', () => {
    render(<Header />);
    expect(screen.getByText(/total p&l/i)).toBeInTheDocument();
  });
});
```

---

## Debugging

### Backend Debugging

**VS Code launch.json**:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": ".NET Core Launch (web)",
      "type": "coreclr",
      "request": "launch",
      "preLaunchTask": "build",
      "program": "${workspaceFolder}/src/CryptoArbitrage.API/bin/Debug/net8.0/CryptoArbitrage.API.dll",
      "args": [],
      "cwd": "${workspaceFolder}/src/CryptoArbitrage.API",
      "stopAtEntry": false,
      "serverReadyAction": {
        "action": "openExternally",
        "pattern": "\\bNow listening on:\\s+(https?://\\S+)"
      },
      "env": {
        "ASPNETCORE_ENVIRONMENT": "Development"
      }
    }
  ]
}
```

**Logging**:

```csharp
// Add detailed logging
_logger.LogDebug("Fetching funding rates for {Count} symbols", symbols.Count);
_logger.LogInformation("Detected {Count} arbitrage opportunities", opportunities.Count);
_logger.LogWarning("Funding rate reversed for {Symbol}", symbol);
_logger.LogError(ex, "Failed to connect to {Exchange}", exchangeName);
```

### Frontend Debugging

**Browser DevTools**:
- Network tab: Monitor SignalR WebSocket
- Console: Check for JavaScript errors
- React DevTools: Inspect component props/state

**Debug logging**:

```typescript
// Enable SignalR debug logging
const connection = new HubConnectionBuilder()
  .withUrl('http://localhost:5000/arbitragehub')
  .configureLogging(LogLevel.Debug) // Verbose logging
  .build();

// Component debugging
useEffect(() => {
  console.log('Component mounted');
  console.log('Current state:', state);
  return () => console.log('Component unmounted');
}, []);
```

---

## Code Style & Standards

### Backend (C#)

**Follow Microsoft conventions**:

```csharp
// Good
public class ExchangeConnector
{
    private readonly ILogger<ExchangeConnector> _logger;
    private const int DefaultTimeout = 30000;

    public async Task<bool> ConnectAsync(string apiKey)
    {
        if (string.IsNullOrEmpty(apiKey))
            throw new ArgumentException("API key is required", nameof(apiKey));

        // Implementation
    }
}

// Use var for obvious types
var opportunities = new List<ArbitrageOpportunity>();
var count = opportunities.Count;

// Explicit types for clarity
IExchangeConnector connector = new BinanceConnector();
```

### Frontend (TypeScript)

```typescript
// Use interfaces for data structures
interface Position {
  id: number;
  symbol: string;
  side: PositionSide;
}

// Use arrow functions for React components
export const MyComponent = ({ prop1, prop2 }: Props) => {
  return <div>{prop1}</div>;
};

// Use descriptive names
const fetchFundingRates = async () => { /* ... */ };

// Avoid magic numbers
const REFRESH_INTERVAL = 5000; // 5 seconds
const MAX_RETRIES = 3;
```

### Prettier Configuration

```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2
}
```

---

## Adding New Features

### Feature: Price Alerts

**Backend**:

1. Create `PriceAlert` entity
2. Add `PriceAlertService` background service
3. Monitor prices and trigger alerts
4. Broadcast via SignalR

**Frontend**:

1. Create `PriceAlertForm` component
2. Add to Zustand store
3. Display alerts in UI
4. Subscribe to SignalR events

### Feature: Backtesting

**Backend**:

1. Create `BacktestService`
2. Load historical funding rate data
3. Simulate trades
4. Calculate metrics

**Frontend**:

1. Create `BacktestDashboard` component
2. Display results with charts
3. Allow parameter adjustment

---

## Performance Optimization

### Backend

**Database**:
- Add indexes on frequently queried columns
- Use AsNoTracking() for read-only queries
- Implement caching for static data

**SignalR**:
- Batch updates instead of sending individually
- Compress large payloads
- Use binary protocol if needed

### Frontend

**React**:
- Memoize expensive calculations
- Use React.memo for pure components
- Lazy load components

**Bundle size**:
- Code splitting
- Tree shaking
- Compress images

---

## Troubleshooting

### Common Issues

**Issue**: Database locked
**Solution**: Use WAL mode or switch to PostgreSQL

**Issue**: SignalR not connecting
**Solution**: Check CORS, verify URL, check browser console

**Issue**: High memory usage
**Solution**: Profile with dotnet-trace, check for memory leaks

**Issue**: Slow queries
**Solution**: Add indexes, use pagination, optimize queries

---

This development guide covers the essentials for extending and maintaining the platform. Refer to other documentation files for specific topics like deployment, architecture, and trading strategies.
