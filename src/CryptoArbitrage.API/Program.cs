using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Hubs;
using CryptoArbitrage.API.Services;
using CryptoArbitrage.API.Config;

var builder = WebApplication.CreateBuilder(args);

// Add configuration
var arbitrageConfig = new ArbitrageConfig
{
    MinSpreadPercentage = 0.1m,
    MaxPositionSizeUsd = 10000m,
    MinPositionSizeUsd = 100m,
    MaxLeverage = 5m,
    MaxTotalExposure = 50000m,
    AutoExecute = false,
    DataRefreshIntervalSeconds = 5,
    WatchedSymbols = new List<string> { "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT" }, // Fallback symbols
    EnabledExchanges = new List<string> { "Binance", "Bybit" },

    // Dynamic symbol discovery settings
    AutoDiscoverSymbols = true,
    MinDailyVolumeUsd = 10_000_000m, // $10M minimum daily volume
    MaxSymbolCount = 50, // Track top 50 symbols by volume
    MinAbsFundingRate = 0.0001m, // 0.01% = 3.65% APR minimum
    SymbolRefreshIntervalHours = 24 // Refresh symbol list daily
};
builder.Services.AddSingleton(arbitrageConfig);

// Add SQLite database
builder.Services.AddDbContext<ArbitrageDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")
        ?? "Data Source=arbitrage.db"));

// Add SignalR
builder.Services.AddSignalR();

// Add CORS
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins("http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:5175")
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
});

// Add controllers
builder.Services.AddControllers();

// Add services
builder.Services.AddScoped<BinanceConnector>();
builder.Services.AddScoped<BybitConnector>();
builder.Services.AddScoped<ArbitrageExecutionService>();
builder.Services.AddHostedService<ArbitrageEngineService>();

// Add Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Initialize database
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();
    db.Database.EnsureCreated();

    // Seed initial exchanges if none exist
    if (!db.Exchanges.Any())
    {
        db.Exchanges.AddRange(
            new CryptoArbitrage.API.Data.Entities.Exchange { Name = "Binance", IsEnabled = true },
            new CryptoArbitrage.API.Data.Entities.Exchange { Name = "Bybit", IsEnabled = false }
        );
        await db.SaveChangesAsync();
    }
}

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors();

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

// Map SignalR hub
app.MapHub<ArbitrageHub>("/arbitragehub");

app.Run();
