using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Hubs;
using CryptoArbitrage.API.Services;
using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.Arbitrage.Detection;
using CryptoArbitrage.API.Services.Arbitrage.Execution;
using CryptoArbitrage.API.Services.Authentication;
using CryptoArbitrage.API.Services.Data;
using CryptoArbitrage.API.Services.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Configuration;
using CryptoArbitrage.API.Services.DataCollection.Collectors;
using CryptoArbitrage.API.Services.DataCollection.Repositories;
using CryptoArbitrage.API.Services.DataCollection.Events;
using CryptoArbitrage.API.Services.Exchanges;
using CryptoArbitrage.API.Services.Notifications;
using CryptoArbitrage.API.Services.Streaming;
using CryptoArbitrage.API.Services.ML;

var builder = WebApplication.CreateBuilder(args);

// Configure HTTP client to use HTTP/1.1 to avoid HTTP/2 connection heartbeat issues
AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2Support", false);

// Add configuration - Read from appsettings.json and appsettings.Development.json
var arbitrageConfig = builder.Configuration.GetSection("ArbitrageConfig").Get<ArbitrageConfig>() ?? new ArbitrageConfig();
builder.Services.AddSingleton(arbitrageConfig);

// Add environment configuration
var environmentConfig = builder.Configuration.GetSection("Environment").Get<EnvironmentConfig>() ?? new EnvironmentConfig();
builder.Services.AddSingleton(environmentConfig);

// Add opportunity dump configuration
var opportunityDumpConfig = builder.Configuration.GetSection("OpportunityDump").Get<OpportunityDumpConfig>() ?? new OpportunityDumpConfig();
builder.Services.AddSingleton(opportunityDumpConfig);

// Add notification configuration
builder.Services.Configure<NotificationSettings>(builder.Configuration.GetSection("NotificationSettings"));

var startupLogger = LoggerFactory.Create(config => config.AddConsole()).CreateLogger("Startup");
startupLogger.LogInformation("Starting application in {Mode} mode (IsLive: {IsLive})",
    environmentConfig.Mode, environmentConfig.IsLive);

// Add PostgreSQL database
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection");

if (string.IsNullOrEmpty(connectionString))
    throw new InvalidOperationException("PostgreSQL connection string not configured. Set ConnectionStrings:DefaultConnection in appsettings.json or DATABASE_URL environment variable");

builder.Services.AddDbContext<ArbitrageDbContext>(options =>
    options.UseNpgsql(connectionString));

// Add ASP.NET Core Identity
builder.Services.AddIdentity<ApplicationUser, IdentityRole>(options =>
{
    // Password settings (relaxed since we're using OAuth)
    options.Password.RequireDigit = false;
    options.Password.RequireLowercase = false;
    options.Password.RequireUppercase = false;
    options.Password.RequireNonAlphanumeric = false;
    options.Password.RequiredLength = 6;

    // User settings
    options.User.RequireUniqueEmail = true;
})
.AddEntityFrameworkStores<ArbitrageDbContext>()
.AddDefaultTokenProviders();

// Add JWT Authentication
var jwtSecretKey = Environment.GetEnvironmentVariable("JWT_SECRET_KEY")
                  ?? builder.Configuration["Jwt:SecretKey"];

if (string.IsNullOrEmpty(jwtSecretKey))
    throw new InvalidOperationException("JWT secret key not configured. Set JWT_SECRET_KEY environment variable or Jwt:SecretKey in appsettings.json");

builder.Services.AddAuthentication(options =>
{
    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(options =>
{
    options.SaveToken = true;
    options.RequireHttpsMetadata = false; // Set to true in production
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuer = true,
        ValidateAudience = true,
        ValidateLifetime = true,
        ValidateIssuerSigningKey = true,
        ValidIssuer = builder.Configuration["Jwt:Issuer"],
        ValidAudience = builder.Configuration["Jwt:Audience"],
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecretKey)),
        ClockSkew = TimeSpan.Zero
    };

    // For SignalR - allow token from query string
    options.Events = new JwtBearerEvents
    {
        OnMessageReceived = context =>
        {
            var accessToken = context.Request.Query["access_token"];
            var path = context.HttpContext.Request.Path;

            if (!string.IsNullOrEmpty(accessToken) &&
                path.StartsWithSegments("/hubs"))
            {
                context.Token = accessToken;
            }
            return Task.CompletedTask;
        }
    };
})
.AddGoogle(options =>
{
    options.ClientId = builder.Configuration["Authentication:Google:ClientId"]
        ?? throw new InvalidOperationException("Google ClientId not configured. Set Authentication:Google:ClientId in appsettings.json or environment");
    options.ClientSecret = builder.Configuration["Authentication:Google:ClientSecret"]
        ?? throw new InvalidOperationException("Google ClientSecret not configured. Set Authentication:Google:ClientSecret in appsettings.json or environment");
});

// Add SignalR
builder.Services.AddSignalR();

// Add CORS
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        var allowedOrigins = new List<string>
        {
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175"
        };

        // Add production domain from environment variable
        var prodDomain = Environment.GetEnvironmentVariable("PROD_DOMAIN");
        if (!string.IsNullOrEmpty(prodDomain))
        {
            allowedOrigins.Add($"http://{prodDomain}");
            allowedOrigins.Add($"https://{prodDomain}");
        }

        policy.WithOrigins(allowedOrigins.ToArray())
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
});

// Add controllers
builder.Services.AddControllers();

// Add HTTP context accessor and custom services
builder.Services.AddHttpContextAccessor();
builder.Services.AddScoped<ICurrentUserService, CurrentUserService>();
builder.Services.AddSingleton<IEncryptionService, AesEncryptionService>();
builder.Services.AddHttpClient(); // For API key validator
builder.Services.AddScoped<IApiKeyValidator, ApiKeyValidator>();

// Add memory cache for in-memory data storage
builder.Services.AddMemoryCache();

// Add resilience service for network error handling
builder.Services.AddSingleton<ConnectionResilienceService>();

// Add refactored modular services
builder.Services.AddSingleton<IOpportunityDetectionService, OpportunityDetectionService>();
builder.Services.AddSingleton<ISignalRStreamingService, SignalRStreamingService>();

// ============================================================================
// DATA COLLECTION SYSTEM - New Architecture
// ============================================================================

// Configuration - bind from appsettings.json
builder.Services.AddSingleton(sp =>
{
    var config = new FundingRateCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:FundingRate").Bind(config);
    return config;
});

builder.Services.AddSingleton(sp =>
{
    var config = new FundingRateHistoryCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:FundingRateHistory").Bind(config);
    return config;
});

builder.Services.AddSingleton(sp =>
{
    var config = new MarketPriceCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:MarketPrice").Bind(config);
    return config;
});

builder.Services.AddSingleton(sp =>
{
    var config = new UserDataCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:UserData").Bind(config);
    return config;
});

builder.Services.AddSingleton(sp =>
{
    var config = new LiquidityCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:Liquidity").Bind(config);
    return config;
});

builder.Services.AddSingleton(sp =>
{
    var config = new OpenOrdersCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:OpenOrders").Bind(config);
    return config;
});

builder.Services.AddSingleton(sp =>
{
    var config = new OrderHistoryCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:OrderHistory").Bind(config);
    return config;
});

builder.Services.AddSingleton(sp =>
{
    var config = new TradeHistoryCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:TradeHistory").Bind(config);
    return config;
});

builder.Services.AddSingleton(sp =>
{
    var config = new TransactionHistoryCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:TransactionHistory").Bind(config);
    return config;
});

builder.Services.AddSingleton(sp =>
{
    var config = new HistoricalPriceCollectorConfiguration();
    builder.Configuration.GetSection("DataCollection:HistoricalPrice").Bind(config);
    return config;
});

// Repositories
builder.Services.AddSingleton<IDataRepository<FundingRateDto>, FundingRateRepository>();
builder.Services.AddSingleton<IDataRepository<MarketDataSnapshot>, MarketDataRepository>();
builder.Services.AddSingleton<IDataRepository<PriceHistoryDto>, MemoryDataRepository<PriceHistoryDto>>();
builder.Services.AddSingleton<IDataRepository<UserDataSnapshot>, UserDataRepository>();
builder.Services.AddSingleton<IDataRepository<LiquidityMetricsDto>, LiquidityMetricsRepository>();
builder.Services.AddSingleton<IDataRepository<ArbitrageOpportunityDto>, OpportunityRepository>();
builder.Services.AddSingleton<IDataRepository<List<OrderDto>>, MemoryDataRepository<List<OrderDto>>>();
builder.Services.AddSingleton<IDataRepository<List<TradeDto>>, MemoryDataRepository<List<TradeDto>>>();
builder.Services.AddSingleton<IDataRepository<List<TransactionDto>>, MemoryDataRepository<List<TransactionDto>>>();
builder.Services.AddSingleton<IDataRepository<Dictionary<string, List<HistoricalPriceDto>>>, MemoryDataRepository<Dictionary<string, List<HistoricalPriceDto>>>>();

// Collectors - register as interfaces so background services can resolve them
builder.Services.AddSingleton<IDataCollector<FundingRateDto, FundingRateCollectorConfiguration>, FundingRateCollector>();
builder.Services.AddSingleton<IDataCollector<FundingRateDto, FundingRateHistoryCollectorConfiguration>, FundingRateHistoryCollector>();
builder.Services.AddSingleton<IDataCollector<MarketDataSnapshot, MarketPriceCollectorConfiguration>, MarketPriceCollector>();
builder.Services.AddSingleton<IDataCollector<UserDataSnapshot, UserDataCollectorConfiguration>, UserDataCollector>();
builder.Services.AddSingleton<IDataCollector<LiquidityMetricsDto, LiquidityCollectorConfiguration>, LiquidityMetricsCollector>();
builder.Services.AddSingleton<IDataCollector<List<OrderDto>, OpenOrdersCollectorConfiguration>, OpenOrdersCollector>();
builder.Services.AddSingleton<IDataCollector<List<OrderDto>, OrderHistoryCollectorConfiguration>, OrderHistoryCollector>();
builder.Services.AddSingleton<IDataCollector<List<TradeDto>, TradeHistoryCollectorConfiguration>, TradeHistoryCollector>();
builder.Services.AddSingleton<IDataCollector<List<TransactionDto>, TransactionHistoryCollectorConfiguration>, TransactionHistoryCollector>();
builder.Services.AddSingleton<IDataCollector<Dictionary<string, List<HistoricalPriceDto>>, HistoricalPriceCollectorConfiguration>, HistoricalPriceCollector>();

// Event Bus - coordinates data flow between collectors, aggregators, enrichers, and broadcasters
builder.Services.AddSingleton<IDataCollectionEventBus, DataCollectionEventBus>();

// Background Services - Data Collection (Layer 1: Pure Collectors)
builder.Services.AddHostedService<FundingRateCollectionBackgroundService>();
builder.Services.AddHostedService<FundingRateHistoryCollectionBackgroundService>();
builder.Services.AddHostedService<MarketPriceCollectionBackgroundService>();
builder.Services.AddHostedService<UserDataCollectionBackgroundService>();
builder.Services.AddHostedService<LiquidityCollectionBackgroundService>();
builder.Services.AddHostedService<OpenOrdersCollectionBackgroundService>();
builder.Services.AddHostedService<OrderHistoryCollectionBackgroundService>();
builder.Services.AddHostedService<TradeHistoryCollectionBackgroundService>();
builder.Services.AddHostedService<TransactionHistoryCollectionBackgroundService>();
builder.Services.AddHostedService<HistoricalPriceCollectionBackgroundService>();

// Background Services - Aggregation (Layer 3: Aggregators)
builder.Services.AddHostedService<OpportunityAggregator>();

// Background Services - Enrichment (Layer 4: Enrichers)
builder.Services.AddHostedService<OpportunityEnricher>();

// Background Services - Dumping (Layer 4.5: Snapshot Dumpers)
builder.Services.AddHostedService<OpportunitySnapshotDumper>();

// Background Services - Broadcasting (Layer 5: Broadcasters)
builder.Services.AddHostedService<SignalRBroadcaster>();

// Symbol Discovery Service - manages symbol auto-discovery
builder.Services.AddSingleton<SymbolDiscoveryService>();

// Consumer Services - clean APIs for reading data
builder.Services.AddSingleton<IMarketDataService, MarketDataService>();
builder.Services.AddSingleton<IFundingRateService, FundingRateService>();

// ============================================================================
// ML SERVICES - XGBoost model integration
// ============================================================================

// ML Configuration
builder.Configuration.AddInMemoryCollection(new Dictionary<string, string?>
{
    ["MLModels:Path"] = "models/ml",
    ["MLApi:Host"] = "localhost",
    ["MLApi:Port"] = "5250"
});

// ML Services - Using HTTP API to call Python Flask server
builder.Services.AddSingleton<PythonMLApiClient>();
builder.Services.AddSingleton<OpportunityMLScorer>();

// ============================================================================

// Add exchange connectors and services
builder.Services.AddScoped<BinanceConnector>();
builder.Services.AddScoped<BybitConnector>();
builder.Services.AddScoped<ArbitrageExecutionService>();
builder.Services.AddScoped<INotificationService, NotificationService>();

// DISABLED: ArbitrageEngineService - Now using event-driven data collection services
// The new architecture uses FundingRateCollectionBackgroundService, MarketPriceCollectionBackgroundService,
// and UserDataCollectionBackgroundService which broadcast data immediately when it changes.
// This eliminates duplicate broadcasting and opportunity detection.
// To re-enable, uncomment the lines below:
// builder.Services.AddSingleton<ArbitrageEngineService>();
// builder.Services.AddHostedService(sp => sp.GetRequiredService<ArbitrageEngineService>());

// Add Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Global exception handler to prevent crashes from unhandled exceptions (e.g., HTTP/2 connection issues)
AppDomain.CurrentDomain.UnhandledException += (sender, eventArgs) =>
{
    var exception = eventArgs.ExceptionObject as Exception;
    var logger = app.Services.GetRequiredService<ILogger<Program>>();

    logger.LogCritical(exception,
        "Unhandled exception occurred. IsTerminating: {IsTerminating}",
        eventArgs.IsTerminating);

    // Log but don't terminate - let the app continue if possible
    if (!eventArgs.IsTerminating)
    {
        logger.LogWarning("Application continuing after unhandled exception");
    }
};

// Handle TaskScheduler unobserved task exceptions (including HTTP/2 connection issues)
TaskScheduler.UnobservedTaskException += (sender, eventArgs) =>
{
    var logger = app.Services.GetRequiredService<ILogger<Program>>();
    logger.LogError(eventArgs.Exception, "Unobserved task exception");

    // Mark as observed to prevent process termination
    eventArgs.SetObserved();
};

// Apply database migrations on startup
using (var scope = app.Services.CreateScope())
{
    try
    {

        var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();
        var db = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

        logger.LogInformation("Applying database migrations...");
        db.Database.Migrate();
        logger.LogInformation("Database migrations applied successfully");
    }
    catch (Exception ex)
    {
        var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();
        logger.LogError(ex, "An error occurred while migrating the database");
        throw; // Re-throw to prevent app from starting with broken database
    }
}

// Check if Python ML API is available on startup
using (var scope = app.Services.CreateScope())
{
    try
    {
        var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();
        logger.LogInformation("Checking Python ML API availability...");

        var mlApiClient = scope.ServiceProvider.GetRequiredService<PythonMLApiClient>();
        var isHealthy = await mlApiClient.HealthCheckAsync();

        if (isHealthy)
        {
            logger.LogInformation("✅ Python ML API is available at http://localhost:5250");
        }
        else
        {
            logger.LogWarning("⚠️ Python ML API is not available - ML predictions will not work");
            logger.LogWarning("   Start the ML API server: cd ml_pipeline && python ml_api_server.py");
        }
    }
    catch (Exception ex)
    {
        var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();
        logger.LogWarning(ex, "Failed to connect to Python ML API - continuing without ML predictions");
        // Don't throw - allow app to start without ML if API is not running
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

// IMPORTANT: Order matters - Authentication must come before Authorization
app.UseAuthentication();
app.UseAuthorization();

app.MapControllers();

// Map SignalR hub
app.MapHub<ArbitrageHub>("/hubs/arbitrage");

app.Run();
