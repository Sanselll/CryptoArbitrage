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

var builder = WebApplication.CreateBuilder(args);

// Configure HTTP client to use HTTP/1.1 to avoid HTTP/2 connection heartbeat issues
AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2Support", false);

// Add configuration - Read from appsettings.json and appsettings.Development.json
var arbitrageConfig = builder.Configuration.GetSection("ArbitrageConfig").Get<ArbitrageConfig>() ?? new ArbitrageConfig();
builder.Services.AddSingleton(arbitrageConfig);

// Add environment configuration
var environmentConfig = builder.Configuration.GetSection("Environment").Get<EnvironmentConfig>() ?? new EnvironmentConfig();
builder.Services.AddSingleton(environmentConfig);

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
builder.Services.AddSingleton<IDataAggregationService, DataAggregationService>();
builder.Services.AddSingleton<IOpportunityDetectionService, OpportunityDetectionService>();
builder.Services.AddSingleton<ISignalRStreamingService, SignalRStreamingService>();

// Add exchange connectors and services
builder.Services.AddScoped<BinanceConnector>();
builder.Services.AddScoped<BybitConnector>();
builder.Services.AddScoped<ArbitrageExecutionService>();
builder.Services.AddScoped<INotificationService, NotificationService>();

// Register ArbitrageEngineService as both a singleton (for DI) and a hosted service
builder.Services.AddSingleton<ArbitrageEngineService>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<ArbitrageEngineService>());

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
