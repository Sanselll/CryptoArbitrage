using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Hubs;
using CryptoArbitrage.API.Services;
using CryptoArbitrage.API.Config;

var builder = WebApplication.CreateBuilder(args);

// Add configuration - Read from appsettings.json and appsettings.Development.json
var arbitrageConfig = builder.Configuration.GetSection("ArbitrageConfig").Get<ArbitrageConfig>() ?? new ArbitrageConfig();
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

// Initialize database with migrations
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();
    db.Database.Migrate();
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
