using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace CryptoArbitrage.API.Data;

/// <summary>
/// Design-time factory for EF migrations
/// </summary>
public class ArbitrageDbContextFactory : IDesignTimeDbContextFactory<ArbitrageDbContext>
{
    public ArbitrageDbContext CreateDbContext(string[] args)
    {
        var optionsBuilder = new DbContextOptionsBuilder<ArbitrageDbContext>();

        // Default connection string for local development
        var connectionString = "Host=localhost;Port=5432;Database=crypto_arbitrage;Username=sansel;Password=";

        // Allow override from environment variable
        var envConnectionString = Environment.GetEnvironmentVariable("DATABASE_URL");
        if (!string.IsNullOrEmpty(envConnectionString))
        {
            connectionString = envConnectionString;
        }

        optionsBuilder.UseNpgsql(connectionString);

        return new ArbitrageDbContext(optionsBuilder.Options);
    }
}
