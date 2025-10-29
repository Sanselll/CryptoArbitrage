using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace CryptoArbitrage.API.Data;

/// <summary>
/// Design-time factory for creating DbContext instances for EF Core migrations
/// </summary>
public class ArbitrageDbContextFactory : IDesignTimeDbContextFactory<ArbitrageDbContext>
{
    public ArbitrageDbContext CreateDbContext(string[] args)
    {
        var optionsBuilder = new DbContextOptionsBuilder<ArbitrageDbContext>();

        // Use a dummy connection string for migrations
        // The actual connection string is configured at runtime
        optionsBuilder.UseNpgsql("Host=localhost;Database=arbitrage_design;Username=postgres;Password=postgres");

        return new ArbitrageDbContext(optionsBuilder.Options);
    }
}
