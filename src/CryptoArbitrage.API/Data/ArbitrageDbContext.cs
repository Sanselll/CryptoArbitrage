using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Data;

public class ArbitrageDbContext : IdentityDbContext<ApplicationUser>
{
    public ArbitrageDbContext(DbContextOptions<ArbitrageDbContext> options)
        : base(options)
    {
    }

    public DbSet<FundingRate> FundingRates { get; set; }
    public DbSet<Execution> Executions { get; set; }
    public DbSet<Position> Positions { get; set; }
    public DbSet<PerformanceMetric> PerformanceMetrics { get; set; }
    public DbSet<UserExchangeApiKey> UserExchangeApiKeys { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder); // IMPORTANT: Call base for Identity tables

        // FundingRate configuration
        modelBuilder.Entity<FundingRate>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Id).ValueGeneratedOnAdd(); // Ensure auto-increment

            // Unique index on Exchange + Symbol for efficient upserts
            entity.HasIndex(e => new { e.Exchange, e.Symbol }).IsUnique();

            // Additional index for querying by timestamp
            entity.HasIndex(e => e.RecordedAt);

            entity.Property(e => e.Exchange).IsRequired().HasMaxLength(50);
            entity.Property(e => e.Symbol).IsRequired().HasMaxLength(20);
            entity.Property(e => e.Rate).HasPrecision(18, 8);
            entity.Property(e => e.AnnualizedRate).HasPrecision(18, 8);
        });

        // Execution configuration
        modelBuilder.Entity<Execution>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => new { e.State, e.StartedAt });
            entity.HasIndex(e => new { e.Symbol, e.Exchange });

            entity.Property(e => e.Symbol).IsRequired().HasMaxLength(20);
            entity.Property(e => e.Exchange).IsRequired().HasMaxLength(50);
            entity.Property(e => e.PositionSizeUsd).HasPrecision(18, 8);
            entity.Property(e => e.FundingEarned).HasPrecision(18, 8);
            entity.Property(e => e.SpotOrderId).HasMaxLength(100);
            entity.Property(e => e.PerpOrderId).HasMaxLength(100);
        });

        // PerformanceMetric configuration
        modelBuilder.Entity<PerformanceMetric>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.Date).IsUnique();
            entity.Property(e => e.TotalPnL).HasPrecision(18, 8);
            entity.Property(e => e.RealizedPnL).HasPrecision(18, 8);
            entity.Property(e => e.UnrealizedPnL).HasPrecision(18, 8);
            entity.Property(e => e.TotalFundingFeeReceived).HasPrecision(18, 8);
            entity.Property(e => e.TotalFundingFeePaid).HasPrecision(18, 8);
            entity.Property(e => e.NetFundingFee).HasPrecision(18, 8);
            entity.Property(e => e.WinRate).HasPrecision(5, 2);
            entity.Property(e => e.LargestWin).HasPrecision(18, 8);
            entity.Property(e => e.LargestLoss).HasPrecision(18, 8);
            entity.Property(e => e.MaxDrawdown).HasPrecision(18, 8);
            entity.Property(e => e.AccountBalance).HasPrecision(18, 8);
        });

        // Position configuration
        modelBuilder.Entity<Position>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => new { e.ExecutionId, e.Type });
            entity.HasIndex(e => new { e.Status, e.OpenedAt });
            entity.HasIndex(e => new { e.Symbol, e.Exchange });

            entity.Property(e => e.Symbol).IsRequired().HasMaxLength(20);
            entity.Property(e => e.Exchange).IsRequired().HasMaxLength(50);
            entity.Property(e => e.EntryPrice).HasPrecision(18, 8);
            entity.Property(e => e.ExitPrice).HasPrecision(18, 8);
            entity.Property(e => e.Quantity).HasPrecision(18, 8);
            entity.Property(e => e.Leverage).HasPrecision(5, 2);
            entity.Property(e => e.InitialMargin).HasPrecision(18, 8);
            entity.Property(e => e.RealizedPnL).HasPrecision(18, 8);
            entity.Property(e => e.UnrealizedPnL).HasPrecision(18, 8);
            entity.Property(e => e.TotalFundingFeePaid).HasPrecision(18, 8);
            entity.Property(e => e.TotalFundingFeeReceived).HasPrecision(18, 8);
            entity.Property(e => e.OrderId).HasMaxLength(100);

            // Optional foreign key to Execution (nullable)
            entity.HasOne(e => e.Execution)
                .WithMany()
                .HasForeignKey(e => e.ExecutionId)
                .OnDelete(DeleteBehavior.SetNull);

            // Foreign key to ApplicationUser
            entity.HasOne(e => e.User)
                .WithMany(u => u.Positions)
                .HasForeignKey(e => e.UserId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        // Configure relationships for multi-user support
        modelBuilder.Entity<ApplicationUser>()
            .HasMany(u => u.ExchangeApiKeys)
            .WithOne(k => k.User)
            .HasForeignKey(k => k.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<ApplicationUser>()
            .HasMany(u => u.Executions)
            .WithOne(e => e.User)
            .HasForeignKey(e => e.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<ApplicationUser>()
            .HasMany(u => u.PerformanceMetrics)
            .WithOne(pm => pm.User)
            .HasForeignKey(pm => pm.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        // Indexes for performance
        modelBuilder.Entity<Position>()
            .HasIndex(p => p.UserId);

        modelBuilder.Entity<Execution>()
            .HasIndex(e => e.UserId);

        modelBuilder.Entity<PerformanceMetric>()
            .HasIndex(pm => new { pm.UserId, pm.Date });

        modelBuilder.Entity<UserExchangeApiKey>()
            .HasIndex(k => new { k.UserId, k.ExchangeName });
    }
}
