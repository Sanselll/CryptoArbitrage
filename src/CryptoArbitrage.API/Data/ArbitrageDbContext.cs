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

    public DbSet<Execution> Executions { get; set; }
    public DbSet<Position> Positions { get; set; }
    public DbSet<PositionTransaction> PositionTransactions { get; set; }
    public DbSet<PerformanceMetric> PerformanceMetrics { get; set; }
    public DbSet<UserExchangeApiKey> UserExchangeApiKeys { get; set; }

    // Agent tables
    public DbSet<AgentConfiguration> AgentConfigurations { get; set; }
    public DbSet<AgentSession> AgentSessions { get; set; }
    public DbSet<AgentStats> AgentStats { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder); // IMPORTANT: Call base for Identity tables

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
            entity.Property(e => e.OrderId).HasMaxLength(100);
            entity.Property(e => e.CloseOrderId).HasMaxLength(100);

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

            // Relationship with PositionTransactions
            entity.HasMany(e => e.Transactions)
                .WithOne(t => t.Position)
                .HasForeignKey(t => t.PositionId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        // PositionTransaction configuration
        modelBuilder.Entity<PositionTransaction>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.PositionId);
            entity.HasIndex(e => new { e.TransactionId, e.Exchange });
            entity.HasIndex(e => e.OrderId);

            entity.Property(e => e.TransactionId).IsRequired().HasMaxLength(100);
            entity.Property(e => e.Exchange).IsRequired().HasMaxLength(50);
            entity.Property(e => e.Symbol).IsRequired().HasMaxLength(20);
            entity.Property(e => e.Amount).HasPrecision(18, 8);
            entity.Property(e => e.Fee).HasPrecision(18, 8);
            entity.Property(e => e.SignedFee).HasPrecision(18, 8);
            entity.Property(e => e.AllocationPercentage).HasPrecision(5, 4);
            entity.Property(e => e.OrderId).HasMaxLength(100);
            entity.Property(e => e.Asset).HasMaxLength(20);
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

        // ===================================================================
        // Agent entity configurations
        // ===================================================================

        // AgentConfiguration
        modelBuilder.Entity<AgentConfiguration>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.UserId);

            entity.Property(e => e.MaxLeverage).HasPrecision(5, 2);
            entity.Property(e => e.TargetUtilization).HasPrecision(5, 4);

            // One-to-many relationship with ApplicationUser
            entity.HasOne(e => e.User)
                .WithMany(u => u.AgentConfigurations)
                .HasForeignKey(e => e.UserId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        // AgentSession
        modelBuilder.Entity<AgentSession>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.UserId);
            entity.HasIndex(e => new { e.UserId, e.Status });
            entity.HasIndex(e => e.StartedAt);

            entity.Property(e => e.FinalPnLUsd).HasPrecision(18, 8);
            entity.Property(e => e.FinalPnLPct).HasPrecision(18, 8);

            // One-to-many relationship with ApplicationUser
            entity.HasOne(e => e.User)
                .WithMany(u => u.AgentSessions)
                .HasForeignKey(e => e.UserId)
                .OnDelete(DeleteBehavior.Cascade);

            // Many-to-one relationship with AgentConfiguration
            entity.HasOne(e => e.AgentConfiguration)
                .WithMany()
                .HasForeignKey(e => e.AgentConfigurationId)
                .OnDelete(DeleteBehavior.Restrict);  // Don't delete config if sessions exist
        });

        // AgentStats
        modelBuilder.Entity<AgentStats>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.UserId);
            entity.HasIndex(e => e.AgentSessionId);

            entity.Property(e => e.WinRate).HasPrecision(5, 2);
            entity.Property(e => e.TotalPnLUsd).HasPrecision(18, 8);
            entity.Property(e => e.TotalPnLPct).HasPrecision(18, 8);
            entity.Property(e => e.TodayPnLUsd).HasPrecision(18, 8);
            entity.Property(e => e.TodayPnLPct).HasPrecision(18, 8);
            entity.Property(e => e.MaxDrawdownPct).HasPrecision(18, 8);
            entity.Property(e => e.AveragePositionDurationHours).HasPrecision(18, 8);

            // One-to-many relationship with ApplicationUser
            entity.HasOne(e => e.User)
                .WithMany(u => u.AgentStats)
                .HasForeignKey(e => e.UserId)
                .OnDelete(DeleteBehavior.Cascade);

            // Many-to-one relationship with AgentSession (nullable)
            entity.HasOne(e => e.AgentSession)
                .WithMany()
                .HasForeignKey(e => e.AgentSessionId)
                .OnDelete(DeleteBehavior.SetNull);
        });
    }
}
