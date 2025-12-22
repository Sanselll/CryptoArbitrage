using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Services.Agent;

/// <summary>
/// Service for managing agent configurations.
/// </summary>
public class AgentConfigurationService : IAgentConfigurationService
{
    private readonly ArbitrageDbContext _context;
    private readonly ILogger<AgentConfigurationService> _logger;

    public AgentConfigurationService(
        ArbitrageDbContext context,
        ILogger<AgentConfigurationService> logger)
    {
        _context = context;
        _logger = logger;
    }

    /// <summary>
    /// Get or create default configuration for user.
    /// </summary>
    public async Task<AgentConfiguration> GetOrCreateConfigurationAsync(string userId)
    {
        // Try to get existing configuration
        var config = await _context.AgentConfigurations
            .Where(c => c.UserId == userId)
            .OrderByDescending(c => c.UpdatedAt)
            .FirstOrDefaultAsync();

        if (config != null)
        {
            _logger.LogDebug("Retrieved existing configuration for user {UserId}", userId);
            return config;
        }

        // Create default configuration (V9: single position mode)
        config = new AgentConfiguration
        {
            UserId = userId,
            MaxLeverage = 2.0m,              // Trained with 2.0x leverage
            TargetUtilization = 0.8m,        // Trained with 80% utilization
            MaxPositions = 1,                // V9: single position only
            PredictionIntervalSeconds = 300, // Trained with 5-minute intervals (300s)
            CreatedAt = DateTime.UtcNow,
            UpdatedAt = DateTime.UtcNow
        };

        _context.AgentConfigurations.Add(config);
        await _context.SaveChangesAsync();

        _logger.LogInformation("Created default configuration for user {UserId}", userId);
        return config;
    }

    /// <summary>
    /// Update configuration for user (only allowed when agent is stopped).
    /// </summary>
    public async Task<AgentConfiguration> UpdateConfigurationAsync(
        string userId,
        decimal maxLeverage,
        decimal targetUtilization,
        int maxPositions)
    {
        // Validate configuration
        var (isValid, errorMessage) = ValidateConfiguration(maxLeverage, targetUtilization, maxPositions);
        if (!isValid)
        {
            throw new ArgumentException(errorMessage);
        }

        // Check if agent is currently running
        var runningSession = await _context.AgentSessions
            .Where(s => s.UserId == userId && s.Status == AgentStatus.Running)
            .FirstOrDefaultAsync();

        if (runningSession != null)
        {
            throw new InvalidOperationException("Cannot update configuration while agent is running. Stop the agent first.");
        }

        // Get or create configuration
        var config = await GetOrCreateConfigurationAsync(userId);

        // Update values
        config.MaxLeverage = maxLeverage;
        config.TargetUtilization = targetUtilization;
        config.MaxPositions = maxPositions;
        config.UpdatedAt = DateTime.UtcNow;

        await _context.SaveChangesAsync();

        _logger.LogInformation(
            "Updated configuration for user {UserId}: Leverage={MaxLeverage}, Utilization={TargetUtilization}, MaxPositions={MaxPositions}",
            userId, maxLeverage, targetUtilization, maxPositions);

        return config;
    }

    /// <summary>
    /// Validate configuration values.
    /// </summary>
    public (bool IsValid, string? ErrorMessage) ValidateConfiguration(
        decimal maxLeverage,
        decimal targetUtilization,
        int maxPositions)
    {
        if (maxLeverage < 1.0m || maxLeverage > 5.0m)
        {
            return (false, "Max leverage must be between 1.0 and 5.0");
        }

        if (targetUtilization < 0.5m || targetUtilization > 1.0m)
        {
            return (false, "Target utilization must be between 0.5 (50%) and 1.0 (100%)");
        }

        // V8: Reduced from 3 to 2 max positions to match optimized model architecture
        if (maxPositions < 1 || maxPositions > 2)
        {
            return (false, "Max positions (executions) must be between 1 and 2. Each execution opens 2 positions (long + short hedge).");
        }

        return (true, null);
    }
}
