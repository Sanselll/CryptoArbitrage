using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Services.Agent;

/// <summary>
/// Service for managing agent configurations.
/// </summary>
public interface IAgentConfigurationService
{
    /// <summary>
    /// Get or create default configuration for user.
    /// </summary>
    Task<AgentConfiguration> GetOrCreateConfigurationAsync(string userId);

    /// <summary>
    /// Update configuration for user (only allowed when agent is stopped).
    /// </summary>
    Task<AgentConfiguration> UpdateConfigurationAsync(string userId, decimal maxLeverage, decimal targetUtilization, int maxPositions);

    /// <summary>
    /// Validate configuration values.
    /// </summary>
    (bool IsValid, string? ErrorMessage) ValidateConfiguration(decimal maxLeverage, decimal targetUtilization, int maxPositions);
}
