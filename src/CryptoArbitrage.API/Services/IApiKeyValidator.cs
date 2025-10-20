namespace CryptoArbitrage.API.Services;

/// <summary>
/// Service for validating exchange API key permissions and restrictions
/// </summary>
public interface IApiKeyValidator
{
    /// <summary>
    /// Validates API key permissions for a specific exchange
    /// </summary>
    /// <param name="exchangeName">Exchange name (Binance, Bybit)</param>
    /// <param name="apiKey">API key to validate</param>
    /// <param name="apiSecret">API secret for authentication</param>
    /// <returns>Validation result with details about permissions and restrictions</returns>
    Task<ApiKeyValidationResult> ValidateApiKeyAsync(string exchangeName, string apiKey, string apiSecret);
}
