using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Services.Data;

/// <summary>
/// Service for accessing funding rate data (both current and historical)
/// </summary>
public interface IFundingRateService
{
    /// <summary>
    /// Get current (cached) funding rates for a specific exchange and symbol
    /// </summary>
    Task<FundingRateDto?> GetCurrentRateAsync(string exchange, string symbol);

    /// <summary>
    /// Get current (cached) funding rates for all symbols on an exchange
    /// </summary>
    Task<List<FundingRateDto>> GetCurrentRatesForExchangeAsync(string exchange);

    /// <summary>
    /// Get historical funding rates from database
    /// </summary>
    Task<List<FundingRateDto>> GetHistoricalRatesAsync(
        string exchange,
        string symbol,
        DateTime from,
        DateTime to);

    /// <summary>
    /// Get the latest funding rate from database for a specific exchange/symbol
    /// </summary>
    Task<FundingRateDto?> GetLatestHistoricalRateAsync(string exchange, string symbol);
}
