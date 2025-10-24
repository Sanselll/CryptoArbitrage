using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;

namespace CryptoArbitrage.API.Services.Data;

/// <summary>
/// Service for accessing funding rate data from memory cache
/// </summary>
public class FundingRateService : IFundingRateService
{
    private readonly IDataRepository<FundingRateDto> _repository;
    private readonly ILogger<FundingRateService> _logger;

    public FundingRateService(
        IDataRepository<FundingRateDto> repository,
        ILogger<FundingRateService> logger)
    {
        _repository = repository;
        _logger = logger;
    }

    public async Task<FundingRateDto?> GetCurrentRateAsync(string exchange, string symbol)
    {
        try
        {
            var key = $"funding:{exchange}:{symbol}";
            var rate = await _repository.GetAsync(key);

            if (rate == null)
            {
                _logger.LogDebug("No cached funding rate found for {Exchange}:{Symbol}", exchange, symbol);
            }

            return rate;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving current funding rate for {Exchange}:{Symbol}",
                exchange, symbol);
            return null;
        }
    }

    public async Task<List<FundingRateDto>> GetCurrentRatesForExchangeAsync(string exchange)
    {
        try
        {
            var pattern = $"funding:{exchange}:*";
            var rates = await _repository.GetByPatternAsync(pattern);

            return rates.Values.ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving current funding rates for {Exchange}", exchange);
            return new List<FundingRateDto>();
        }
    }

    public Task<List<FundingRateDto>> GetHistoricalRatesAsync(
        string exchange,
        string symbol,
        DateTime from,
        DateTime to)
    {
        // Memory-only storage doesn't support historical queries
        _logger.LogWarning("Historical funding rate queries not supported with memory-only storage");
        return Task.FromResult(new List<FundingRateDto>());
    }

    public async Task<FundingRateDto?> GetLatestHistoricalRateAsync(string exchange, string symbol)
    {
        // For memory-only, just return the current cached rate
        return await GetCurrentRateAsync(exchange, symbol);
    }
}
