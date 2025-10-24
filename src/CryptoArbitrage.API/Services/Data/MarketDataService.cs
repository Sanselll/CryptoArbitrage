using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;

namespace CryptoArbitrage.API.Services.Data;

/// <summary>
/// Service for reading cached market data from repositories
/// </summary>
public class MarketDataService : IMarketDataService
{
    private readonly IDataRepository<FundingRateDto> _fundingRateRepository;
    private readonly IDataRepository<MarketDataSnapshot> _marketDataRepository;
    private readonly ILogger<MarketDataService> _logger;

    public MarketDataService(
        IDataRepository<FundingRateDto> fundingRateRepository,
        IDataRepository<MarketDataSnapshot> marketDataRepository,
        ILogger<MarketDataService> logger)
    {
        _fundingRateRepository = fundingRateRepository;
        _marketDataRepository = marketDataRepository;
        _logger = logger;
    }

    public async Task<Dictionary<string, List<FundingRateDto>>> GetFundingRatesAsync()
    {
        try
        {
            // Get all funding rates from repository using pattern matching
            // Keys are in format: funding:{exchange}:{symbol}
            var allRatesDict = await _fundingRateRepository.GetByPatternAsync("funding:*");

            if (!allRatesDict.Any())
            {
                _logger.LogDebug("No funding rates found in repository");
                return new Dictionary<string, List<FundingRateDto>>();
            }

            // Transform from Dictionary<string, FundingRateDto> to Dictionary<string, List<FundingRateDto>>
            // Grouped by exchange name
            var ratesByExchange = new Dictionary<string, List<FundingRateDto>>();

            foreach (var (key, rate) in allRatesDict)
            {
                // Parse exchange from key format "funding:{exchange}:{symbol}"
                var parts = key.Split(':');
                if (parts.Length >= 2)
                {
                    var exchange = parts[1];

                    if (!ratesByExchange.ContainsKey(exchange))
                    {
                        ratesByExchange[exchange] = new List<FundingRateDto>();
                    }

                    ratesByExchange[exchange].Add(rate);
                }
            }

            _logger.LogDebug("Retrieved {Count} funding rates from {Exchanges} exchanges",
                allRatesDict.Count, ratesByExchange.Keys.Count);

            return ratesByExchange;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving funding rates");
            return new Dictionary<string, List<FundingRateDto>>();
        }
    }

    public async Task<Dictionary<string, Dictionary<string, PriceDto>>> GetSpotPricesAsync()
    {
        try
        {
            var snapshot = await GetMarketDataSnapshotAsync();
            return snapshot?.SpotPrices ?? new Dictionary<string, Dictionary<string, PriceDto>>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving spot prices");
            return new Dictionary<string, Dictionary<string, PriceDto>>();
        }
    }

    public async Task<Dictionary<string, Dictionary<string, PriceDto>>> GetPerpetualPricesAsync()
    {
        try
        {
            var snapshot = await GetMarketDataSnapshotAsync();
            return snapshot?.PerpPrices ?? new Dictionary<string, Dictionary<string, PriceDto>>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving perpetual prices");
            return new Dictionary<string, Dictionary<string, PriceDto>>();
        }
    }

    public async Task<MarketDataSnapshot?> GetMarketDataSnapshotAsync()
    {
        try
        {
            const string key = "market_data_snapshot";
            var snapshot = await _marketDataRepository.GetAsync(key);

            if (snapshot == null)
            {
                _logger.LogDebug("No market data snapshot found in cache");
            }

            return snapshot;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error retrieving market data snapshot");
            return null;
        }
    }
}
