using Microsoft.Extensions.Caching.Memory;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Config;

namespace CryptoArbitrage.API.Services;

public class DataAggregationService : IDataAggregationService
{
    private readonly IMemoryCache _cache;
    private readonly ArbitrageConfig _config;
    private readonly ILogger<DataAggregationService> _logger;
    private const string CACHE_KEY_PREFIX_FUNDING = "funding_";
    private const string CACHE_KEY_PREFIX_SPOT = "spot_";
    private const string CACHE_KEY_PREFIX_PERP = "perp_";

    public DataAggregationService(
        IMemoryCache cache,
        ArbitrageConfig config,
        ILogger<DataAggregationService> logger)
    {
        _cache = cache;
        _config = config;
        _logger = logger;
    }

    public async Task<MarketDataSnapshot> FetchAndCacheMarketDataAsync(
        List<string> symbols,
        Dictionary<string, IExchangeConnector> exchangeConnectors,
        CancellationToken cancellationToken = default)
    {
        var snapshot = new MarketDataSnapshot { FetchedAt = DateTime.UtcNow };

        foreach (var (exchangeName, connector) in exchangeConnectors)
        {
            try
            {
                // Fetch funding rates
                var rates = await connector.GetFundingRatesAsync(symbols);

                // Enrich with previous rates from cache
                foreach (var rate in rates)
                {
                    var cacheKey = $"{CACHE_KEY_PREFIX_FUNDING}{exchangeName}:{rate.Symbol}";
                    if (_cache.TryGetValue<FundingRateCacheEntry>(cacheKey, out var cachedEntry))
                    {
                        rate.PreviousRate = cachedEntry.CurrentRate.Rate;
                        rate.PreviousAnnualizedRate = cachedEntry.CurrentRate.AnnualizedRate;
                    }

                    // Update cache
                    var newEntry = new FundingRateCacheEntry
                    {
                        CurrentRate = rate,
                        PreviousRate = cachedEntry?.CurrentRate,
                        LastUpdated = DateTime.UtcNow
                    };

                    _cache.Set(cacheKey, newEntry, TimeSpan.FromMinutes(_config.FundingRateCacheDurationMinutes));
                }

                snapshot.FundingRates[exchangeName] = rates;

                // Fetch spot prices
                var spotPrices = await connector.GetSpotPricesAsync(symbols);
                snapshot.SpotPrices[exchangeName] = spotPrices;
                _cache.Set($"{CACHE_KEY_PREFIX_SPOT}{exchangeName}", spotPrices,
                    TimeSpan.FromMinutes(_config.FundingRateCacheDurationMinutes));

                // Fetch perpetual prices
                var perpPrices = await connector.GetPerpetualPricesAsync(symbols);
                snapshot.PerpPrices[exchangeName] = perpPrices;
                _cache.Set($"{CACHE_KEY_PREFIX_PERP}{exchangeName}", perpPrices,
                    TimeSpan.FromMinutes(_config.FundingRateCacheDurationMinutes));

                // Fetch 24h volumes and enrich funding rates
                var volumes = await connector.Get24hVolumeAsync(symbols);
                foreach (var rate in rates)
                {
                    if (volumes.TryGetValue(rate.Symbol, out var volume))
                    {
                        rate.Volume24h = volume;
                    }
                }

                _logger.LogInformation("Fetched and cached data for {Exchange}: {RateCount} rates, {SpotCount} spot prices, {PerpCount} perp prices, {VolumeCount} volumes",
                    exchangeName, rates.Count, spotPrices.Count, perpPrices.Count, volumes.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching data from {Exchange}", exchangeName);
            }
        }

        return snapshot;
    }

    public Dictionary<string, List<FundingRateDto>> GetCachedFundingRates()
    {
        var result = new Dictionary<string, List<FundingRateDto>>();

        // Note: We'd need to track exchange names separately to iterate efficiently
        // For now, return empty - this would be populated during fetch

        return result;
    }

    public Dictionary<string, Dictionary<string, SpotPriceDto>> GetCachedSpotPrices()
    {
        var result = new Dictionary<string, Dictionary<string, SpotPriceDto>>();

        // Iterate through known exchanges (from config)
        foreach (var exchange in _config.Exchanges.Where(e => e.IsEnabled))
        {
            var cacheKey = $"{CACHE_KEY_PREFIX_SPOT}{exchange.Name}";
            if (_cache.TryGetValue<Dictionary<string, SpotPriceDto>>(cacheKey, out var prices))
            {
                result[exchange.Name] = prices;
            }
        }

        return result;
    }

    public Dictionary<string, Dictionary<string, decimal>> GetCachedPerpPrices()
    {
        var result = new Dictionary<string, Dictionary<string, decimal>>();

        foreach (var exchange in _config.Exchanges.Where(e => e.IsEnabled))
        {
            var cacheKey = $"{CACHE_KEY_PREFIX_PERP}{exchange.Name}";
            if (_cache.TryGetValue<Dictionary<string, decimal>>(cacheKey, out var prices))
            {
                result[exchange.Name] = prices;
            }
        }

        return result;
    }
}
