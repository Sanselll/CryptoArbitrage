using System.Diagnostics;
using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.Data;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;

namespace CryptoArbitrage.API.Services.DataCollection;

/// <summary>
/// Background service that collects production data snapshots every 5 minutes for ML training
/// </summary>
public class ProductionDataCollectionService : BackgroundService
{
    private readonly ILogger<ProductionDataCollectionService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly IConfiguration _configuration;
    private readonly bool _enabled;
    private readonly int _intervalMinutes;

    public ProductionDataCollectionService(
        ILogger<ProductionDataCollectionService> logger,
        IServiceProvider serviceProvider,
        IConfiguration configuration)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _configuration = configuration;

        _enabled = _configuration.GetValue<bool>("ProductionDataCollection:Enabled", true);
        _intervalMinutes = _configuration.GetValue<int>("ProductionDataCollection:IntervalMinutes", 5);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!_enabled)
        {
            _logger.LogInformation("Production data collection is disabled");
            return;
        }

        _logger.LogInformation(
            "Production data collection started. Interval: {IntervalMinutes} minutes",
            _intervalMinutes);

        // Wait for initial startup
        await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Calculate time until next collection interval
                var now = DateTime.UtcNow;
                var nextRun = GetNextIntervalTime(now, _intervalMinutes);
                var delay = nextRun - now;

                if (delay.TotalMilliseconds > 0)
                {
                    _logger.LogDebug(
                        "Next collection at {NextRun} (in {Delay})",
                        nextRun.ToString("HH:mm:ss"), delay.ToString(@"mm\:ss"));

                    await Task.Delay(delay, stoppingToken);
                }

                // Collect snapshot
                await CollectSnapshotAsync(stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in production data collection cycle");
                // Wait a bit before retrying
                await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
            }
        }

        _logger.LogInformation("Production data collection stopped");
    }

    private async Task CollectSnapshotAsync(CancellationToken cancellationToken)
    {
        var stopwatch = Stopwatch.StartNew();
        var timestamp = DateTime.UtcNow;

        try
        {
            using var scope = _serviceProvider.CreateScope();

            var marketDataService = scope.ServiceProvider.GetRequiredService<IMarketDataService>();
            var fundingRateRepository = scope.ServiceProvider.GetRequiredService<IDataRepository<FundingRateDto>>();
            var opportunityRepository = scope.ServiceProvider.GetRequiredService<IDataRepository<ArbitrageOpportunityDto>>();
            var persister = scope.ServiceProvider.GetRequiredService<ProductionDataPersister>();

            _logger.LogDebug("Starting production data collection at {Timestamp}", timestamp.ToString("yyyy-MM-dd HH:mm:ss"));

            // Get market prices snapshot
            var marketSnapshot = await marketDataService.GetMarketDataSnapshotAsync();
            if (marketSnapshot == null)
            {
                _logger.LogWarning("Market data snapshot is null, skipping collection");
                return;
            }

            // Get funding rates from repository using correct pattern
            var fundingRatesDict = await fundingRateRepository.GetByPatternAsync(
                DataCollectionConstants.CacheKeys.FundingRatePattern,
                cancellationToken);
            var fundingRates = fundingRatesDict.Values.ToList();

            _logger.LogDebug(
                "Collected funding rates: {Count} rates from repository",
                fundingRates.Count);

            // Get all opportunities
            var opportunitiesDict = await opportunityRepository.GetByPatternAsync("opportunity:*", cancellationToken);
            var opportunities = opportunitiesDict.Values.ToList();

            // Build symbol market data from snapshot and funding rates
            var symbolMarketData = BuildSymbolMarketData(marketSnapshot, fundingRates, timestamp, _intervalMinutes);

            // Create production snapshot
            var snapshot = new ProductionSnapshot
            {
                Timestamp = timestamp,
                IntervalMinutes = _intervalMinutes,
                MarketData = symbolMarketData,
                Opportunities = opportunities,
                Metadata = new CollectionMetadata
                {
                    TotalSymbols = symbolMarketData.Count,
                    TotalOpportunities = opportunities.Count,
                    CollectionDurationMs = stopwatch.ElapsedMilliseconds,
                    Source = "production"
                }
            };

            // Save snapshot
            await persister.SaveSnapshotAsync(snapshot, cancellationToken);

            stopwatch.Stop();
            _logger.LogDebug(
                "Production data collection completed. Symbols: {SymbolCount}, Opportunities: {OpportunityCount}, Duration: {Duration}ms",
                snapshot.Metadata.TotalSymbols, snapshot.Metadata.TotalOpportunities, stopwatch.ElapsedMilliseconds);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to collect production data snapshot");
            throw;
        }
    }

    private List<SymbolMarketData> BuildSymbolMarketData(
        MarketDataSnapshot snapshot,
        List<FundingRateDto> fundingRates,
        DateTime collectionTime,
        int intervalMinutes)
    {
        var result = new List<SymbolMarketData>();

        // Group funding rates by exchange and symbol for lookup
        var fundingByExchangeAndSymbol = fundingRates
            .GroupBy(f => f.Exchange)
            .ToDictionary(
                g => g.Key,
                g => g.ToDictionary(f => f.Symbol));

        // Process each exchange separately
        var exchanges = new[] { "Binance", "Bybit" };

        foreach (var exchange in exchanges)
        {
            if (!snapshot.PerpPrices.TryGetValue(exchange, out var exchangePrices))
                continue;

            foreach (var (symbol, priceDto) in exchangePrices)
            {
                var marketData = new SymbolMarketData
                {
                    Symbol = symbol,
                    Exchange = exchange,
                    Price = priceDto.Price,
                    Volume24h = priceDto.Volume24h
                };

                // Get funding rate for this exchange-symbol pair
                if (fundingByExchangeAndSymbol.TryGetValue(exchange, out var exchangeFunding) &&
                    exchangeFunding.TryGetValue(symbol, out var fundingRate))
                {
                    // Check if funding is paid in this 5-minute window
                    bool isFundingPaid = IsFundingPaidInInterval(
                        fundingRate.NextFundingTime,
                        collectionTime,
                        intervalMinutes);

                    // Set rate to actual value if paid, otherwise 0
                    marketData.FundingRate = isFundingPaid ? fundingRate.Rate : 0.0m;
                    marketData.AnnualizedFundingRate = isFundingPaid ? fundingRate.AnnualizedRate : 0.0m;
                }
                else
                {
                    // No funding data available - set to 0
                    marketData.FundingRate = 0.0m;
                    marketData.AnnualizedFundingRate = 0.0m;
                }

                result.Add(marketData);
            }
        }

        return result;
    }

    /// <summary>
    /// Determines if a funding payment occurs within the NEXT collection interval window.
    /// We record the funding rate in the interval BEFORE payment so the data is available
    /// when the payment actually happens.
    /// Example: At 17:55, NextFundingTime=18:00 → returns TRUE (record rate now, payment in next interval)
    ///          At 18:04, NextFundingTime=19:00 → returns FALSE (payment is far in future)
    /// </summary>
    private static bool IsFundingPaidInInterval(DateTime nextFundingTime, DateTime collectionTime, int intervalMinutes)
    {
        // Round collection time to the interval boundary (e.g., 12:25:00 for 5-min intervals)
        var intervalStart = new DateTime(
            collectionTime.Year,
            collectionTime.Month,
            collectionTime.Day,
            collectionTime.Hour,
            (collectionTime.Minute / intervalMinutes) * intervalMinutes,
            0,
            DateTimeKind.Utc);

        var intervalEnd = intervalStart.AddMinutes(intervalMinutes);

        // Check if NextFundingTime falls within the NEXT interval [intervalEnd, intervalEnd + intervalMinutes)
        // This records the rate in the interval BEFORE payment occurs
        var nextIntervalStart = intervalEnd;
        var nextIntervalEnd = nextIntervalStart.AddMinutes(intervalMinutes);

        return nextFundingTime >= nextIntervalStart && nextFundingTime < nextIntervalEnd;
    }

    private static DateTime GetNextIntervalTime(DateTime now, int intervalMinutes)
    {
        // Calculate minutes into the current hour (round up if there are seconds)
        var currentMinute = now.Minute + (now.Second > 0 ? 1 : 0);

        // Find NEXT interval (always future, never current)
        var nextIntervalMinute = ((currentMinute / intervalMinutes) + 1) * intervalMinutes;

        if (nextIntervalMinute >= 60)
        {
            return new DateTime(now.Year, now.Month, now.Day, now.Hour, 0, 0, DateTimeKind.Utc)
                .AddHours(1);
        }

        return new DateTime(now.Year, now.Month, now.Day, now.Hour, nextIntervalMinute, 0, DateTimeKind.Utc);
    }
}
