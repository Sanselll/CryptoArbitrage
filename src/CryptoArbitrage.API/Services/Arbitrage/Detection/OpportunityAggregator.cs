using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Events;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Services.Arbitrage.Detection;

/// <summary>
/// Aggregates funding rates and market prices, then detects arbitrage opportunities.
/// PURE aggregator: Waits for fresh data from multiple sources before processing.
/// Ensures temporal consistency - only processes when ALL required data is fresh.
/// </summary>
public class OpportunityAggregator : IHostedService
{
    private readonly ILogger<OpportunityAggregator> _logger;
    private readonly IDataCollectionEventBus _eventBus;
    private readonly IDataRepository<FundingRateDto> _fundingRateRepository;
    private readonly IDataRepository<MarketDataSnapshot> _marketPriceRepository;
    private readonly IOpportunityDetectionService _opportunityDetectionService;
    private readonly IServiceScopeFactory _serviceScopeFactory;

    // Track freshness of data sources
    private DateTime? _lastFundingRatesUpdate;
    private DateTime? _lastMarketPricesUpdate;
    private readonly TimeSpan _freshnessWindow = TimeSpan.FromSeconds(60);

    // Lock for thread-safe aggregation
    private readonly SemaphoreSlim _aggregationLock = new(1, 1);

    public OpportunityAggregator(
        ILogger<OpportunityAggregator> logger,
        IDataCollectionEventBus eventBus,
        IDataRepository<FundingRateDto> fundingRateRepository,
        IDataRepository<MarketDataSnapshot> marketPriceRepository,
        IOpportunityDetectionService opportunityDetectionService,
        IServiceScopeFactory serviceScopeFactory)
    {
        _logger = logger;
        _eventBus = eventBus;
        _fundingRateRepository = fundingRateRepository;
        _marketPriceRepository = marketPriceRepository;
        _opportunityDetectionService = opportunityDetectionService;
        _serviceScopeFactory = serviceScopeFactory;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        // Subscribe to data collection events
        _eventBus.Subscribe<IDictionary<string, FundingRateDto>>(
            DataCollectionConstants.EventTypes.FundingRatesCollected,
            OnFundingRatesCollectedAsync);

        _eventBus.Subscribe<IDictionary<string, MarketDataSnapshot>>(
            DataCollectionConstants.EventTypes.MarketPricesCollected,
            OnMarketPricesCollectedAsync);

        _logger.LogInformation("OpportunityAggregator started and subscribed to events");
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("OpportunityAggregator stopped");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Called when funding rates are collected
    /// </summary>
    private async Task OnFundingRatesCollectedAsync(DataCollectionEvent<IDictionary<string, FundingRateDto>> @event)
    {
        _lastFundingRatesUpdate = @event.Timestamp;
        _logger.LogDebug("Funding rates updated at {Timestamp}", @event.Timestamp);

        // Try to aggregate if all sources are fresh
        await TryAggregateAndDetectAsync();
    }

    /// <summary>
    /// Called when market prices are collected
    /// </summary>
    private async Task OnMarketPricesCollectedAsync(DataCollectionEvent<IDictionary<string, MarketDataSnapshot>> @event)
    {
        _lastMarketPricesUpdate = @event.Timestamp;
        _logger.LogDebug("Market prices updated at {Timestamp}", @event.Timestamp);

        // Try to aggregate if all sources are fresh
        await TryAggregateAndDetectAsync();
    }

    /// <summary>
    /// Attempts to aggregate data and detect opportunities if all sources are fresh
    /// </summary>
    private async Task TryAggregateAndDetectAsync()
    {
        // Check if we have fresh data from all sources
        if (!IsDataFresh())
        {
            _logger.LogDebug(
                "Data not fresh enough for aggregation. Funding: {FundingAge}s ago, Prices: {PriceAge}s ago",
                GetDataAge(_lastFundingRatesUpdate),
                GetDataAge(_lastMarketPricesUpdate));
            return;
        }

        // Acquire lock to prevent concurrent aggregations
        if (!await _aggregationLock.WaitAsync(0))
        {
            _logger.LogDebug("Aggregation already in progress, skipping");
            return;
        }

        try
        {
            await AggregateAndDetectOpportunitiesAsync();
        }
        finally
        {
            _aggregationLock.Release();
        }
    }

    /// <summary>
    /// Checks if all required data sources have been updated within the freshness window
    /// </summary>
    private bool IsDataFresh()
    {
        var now = DateTime.UtcNow;

        if (_lastFundingRatesUpdate == null || _lastMarketPricesUpdate == null)
        {
            return false;
        }

        var fundingAge = now - _lastFundingRatesUpdate.Value;
        var priceAge = now - _lastMarketPricesUpdate.Value;

        return fundingAge <= _freshnessWindow && priceAge <= _freshnessWindow;
    }

    /// <summary>
    /// Gets the age of data in seconds (for logging)
    /// </summary>
    private double GetDataAge(DateTime? timestamp)
    {
        if (timestamp == null) return -1;
        return (DateTime.UtcNow - timestamp.Value).TotalSeconds;
    }

    /// <summary>
    /// Aggregates data from all sources and detects opportunities
    /// </summary>
    private async Task AggregateAndDetectOpportunitiesAsync()
    {
        try
        {
            var startTime = DateTime.UtcNow;

            // Read funding rates from cache
            var fundingRatesDict = await _fundingRateRepository.GetByPatternAsync(DataCollectionConstants.CacheKeys.FundingRatePattern);

            // Read market prices from cache (single snapshot with all exchanges)
            var marketSnapshot = await _marketPriceRepository.GetAsync(DataCollectionConstants.CacheKeys.MarketDataSnapshot);
            var marketSnapshotsDict = marketSnapshot != null
                ? new Dictionary<string, MarketDataSnapshot> { { DataCollectionConstants.CacheKeys.MarketDataSnapshot, marketSnapshot } }
                : new Dictionary<string, MarketDataSnapshot>();

            _logger.LogDebug(
                "Aggregating data: {FundingCount} funding rates, {SnapshotCount} market snapshots",
                fundingRatesDict.Count,
                marketSnapshotsDict.Count);

            // Build complete MarketDataSnapshot
            var aggregatedSnapshot = BuildMarketSnapshot(fundingRatesDict, marketSnapshotsDict);


            // Detect opportunities
            var detectedOpportunities = await _opportunityDetectionService.DetectOpportunitiesAsync(aggregatedSnapshot);

            // Get open position keys
            var openPositionKeys = await GetOpenPositionKeysAsync();

            // Mark opportunities that have existing positions
            MarkExistingPositions(detectedOpportunities, openPositionKeys);

            var duration = DateTime.UtcNow - startTime;


            // Publish event with detected opportunities (with position flags and RL predictions)
            await _eventBus.PublishAsync(new DataCollectionEvent<List<ArbitrageOpportunityDto>>
            {
                EventType = DataCollectionConstants.EventTypes.OpportunitiesDetected,
                Data = detectedOpportunities,
                Timestamp = DateTime.UtcNow,
                CollectionDuration = duration,
                Success = true,
                ItemCount = detectedOpportunities.Count
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error aggregating data and detecting opportunities");
        }
    }

    /// <summary>
    /// Builds a complete MarketDataSnapshot from cached funding rates and market snapshots
    /// </summary>
    private MarketDataSnapshot BuildMarketSnapshot(
        IDictionary<string, FundingRateDto> fundingRatesDict,
        IDictionary<string, MarketDataSnapshot> marketSnapshotsDict)
    {
        var snapshot = new MarketDataSnapshot
        {
            FetchedAt = DateTime.UtcNow
        };

        // Group funding rates by exchange (to match OpportunityDetectionService expectations)
        foreach (var kvp in fundingRatesDict)
        {
            var fundingRate = kvp.Value;
            var exchange = fundingRate.Exchange;

            if (!snapshot.FundingRates.ContainsKey(exchange))
            {
                snapshot.FundingRates[exchange] = new List<FundingRateDto>();
            }

            snapshot.FundingRates[exchange].Add(fundingRate);
        }

        // Aggregate spot and perp prices from all exchange snapshots
        foreach (var kvp in marketSnapshotsDict)
        {
            var exchangeSnapshot = kvp.Value;

            // Merge spot prices
            foreach (var spotKvp in exchangeSnapshot.SpotPrices)
            {
                var exchange = spotKvp.Key;
                var prices = spotKvp.Value;

                if (!snapshot.SpotPrices.ContainsKey(exchange))
                {
                    snapshot.SpotPrices[exchange] = new Dictionary<string, PriceDto>();
                }

                foreach (var priceKvp in prices)
                {
                    snapshot.SpotPrices[exchange][priceKvp.Key] = priceKvp.Value;
                }
            }

            // Merge perp prices
            foreach (var perpKvp in exchangeSnapshot.PerpPrices)
            {
                var exchange = perpKvp.Key;
                var prices = perpKvp.Value;

                if (!snapshot.PerpPrices.ContainsKey(exchange))
                {
                    snapshot.PerpPrices[exchange] = new Dictionary<string, PriceDto>();
                }

                foreach (var priceKvp in prices)
                {
                    snapshot.PerpPrices[exchange][priceKvp.Key] = priceKvp.Value;
                }
            }
        }

        return snapshot;
    }

    /// <summary>
    /// Gets position keys for all open positions (from all users)
    /// Returns keys in format: "SYMBOL|LongExchange|ShortExchange"
    /// </summary>
    private async Task<HashSet<string>> GetOpenPositionKeysAsync()
    {
        var positionKeys = new HashSet<string>();

        try
        {
            // Create a scope to get DbContext (since this is a singleton hosted service)
            using var scope = _serviceScopeFactory.CreateScope();
            var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

            // Get ALL open positions (from all users)
            var openPositions = await dbContext.Positions
                .Where(p => p.Status == PositionStatus.Open)
                .ToListAsync();

            _logger.LogDebug("Found {Count} open positions", openPositions.Count);

            // Group positions by execution to find pairs
            var positionGroups = openPositions
                .GroupBy(p => p.ExecutionId)
                .ToList();

            foreach (var group in positionGroups)
            {
                var positions = group.ToList();
                if (positions.Count != 2) continue; // Skip incomplete pairs

                // Identify long and short positions
                var longPosition = positions.FirstOrDefault(p => p.Side == PositionSide.Long);
                var shortPosition = positions.FirstOrDefault(p => p.Side == PositionSide.Short);

                if (longPosition == null || shortPosition == null) continue;

                // Create position key
                var key = $"{longPosition.Symbol}|{longPosition.Exchange}|{shortPosition.Exchange}";
                positionKeys.Add(key);
            }

            _logger.LogInformation("Found {Count} active position pairs", positionKeys.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting open position keys");
        }

        return positionKeys;
    }

    /// <summary>
    /// Marks detected opportunities that have existing open positions
    /// Sets IsExistingPosition = true for matching opportunities
    /// </summary>
    private void MarkExistingPositions(
        List<ArbitrageOpportunityDto> opportunities,
        HashSet<string> positionKeys)
    {
        int markedCount = 0;

        foreach (var opp in opportunities)
        {
            var key = $"{opp.Symbol}|{opp.LongExchange}|{opp.ShortExchange}";
            if (positionKeys.Contains(key))
            {
                opp.IsExistingPosition = true;
                markedCount++;
            }
        }

        _logger.LogDebug(
            "Marked {Marked} out of {Total} opportunities as existing positions",
            markedCount,
            opportunities.Count);
    }
}
