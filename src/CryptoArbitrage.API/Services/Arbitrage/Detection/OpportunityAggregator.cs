using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Events;
using CryptoArbitrage.API.Services.ML;
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
    private readonly RLPredictionService _rlPredictionService;
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
        RLPredictionService rlPredictionService,
        IServiceScopeFactory serviceScopeFactory)
    {
        _logger = logger;
        _eventBus = eventBus;
        _fundingRateRepository = fundingRateRepository;
        _marketPriceRepository = marketPriceRepository;
        _opportunityDetectionService = opportunityDetectionService;
        _rlPredictionService = rlPredictionService;
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

            // Add RL predictions to opportunities
            await EnrichWithRLPredictionsAsync(detectedOpportunities);

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

    /// <summary>
    /// Enriches opportunities with RL model predictions
    /// Adds ENTER probability, confidence, and state value estimates
    /// </summary>
    private async Task EnrichWithRLPredictionsAsync(List<ArbitrageOpportunityDto> opportunities)
    {
        if (opportunities.Count == 0)
            return;

        try
        {
            // Build portfolio state
            var portfolioState = await BuildPortfolioStateAsync();

            // Get RL predictions for all opportunities (ML API handles batching internally)
            var predictions = await _rlPredictionService.EvaluateOpportunitiesAsync(
                opportunities,
                portfolioState);

            // RL prediction enrichment removed - fields no longer exist in ArbitrageOpportunityDto

            _logger.LogInformation(
                "Enriched {Count} opportunities with RL predictions",
                predictions.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to enrich opportunities with RL predictions");
            // Don't fail the entire pipeline if RL predictions fail
        }
    }

    /// <summary>
    /// Builds current portfolio state for RL model evaluation
    /// </summary>
    private async Task<RLPortfolioState> BuildPortfolioStateAsync()
    {
        try
        {
            using var scope = _serviceScopeFactory.CreateScope();
            var dbContext = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

            // Get all open positions for all users (include transactions for funding fee calculation)
            var openPositions = await dbContext.Positions
                .Include(p => p.Transactions)
                .Where(p => p.Status == PositionStatus.Open)
                .ToListAsync();

            // Calculate portfolio metrics
            decimal totalCapital = 10000m; // Default initial capital
            decimal totalPnL = openPositions.Sum(p => p.RealizedPnLUsd + p.UnrealizedPnL);
            decimal totalMargin = openPositions.Sum(p => p.InitialMargin);

            var portfolioState = new RLPortfolioState
            {
                Capital = totalCapital + totalPnL,
                InitialCapital = totalCapital,
                NumPositions = openPositions.Count,
                Utilization = totalCapital > 0 ? (float)(totalMargin / totalCapital) : 0f,
                TotalPnlPct = totalCapital > 0 ? (float)(totalPnL / totalCapital * 100) : 0f,
                Drawdown = 0f, // Calculate if needed
                Positions = openPositions.Take(3).Select(p =>
                {
                    // Calculate net funding fee from transactions
                    var fundingReceived = p.Transactions
                        .Where(t => t.TransactionType == TransactionType.FundingFee && t.Amount > 0)
                        .Sum(t => t.Amount);
                    var fundingPaid = p.Transactions
                        .Where(t => t.TransactionType == TransactionType.FundingFee && t.Amount < 0)
                        .Sum(t => Math.Abs(t.Amount));
                    var netFunding = fundingReceived - fundingPaid;

                    return new RLPositionState
                    {
                        PnlPct = p.InitialMargin > 0 ? (float)(p.UnrealizedPnL / p.InitialMargin * 100) : 0f,
                        HoursHeld = (float)(DateTime.UtcNow - p.OpenedAt).TotalHours,
                        FundingRate = p.InitialMargin > 0
                            ? (float)(netFunding / p.InitialMargin / (decimal)Math.Max(1, (DateTime.UtcNow - p.OpenedAt).TotalHours) * 8 * 100)
                            : 0f
                    };
                }).ToList()
            };

            return portfolioState;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to build portfolio state for RL predictions");

            // Return default state if error
            return new RLPortfolioState
            {
                Capital = 10000m,
                InitialCapital = 10000m,
                NumPositions = 0,
                Utilization = 0f,
                TotalPnlPct = 0f,
                Drawdown = 0f,
                Positions = new List<RLPositionState>()
            };
        }
    }
}
