using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Constants;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Services.DataCollection.Events;
using CryptoArbitrage.API.Services.Suggestions;
using Microsoft.Extensions.Options;

namespace CryptoArbitrage.API.Services.Arbitrage.Detection;

/// <summary>
/// Enriches detected opportunities with supplementary data (volume, liquidity metrics, AI suggestions).
/// PURE enricher: Receives opportunities, adds data, publishes enriched version.
/// </summary>
public class OpportunityEnricher : IHostedService
{
    private readonly ILogger<OpportunityEnricher> _logger;
    private readonly IDataCollectionEventBus _eventBus;
    private readonly IDataRepository<MarketDataSnapshot> _marketDataRepository;
    private readonly IDataRepository<LiquidityMetricsDto> _liquidityRepository;
    private readonly IDataRepository<ArbitrageOpportunityDto> _opportunityRepository;
    private readonly IDataRepository<UserDataSnapshot> _userDataRepository;
    private readonly OpportunitySuggestionService _suggestionService;
    private readonly ArbitrageConfig _config;

    public OpportunityEnricher(
        ILogger<OpportunityEnricher> logger,
        IDataCollectionEventBus eventBus,
        IDataRepository<MarketDataSnapshot> marketDataRepository,
        IDataRepository<LiquidityMetricsDto> liquidityRepository,
        IDataRepository<ArbitrageOpportunityDto> opportunityRepository,
        IDataRepository<UserDataSnapshot> userDataRepository,
        OpportunitySuggestionService suggestionService,
        IOptions<ArbitrageConfig> config)
    {
        _logger = logger;
        _eventBus = eventBus;
        _marketDataRepository = marketDataRepository;
        _liquidityRepository = liquidityRepository;
        _opportunityRepository = opportunityRepository;
        _userDataRepository = userDataRepository;
        _suggestionService = suggestionService;
        _config = config.Value;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        // Subscribe to opportunities detected event
        _eventBus.Subscribe<List<ArbitrageOpportunityDto>>(
            DataCollectionConstants.EventTypes.OpportunitiesDetected,
            OnOpportunitiesDetectedAsync);

        _logger.LogInformation("OpportunityEnricher started and subscribed to OpportunitiesDetected events");
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("OpportunityEnricher stopped");
        return Task.CompletedTask;
    }

    /// <summary>
    /// Called when opportunities are detected
    /// </summary>
    private async Task OnOpportunitiesDetectedAsync(DataCollectionEvent<List<ArbitrageOpportunityDto>> @event)
    {
        if (@event.Data == null || @event.Data.Count == 0)
        {
            _logger.LogDebug("No opportunities to enrich");

            // Still publish event even if empty
            await PublishEnrichedOpportunitiesAsync(@event.Data ?? new List<ArbitrageOpportunityDto>());
            return;
        }

        try
        {
            var startTime = DateTime.UtcNow;

            _logger.LogDebug("Enriching {Count} opportunities", @event.Data.Count);

            // Enrich each opportunity
            var enrichedOpportunities = await EnrichOpportunitiesAsync(@event.Data);

            var duration = DateTime.UtcNow - startTime;

            _logger.LogInformation(
                "Enriched {Count} opportunities in {Duration}ms",
                enrichedOpportunities.Count,
                duration.TotalMilliseconds);

            // Publish enriched opportunities
            await PublishEnrichedOpportunitiesAsync(enrichedOpportunities);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error enriching opportunities");

            // On error, publish original opportunities without enrichment
            await PublishEnrichedOpportunitiesAsync(@event.Data);
        }
    }

    /// <summary>
    /// Enriches opportunities with volume and liquidity data
    /// </summary>
    private async Task<List<ArbitrageOpportunityDto>> EnrichOpportunitiesAsync(List<ArbitrageOpportunityDto> opportunities)
    {
        try
        {
            // Fetch market data snapshot from cache (contains Volume24h in PriceDto)
            var marketDataSnapshot = await _marketDataRepository.GetAsync(
                DataCollectionConstants.CacheKeys.MarketDataSnapshot);

            // Fetch all liquidity metrics from cache
            var liquidityMetricsDict = await _liquidityRepository.GetByPatternAsync(
                DataCollectionConstants.CacheKeys.LiquidityMetricPattern);

            // Enrich each opportunity
            foreach (var opportunity in opportunities)
            {
                await EnrichSingleOpportunityAsync(opportunity, marketDataSnapshot, liquidityMetricsDict);
            }

            return opportunities;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error enriching opportunities, returning opportunities without enrichment");
            return opportunities;
        }
    }

    /// <summary>
    /// Enriches a single opportunity with volume, liquidity data, and AI suggestion
    /// </summary>
    private async Task EnrichSingleOpportunityAsync(
        ArbitrageOpportunityDto opportunity,
        MarketDataSnapshot? marketDataSnapshot,
        IDictionary<string, LiquidityMetricsDto> liquidityMetricsDict)
    {
        try
        {
            // Phase 1: Copy Volume24h from market data
            await EnrichVolumeAsync(opportunity, marketDataSnapshot);

            // Phase 2: Apply liquidity metrics
            await EnrichLiquidityAsync(opportunity, liquidityMetricsDict);

            // Phase 3: Generate AI-driven trading suggestion
            await EnrichSuggestionAsync(opportunity);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error enriching opportunity {Symbol}", opportunity.Symbol);
        }
    }

    /// <summary>
    /// Enriches opportunity with 24h volume from market data
    /// </summary>
    private Task EnrichVolumeAsync(
        ArbitrageOpportunityDto opportunity,
        MarketDataSnapshot? marketDataSnapshot)
    {
        try
        {
            if (marketDataSnapshot == null)
            {
                _logger.LogDebug("No market data snapshot available for volume enrichment");
                return Task.CompletedTask;
            }

            if (opportunity.Strategy == ArbitrageStrategy.SpotPerpetual)
            {
                // For spot-perpetual, get volume from perpetual prices
                if (marketDataSnapshot.PerpPrices.TryGetValue(opportunity.Exchange, out var perpPrices))
                {
                    if (perpPrices.TryGetValue(opportunity.Symbol, out var priceDto))
                    {
                        opportunity.Volume24h = priceDto.Volume24h;
                        _logger.LogDebug("Enriched {Symbol} on {Exchange} with Volume24h: ${Volume}",
                            opportunity.Symbol, opportunity.Exchange, priceDto.Volume24h);
                    }
                }
            }
            else // CrossExchange
            {
                // Get volume from both exchanges and use the minimum (bottleneck)
                decimal longVolume = 0, shortVolume = 0;

                // Get long exchange volume (perp)
                if (marketDataSnapshot.PerpPrices.TryGetValue(opportunity.LongExchange, out var longPerpPrices))
                {
                    if (longPerpPrices.TryGetValue(opportunity.Symbol, out var longPrice))
                    {
                        longVolume = longPrice.Volume24h;
                    }
                }

                // Get short exchange volume (perp)
                if (marketDataSnapshot.PerpPrices.TryGetValue(opportunity.ShortExchange, out var shortPerpPrices))
                {
                    if (shortPerpPrices.TryGetValue(opportunity.Symbol, out var shortPrice))
                    {
                        shortVolume = shortPrice.Volume24h;
                    }
                }

                // Store individual exchange volumes
                opportunity.LongVolume24h = longVolume;
                opportunity.ShortVolume24h = shortVolume;

                // Use the minimum of the two volumes (bottleneck)
                opportunity.Volume24h = Math.Min(
                    longVolume > 0 ? longVolume : decimal.MaxValue,
                    shortVolume > 0 ? shortVolume : decimal.MaxValue);

                if (opportunity.Volume24h == decimal.MaxValue)
                    opportunity.Volume24h = 0;

                _logger.LogDebug("Enriched {Symbol} cross-exchange with Volume24h: ${Volume} (Long: ${LongVol}, Short: ${ShortVol})",
                    opportunity.Symbol, opportunity.Volume24h, longVolume, shortVolume);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error enriching volume for {Symbol}", opportunity.Symbol);
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Enriches opportunity with liquidity metrics
    /// </summary>
    private Task EnrichLiquidityAsync(
        ArbitrageOpportunityDto opportunity,
        IDictionary<string, LiquidityMetricsDto> liquidityMetricsDict)
    {
        try
        {
            // For spot-perpetual: check liquidity on the exchange
            // For cross-exchange: check liquidity on both exchanges (use worst case)
            if (opportunity.Strategy == ArbitrageStrategy.SpotPerpetual)
            {
                var liquidityKey = DataCollectionConstants.CacheKeys.BuildLiquidityMetricKey(
                    opportunity.Exchange, opportunity.Symbol);

                if (liquidityMetricsDict.TryGetValue(liquidityKey, out var metrics))
                {
                    ApplyLiquidityMetrics(opportunity, metrics);
                }
            }
            else // CrossExchange
            {
                // Get liquidity for both exchanges
                var longLiquidityKey = DataCollectionConstants.CacheKeys.BuildLiquidityMetricKey(
                    opportunity.LongExchange, opportunity.Symbol);
                var shortLiquidityKey = DataCollectionConstants.CacheKeys.BuildLiquidityMetricKey(
                    opportunity.ShortExchange, opportunity.Symbol);

                liquidityMetricsDict.TryGetValue(longLiquidityKey, out var longMetrics);
                liquidityMetricsDict.TryGetValue(shortLiquidityKey, out var shortMetrics);

                // Use worst-case scenario (highest spread, lowest depth)
                if (longMetrics != null && shortMetrics != null)
                {
                    var worstCaseMetrics = new LiquidityMetricsDto
                    {
                        BidAskSpreadPercent = Math.Max(longMetrics.BidAskSpreadPercent, shortMetrics.BidAskSpreadPercent),
                        OrderbookDepthUsd = Math.Min(longMetrics.OrderbookDepthUsd, shortMetrics.OrderbookDepthUsd)
                    };
                    ApplyLiquidityMetrics(opportunity, worstCaseMetrics);
                }
                else if (longMetrics != null)
                {
                    ApplyLiquidityMetrics(opportunity, longMetrics);
                }
                else if (shortMetrics != null)
                {
                    ApplyLiquidityMetrics(opportunity, shortMetrics);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error enriching liquidity for {Symbol}", opportunity.Symbol);
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Applies liquidity metrics to opportunity and evaluates liquidity status
    /// </summary>
    private void ApplyLiquidityMetrics(ArbitrageOpportunityDto opportunity, LiquidityMetricsDto metrics)
    {
        opportunity.BidAskSpreadPercent = metrics.BidAskSpreadPercent;
        opportunity.OrderbookDepthUsd = metrics.OrderbookDepthUsd;

        // Evaluate liquidity status based on thresholds
        var status = EvaluateLiquidity(metrics.BidAskSpreadPercent, metrics.OrderbookDepthUsd);
        opportunity.LiquidityStatus = status;

        // Set warning message if liquidity is not good
        if (status == LiquidityStatus.Medium)
        {
            opportunity.LiquidityWarning = "Medium liquidity - execution may have some slippage";
        }
        else if (status == LiquidityStatus.Low)
        {
            opportunity.LiquidityWarning = "Low liquidity - high slippage risk";
        }

        _logger.LogDebug("Enriched {Symbol} with liquidity: Spread={Spread}%, Depth=${Depth}, Status={Status}",
            opportunity.Symbol, metrics.BidAskSpreadPercent * 100, metrics.OrderbookDepthUsd, status);
    }

    /// <summary>
    /// Evaluates liquidity status based on bid-ask spread and orderbook depth
    /// </summary>
    private LiquidityStatus EvaluateLiquidity(decimal bidAskSpreadPercent, decimal orderbookDepthUsd)
    {
        // Check if both metrics meet "Good" thresholds
        bool spreadIsGood = bidAskSpreadPercent <= _config.MaxBidAskSpreadPercent;
        bool depthIsGood = orderbookDepthUsd >= _config.MinOrderbookDepthUsd;

        if (spreadIsGood && depthIsGood)
        {
            return LiquidityStatus.Good;
        }

        // Check if metrics are too far from thresholds (Low liquidity)
        // Consider "Low" if spread is more than 2x threshold or depth is less than 50% of threshold
        bool spreadIsBad = bidAskSpreadPercent > (_config.MaxBidAskSpreadPercent * 2);
        bool depthIsBad = orderbookDepthUsd < (_config.MinOrderbookDepthUsd * 0.5m);

        if (spreadIsBad || depthIsBad)
        {
            return LiquidityStatus.Low;
        }

        // Otherwise, it's Medium liquidity
        return LiquidityStatus.Medium;
    }

    /// <summary>
    /// Enriches opportunity with AI-driven trading suggestion
    /// </summary>
    private async Task EnrichSuggestionAsync(ArbitrageOpportunityDto opportunity)
    {
        try
        {
            // Fetch available balances from all users (aggregate for suggestions)
            var availableBalances = await GetAggregatedAvailableBalancesAsync();

            _logger.LogInformation(
                "Fetched balances for {Symbol}: {BalanceCount} exchanges, Total: ${Total:N2}",
                opportunity.Symbol,
                availableBalances.Count,
                availableBalances.Values.Sum());

            // Generate suggestion using the suggestion service (with balance awareness)
            var suggestion = _suggestionService.GenerateSuggestion(opportunity, availableBalances);
            opportunity.Suggestion = suggestion;

            _logger.LogInformation(
                "Generated suggestion for {Symbol}: Strategy={Strategy}, Confidence={Confidence:F1}, Recommendation={Recommendation}, Size=${Size:N0}, Leverage={Leverage:F1}x, FundingQuality={FQ:F1}, ProfitPotential={PP:F1}, SpreadEfficiency={SE:F1}, MarketQuality={MQ:F1}",
                opportunity.Symbol,
                suggestion.RecommendedStrategy,
                suggestion.ConfidenceScore,
                suggestion.EntryRecommendation,
                suggestion.SuggestedPositionSizeUsd,
                suggestion.SuggestedLeverage,
                suggestion.ScoreBreakdown.FundingQualityScore,
                suggestion.ScoreBreakdown.ProfitPotentialScore,
                suggestion.ScoreBreakdown.SpreadEfficiencyScore,
                suggestion.ScoreBreakdown.MarketQualityScore);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error generating suggestion for {Symbol}", opportunity.Symbol);
            // Don't fail enrichment if suggestion generation fails
            opportunity.Suggestion = null;
        }
    }

    /// <summary>
    /// Gets aggregated available balances across all users by exchange
    /// This provides total buying power for position sizing
    /// </summary>
    private async Task<IDictionary<string, decimal>> GetAggregatedAvailableBalancesAsync()
    {
        try
        {
            // Fetch all user data snapshots (pattern: "userdata:*")
            var userDataSnapshots = await _userDataRepository.GetByPatternAsync("userdata:*");

            if (userDataSnapshots == null || !userDataSnapshots.Any())
            {
                _logger.LogWarning("No user balance data available for suggestions - position sizing will use defaults");
                return new Dictionary<string, decimal>();
            }

            _logger.LogInformation("Found {Count} user data snapshots for balance aggregation", userDataSnapshots.Count);

            // Aggregate balances by exchange
            var balancesByExchange = new Dictionary<string, decimal>();

            foreach (var snapshot in userDataSnapshots.Values)
            {
                if (snapshot?.Balance != null)
                {
                    var exchange = snapshot.Exchange;
                    var availableBalance = snapshot.Balance.FuturesAvailableUsd;

                    if (!balancesByExchange.ContainsKey(exchange))
                    {
                        balancesByExchange[exchange] = 0m;
                    }

                    balancesByExchange[exchange] += availableBalance;

                    _logger.LogDebug(
                        "Added balance from user snapshot: Exchange={Exchange}, Available=${Available:N2}",
                        exchange, availableBalance);
                }
                else
                {
                    _logger.LogWarning("User snapshot found but Balance is null: {Snapshot}", snapshot);
                }
            }

            _logger.LogInformation(
                "Aggregated balances for suggestions: {Balances}",
                string.Join(", ", balancesByExchange.Select(kvp => $"{kvp.Key}: ${kvp.Value:N2}")));

            return balancesByExchange;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error fetching user balances for suggestions");
            return new Dictionary<string, decimal>();
        }
    }

    /// <summary>
    /// Publishes enriched opportunities event
    /// </summary>
    private async Task PublishEnrichedOpportunitiesAsync(List<ArbitrageOpportunityDto> opportunities)
    {
        // Store opportunities in repository (cache) for retrieval on page refresh
        try
        {
            var opportunitiesDict = opportunities.ToDictionary(
                o => $"opportunity:{o.UniqueKey}",
                o => o
            );

            await _opportunityRepository.StoreBatchAsync(
                opportunitiesDict,
                TimeSpan.FromMinutes(5));

            _logger.LogDebug("Stored {Count} opportunities in cache", opportunities.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error storing opportunities in cache");
        }

        // Publish event for SignalR broadcasting
        await _eventBus.PublishAsync(new DataCollectionEvent<List<ArbitrageOpportunityDto>>
        {
            EventType = DataCollectionConstants.EventTypes.OpportunitiesEnriched,
            Data = opportunities,
            Timestamp = DateTime.UtcNow,
            CollectionDuration = TimeSpan.Zero,
            Success = true,
            ItemCount = opportunities.Count
        });

        _logger.LogDebug("Published OpportunitiesEnriched event with {Count} items", opportunities.Count);
    }
}
