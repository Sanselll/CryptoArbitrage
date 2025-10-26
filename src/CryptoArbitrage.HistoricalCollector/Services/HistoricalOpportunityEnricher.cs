using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.HistoricalCollector.Services;

/// <summary>
/// Enriches detected opportunities with volume, liquidity, and spread metrics
/// Adapted from backend OpportunityEnricher but works with historical data
/// </summary>
public class HistoricalOpportunityEnricher
{
    private readonly ILogger<HistoricalOpportunityEnricher> _logger;
    private readonly ArbitrageConfig _config;

    public HistoricalOpportunityEnricher(
        ILogger<HistoricalOpportunityEnricher> logger,
        ArbitrageConfig config)
    {
        _logger = logger;
        _config = config;
    }

    /// <summary>
    /// Enrich opportunities with volume, liquidity, and spread metrics
    /// </summary>
    public async Task<List<ArbitrageOpportunityDto>> EnrichOpportunitiesAsync(
        List<ArbitrageOpportunityDto> opportunities,
        MarketDataSnapshot marketDataSnapshot,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>> liquidityMetrics)
    {
        if (opportunities == null || opportunities.Count == 0)
        {
            return opportunities ?? new List<ArbitrageOpportunityDto>();
        }

        try
        {
            _logger.LogDebug("Enriching {Count} opportunities", opportunities.Count);

            foreach (var opportunity in opportunities)
            {
                await EnrichSingleOpportunityAsync(opportunity, marketDataSnapshot, liquidityMetrics);
            }

            return opportunities;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error enriching opportunities");
            return opportunities;
        }
    }

    /// <summary>
    /// Enriches a single opportunity with all metrics
    /// </summary>
    private async Task EnrichSingleOpportunityAsync(
        ArbitrageOpportunityDto opportunity,
        MarketDataSnapshot marketDataSnapshot,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>> liquidityMetrics)
    {
        try
        {
            // Phase 1: Volume enrichment
            EnrichVolume(opportunity, marketDataSnapshot);

            // Phase 2: Liquidity enrichment
            EnrichLiquidity(opportunity, liquidityMetrics);

            // Phase 3: Spread metrics (for cross-exchange only)
            EnrichSpreadMetrics(opportunity, marketDataSnapshot);

            await Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error enriching opportunity {Symbol}", opportunity.Symbol);
        }
    }

    /// <summary>
    /// Enriches opportunity with 24h volume from market data
    /// Logic copied from backend OpportunityEnricher.EnrichVolumeAsync
    /// </summary>
    private void EnrichVolume(
        ArbitrageOpportunityDto opportunity,
        MarketDataSnapshot marketDataSnapshot)
    {
        try
        {
            if (marketDataSnapshot == null)
            {
                _logger.LogDebug("No market data snapshot available for volume enrichment");
                return;
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
    }

    /// <summary>
    /// Enriches opportunity with liquidity metrics
    /// Logic copied from backend OpportunityEnricher.EnrichLiquidityAsync
    /// </summary>
    private void EnrichLiquidity(
        ArbitrageOpportunityDto opportunity,
        Dictionary<string, Dictionary<string, LiquidityMetricsDto>> liquidityMetrics)
    {
        try
        {
            if (opportunity.Strategy == ArbitrageStrategy.SpotPerpetual)
            {
                // Get liquidity for single exchange
                if (liquidityMetrics.TryGetValue(opportunity.Exchange, out var exchangeMetrics))
                {
                    if (exchangeMetrics.TryGetValue(opportunity.Symbol, out var metrics))
                    {
                        ApplyLiquidityMetrics(opportunity, metrics);
                    }
                }
            }
            else // CrossExchange
            {
                // Get liquidity for both exchanges
                LiquidityMetricsDto? longMetrics = null;
                LiquidityMetricsDto? shortMetrics = null;

                if (liquidityMetrics.TryGetValue(opportunity.LongExchange, out var longExchangeMetrics))
                {
                    longExchangeMetrics.TryGetValue(opportunity.Symbol, out longMetrics);
                }

                if (liquidityMetrics.TryGetValue(opportunity.ShortExchange, out var shortExchangeMetrics))
                {
                    shortExchangeMetrics.TryGetValue(opportunity.Symbol, out shortMetrics);
                }

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
    }

    /// <summary>
    /// Applies liquidity metrics to opportunity and evaluates liquidity status
    /// Logic copied from backend OpportunityEnricher.ApplyLiquidityMetrics
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
    /// Logic copied from backend OpportunityEnricher.EvaluateLiquidity
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
    /// Enriches opportunity with spread metrics from price history (Cross-Exchange only)
    /// Logic copied from backend OpportunityEnricher.EnrichSpreadMetricsAsync
    /// </summary>
    private void EnrichSpreadMetrics(
        ArbitrageOpportunityDto opportunity,
        MarketDataSnapshot marketDataSnapshot)
    {
        try
        {
            // Only process Cross-Exchange opportunities
            if (opportunity.Strategy != ArbitrageStrategy.CrossExchange)
            {
                return;
            }

            if (marketDataSnapshot?.PerpPriceHistory == null)
            {
                _logger.LogDebug("No price history available for spread metrics enrichment");
                return;
            }

            // Get price histories for long and short exchanges
            PriceHistoryDto? longHistory = null;
            PriceHistoryDto? shortHistory = null;

            if (marketDataSnapshot.PerpPriceHistory.TryGetValue(opportunity.LongExchange, out var longExchangeHistory))
            {
                longExchangeHistory.TryGetValue(opportunity.Symbol, out longHistory);
            }

            if (marketDataSnapshot.PerpPriceHistory.TryGetValue(opportunity.ShortExchange, out var shortExchangeHistory))
            {
                shortExchangeHistory.TryGetValue(opportunity.Symbol, out shortHistory);
            }

            // Both histories must exist and have matching sample counts
            if (longHistory == null || shortHistory == null)
            {
                _logger.LogDebug("Price history not available for {Symbol} on {LongExchange} or {ShortExchange}",
                    opportunity.Symbol, opportunity.LongExchange, opportunity.ShortExchange);
                return;
            }

            if (longHistory.SampleCount == 0 || shortHistory.SampleCount == 0)
            {
                _logger.LogDebug("Insufficient price history samples for {Symbol}", opportunity.Symbol);
                return;
            }

            // Calculate spreads for each sample pair
            // Use the minimum count to ensure we have matching pairs
            int sampleCount = Math.Min(longHistory.SampleCount, shortHistory.SampleCount);
            var spreads = new List<decimal>();

            for (int i = 0; i < sampleCount; i++)
            {
                var longPrice = longHistory.PriceHistory[i];
                var shortPrice = shortHistory.PriceHistory[i];

                // Spread = (ShortPrice - LongPrice) / LongPrice
                if (longPrice > 0)
                {
                    var spread = (shortPrice - longPrice) / longPrice;
                    spreads.Add(spread);
                }
            }

            if (spreads.Count == 0)
            {
                return;
            }

            // Calculate average spread
            var avgSpread = spreads.Average();
            opportunity.Spread30SampleAvg = avgSpread;

            // Calculate standard deviation
            var variance = spreads.Sum(s => (s - avgSpread) * (s - avgSpread)) / spreads.Count;
            var stdDev = (decimal)Math.Sqrt((double)variance);
            opportunity.SpreadVolatilityStdDev = stdDev;

            // Calculate coefficient of variation (CV = StdDev / Mean)
            // Avoid division by zero
            if (Math.Abs(avgSpread) > 0.0000001m)
            {
                opportunity.SpreadVolatilityCv = stdDev / Math.Abs(avgSpread);
            }

            _logger.LogDebug(
                "Enriched {Symbol} with spread metrics: Avg={Avg:P4}, StdDev={StdDev:P4}, CV={CV:F2} ({Samples} samples)",
                opportunity.Symbol,
                avgSpread,
                stdDev,
                opportunity.SpreadVolatilityCv,
                spreads.Count);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error enriching spread metrics for {Symbol}", opportunity.Symbol);
        }
    }
}
