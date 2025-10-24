using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Models.Suggestions;

namespace CryptoArbitrage.API.Services.Suggestions;

/// <summary>
/// Monitors positions and generates exit signals based on multiple conditions
/// </summary>
public class ExitStrategyMonitor
{
    private readonly ILogger<ExitStrategyMonitor> _logger;

    // Exit condition thresholds
    private const decimal FUNDING_REVERSAL_THRESHOLD = 0.30m; // 30% deterioration
    private const decimal VOLUME_DROP_THRESHOLD = 0.40m; // 40% volume drop
    private const decimal SPREAD_WIDENING_THRESHOLD = 0.50m; // 50% spread widening

    public ExitStrategyMonitor(ILogger<ExitStrategyMonitor> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Evaluates all exit conditions for a position and returns triggered signals
    /// </summary>
    public List<ExitSignal> EvaluateExitConditions(
        Position position,
        Execution? execution,
        MarketDataSnapshot currentMarketData)
    {
        var signals = new List<ExitSignal>();

        try
        {
            // Get current market data for this position's symbol
            var currentFundingRate = GetCurrentFundingRate(currentMarketData, position.Symbol, position.Exchange);
            var currentPrices = GetCurrentPrices(currentMarketData, position.Symbol, position.Exchange);
            var currentVolume = GetCurrentVolume(currentMarketData, position.Symbol, position.Exchange);

            // Condition 1: Profit Target
            var profitSignal = EvaluateProfitTarget(position, execution);
            if (profitSignal != null) signals.Add(profitSignal);

            // Condition 2: Funding Rate Reversal
            if (position.EntryFundingRate.HasValue && currentFundingRate.HasValue)
            {
                var fundingSignal = EvaluateFundingReversal(position, currentFundingRate.Value);
                if (fundingSignal != null) signals.Add(fundingSignal);
            }

            // Condition 3: Time Limit
            if (position.MaxHoldingHours.HasValue)
            {
                var timeSignal = EvaluateTimeLimit(position);
                if (timeSignal != null) signals.Add(timeSignal);
            }

            // Condition 4: Market Degradation
            var marketSignal = EvaluateMarketDegradation(position, currentVolume, currentMarketData);
            if (marketSignal != null) signals.Add(marketSignal);

            if (signals.Any())
            {
                _logger.LogInformation(
                    "Exit signals triggered for Position {PositionId} ({Symbol}): {SignalCount} conditions met",
                    position.Id, position.Symbol, signals.Count);
            }

            return signals;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating exit conditions for Position {PositionId}", position.Id);
            return signals;
        }
    }

    private ExitSignal? EvaluateProfitTarget(Position position, Execution? execution)
    {
        if (!position.ProfitTargetPercent.HasValue || execution == null)
        {
            return null;
        }

        decimal targetProfitPercent = position.ProfitTargetPercent.Value;
        decimal currentProfitPercent = CalculateCurrentProfitPercent(position, execution);

        if (currentProfitPercent >= targetProfitPercent)
        {
            decimal confidence = Math.Min(100m, 70m + (30m * (currentProfitPercent / targetProfitPercent)));

            return new ExitSignal
            {
                ConditionType = ExitConditionType.ProfitTarget,
                IsTriggered = true,
                Confidence = confidence,
                Urgency = DetermineUrgency(confidence),
                RecommendedAction = "Close position - profit target reached",
                Message = $"Profit target of {targetProfitPercent:F2}% reached. Current profit: {currentProfitPercent:F2}%",
                CurrentValue = currentProfitPercent,
                ThresholdValue = targetProfitPercent
            };
        }

        return null;
    }

    private ExitSignal? EvaluateFundingReversal(Position position, decimal currentFundingRate)
    {
        decimal entryRate = position.EntryFundingRate!.Value;

        // Check for sign flip (most critical)
        bool signFlipped = Math.Sign(entryRate) != Math.Sign(currentFundingRate) && Math.Abs(entryRate) > 0.0001m;

        if (signFlipped)
        {
            return new ExitSignal
            {
                ConditionType = ExitConditionType.FundingReversal,
                IsTriggered = true,
                Confidence = 90m,
                Urgency = ExitUrgency.High,
                RecommendedAction = "Close position immediately - funding rate reversed direction",
                Message = $"Funding rate flipped from {entryRate:F4} to {currentFundingRate:F4}. Position now paying instead of earning.",
                CurrentValue = currentFundingRate,
                ThresholdValue = entryRate
            };
        }

        // Check for significant deterioration (same sign but much worse)
        if (Math.Abs(entryRate) > 0.0001m) // Only if entry rate was meaningful
        {
            decimal deterioration = Math.Abs((currentFundingRate - entryRate) / entryRate);

            if (deterioration >= FUNDING_REVERSAL_THRESHOLD)
            {
                decimal confidence = Math.Min(100m, 60m + (deterioration * 100m));

                return new ExitSignal
                {
                    ConditionType = ExitConditionType.FundingReversal,
                    IsTriggered = true,
                    Confidence = confidence,
                    Urgency = deterioration >= 0.5m ? ExitUrgency.High : ExitUrgency.Medium,
                    RecommendedAction = "Consider closing position - funding rate deteriorated significantly",
                    Message = $"Funding rate deteriorated {deterioration:P0} from entry ({entryRate:F4} → {currentFundingRate:F4})",
                    CurrentValue = currentFundingRate,
                    ThresholdValue = entryRate
                };
            }
        }

        return null;
    }

    private ExitSignal? EvaluateTimeLimit(Position position)
    {
        decimal maxHours = position.MaxHoldingHours!.Value;
        decimal hoursHeld = (decimal)(DateTime.UtcNow - position.OpenedAt).TotalHours;

        if (hoursHeld >= maxHours)
        {
            decimal overage = hoursHeld - maxHours;
            decimal confidence = Math.Min(100m, 70m + (overage / maxHours * 30m));

            return new ExitSignal
            {
                ConditionType = ExitConditionType.TimeLimit,
                IsTriggered = true,
                Confidence = confidence,
                Urgency = overage > maxHours * 0.2m ? ExitUrgency.High : ExitUrgency.Medium,
                RecommendedAction = "Close position - maximum holding time reached",
                Message = $"Position held for {hoursHeld:F1} hours, exceeding max holding time of {maxHours:F1} hours",
                CurrentValue = hoursHeld,
                ThresholdValue = maxHours
            };
        }
        else if (hoursHeld >= maxHours * 0.9m) // Approaching limit (90%)
        {
            return new ExitSignal
            {
                ConditionType = ExitConditionType.TimeLimit,
                IsTriggered = false, // Warning, not triggered
                Confidence = 50m,
                Urgency = ExitUrgency.Low,
                RecommendedAction = "Monitor position - approaching time limit",
                Message = $"Position held for {hoursHeld:F1} hours, approaching max of {maxHours:F1} hours",
                CurrentValue = hoursHeld,
                ThresholdValue = maxHours
            };
        }

        return null;
    }

    private ExitSignal? EvaluateMarketDegradation(Position position, decimal? currentVolume, MarketDataSnapshot marketData)
    {
        var degradationFactors = new List<string>();
        decimal degradationScore = 0m;

        // Check volume drop
        if (currentVolume.HasValue)
        {
            // We don't have entry volume stored, so we compare against a reasonable baseline
            // In a production system, you might store entry volume or use historical averages
            if (currentVolume.Value < 100_000m) // Less than $100K volume
            {
                degradationFactors.Add($"Low current volume: ${currentVolume:N0}");
                degradationScore += 30m;
            }
        }

        // Check liquidity degradation
        var liquidityStatus = GetCurrentLiquidityStatus(marketData, position.Symbol, position.Exchange);
        if (liquidityStatus == LiquidityStatus.Low)
        {
            degradationFactors.Add("Liquidity degraded to Low status");
            degradationScore += 40m;
        }
        else if (liquidityStatus == LiquidityStatus.Medium)
        {
            degradationFactors.Add("Liquidity degraded to Medium status");
            degradationScore += 20m;
        }

        // Check spread widening (if we have bid-ask spread data)
        var currentSpread = GetCurrentBidAskSpread(marketData, position.Symbol, position.Exchange);
        if (currentSpread.HasValue && currentSpread.Value > 0.2m) // > 0.2% spread
        {
            degradationFactors.Add($"Wide bid-ask spread: {currentSpread.Value:P2}");
            degradationScore += 30m;
        }

        if (degradationScore >= 50m) // Significant degradation
        {
            return new ExitSignal
            {
                ConditionType = ExitConditionType.MarketDegradation,
                IsTriggered = true,
                Confidence = Math.Min(100m, degradationScore),
                Urgency = degradationScore >= 70m ? ExitUrgency.High : ExitUrgency.Medium,
                RecommendedAction = "Consider closing position - market quality degraded",
                Message = $"Market degradation detected: {string.Join(", ", degradationFactors)}",
                CurrentValue = degradationScore,
                ThresholdValue = 50m
            };
        }

        return null;
    }

    // Helper methods to extract data from MarketDataSnapshot

    private decimal? GetCurrentFundingRate(MarketDataSnapshot snapshot, string symbol, string exchange)
    {
        // FundingRates is Dictionary<string, List<FundingRateDto>>
        // Key format is typically "exchange"
        if (snapshot.FundingRates.TryGetValue(exchange, out var fundingRates))
        {
            var fundingRate = fundingRates.FirstOrDefault(fr => fr.Symbol == symbol);
            return fundingRate?.Rate;
        }
        return null;
    }

    private (decimal? Spot, decimal? Perp) GetCurrentPrices(MarketDataSnapshot snapshot, string symbol, string exchange)
    {
        // SpotPrices and PerpPrices are Dictionary<string, Dictionary<string, PriceDto>>
        // First key is exchange, second key is symbol
        decimal? spotPrice = null;
        decimal? perpPrice = null;

        if (snapshot.SpotPrices.TryGetValue(exchange, out var spotPrices))
        {
            if (spotPrices.TryGetValue(symbol, out var spot))
            {
                spotPrice = spot.Price;
            }
        }

        if (snapshot.PerpPrices.TryGetValue(exchange, out var perpPrices))
        {
            if (perpPrices.TryGetValue(symbol, out var perp))
            {
                perpPrice = perp.Price;
            }
        }

        return (spotPrice, perpPrice);
    }

    private decimal? GetCurrentVolume(MarketDataSnapshot snapshot, string symbol, string exchange)
    {
        if (snapshot.PerpPrices.TryGetValue(exchange, out var perpPrices))
        {
            if (perpPrices.TryGetValue(symbol, out var perp))
            {
                return perp.Volume24h;
            }
        }
        return null;
    }

    private LiquidityStatus GetCurrentLiquidityStatus(MarketDataSnapshot snapshot, string symbol, string exchange)
    {
        var volume = GetCurrentVolume(snapshot, symbol, exchange);

        // Simple heuristic based on volume
        if (volume.HasValue)
        {
            if (volume.Value >= 1_000_000m) return LiquidityStatus.Good;
            if (volume.Value >= 100_000m) return LiquidityStatus.Medium;
        }
        return LiquidityStatus.Low;
    }

    private decimal? GetCurrentBidAskSpread(MarketDataSnapshot snapshot, string symbol, string exchange)
    {
        // This would require bid/ask data in the snapshot
        // For now, return null as we don't have this data
        return null;
    }

    private decimal CalculateCurrentProfitPercent(Position position, Execution execution)
    {
        // Calculate profit as percentage of position value
        decimal positionValue = position.Quantity * position.EntryPrice;

        if (positionValue == 0) return 0m;

        // Total profit = PnL + Net Funding
        decimal totalProfit = position.RealizedPnL + position.UnrealizedPnL + position.NetFundingFee;

        // Also include execution-level funding if available
        if (execution != null)
        {
            totalProfit += execution.FundingEarned;
        }

        return (totalProfit / positionValue) * 100m;
    }

    private ExitUrgency DetermineUrgency(decimal confidence)
    {
        if (confidence >= 85m) return ExitUrgency.Critical;
        if (confidence >= 70m) return ExitUrgency.High;
        if (confidence >= 50m) return ExitUrgency.Medium;
        return ExitUrgency.Low;
    }
}
