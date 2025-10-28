using CryptoArbitrage.API.Models;
using CryptoArbitrage.HistoricalCollector.Models;
using Microsoft.Extensions.Logging;

namespace CryptoArbitrage.HistoricalCollector.Services;

/// <summary>
/// Generates position snapshots during simulated executions.
/// Handles snapshot creation, feature calculation, and labeling for ML training.
/// </summary>
public class SnapshotGenerator
{
    private readonly ILogger<SnapshotGenerator> _logger;

    public SnapshotGenerator(ILogger<SnapshotGenerator> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Intermediate data structure to track position state at a checkpoint
    /// </summary>
    public class CheckpointData
    {
        public int Index { get; set; }
        public DateTime Time { get; set; }
        public decimal UnrealizedPnLPercent { get; set; }
        public decimal LongPrice { get; set; }
        public decimal ShortPrice { get; set; }
        public decimal LongFundingRate { get; set; }
        public decimal ShortFundingRate { get; set; }
        public decimal SpreadPercent { get; set; }
        public decimal Volume24h { get; set; }
        public decimal CumulativeFundingPercent { get; set; }
    }

    /// <summary>
    /// Generate snapshots from checkpoint data collected during simulation
    /// </summary>
    public List<PositionSnapshot> GenerateSnapshots(
        int executionId,
        ArbitrageOpportunityDto opportunity,
        HistoricalMarketSnapshot entrySnapshot,
        List<CheckpointData> checkpoints,
        int optimalExitIndex,
        string exitReason,
        decimal optimalExitPnL)
    {
        var snapshots = new List<PositionSnapshot>();

        if (checkpoints.Count == 0)
            return snapshots;

        var entryTime = entrySnapshot.Timestamp;

        // Track derived metrics across snapshots
        decimal peakPnL = decimal.MinValue;
        decimal maxDrawdown = 0m;
        decimal? previousPnL = null;
        int consecutiveNegative = 0;
        int consecutivePositive = 0;

        for (int i = 0; i < checkpoints.Count; i++)
        {
            var checkpoint = checkpoints[i];
            var timeInPositionHours = (decimal)(checkpoint.Time - entryTime).TotalHours;

            // Update peak and drawdown
            peakPnL = Math.Max(peakPnL, checkpoint.UnrealizedPnLPercent);
            maxDrawdown = Math.Min(maxDrawdown, checkpoint.UnrealizedPnLPercent);
            var drawdownFromPeak = checkpoint.UnrealizedPnLPercent - peakPnL;

            // Calculate P&L velocity (if we have previous snapshot)
            decimal? pnlVelocity = null;
            if (previousPnL.HasValue && timeInPositionHours > 0)
            {
                var pnlChange = checkpoint.UnrealizedPnLPercent - previousPnL.Value;
                pnlVelocity = pnlChange / 0.0833m; // per hour (assuming 5min intervals)
            }

            // Track consecutive negative/positive P&L movements
            if (previousPnL.HasValue)
            {
                if (checkpoint.UnrealizedPnLPercent < previousPnL.Value)
                {
                    consecutiveNegative++;
                    consecutivePositive = 0;
                }
                else if (checkpoint.UnrealizedPnLPercent > previousPnL.Value)
                {
                    consecutivePositive++;
                    consecutiveNegative = 0;
                }
            }

            // Calculate current funding rate differential
            var currentFundingDiff = checkpoint.ShortFundingRate - checkpoint.LongFundingRate;
            var entryFundingDiff = opportunity.ShortFundingRate - opportunity.LongFundingRate;
            var fundingDiffChange = currentFundingDiff - entryFundingDiff;
            var fundingReversalMagnitude = entryFundingDiff != 0
                ? Math.Abs((entryFundingDiff - currentFundingDiff) / entryFundingDiff)
                : 0m;

            // Calculate spread changes
            var entrySpread = opportunity.CurrentPriceSpreadPercent ?? 0m;
            var spreadChange = checkpoint.SpreadPercent - entrySpread;

            // Calculate volume change
            var entryVolume = opportunity.Volume24h;
            var volumeChangeRatio = entryVolume > 0 ? checkpoint.Volume24h / entryVolume : 1m;

            // Calculate position maturity
            var positionMaturity = opportunity.MLPredictedDurationHours.HasValue && opportunity.MLPredictedDurationHours.Value > 0
                ? timeInPositionHours / opportunity.MLPredictedDurationHours.Value
                : (decimal?)null;

            // Calculate hold efficiency
            var holdEfficiency = timeInPositionHours > 0
                ? checkpoint.UnrealizedPnLPercent / timeInPositionHours
                : 0m;

            // Check if optimal time reached
            var optimalTimeReached = opportunity.MLPredictedDurationHours.HasValue
                && timeInPositionHours >= opportunity.MLPredictedDurationHours.Value;

            // Calculate hours until optimal exit
            var hoursUntilExit = i < optimalExitIndex
                ? (decimal)(checkpoints[optimalExitIndex].Time - checkpoint.Time).TotalHours
                : 0m;

            // Calculate potential P&L loss if continuing
            var potentialLoss = checkpoint.UnrealizedPnLPercent - optimalExitPnL;

            // Create snapshot
            var snapshot = new PositionSnapshot
            {
                // Identification
                ExecutionId = executionId,
                SnapshotIndex = i,
                SnapshotTime = checkpoint.Time,
                EntryTime = entryTime,

                // Basic info (static)
                Symbol = opportunity.Symbol,
                Strategy = opportunity.SubType.ToString(),
                LongExchange = opportunity.LongExchange,
                ShortExchange = opportunity.ShortExchange,

                // Entry timing
                EntryHourOfDay = entryTime.Hour,
                EntryDayOfWeek = (int)entryTime.DayOfWeek,

                // Entry funding rates (static)
                EntryLongFundingRate = opportunity.LongFundingRate,
                EntryShortFundingRate = opportunity.ShortFundingRate,
                EntryFundingRateDifferential = entryFundingDiff,
                EntryFundProfit8h = opportunity.FundProfit8h,
                EntryFundApr = opportunity.FundApr,
                EntryFundProfit8h24hProj = opportunity.FundProfit8h24hProj,
                EntryFundApr24hProj = opportunity.FundApr24hProj,
                EntryFundProfit8h3dProj = opportunity.FundProfit8h3dProj,
                EntryFundApr3dProj = opportunity.FundApr3dProj,

                // Entry spread (static)
                EntryPriceSpreadPercent = opportunity.CurrentPriceSpreadPercent,
                EntryPriceSpread24hAvg = opportunity.PriceSpread24hAvg,
                EntryPriceSpread3dAvg = opportunity.PriceSpread3dAvg,
                EntrySpread30SampleAvg = opportunity.Spread30SampleAvg,
                EntrySpreadVolatilityStdDev = opportunity.SpreadVolatilityStdDev,
                EntrySpreadVolatilityCv = opportunity.SpreadVolatilityCv,

                // Entry liquidity (static)
                EntryVolume24h = opportunity.Volume24h,
                EntryBidAskSpreadPercent = opportunity.BidAskSpreadPercent,
                EntryOrderbookDepthUsd = opportunity.OrderbookDepthUsd,

                // ML predictions at entry (static)
                MLPredictedProfitPercent = opportunity.MLPredictedProfitPercent,
                MLPredictedSuccessProbability = opportunity.MLSuccessProbability,
                MLPredictedDurationHours = opportunity.MLPredictedDurationHours,

                // Dynamic features (current position state)
                TimeInPositionHours = timeInPositionHours,
                CurrentPnLPercent = checkpoint.UnrealizedPnLPercent,
                PeakPnLPercent = peakPnL,
                DrawdownFromPeakPercent = drawdownFromPeak,
                MaxDrawdownPercent = maxDrawdown,
                PnLVelocityPerHour = pnlVelocity,

                // Funding (for now, simplified - could be enhanced)
                FundingPaymentsReceived = 0, // TODO: Calculate from actual funding events
                FundingEarnedUsd = 0m,
                FundingEarnedPercent = 0m,

                // Current market state
                CurrentFundingRateDifferential = currentFundingDiff,
                FundingRateDifferentialChange = fundingDiffChange,
                FundingReversalMagnitude = fundingReversalMagnitude,
                CurrentPriceSpreadPercent = checkpoint.SpreadPercent,
                SpreadChangeSinceEntryPercent = spreadChange,
                CurrentSpreadVolatilityStdDev = null, // TODO: Calculate if needed
                VolatilityChangeRatio = null,
                CurrentVolume24h = checkpoint.Volume24h,
                VolumeChangeRatio = volumeChangeRatio,
                MinutesToNextFunding = 240m, // TODO: Calculate actual

                // Engineered features
                PositionMaturity = positionMaturity,
                HoldEfficiency = holdEfficiency,
                OptimalTimeReached = optimalTimeReached,
                ConsecutiveNegativeSamples = consecutiveNegative,
                ConsecutivePositiveSamples = consecutivePositive,

                // Labels (will be set by LabelSnapshots method)
                ShouldExitNow = 0,
                ExitReason = null,
                HoursUntilOptimalExit = hoursUntilExit,
                OptimalExitPnLPercent = optimalExitPnL,
                PotentialPnLLoss = potentialLoss
            };

            snapshots.Add(snapshot);
            previousPnL = checkpoint.UnrealizedPnLPercent;
        }

        // Label all snapshots
        LabelSnapshots(snapshots, optimalExitIndex, exitReason, optimalExitPnL);

        return snapshots;
    }

    /// <summary>
    /// Label snapshots with exit signals based on optimal hindsight
    /// </summary>
    private void LabelSnapshots(
        List<PositionSnapshot> snapshots,
        int optimalExitIndex,
        string exitReason,
        decimal optimalExitPnL)
    {
        if (snapshots.Count == 0 || optimalExitIndex >= snapshots.Count)
            return;

        for (int i = 0; i < snapshots.Count; i++)
        {
            var snapshot = snapshots[i];
            var hoursUntilExit = snapshot.HoursUntilOptimalExit;

            // Label 1: should_exit_now
            // Mark as 1 if:
            // - Within 30 minutes (0.5 hours) of optimal exit
            // - OR continuing would cause >0.5% loss of profit
            var withinExitWindow = hoursUntilExit <= 0.5m;
            var wouldLoseSignificantProfit = snapshot.PotentialPnLLoss > 0.5m;

            snapshot.ShouldExitNow = (withinExitWindow || wouldLoseSignificantProfit) ? 1 : 0;

            // Label 2: exit_reason (only for exit snapshots)
            if (snapshot.ShouldExitNow == 1)
            {
                snapshot.ExitReason = exitReason;
            }

            // HoursUntilOptimalExit is already set in GenerateSnapshots
        }

        _logger.LogDebug(
            "Labeled {Total} snapshots: {ExitSignals} exit signals ({Percent:F1}%)",
            snapshots.Count,
            snapshots.Count(s => s.ShouldExitNow == 1),
            snapshots.Count > 0 ? snapshots.Count(s => s.ShouldExitNow == 1) * 100.0 / snapshots.Count : 0);
    }
}
