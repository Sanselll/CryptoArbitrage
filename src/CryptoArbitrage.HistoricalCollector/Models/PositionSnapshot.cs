namespace CryptoArbitrage.HistoricalCollector.Models;

/// <summary>
/// Represents a single snapshot of a position's state during its lifecycle.
/// Used for generating time-series training data for exit prediction models.
/// Each simulated execution will have multiple snapshots (one every 5 minutes).
/// </summary>
public class PositionSnapshot
{
    // ===================================================================
    // IDENTIFICATION
    // ===================================================================

    /// <summary>
    /// ID of the parent simulated execution
    /// </summary>
    public int ExecutionId { get; set; }

    /// <summary>
    /// Snapshot sequence number (0 = entry, increases every 5 min)
    /// </summary>
    public int SnapshotIndex { get; set; }

    /// <summary>
    /// Time this snapshot was taken
    /// </summary>
    public DateTime SnapshotTime { get; set; }

    /// <summary>
    /// Position entry time (from parent execution)
    /// </summary>
    public DateTime EntryTime { get; set; }

    // ===================================================================
    // STATIC FEATURES (from entry - don't change during position)
    // These are copied from the opportunity at entry time
    // ===================================================================

    // Basic info
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string LongExchange { get; set; } = string.Empty;
    public string ShortExchange { get; set; } = string.Empty;

    // Entry timing
    public int EntryHourOfDay { get; set; }
    public int EntryDayOfWeek { get; set; }

    // Entry funding rates
    public decimal EntryLongFundingRate { get; set; }
    public decimal EntryShortFundingRate { get; set; }
    public decimal EntryFundingRateDifferential { get; set; }
    public decimal EntryFundProfit8h { get; set; }
    public decimal EntryFundApr { get; set; }

    // Entry funding projections
    public decimal? EntryFundProfit8h24hProj { get; set; }
    public decimal? EntryFundApr24hProj { get; set; }
    public decimal? EntryFundProfit8h3dProj { get; set; }
    public decimal? EntryFundApr3dProj { get; set; }

    // Entry spread metrics
    public decimal? EntryPriceSpreadPercent { get; set; }
    public decimal? EntryPriceSpread24hAvg { get; set; }
    public decimal? EntryPriceSpread3dAvg { get; set; }
    public decimal? EntrySpread30SampleAvg { get; set; }
    public decimal? EntrySpreadVolatilityStdDev { get; set; }
    public decimal? EntrySpreadVolatilityCv { get; set; }

    // Entry liquidity
    public decimal EntryVolume24h { get; set; }
    public decimal? EntryBidAskSpreadPercent { get; set; }
    public decimal? EntryOrderbookDepthUsd { get; set; }

    // ML predictions at entry
    public decimal? MLPredictedProfitPercent { get; set; }
    public decimal? MLPredictedSuccessProbability { get; set; }
    public decimal? MLPredictedDurationHours { get; set; }

    // ===================================================================
    // DYNAMIC FEATURES (current position state at this snapshot)
    // These change as the position evolves
    // ===================================================================

    /// <summary>
    /// Hours since position was opened
    /// </summary>
    public decimal TimeInPositionHours { get; set; }

    /// <summary>
    /// Current unrealized P&L percentage
    /// </summary>
    public decimal CurrentPnLPercent { get; set; }

    /// <summary>
    /// Peak unrealized P&L achieved so far
    /// </summary>
    public decimal PeakPnLPercent { get; set; }

    /// <summary>
    /// Current drawdown from peak (negative value)
    /// </summary>
    public decimal DrawdownFromPeakPercent { get; set; }

    /// <summary>
    /// Maximum drawdown experienced so far
    /// </summary>
    public decimal MaxDrawdownPercent { get; set; }

    /// <summary>
    /// P&L velocity: rate of change per hour
    /// Calculated as change in P&L since last snapshot / time delta
    /// </summary>
    public decimal? PnLVelocityPerHour { get; set; }

    /// <summary>
    /// Number of funding payments received so far
    /// </summary>
    public int FundingPaymentsReceived { get; set; }

    /// <summary>
    /// Total funding earned so far (USD)
    /// </summary>
    public decimal FundingEarnedUsd { get; set; }

    /// <summary>
    /// Total funding earned as percentage of position size
    /// </summary>
    public decimal FundingEarnedPercent { get; set; }

    // Current market state
    /// <summary>
    /// Current funding rate differential
    /// </summary>
    public decimal CurrentFundingRateDifferential { get; set; }

    /// <summary>
    /// Change in funding rate differential since entry
    /// </summary>
    public decimal FundingRateDifferentialChange { get; set; }

    /// <summary>
    /// Funding reversal magnitude: how much rates have reversed (0-1)
    /// Formula: (entry_diff - current_diff) / entry_diff
    /// </summary>
    public decimal FundingReversalMagnitude { get; set; }

    /// <summary>
    /// Current price spread percentage
    /// </summary>
    public decimal? CurrentPriceSpreadPercent { get; set; }

    /// <summary>
    /// Change in spread since entry (percentage points)
    /// Positive = widened (worse), Negative = narrowed (better)
    /// </summary>
    public decimal? SpreadChangeSinceEntryPercent { get; set; }

    /// <summary>
    /// Current spread volatility
    /// </summary>
    public decimal? CurrentSpreadVolatilityStdDev { get; set; }

    /// <summary>
    /// Ratio of current volatility to entry volatility
    /// > 1.0 = more volatile, < 1.0 = less volatile
    /// </summary>
    public decimal? VolatilityChangeRatio { get; set; }

    /// <summary>
    /// Current 24h trading volume
    /// </summary>
    public decimal CurrentVolume24h { get; set; }

    /// <summary>
    /// Volume change ratio (current / entry)
    /// </summary>
    public decimal VolumeChangeRatio { get; set; }

    /// <summary>
    /// Minutes until next funding payment
    /// </summary>
    public decimal MinutesToNextFunding { get; set; }

    // ===================================================================
    // ENGINEERED FEATURES (derived from dynamic features)
    // ===================================================================

    /// <summary>
    /// Position maturity: time_in_position / predicted_duration
    /// > 1.0 means held longer than predicted
    /// </summary>
    public decimal? PositionMaturity { get; set; }

    /// <summary>
    /// Hold efficiency: profit per hour held
    /// Formula: current_pnl / time_in_position
    /// </summary>
    public decimal HoldEfficiency { get; set; }

    /// <summary>
    /// Has reached predicted optimal duration?
    /// </summary>
    public bool OptimalTimeReached { get; set; }

    /// <summary>
    /// Consecutive snapshots with declining P&L
    /// </summary>
    public int ConsecutiveNegativeSamples { get; set; }

    /// <summary>
    /// Consecutive snapshots with improving P&L
    /// </summary>
    public int ConsecutivePositiveSamples { get; set; }

    // ===================================================================
    // LABELS (for ML training - known from optimal hindsight)
    // ===================================================================

    /// <summary>
    /// Should exit now? (0 or 1)
    /// Labeled as 1 if within 30 minutes of optimal exit
    /// OR if continuing causes >0.5% additional loss
    /// </summary>
    public int ShouldExitNow { get; set; }

    /// <summary>
    /// Exit reason (only populated if ShouldExitNow = 1)
    /// Values: PROFIT_TARGET, STOP_LOSS, TRAILING_STOP, FUNDING_REVERSAL,
    ///         VOLATILITY_SPIKE, MAX_HOLD_TIME, OPTIMAL_HINDSIGHT
    /// </summary>
    public string? ExitReason { get; set; }

    /// <summary>
    /// Hours until optimal exit (from this snapshot)
    /// Always >= 0 (clamped to 0 at optimal exit point)
    /// </summary>
    public decimal HoursUntilOptimalExit { get; set; }

    /// <summary>
    /// P&L at optimal exit point (for reference)
    /// </summary>
    public decimal OptimalExitPnLPercent { get; set; }

    /// <summary>
    /// P&L that will be lost if continuing from this snapshot
    /// Formula: current_pnl - optimal_exit_pnl
    /// Positive = would lose profit, Negative = still gaining
    /// </summary>
    public decimal PotentialPnLLoss { get; set; }
}
