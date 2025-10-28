namespace CryptoArbitrage.HistoricalCollector.Models;

/// <summary>
/// Represents a simulated position execution with all features and outcomes
/// Used for generating ML training data
/// </summary>
public class SimulatedExecution
{
    // ===================================================================
    // INPUT FEATURES (X) - What we know at entry time
    // ===================================================================

    /// <summary>
    /// Full opportunity snapshot as JSON (for complete reconstruction)
    /// </summary>
    public string OpportunitySnapshotJson { get; set; } = string.Empty;

    // Timing
    public DateTime EntryTime { get; set; }
    public DateTime ExitTime { get; set; }
    public int HourOfDay { get; set; }
    public int DayOfWeek { get; set; }

    // Market context at entry
    public decimal BtcPriceAtEntry { get; set; }
    public string MarketRegimeAtEntry { get; set; } = string.Empty;

    // Opportunity features (extracted for quick access and CSV export)
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string LongExchange { get; set; } = string.Empty;
    public string ShortExchange { get; set; } = string.Empty;

    // Funding rate details
    public decimal LongFundingRate { get; set; }
    public decimal ShortFundingRate { get; set; }
    public int? LongFundingIntervalHours { get; set; }
    public int? ShortFundingIntervalHours { get; set; }
    public decimal? LongNextFundingTimeMinutes { get; set; }  // Minutes until next funding (easier for ML)
    public decimal? ShortNextFundingTimeMinutes { get; set; } // Minutes until next funding

    // Price spread
    public decimal? CurrentPriceSpreadPercent { get; set; }

    // Profitability metrics at entry
    public decimal FundProfit8h { get; set; }
    public decimal FundApr { get; set; }
    public decimal? FundProfit8h24hProj { get; set; }
    public decimal? FundApr24hProj { get; set; }
    public decimal? FundBreakEvenTime24hProj { get; set; }
    public decimal? FundProfit8h3dProj { get; set; }
    public decimal? FundApr3dProj { get; set; }
    public decimal? FundBreakEvenTime3dProj { get; set; }
    public decimal? BreakEvenTimeHours { get; set; }

    // Price spread statistics
    public decimal? PriceSpread24hAvg { get; set; }
    public decimal? PriceSpread3dAvg { get; set; }

    // Risk metrics
    public decimal? SpreadVolatilityCv { get; set; }
    public decimal? SpreadVolatilityStdDev { get; set; }
    public decimal? Spread30SampleAvg { get; set; }

    // Liquidity metrics
    public decimal Volume24h { get; set; }
    public decimal? LongVolume24h { get; set; }
    public decimal? ShortVolume24h { get; set; }
    public decimal? BidAskSpreadPercent { get; set; }
    public decimal? OrderbookDepthUsd { get; set; }
    public string? LiquidityStatus { get; set; }

    // Position cost
    public decimal PositionCostPercent { get; set; }
    public decimal PositionSizeUsd { get; set; } = 1000m; // Default simulation size

    // ===================================================================
    // TARGET VARIABLES (y) - What we're trying to predict
    // ===================================================================

    /// <summary>
    /// Strategy used for this simulation
    /// </summary>
    public string StrategyName { get; set; } = string.Empty;

    /// <summary>
    /// Why the position was exited
    /// </summary>
    public string ExitReason { get; set; } = string.Empty;

    /// <summary>
    /// Actual hold duration in hours
    /// </summary>
    public decimal ActualHoldHours { get; set; }

    /// <summary>
    /// Actual profit percentage (including all fees and slippage)
    /// </summary>
    public decimal ActualProfitPercent { get; set; }

    /// <summary>
    /// Actual profit in USD
    /// </summary>
    public decimal ActualProfitUsd { get; set; }

    /// <summary>
    /// Binary: was this position profitable?
    /// </summary>
    public bool WasProfitable { get; set; }

    /// <summary>
    /// Did this exit hit the profit target?
    /// </summary>
    public bool HitProfitTarget { get; set; }

    /// <summary>
    /// Did this exit hit the stop loss?
    /// </summary>
    public bool HitStopLoss { get; set; }

    // ===================================================================
    // PERFORMANCE METRICS
    // ===================================================================

    /// <summary>
    /// Peak unrealized profit during hold period
    /// </summary>
    public decimal PeakUnrealizedProfitPercent { get; set; }

    /// <summary>
    /// Maximum drawdown during hold period
    /// </summary>
    public decimal MaxDrawdownPercent { get; set; }

    /// <summary>
    /// Number of funding payments received
    /// </summary>
    public int FundingPaymentsCount { get; set; }

    /// <summary>
    /// Total funding earned in USD
    /// </summary>
    public decimal TotalFundingEarnedUsd { get; set; }

    // ===================================================================
    // EXECUTION QUALITY METRICS
    // ===================================================================

    /// <summary>
    /// Total fees in USD (trading fees for entry and exit)
    /// </summary>
    public decimal TotalFeesUsd { get; set; }

    // ===================================================================
    // PRICE DATA
    // ===================================================================
    // NOTE: These prices already include slippage adjustments
    // - Entry long: market price + slippage (paying ask)
    // - Entry short: market price - slippage (receiving bid)
    // - Exit long: market price - slippage (receiving bid)
    // - Exit short: market price + slippage (paying ask)

    public decimal EntryLongPrice { get; set; }
    public decimal EntryShortPrice { get; set; }
    public decimal ExitLongPrice { get; set; }
    public decimal ExitShortPrice { get; set; }

    // ===================================================================
    // TIME-SERIES SNAPSHOTS
    // ===================================================================

    /// <summary>
    /// Position snapshots taken during the position lifecycle.
    /// Sampled every 5 minutes from entry to exit.
    /// Used for training exit prediction models.
    /// Empty for entry-only training data (backward compatibility).
    /// </summary>
    public List<PositionSnapshot> Snapshots { get; set; } = new();
}
