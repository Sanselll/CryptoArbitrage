namespace CryptoArbitrage.HistoricalCollector.Config;

/// <summary>
/// Configuration for position exit strategies
/// Defines realistic exit conditions instead of fixed time-based exits
/// </summary>
public class ExitStrategyConfig
{
    /// <summary>
    /// Strategy name for identification
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Profit target percentage - exit when profit exceeds this
    /// Example: 2.0 = exit when +2% profit
    /// </summary>
    public decimal? ProfitTargetPercent { get; set; }

    /// <summary>
    /// Stop loss percentage - exit when loss exceeds this (negative value)
    /// Example: -3.0 = exit when -3% loss
    /// </summary>
    public decimal? StopLossPercent { get; set; }

    /// <summary>
    /// Trailing stop percentage - lock in profits after they're achieved
    /// Example: 0.5 = if profit reaches +2%, exit if it drops back to +1.5%
    /// </summary>
    public decimal? TrailingStopPercent { get; set; }

    /// <summary>
    /// Minimum profit required to activate trailing stop
    /// Example: 1.0 = trailing stop only activates after +1% profit
    /// </summary>
    public decimal? TrailingStopActivationPercent { get; set; }

    /// <summary>
    /// Exit when funding rate differential drops by this percentage of original
    /// Example: 0.5 = exit when differential drops to 50% of entry differential
    /// </summary>
    public decimal? FundingReversalThreshold { get; set; }

    /// <summary>
    /// Exit when spread volatility increases beyond this multiplier
    /// Example: 2.0 = exit if volatility doubles from entry
    /// </summary>
    public decimal? VolatilitySpikeMultiplier { get; set; }

    /// <summary>
    /// Maximum hold time in hours - always exit after this duration
    /// Example: 72 = force exit after 72 hours regardless of P&L
    /// </summary>
    public decimal MaxHoldHours { get; set; } = 72m;

    /// <summary>
    /// Minimum hold time in hours - don't exit before this (except stop loss)
    /// Example: 0.5 = don't take profit exits before 30 minutes
    /// </summary>
    public decimal MinHoldHours { get; set; } = 0.5m;

    /// <summary>
    /// Sample interval in hours for checking exit conditions
    /// Example: 0.5 = check every 30 minutes
    /// Lower values = more accurate exits but slower simulation
    /// </summary>
    public decimal SampleIntervalHours { get; set; } = 0.5m;

    /// <summary>
    /// Priority order for exit conditions when multiple trigger
    /// 1 = highest priority (stop loss should be 1)
    /// </summary>
    public int StopLossPriority { get; set; } = 1;
    public int VolatilityExitPriority { get; set; } = 2;
    public int ProfitTargetPriority { get; set; } = 3;
    public int TrailingStopPriority { get; set; } = 4;
    public int FundingReversalPriority { get; set; } = 5;
    public int MaxHoldPriority { get; set; } = 6;

    /// <summary>
    /// Predefined strategy configurations
    /// </summary>
    public static class Presets
    {
        /// <summary>
        /// Conservative strategy: Quick profit taking, tight stop loss
        /// Target: Small consistent wins, minimize losses
        /// </summary>
        public static ExitStrategyConfig Conservative => new()
        {
            Name = "Conservative",
            ProfitTargetPercent = 1.5m,      // Exit at +1.5%
            StopLossPercent = -2.0m,          // Cut losses at -2%
            TrailingStopPercent = 0.5m,       // Protect profits (drop 0.5%)
            TrailingStopActivationPercent = 1.0m,  // After +1%
            FundingReversalThreshold = 0.6m,  // Exit if funding drops 40%
            MaxHoldHours = 24m,               // Max 24 hours
            MinHoldHours = 1.0m,              // Min 1 hour
            SampleIntervalHours = 0.5m        // Check every 30min
        };

        /// <summary>
        /// Aggressive strategy: Higher targets, wider stops
        /// Target: Maximize wins, accept larger drawdowns
        /// </summary>
        public static ExitStrategyConfig Aggressive => new()
        {
            Name = "Aggressive",
            ProfitTargetPercent = 3.0m,       // Exit at +3%
            StopLossPercent = -5.0m,          // Wider stop at -5%
            TrailingStopPercent = 1.0m,       // Wider trailing (drop 1%)
            TrailingStopActivationPercent = 2.0m,  // After +2%
            VolatilitySpikeMultiplier = 3.0m, // Only exit if vol triples
            MaxHoldHours = 72m,               // Max 72 hours
            MinHoldHours = 0.5m,              // Min 30min
            SampleIntervalHours = 1.0m        // Check hourly
        };

        /// <summary>
        /// Funding-based strategy: Hold as long as funding is favorable
        /// Target: Maximize funding revenue, exit on reversals
        /// </summary>
        public static ExitStrategyConfig FundingBased => new()
        {
            Name = "FundingBased",
            ProfitTargetPercent = 5.0m,       // High target (rely on funding)
            StopLossPercent = -4.0m,          // Moderate stop
            FundingReversalThreshold = 0.5m,  // Exit if funding drops 50%
            VolatilitySpikeMultiplier = 2.5m, // Exit on significant vol increase
            MaxHoldHours = 48m,               // Max 48 hours
            MinHoldHours = 2.0m,              // Min 2 hours (wait for funding)
            SampleIntervalHours = 0.5m        // Check every 30min
        };

        /// <summary>
        /// Scalping strategy: Very quick exits, tight management
        /// Target: Many small wins, very low risk
        /// </summary>
        public static ExitStrategyConfig Scalping => new()
        {
            Name = "Scalping",
            ProfitTargetPercent = 0.8m,       // Exit at +0.8%
            StopLossPercent = -1.0m,          // Very tight stop at -1%
            TrailingStopPercent = 0.3m,       // Very tight trailing
            TrailingStopActivationPercent = 0.5m,  // After +0.5%
            MaxHoldHours = 8m,                // Max 8 hours
            MinHoldHours = 0.5m,              // Min 30min
            SampleIntervalHours = 0.25m       // Check every 15min
        };

        /// <summary>
        /// Get all preset strategies for simulation
        /// </summary>
        public static List<ExitStrategyConfig> AllPresets => new()
        {
            Conservative,
            Aggressive,
            FundingBased,
            Scalping
        };

        /// <summary>
        /// Get recommended strategies for production ML training
        /// (Excludes scalping which may have too much noise)
        /// </summary>
        public static List<ExitStrategyConfig> Recommended => new()
        {
            Conservative,
            Aggressive,
            FundingBased
        };
    }
}

/// <summary>
/// Exit result from strategy simulation
/// </summary>
public class ExitResult
{
    /// <summary>
    /// Snapshot index where exit occurred
    /// </summary>
    public int ExitSnapshotIndex { get; set; }

    /// <summary>
    /// Exit timestamp
    /// </summary>
    public DateTime ExitTime { get; set; }

    /// <summary>
    /// Reason for exit
    /// </summary>
    public ExitReason Reason { get; set; }

    /// <summary>
    /// Profit/loss at exit
    /// </summary>
    public decimal ProfitPercent { get; set; }

    /// <summary>
    /// Peak profit achieved before exit
    /// </summary>
    public decimal PeakProfitPercent { get; set; }

    /// <summary>
    /// Maximum drawdown experienced
    /// </summary>
    public decimal MaxDrawdownPercent { get; set; }

    /// <summary>
    /// Hours held
    /// </summary>
    public decimal HoursHeld { get; set; }

    /// <summary>
    /// Did this exit hit the profit target?
    /// </summary>
    public bool HitProfitTarget { get; set; }

    /// <summary>
    /// Did this exit hit the stop loss?
    /// </summary>
    public bool HitStopLoss { get; set; }
}

/// <summary>
/// Reasons a position was exited
/// </summary>
public enum ExitReason
{
    /// <summary>
    /// Profit target was reached
    /// </summary>
    PROFIT_TARGET,

    /// <summary>
    /// Stop loss was hit
    /// </summary>
    STOP_LOSS,

    /// <summary>
    /// Trailing stop was triggered
    /// </summary>
    TRAILING_STOP,

    /// <summary>
    /// Funding rate differential reversed
    /// </summary>
    FUNDING_REVERSAL,

    /// <summary>
    /// Spread volatility spiked
    /// </summary>
    VOLATILITY_SPIKE,

    /// <summary>
    /// Maximum hold time reached
    /// </summary>
    MAX_HOLD_TIME,

    /// <summary>
    /// Insufficient data to continue simulation
    /// </summary>
    INSUFFICIENT_DATA,

    /// <summary>
    /// Position exited at the optimal profit point identified by hindsight analysis
    /// </summary>
    OPTIMAL_HINDSIGHT
}
