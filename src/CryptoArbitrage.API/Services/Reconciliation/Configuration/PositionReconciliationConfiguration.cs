namespace CryptoArbitrage.API.Services.Reconciliation.Configuration;

public class PositionReconciliationConfiguration
{
    /// <summary>
    /// Enable/disable reconciliation service
    /// </summary>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// How often to run reconciliation (seconds)
    /// </summary>
    public int ReconciliationIntervalSeconds { get; set; } = 30;

    /// <summary>
    /// Time window for matching funding fees (minutes before/after position time)
    /// </summary>
    public int FundingFeeTimeWindowMinutes { get; set; } = 5;

    /// <summary>
    /// Time window for matching commissions (minutes before position open)
    /// </summary>
    public int CommissionPreMatchWindowMinutes { get; set; } = 1;

    /// <summary>
    /// Time window for matching commissions (minutes after position close)
    /// </summary>
    public int CommissionPostMatchWindowMinutes { get; set; } = 5;

    /// <summary>
    /// Hours before a position is marked as stale if not fully reconciled
    /// </summary>
    public int StaleThresholdHours { get; set; } = 24;

    /// <summary>
    /// Minimum hours a perpetual position must be open to expect funding fees
    /// </summary>
    public decimal MinHoursForFundingFees { get; set; } = 8.0m;

    /// <summary>
    /// Funding interval in hours (8 hours on most exchanges)
    /// </summary>
    public decimal FundingIntervalHours { get; set; } = 8.0m;

    /// <summary>
    /// Maximum number of positions to reconcile per cycle
    /// </summary>
    public int MaxPositionsPerCycle { get; set; } = 100;

    /// <summary>
    /// Retry reconciliation for partially reconciled positions
    /// </summary>
    public bool RetryPartiallyReconciled { get; set; } = true;
}
