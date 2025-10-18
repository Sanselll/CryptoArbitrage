namespace CryptoArbitrage.API.Models;

/// <summary>
/// Balance information needed for execution validation
/// </summary>
public class ExecutionBalancesDto
{
    public string Exchange { get; set; } = string.Empty;

    /// <summary>
    /// Available spot USDT balance (for spot purchases)
    /// </summary>
    public decimal SpotUsdtAvailable { get; set; }

    /// <summary>
    /// Available futures/margin balance (for perpetual positions)
    /// </summary>
    public decimal FuturesAvailable { get; set; }

    /// <summary>
    /// Total available balance in unified account (Bybit) or combined (Binance)
    /// </summary>
    public decimal TotalAvailable { get; set; }

    /// <summary>
    /// Whether this exchange uses unified margin (true for Bybit)
    /// </summary>
    public bool IsUnifiedAccount { get; set; }

    /// <summary>
    /// Current margin usage percentage
    /// </summary>
    public decimal MarginUsagePercent { get; set; }

    /// <summary>
    /// Maximum position size that can be opened with current balance
    /// </summary>
    public decimal MaxPositionSize { get; set; }
}

/// <summary>
/// Balance validation result for multi-exchange execution
/// </summary>
public class MultiExchangeBalancesDto
{
    public ExecutionBalancesDto? LongExchangeBalances { get; set; }
    public ExecutionBalancesDto? ShortExchangeBalances { get; set; }
    public bool CanExecute { get; set; }
    public string? ValidationMessage { get; set; }
}
