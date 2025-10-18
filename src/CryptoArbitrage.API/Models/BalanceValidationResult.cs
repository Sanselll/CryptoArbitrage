namespace CryptoArbitrage.API.Models;

/// <summary>
/// Result of balance validation for execution
/// </summary>
public class BalanceValidationResult
{
    public bool IsValid { get; set; }
    public List<string> Errors { get; set; } = new();
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Required spot USDT for execution
    /// </summary>
    public decimal RequiredSpotUsdt { get; set; }

    /// <summary>
    /// Required futures margin for execution
    /// </summary>
    public decimal RequiredFuturesMargin { get; set; }

    /// <summary>
    /// Total required capital
    /// </summary>
    public decimal TotalRequired { get; set; }

    /// <summary>
    /// Available spot USDT
    /// </summary>
    public decimal AvailableSpotUsdt { get; set; }

    /// <summary>
    /// Available futures margin
    /// </summary>
    public decimal AvailableFuturesMargin { get; set; }

    /// <summary>
    /// Exchange name for single-exchange validation
    /// </summary>
    public string? Exchange { get; set; }

    /// <summary>
    /// For multi-exchange: long exchange validation
    /// </summary>
    public string? LongExchange { get; set; }

    /// <summary>
    /// For multi-exchange: short exchange validation
    /// </summary>
    public string? ShortExchange { get; set; }

    /// <summary>
    /// Whether this is for a unified margin account
    /// </summary>
    public bool IsUnifiedAccount { get; set; }
}
