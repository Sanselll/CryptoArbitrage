namespace CryptoArbitrage.API.Models;

/// <summary>
/// Fee information for a user on a specific exchange
/// </summary>
public class FeeInfoDto
{
    public string UserId { get; set; } = string.Empty;
    public string Exchange { get; set; } = string.Empty;

    /// <summary>
    /// Maker fee rate (e.g., 0.0002 for 0.02%)
    /// </summary>
    public decimal MakerFeeRate { get; set; }

    /// <summary>
    /// Taker fee rate (e.g., 0.0004 for 0.04%)
    /// </summary>
    public decimal TakerFeeRate { get; set; }

    /// <summary>
    /// User's fee tier or VIP level (if available from exchange)
    /// </summary>
    public string? FeeTier { get; set; }

    /// <summary>
    /// When this fee information was collected
    /// </summary>
    public DateTime CollectedAt { get; set; }
}
