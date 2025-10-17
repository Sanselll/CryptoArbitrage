namespace CryptoArbitrage.API.Models;

public class AccountBalanceDto
{
    public string Exchange { get; set; } = string.Empty;

    // Combined totals (Spot + Futures)
    public decimal TotalBalance { get; set; }
    public decimal AvailableBalance { get; set; }

    // Operational balance (USDT + coins in positions + futures balance)
    public decimal OperationalBalanceUsd { get; set; }

    // Spot balances
    public decimal SpotBalanceUsd { get; set; }
    public decimal SpotAvailableUsd { get; set; }
    public Dictionary<string, decimal> SpotAssets { get; set; } = new Dictionary<string, decimal>();

    // Futures balances
    public decimal FuturesBalanceUsd { get; set; }
    public decimal FuturesAvailableUsd { get; set; }
    public decimal MarginUsed { get; set; }
    public decimal UnrealizedPnL { get; set; }

    public DateTime UpdatedAt { get; set; }
}
