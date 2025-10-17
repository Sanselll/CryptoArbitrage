namespace CryptoArbitrage.API.Models;

public class AccountBalanceDto
{
    public string Exchange { get; set; } = string.Empty;
    public decimal TotalBalance { get; set; }
    public decimal AvailableBalance { get; set; }
    public decimal MarginUsed { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public DateTime UpdatedAt { get; set; }
}
