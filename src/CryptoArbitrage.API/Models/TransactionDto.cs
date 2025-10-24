namespace CryptoArbitrage.API.Models;

public class TransactionDto
{
    public string Exchange { get; set; } = string.Empty;
    public string TransactionId { get; set; } = string.Empty;
    public string? TxHash { get; set; }
    public TransactionType Type { get; set; }
    public string Asset { get; set; } = string.Empty;
    public decimal Amount { get; set; }
    public TransactionStatus Status { get; set; }
    public string? FromAddress { get; set; }
    public string? ToAddress { get; set; }
    public string? Network { get; set; }
    public string? Info { get; set; }
    public string? Symbol { get; set; }
    public string? TradeId { get; set; }
    public decimal Fee { get; set; }
    public string? FeeAsset { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? ConfirmedAt { get; set; }
}
