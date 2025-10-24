using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Data.Entities;

public class PerformanceMetric
{
    public int Id { get; set; }

    // Multi-user support
    [Required]
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;

    public DateTime Date { get; set; }
    public decimal TotalPnL { get; set; }
    public decimal RealizedPnL { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public decimal TotalFundingFeeReceived { get; set; }
    public decimal TotalFundingFeePaid { get; set; }
    public decimal NetFundingFee { get; set; }
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public decimal WinRate { get; set; }
    public decimal LargestWin { get; set; }
    public decimal LargestLoss { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal AccountBalance { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
