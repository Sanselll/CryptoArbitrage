using System.ComponentModel.DataAnnotations;
using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Data.Entities;

/// <summary>
/// Links transactions (trades, commissions, funding fees) to positions for accurate P&L calculation
/// </summary>
public class PositionTransaction
{
    public int Id { get; set; }

    // Link to Position
    [Required]
    public int PositionId { get; set; }
    public Position Position { get; set; } = null!;

    // Transaction identification
    [Required]
    public string TransactionId { get; set; } = string.Empty;  // Exchange transaction ID
    [Required]
    public string Exchange { get; set; } = string.Empty;
    [Required]
    public string Symbol { get; set; } = string.Empty;

    // Transaction details
    public TransactionType TransactionType { get; set; }
    public decimal Amount { get; set; }
    public decimal Fee { get; set; }
    public decimal? SignedFee { get; set; }

    // Linking metadata
    public string? OrderId { get; set; }  // For trade/commission matching
    public decimal? AllocationPercentage { get; set; }  // For split funding fees (if multiple positions)

    // Timestamps
    public DateTime TransactionCreatedAt { get; set; }  // When transaction occurred on exchange
    public DateTime LinkedAt { get; set; } = DateTime.UtcNow;  // When we linked it to this position

    // Optional details
    public string? Asset { get; set; }
    public string? Notes { get; set; }
}
