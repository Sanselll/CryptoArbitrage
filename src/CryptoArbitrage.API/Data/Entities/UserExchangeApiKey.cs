using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Data.Entities;

public class UserExchangeApiKey
{
    public int Id { get; set; }

    [Required]
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;

    [Required]
    [MaxLength(50)]
    public string ExchangeName { get; set; } = string.Empty; // "Binance", "Bybit"

    [Required]
    public string EncryptedApiKey { get; set; } = string.Empty;

    [Required]
    public string EncryptedApiSecret { get; set; } = string.Empty;

    public bool IsEnabled { get; set; } = true;

    public DateTime CreatedAt { get; set; }
    public DateTime? LastTestedAt { get; set; }
    public string? LastTestResult { get; set; }
}
