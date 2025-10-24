namespace CryptoArbitrage.API.Models;

/// <summary>
/// Universal price DTO for both spot and perpetual prices
/// </summary>
public class PriceDto
{
    public string Exchange { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public decimal Price { get; set; }
    public decimal Volume24h { get; set; }
    public DateTime Timestamp { get; set; }
}
