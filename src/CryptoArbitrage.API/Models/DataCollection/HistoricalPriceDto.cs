namespace CryptoArbitrage.API.Models.DataCollection;

/// <summary>
/// Represents historical price data for a specific exchange/symbol at a point in time
/// Used to calculate average price spreads over time
/// </summary>
public class HistoricalPriceDto
{
    public string Exchange { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public decimal Price { get; set; }
    public DateTime Timestamp { get; set; }
}
