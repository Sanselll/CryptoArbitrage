namespace CryptoArbitrage.API.Models;

public class TradeDto
{
    public string Exchange { get; set; } = string.Empty;
    public string TradeId { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public OrderSide Side { get; set; }
    public decimal Price { get; set; }
    public decimal Quantity { get; set; }
    public decimal QuoteQuantity { get; set; }
    public decimal Fee { get; set; }
    public string? FeeAsset { get; set; }
    public decimal? Commission { get; set; }
    public string? CommissionAsset { get; set; }
    public bool IsMaker { get; set; }
    public bool IsBuyer { get; set; }
    public DateTime ExecutedAt { get; set; }
    public string? OrderType { get; set; }
    public string? PositionSide { get; set; }
}
