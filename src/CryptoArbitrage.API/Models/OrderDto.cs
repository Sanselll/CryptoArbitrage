namespace CryptoArbitrage.API.Models;

public class OrderDto
{
    public string Exchange { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public string? ClientOrderId { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public OrderSide Side { get; set; }
    public OrderType Type { get; set; }
    public OrderStatus Status { get; set; }
    public string? TimeInForce { get; set; }
    public decimal? Price { get; set; }
    public decimal? AveragePrice { get; set; }
    public decimal? StopPrice { get; set; }
    public decimal Quantity { get; set; }
    public decimal FilledQuantity { get; set; }
    public decimal Fee { get; set; }
    public string? FeeAsset { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
    public DateTime? WorkingTime { get; set; }
    public string? ReduceOnly { get; set; }
    public string? PostOnly { get; set; }
}
