namespace CryptoArbitrage.API.Data.Entities;

public class FundingRate
{
    public int Id { get; set; }
    public int ExchangeId { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public decimal Rate { get; set; }
    public decimal AnnualizedRate { get; set; }
    public DateTime FundingTime { get; set; }
    public DateTime NextFundingTime { get; set; }
    public DateTime RecordedAt { get; set; } = DateTime.UtcNow;

    // Navigation properties
    public Exchange Exchange { get; set; } = null!;
}
