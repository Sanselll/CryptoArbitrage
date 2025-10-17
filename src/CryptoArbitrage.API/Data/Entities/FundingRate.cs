namespace CryptoArbitrage.API.Data.Entities;

public class FundingRate
{
    public int Id { get; set; }
    public string Exchange { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public decimal Rate { get; set; }
    public decimal AnnualizedRate { get; set; }
    public DateTime FundingTime { get; set; }
    public DateTime NextFundingTime { get; set; }
    public DateTime RecordedAt { get; set; } = DateTime.UtcNow;
}
