using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Models;

public class FundingRateDto
{
    public string Exchange { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public decimal Rate { get; set; }
    public decimal AnnualizedRate { get; set; }
    public int FundingIntervalHours { get; set; } = 8;
    public decimal? Average3DayRate { get; set; }
    public decimal? Average24hRate { get; set; }
    public FundingDirection? Direction { get; set; }
    public decimal? PreviousRate { get; set; }
    public decimal? PreviousAnnualizedRate { get; set; }
    public decimal? FundingCap { get; set; }
    public decimal? FundingFloor { get; set; }
    public DateTime FundingTime { get; set; }
    public DateTime NextFundingTime { get; set; }
    public DateTime RecordedAt { get; set; }
}
