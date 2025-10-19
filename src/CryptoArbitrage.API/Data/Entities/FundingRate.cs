namespace CryptoArbitrage.API.Data.Entities;

public class FundingRate
{
    public int Id { get; set; }
    public string Exchange { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public decimal Rate { get; set; }
    public decimal AnnualizedRate { get; set; }
    public int FundingIntervalHours { get; set; } = 8; // Default 8h interval
    public decimal? Average3DayRate { get; set; } // 3-day average rate
    public FundingDirection? Direction { get; set; } // Who pays whom
    public decimal? PreviousRate { get; set; } // Previous settled funding rate
    public decimal? PreviousAnnualizedRate { get; set; } // Previous annualized rate
    public decimal? FundingCap { get; set; } // Maximum funding rate limit
    public decimal? FundingFloor { get; set; } // Minimum funding rate limit
    public DateTime FundingTime { get; set; }
    public DateTime NextFundingTime { get; set; }
    public DateTime RecordedAt { get; set; } = DateTime.UtcNow;
}
