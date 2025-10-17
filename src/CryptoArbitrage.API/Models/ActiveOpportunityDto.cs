namespace CryptoArbitrage.API.Models;

public class ActiveOpportunityDto
{
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Exchange { get; set; } = string.Empty;
    public string? LongExchange { get; set; }
    public string? ShortExchange { get; set; }
    public int Strategy { get; set; } // 1 = SpotPerpetual, 2 = CrossExchange
    public decimal FundingRate { get; set; }
    public decimal? LongFundingRate { get; set; }
    public decimal? ShortFundingRate { get; set; }
    public decimal SpreadRate { get; set; }
    public decimal AnnualizedSpread { get; set; }
    public decimal EstimatedProfitPercentage { get; set; }
    public DateTime ExecutedAt { get; set; }
    public DateTime? ClosedAt { get; set; }
    public decimal PositionSizeUsd { get; set; }
    public decimal Leverage { get; set; }
    public decimal? StopLossPercentage { get; set; }
    public decimal? TakeProfitPercentage { get; set; }
    public decimal CurrentPnL { get; set; }
    public decimal NetFundingFees { get; set; }
    public bool IsActive { get; set; }
    public string? Notes { get; set; }
}
