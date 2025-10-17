namespace CryptoArbitrage.API.Models;

public class DashboardDataDto
{
    public List<FundingRateDto> FundingRates { get; set; } = new();
    public List<PositionDto> OpenPositions { get; set; } = new();
    public List<ArbitrageOpportunityDto> Opportunities { get; set; } = new();
    public List<AccountBalanceDto> Balances { get; set; } = new();
    public decimal TotalPnL { get; set; }
    public decimal TodayPnL { get; set; }
    public decimal TotalMarginUsed { get; set; }
    public int ActiveOpportunities { get; set; }
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}
