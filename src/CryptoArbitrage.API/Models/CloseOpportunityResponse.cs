namespace CryptoArbitrage.API.Models;

public class CloseOpportunityResponse
{
    public bool Success { get; set; }
    public string? Message { get; set; }
    public int ActiveOpportunityId { get; set; }
    public List<int> ClosedPositionIds { get; set; } = new();
    public decimal FinalPnL { get; set; }
    public decimal NetFundingFees { get; set; }
    public string? ErrorMessage { get; set; }
}
