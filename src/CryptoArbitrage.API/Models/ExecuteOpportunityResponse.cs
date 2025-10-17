namespace CryptoArbitrage.API.Models;

public class ExecuteOpportunityResponse
{
    public bool Success { get; set; }
    public string? Message { get; set; }
    public List<int> PositionIds { get; set; } = new();
    public List<string> OrderIds { get; set; } = new();
    public decimal TotalPositionSize { get; set; }
    public string? ErrorMessage { get; set; }
}
