using Microsoft.AspNetCore.Identity;

namespace CryptoArbitrage.API.Data.Entities;

public class ApplicationUser : IdentityUser
{
    public string? GoogleId { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? LastLoginAt { get; set; }

    // Navigation properties
    public ICollection<UserExchangeApiKey> ExchangeApiKeys { get; set; } = new List<UserExchangeApiKey>();
    public ICollection<Position> Positions { get; set; } = new List<Position>();
    public ICollection<Execution> Executions { get; set; } = new List<Execution>();
    public ICollection<PerformanceMetric> PerformanceMetrics { get; set; } = new List<PerformanceMetric>();
}
