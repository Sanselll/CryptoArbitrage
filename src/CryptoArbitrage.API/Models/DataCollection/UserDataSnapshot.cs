using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Models.DataCollection;

/// <summary>
/// Snapshot of user data (balances and positions) for a specific user and exchange
/// </summary>
public class UserDataSnapshot
{
    public string UserId { get; set; } = string.Empty;
    public string Exchange { get; set; } = string.Empty;
    public AccountBalanceDto? Balance { get; set; }
    public List<PositionDto> Positions { get; set; } = new();
    public DateTime CollectedAt { get; set; } = DateTime.UtcNow;
}
