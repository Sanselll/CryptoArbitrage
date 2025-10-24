namespace CryptoArbitrage.API.Models;

public class NotificationDto
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public NotificationType Type { get; set; }
    public NotificationSeverity Severity { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public object? Data { get; set; }
    public bool AutoClose { get; set; } = true;
    public int? AutoCloseDelay { get; set; } = 5000; // milliseconds
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public enum NotificationType
{
    NegativeFunding,
    ExecutionStateChange,
    ExchangeConnectivity,
    OpportunityDetected,
    LiquidationRisk,
    General
}

public enum NotificationSeverity
{
    Info,
    Success,
    Warning,
    Error
}
