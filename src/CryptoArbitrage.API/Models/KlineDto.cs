namespace CryptoArbitrage.API.Models;

/// <summary>
/// Represents a candlestick/kline data point (OHLCV)
/// </summary>
public class KlineDto
{
    public string Exchange { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public DateTime OpenTime { get; set; }
    public DateTime CloseTime { get; set; }
    public decimal Open { get; set; }
    public decimal High { get; set; }
    public decimal Low { get; set; }
    public decimal Close { get; set; }
    public decimal Volume { get; set; }
}

/// <summary>
/// Kline interval for historical data collection
/// </summary>
public enum KlineInterval
{
    OneMinute,
    FiveMinutes,
    FifteenMinutes,
    ThirtyMinutes,
    OneHour,
    FourHours,
    OneDay
}
