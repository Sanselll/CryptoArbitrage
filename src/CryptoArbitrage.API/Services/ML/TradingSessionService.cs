using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Services.ML;

/// <summary>
/// Service for tracking trading session state for ML model predictions.
/// Maintains session-level metrics like total P&L and episode progress.
/// </summary>
public interface ITradingSessionService
{
    /// <summary>
    /// Gets or initializes the current session start time.
    /// </summary>
    DateTime GetSessionStartTime();

    /// <summary>
    /// Gets the initial capital for the current session.
    /// </summary>
    decimal GetSessionInitialCapital();

    /// <summary>
    /// Calculates session-level total P&L percentage.
    /// </summary>
    Task<float> CalculateSessionTotalPnlPctAsync();

    /// <summary>
    /// Calculates episode progress (0.0 to 1.0) based on session duration.
    /// </summary>
    float CalculateEpisodeProgress();

    /// <summary>
    /// Resets the session (call this at session boundaries like midnight).
    /// </summary>
    void ResetSession();
}

public class TradingSessionService : ITradingSessionService
{
    private readonly ArbitrageDbContext _db;
    private readonly ILogger<TradingSessionService> _logger;

    // Session state (in-memory for now, could be persisted)
    private static DateTime _sessionStartTime = DateTime.UtcNow;
    private static decimal _sessionInitialCapital = 10000m; // TODO: Get from user settings
    private static readonly TimeSpan SessionDuration = TimeSpan.FromHours(24); // 24-hour sessions

    public TradingSessionService(ArbitrageDbContext db, ILogger<TradingSessionService> logger)
    {
        _db = db;
        _logger = logger;

        // Auto-reset session if expired
        if (DateTime.UtcNow - _sessionStartTime > SessionDuration)
        {
            _logger.LogInformation("Session expired, resetting...");
            ResetSession();
        }
    }

    public DateTime GetSessionStartTime() => _sessionStartTime;

    public decimal GetSessionInitialCapital() => _sessionInitialCapital;

    public async Task<float> CalculateSessionTotalPnlPctAsync()
    {
        try
        {
            // Get all positions that were opened during or before this session
            var positions = await _db.Positions
                .Where(p => p.OpenedAt >= _sessionStartTime)
                .ToListAsync();

            // Calculate realized P&L from closed positions in this session
            var realizedPnl = positions
                .Where(p => p.Status == PositionStatus.Closed && p.ClosedAt.HasValue)
                .Sum(p => p.RealizedPnLUsd);

            // Calculate unrealized P&L from currently open positions
            var unrealizedPnl = positions
                .Where(p => p.Status != PositionStatus.Closed)
                .Sum(p => p.UnrealizedPnL);

            // Total P&L = realized + unrealized
            var totalPnl = realizedPnl + unrealizedPnl;

            // Calculate percentage relative to session initial capital
            var totalPnlPct = (float)(totalPnl / _sessionInitialCapital * 100m);

            _logger.LogDebug(
                "Session P&L: Realized={RealizedPnL}, Unrealized={UnrealizedPnl}, Total={TotalPnl}, Pct={TotalPnlPct}%",
                realizedPnl, unrealizedPnl, totalPnl, totalPnlPct);

            return totalPnlPct;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating session total P&L");
            return 0.0f; // Fallback to 0 on error
        }
    }

    public float CalculateEpisodeProgress()
    {
        var elapsed = DateTime.UtcNow - _sessionStartTime;
        var progress = (float)(elapsed.TotalMilliseconds / SessionDuration.TotalMilliseconds);

        // Clamp to [0.0, 1.0]
        return Math.Clamp(progress, 0.0f, 1.0f);
    }

    public void ResetSession()
    {
        _sessionStartTime = DateTime.UtcNow;

        // Get current equity to use as new initial capital
        // This allows the session to "carry forward" accumulated profits/losses
        var currentEquity = CalculateCurrentEquity();
        _sessionInitialCapital = currentEquity > 0 ? currentEquity : 10000m;

        _logger.LogInformation(
            "Session reset at {SessionStartTime}, Initial Capital = ${InitialCapital}",
            _sessionStartTime, _sessionInitialCapital);
    }

    private decimal CalculateCurrentEquity()
    {
        try
        {
            // Get unrealized P&L from all open positions
            var unrealizedPnl = _db.Positions
                .Where(p => p.Status != PositionStatus.Closed)
                .Sum(p => p.UnrealizedPnL);

            // Current equity = initial capital + all P&L
            return _sessionInitialCapital + unrealizedPnl;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating current equity");
            return _sessionInitialCapital; // Fallback to initial capital
        }
    }
}
