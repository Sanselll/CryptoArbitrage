using System.Collections.Concurrent;

namespace CryptoArbitrage.API.Services.Agent;

/// <summary>
/// In-memory repository for storing agent decisions
/// </summary>
public class AgentDecisionRepository
{
    private readonly ConcurrentDictionary<string, AgentDecisionRecord> _decisions = new();
    private readonly ILogger<AgentDecisionRepository> _logger;

    public AgentDecisionRepository(ILogger<AgentDecisionRepository> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Add or update a decision
    /// </summary>
    public void AddOrUpdate(AgentDecisionRecord decision)
    {
        _decisions[decision.Id] = decision;
        _logger.LogDebug("Stored decision {DecisionId} for session {SessionId}",
            decision.Id, decision.SessionId);
    }

    /// <summary>
    /// Get a specific decision by ID
    /// </summary>
    public AgentDecisionRecord? GetById(string id)
    {
        _decisions.TryGetValue(id, out var decision);
        return decision;
    }

    /// <summary>
    /// Get all decisions for a session
    /// </summary>
    public List<AgentDecisionRecord> GetBySessionId(Guid sessionId)
    {
        return _decisions.Values
            .Where(d => d.SessionId == sessionId)
            .OrderByDescending(d => d.Timestamp)
            .ToList();
    }

    /// <summary>
    /// Get decisions for a user (across all sessions)
    /// </summary>
    public List<AgentDecisionRecord> GetByUserId(string userId)
    {
        return _decisions.Values
            .Where(d => d.UserId == userId)
            .OrderByDescending(d => d.Timestamp)
            .Take(100) // Limit to last 100
            .ToList();
    }

    /// <summary>
    /// Update profit for a decision (called after reconciliation)
    /// </summary>
    public bool UpdateProfit(string decisionId, decimal profitUsd, decimal profitPct, bool isReconciled = true)
    {
        if (_decisions.TryGetValue(decisionId, out var decision))
        {
            decision.ProfitUsd = profitUsd;
            decision.ProfitPct = profitPct;
            decision.IsReconciled = isReconciled;
            decision.ReconciledAt = DateTime.UtcNow;

            _logger.LogInformation(
                "Updated decision {DecisionId} profit: ${ProfitUsd:F2} ({ProfitPct:F2}%), Reconciled: {IsReconciled}",
                decisionId, profitUsd, profitPct, isReconciled);

            return true;
        }

        _logger.LogWarning("Decision {DecisionId} not found for profit update", decisionId);
        return false;
    }

    /// <summary>
    /// Calculate session P&L from decisions
    /// </summary>
    public (decimal pnlUsd, decimal pnlPct, int winningTrades, int losingTrades) GetSessionMetrics(Guid sessionId)
    {
        var decisions = GetBySessionId(sessionId)
            .Where(d => d.Action == "EXIT" && d.ExecutionStatus == "success")
            .ToList();

        if (!decisions.Any())
            return (0, 0, 0, 0);

        var totalPnlUsd = decisions.Sum(d => d.ProfitUsd ?? 0);
        var winningTrades = decisions.Count(d => (d.ProfitUsd ?? 0) > 0);
        var losingTrades = decisions.Count(d => (d.ProfitUsd ?? 0) < 0);

        // Calculate average profit percentage across all exits
        var avgProfitPct = decisions.Count > 0
            ? decisions.Average(d => d.ProfitPct ?? 0)
            : 0;

        return (totalPnlUsd, avgProfitPct, winningTrades, losingTrades);
    }

    /// <summary>
    /// Clear old decisions (older than specified hours)
    /// </summary>
    public int ClearOldDecisions(int hoursToKeep = 24)
    {
        var cutoff = DateTime.UtcNow.AddHours(-hoursToKeep);
        var oldDecisions = _decisions.Values
            .Where(d => d.Timestamp < cutoff)
            .Select(d => d.Id)
            .ToList();

        int removed = 0;
        foreach (var id in oldDecisions)
        {
            if (_decisions.TryRemove(id, out _))
                removed++;
        }

        if (removed > 0)
        {
            _logger.LogInformation("Cleared {Count} decisions older than {Hours} hours", removed, hoursToKeep);
        }

        return removed;
    }

    /// <summary>
    /// Get decision by execution ID (for linking reconciled positions back to decisions)
    /// </summary>
    public AgentDecisionRecord? GetByExecutionId(string executionId)
    {
        return _decisions.Values
            .Where(d => d.ExecutionId == executionId)
            .OrderByDescending(d => d.Timestamp)
            .FirstOrDefault();
    }
}

/// <summary>
/// Agent decision record stored in memory
/// </summary>
public class AgentDecisionRecord
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public Guid SessionId { get; set; }
    public string UserId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string Action { get; set; } = string.Empty;
    public string? Symbol { get; set; }
    public string Confidence { get; set; } = string.Empty;
    public double? EnterProbability { get; set; }
    public string? Reasoning { get; set; }
    public int NumOpportunities { get; set; }
    public int NumPositions { get; set; }

    // Execution results
    public string ExecutionStatus { get; set; } = string.Empty;
    public string? ErrorMessage { get; set; }

    // ENTER specific
    public decimal? AmountUsd { get; set; }
    public string? ExecutionId { get; set; }

    // EXIT specific
    public decimal? ProfitUsd { get; set; }
    public decimal? ProfitPct { get; set; }
    public double? DurationHours { get; set; }

    // Reconciliation tracking
    public bool IsReconciled { get; set; }
    public DateTime? ReconciledAt { get; set; }
}
