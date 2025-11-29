using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Services;

public interface IExecutionHistoryService
{
    Task<List<ExecutionHistoryDto>> GetExecutionHistoryAsync(string userId, CancellationToken cancellationToken = default);
}

public class ExecutionHistoryService : IExecutionHistoryService
{
    private readonly ArbitrageDbContext _context;
    private readonly ILogger<ExecutionHistoryService> _logger;

    public ExecutionHistoryService(ArbitrageDbContext context, ILogger<ExecutionHistoryService> logger)
    {
        _context = context;
        _logger = logger;
    }

    public async Task<List<ExecutionHistoryDto>> GetExecutionHistoryAsync(string userId, CancellationToken cancellationToken = default)
    {
        try
        {
            // Query closed/failed executions for this user
            var executions = await _context.Executions
                .Where(e => e.UserId == userId && e.State != ExecutionState.Running)
                .OrderByDescending(e => e.StoppedAt)
                .ToListAsync(cancellationToken);

            if (!executions.Any())
            {
                return new List<ExecutionHistoryDto>();
            }

            var executionIds = executions.Select(e => e.Id).ToList();

            // Get all positions for these executions
            var positions = await _context.Positions
                .Where(p => executionIds.Contains(p.ExecutionId))
                .ToListAsync(cancellationToken);

            // Group positions by execution
            var positionsByExecution = positions
                .GroupBy(p => p.ExecutionId)
                .ToDictionary(g => g.Key, g => g.ToList());

            var result = new List<ExecutionHistoryDto>();

            foreach (var execution in executions)
            {
                if (!positionsByExecution.TryGetValue(execution.Id, out var executionPositions) || !executionPositions.Any())
                {
                    continue;
                }

                // Aggregate P&L from all positions in this execution
                var totalPricePnL = executionPositions.Sum(p => p.PricePnLUsd);
                var totalFundingEarned = executionPositions.Sum(p => p.FundingEarnedUsd);
                var totalTradingFees = executionPositions.Sum(p => p.TradingFeesUsd);
                var totalPnL = executionPositions.Sum(p => p.RealizedPnLUsd);

                // Calculate duration
                var startedAt = execution.StartedAt;
                var closedAt = execution.StoppedAt ?? DateTime.UtcNow;
                var duration = closedAt - startedAt;

                // Calculate P&L percentage based on position size
                var totalPnLPct = execution.PositionSizeUsd > 0
                    ? (totalPnL / execution.PositionSizeUsd) * 100
                    : 0;

                // Map individual positions
                var positionDtos = executionPositions.Select(p => new ExecutionHistoryPositionDto
                {
                    Id = p.Id,
                    Exchange = p.Exchange,
                    Type = p.Type,
                    Side = p.Side,
                    EntryPrice = p.EntryPrice,
                    ExitPrice = p.ExitPrice ?? 0,
                    Quantity = p.Quantity,
                    Leverage = p.Leverage,
                    PricePnL = p.PricePnLUsd,
                    FundingEarned = p.FundingEarnedUsd,
                    TradingFees = p.TradingFeesUsd,
                    RealizedPnL = p.RealizedPnLUsd
                }).ToList();

                result.Add(new ExecutionHistoryDto
                {
                    Id = execution.Id,
                    Symbol = execution.Symbol,
                    Exchange = execution.Exchange,
                    LongExchange = execution.LongExchange,
                    ShortExchange = execution.ShortExchange,
                    Strategy = execution.SubType,
                    PositionSizeUsd = execution.PositionSizeUsd,
                    TotalPricePnL = totalPricePnL,
                    TotalFundingEarned = totalFundingEarned,
                    TotalTradingFees = totalTradingFees,
                    TotalPnL = totalPnL,
                    TotalPnLPct = totalPnLPct,
                    StartedAt = startedAt,
                    ClosedAt = closedAt,
                    DurationSeconds = duration.TotalSeconds,
                    State = execution.State,
                    Positions = positionDtos
                });
            }

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching execution history for user {UserId}", userId);
            return new List<ExecutionHistoryDto>();
        }
    }
}
