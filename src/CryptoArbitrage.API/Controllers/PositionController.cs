using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Models.DataCollection;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Services;
using CryptoArbitrage.API.Services.Authentication;
using CryptoArbitrage.API.Services.Suggestions;
using CryptoArbitrage.API.Services.DataCollection.Abstractions;
using CryptoArbitrage.API.Constants;

namespace CryptoArbitrage.API.Controllers;

/// <summary>
/// Manages positions for the authenticated user.
/// All endpoints require [Authorize] attribute for multi-user support.
/// User identity is extracted from JWT token via ICurrentUserService.
/// Positions are automatically filtered by UserId - users can only see their own positions.
/// </summary>
[Authorize]
[ApiController]
[Route("api/[controller]")]
public class PositionController : BaseController
{
    private readonly ArbitrageDbContext _context;
    private readonly ICurrentUserService _currentUser;
    private readonly ExitStrategyMonitor _exitMonitor;
    private readonly IDataRepository<MarketDataSnapshot> _marketDataRepository;

    public PositionController(
        ArbitrageDbContext context,
        ILogger<PositionController> logger,
        ICurrentUserService currentUser,
        ExitStrategyMonitor exitMonitor,
        IDataRepository<MarketDataSnapshot> marketDataRepository)
        : base(logger)
    {
        _context = context;
        _currentUser = currentUser;
        _exitMonitor = exitMonitor;
        _marketDataRepository = marketDataRepository;
    }

    /// <summary>
    /// Gets all positions for the authenticated user.
    /// Automatically filters by UserId from JWT token.
    /// </summary>
    [HttpGet]
    public async Task<ActionResult<List<PositionDto>>> GetPositions([FromQuery] string? status = null)
    {
        return await ExecuteAuthenticatedAsync(_currentUser.UserId, async () =>
        {
            var query = _context.Positions
                .Include(p => p.Execution) // Include execution for exit signal calculation
                .Where(p => p.UserId == _currentUser.UserId); // CRITICAL: Filter by authenticated user

            // Filter by status if provided
            if (!string.IsNullOrEmpty(status) && Enum.TryParse<PositionStatus>(status, true, out var positionStatus))
            {
                query = query.Where(p => p.Status == positionStatus);
            }

            var positionEntities = await query
                .OrderByDescending(p => p.OpenedAt)
                .ToListAsync();

            // Get market data snapshot for exit signal calculation
            MarketDataSnapshot? marketSnapshot = null;
            try
            {
                marketSnapshot = await _marketDataRepository.GetAsync(DataCollectionConstants.CacheKeys.MarketDataSnapshot);
            }
            catch (Exception ex)
            {
                Logger.LogWarning(ex, "Failed to get market data snapshot for exit signals");
            }

            var positions = positionEntities.Select(p =>
            {
                var dto = new PositionDto
                {
                    Id = p.Id,
                    ExecutionId = p.ExecutionId,
                    Exchange = p.Exchange,
                    Symbol = p.Symbol,
                    Type = p.Type,
                    Side = p.Side,
                    Status = p.Status,
                    EntryPrice = p.EntryPrice,
                    ExitPrice = p.ExitPrice,
                    Quantity = p.Quantity,
                    Leverage = p.Leverage,
                    InitialMargin = p.InitialMargin,
                    RealizedPnL = p.RealizedPnL,
                    UnrealizedPnL = p.UnrealizedPnL,
                    TotalFundingFeePaid = p.TotalFundingFeePaid,
                    TotalFundingFeeReceived = p.TotalFundingFeeReceived,
                    OpenedAt = p.OpenedAt,
                    ClosedAt = p.ClosedAt,
                    ActiveOpportunityId = p.ExecutionId
                };

                // Calculate exit signals for open positions
                if (p.Status == PositionStatus.Open && marketSnapshot != null)
                {
                    try
                    {
                        dto.ExitSignals = _exitMonitor.EvaluateExitConditions(p, p.Execution, marketSnapshot);
                    }
                    catch (Exception ex)
                    {
                        Logger.LogWarning(ex, "Failed to evaluate exit signals for position {PositionId}", p.Id);
                        dto.ExitSignals = new List<CryptoArbitrage.API.Models.Suggestions.ExitSignal>();
                    }
                }

                return dto;
            }).ToList();

            Logger.LogDebug("User {UserId} retrieved {Count} positions", _currentUser.UserId, positions.Count);
            return positions;
        }, $"fetching positions for user {_currentUser.UserId}");
    }

    /// <summary>
    /// Gets a specific position by ID.
    /// CRITICAL: Validates the authenticated user owns the position before returning it.
    /// </summary>
    [HttpGet("{id}")]
    public async Task<ActionResult<PositionDto>> GetPosition(int id)
    {
        return await ExecuteActionAsync(async () =>
        {
            var authResult = ValidateAuthentication(_currentUser.UserId);
            if (authResult != null)
                return authResult;

            var position = await _context.Positions
                .Include(p => p.Execution) // Include execution for exit signal calculation
                .Where(p => p.Id == id && p.UserId == _currentUser.UserId) // CRITICAL: Filter by user
                .FirstOrDefaultAsync();

            if (position == null)
            {
                Logger.LogWarning("User {UserId} attempted to access position {Id} which does not exist or is not owned by them",
                    _currentUser.UserId, id);
                return NotFound(new { errorMessage = $"Position {id} not found" });
            }

            var positionDto = new PositionDto
            {
                Id = position.Id,
                ExecutionId = position.ExecutionId,
                Exchange = position.Exchange,
                Symbol = position.Symbol,
                Type = position.Type,
                Side = position.Side,
                Status = position.Status,
                EntryPrice = position.EntryPrice,
                ExitPrice = position.ExitPrice,
                Quantity = position.Quantity,
                Leverage = position.Leverage,
                InitialMargin = position.InitialMargin,
                RealizedPnL = position.RealizedPnL,
                UnrealizedPnL = position.UnrealizedPnL,
                TotalFundingFeePaid = position.TotalFundingFeePaid,
                TotalFundingFeeReceived = position.TotalFundingFeeReceived,
                OpenedAt = position.OpenedAt,
                ClosedAt = position.ClosedAt,
                ActiveOpportunityId = position.ExecutionId
            };

            // Calculate exit signals for open positions
            if (position.Status == PositionStatus.Open)
            {
                try
                {
                    var marketSnapshot = await _marketDataRepository.GetAsync(DataCollectionConstants.CacheKeys.MarketDataSnapshot);
                    if (marketSnapshot != null)
                    {
                        positionDto.ExitSignals = _exitMonitor.EvaluateExitConditions(position, position.Execution, marketSnapshot);
                    }
                }
                catch (Exception ex)
                {
                    Logger.LogWarning(ex, "Failed to evaluate exit signals for position {PositionId}", position.Id);
                    positionDto.ExitSignals = new List<CryptoArbitrage.API.Models.Suggestions.ExitSignal>();
                }
            }

            Logger.LogDebug("User {UserId} retrieved position {Id}", _currentUser.UserId, id);
            return Ok(positionDto);
        }, $"fetching position {id} for user {_currentUser.UserId}");
    }
}
