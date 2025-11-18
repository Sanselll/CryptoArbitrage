using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Services;
using CryptoArbitrage.API.Services.Authentication;
using CryptoArbitrage.API.Services.ML;

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

    public PositionController(
        ArbitrageDbContext context,
        ILogger<PositionController> logger,
        ICurrentUserService currentUser)
        : base(logger)
    {
        _context = context;
        _currentUser = currentUser;
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
                .Where(p => p.UserId == _currentUser.UserId); // CRITICAL: Filter by authenticated user

            // Filter by status if provided
            if (!string.IsNullOrEmpty(status) && Enum.TryParse<PositionStatus>(status, true, out var positionStatus))
            {
                query = query.Where(p => p.Status == positionStatus);
            }

            // Load positions from database
            var dbPositions = await query
                .OrderByDescending(p => p.OpenedAt)
                .ToListAsync();

            // Get all position IDs to load transactions
            var positionIds = dbPositions.Select(p => p.Id).ToList();

            // Load all PositionTransactions for these positions (single query)
            var allTransactions = await _context.PositionTransactions
                .Where(pt => positionIds.Contains(pt.PositionId))
                .ToListAsync();

            // Map to DTOs and calculate fees in memory
            var positions = dbPositions.Select(p =>
            {
                var positionTransactions = allTransactions.Where(pt => pt.PositionId == p.Id).ToList();

                return new PositionDto
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
                    FundingEarnedUsd = p.FundingEarnedUsd,
                    TradingFeesUsd = p.TradingFeesUsd,
                    PricePnLUsd = p.PricePnLUsd,
                    RealizedPnLUsd = p.RealizedPnLUsd,
                    RealizedPnLPct = p.RealizedPnLPct,
                    UnrealizedPnL = p.UnrealizedPnL,
                    // Calculate fees from PositionTransaction (single source of truth)
                    TotalFundingFeePaid = positionTransactions
                        .Where(pt => pt.TransactionType == TransactionType.FundingFee && pt.SignedFee < 0)
                        .Sum(pt => Math.Abs(pt.SignedFee ?? 0m)),
                    TotalFundingFeeReceived = positionTransactions
                        .Where(pt => pt.TransactionType == TransactionType.FundingFee && pt.SignedFee > 0)
                        .Sum(pt => pt.SignedFee ?? 0m),
                    TradingFeePaid = positionTransactions
                        .Where(pt => pt.TransactionType == TransactionType.Commission || pt.TransactionType == TransactionType.Trade)
                        .Sum(pt => pt.Fee),
                    ReconciliationStatus = p.ReconciliationStatus,
                    ReconciliationCompletedAt = p.ReconciliationCompletedAt,
                    OpenedAt = p.OpenedAt,
                    ClosedAt = p.ClosedAt,
                    ActiveOpportunityId = p.ExecutionId
                };
            }).ToList();

            // Enrich open positions with RL predictions
            var openPositions = positions.Where(p => p.Status == PositionStatus.Open).ToList();
            Logger.LogInformation("Found {Count} open positions to enrich with RL predictions", openPositions.Count);

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
                .Where(p => p.Id == id && p.UserId == _currentUser.UserId) // CRITICAL: Filter by user
                .FirstOrDefaultAsync();

            if (position == null)
            {
                Logger.LogWarning("User {UserId} attempted to access position {Id} which does not exist or is not owned by them",
                    _currentUser.UserId, id);
                return NotFound(new { errorMessage = $"Position {id} not found" });
            }

            // Load PositionTransactions to calculate fees
            var positionTransactions = await _context.PositionTransactions
                .Where(pt => pt.PositionId == position.Id)
                .ToListAsync();

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
                FundingEarnedUsd = position.FundingEarnedUsd,
                TradingFeesUsd = position.TradingFeesUsd,
                PricePnLUsd = position.PricePnLUsd,
                RealizedPnLUsd = position.RealizedPnLUsd,
                RealizedPnLPct = position.RealizedPnLPct,
                UnrealizedPnL = position.UnrealizedPnL,
                // Calculate fees from PositionTransaction (single source of truth)
                TotalFundingFeePaid = positionTransactions
                    .Where(pt => pt.TransactionType == TransactionType.FundingFee && pt.SignedFee < 0)
                    .Sum(pt => Math.Abs(pt.SignedFee ?? 0m)),
                TotalFundingFeeReceived = positionTransactions
                    .Where(pt => pt.TransactionType == TransactionType.FundingFee && pt.SignedFee > 0)
                    .Sum(pt => pt.SignedFee ?? 0m),
                TradingFeePaid = positionTransactions
                    .Where(pt => pt.TransactionType == TransactionType.Commission || pt.TransactionType == TransactionType.Trade)
                    .Sum(pt => pt.Fee),
                ReconciliationStatus = position.ReconciliationStatus,
                ReconciliationCompletedAt = position.ReconciliationCompletedAt,
                OpenedAt = position.OpenedAt,
                ClosedAt = position.ClosedAt,
                ActiveOpportunityId = position.ExecutionId
            };

            Logger.LogDebug("User {UserId} retrieved position {Id}", _currentUser.UserId, id);
            return Ok(positionDto);
        }, $"fetching position {id} for user {_currentUser.UserId}");
    }

}
