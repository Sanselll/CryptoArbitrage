using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Services;

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
public class PositionController : ControllerBase
{
    private readonly ArbitrageDbContext _context;
    private readonly ILogger<PositionController> _logger;
    private readonly ICurrentUserService _currentUser;

    public PositionController(
        ArbitrageDbContext context,
        ILogger<PositionController> logger,
        ICurrentUserService currentUser)
    {
        _context = context;
        _logger = logger;
        _currentUser = currentUser;
    }

    /// <summary>
    /// Gets all positions for the authenticated user.
    /// Automatically filters by UserId from JWT token.
    /// </summary>
    [HttpGet]
    public async Task<ActionResult<List<PositionDto>>> GetPositions([FromQuery] string? status = null)
    {
        try
        {
            if (string.IsNullOrEmpty(_currentUser.UserId))
                return Unauthorized(new { error = "User not authenticated" });

            var query = _context.Positions
                .Where(p => p.UserId == _currentUser.UserId); // CRITICAL: Filter by authenticated user

            // Filter by status if provided
            if (!string.IsNullOrEmpty(status) && Enum.TryParse<PositionStatus>(status, true, out var positionStatus))
            {
                query = query.Where(p => p.Status == positionStatus);
            }

            var positions = await query
                .OrderByDescending(p => p.OpenedAt)
                .Select(p => new PositionDto
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
                })
                .ToListAsync();

            _logger.LogDebug("User {UserId} retrieved {Count} positions", _currentUser.UserId, positions.Count);
            return Ok(positions);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching positions for user {UserId}", _currentUser.UserId);
            return StatusCode(500, new { errorMessage = $"Server error: {ex.Message}" });
        }
    }

    /// <summary>
    /// Gets a specific position by ID.
    /// CRITICAL: Validates the authenticated user owns the position before returning it.
    /// </summary>
    [HttpGet("{id}")]
    public async Task<ActionResult<PositionDto>> GetPosition(int id)
    {
        try
        {
            if (string.IsNullOrEmpty(_currentUser.UserId))
                return Unauthorized(new { error = "User not authenticated" });

            var position = await _context.Positions
                .Where(p => p.Id == id && p.UserId == _currentUser.UserId) // CRITICAL: Filter by user
                .FirstOrDefaultAsync();

            if (position == null)
            {
                _logger.LogWarning("User {UserId} attempted to access position {Id} which does not exist or is not owned by them",
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

            _logger.LogDebug("User {UserId} retrieved position {Id}", _currentUser.UserId, id);
            return Ok(positionDto);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching position {Id} for user {UserId}", id, _currentUser.UserId);
            return StatusCode(500, new { errorMessage = $"Server error: {ex.Message}" });
        }
    }
}
