using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Data.Entities;

namespace CryptoArbitrage.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PositionController : ControllerBase
{
    private readonly ArbitrageDbContext _context;
    private readonly ILogger<PositionController> _logger;

    public PositionController(
        ArbitrageDbContext context,
        ILogger<PositionController> logger)
    {
        _context = context;
        _logger = logger;
    }

    [HttpGet]
    public async Task<ActionResult<List<PositionDto>>> GetPositions([FromQuery] string? status = null)
    {
        try
        {
            var query = _context.Positions.AsQueryable();

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

            return Ok(positions);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching positions");
            return StatusCode(500, new { errorMessage = $"Server error: {ex.Message}" });
        }
    }

    [HttpGet("{id}")]
    public async Task<ActionResult<PositionDto>> GetPosition(int id)
    {
        try
        {
            var position = await _context.Positions
                .Where(p => p.Id == id)
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
                .FirstOrDefaultAsync();

            if (position == null)
            {
                return NotFound(new { errorMessage = $"Position {id} not found" });
            }

            return Ok(position);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching position {Id}", id);
            return StatusCode(500, new { errorMessage = $"Server error: {ex.Message}" });
        }
    }
}
