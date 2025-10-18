using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services;

namespace CryptoArbitrage.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class OpportunityController : ControllerBase
{
    private readonly ArbitrageDbContext _context;
    private readonly ILogger<OpportunityController> _logger;
    private readonly ArbitrageExecutionService _executionService;

    public OpportunityController(
        ArbitrageDbContext context,
        ILogger<OpportunityController> logger,
        ArbitrageExecutionService executionService)
    {
        _context = context;
        _logger = logger;
        _executionService = executionService;
    }

    // COMMENTED OUT: ArbitrageOpportunities table no longer exists - opportunities are now in-memory only
    // [HttpGet]
    // public async Task<ActionResult<List<ArbitrageOpportunityDto>>> GetOpportunities(
    //     [FromQuery] int limit = 50,
    //     [FromQuery] string? status = null)
    // {
    //     var query = _context.ArbitrageOpportunities
    //         .Include(o => o.LongExchange)
    //         .Include(o => o.ShortExchange)
    //         .AsQueryable();
    //
    //     if (!string.IsNullOrEmpty(status) && Enum.TryParse<Data.Entities.OpportunityStatus>(status, out var oppStatus))
    //     {
    //         query = query.Where(o => o.Status == oppStatus);
    //     }
    //
    //     var opportunities = await query
    //         .OrderByDescending(o => o.DetectedAt)
    //         .Take(limit)
    //         .Select(o => new ArbitrageOpportunityDto
    //         {
    //             Id = o.Id,
    //             Symbol = o.Symbol,
    //             LongExchange = o.LongExchange.Name,
    //             ShortExchange = o.ShortExchange.Name,
    //             LongFundingRate = o.LongFundingRate,
    //             ShortFundingRate = o.ShortFundingRate,
    //             SpreadRate = o.SpreadRate,
    //             AnnualizedSpread = o.AnnualizedSpread,
    //             EstimatedProfitPercentage = o.EstimatedProfitPercentage,
    //             Status = o.Status,
    //             DetectedAt = o.DetectedAt,
    //             ExecutedAt = o.ExecutedAt
    //         })
    //         .ToListAsync();
    //
    //     return opportunities;
    // }

    // COMMENTED OUT: ArbitrageOpportunities table no longer exists - opportunities are now in-memory only
    // [HttpGet("active")]
    // public async Task<ActionResult<List<ArbitrageOpportunityDto>>> GetActiveOpportunities()
    // {
    //     var opportunities = await _context.ArbitrageOpportunities
    //         .Include(o => o.LongExchange)
    //         .Include(o => o.ShortExchange)
    //         .Where(o => o.Status == Data.Entities.OpportunityStatus.Detected)
    //         .Where(o => o.DetectedAt > DateTime.UtcNow.AddMinutes(-10))
    //         .OrderByDescending(o => o.AnnualizedSpread)
    //         .Select(o => new ArbitrageOpportunityDto
    //         {
    //             Id = o.Id,
    //             Symbol = o.Symbol,
    //             LongExchange = o.LongExchange.Name,
    //             ShortExchange = o.ShortExchange.Name,
    //             LongFundingRate = o.LongFundingRate,
    //             ShortFundingRate = o.ShortFundingRate,
    //             SpreadRate = o.SpreadRate,
    //             AnnualizedSpread = o.AnnualizedSpread,
    //             EstimatedProfitPercentage = o.EstimatedProfitPercentage,
    //             Status = o.Status,
    //             DetectedAt = o.DetectedAt,
    //             ExecutedAt = o.ExecutedAt
    //         })
    //         .ToListAsync();
    //
    //     return opportunities;
    // }

    // COMMENTED OUT: ActiveOpportunity.ArbitrageOpportunity navigation property no longer exists
    // [HttpGet("executing")]
    // public async Task<ActionResult<List<ActiveOpportunityDto>>> GetExecutingOpportunities()
    // {
    //     var activeOpportunities = await _context.ActiveOpportunities
    //         .Include(ao => ao.ArbitrageOpportunity)
    //             .ThenInclude(o => o.LongExchange)
    //         .Include(ao => ao.ArbitrageOpportunity)
    //             .ThenInclude(o => o.ShortExchange)
    //         .Where(ao => ao.IsActive)
    //         .OrderByDescending(ao => ao.ExecutedAt)
    //         .Select(ao => new ActiveOpportunityDto
    //         {
    //             Id = ao.Id,
    //             Symbol = ao.ArbitrageOpportunity.Symbol,
    //             Exchange = ao.Exchange ?? "",
    //             LongExchange = ao.ArbitrageOpportunity.LongExchange.Name,
    //             ShortExchange = ao.ArbitrageOpportunity.ShortExchange.Name,
    //             Strategy = (int)ao.Strategy,
    //             FundingRate = ao.Strategy == Models.ArbitrageStrategy.SpotPerpetual
    //                 ? ao.ArbitrageOpportunity.ShortFundingRate
    //                 : 0,
    //             LongFundingRate = ao.ArbitrageOpportunity.LongFundingRate,
    //             ShortFundingRate = ao.ArbitrageOpportunity.ShortFundingRate,
    //             SpreadRate = ao.ArbitrageOpportunity.SpreadRate,
    //             AnnualizedSpread = ao.ArbitrageOpportunity.AnnualizedSpread,
    //             EstimatedProfitPercentage = ao.ArbitrageOpportunity.EstimatedProfitPercentage,
    //             ExecutedAt = ao.ExecutedAt,
    //             ClosedAt = ao.ClosedAt,
    //             PositionSizeUsd = ao.PositionSizeUsd,
    //             Leverage = ao.Leverage,
    //             StopLossPercentage = ao.StopLossPercentage,
    //             TakeProfitPercentage = ao.TakeProfitPercentage,
    //             CurrentPnL = ao.CurrentPnL,
    //             NetFundingFees = ao.NetFundingFees,
    //             IsActive = ao.IsActive,
    //             Notes = ao.Notes
    //         })
    //         .ToListAsync();
    //
    //     return activeOpportunities;
    // }

    [HttpGet("execution-balances")]
    public async Task<ActionResult<ExecutionBalancesDto>> GetExecutionBalances(
        [FromQuery] string exchange,
        [FromQuery] decimal maxLeverage = 5)
    {
        try
        {
            if (string.IsNullOrEmpty(exchange))
            {
                return BadRequest("Exchange parameter is required");
            }

            var balances = await _executionService.GetExecutionBalancesAsync(exchange, maxLeverage);
            return Ok(balances);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching execution balances for {Exchange}", exchange);
            return StatusCode(500, new { error = $"Failed to fetch balances: {ex.Message}" });
        }
    }

    [HttpPost("execute")]
    public async Task<ActionResult<ExecuteOpportunityResponse>> ExecuteOpportunity([FromBody] ExecuteOpportunityRequest request)
    {
        try
        {
            _logger.LogInformation(
                "Execute request received for {Symbol} on {Exchange}, Strategy: {Strategy}",
                request.Symbol, request.Exchange, request.Strategy);

            var response = await _executionService.ExecuteOpportunityAsync(request);

            if (!response.Success)
            {
                return BadRequest(response);
            }

            return Ok(response);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in execute endpoint");
            return StatusCode(500, new ExecuteOpportunityResponse
            {
                Success = false,
                ErrorMessage = $"Server error: {ex.Message}"
            });
        }
    }

    [HttpPost("close/{activeOpportunityId}")]
    public async Task<ActionResult<CloseOpportunityResponse>> CloseOpportunity(int activeOpportunityId)
    {
            _logger.LogInformation("Close request received for ActiveOpportunity {Id} (deprecated endpoint - using StopExecutionAsync)", activeOpportunityId);

            // Deprecated: This endpoint now calls StopExecutionAsync instead of the old CloseOpportunityAsync
            return await StopExecution(activeOpportunityId);
    }

    [HttpPost("stop/{executionId}")]
    public async Task<ActionResult<CloseOpportunityResponse>> StopExecution(int executionId)
    {
        try
        {
            _logger.LogInformation("Stop request received for Execution {Id}", executionId);

            var response = await _executionService.StopExecutionAsync(executionId);

            if (!response.Success)
            {
                return BadRequest(response);
            }

            return Ok(response);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in stop execution endpoint");
            return StatusCode(500, new CloseOpportunityResponse
            {
                Success = false,
                ErrorMessage = $"Server error: {ex.Message}"
            });
        }
    }
}
