using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Models;
using CryptoArbitrage.API.Services;
using CryptoArbitrage.API.Services.Arbitrage.Execution;
using CryptoArbitrage.API.Services.Authentication;

namespace CryptoArbitrage.API.Controllers;

/// <summary>
/// Handles arbitrage opportunity detection and execution.
/// All endpoints require [Authorize] attribute for multi-user support.
/// User identity is extracted from JWT token via ICurrentUserService.
/// </summary>
[Authorize]
[ApiController]
[Route("api/[controller]")]
public class OpportunityController : BaseController
{
    private readonly ArbitrageDbContext _context;
    private readonly ArbitrageExecutionService _executionService;
    private readonly ICurrentUserService _currentUser;

    public OpportunityController(
        ArbitrageDbContext context,
        ILogger<OpportunityController> logger,
        ArbitrageExecutionService executionService,
        ICurrentUserService currentUser)
        : base(logger)
    {
        _context = context;
        _executionService = executionService;
        _currentUser = currentUser;
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
        return await ExecuteActionAsync(async () =>
        {
            if (string.IsNullOrEmpty(exchange))
            {
                return BadRequest("Exchange parameter is required");
            }

            var balances = await _executionService.GetExecutionBalancesAsync(exchange, maxLeverage);
            return Ok(balances);
        }, $"fetching execution balances for {exchange}");
    }

    [HttpPost("execute")]
    public async Task<ActionResult<ExecuteOpportunityResponse>> ExecuteOpportunity([FromBody] ExecuteOpportunityRequest request)
    {
        return await ExecuteActionAsync(async () =>
        {
            var authResult = ValidateAuthentication(_currentUser.UserId);
            if (authResult != null)
                return authResult;

            Logger.LogInformation(
                "Execute request received by user {UserId} for {Symbol} on {Exchange}, Strategy: {Strategy}",
                _currentUser.UserId, request.Symbol, request.Exchange, request.Strategy);

            // Execute opportunity (user ID is tracked in Execution record via controller)
            var response = await _executionService.ExecuteOpportunityAsync(request);

            if (!response.Success)
            {
                return BadRequest(response);
            }

            Logger.LogInformation(
                "Execution successful for user {UserId}: {Response}",
                _currentUser.UserId, response);

            return Ok(response);
        }, $"executing opportunity {request.Symbol} for user {_currentUser.UserId}");
    }

    [HttpPost("close/{activeOpportunityId}")]
    public async Task<ActionResult<CloseOpportunityResponse>> CloseOpportunity(int activeOpportunityId)
    {
            Logger.LogInformation("Close request received for ActiveOpportunity {Id} (deprecated endpoint - using StopExecutionAsync)", activeOpportunityId);

            // Deprecated: This endpoint now calls StopExecutionAsync instead of the old CloseOpportunityAsync
            return await StopExecution(activeOpportunityId);
    }

    [HttpPost("stop/{executionId}")]
    public async Task<ActionResult<CloseOpportunityResponse>> StopExecution(int executionId)
    {
        return await ExecuteActionAsync(async () =>
        {
            var authResult = ValidateAuthentication(_currentUser.UserId);
            if (authResult != null)
                return authResult;

            Logger.LogInformation("Stop request received by user {UserId} for Execution {Id}",
                _currentUser.UserId, executionId);

            // CRITICAL: Validate user owns this execution before allowing stop
            var execution = await _context.Executions.FindAsync(executionId);
            if (execution == null)
                return NotFound(new { error = "Execution not found" });

            // Verify user owns this execution
            try
            {
                _currentUser.ValidateUserOwnsResource(execution.UserId);
            }
            catch (UnauthorizedAccessException)
            {
                Logger.LogWarning("User {UserId} attempted to stop execution {ExecutionId} owned by {Owner}",
                    _currentUser.UserId, executionId, execution.UserId);
                return Forbid();
            }

            var response = await _executionService.StopExecutionAsync(executionId);

            if (!response.Success)
            {
                return BadRequest(response);
            }

            Logger.LogInformation("Execution {Id} stopped successfully for user {UserId}",
                executionId, _currentUser.UserId);

            return Ok(response);
        }, $"stopping execution {executionId} for user {_currentUser.UserId}");
    }
}
