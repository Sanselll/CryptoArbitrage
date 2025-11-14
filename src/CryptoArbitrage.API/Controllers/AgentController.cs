using System.Text.Json;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models.Agent;
using CryptoArbitrage.API.Services.Agent;
using CryptoArbitrage.API.Services.Authentication;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Controllers;

/// <summary>
/// Controller for autonomous trading agent management.
/// Allows users to start/stop agents, configure settings, and view statistics.
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Authorize]
public class AgentController : BaseController
{
    private readonly ArbitrageDbContext _context;
    private readonly IAgentConfigurationService _configService;
    private readonly ICurrentUserService _currentUser;
    private readonly IConfiguration _configuration;
    private readonly AgentDecisionRepository _decisionRepository;

    public AgentController(
        ArbitrageDbContext context,
        IAgentConfigurationService configService,
        ICurrentUserService currentUser,
        IConfiguration configuration,
        AgentDecisionRepository decisionRepository,
        ILogger<AgentController> logger) : base(logger)
    {
        _context = context;
        _configService = configService;
        _currentUser = currentUser;
        _configuration = configuration;
        _decisionRepository = decisionRepository;
    }

    /// <summary>
    /// Start autonomous trading agent for current user.
    /// </summary>
    /// <param name="config">Optional configuration (uses existing/default if not provided)</param>
    [HttpPost("start")]
    public async Task<ActionResult<AgentStatusDto>> StartAgent([FromBody] AgentConfigDto? config = null)
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = _currentUser.UserId!;

            // Check if already running
            var existingSession = await _context.AgentSessions
                .Where(s => s.UserId == userId && s.Status == AgentStatus.Running)
                .FirstOrDefaultAsync();

            if (existingSession != null)
            {
                return BadRequest(new { error = "Agent is already running" });
            }

            // Get or update configuration
            var agentConfig = config != null
                ? await _configService.UpdateConfigurationAsync(
                    userId,
                    config.MaxLeverage,
                    config.TargetUtilization,
                    config.MaxPositions)
                : await _configService.GetOrCreateConfigurationAsync(userId);

            // Create new session
            var session = new AgentSession
            {
                UserId = userId,
                AgentConfigurationId = agentConfig.Id,
                Status = AgentStatus.Running,
                StartedAt = DateTime.UtcNow,
                HoldDecisions = 0,
                EnterDecisions = 0,
                ExitDecisions = 0,
                WinningTrades = 0,
                LosingTrades = 0,
                SessionPnLUsd = 0,
                SessionPnLPct = 0,
                ActivePositions = 0,
                MaxActivePositions = 0,
                CreatedAt = DateTime.UtcNow,
                UpdatedAt = DateTime.UtcNow
            };

            _context.AgentSessions.Add(session);
            await _context.SaveChangesAsync();

            Logger.LogInformation("Started agent for user {UserId}", userId);

            // Return status
            return Ok(await GetAgentStatusInternalAsync(userId));
        }, "start-agent");
    }

    /// <summary>
    /// Stop autonomous trading agent for current user.
    /// </summary>
    [HttpPost("stop")]
    public async Task<ActionResult> StopAgent()
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = _currentUser.UserId!;

            var session = await _context.AgentSessions
                .Where(s => s.UserId == userId && s.Status == AgentStatus.Running)
                .FirstOrDefaultAsync();

            if (session == null)
            {
                return NotFound(new { error = "No running agent found" });
            }

            // Update session status
            session.Status = AgentStatus.Stopped;
            session.StoppedAt = DateTime.UtcNow;
            session.UpdatedAt = DateTime.UtcNow;

            await _context.SaveChangesAsync();

            Logger.LogInformation("Stopped agent for user {UserId}", userId);

            return Ok(new { message = "Agent stopped successfully" });
        }, "stop-agent");
    }

    /// <summary>
    /// Pause autonomous trading agent for current user.
    /// </summary>
    [HttpPost("pause")]
    public async Task<ActionResult> PauseAgent()
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = _currentUser.UserId!;

            var session = await _context.AgentSessions
                .Where(s => s.UserId == userId && s.Status == AgentStatus.Running)
                .FirstOrDefaultAsync();

            if (session == null)
            {
                return NotFound(new { error = "No running agent found" });
            }

            // Update session status
            session.Status = AgentStatus.Paused;
            session.PausedAt = DateTime.UtcNow;
            session.UpdatedAt = DateTime.UtcNow;

            await _context.SaveChangesAsync();

            Logger.LogInformation("Paused agent for user {UserId}", userId);

            return Ok(new { message = "Agent paused successfully" });
        }, "pause-agent");
    }

    /// <summary>
    /// Resume paused trading agent for current user.
    /// </summary>
    [HttpPost("resume")]
    public async Task<ActionResult> ResumeAgent()
    {
        return await ExecuteActionAsync(async () =>
        {
            var userId = _currentUser.UserId!;

            var session = await _context.AgentSessions
                .Where(s => s.UserId == userId && s.Status == AgentStatus.Paused)
                .FirstOrDefaultAsync();

            if (session == null)
            {
                return NotFound(new { error = "No paused agent found" });
            }

            // Update session status
            session.Status = AgentStatus.Running;
            session.PausedAt = null;
            session.UpdatedAt = DateTime.UtcNow;

            await _context.SaveChangesAsync();

            Logger.LogInformation("Resumed agent for user {UserId}", userId);

            return Ok(new { message = "Agent resumed successfully" });
        }, "resume-agent");
    }

    /// <summary>
    /// Get agent status and statistics for current user.
    /// </summary>
    [HttpGet("status")]
    public async Task<ActionResult<AgentStatusDto>> GetStatus()
    {
        return await ExecuteAsync(async () =>
        {
            var userId = _currentUser.UserId!;
            return await GetAgentStatusInternalAsync(userId);
        }, "get-agent-status");
    }

    /// <summary>
    /// Update agent configuration (only allowed when stopped).
    /// </summary>
    /// <param name="config">New configuration</param>
    [HttpPut("config")]
    public async Task<ActionResult<AgentConfigDto>> UpdateConfig([FromBody] AgentConfigDto config)
    {
        return await ExecuteAsync(async () =>
        {
            var userId = _currentUser.UserId!;

            var updatedConfig = await _configService.UpdateConfigurationAsync(
                userId,
                config.MaxLeverage,
                config.TargetUtilization,
                config.MaxPositions);

            Logger.LogInformation("Updated agent config for user {UserId}", userId);

            return new AgentConfigDto
            {
                MaxLeverage = updatedConfig.MaxLeverage,
                TargetUtilization = updatedConfig.TargetUtilization,
                MaxPositions = updatedConfig.MaxPositions,
                PredictionIntervalSeconds = updatedConfig.PredictionIntervalSeconds
            };
        }, "update-agent-config");
    }

    /// <summary>
    /// Get agent configuration for current user.
    /// </summary>
    [HttpGet("config")]
    public async Task<ActionResult<AgentConfigDto>> GetConfig()
    {
        return await ExecuteAsync(async () =>
        {
            var userId = _currentUser.UserId!;
            var config = await _configService.GetOrCreateConfigurationAsync(userId);

            return new AgentConfigDto
            {
                MaxLeverage = config.MaxLeverage,
                TargetUtilization = config.TargetUtilization,
                MaxPositions = config.MaxPositions,
                PredictionIntervalSeconds = config.PredictionIntervalSeconds
            };
        }, "get-agent-config");
    }

    /// <summary>
    /// Get agent statistics for current user (from current session).
    /// </summary>
    [HttpGet("stats")]
    public async Task<ActionResult<AgentStatsDto>> GetStats()
    {
        return await ExecuteAsync(async () =>
        {
            var userId = _currentUser.UserId!;

            var session = await _context.AgentSessions
                .Where(s => s.UserId == userId)
                .OrderByDescending(s => s.CreatedAt)
                .FirstOrDefaultAsync();

            if (session == null)
            {
                return new AgentStatsDto
                {
                    TotalDecisions = 0,
                    TotalTrades = 0,
                    WinRate = 0
                };
            }

            // Calculate stats from decision repository (real-time, accurate)
            var (pnlUsd, pnlPct, winningTrades, losingTrades) = _decisionRepository.GetSessionMetrics(session.Id);
            var totalTrades = winningTrades + losingTrades;
            var winRate = totalTrades > 0 ? (decimal)winningTrades / totalTrades : 0m;

            // Use database fields for decision counts
            var totalDecisions = session.HoldDecisions + session.EnterDecisions + session.ExitDecisions;

            return new AgentStatsDto
            {
                TotalDecisions = totalDecisions,
                HoldDecisions = session.HoldDecisions,
                EnterDecisions = session.EnterDecisions,
                ExitDecisions = session.ExitDecisions,
                TotalTrades = totalTrades,
                WinningTrades = winningTrades,
                LosingTrades = losingTrades,
                WinRate = winRate,
                TotalPnLUsd = pnlUsd,
                TotalPnLPct = pnlPct,
                TodayPnLUsd = pnlUsd,  // Same as total for now (no multi-day sessions)
                TodayPnLPct = pnlPct,
                ActivePositions = session.ActivePositions
            };
        }, "get-agent-stats");
    }

    /// <summary>
    /// Get agent decisions from in-memory repository for current user.
    /// </summary>
    /// <param name="limit">Maximum number of decisions to return (default: 100)</param>
    [HttpGet("decisions")]
    public async Task<ActionResult<List<AgentDecisionRecord>>> GetDecisions([FromQuery] int limit = 100)
    {
        return await ExecuteAsync(async () =>
        {
            var userId = _currentUser.UserId!;

            // Get decisions from in-memory repository
            var decisions = _decisionRepository.GetByUserId(userId);

            // The repository already limits to 100, but respect the requested limit
            if (limit > 0 && limit < 100)
            {
                decisions = decisions.Take(limit).ToList();
            }

            return decisions;
        }, "get-agent-decisions");
    }

    /// <summary>
    /// Internal helper to get agent status
    /// </summary>
    private async Task<AgentStatusDto> GetAgentStatusInternalAsync(string userId)
    {
        var session = await _context.AgentSessions
            .Where(s => s.UserId == userId)
            .OrderByDescending(s => s.CreatedAt)
            .Include(s => s.AgentConfiguration)
            .FirstOrDefaultAsync();

        if (session == null)
        {
            return new AgentStatusDto
            {
                Status = "stopped",
                TotalPredictions = 0
            };
        }

        var durationSeconds = session.Status == AgentStatus.Running && session.StartedAt.HasValue
            ? (int)(DateTime.UtcNow - session.StartedAt.Value).TotalSeconds
            : 0;

        // Calculate stats from decision repository (real-time, accurate)
        var (pnlUsd, pnlPct, winningTrades, losingTrades) = _decisionRepository.GetSessionMetrics(session.Id);
        var totalTrades = winningTrades + losingTrades;
        var winRate = totalTrades > 0 ? (decimal)winningTrades / totalTrades : 0m;

        // Use database fields for decision counts
        var totalDecisions = session.HoldDecisions + session.EnterDecisions + session.ExitDecisions;

        return new AgentStatusDto
        {
            Status = session.Status.ToString().ToLower(),
            StartedAt = session.StartedAt,
            PausedAt = session.PausedAt,
            DurationSeconds = durationSeconds,
            ErrorMessage = session.ErrorMessage,
            TotalPredictions = totalDecisions,
            Config = session.AgentConfiguration != null ? new AgentConfigDto
            {
                MaxLeverage = session.AgentConfiguration.MaxLeverage,
                TargetUtilization = session.AgentConfiguration.TargetUtilization,
                MaxPositions = session.AgentConfiguration.MaxPositions,
                PredictionIntervalSeconds = session.AgentConfiguration.PredictionIntervalSeconds
            } : null,
            Stats = new AgentStatsDto
            {
                TotalDecisions = totalDecisions,
                HoldDecisions = session.HoldDecisions,
                EnterDecisions = session.EnterDecisions,
                ExitDecisions = session.ExitDecisions,
                TotalTrades = totalTrades,
                WinningTrades = winningTrades,
                LosingTrades = losingTrades,
                WinRate = winRate,
                TotalPnLUsd = pnlUsd,
                TotalPnLPct = pnlPct,
                TodayPnLUsd = pnlUsd,  // Same as total for now (no multi-day sessions)
                TodayPnLPct = pnlPct,
                ActivePositions = session.ActivePositions
            }
        };
    }
}
