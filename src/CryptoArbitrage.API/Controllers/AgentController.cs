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

    public AgentController(
        ArbitrageDbContext context,
        IAgentConfigurationService configService,
        ICurrentUserService currentUser,
        IConfiguration configuration,
        ILogger<AgentController> logger) : base(logger)
    {
        _context = context;
        _configService = configService;
        _currentUser = currentUser;
        _configuration = configuration;
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

            var totalDecisions = session.HoldDecisions + session.EnterDecisions + session.ExitDecisions;
            var totalTrades = session.WinningTrades + session.LosingTrades;
            var winRate = totalTrades > 0 ? (decimal)session.WinningTrades / totalTrades : 0m;

            return new AgentStatsDto
            {
                TotalDecisions = totalDecisions,
                HoldDecisions = session.HoldDecisions,
                EnterDecisions = session.EnterDecisions,
                ExitDecisions = session.ExitDecisions,
                TotalTrades = totalTrades,
                WinningTrades = session.WinningTrades,
                LosingTrades = session.LosingTrades,
                WinRate = winRate,
                TotalPnLUsd = session.SessionPnLUsd,
                TotalPnLPct = session.SessionPnLPct,
                TodayPnLUsd = session.SessionPnLUsd, // Session stats = today's stats
                TodayPnLPct = session.SessionPnLPct,
                ActivePositions = session.ActivePositions
            };
        }, "get-agent-stats");
    }

    /// <summary>
    /// Get agent decisions from ML API for current user.
    /// </summary>
    /// <param name="limit">Maximum number of decisions to return (default: 100)</param>
    [HttpGet("decisions")]
    public async Task<ActionResult<List<Dictionary<string, object>>>> GetDecisions([FromQuery] int limit = 100)
    {
        return await ExecuteAsync(async () =>
        {
            var userId = _currentUser.UserId!;

            // Call ML API to get decisions
            var mlApiUrl = _configuration["MLApi:Host"] ?? "localhost";
            var mlApiPort = _configuration["MLApi:Port"] ?? "5250";
            var baseUrl = $"http://{mlApiUrl}:{mlApiPort}";

            using var httpClient = new HttpClient { BaseAddress = new Uri(baseUrl) };
            var response = await httpClient.GetAsync($"/agent/decisions?user_id={userId}&limit={limit}");

            if (!response.IsSuccessStatusCode)
            {
                Logger.LogWarning("Failed to fetch decisions from ML API: {StatusCode}", response.StatusCode);
                return new List<Dictionary<string, object>>();
            }

            var responseJson = await response.Content.ReadAsStringAsync();
            var jsonDoc = JsonDocument.Parse(responseJson);
            var decisions = jsonDoc.RootElement.GetProperty("decisions");

            var result = new List<Dictionary<string, object>>();
            foreach (var decision in decisions.EnumerateArray())
            {
                var dict = new Dictionary<string, object>();
                foreach (var property in decision.EnumerateObject())
                {
                    dict[property.Name] = property.Value.ValueKind switch
                    {
                        JsonValueKind.String => property.Value.GetString() ?? string.Empty,
                        JsonValueKind.Number => property.Value.GetDecimal(),
                        JsonValueKind.True => true,
                        JsonValueKind.False => false,
                        JsonValueKind.Null => null!,
                        _ => property.Value.ToString()
                    };
                }
                result.Add(dict);
            }

            return result;
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

        // Calculate stats from session data
        var totalDecisions = session.HoldDecisions + session.EnterDecisions + session.ExitDecisions;
        var totalTrades = session.WinningTrades + session.LosingTrades;
        var winRate = totalTrades > 0 ? (decimal)session.WinningTrades / totalTrades : 0m;

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
                WinningTrades = session.WinningTrades,
                LosingTrades = session.LosingTrades,
                WinRate = winRate,
                TotalPnLUsd = session.SessionPnLUsd,
                TotalPnLPct = session.SessionPnLPct,
                TodayPnLUsd = session.SessionPnLUsd,
                TodayPnLPct = session.SessionPnLPct,
                ActivePositions = session.ActivePositions
            }
        };
    }
}
