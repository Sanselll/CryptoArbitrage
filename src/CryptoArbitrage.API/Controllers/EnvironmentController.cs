using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;

namespace CryptoArbitrage.API.Controllers;

/// <summary>
/// Provides environment information to the client application.
/// This endpoint is public (no [Authorize]) so the UI can display environment mode before login.
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class EnvironmentController : ControllerBase
{
    private readonly IConfiguration _configuration;
    private readonly ILogger<EnvironmentController> _logger;

    public EnvironmentController(IConfiguration configuration, ILogger<EnvironmentController> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    /// <summary>
    /// Gets the current environment status (Demo or Live).
    /// Public endpoint - no authentication required.
    /// </summary>
    [HttpGet("status")]
    public IActionResult GetStatus()
    {
        var isLive = _configuration.GetValue<bool>("Environment:IsLive");
        var mode = _configuration.GetValue<string>("Environment:Mode") ?? (isLive ? "Live" : "Demo");

        _logger.LogDebug("Environment status requested: IsLive={IsLive}, Mode={Mode}", isLive, mode);

        return Ok(new
        {
            isLive = isLive,
            mode = mode,
            timestamp = DateTime.UtcNow
        });
    }

    /// <summary>
    /// Gets the list of supported exchanges configured in the system.
    /// Public endpoint - no authentication required.
    /// </summary>
    [HttpGet("exchanges")]
    public IActionResult GetSupportedExchanges()
    {
        var exchanges = _configuration.GetSection("ArbitrageConfig:Exchanges")
            .GetChildren()
            .Where(e => e.GetValue<bool>("IsEnabled"))
            .Select(e => e.GetValue<string>("Name"))
            .Where(name => !string.IsNullOrEmpty(name))
            .ToList();

        _logger.LogDebug("Supported exchanges requested: {Count} exchanges", exchanges.Count);

        return Ok(new
        {
            exchanges = exchanges,
            timestamp = DateTime.UtcNow
        });
    }
}
