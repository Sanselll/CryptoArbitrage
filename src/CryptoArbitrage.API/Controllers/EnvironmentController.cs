using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;

namespace CryptoArbitrage.API.Controllers;

/// <summary>
/// Provides environment information to the client application.
/// This endpoint is public (no [Authorize]) so the UI can display environment mode before login.
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class EnvironmentController : BaseController
{
    private readonly IConfiguration _configuration;
    private readonly IHttpClientFactory _httpClientFactory;

    public EnvironmentController(IConfiguration configuration, ILogger<EnvironmentController> logger, IHttpClientFactory httpClientFactory)
        : base(logger)
    {
        _configuration = configuration;
        _httpClientFactory = httpClientFactory;
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

        Logger.LogDebug("Environment status requested: IsLive={IsLive}, Mode={Mode}", isLive, mode);

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

        Logger.LogDebug("Supported exchanges requested: {Count} exchanges", exchanges.Count);

        return Ok(new
        {
            exchanges = exchanges,
            timestamp = DateTime.UtcNow
        });
    }

    /// <summary>
    /// Gets the server's public IP address for API key whitelisting.
    /// Public endpoint - no authentication required.
    /// </summary>
    [HttpGet("server-ip")]
    public async Task<IActionResult> GetServerIp()
    {
        return await ExecuteActionAsync(async () =>
        {
            var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(5);
            var ip = await client.GetStringAsync("https://api.ipify.org");

            Logger.LogDebug("Server IP requested: {ServerIp}", ip?.Trim());

            return Ok(new
            {
                ip = ip?.Trim(),
                timestamp = DateTime.UtcNow
            });
        }, "fetching server IP");
    }
}
