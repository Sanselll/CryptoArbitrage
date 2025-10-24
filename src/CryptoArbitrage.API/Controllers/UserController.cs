using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Services;
using CryptoArbitrage.API.Services.Authentication;
using CryptoArbitrage.API.Services.Exchanges;

namespace CryptoArbitrage.API.Controllers;

/// <summary>
/// Manages user profile and exchange API keys.
/// All endpoints are protected with [Authorize] attribute.
/// User identity is extracted from JWT token, never from client input.
/// </summary>
[Authorize]
[ApiController]
[Route("api/[controller]")]
public class UserController : BaseController
{
    private readonly ArbitrageDbContext _db;
    private readonly ICurrentUserService _currentUser;
    private readonly IEncryptionService _encryption;
    private readonly IApiKeyValidator _apiKeyValidator;

    public UserController(
        ArbitrageDbContext db,
        ICurrentUserService currentUser,
        IEncryptionService encryption,
        ILogger<UserController> logger,
        IApiKeyValidator apiKeyValidator)
        : base(logger)
    {
        _db = db;
        _currentUser = currentUser;
        _encryption = encryption;
        _apiKeyValidator = apiKeyValidator;
    }

    /// <summary>
    /// Gets the current user's profile information.
    /// </summary>
    [HttpGet("profile")]
    public async Task<IActionResult> GetProfile()
    {
        if (string.IsNullOrEmpty(_currentUser.UserId))
            return Unauthorized(new { error = "User not authenticated" });

        var user = await _db.Users.FindAsync(_currentUser.UserId);
        if (user == null)
            return NotFound(new { error = "User not found" });

        return Ok(new
        {
            id = user.Id,
            email = user.Email,
            userName = user.UserName,
            createdAt = user.CreatedAt,
            lastLoginAt = user.LastLoginAt
        });
    }

    /// <summary>
    /// Gets all exchange API keys for the authenticated user.
    /// API keys are masked in the response (only last 4 characters visible).
    /// </summary>
    [HttpGet("apikeys")]
    public async Task<IActionResult> GetApiKeys()
    {
        var keys = await _db.UserExchangeApiKeys
            .Where(k => k.UserId == _currentUser.UserId)
            .Select(k => new
            {
                k.Id,
                k.ExchangeName,
                apiKey = MaskApiKey(k.EncryptedApiKey),
                k.IsEnabled,
                k.CreatedAt,
                k.LastTestedAt,
                k.LastTestResult
            })
            .ToListAsync();

        Logger.LogDebug("User {UserId} retrieved {Count} API keys", _currentUser.UserId, keys.Count);
        return Ok(keys);
    }

    /// <summary>
    /// Adds a new exchange API key for the authenticated user.
    /// API key and secret are encrypted before storage.
    /// </summary>
    [HttpPost("apikeys")]
    public async Task<IActionResult> AddApiKey([FromBody] AddApiKeyRequest request, [FromServices] IServiceProvider serviceProvider)
    {
        if (string.IsNullOrEmpty(_currentUser.UserId))
            return Unauthorized(new { error = "User not authenticated" });

        // Verify user exists in database (prevent FK constraint violations)
        var userExists = await _db.Users.FindAsync(_currentUser.UserId);
        if (userExists == null)
        {
            Logger.LogWarning("User {UserId} has valid JWT but doesn't exist in database. User needs to re-authenticate.", _currentUser.UserId);
            return Unauthorized(new { error = "Session expired. Please log out and log back in." });
        }

        // Validate exchange name
        if (!new[] { "Binance", "Bybit" }.Contains(request.ExchangeName))
            return BadRequest(new { error = "Invalid exchange name. Supported: Binance, Bybit" });

        // Validate API credentials
        if (string.IsNullOrWhiteSpace(request.ApiKey) || string.IsNullOrWhiteSpace(request.ApiSecret))
            return BadRequest(new { error = "API key and secret are required" });

        try
        {
            // Check if user already has a key for this exchange
            var existingKey = await _db.UserExchangeApiKeys
                .FirstOrDefaultAsync(k => k.UserId == _currentUser.UserId && k.ExchangeName == request.ExchangeName);

            if (existingKey != null)
            {
                return BadRequest(new { error = $"API key already exists for {request.ExchangeName}" });
            }

            // VALIDATE API KEY BEFORE SAVING TO DATABASE
            Logger.LogInformation("Validating API credentials for {Exchange} before saving...", request.ExchangeName);

            IExchangeConnector? connector = null;
            try
            {
                connector = request.ExchangeName switch
                {
                    "Binance" => serviceProvider.GetRequiredService<BinanceConnector>(),
                    "Bybit" => serviceProvider.GetRequiredService<BybitConnector>(),
                    _ => null
                };

                if (connector == null)
                {
                    return BadRequest(new { error = $"Exchange {request.ExchangeName} is not supported" });
                }

                // Test the API credentials by connecting to the exchange
                var connectionSuccess = await connector.ConnectAsync(request.ApiKey, request.ApiSecret);

                if (!connectionSuccess)
                {
                    Logger.LogWarning("API key validation failed for user {UserId} on {Exchange}",
                        _currentUser.UserId, request.ExchangeName);
                    return BadRequest(new { error = $"Invalid API credentials. Failed to connect to {request.ExchangeName}. Please verify your API key and secret." });
                }

                Logger.LogInformation("API credentials validated successfully for {Exchange}", request.ExchangeName);

                // Validate API key permissions and restrictions
                Logger.LogInformation("Checking API key permissions and restrictions for {Exchange}...", request.ExchangeName);
                var validationResult = await _apiKeyValidator.ValidateApiKeyAsync(request.ExchangeName, request.ApiKey, request.ApiSecret);

                if (!validationResult.IsValid)
                {
                    Logger.LogWarning("API key permissions check failed for user {UserId} on {Exchange}: {Message}",
                        _currentUser.UserId, request.ExchangeName, validationResult.DetailedMessage);

                    return BadRequest(new
                    {
                        error = "API key validation failed",
                        detailedMessage = validationResult.DetailedMessage,
                        missingPermissions = validationResult.MissingPermissions,
                        isIpRestricted = validationResult.IsIpRestricted,
                        allowedIps = validationResult.AllowedIps,
                        serverIp = validationResult.ServerIp
                    });
                }

                Logger.LogInformation("API key permissions validated successfully for {Exchange}", request.ExchangeName);
            }
            finally
            {
                // Cleanup: disconnect after validation
                if (connector != null)
                {
                    try
                    {
                        await connector.DisconnectAsync();
                    }
                    catch
                    {
                        // Ignore disconnect errors
                    }
                }
            }

            // API key is valid - now save it to database
            var encryptedKey = _encryption.Encrypt(request.ApiKey);
            var encryptedSecret = _encryption.Encrypt(request.ApiSecret);

            var apiKey = new UserExchangeApiKey
            {
                UserId = _currentUser.UserId,
                ExchangeName = request.ExchangeName,
                EncryptedApiKey = encryptedKey,
                EncryptedApiSecret = encryptedSecret,
                IsEnabled = true,
                CreatedAt = DateTime.UtcNow,
                LastTestedAt = DateTime.UtcNow,
                LastTestResult = "Success: Connected to exchange"
            };

            _db.UserExchangeApiKeys.Add(apiKey);
            await _db.SaveChangesAsync();

            Logger.LogInformation(
                "User {UserId} added and validated API key for {Exchange}",
                _currentUser.UserId, request.ExchangeName);

            return Ok(new { id = apiKey.Id, message = "API key added and validated successfully" });
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error adding API key for user {UserId}", _currentUser.UserId);
            return StatusCode(500, new { error = "Failed to add API key" });
        }
    }

    /// <summary>
    /// Updates an existing exchange API key.
    /// Can update API key, secret, enabled status, or demo trading flag.
    /// CRITICAL: Validates user owns the API key before allowing updates.
    /// </summary>
    [HttpPut("apikeys/{id}")]
    public async Task<IActionResult> UpdateApiKey(int id, [FromBody] UpdateApiKeyRequest request)
    {
        var apiKey = await _db.UserExchangeApiKeys.FindAsync(id);
        if (apiKey == null)
            return NotFound(new { error = "API key not found" });

        // CRITICAL: Verify user owns this API key
        try
        {
            _currentUser.ValidateUserOwnsResource(apiKey.UserId);
        }
        catch (UnauthorizedAccessException)
        {
            Logger.LogWarning("User {UserId} attempted to update API key {KeyId} owned by {Owner}",
                _currentUser.UserId, id, apiKey.UserId);
            return Forbid();
        }

        try
        {
            // Update only provided fields
            if (!string.IsNullOrEmpty(request.ApiKey))
                apiKey.EncryptedApiKey = _encryption.Encrypt(request.ApiKey);

            if (!string.IsNullOrEmpty(request.ApiSecret))
                apiKey.EncryptedApiSecret = _encryption.Encrypt(request.ApiSecret);

            if (request.IsEnabled.HasValue)
                apiKey.IsEnabled = request.IsEnabled.Value;

            await _db.SaveChangesAsync();

            Logger.LogInformation("User {UserId} updated API key {KeyId} for {Exchange}",
                _currentUser.UserId, id, apiKey.ExchangeName);

            return Ok(new { message = "API key updated successfully" });
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error updating API key {KeyId} for user {UserId}", id, _currentUser.UserId);
            return StatusCode(500, new { error = "Failed to update API key" });
        }
    }

    /// <summary>
    /// Deletes an exchange API key.
    /// CRITICAL: Validates user owns the API key before allowing deletion.
    /// </summary>
    [HttpDelete("apikeys/{id}")]
    public async Task<IActionResult> DeleteApiKey(int id)
    {
        var apiKey = await _db.UserExchangeApiKeys.FindAsync(id);
        if (apiKey == null)
            return NotFound(new { error = "API key not found" });

        // CRITICAL: Verify user owns this API key
        try
        {
            _currentUser.ValidateUserOwnsResource(apiKey.UserId);
        }
        catch (UnauthorizedAccessException)
        {
            Logger.LogWarning("User {UserId} attempted to delete API key {KeyId} owned by {Owner}",
                _currentUser.UserId, id, apiKey.UserId);
            return Forbid();
        }

        try
        {
            _db.UserExchangeApiKeys.Remove(apiKey);
            await _db.SaveChangesAsync();

            Logger.LogInformation("User {UserId} deleted API key {KeyId} for {Exchange}",
                _currentUser.UserId, id, apiKey.ExchangeName);

            return Ok(new { message = "API key deleted successfully" });
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error deleting API key {KeyId} for user {UserId}", id, _currentUser.UserId);
            return StatusCode(500, new { error = "Failed to delete API key" });
        }
    }

    /// <summary>
    /// Tests an API key by attempting to connect to the actual exchange API.
    /// Uses the same ApiKeyValidator as the Add API Key endpoint for consistency.
    /// CRITICAL: Validates user owns the API key before testing.
    /// </summary>
    [HttpPost("apikeys/{id}/test")]
    public async Task<IActionResult> TestApiKey(int id)
    {
        var apiKey = await _db.UserExchangeApiKeys.FindAsync(id);
        if (apiKey == null)
            return NotFound(new { error = "API key not found" });

        // CRITICAL: Verify user owns this API key
        try
        {
            _currentUser.ValidateUserOwnsResource(apiKey.UserId);
        }
        catch (UnauthorizedAccessException)
        {
            Logger.LogWarning("User {UserId} attempted to test API key {KeyId} owned by {Owner}",
                _currentUser.UserId, id, apiKey.UserId);
            return Forbid();
        }

        try
        {
            // Decrypt API credentials
            var key = _encryption.Decrypt(apiKey.EncryptedApiKey);
            var secret = _encryption.Decrypt(apiKey.EncryptedApiSecret);

            if (string.IsNullOrEmpty(key) || string.IsNullOrEmpty(secret))
            {
                apiKey.LastTestedAt = DateTime.UtcNow;
                apiKey.LastTestResult = "Failed: Invalid decryption";
                await _db.SaveChangesAsync();

                return Ok(new
                {
                    success = false,
                    message = "Failed to decrypt API credentials",
                    testedAt = apiKey.LastTestedAt
                });
            }

            // Use the same ApiKeyValidator as the Add API Key endpoint
            Logger.LogInformation("Testing API key {KeyId} for user {UserId} on {Exchange}",
                id, _currentUser.UserId, apiKey.ExchangeName);

            var validationResult = await _apiKeyValidator.ValidateApiKeyAsync(apiKey.ExchangeName, key, secret);

            // Update last test result
            apiKey.LastTestedAt = DateTime.UtcNow;

            if (validationResult.IsValid)
            {
                apiKey.LastTestResult = "Success: Connected to exchange";
            }
            else if (validationResult.IsIpRestricted)
            {
                apiKey.LastTestResult = $"Failed: IP-restricted (Server IP: {validationResult.ServerIp ?? "unknown"})";
            }
            else
            {
                apiKey.LastTestResult = $"Failed: {validationResult.DetailedMessage}";
            }

            await _db.SaveChangesAsync();

            Logger.LogInformation("User {UserId} tested API key {KeyId} for {Exchange}: {Result}",
                _currentUser.UserId, id, apiKey.ExchangeName, apiKey.LastTestResult);

            // Return detailed result
            if (validationResult.IsValid)
            {
                return Ok(new
                {
                    success = true,
                    message = $"Successfully connected to {apiKey.ExchangeName} API",
                    testedAt = apiKey.LastTestedAt
                });
            }
            else
            {
                return Ok(new
                {
                    success = false,
                    message = validationResult.DetailedMessage,
                    isIpRestricted = validationResult.IsIpRestricted,
                    serverIp = validationResult.ServerIp,
                    allowedIps = validationResult.AllowedIps,
                    missingPermissions = validationResult.MissingPermissions,
                    testedAt = apiKey.LastTestedAt
                });
            }
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, "Error testing API key {KeyId} for user {UserId}", id, _currentUser.UserId);

            apiKey.LastTestedAt = DateTime.UtcNow;
            apiKey.LastTestResult = $"Error: {ex.Message}";
            await _db.SaveChangesAsync();

            return Ok(new
            {
                success = false,
                message = "API key test failed",
                error = ex.Message,
                testedAt = apiKey.LastTestedAt
            });
        }
    }

    /// <summary>
    /// Masks an API key for display purposes.
    /// Shows only the first 4 and last 4 characters (or **** if too short).
    /// </summary>
    private static string MaskApiKey(string encryptedKey)
    {
        if (string.IsNullOrEmpty(encryptedKey) || encryptedKey.Length < 8)
            return "****";

        return encryptedKey.Substring(0, 4) + "****" + encryptedKey.Substring(encryptedKey.Length - 4);
    }
}

/// <summary>
/// Request model for adding a new API key.
/// </summary>
public record AddApiKeyRequest(string ExchangeName, string ApiKey, string ApiSecret);

/// <summary>
/// Request model for updating an existing API key.
/// All fields are optional - only provided fields will be updated.
/// </summary>
public record UpdateApiKeyRequest(string? ApiKey, string? ApiSecret, bool? IsEnabled);
