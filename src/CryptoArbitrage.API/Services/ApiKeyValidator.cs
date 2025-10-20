using Binance.Net;
using Binance.Net.Clients;
using Binance.Net.Objects;
using Bybit.Net;
using Bybit.Net.Clients;
using Bybit.Net.Objects;
using CryptoExchange.Net.Authentication;

namespace CryptoArbitrage.API.Services;

/// <summary>
/// Service for validating exchange API key permissions and restrictions
/// </summary>
public class ApiKeyValidator : IApiKeyValidator
{
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<ApiKeyValidator> _logger;
    private readonly IConfiguration _configuration;

    public ApiKeyValidator(
        IHttpClientFactory httpClientFactory,
        ILogger<ApiKeyValidator> logger,
        IConfiguration configuration)
    {
        _httpClientFactory = httpClientFactory;
        _logger = logger;
        _configuration = configuration;
    }

    public async Task<ApiKeyValidationResult> ValidateApiKeyAsync(string exchangeName, string apiKey, string apiSecret)
    {
        _logger.LogInformation("=== API Key Validation Started ===");
        _logger.LogInformation("Exchange: {Exchange}", exchangeName);
        _logger.LogInformation("API Key: {ApiKey}", apiKey?.Substring(0, Math.Min(8, apiKey?.Length ?? 0)) + "...");

        try
        {
            // Detect server's public IP address
            _logger.LogInformation("Detecting server IP address...");
            var serverIp = await DetectServerIpAsync();
            _logger.LogInformation("Server IP detected: {ServerIp}", serverIp ?? "null");

            var result = exchangeName.ToLower() switch
            {
                "binance" => await ValidateBinanceApiKeyAsync(apiKey, apiSecret, serverIp),
                "bybit" => await ValidateBybitApiKeyAsync(apiKey, apiSecret, serverIp),
                _ => ApiKeyValidationResult.Failure(
                    new List<string> { $"Unsupported exchange: {exchangeName}" },
                    false,
                    null,
                    serverIp)
            };

            _logger.LogInformation("=== Validation Result ===");
            _logger.LogInformation("IsValid: {IsValid}", result.IsValid);
            _logger.LogInformation("IsIpRestricted: {IsIpRestricted}", result.IsIpRestricted);
            _logger.LogInformation("MissingPermissions: {Count}", result.MissingPermissions?.Count ?? 0);
            _logger.LogInformation("DetailedMessage: {Message}", result.DetailedMessage);
            _logger.LogInformation("=========================");

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating API key for {Exchange}", exchangeName);
            return ApiKeyValidationResult.Failure(
                new List<string> { $"Validation failed: {ex.Message}" });
        }
    }

    private async Task<string?> DetectServerIpAsync()
    {
        try
        {
            var client = _httpClientFactory.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(5);
            var response = await client.GetStringAsync("https://api.ipify.org");
            return response.Trim();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to detect server IP address");
            return null;
        }
    }

    private async Task<ApiKeyValidationResult> ValidateBinanceApiKeyAsync(string apiKey, string apiSecret, string? serverIp)
    {
        try
        {
            _logger.LogInformation("--- Binance Validation Started ---");

            // Use Binance.Net library
            var isLive = _configuration.GetValue<bool>("Environment:IsLive");
            _logger.LogInformation("Environment: {Environment} (IsLive: {IsLive})",
                isLive ? "Live" : "Demo", isLive);

            var credentials = new ApiCredentials(apiKey, apiSecret);
            var environment = isLive ? BinanceEnvironment.Live : BinanceEnvironment.Demo;

            using var client = new BinanceRestClient(options =>
            {
                options.ApiCredentials = credentials;
                options.Environment = environment;
                options.RequestTimeout = TimeSpan.FromSeconds(10);
            });

            // Get API Key Permissions using the official endpoint
            _logger.LogInformation("Calling Binance SpotApi.Account.GetAPIKeyPermissionsAsync()...");
            var permissionsResult = await client.SpotApi.Account.GetAPIKeyPermissionsAsync();

            _logger.LogInformation("API Call completed. Success: {Success}", permissionsResult.Success);

            if (!permissionsResult.Success)
            {
                _logger.LogWarning("Binance API key permission check failed");
                _logger.LogWarning("Error Code: {Code}", permissionsResult.Error?.Code);
                _logger.LogWarning("Error Message: {Message}", permissionsResult.Error?.Message);

                // Check if it's an IP restriction error
                var isIpError = permissionsResult.Error != null && (
                    permissionsResult.Error.Message.Contains("IP", StringComparison.OrdinalIgnoreCase) ||
                    permissionsResult.Error.Code == -2015);

                if (isIpError)
                {
                    _logger.LogWarning("IP Restriction Detected!");
                    return ApiKeyValidationResult.Failure(
                        new List<string> { "API key is IP-restricted" },
                        isIpRestricted: true,
                        allowedIps: null,
                        serverIp: serverIp);
                }

                return ApiKeyValidationResult.Failure(
                    new List<string> { $"API validation failed: {permissionsResult.Error?.Message ?? "Unknown error"}" });
            }

            // Log each permission
            _logger.LogInformation("--- API Key Permissions ---");
            var permissions = permissionsResult.Data;
            _logger.LogInformation("[Permission] EnableReading: {Value}", permissions.EnableReading);
            _logger.LogInformation("[Permission] EnableSpotAndMarginTrading: {Value}", permissions.EnableSpotAndMarginTrading);
            _logger.LogInformation("[Permission] EnableFutures: {Value}", permissions.EnableFutures);
            _logger.LogInformation("[Permission] EnableWithdrawals: {Value}", permissions.EnableWithdrawals);
            _logger.LogInformation("[Permission] EnableInternalTransfer: {Value}", permissions.EnableInternalTransfer);
            _logger.LogInformation("[Permission] EnableMargin: {Value}", permissions.EnableMargin);
            _logger.LogInformation("[Permission] EnablePortfolioMarginTrading: {Value}", permissions.EnablePortfolioMarginTrading);
            _logger.LogInformation("[Permission] EnableVanillaOptions: {Value}", permissions.EnableVanillaOptions);

            // IP Restrictions
            _logger.LogInformation("--- IP Restrictions ---");
            _logger.LogInformation("IP Restrict: {Value}", permissions.IpRestrict);
            if (permissions.IpRestrict)
            {
                _logger.LogInformation("IP Restriction is ENABLED - Note: Binance API does not return the list of allowed IPs");
                if (!string.IsNullOrEmpty(serverIp))
                {
                    _logger.LogInformation("Server IP: {ServerIp}", serverIp);
                    _logger.LogInformation("If connection fails, add {ServerIp} to your Binance API key whitelist", serverIp);
                }
            }
            else
            {
                _logger.LogInformation("IP Restriction: DISABLED (unrestricted)");

                // SECURITY REQUIREMENT: IP restriction MUST be enabled for live trading
                if (isLive)
                {
                    _logger.LogError("SECURITY ERROR: IP restriction is REQUIRED for live trading but is currently DISABLED");
                    return ApiKeyValidationResult.Failure(
                        new List<string> { "IP Restriction REQUIRED - For security, IP restriction must be enabled for live trading. Please enable IP whitelist in your Binance API key settings." });
                }
                else
                {
                    _logger.LogInformation("IP restriction is optional for demo trading");
                }
            }

            // Validate required permissions
            var missingPermissions = new List<string>();

            if (!permissions.EnableReading)
            {
                missingPermissions.Add("EnableReading - Required to read account data");
            }

            if (!permissions.EnableSpotAndMarginTrading)
            {
                missingPermissions.Add("EnableSpotAndMarginTrading - Required for spot and margin trading");
            }

            if (!permissions.EnableFutures)
            {
                missingPermissions.Add("EnableFutures - Required for futures trading");
            }

            // Note: We cannot check if server IP is in the whitelist because Binance API
            // doesn't return the list of allowed IPs. IP restriction errors will be caught
            // by the error handler above (error code -2015)

            // Summary
            _logger.LogInformation("--- Permission Check Summary ---");
            _logger.LogInformation("Missing Permissions: {Count}", missingPermissions.Count);
            if (missingPermissions.Any())
            {
                foreach (var perm in missingPermissions)
                {
                    _logger.LogWarning("  ✗ {Permission}", perm);
                }
                return ApiKeyValidationResult.Failure(missingPermissions);
            }

            _logger.LogInformation("✓ All required permissions validated successfully!");
            return ApiKeyValidationResult.Success();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating Binance API key");
            return ApiKeyValidationResult.Failure(
                new List<string> { $"Validation error: {ex.Message}" });
        }
    }

    private async Task<ApiKeyValidationResult> ValidateBybitApiKeyAsync(string apiKey, string apiSecret, string? serverIp)
    {
        try
        {
            _logger.LogInformation("--- Bybit Validation Started ---");

            // Use Bybit.Net library
            var isLive = _configuration.GetValue<bool>("Environment:IsLive");
            _logger.LogInformation("Environment: {Environment} (IsLive: {IsLive})",
                isLive ? "Live" : "DemoTrading", isLive);

            var credentials = new ApiCredentials(apiKey, apiSecret);
            var environment = isLive ? BybitEnvironment.Live : BybitEnvironment.DemoTrading;

            using var client = new BybitRestClient(options =>
            {
                options.ApiCredentials = credentials;
                options.Environment = environment;
                options.RequestTimeout = TimeSpan.FromSeconds(10);
            });

            var missingPermissions = new List<string>();

            // SECURITY WARNING: For Bybit, we cannot directly check if IP restriction is enabled
            // We can only detect IP restriction errors when they occur
            if (isLive)
            {
                _logger.LogWarning("--- IP Restriction WARNING ---");
                _logger.LogWarning("IMPORTANT: For live trading, ensure IP restriction is ENABLED in your Bybit API key settings");
                _logger.LogWarning("Bybit API does not provide a way to check IP restriction status programmatically");
                if (!string.IsNullOrEmpty(serverIp))
                {
                    _logger.LogWarning("Server IP: {ServerIp} - Add this to your Bybit API key whitelist", serverIp);
                }
            }

            // Test 1: Read Permission - Get Wallet Balance
            _logger.LogInformation("[Permission Test 1/2] Testing READ permission (Wallet Balance)...");
            _logger.LogInformation("  API Call: V5Api.Account.GetBalancesAsync(AccountType.Unified)");
            var balanceResult = await client.V5Api.Account.GetBalancesAsync(Bybit.Net.Enums.AccountType.Unified);
            _logger.LogInformation("  Result: {Success}", balanceResult.Success ? "✓ PASS" : "✗ FAIL");
            if (!balanceResult.Success)
            {
                _logger.LogWarning("  Error Code: {Code}", balanceResult.Error?.Code);
                _logger.LogWarning("  Error Message: {Message}", balanceResult.Error?.Message);

                // Check if it's an IP restriction error
                var isIpError = balanceResult.Error != null &&
                    balanceResult.Error.Message.Contains("IP", StringComparison.OrdinalIgnoreCase);

                if (isIpError)
                {
                    _logger.LogWarning("  IP Restriction Detected!");
                    return ApiKeyValidationResult.Failure(
                        new List<string> { "API key is IP-restricted" },
                        isIpRestricted: true,
                        allowedIps: null,
                        serverIp: serverIp);
                }

                missingPermissions.Add("READ - Cannot read wallet balance");
            }

            // Test 2: Position Permission - Get Positions
            _logger.LogInformation("[Permission Test 2/2] Testing POSITION permission...");
            _logger.LogInformation("  API Call: V5Api.Trading.GetPositionsAsync(Category.Linear)");
            var positionsResult = await client.V5Api.Trading.GetPositionsAsync(Bybit.Net.Enums.Category.Linear);
            _logger.LogInformation("  Result: {Success}", positionsResult.Success ? "✓ PASS" : "✗ FAIL");
            if (!positionsResult.Success)
            {
                _logger.LogWarning("  Error Code: {Code}", positionsResult.Error?.Code);
                _logger.LogWarning("  Error Message: {Message}", positionsResult.Error?.Message);

                // Check if it's an IP restriction error
                var isIpError = positionsResult.Error != null &&
                    positionsResult.Error.Message.Contains("IP", StringComparison.OrdinalIgnoreCase);

                if (isIpError)
                {
                    _logger.LogWarning("  IP Restriction Detected!");
                    return ApiKeyValidationResult.Failure(
                        new List<string> { "API key is IP-restricted" },
                        isIpRestricted: true,
                        allowedIps: null,
                        serverIp: serverIp);
                }

                missingPermissions.Add("POSITION - Cannot read positions");
            }

            // Summary
            _logger.LogInformation("--- Permission Test Summary ---");
            _logger.LogInformation("Total Permissions Tested: 2 (READ + POSITION)");
            _logger.LogInformation("Failed Permissions: {Count}", missingPermissions.Count);
            if (missingPermissions.Any())
            {
                foreach (var perm in missingPermissions)
                {
                    _logger.LogWarning("  ✗ {Permission}", perm);
                }
            }

            if (missingPermissions.Any())
            {
                return ApiKeyValidationResult.Failure(missingPermissions);
            }

            _logger.LogInformation("✓ All required permissions validated successfully!");
            return ApiKeyValidationResult.Success();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating Bybit API key");
            return ApiKeyValidationResult.Failure(
                new List<string> { $"Validation error: {ex.Message}" });
        }
    }
}
