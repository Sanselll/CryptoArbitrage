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

            // DEMO MODE: Skip strict validation as testnet doesn't support permission checks
            if (!isLive)
            {
                _logger.LogInformation("Demo mode detected - performing simple connection test instead of strict validation");

                // Simple connection test - try to get account info
                var accountResult = await client.SpotApi.Account.GetAccountInfoAsync();

                if (!accountResult.Success)
                {
                    _logger.LogWarning("Demo connection test failed: {Error}", accountResult.Error?.Message);

                    // Check if it's an IP restriction error
                    var isIpError = accountResult.Error != null && (
                        accountResult.Error.Message.Contains("IP", StringComparison.OrdinalIgnoreCase) ||
                        accountResult.Error.Code == -2015);

                    if (isIpError)
                    {
                        return ApiKeyValidationResult.Failure(
                            new List<string> { "API key is IP-restricted" },
                            isIpRestricted: true,
                            allowedIps: null,
                            serverIp: serverIp);
                    }

                    return ApiKeyValidationResult.Failure(
                        new List<string> { $"Connection test failed: {accountResult.Error?.Message ?? "Unknown error"}" });
                }

                _logger.LogInformation("✓ Demo connection test successful - API key is valid");
                return ApiKeyValidationResult.Success();
            }

            // LIVE MODE: Perform strict validation with permission checks
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

            // Check if permissions data is null
            if (permissionsResult.Data == null)
            {
                _logger.LogError("API returned success but permissions data is null");
                return ApiKeyValidationResult.Failure(
                    new List<string> { "API returned empty permissions data" });
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

            // Get API Key Information using the official endpoint
            _logger.LogInformation("Calling Bybit V5Api.Account.GetApiKeyInfoAsync()...");
            var apiKeyInfoResult = await client.V5Api.Account.GetApiKeyInfoAsync();
            _logger.LogInformation("API Call completed. Success: {Success}", apiKeyInfoResult.Success);

            if (!apiKeyInfoResult.Success)
            {
                _logger.LogWarning("Bybit API key information check failed");
                _logger.LogWarning("Error Code: {Code}", apiKeyInfoResult.Error?.Code);
                _logger.LogWarning("Error Message: {Message}", apiKeyInfoResult.Error?.Message);

                // Check if it's an IP restriction error
                var isIpError = apiKeyInfoResult.Error != null && (
                    apiKeyInfoResult.Error.Message.Contains("IP", StringComparison.OrdinalIgnoreCase) ||
                    apiKeyInfoResult.Error.Code == 10003); // Bybit IP restriction error code

                if (isIpError)
                {
                    _logger.LogWarning("IP Restriction Detected!");
                    return ApiKeyValidationResult.Failure(
                        new List<string> { "API key is IP-restricted and server IP is not whitelisted" },
                        isIpRestricted: true,
                        allowedIps: null,
                        serverIp: serverIp);
                }

                return ApiKeyValidationResult.Failure(
                    new List<string> { $"API validation failed: {apiKeyInfoResult.Error?.Message ?? "Unknown error"}" });
            }

            // Check if API key info data is null
            if (apiKeyInfoResult.Data == null)
            {
                _logger.LogError("API returned success but key info data is null");
                return ApiKeyValidationResult.Failure(
                    new List<string> { "API returned empty key information data" });
            }

            var keyInfo = apiKeyInfoResult.Data;

            // DEMO MODE: Simple validation
            if (!isLive)
            {
                _logger.LogInformation("Demo mode detected - performing basic validation");
                _logger.LogInformation("✓ Demo connection test successful - API key is valid");
                return ApiKeyValidationResult.Success();
            }

            // LIVE MODE: Perform strict validation
            _logger.LogInformation("--- API Key Information ---");
            _logger.LogInformation("Read Only: {ReadOnly}", keyInfo.Readonly);
            _logger.LogInformation("API Key Type: {Type}", keyInfo.ApiKeyType == 1 ? "Personal" : "Third-party");

            // 1. Check if API key is read-only
            if (keyInfo.Readonly)
            {
                _logger.LogError("API key is READ-ONLY - Write permissions required for trading");
                return ApiKeyValidationResult.Failure(
                    new List<string> { "API key is read-only. Trading requires read-write permissions. Please create a new API key with write permissions enabled." });
            }

            // 2. Check IP Restrictions
            _logger.LogInformation("--- IP Restrictions ---");
            var hasIpRestriction = keyInfo.Ips != null && keyInfo.Ips.Length > 0;
            _logger.LogInformation("IP Restriction Enabled: {Enabled}", hasIpRestriction);

            if (hasIpRestriction)
            {
                _logger.LogInformation("Whitelisted IPs ({Count}):", keyInfo.Ips.Length);
                foreach (var ip in keyInfo.Ips)
                {
                    _logger.LogInformation("  - {IP}", ip);
                }

                // Check if server IP is in the whitelist
                if (!string.IsNullOrEmpty(serverIp))
                {
                    _logger.LogInformation("Server IP: {ServerIp}", serverIp);
                    var isServerIpAllowed = keyInfo.Ips.Contains(serverIp);

                    if (!isServerIpAllowed)
                    {
                        _logger.LogError("Server IP {ServerIp} is NOT in the whitelist", serverIp);
                        return ApiKeyValidationResult.Failure(
                            new List<string> { $"Server IP {serverIp} is not whitelisted. Please add it to your Bybit API key IP whitelist." },
                            isIpRestricted: true,
                            allowedIps: keyInfo.Ips.ToList(),
                            serverIp: serverIp);
                    }

                    _logger.LogInformation("✓ Server IP is whitelisted");
                }
            }
            else
            {
                _logger.LogError("SECURITY ERROR: IP restriction is REQUIRED for live trading but is currently DISABLED");
                return ApiKeyValidationResult.Failure(
                    new List<string> { "IP Restriction REQUIRED - For security, IP restriction must be enabled for live trading. Please enable IP whitelist in your Bybit API key settings." });
            }

            // 3. Validate Permissions
            _logger.LogInformation("--- API Key Permissions ---");
            var permissions = keyInfo.Permissions;
            var missingPermissions = new List<string>();

            // ContractTrade permissions (for futures/perpetuals)
            _logger.LogInformation("[ContractTrade] Permissions: [{Permissions}]", string.Join(", ", permissions.ContractTrade));
            if (!permissions.ContractTrade.Contains("Order"))
            {
                missingPermissions.Add("ContractTrade.Order - Required to place and manage futures orders");
            }
            if (!permissions.ContractTrade.Contains("Position"))
            {
                missingPermissions.Add("ContractTrade.Position - Required to manage futures positions");
            }

            // Spot permissions
            _logger.LogInformation("[Spot] Permissions: [{Permissions}]", string.Join(", ", permissions.Spot));
            if (!permissions.Spot.Contains("SpotTrade"))
            {
                missingPermissions.Add("Spot.SpotTrade - Required for spot trading operations");
            }

            // Wallet permissions
            _logger.LogInformation("[Wallet] Permissions: [{Permissions}]", string.Join(", ", permissions.Wallet));
            if (!permissions.Wallet.Contains("AccountTransfer"))
            {
                missingPermissions.Add("Wallet.AccountTransfer - Required to transfer funds between accounts");
            }

            // Derivatives permissions (for Unified Trading Account)
            _logger.LogInformation("[Derivatives] Permissions: [{Permissions}]", string.Join(", ", permissions.Derivatives));
            if (keyInfo.Uta && !permissions.Derivatives.Contains("DerivativesTrade"))
            {
                missingPermissions.Add("Derivatives.DerivativesTrade - Required for unified account derivatives trading");
            }

            // Log other permissions for reference
            _logger.LogInformation("[Options] Permissions: [{Permissions}]", string.Join(", ", permissions.Options));
            _logger.LogInformation("[Exchange] Permissions: [{Permissions}]", string.Join(", ", permissions.Exchange));
            _logger.LogInformation("[CopyTrading] Permissions: [{Permissions}]", string.Join(", ", permissions.CopyTrading));
            _logger.LogInformation("[BlockTrade] Permissions: [{Permissions}]", string.Join(", ", permissions.BlockTrade));

            // Summary
            _logger.LogInformation("--- Permission Validation Summary ---");
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
            _logger.LogError(ex, "Error validating Bybit API key");
            return ApiKeyValidationResult.Failure(
                new List<string> { $"Validation error: {ex.Message}" });
        }
    }
}
