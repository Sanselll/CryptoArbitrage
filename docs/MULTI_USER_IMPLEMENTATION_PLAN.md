# Multi-User Architecture Implementation Plan

## Executive Summary

Transform the single-tenant crypto arbitrage system into a secure multi-user platform with:
- **OAuth2/Google authentication** with email whitelist
- **Token-based authorization** (JWT) - all user identity from token claims, never client input
- **Encrypted per-user exchange API keys** (AES-256, environment-based encryption)
- **Complete user data isolation** - positions, executions, balances, P&L
- **Shared global data** - funding rates (efficient), opportunities detection
- **Profile settings** - API key management interface

## Security Principles

🔒 **NEVER trust client-provided user IDs** - Always extract user identity from authenticated JWT token

🔒 **Server-side authorization on every request** - Validate token claims before any data access

🔒 **SignalR authorization per message** - Verify user owns requested data before broadcasting

🔒 **Database queries always filtered by authenticated user** - No way to access other users' data

🔒 **Resource ownership validation** - Before any update/delete, verify user owns the resource

---

## Phase 1: Backend Authentication & Authorization

### 1.1 Install Required NuGet Packages

```bash
cd src/CryptoArbitrage.API

dotnet add package Microsoft.AspNetCore.Identity.EntityFrameworkCore
dotnet add package Microsoft.AspNetCore.Authentication.JwtBearer
dotnet add package Microsoft.AspNetCore.Authentication.Google
```

### 1.2 Configuration Updates

**File: `src/CryptoArbitrage.API/appsettings.json`**

Add new sections:

```json
{
  "Authentication": {
    "Google": {
      "ClientId": "YOUR_GOOGLE_CLIENT_ID",
      "ClientSecret": "YOUR_GOOGLE_CLIENT_SECRET"
    },
    "AllowedUsers": [
      "user1@gmail.com",
      "user2@gmail.com",
      "admin@yourdomain.com"
    ]
  },
  "Jwt": {
    "SecretKey": "YOUR_256_BIT_SECRET_KEY_MINIMUM_32_CHARACTERS_LONG",
    "Issuer": "CryptoArbitrage",
    "Audience": "CryptoArbitrageClient",
    "ExpirationMinutes": 1440
  },
  "Encryption": {
    "Key": "YOUR_32_CHARACTER_ENCRYPTION_KEY"
  }
}
```

**Environment Variable Override (Recommended for Production):**
```bash
export ENCRYPTION_KEY="your-production-encryption-key-32-chars"
export JWT_SECRET_KEY="your-production-jwt-secret-key-min-32-chars"
```

### 1.3 Database Schema Changes

#### New Entity: ApplicationUser

**File: `src/CryptoArbitrage.API/Data/Entities/ApplicationUser.cs`** (NEW)

```csharp
using Microsoft.AspNetCore.Identity;

namespace CryptoArbitrage.API.Data.Entities;

public class ApplicationUser : IdentityUser
{
    public string? GoogleId { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? LastLoginAt { get; set; }

    // Navigation properties
    public ICollection<UserExchangeApiKey> ExchangeApiKeys { get; set; } = new List<UserExchangeApiKey>();
    public ICollection<Position> Positions { get; set; } = new List<Position>();
    public ICollection<Execution> Executions { get; set; } = new List<Execution>();
    public ICollection<PerformanceMetric> PerformanceMetrics { get; set; } = new List<PerformanceMetric>();
}
```

#### New Entity: UserExchangeApiKey

**File: `src/CryptoArbitrage.API/Data/Entities/UserExchangeApiKey.cs`** (NEW)

```csharp
using System.ComponentModel.DataAnnotations;

namespace CryptoArbitrage.API.Data.Entities;

public class UserExchangeApiKey
{
    public int Id { get; set; }

    [Required]
    public string UserId { get; set; } = string.Empty;
    public ApplicationUser User { get; set; } = null!;

    [Required]
    [MaxLength(50)]
    public string ExchangeName { get; set; } = string.Empty; // "Binance", "Bybit"

    [Required]
    public string EncryptedApiKey { get; set; } = string.Empty;

    [Required]
    public string EncryptedApiSecret { get; set; } = string.Empty;

    public bool IsEnabled { get; set; } = true;
    public bool UseDemoTrading { get; set; } = false;

    public DateTime CreatedAt { get; set; }
    public DateTime? LastTestedAt { get; set; }
    public string? LastTestResult { get; set; }
}
```

#### Modify Existing Entities

Add `UserId` property to:

**1. Position.cs** - Add:
```csharp
[Required]
public string UserId { get; set; } = string.Empty;
public ApplicationUser User { get; set; } = null!;
```

**2. Execution.cs** - Add:
```csharp
[Required]
public string UserId { get; set; } = string.Empty;
public ApplicationUser User { get; set; } = null!;
```

**3. PerformanceMetric.cs** - Add:
```csharp
[Required]
public string UserId { get; set; } = string.Empty;
public ApplicationUser User { get; set; } = null!;
```

**4. FundingRate.cs** - NO CHANGES (global data shared across all users)

### 1.4 Update DbContext

**File: `src/CryptoArbitrage.API/Data/ArbitrageDbContext.cs`** (MODIFY)

```csharp
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Data;

public class ArbitrageDbContext : IdentityDbContext<ApplicationUser>
{
    public ArbitrageDbContext(DbContextOptions<ArbitrageDbContext> options)
        : base(options)
    {
    }

    public DbSet<FundingRate> FundingRates { get; set; }
    public DbSet<Position> Positions { get; set; }
    public DbSet<Execution> Executions { get; set; }
    public DbSet<PerformanceMetric> PerformanceMetrics { get; set; }
    public DbSet<UserExchangeApiKey> UserExchangeApiKeys { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder); // IMPORTANT: Call base for Identity tables

        // Configure relationships
        modelBuilder.Entity<ApplicationUser>()
            .HasMany(u => u.ExchangeApiKeys)
            .WithOne(k => k.User)
            .HasForeignKey(k => k.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<ApplicationUser>()
            .HasMany(u => u.Positions)
            .WithOne(p => p.User)
            .HasForeignKey(p => p.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<ApplicationUser>()
            .HasMany(u => u.Executions)
            .WithOne(e => e.User)
            .HasForeignKey(e => e.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<ApplicationUser>()
            .HasMany(u => u.PerformanceMetrics)
            .WithOne(pm => pm.User)
            .HasForeignKey(pm => pm.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        // Indexes for performance
        modelBuilder.Entity<Position>()
            .HasIndex(p => p.UserId);

        modelBuilder.Entity<Execution>()
            .HasIndex(e => e.UserId);

        modelBuilder.Entity<UserExchangeApiKey>()
            .HasIndex(k => new { k.UserId, k.ExchangeName });

        // Existing configurations...
    }
}
```

### 1.5 Encryption Service

**File: `src/CryptoArbitrage.API/Services/IEncryptionService.cs`** (NEW)

```csharp
namespace CryptoArbitrage.API.Services;

public interface IEncryptionService
{
    string Encrypt(string plainText);
    string Decrypt(string cipherText);
}
```

**File: `src/CryptoArbitrage.API/Services/AesEncryptionService.cs`** (NEW)

```csharp
using System.Security.Cryptography;
using System.Text;

namespace CryptoArbitrage.API.Services;

public class AesEncryptionService : IEncryptionService
{
    private readonly byte[] _key;

    public AesEncryptionService(IConfiguration config)
    {
        var keyString = Environment.GetEnvironmentVariable("ENCRYPTION_KEY")
                       ?? config["Encryption:Key"];

        if (string.IsNullOrEmpty(keyString))
            throw new InvalidOperationException("Encryption key not configured. Set ENCRYPTION_KEY environment variable or Encryption:Key in appsettings.json");

        // Ensure 32 bytes for AES-256
        _key = Encoding.UTF8.GetBytes(keyString.PadRight(32).Substring(0, 32));
    }

    public string Encrypt(string plainText)
    {
        if (string.IsNullOrEmpty(plainText))
            throw new ArgumentNullException(nameof(plainText));

        using var aes = Aes.Create();
        aes.Key = _key;
        aes.GenerateIV();

        using var encryptor = aes.CreateEncryptor(aes.Key, aes.IV);
        using var ms = new MemoryStream();

        // Prepend IV to ciphertext
        ms.Write(aes.IV, 0, aes.IV.Length);

        using (var cs = new CryptoStream(ms, encryptor, CryptoStreamMode.Write))
        using (var sw = new StreamWriter(cs))
        {
            sw.Write(plainText);
        }

        return Convert.ToBase64String(ms.ToArray());
    }

    public string Decrypt(string cipherText)
    {
        if (string.IsNullOrEmpty(cipherText))
            throw new ArgumentNullException(nameof(cipherText));

        var fullCipher = Convert.FromBase64String(cipherText);

        using var aes = Aes.Create();
        aes.Key = _key;

        // Extract IV from first 16 bytes
        var iv = new byte[16];
        Array.Copy(fullCipher, 0, iv, 0, iv.Length);
        aes.IV = iv;

        using var decryptor = aes.CreateDecryptor(aes.Key, aes.IV);
        using var ms = new MemoryStream(fullCipher, iv.Length, fullCipher.Length - iv.Length);
        using var cs = new CryptoStream(ms, decryptor, CryptoStreamMode.Read);
        using var sr = new StreamReader(cs);

        return sr.ReadToEnd();
    }
}
```

### 1.6 Current User Service (Authorization Core)

**File: `src/CryptoArbitrage.API/Services/ICurrentUserService.cs`** (NEW)

```csharp
namespace CryptoArbitrage.API.Services;

public interface ICurrentUserService
{
    string? UserId { get; }
    string? Email { get; }
    bool IsAuthenticated { get; }
    void ValidateUserOwnsResource(string resourceUserId);
}
```

**File: `src/CryptoArbitrage.API/Services/CurrentUserService.cs`** (NEW)

```csharp
using System.Security.Claims;

namespace CryptoArbitrage.API.Services;

public class CurrentUserService : ICurrentUserService
{
    private readonly IHttpContextAccessor _httpContextAccessor;

    public CurrentUserService(IHttpContextAccessor httpContextAccessor)
    {
        _httpContextAccessor = httpContextAccessor;
    }

    public string? UserId => _httpContextAccessor.HttpContext?.User
        ?.FindFirst(ClaimTypes.NameIdentifier)?.Value;

    public string? Email => _httpContextAccessor.HttpContext?.User
        ?.FindFirst(ClaimTypes.Email)?.Value;

    public bool IsAuthenticated => !string.IsNullOrEmpty(UserId);

    public void ValidateUserOwnsResource(string resourceUserId)
    {
        if (string.IsNullOrEmpty(UserId))
            throw new UnauthorizedAccessException("User not authenticated");

        if (UserId != resourceUserId)
            throw new UnauthorizedAccessException("Access denied to resource");
    }
}
```

### 1.7 Update Program.cs

**File: `src/CryptoArbitrage.API/Program.cs`** (MODIFY)

Add after `var builder = WebApplication.CreateBuilder(args);`:

```csharp
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Identity;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using CryptoArbitrage.API.Data.Entities;

// Add Identity
builder.Services.AddIdentity<ApplicationUser, IdentityRole>(options =>
{
    // Password settings (can be relaxed since we're using OAuth)
    options.Password.RequireDigit = false;
    options.Password.RequireLowercase = false;
    options.Password.RequireUppercase = false;
    options.Password.RequireNonAlphanumeric = false;
    options.Password.RequiredLength = 6;

    // User settings
    options.User.RequireUniqueEmail = true;
})
.AddEntityFrameworkStores<ArbitrageDbContext>()
.AddDefaultTokenProviders();

// Add JWT Authentication
var jwtSecretKey = Environment.GetEnvironmentVariable("JWT_SECRET_KEY")
                  ?? builder.Configuration["Jwt:SecretKey"];

if (string.IsNullOrEmpty(jwtSecretKey))
    throw new InvalidOperationException("JWT secret key not configured");

builder.Services.AddAuthentication(options =>
{
    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(options =>
{
    options.SaveToken = true;
    options.RequireHttpsMetadata = false; // Set to true in production
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuer = true,
        ValidateAudience = true,
        ValidateLifetime = true,
        ValidateIssuerSigningKey = true,
        ValidIssuer = builder.Configuration["Jwt:Issuer"],
        ValidAudience = builder.Configuration["Jwt:Audience"],
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecretKey)),
        ClockSkew = TimeSpan.Zero
    };

    // For SignalR - allow token from query string
    options.Events = new JwtBearerEvents
    {
        OnMessageReceived = context =>
        {
            var accessToken = context.Request.Query["access_token"];
            var path = context.HttpContext.Request.Path;

            if (!string.IsNullOrEmpty(accessToken) &&
                path.StartsWithSegments("/hubs"))
            {
                context.Token = accessToken;
            }
            return Task.CompletedTask;
        }
    };
})
.AddGoogle(options =>
{
    options.ClientId = builder.Configuration["Authentication:Google:ClientId"]
        ?? throw new InvalidOperationException("Google ClientId not configured");
    options.ClientSecret = builder.Configuration["Authentication:Google:ClientSecret"]
        ?? throw new InvalidOperationException("Google ClientSecret not configured");
});

// Register custom services
builder.Services.AddHttpContextAccessor();
builder.Services.AddScoped<ICurrentUserService, CurrentUserService>();
builder.Services.AddSingleton<IEncryptionService, AesEncryptionService>();
```

Add BEFORE `app.Run();`:

```csharp
// IMPORTANT: Order matters!
app.UseAuthentication();
app.UseAuthorization();
```

---

## Phase 2: API Controllers

### 2.1 AuthController (NEW)

**File: `src/CryptoArbitrage.API/Controllers/AuthController.cs`** (NEW)

```csharp
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using CryptoArbitrage.API.Data.Entities;
using Google.Apis.Auth;

namespace CryptoArbitrage.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AuthController : ControllerBase
{
    private readonly UserManager<ApplicationUser> _userManager;
    private readonly IConfiguration _config;
    private readonly ILogger<AuthController> _logger;

    public AuthController(
        UserManager<ApplicationUser> userManager,
        IConfiguration config,
        ILogger<AuthController> logger)
    {
        _userManager = userManager;
        _config = config;
        _logger = logger;
    }

    [HttpPost("google-signin")]
    public async Task<IActionResult> GoogleSignIn([FromBody] GoogleSignInRequest request)
    {
        try
        {
            // 1. Validate Google token
            var payload = await GoogleJsonWebSignature.ValidateAsync(request.IdToken, new GoogleJsonWebSignature.ValidationSettings
            {
                Audience = new[] { _config["Authentication:Google:ClientId"] }
            });

            if (payload == null)
                return Unauthorized(new { error = "Invalid Google token" });

            // 2. Check whitelist
            var allowedUsers = _config.GetSection("Authentication:AllowedUsers").Get<string[]>() ?? Array.Empty<string>();
            if (!allowedUsers.Contains(payload.Email, StringComparer.OrdinalIgnoreCase))
            {
                _logger.LogWarning("Login attempt from non-whitelisted email: {Email}", payload.Email);
                return Unauthorized(new { error = "User not authorized" });
            }

            // 3. Find or create user
            var user = await _userManager.FindByEmailAsync(payload.Email);
            if (user == null)
            {
                user = new ApplicationUser
                {
                    UserName = payload.Email,
                    Email = payload.Email,
                    EmailConfirmed = true,
                    GoogleId = payload.Subject,
                    CreatedAt = DateTime.UtcNow
                };

                var createResult = await _userManager.CreateAsync(user);
                if (!createResult.Succeeded)
                {
                    _logger.LogError("Failed to create user: {Errors}", string.Join(", ", createResult.Errors.Select(e => e.Description)));
                    return StatusCode(500, new { error = "Failed to create user" });
                }

                _logger.LogInformation("Created new user: {Email}", user.Email);
            }

            // 4. Update last login
            user.LastLoginAt = DateTime.UtcNow;
            await _userManager.UpdateAsync(user);

            // 5. Generate JWT token
            var token = GenerateJwtToken(user);

            _logger.LogInformation("User logged in: {Email}", user.Email);

            return Ok(new
            {
                token,
                user = new
                {
                    id = user.Id,
                    email = user.Email,
                    createdAt = user.CreatedAt
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during Google sign-in");
            return StatusCode(500, new { error = "Authentication failed" });
        }
    }

    private string GenerateJwtToken(ApplicationUser user)
    {
        var claims = new[]
        {
            new Claim(ClaimTypes.NameIdentifier, user.Id),
            new Claim(ClaimTypes.Email, user.Email!),
            new Claim(JwtRegisteredClaimNames.Sub, user.Id),
            new Claim(JwtRegisteredClaimNames.Email, user.Email!),
            new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString())
        };

        var jwtSecretKey = Environment.GetEnvironmentVariable("JWT_SECRET_KEY")
                          ?? _config["Jwt:SecretKey"];
        var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecretKey!));
        var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);
        var expires = DateTime.UtcNow.AddMinutes(double.Parse(_config["Jwt:ExpirationMinutes"]!));

        var token = new JwtSecurityToken(
            issuer: _config["Jwt:Issuer"],
            audience: _config["Jwt:Audience"],
            claims: claims,
            expires: expires,
            signingCredentials: creds
        );

        return new JwtSecurityTokenHandler().WriteToken(token);
    }
}

public record GoogleSignInRequest(string IdToken);
```

**Install Google Auth Library:**
```bash
dotnet add package Google.Apis.Auth
```

### 2.2 UserController (NEW)

**File: `src/CryptoArbitrage.API/Controllers/UserController.cs`** (NEW)

```csharp
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Services;

namespace CryptoArbitrage.API.Controllers;

[Authorize]
[ApiController]
[Route("api/[controller]")]
public class UserController : ControllerBase
{
    private readonly ArbitrageDbContext _db;
    private readonly ICurrentUserService _currentUser;
    private readonly IEncryptionService _encryption;
    private readonly ILogger<UserController> _logger;

    public UserController(
        ArbitrageDbContext db,
        ICurrentUserService currentUser,
        IEncryptionService encryption,
        ILogger<UserController> logger)
    {
        _db = db;
        _currentUser = currentUser;
        _encryption = encryption;
        _logger = logger;
    }

    [HttpGet("profile")]
    public async Task<IActionResult> GetProfile()
    {
        var user = await _db.Users.FindAsync(_currentUser.UserId);
        if (user == null)
            return NotFound();

        return Ok(new
        {
            id = user.Id,
            email = user.Email,
            createdAt = user.CreatedAt,
            lastLoginAt = user.LastLoginAt
        });
    }

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
                k.UseDemoTrading,
                k.CreatedAt,
                k.LastTestedAt,
                k.LastTestResult
            })
            .ToListAsync();

        return Ok(keys);
    }

    [HttpPost("apikeys")]
    public async Task<IActionResult> AddApiKey([FromBody] AddApiKeyRequest request)
    {
        if (!new[] { "Binance", "Bybit" }.Contains(request.ExchangeName))
            return BadRequest(new { error = "Invalid exchange name" });

        if (string.IsNullOrWhiteSpace(request.ApiKey) || string.IsNullOrWhiteSpace(request.ApiSecret))
            return BadRequest(new { error = "API key and secret are required" });

        try
        {
            var encryptedKey = _encryption.Encrypt(request.ApiKey);
            var encryptedSecret = _encryption.Encrypt(request.ApiSecret);

            var apiKey = new UserExchangeApiKey
            {
                UserId = _currentUser.UserId!,
                ExchangeName = request.ExchangeName,
                EncryptedApiKey = encryptedKey,
                EncryptedApiSecret = encryptedSecret,
                IsEnabled = true,
                UseDemoTrading = request.UseDemoTrading,
                CreatedAt = DateTime.UtcNow
            };

            _db.UserExchangeApiKeys.Add(apiKey);
            await _db.SaveChangesAsync();

            _logger.LogInformation("User {UserId} added API key for {Exchange}", _currentUser.UserId, request.ExchangeName);

            return Ok(new { id = apiKey.Id, message = "API key added successfully" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding API key");
            return StatusCode(500, new { error = "Failed to add API key" });
        }
    }

    [HttpPut("apikeys/{id}")]
    public async Task<IActionResult> UpdateApiKey(int id, [FromBody] UpdateApiKeyRequest request)
    {
        var apiKey = await _db.UserExchangeApiKeys.FindAsync(id);
        if (apiKey == null)
            return NotFound();

        // CRITICAL: Verify user owns this API key
        _currentUser.ValidateUserOwnsResource(apiKey.UserId);

        try
        {
            if (!string.IsNullOrEmpty(request.ApiKey))
                apiKey.EncryptedApiKey = _encryption.Encrypt(request.ApiKey);

            if (!string.IsNullOrEmpty(request.ApiSecret))
                apiKey.EncryptedApiSecret = _encryption.Encrypt(request.ApiSecret);

            if (request.IsEnabled.HasValue)
                apiKey.IsEnabled = request.IsEnabled.Value;

            if (request.UseDemoTrading.HasValue)
                apiKey.UseDemoTrading = request.UseDemoTrading.Value;

            await _db.SaveChangesAsync();

            _logger.LogInformation("User {UserId} updated API key {KeyId}", _currentUser.UserId, id);

            return Ok(new { message = "API key updated successfully" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating API key");
            return StatusCode(500, new { error = "Failed to update API key" });
        }
    }

    [HttpDelete("apikeys/{id}")]
    public async Task<IActionResult> DeleteApiKey(int id)
    {
        var apiKey = await _db.UserExchangeApiKeys.FindAsync(id);
        if (apiKey == null)
            return NotFound();

        // CRITICAL: Verify user owns this API key
        _currentUser.ValidateUserOwnsResource(apiKey.UserId);

        _db.UserExchangeApiKeys.Remove(apiKey);
        await _db.SaveChangesAsync();

        _logger.LogInformation("User {UserId} deleted API key {KeyId} for {Exchange}",
            _currentUser.UserId, id, apiKey.ExchangeName);

        return Ok(new { message = "API key deleted successfully" });
    }

    [HttpPost("apikeys/{id}/test")]
    public async Task<IActionResult> TestApiKey(int id)
    {
        var apiKey = await _db.UserExchangeApiKeys.FindAsync(id);
        if (apiKey == null)
            return NotFound();

        // CRITICAL: Verify user owns this API key
        _currentUser.ValidateUserOwnsResource(apiKey.UserId);

        try
        {
            var key = _encryption.Decrypt(apiKey.EncryptedApiKey);
            var secret = _encryption.Decrypt(apiKey.EncryptedApiSecret);

            // TODO: Implement actual exchange connection test
            // For now, just validate decryption worked
            var success = !string.IsNullOrEmpty(key) && !string.IsNullOrEmpty(secret);

            apiKey.LastTestedAt = DateTime.UtcNow;
            apiKey.LastTestResult = success ? "Success" : "Failed";
            await _db.SaveChangesAsync();

            return Ok(new
            {
                success,
                message = success ? "API key is valid" : "API key test failed",
                testedAt = apiKey.LastTestedAt
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error testing API key");
            apiKey.LastTestedAt = DateTime.UtcNow;
            apiKey.LastTestResult = $"Error: {ex.Message}";
            await _db.SaveChangesAsync();

            return Ok(new
            {
                success = false,
                message = "API key test failed",
                error = ex.Message
            });
        }
    }

    private static string MaskApiKey(string encryptedKey)
    {
        if (string.IsNullOrEmpty(encryptedKey) || encryptedKey.Length < 8)
            return "****";

        return encryptedKey.Substring(0, 4) + "****" + encryptedKey.Substring(encryptedKey.Length - 4);
    }
}

public record AddApiKeyRequest(string ExchangeName, string ApiKey, string ApiSecret, bool UseDemoTrading);
public record UpdateApiKeyRequest(string? ApiKey, string? ApiSecret, bool? IsEnabled, bool? UseDemoTrading);
```

### 2.3 Update OpportunityController

**File: `src/CryptoArbitrage.API/Controllers/OpportunityController.cs`** (MODIFY)

Add at the top of the class:

```csharp
[Authorize] // ADD THIS
[ApiController]
[Route("api/[controller]")]
public class OpportunityController : ControllerBase
{
    private readonly ICurrentUserService _currentUser; // ADD THIS
    private readonly ArbitrageDbContext _db; // ADD THIS

    // Inject ICurrentUserService and ArbitrageDbContext in constructor

    [HttpPost("execute")]
    public async Task<IActionResult> ExecuteOpportunity([FromBody] ExecuteOpportunityRequest request)
    {
        // Get user's API keys - automatically filtered by JWT token
        var userApiKeys = await _db.UserExchangeApiKeys
            .Where(k => k.UserId == _currentUser.UserId && k.IsEnabled)
            .ToListAsync();

        // Validate user has API keys for required exchanges
        var hasLongExchange = userApiKeys.Any(k => k.ExchangeName == request.LongExchange);
        var hasShortExchange = userApiKeys.Any(k => k.ExchangeName == request.ShortExchange);

        if (!hasLongExchange || !hasShortExchange)
            return BadRequest(new { error = "Missing API keys for required exchanges" });

        // Create execution record - tied to authenticated user
        var execution = new Execution
        {
            UserId = _currentUser.UserId!, // From JWT
            Symbol = request.Symbol,
            LongExchange = request.LongExchange,
            ShortExchange = request.ShortExchange,
            State = ExecutionState.Running,
            StartedAt = DateTime.UtcNow
        };

        _db.Executions.Add(execution);
        await _db.SaveChangesAsync();

        // TODO: Trigger actual trade execution with user's API keys

        return Ok(new { executionId = execution.Id });
    }

    [HttpPost("stop/{executionId}")]
    public async Task<IActionResult> StopExecution(int executionId)
    {
        var execution = await _db.Executions.FindAsync(executionId);
        if (execution == null)
            return NotFound();

        // CRITICAL: Verify user owns this execution
        _currentUser.ValidateUserOwnsResource(execution.UserId);

        execution.State = ExecutionState.Stopped;
        execution.StoppedAt = DateTime.UtcNow;
        await _db.SaveChangesAsync();

        return Ok(new { message = "Execution stopped" });
    }
}
```

### 2.4 Update PositionController

**File: `src/CryptoArbitrage.API/Controllers/PositionController.cs`** (MODIFY)

```csharp
[Authorize] // ADD THIS
[ApiController]
[Route("api/[controller]")]
public class PositionController : ControllerBase
{
    private readonly ICurrentUserService _currentUser; // ADD THIS
    private readonly ArbitrageDbContext _db;

    // Inject ICurrentUserService in constructor

    [HttpGet]
    public async Task<IActionResult> GetPositions()
    {
        // Automatically filtered by authenticated user from JWT
        var positions = await _db.Positions
            .Where(p => p.UserId == _currentUser.UserId)
            .ToListAsync();

        return Ok(positions);
    }

    [HttpGet("{id}")]
    public async Task<IActionResult> GetPosition(int id)
    {
        var position = await _db.Positions.FindAsync(id);
        if (position == null)
            return NotFound();

        // CRITICAL: Verify user owns this position
        _currentUser.ValidateUserOwnsResource(position.UserId);

        return Ok(position);
    }
}
```

---

## Phase 3: SignalR Multi-User Isolation

### 3.1 Update ArbitrageHub

**File: `src/CryptoArbitrage.API/Hubs/ArbitrageHub.cs`** (MODIFY)

```csharp
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using CryptoArbitrage.API.Data;

namespace CryptoArbitrage.API.Hubs;

[Authorize] // Require authentication to connect
public class ArbitrageHub : Hub
{
    private readonly ArbitrageDbContext _db;
    private readonly ILogger<ArbitrageHub> _logger;

    public ArbitrageHub(ArbitrageDbContext db, ILogger<ArbitrageHub> logger)
    {
        _db = db;
        _logger = logger;
    }

    public override async Task OnConnectedAsync()
    {
        // Get userId from authenticated JWT token
        var userId = Context.User?.FindFirst(ClaimTypes.NameIdentifier)?.Value;

        if (string.IsNullOrEmpty(userId))
        {
            _logger.LogWarning("Connection attempt without valid user ID");
            Context.Abort();
            return;
        }

        // Add connection to user-specific group
        await Groups.AddToGroupAsync(Context.ConnectionId, $"user_{userId}");

        _logger.LogInformation("User {UserId} connected to SignalR (ConnectionId: {ConnectionId})",
            userId, Context.ConnectionId);

        await base.OnConnectedAsync();
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        var userId = Context.User?.FindFirst(ClaimTypes.NameIdentifier)?.Value;

        if (!string.IsNullOrEmpty(userId))
        {
            await Groups.RemoveFromGroupAsync(Context.ConnectionId, $"user_{userId}");
            _logger.LogInformation("User {UserId} disconnected from SignalR", userId);
        }

        await base.OnDisconnectedAsync(exception);
    }

    // Client can request their positions
    public async Task GetPositions()
    {
        var userId = Context.User?.FindFirst(ClaimTypes.NameIdentifier)?.Value;
        if (string.IsNullOrEmpty(userId))
            return;

        // CRITICAL: Query filtered by authenticated user
        var positions = await _db.Positions
            .Where(p => p.UserId == userId)
            .ToListAsync();

        // Send ONLY to this specific user
        await Clients.Caller.SendAsync("ReceivePositions", positions);
    }

    // Add similar methods for other data types...
}
```

### 3.2 Update ArbitrageEngineService for Multi-User Broadcasting

**File: `src/CryptoArbitrage.API/Services/ArbitrageEngineService.cs`** (MODIFY)

```csharp
public class ArbitrageEngineService : BackgroundService
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IHubContext<ArbitrageHub> _hubContext;
    private readonly IEncryptionService _encryption;
    private readonly ILogger<ArbitrageEngineService> _logger;

    // Constructor...

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                using var scope = _serviceProvider.CreateScope();
                var db = scope.ServiceProvider.GetRequiredService<ArbitrageDbContext>();

                // 1. Fetch global funding rates (shared data)
                var fundingRates = await FetchGlobalFundingRates();

                if (fundingRates.Any())
                {
                    await db.FundingRates.AddRangeAsync(fundingRates);
                    await db.SaveChangesAsync();

                    // Broadcast to all users (global data)
                    await _hubContext.Clients.All.SendAsync("ReceiveFundingRates", fundingRates, stoppingToken);
                }

                // 2. Process each user individually
                var activeUsers = await db.Users
                    .Where(u => u.ExchangeApiKeys.Any(k => k.IsEnabled))
                    .Include(u => u.ExchangeApiKeys.Where(k => k.IsEnabled))
                    .ToListAsync(stoppingToken);

                foreach (var user in activeUsers)
                {
                    await ProcessUserData(user, db, stoppingToken);
                }

                await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in arbitrage engine loop");
                await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
            }
        }
    }

    private async Task ProcessUserData(ApplicationUser user, ArbitrageDbContext db, CancellationToken ct)
    {
        try
        {
            var userId = user.Id;

            // Initialize exchange connectors for THIS user with decrypted keys
            var connectors = new Dictionary<string, IExchangeConnector>();

            foreach (var apiKey in user.ExchangeApiKeys)
            {
                var key = _encryption.Decrypt(apiKey.EncryptedApiKey);
                var secret = _encryption.Decrypt(apiKey.EncryptedApiSecret);

                var connector = CreateConnector(apiKey.ExchangeName, key, secret, apiKey.UseDemoTrading);
                connectors[apiKey.ExchangeName] = connector;
            }

            // Fetch user-specific positions
            var positions = await db.Positions
                .Where(p => p.UserId == userId)
                .ToListAsync(ct);

            // Update positions with live data from exchanges
            foreach (var position in positions)
            {
                if (connectors.TryGetValue(position.Exchange, out var connector))
                {
                    await UpdatePositionFromExchange(position, connector);
                }
            }

            await db.SaveChangesAsync(ct);

            // Broadcast to THIS USER ONLY
            await _hubContext.Clients.Group($"user_{userId}")
                .SendAsync("ReceivePositions", positions.Select(MapToDto), ct);

            // Fetch and broadcast user-specific balances
            var balances = new List<AccountBalanceDto>();
            foreach (var kvp in connectors)
            {
                try
                {
                    var balance = await kvp.Value.GetAccountBalanceAsync();
                    balances.Add(balance);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error fetching balance for user {UserId} from {Exchange}",
                        userId, kvp.Key);
                }
            }

            await _hubContext.Clients.Group($"user_{userId}")
                .SendAsync("ReceiveBalances", balances, ct);

            // Calculate and broadcast user-specific P&L
            var totalPnL = positions.Sum(p => p.UnrealizedPnL + p.RealizedPnL);
            var todayPnL = await CalculateTodayPnL(db, userId);

            await _hubContext.Clients.Group($"user_{userId}")
                .SendAsync("ReceivePnLUpdate", new { totalPnL, todayPnL }, ct);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing data for user {UserId}", user.Id);
        }
    }

    private IExchangeConnector CreateConnector(string exchangeName, string apiKey, string apiSecret, bool demoMode)
    {
        // Create exchange-specific connector with user's credentials
        return exchangeName switch
        {
            "Binance" => new BinanceConnector(apiKey, apiSecret, demoMode),
            "Bybit" => new BybitConnector(apiKey, apiSecret, demoMode),
            _ => throw new NotSupportedException($"Exchange {exchangeName} not supported")
        };
    }

    private async Task<decimal> CalculateTodayPnL(ArbitrageDbContext db, string userId)
    {
        var today = DateTime.UtcNow.Date;

        var metric = await db.PerformanceMetrics
            .Where(m => m.UserId == userId && m.Date == today)
            .FirstOrDefaultAsync();

        return metric?.RealizedPnL ?? 0;
    }

    // Other helper methods...
}
```

---

## Phase 4: Database Migration

### 4.1 Create Migration

```bash
cd src/CryptoArbitrage.API

dotnet ef migrations add AddMultiUserSupport
```

### 4.2 Review Migration

Check the generated migration file in `Data/Migrations/` to ensure:
- AspNetUsers, AspNetRoles, AspNetUserRoles, etc. tables are created
- UserExchangeApiKeys table is created
- UserId columns added to Position, Execution, PerformanceMetric tables
- Foreign key constraints are correct
- Indexes are created

### 4.3 Apply Migration

**IMPORTANT: This will modify your database schema. Backup database first if production data exists.**

```bash
# Backup database (if using SQLite)
cp arbitrage.db arbitrage.db.backup

# Apply migration
dotnet ef database update
```

### 4.4 Data Migration (if needed)

If you have existing data that needs to be assigned to a user:

```sql
-- Create a default admin user first through the app, then:
UPDATE Positions SET UserId = 'admin-user-id-here';
UPDATE Executions SET UserId = 'admin-user-id-here';
UPDATE PerformanceMetrics SET UserId = 'admin-user-id-here';
```

---

## Phase 5: Frontend Implementation

### 5.1 Install Dependencies

```bash
cd client

npm install @microsoft/signalr
npm install axios
npm install zustand
npm install react-router-dom
npm install @react-oauth/google
```

### 5.2 Auth Store

**File: `client/src/stores/authStore.ts`** (NEW)

```typescript
import { create } from 'zustand';

interface User {
  id: string;
  email: string;
  createdAt: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  login: (googleToken: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: localStorage.getItem('jwt_token'),
  isAuthenticated: !!localStorage.getItem('jwt_token'),
  isLoading: false,

  login: async (googleToken: string) => {
    set({ isLoading: true });
    try {
      const response = await fetch('http://localhost:5052/api/auth/google-signin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ idToken: googleToken })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Login failed');
      }

      const { token, user } = await response.json();
      localStorage.setItem('jwt_token', token);
      set({ token, user, isAuthenticated: true, isLoading: false });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  logout: () => {
    localStorage.removeItem('jwt_token');
    set({ user: null, token: null, isAuthenticated: false });
  },

  checkAuth: () => {
    const token = localStorage.getItem('jwt_token');
    set({ isAuthenticated: !!token, token });
  }
}));
```

### 5.3 API Client with Auth Interceptor

**File: `client/src/services/apiClient.ts`** (NEW)

```typescript
import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:5052/api'
});

// CRITICAL: Attach JWT token from localStorage to every request
apiClient.interceptors.request.use(config => {
  const token = localStorage.getItem('jwt_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 Unauthorized - redirect to login
apiClient.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      localStorage.removeItem('jwt_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default apiClient;
```

### 5.4 Update SignalR Service

**File: `client/src/services/signalRService.ts`** (MODIFY)

```typescript
import * as signalR from '@microsoft/signalr';

class SignalRService {
  private connection: signalR.HubConnection | null = null;

  async connect() {
    const token = localStorage.getItem('jwt_token');
    if (!token) {
      throw new Error('No authentication token');
    }

    this.connection = new signalR.HubConnectionBuilder()
      .withUrl('http://localhost:5052/hubs/arbitrage', {
        // CRITICAL: Send JWT token with connection
        accessTokenFactory: () => token
      })
      .withAutomaticReconnect()
      .build();

    try {
      await this.connection.start();
      console.log('SignalR connected');
    } catch (error) {
      console.error('SignalR connection error:', error);
      throw error;
    }
  }

  disconnect() {
    if (this.connection) {
      this.connection.stop();
      this.connection = null;
    }
  }

  onFundingRates(callback: (rates: any[]) => void) {
    this.connection?.on('ReceiveFundingRates', callback);
    return () => this.connection?.off('ReceiveFundingRates', callback);
  }

  onPositions(callback: (positions: any[]) => void) {
    this.connection?.on('ReceivePositions', callback);
    return () => this.connection?.off('ReceivePositions', callback);
  }

  onBalances(callback: (balances: any[]) => void) {
    this.connection?.on('ReceiveBalances', callback);
    return () => this.connection?.off('ReceiveBalances', callback);
  }

  onPnLUpdate(callback: (pnl: any) => void) {
    this.connection?.on('ReceivePnLUpdate', callback);
    return () => this.connection?.off('ReceivePnLUpdate', callback);
  }

  onOpportunities(callback: (opportunities: any[]) => void) {
    this.connection?.on('ReceiveOpportunities', callback);
    return () => this.connection?.off('ReceiveOpportunities', callback);
  }
}

export default new SignalRService();
```

### 5.5 Protected Route Component

**File: `client/src/components/ProtectedRoute.tsx`** (NEW)

```typescript
import { Navigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  const isAuthenticated = useAuthStore(state => state.isAuthenticated);

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};
```

### 5.6 Login Page

**File: `client/src/pages/LoginPage.tsx`** (NEW)

```typescript
import { GoogleLogin } from '@react-oauth/google';
import { useAuthStore } from '../stores/authStore';
import { useNavigate } from 'react-router-dom';

export const LoginPage = () => {
  const login = useAuthStore(state => state.login);
  const navigate = useNavigate();

  const handleGoogleSuccess = async (credentialResponse: any) => {
    try {
      await login(credentialResponse.credential);
      navigate('/');
    } catch (error: any) {
      alert(error.message || 'Login failed');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900">
      <div className="bg-gray-800 p-8 rounded-lg shadow-xl max-w-md w-full">
        <h1 className="text-3xl font-bold text-white mb-6 text-center">
          Crypto Arbitrage Platform
        </h1>
        <p className="text-gray-400 mb-8 text-center">
          Sign in with your authorized Google account
        </p>

        <div className="flex justify-center">
          <GoogleLogin
            onSuccess={handleGoogleSuccess}
            onError={() => alert('Login failed')}
            useOneTap
          />
        </div>
      </div>
    </div>
  );
};
```

### 5.7 Profile Settings Page

**File: `client/src/pages/ProfileSettings.tsx`** (NEW)

```typescript
import { useEffect, useState } from 'react';
import apiClient from '../services/apiClient';

interface ApiKey {
  id: number;
  exchangeName: string;
  apiKey: string;
  isEnabled: boolean;
  useDemoTrading: boolean;
  createdAt: string;
  lastTestedAt?: string;
  lastTestResult?: string;
}

export const ProfileSettings = () => {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [isAdding, setIsAdding] = useState(false);
  const [newKey, setNewKey] = useState({
    exchangeName: 'Binance',
    apiKey: '',
    apiSecret: '',
    useDemoTrading: false
  });

  useEffect(() => {
    loadApiKeys();
  }, []);

  const loadApiKeys = async () => {
    try {
      const response = await apiClient.get('/user/apikeys');
      setApiKeys(response.data);
    } catch (error) {
      console.error('Error loading API keys:', error);
    }
  };

  const addApiKey = async () => {
    try {
      await apiClient.post('/user/apikeys', newKey);
      setNewKey({ exchangeName: 'Binance', apiKey: '', apiSecret: '', useDemoTrading: false });
      setIsAdding(false);
      loadApiKeys();
    } catch (error: any) {
      alert(error.response?.data?.error || 'Failed to add API key');
    }
  };

  const deleteApiKey = async (id: number) => {
    if (!confirm('Are you sure you want to delete this API key?')) return;

    try {
      await apiClient.delete(`/user/apikeys/${id}`);
      loadApiKeys();
    } catch (error) {
      alert('Failed to delete API key');
    }
  };

  const testApiKey = async (id: number) => {
    try {
      const response = await apiClient.post(`/user/apikeys/${id}/test`);
      alert(response.data.message);
      loadApiKeys();
    } catch (error) {
      alert('Failed to test API key');
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-white mb-8">Profile Settings</h1>

      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-white">Exchange API Keys</h2>
          <button
            onClick={() => setIsAdding(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
          >
            Add API Key
          </button>
        </div>

        {isAdding && (
          <div className="bg-gray-700 p-4 rounded mb-4">
            <h3 className="text-white font-semibold mb-3">Add New API Key</h3>
            <div className="space-y-3">
              <div>
                <label className="text-gray-300 block mb-1">Exchange</label>
                <select
                  value={newKey.exchangeName}
                  onChange={e => setNewKey({...newKey, exchangeName: e.target.value})}
                  className="w-full bg-gray-600 text-white rounded px-3 py-2"
                >
                  <option>Binance</option>
                  <option>Bybit</option>
                </select>
              </div>
              <div>
                <label className="text-gray-300 block mb-1">API Key</label>
                <input
                  type="text"
                  value={newKey.apiKey}
                  onChange={e => setNewKey({...newKey, apiKey: e.target.value})}
                  className="w-full bg-gray-600 text-white rounded px-3 py-2"
                  placeholder="Enter API key"
                />
              </div>
              <div>
                <label className="text-gray-300 block mb-1">API Secret</label>
                <input
                  type="password"
                  value={newKey.apiSecret}
                  onChange={e => setNewKey({...newKey, apiSecret: e.target.value})}
                  className="w-full bg-gray-600 text-white rounded px-3 py-2"
                  placeholder="Enter API secret"
                />
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  checked={newKey.useDemoTrading}
                  onChange={e => setNewKey({...newKey, useDemoTrading: e.target.checked})}
                  className="mr-2"
                />
                <label className="text-gray-300">Use Demo Trading (Testnet)</label>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={addApiKey}
                  className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded"
                >
                  Save
                </button>
                <button
                  onClick={() => setIsAdding(false)}
                  className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="space-y-3">
          {apiKeys.map(key => (
            <div key={key.id} className="bg-gray-700 p-4 rounded flex justify-between items-center">
              <div>
                <div className="text-white font-semibold">{key.exchangeName}</div>
                <div className="text-gray-400 text-sm">API Key: {key.apiKey}</div>
                <div className="text-gray-400 text-sm">
                  Status: {key.isEnabled ? 'Enabled' : 'Disabled'}
                  {key.useDemoTrading && ' (Demo)'}
                </div>
                {key.lastTestResult && (
                  <div className="text-gray-400 text-sm">
                    Last Test: {key.lastTestResult} ({new Date(key.lastTestedAt!).toLocaleString()})
                  </div>
                )}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => testApiKey(key.id)}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm"
                >
                  Test
                </button>
                <button
                  onClick={() => deleteApiKey(key.id)}
                  className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}

          {apiKeys.length === 0 && !isAdding && (
            <div className="text-gray-400 text-center py-8">
              No API keys configured. Add one to get started.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
```

### 5.8 Update App.tsx

**File: `client/src/App.tsx`** (MODIFY)

```typescript
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { LoginPage } from './pages/LoginPage';
import { ProfileSettings } from './pages/ProfileSettings';
import { Dashboard } from './pages/Dashboard'; // Your existing dashboard
import { ProtectedRoute } from './components/ProtectedRoute';
import { useAuthStore } from './stores/authStore';
import { useEffect } from 'react';

function App() {
  const checkAuth = useAuthStore(state => state.checkAuth);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  return (
    <GoogleOAuthProvider clientId="YOUR_GOOGLE_CLIENT_ID">
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } />
          <Route path="/profile" element={
            <ProtectedRoute>
              <ProfileSettings />
            </ProtectedRoute>
          } />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </GoogleOAuthProvider>
  );
}

export default App;
```

### 5.9 Update Arbitrage Store

**File: `client/src/stores/arbitrageStore.ts`** (MODIFY)

Update SignalR connection logic to disconnect on logout:

```typescript
import signalRService from '../services/signalRService';
import { useAuthStore } from './authStore';

// In your store, add connection management:
export const useArbitrageStore = create<ArbitrageState>((set) => ({
  // ... existing state

  connect: async () => {
    const isAuthenticated = useAuthStore.getState().isAuthenticated;
    if (!isAuthenticated) {
      console.warn('Cannot connect to SignalR - not authenticated');
      return;
    }

    try {
      await signalRService.connect();
      set({ isConnected: true });

      // Set up listeners
      signalRService.onPositions((positions) => set({ positions }));
      signalRService.onBalances((balances) => set({ balances }));
      signalRService.onFundingRates((rates) => set({ fundingRates: rates }));
      // ... other listeners
    } catch (error) {
      console.error('SignalR connection failed:', error);
      set({ isConnected: false });
    }
  },

  disconnect: () => {
    signalRService.disconnect();
    set({ isConnected: false });
  }
}));
```

---

## Phase 6: Testing & Validation

### 6.1 Security Testing Checklist

- [ ] Attempt to access protected endpoints without JWT token (should return 401)
- [ ] Attempt to access another user's positions via API (should return 403/404)
- [ ] Attempt to modify another user's API keys (should return 403)
- [ ] Attempt to connect to SignalR without JWT token (should fail)
- [ ] Verify SignalR broadcasts only to correct user
- [ ] Verify non-whitelisted email cannot log in
- [ ] Verify API keys are encrypted in database
- [ ] Verify API keys are never logged in plaintext
- [ ] Verify JWT token expires after configured time
- [ ] Verify frontend redirects to login on 401

### 6.2 Functional Testing

1. **User Registration**
   - Log in with whitelisted Google account
   - Verify JWT token is stored in localStorage
   - Verify user record created in database

2. **API Key Management**
   - Add Binance API key
   - Add Bybit API key
   - Test API key connection
   - Enable/disable API key
   - Delete API key

3. **Multi-User Isolation**
   - Create two test users (User A and User B)
   - User A adds API keys and creates positions
   - User B logs in and should see EMPTY positions list
   - Verify SignalR broadcasts don't cross users

4. **Background Service**
   - Verify funding rates fetched globally
   - Verify positions updated for each user independently
   - Verify balances fetched using correct user API keys

### 6.3 Performance Testing

- Monitor database query performance with multiple users
- Verify SignalR group broadcasting scales
- Monitor memory usage with multiple user connectors

---

## Phase 7: Deployment & Production Setup

### 7.1 Environment Variables (Production)

```bash
export ENCRYPTION_KEY="your-strong-32-character-production-key"
export JWT_SECRET_KEY="your-strong-jwt-secret-minimum-32-chars"
export GOOGLE_CLIENT_ID="your-google-client-id"
export GOOGLE_CLIENT_SECRET="your-google-client-secret"
```

### 7.2 Production Configuration Updates

**appsettings.Production.json:**

```json
{
  "Jwt": {
    "RequireHttpsMetadata": true
  },
  "Cors": {
    "AllowedOrigins": ["https://yourdomain.com"]
  }
}
```

### 7.3 Security Hardening

- Enable HTTPS only
- Set secure cookie flags
- Enable CORS for production domain only
- Implement rate limiting on login endpoint
- Add request logging for audit trail
- Set up monitoring/alerts for failed auth attempts

---

## Implementation Order

1. ✅ Backend: Database entities + DbContext updates
2. ✅ Backend: Install NuGet packages
3. ✅ Backend: Encryption service + CurrentUserService
4. ✅ Backend: Update Program.cs with Identity + JWT + Google OAuth
5. ✅ Backend: AuthController + UserController
6. ✅ Backend: Update OpportunityController + PositionController
7. ✅ Backend: Update ArbitrageHub for user groups
8. ✅ Backend: Update ArbitrageEngineService for multi-user
9. ✅ Backend: Create and apply EF migration
10. ✅ Frontend: Install npm packages
11. ✅ Frontend: Auth store + API client
12. ✅ Frontend: Update SignalR service
13. ✅ Frontend: Login page + Protected routes
14. ✅ Frontend: Profile settings page
15. ✅ Frontend: Update App.tsx routing
16. ✅ Testing: Full security & functional testing
17. ✅ Production: Environment setup & deployment

---

## Key Security Features Summary

✅ **Token-based authorization**: User identity extracted from JWT claims, never client input

✅ **Automatic query filtering**: All database queries use `_currentUser.UserId` from token

✅ **Resource ownership validation**: Every update/delete validates user owns the resource

✅ **Encrypted API keys**: AES-256 encryption with environment-based key

✅ **SignalR user isolation**: User-specific groups, filtered broadcasts

✅ **Whitelist access control**: Only pre-approved emails can log in

✅ **No trust in client**: Server validates everything from authenticated token claims

✅ **Audit logging**: All user actions logged with userId from token

---

## Troubleshooting

### Issue: "JWT token validation failed"
**Solution**: Verify JWT secret key is same in appsettings and token generation

### Issue: "Google authentication fails"
**Solution**: Check Google OAuth credentials, callback URL configuration

### Issue: "SignalR connection unauthorized"
**Solution**: Verify JWT token is passed in `accessTokenFactory` on frontend

### Issue: "User sees other users' data"
**Solution**: Check all queries include `.Where(x => x.UserId == _currentUser.UserId)`

### Issue: "API keys not decrypting"
**Solution**: Verify ENCRYPTION_KEY environment variable is set correctly

---

## Next Steps (Future Enhancements)

- Add email/SMS notifications for opportunities
- Add per-user trading preferences (symbols, leverage, position size limits)
- Add per-user risk management settings
- Add user dashboard with custom watchlists
- Add social features (leaderboard, shared strategies - with opt-in)
- Add two-factor authentication (2FA)
- Add API key rotation mechanism
- Add comprehensive audit log viewer
