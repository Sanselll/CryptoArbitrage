using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using CryptoArbitrage.API.Data.Entities;
using Google.Apis.Auth;

namespace CryptoArbitrage.API.Controllers;

/// <summary>
/// Handles authentication operations, specifically Google OAuth sign-in.
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class AuthController : BaseController
{
    private readonly UserManager<ApplicationUser> _userManager;
    private readonly IConfiguration _config;

    public AuthController(
        UserManager<ApplicationUser> userManager,
        IConfiguration config,
        ILogger<AuthController> logger)
        : base(logger)
    {
        _userManager = userManager;
        _config = config;
    }

    /// <summary>
    /// Authenticates a user via Google OAuth token.
    /// Validates the token, checks email whitelist, creates/updates user, and returns JWT token.
    /// </summary>
    [HttpPost("google-signin")]
    public async Task<IActionResult> GoogleSignIn([FromBody] GoogleSignInRequest request)
    {
        return await ExecuteActionAsync(async () =>
        {
            // 1. Validate Google token
            var payload = await GoogleJsonWebSignature.ValidateAsync(request.IdToken, new GoogleJsonWebSignature.ValidationSettings
            {
                Audience = new[] { _config["Authentication:Google:ClientId"] }
            });

            if (payload == null)
            {
                Logger.LogWarning("Invalid or expired Google token received");
                return Unauthorized(new { error = "Invalid Google token" });
            }

            // 2. Check email whitelist
            var allowedUsers = _config.GetSection("Authentication:AllowedUsers").Get<string[]>() ?? Array.Empty<string>();
            if (!allowedUsers.Contains(payload.Email, StringComparer.OrdinalIgnoreCase))
            {
                Logger.LogWarning("Login attempt from non-whitelisted email: {Email}", payload.Email);
                return Unauthorized(new { error = "User not authorized. Email not in whitelist." });
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
                    var errors = string.Join(", ", createResult.Errors.Select(e => e.Description));
                    Logger.LogError("Failed to create user {Email}: {Errors}", payload.Email, errors);
                    return StatusCode(500, new { error = "Failed to create user account" });
                }

                Logger.LogInformation("Created new user account: {Email} (GoogleId: {GoogleId})", payload.Email, payload.Subject);
            }
            else
            {
                // Update GoogleId if not already set
                if (string.IsNullOrEmpty(user.GoogleId))
                {
                    user.GoogleId = payload.Subject;
                    await _userManager.UpdateAsync(user);
                }
            }

            // 4. Update last login
            user.LastLoginAt = DateTime.UtcNow;
            await _userManager.UpdateAsync(user);

            // 5. Generate JWT token
            var token = GenerateJwtToken(user);

            Logger.LogInformation("User successfully authenticated: {Email} (UserId: {UserId})", user.Email, user.Id);

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
        }, "Google sign-in");
    }

    /// <summary>
    /// Dev sign-in for whitelisted users using email + password.
    /// Temporary bypass for Google OAuth.
    /// </summary>
    [HttpPost("dev-signin")]
    public async Task<IActionResult> DevSignIn([FromBody] DevSignInRequest request)
    {
        return await ExecuteActionAsync(async () =>
        {
            // 1. Validate dev password
            var devPassword = _config["Authentication:DevPassword"];
            if (string.IsNullOrEmpty(devPassword) || request.Password != devPassword)
            {
                Logger.LogWarning("Invalid dev password attempt for: {Email}", request.Email);
                return Unauthorized(new { error = "Invalid credentials" });
            }

            // 2. Check email whitelist
            var allowedUsers = _config.GetSection("Authentication:AllowedUsers").Get<string[]>() ?? Array.Empty<string>();
            if (!allowedUsers.Contains(request.Email, StringComparer.OrdinalIgnoreCase))
            {
                Logger.LogWarning("Dev login attempt from non-whitelisted email: {Email}", request.Email);
                return Unauthorized(new { error = "User not authorized. Email not in whitelist." });
            }

            // 3. Find or create user (same as Google flow)
            var user = await _userManager.FindByEmailAsync(request.Email);
            if (user == null)
            {
                user = new ApplicationUser
                {
                    UserName = request.Email,
                    Email = request.Email,
                    EmailConfirmed = true,
                    CreatedAt = DateTime.UtcNow
                };

                var createResult = await _userManager.CreateAsync(user);
                if (!createResult.Succeeded)
                {
                    var errors = string.Join(", ", createResult.Errors.Select(e => e.Description));
                    Logger.LogError("Failed to create user {Email}: {Errors}", request.Email, errors);
                    return StatusCode(500, new { error = "Failed to create user account" });
                }

                Logger.LogInformation("Created new user account via dev auth: {Email}", request.Email);
            }

            // 4. Update last login
            user.LastLoginAt = DateTime.UtcNow;
            await _userManager.UpdateAsync(user);

            // 5. Generate JWT token
            var token = GenerateJwtToken(user);

            Logger.LogInformation("User authenticated via dev auth: {Email} (UserId: {UserId})", user.Email, user.Id);

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
        }, "Dev sign-in");
    }

    /// <summary>
    /// Generates a JWT token for the authenticated user.
    /// Token includes standard claims (NameIdentifier, Email, Sub, Jti) and expiration.
    /// </summary>
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

        if (string.IsNullOrEmpty(jwtSecretKey))
            throw new InvalidOperationException("JWT secret key not configured");

        var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecretKey));
        var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

        var expirationMinutes = double.Parse(_config["Jwt:ExpirationMinutes"] ?? "1440");
        var expires = DateTime.UtcNow.AddMinutes(expirationMinutes);

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

/// <summary>
/// Request model for Google sign-in endpoint.
/// </summary>
public record GoogleSignInRequest(string IdToken);

/// <summary>
/// Request model for dev sign-in endpoint.
/// </summary>
public record DevSignInRequest(string Email, string Password);
