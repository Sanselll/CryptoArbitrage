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

    /// <summary>
    /// Authenticates a user via Google OAuth token.
    /// Validates the token, checks email whitelist, creates/updates user, and returns JWT token.
    /// </summary>
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
            {
                _logger.LogWarning("Invalid or expired Google token received");
                return Unauthorized(new { error = "Invalid Google token" });
            }

            // 2. Check email whitelist
            var allowedUsers = _config.GetSection("Authentication:AllowedUsers").Get<string[]>() ?? Array.Empty<string>();
            if (!allowedUsers.Contains(payload.Email, StringComparer.OrdinalIgnoreCase))
            {
                _logger.LogWarning("Login attempt from non-whitelisted email: {Email}", payload.Email);
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
                    _logger.LogError("Failed to create user {Email}: {Errors}", payload.Email, errors);
                    return StatusCode(500, new { error = "Failed to create user account" });
                }

                _logger.LogInformation("Created new user account: {Email} (GoogleId: {GoogleId})", payload.Email, payload.Subject);
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

            _logger.LogInformation("User successfully authenticated: {Email} (UserId: {UserId})", user.Email, user.Id);

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
        catch (InvalidOperationException ex)
        {
            _logger.LogError(ex, "Configuration error during Google sign-in");
            return StatusCode(500, new { error = "Server configuration error" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error during Google sign-in");
            return StatusCode(500, new { error = "Authentication failed" });
        }
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
