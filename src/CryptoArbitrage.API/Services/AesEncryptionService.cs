using System.Security.Cryptography;
using System.Text;

namespace CryptoArbitrage.API.Services;

/// <summary>
/// Implements AES-256 encryption service for securing sensitive data like API keys.
/// Encryption key is sourced from environment variable (recommended) or appsettings.json (development).
/// The IV (initialization vector) is prepended to the ciphertext for stateless decryption.
/// </summary>
public class AesEncryptionService : IEncryptionService
{
    private readonly byte[] _key;
    private readonly ILogger<AesEncryptionService> _logger;

    public AesEncryptionService(IConfiguration config, ILogger<AesEncryptionService> logger)
    {
        _logger = logger;

        // Prefer environment variable for production security
        var keyString = Environment.GetEnvironmentVariable("ENCRYPTION_KEY")
                       ?? config["Encryption:Key"];

        if (string.IsNullOrEmpty(keyString))
        {
            throw new InvalidOperationException(
                "Encryption key not configured. Set ENCRYPTION_KEY environment variable or Encryption:Key in appsettings.json");
        }

        // Ensure exactly 32 bytes for AES-256
        _key = Encoding.UTF8.GetBytes(keyString.PadRight(32).Substring(0, 32));
        _logger.LogInformation("AES encryption service initialized successfully");
    }

    /// <summary>
    /// Encrypts plaintext using AES-256-CBC with random IV.
    /// Returns Base64 string with IV prepended (16 bytes) followed by ciphertext.
    /// </summary>
    public string Encrypt(string plainText)
    {
        if (string.IsNullOrEmpty(plainText))
        {
            throw new ArgumentNullException(nameof(plainText), "Cannot encrypt empty or null plaintext");
        }

        try
        {
            using var aes = Aes.Create();
            aes.Key = _key;
            aes.GenerateIV();

            using var encryptor = aes.CreateEncryptor(aes.Key, aes.IV);
            using var ms = new MemoryStream();

            // Prepend IV to output for later decryption
            ms.Write(aes.IV, 0, aes.IV.Length);

            using (var cs = new CryptoStream(ms, encryptor, CryptoStreamMode.Write))
            using (var sw = new StreamWriter(cs))
            {
                sw.Write(plainText);
            }

            return Convert.ToBase64String(ms.ToArray());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error encrypting data");
            throw;
        }
    }

    /// <summary>
    /// Decrypts Base64-encoded ciphertext (with IV prepended).
    /// Extracts IV from first 16 bytes, then decrypts the remaining data.
    /// </summary>
    public string Decrypt(string cipherText)
    {
        if (string.IsNullOrEmpty(cipherText))
        {
            throw new ArgumentNullException(nameof(cipherText), "Cannot decrypt empty or null ciphertext");
        }

        try
        {
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
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error decrypting data");
            throw;
        }
    }
}
