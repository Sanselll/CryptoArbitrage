namespace CryptoArbitrage.API.Services.Authentication;

/// <summary>
/// Service for encrypting and decrypting sensitive data (e.g., API keys).
/// Implementations should handle the encryption/decryption logic securely.
/// </summary>
public interface IEncryptionService
{
    /// <summary>
    /// Encrypts the provided plaintext string.
    /// </summary>
    /// <param name="plainText">The text to encrypt</param>
    /// <returns>Base64-encoded encrypted string</returns>
    string Encrypt(string plainText);

    /// <summary>
    /// Decrypts the provided ciphertext string.
    /// </summary>
    /// <param name="cipherText">The Base64-encoded ciphertext to decrypt</param>
    /// <returns>Decrypted plaintext string</returns>
    string Decrypt(string cipherText);
}
