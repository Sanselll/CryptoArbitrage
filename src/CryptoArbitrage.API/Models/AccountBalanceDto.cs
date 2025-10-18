namespace CryptoArbitrage.API.Models;

/// <summary>
/// Account balance information from an exchange.
/// Note: Field interpretations vary based on exchange architecture.
/// </summary>
public class AccountBalanceDto
{
    public string Exchange { get; set; } = string.Empty;

    /// <summary>
    /// Total account value in USD.
    /// Binance (Separated): Spot wallet + Futures wallet
    /// Bybit (Unified): USDT balance + value of non-USDT assets
    /// </summary>
    public decimal TotalBalance { get; set; }

    /// <summary>
    /// Available balance (not locked in positions or orders).
    /// Binance: Spot available + Futures available
    /// Bybit: Available USDT + value of coins in positions
    /// </summary>
    public decimal AvailableBalance { get; set; }

    /// <summary>
    /// Operational balance used for trading calculations.
    /// Binance: USDT + coins in positions + futures balance
    /// Bybit: Same as TotalBalance (unified account)
    /// </summary>
    public decimal OperationalBalanceUsd { get; set; }

    /// <summary>
    /// Spot wallet balance in USD equivalent.
    /// Binance: Full spot wallet (USDT + all coins)
    /// Bybit: Only non-USDT assets (coins held in positions)
    /// </summary>
    public decimal SpotBalanceUsd { get; set; }

    /// <summary>
    /// Available spot balance (not locked).
    /// Binance: Available portion of spot wallet
    /// Bybit: Same as SpotBalanceUsd (non-USDT assets)
    /// </summary>
    public decimal SpotAvailableUsd { get; set; }

    /// <summary>
    /// Individual asset balances in spot wallet.
    /// Key: Asset symbol (e.g., "BTC", "ETH", "USDT")
    /// Value: Quantity of that asset
    /// </summary>
    public Dictionary<string, decimal> SpotAssets { get; set; } = new Dictionary<string, decimal>();

    /// <summary>
    /// Futures/derivatives balance in USD.
    /// Binance: Futures wallet balance (separate from spot)
    /// Bybit: USDT balance that backs all positions (unified account)
    /// </summary>
    public decimal FuturesBalanceUsd { get; set; }

    /// <summary>
    /// Available futures balance (not used as margin).
    /// Binance: Available portion of futures wallet
    /// Bybit: USDT available to withdraw from unified account
    /// </summary>
    public decimal FuturesAvailableUsd { get; set; }

    /// <summary>
    /// Margin locked in open positions.
    /// Binance: (WalletBalance - AvailableBalance) in futures wallet
    /// Bybit: (WalletBalance - AvailableToWithdraw) in unified account
    /// </summary>
    public decimal MarginUsed { get; set; }

    /// <summary>
    /// Unrealized profit/loss from open positions.
    /// </summary>
    public decimal UnrealizedPnL { get; set; }

    public DateTime UpdatedAt { get; set; }
}
