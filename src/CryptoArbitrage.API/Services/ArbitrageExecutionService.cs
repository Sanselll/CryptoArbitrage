using CryptoArbitrage.API.Config;
using CryptoArbitrage.API.Data;
using CryptoArbitrage.API.Data.Entities;
using CryptoArbitrage.API.Models;
using Microsoft.EntityFrameworkCore;

namespace CryptoArbitrage.API.Services;

public class ArbitrageExecutionService
{
    private readonly ILogger<ArbitrageExecutionService> _logger;
    private readonly ArbitrageDbContext _dbContext;
    private readonly ArbitrageConfig _config;
    private readonly Dictionary<string, IExchangeConnector> _exchangeConnectors;

    public ArbitrageExecutionService(
        ILogger<ArbitrageExecutionService> _logger,
        ArbitrageDbContext dbContext,
        ArbitrageConfig config,
        BinanceConnector binanceConnector,
        BybitConnector bybitConnector)
    {
        this._logger = _logger;
        _dbContext = dbContext;
        _config = config;

        _exchangeConnectors = new Dictionary<string, IExchangeConnector>
        {
            ["Binance"] = binanceConnector,
            ["Bybit"] = bybitConnector
        };
    }

    public async Task<ExecuteOpportunityResponse> ExecuteOpportunityAsync(ExecuteOpportunityRequest request)
    {
        try
        {
            // Validate request
            var validationError = await ValidateRequestAsync(request);
            if (validationError != null)
            {
                return new ExecuteOpportunityResponse
                {
                    Success = false,
                    ErrorMessage = validationError
                };
            }

            // Execute based on strategy
            if (request.Strategy == ArbitrageStrategy.SpotPerpetual)
            {
                return await ExecuteSpotPerpetualAsync(request);
            }
            else if (request.Strategy == ArbitrageStrategy.CrossExchange)
            {
                return await ExecuteCrossExchangeAsync(request);
            }
            else
            {
                return new ExecuteOpportunityResponse
                {
                    Success = false,
                    ErrorMessage = "Unknown strategy type"
                };
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing arbitrage opportunity");
            return new ExecuteOpportunityResponse
            {
                Success = false,
                ErrorMessage = $"Execution failed: {ex.Message}"
            };
        }
    }

    private async Task<string?> ValidateRequestAsync(ExecuteOpportunityRequest request)
    {
        // Check position size limits
        if (request.PositionSizeUsd < _config.MinPositionSizeUsd)
        {
            return $"Position size ${request.PositionSizeUsd:N2} is below minimum ${_config.MinPositionSizeUsd:N2}";
        }

        if (request.PositionSizeUsd > _config.MaxPositionSizeUsd)
        {
            return $"Position size ${request.PositionSizeUsd:N2} exceeds maximum ${_config.MaxPositionSizeUsd:N2}";
        }

        // Check leverage limits
        if (request.Leverage < 1 || request.Leverage > _config.MaxLeverage)
        {
            return $"Leverage {request.Leverage}x must be between 1x and {_config.MaxLeverage}x";
        }

        // Check total exposure (from running executions)
        var currentExposure = _dbContext.Executions
            .Where(e => e.State == ExecutionState.Running)
            .Sum(e => e.PositionSizeUsd);

        if (currentExposure + request.PositionSizeUsd > _config.MaxTotalExposure)
        {
            return $"Total exposure ${currentExposure + request.PositionSizeUsd:N2} would exceed maximum ${_config.MaxTotalExposure:N2}";
        }

        // Validate exchange connector exists
        if (request.Strategy == ArbitrageStrategy.SpotPerpetual)
        {
            if (!_exchangeConnectors.ContainsKey(request.Exchange))
            {
                return $"Exchange '{request.Exchange}' is not supported";
            }

            // Validate funding rate direction - only positive funding is profitable with USDT-only capital
            // Positive funding = shorts pay longs -> BUY spot + SHORT perp -> collect funding
            // Negative funding would require SELLING spot (which we can't do without owning the coins)
            if (request.FundingRate <= 0)
            {
                return $"Cannot execute negative funding rate ({request.FundingRate * 100:F4}%). Spot-perpetual strategy requires positive funding rates (requires spot BUY + perp SHORT).";
            }

            // Validate balance availability
            var balanceValidation = await ValidateBalanceForSpotPerpetualAsync(request);
            if (!balanceValidation.IsValid)
            {
                return string.Join("; ", balanceValidation.Errors);
            }
        }
        else if (request.Strategy == ArbitrageStrategy.CrossExchange)
        {
            if (string.IsNullOrEmpty(request.LongExchange) || string.IsNullOrEmpty(request.ShortExchange))
            {
                return "Long and short exchanges must be specified for cross-exchange arbitrage";
            }

            if (!_exchangeConnectors.ContainsKey(request.LongExchange))
            {
                return $"Long exchange '{request.LongExchange}' is not supported";
            }

            if (!_exchangeConnectors.ContainsKey(request.ShortExchange))
            {
                return $"Short exchange '{request.ShortExchange}' is not supported";
            }

            // Validate balance availability on both exchanges
            var balanceValidation = await ValidateBalanceForCrossExchangeAsync(request);
            if (!balanceValidation.IsValid)
            {
                return string.Join("; ", balanceValidation.Errors);
            }
        }

        return null;
    }

    /// <summary>
    /// Validate balance for spot-perpetual execution on a single exchange
    /// </summary>
    private async Task<BalanceValidationResult> ValidateBalanceForSpotPerpetualAsync(ExecuteOpportunityRequest request)
    {
        var result = new BalanceValidationResult
        {
            Exchange = request.Exchange,
            RequiredSpotUsdt = request.PositionSizeUsd,
            RequiredFuturesMargin = request.PositionSizeUsd / request.Leverage,
            TotalRequired = request.PositionSizeUsd + (request.PositionSizeUsd / request.Leverage)
        };

        try
        {
            var connector = _exchangeConnectors[request.Exchange];
            await EnsureConnectedAsync(request.Exchange, connector);

            var balance = await connector.GetAccountBalanceAsync();
            result.IsUnifiedAccount = request.Exchange == "Bybit"; // Bybit uses unified account

            if (result.IsUnifiedAccount)
            {
                // Bybit: Check unified account balance
                result.AvailableSpotUsdt = balance.FuturesAvailableUsd;
                result.AvailableFuturesMargin = balance.FuturesAvailableUsd;

                if (balance.FuturesAvailableUsd < result.TotalRequired)
                {
                    result.Errors.Add(
                        $"Insufficient balance on {request.Exchange}. " +
                        $"Required: ${result.TotalRequired:N2} (${result.RequiredSpotUsdt:N2} spot + ${result.RequiredFuturesMargin:N2} margin), " +
                        $"Available: ${balance.FuturesAvailableUsd:N2}"
                    );
                }
                else
                {
                    result.IsValid = true;
                }
            }
            else
            {
                // Binance: Check separate spot and futures balances
                result.AvailableSpotUsdt = balance.SpotAvailableUsd;
                result.AvailableFuturesMargin = balance.FuturesAvailableUsd;

                // Check spot USDT balance
                if (balance.SpotAvailableUsd < result.RequiredSpotUsdt)
                {
                    result.Errors.Add(
                        $"Insufficient spot USDT on {request.Exchange}. " +
                        $"Required: ${result.RequiredSpotUsdt:N2}, Available: ${balance.SpotAvailableUsd:N2}"
                    );
                }

                // Check futures margin balance
                if (balance.FuturesAvailableUsd < result.RequiredFuturesMargin)
                {
                    result.Errors.Add(
                        $"Insufficient futures margin on {request.Exchange}. " +
                        $"Required: ${result.RequiredFuturesMargin:N2}, Available: ${balance.FuturesAvailableUsd:N2}"
                    );
                }

                result.IsValid = result.Errors.Count == 0;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating balance for {Exchange}", request.Exchange);
            result.Errors.Add($"Failed to check balance on {request.Exchange}: {ex.Message}");
        }

        return result;
    }

    /// <summary>
    /// Validate balance for cross-exchange execution on multiple exchanges
    /// </summary>
    private async Task<BalanceValidationResult> ValidateBalanceForCrossExchangeAsync(ExecuteOpportunityRequest request)
    {
        var result = new BalanceValidationResult
        {
            LongExchange = request.LongExchange,
            ShortExchange = request.ShortExchange,
            RequiredFuturesMargin = request.PositionSizeUsd / request.Leverage,
            TotalRequired = (request.PositionSizeUsd / request.Leverage) * 2 // Both exchanges need margin
        };

        try
        {
            // Check long exchange balance
            var longConnector = _exchangeConnectors[request.LongExchange!];
            await EnsureConnectedAsync(request.LongExchange!, longConnector);
            var longBalance = await longConnector.GetAccountBalanceAsync();

            if (longBalance.FuturesAvailableUsd < result.RequiredFuturesMargin)
            {
                result.Errors.Add(
                    $"Insufficient futures margin on {request.LongExchange}. " +
                    $"Required: ${result.RequiredFuturesMargin:N2}, Available: ${longBalance.FuturesAvailableUsd:N2}"
                );
            }

            // Check short exchange balance
            var shortConnector = _exchangeConnectors[request.ShortExchange!];
            await EnsureConnectedAsync(request.ShortExchange!, shortConnector);
            var shortBalance = await shortConnector.GetAccountBalanceAsync();

            if (shortBalance.FuturesAvailableUsd < result.RequiredFuturesMargin)
            {
                result.Errors.Add(
                    $"Insufficient futures margin on {request.ShortExchange}. " +
                    $"Required: ${result.RequiredFuturesMargin:N2}, Available: ${shortBalance.FuturesAvailableUsd:N2}"
                );
            }

            result.IsValid = result.Errors.Count == 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating balance for cross-exchange");
            result.Errors.Add($"Failed to check balance: {ex.Message}");
        }

        return result;
    }

    /// <summary>
    /// Get execution balances for UI display
    /// </summary>
    public async Task<ExecutionBalancesDto> GetExecutionBalancesAsync(string exchange, decimal maxLeverage)
    {
        var result = new ExecutionBalancesDto
        {
            Exchange = exchange,
            IsUnifiedAccount = exchange == "Bybit"
        };

        try
        {
            var connector = _exchangeConnectors[exchange];
            await EnsureConnectedAsync(exchange, connector);

            var balance = await connector.GetAccountBalanceAsync();

            result.SpotUsdtAvailable = balance.SpotAvailableUsd;
            result.FuturesAvailable = balance.FuturesAvailableUsd;
            result.TotalAvailable = result.IsUnifiedAccount
                ? balance.FuturesAvailableUsd
                : balance.SpotAvailableUsd + balance.FuturesAvailableUsd;

            // Calculate margin usage
            if (balance.TotalBalance > 0)
            {
                result.MarginUsagePercent = (balance.MarginUsed / balance.TotalBalance) * 100;
            }

            // Calculate max position size based on available balance
            // For unified: total available / (1 + 1/leverage)
            // For separated: min(spot available, futures available * leverage)
            if (result.IsUnifiedAccount)
            {
                result.MaxPositionSize = result.TotalAvailable / (1 + 1 / maxLeverage);
            }
            else
            {
                result.MaxPositionSize = Math.Min(
                    result.SpotUsdtAvailable,
                    result.FuturesAvailable * maxLeverage
                );
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching execution balances for {Exchange}", exchange);
        }

        return result;
    }

    private async Task<ExecuteOpportunityResponse> ExecuteSpotPerpetualAsync(ExecuteOpportunityRequest request)
    {
        var connector = _exchangeConnectors[request.Exchange];
        var response = new ExecuteOpportunityResponse();

        try
        {
            // Ensure connector is connected
            await EnsureConnectedAsync(request.Exchange, connector);

            _logger.LogInformation(
                "Executing spot-perpetual arbitrage for {Symbol} on {Exchange}, size: ${Size}, leverage: {Leverage}x",
                request.Symbol, request.Exchange, request.PositionSizeUsd, request.Leverage);

            // Get current prices
            var spotPrices = await connector.GetSpotPricesAsync(new List<string> { request.Symbol });
            var perpPrices = await connector.GetPerpetualPricesAsync(new List<string> { request.Symbol });

            if (!spotPrices.ContainsKey(request.Symbol) || !perpPrices.ContainsKey(request.Symbol))
            {
                response.ErrorMessage = $"Unable to fetch prices for {request.Symbol}";
                return response;
            }

            var spotPrice = spotPrices[request.Symbol].Price;
            var perpPrice = perpPrices[request.Symbol];

            // Calculate quantities
            // For spot-perpetual: buy spot (no leverage), short perpetual (with leverage)
            // CRITICAL: Both legs MUST use EXACTLY the same quantity to maintain a proper hedge

            // Step 1: Calculate base quantity from position size and spot price
            // PositionSizeUsd is the amount to invest in coins (same coin quantity on both spot and perpetual)
            var baseQuantity = request.PositionSizeUsd / spotPrice;

            // Step 2: Get instrument info for BOTH spot and perpetual to validate quantity
            decimal finalQuantity;

            if (connector is BybitConnector bybitConnector)
            {
                // For Bybit: Get both spot and perpetual instrument info
                var spotInstrumentInfo = await bybitConnector.GetInstrumentInfoAsync(request.Symbol, isSpot: true);
                var perpInstrumentInfo = await bybitConnector.GetInstrumentInfoAsync(request.Symbol, isSpot: false);

                // Validate quantity with spot constraints
                var spotValidatedQty = spotInstrumentInfo != null
                    ? bybitConnector.ValidateAndAdjustQuantity(baseQuantity, spotInstrumentInfo)
                    : baseQuantity;

                // Validate quantity with perp constraints
                var perpValidatedQty = perpInstrumentInfo != null
                    ? bybitConnector.ValidateAndAdjustQuantity(baseQuantity, perpInstrumentInfo)
                    : baseQuantity;

                // Use the SMALLER quantity to ensure both orders can execute with the same quantity
                finalQuantity = Math.Min(spotValidatedQty, perpValidatedQty);

                _logger.LogInformation(
                    "Quantity validation for {Symbol}: Base={Base}, SpotValidated={Spot}, PerpValidated={Perp}, Final={Final}",
                    request.Symbol, baseQuantity, spotValidatedQty, perpValidatedQty, finalQuantity);
            }
            else
            {
                // For Binance and other exchanges: use simple rounding
                finalQuantity = RoundQuantity(baseQuantity, request.Symbol);
            }

            // Step 3: Place spot buy order and get ACTUAL filled quantity
            string spotOrderId;
            decimal actualFilledQuantity;
            try
            {
                var (orderId, filledQty) = await connector.PlaceSpotBuyOrderAsync(
                    request.Symbol,
                    finalQuantity
                );
                spotOrderId = orderId;
                actualFilledQuantity = filledQty;

                response.OrderIds.Add(spotOrderId);
                _logger.LogInformation("Placed spot buy order {OrderId} for {Symbol} @ ${Price}, Requested: {Requested}, Filled: {Filled}",
                    spotOrderId, request.Symbol, spotPrice, finalQuantity, actualFilledQuantity);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to place spot buy order");
                response.ErrorMessage = $"Failed to place spot buy order: {ex.Message}";
                return response;
            }

            // Step 4: Place perpetual short order using the EXACT FILLED quantity from spot order
            // IMPORTANT: The perpetual market may have different quantity precision than spot
            // We need to round the filled quantity to match perpetual's precision
            decimal perpQuantity = actualFilledQuantity;

            if (connector is BybitConnector bybitConn)
            {
                // Get perpetual instrument info to validate quantity precision
                var perpInstrumentInfo = await bybitConn.GetInstrumentInfoAsync(request.Symbol, isSpot: false);
                if (perpInstrumentInfo != null)
                {
                    perpQuantity = bybitConn.ValidateAndAdjustQuantity(actualFilledQuantity, perpInstrumentInfo);
                    if (perpQuantity != actualFilledQuantity)
                    {
                        _logger.LogWarning("Adjusted perpetual quantity from {Original} to {Adjusted} to match instrument precision",
                            actualFilledQuantity, perpQuantity);
                    }
                }
            }
            else
            {
                // For other exchanges, use simple rounding
                perpQuantity = RoundQuantity(actualFilledQuantity, request.Symbol);
            }

            string perpOrderId;
            try
            {
                _logger.LogInformation("Placing perpetual SHORT for filled quantity: {Quantity}", perpQuantity);

                perpOrderId = await connector.PlaceMarketOrderAsync(
                    request.Symbol,
                    PositionSide.Short,
                    perpQuantity,
                    request.Leverage
                );
                response.OrderIds.Add(perpOrderId);
                _logger.LogInformation("Placed perpetual short order {OrderId} for {Quantity} {Symbol} @ ${Price} with {Leverage}x leverage",
                    perpOrderId, perpQuantity, request.Symbol, perpPrice, request.Leverage);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to place perpetual short order. Spot order was placed: {SpotOrderId} with {Quantity} coins",
                    spotOrderId, actualFilledQuantity);
                response.ErrorMessage = $"Failed to place perpetual short order: {ex.Message}. WARNING: Spot position is open with {actualFilledQuantity} coins!";
                return response;
            }

            // Create lightweight Execution record (minimal tracking only)
            var execution = new Execution
            {
                Symbol = request.Symbol,
                Exchange = request.Exchange,
                StartedAt = DateTime.UtcNow,
                State = ExecutionState.Running,
                FundingEarned = 0,
                PositionSizeUsd = request.PositionSizeUsd,
                SpotOrderId = spotOrderId,
                PerpOrderId = perpOrderId
            };

            _dbContext.Executions.Add(execution);
            await _dbContext.SaveChangesAsync(); // Save to get execution ID

            // Create Position records for tracking (one for perpetual, one for spot)
            var perpPosition = new Position
            {
                ExecutionId = execution.Id,
                Symbol = request.Symbol,
                Exchange = request.Exchange,
                Type = PositionType.Perpetual,
                Side = PositionSide.Short,
                Status = PositionStatus.Open,
                EntryPrice = perpPrice,
                Quantity = perpQuantity, // Use the rounded/adjusted perpetual quantity
                Leverage = request.Leverage,
                InitialMargin = request.PositionSizeUsd / request.Leverage,
                RealizedPnL = 0,
                UnrealizedPnL = 0,
                OrderId = perpOrderId,
                OpenedAt = DateTime.UtcNow
            };

            var spotPosition = new Position
            {
                ExecutionId = execution.Id,
                Symbol = request.Symbol,
                Exchange = request.Exchange,
                Type = PositionType.Spot,
                Side = PositionSide.Long,
                Status = PositionStatus.Open,
                EntryPrice = spotPrice,
                Quantity = actualFilledQuantity,
                Leverage = 1m, // Spot positions don't use leverage
                InitialMargin = request.PositionSizeUsd,
                RealizedPnL = 0,
                UnrealizedPnL = 0,
                OrderId = spotOrderId,
                OpenedAt = DateTime.UtcNow
            };

            _dbContext.Positions.Add(perpPosition);
            _dbContext.Positions.Add(spotPosition);
            await _dbContext.SaveChangesAsync();

            response.Success = true;
            response.Message = $"Successfully opened spot-perpetual arbitrage for {request.Symbol}";
            response.TotalPositionSize = request.PositionSizeUsd;

            _logger.LogInformation(
                "Spot-perpetual arbitrage executed: {Symbol} on {Exchange}, Spot @ ${SpotPrice}, Perp @ ${PerpPrice}, Execution ID: {Id}",
                request.Symbol, request.Exchange, spotPrice, perpPrice, execution.Id);

            return response;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in spot-perpetual execution");
            response.ErrorMessage = $"Execution error: {ex.Message}";
            return response;
        }
    }

    private async Task<ExecuteOpportunityResponse> ExecuteCrossExchangeAsync(ExecuteOpportunityRequest request)
    {
        if (string.IsNullOrEmpty(request.LongExchange) || string.IsNullOrEmpty(request.ShortExchange))
        {
            return new ExecuteOpportunityResponse
            {
                Success = false,
                ErrorMessage = "Long and short exchanges must be specified"
            };
        }

        var longConnector = _exchangeConnectors[request.LongExchange];
        var shortConnector = _exchangeConnectors[request.ShortExchange];
        var response = new ExecuteOpportunityResponse();

        try
        {
            // Ensure both connectors are connected
            await EnsureConnectedAsync(request.LongExchange, longConnector);
            await EnsureConnectedAsync(request.ShortExchange, shortConnector);

            // Determine if this is Futures/Futures or Spot/Futures cross-exchange
            bool isSpotFutures = request.SubType == StrategySubType.CrossExchangeSpotFutures;

            _logger.LogInformation(
                "Executing {SubType} cross-exchange arbitrage for {Symbol}: LONG on {LongExchange}, SHORT on {ShortExchange}, size: ${Size}, leverage: {Leverage}x",
                request.SubType, request.Symbol, request.LongExchange, request.ShortExchange, request.PositionSizeUsd, request.Leverage);

            decimal longPrice, shortPrice;

            if (isSpotFutures)
            {
                // Spot/Futures: Get spot price from long exchange, perpetual price from short exchange
                var spotPrices = await longConnector.GetSpotPricesAsync(new List<string> { request.Symbol });
                var perpPrices = await shortConnector.GetPerpetualPricesAsync(new List<string> { request.Symbol });

                if (!spotPrices.ContainsKey(request.Symbol) || !perpPrices.ContainsKey(request.Symbol))
                {
                    response.ErrorMessage = $"Unable to fetch prices for {request.Symbol} from both exchanges";
                    return response;
                }

                longPrice = spotPrices[request.Symbol].Price;
                shortPrice = perpPrices[request.Symbol];
            }
            else
            {
                // Futures/Futures: Get perpetual prices from both exchanges
                var longPrices = await longConnector.GetPerpetualPricesAsync(new List<string> { request.Symbol });
                var shortPrices = await shortConnector.GetPerpetualPricesAsync(new List<string> { request.Symbol });

                if (!longPrices.ContainsKey(request.Symbol) || !shortPrices.ContainsKey(request.Symbol))
                {
                    response.ErrorMessage = $"Unable to fetch prices for {request.Symbol} from both exchanges";
                    return response;
                }

                longPrice = longPrices[request.Symbol];
                shortPrice = shortPrices[request.Symbol];
            }

            // Calculate quantities - use same quantity for both legs to maintain proper hedge
            var avgPrice = (longPrice + shortPrice) / 2;
            var baseQuantity = request.PositionSizeUsd / avgPrice;

            // Get instrument info from both exchanges to ensure proper quantity precision
            decimal finalQuantity;

            // For futures/futures cross-exchange, get instrument info from both exchanges
            if (!isSpotFutures)
            {
                InstrumentInfo? longInstrumentInfo = null;
                InstrumentInfo? shortInstrumentInfo = null;

                // Get instrument info from long exchange
                if (longConnector is BinanceConnector binanceLong)
                {
                    longInstrumentInfo = await binanceLong.GetInstrumentInfoAsync(request.Symbol, isSpot: false);
                }
                else if (longConnector is BybitConnector bybitLong)
                {
                    longInstrumentInfo = await bybitLong.GetInstrumentInfoAsync(request.Symbol, isSpot: false);
                }

                // Get instrument info from short exchange
                if (shortConnector is BinanceConnector binanceShort)
                {
                    shortInstrumentInfo = await binanceShort.GetInstrumentInfoAsync(request.Symbol, isSpot: false);
                }
                else if (shortConnector is BybitConnector bybitShort)
                {
                    shortInstrumentInfo = await bybitShort.GetInstrumentInfoAsync(request.Symbol, isSpot: false);
                }

                // Validate quantity against both exchanges' requirements
                decimal longValidatedQty = baseQuantity;
                decimal shortValidatedQty = baseQuantity;

                if (longInstrumentInfo != null)
                {
                    if (longConnector is BinanceConnector binanceLongConn)
                    {
                        longValidatedQty = binanceLongConn.ValidateAndAdjustQuantity(baseQuantity, longInstrumentInfo);
                    }
                    else if (longConnector is BybitConnector bybitLongConn)
                    {
                        longValidatedQty = bybitLongConn.ValidateAndAdjustQuantity(baseQuantity, longInstrumentInfo);
                    }
                }

                if (shortInstrumentInfo != null)
                {
                    if (shortConnector is BinanceConnector binanceShortConn)
                    {
                        shortValidatedQty = binanceShortConn.ValidateAndAdjustQuantity(baseQuantity, shortInstrumentInfo);
                    }
                    else if (shortConnector is BybitConnector bybitShortConn)
                    {
                        shortValidatedQty = bybitShortConn.ValidateAndAdjustQuantity(baseQuantity, shortInstrumentInfo);
                    }
                }

                // Use the SMALLER quantity to ensure both orders can execute with the same quantity
                finalQuantity = Math.Min(longValidatedQty, shortValidatedQty);

                _logger.LogInformation(
                    "Cross-exchange quantity validation for {Symbol}: Base={Base}, LongValidated={Long}, ShortValidated={Short}, Final={Final}",
                    request.Symbol, baseQuantity, longValidatedQty, shortValidatedQty, finalQuantity);
            }
            else
            {
                // For spot/futures cross-exchange, use simple rounding (spot side will handle its own validation)
                finalQuantity = RoundQuantity(baseQuantity, request.Symbol);
            }

            var quantity = finalQuantity;

            // Place long order on first exchange
            string longOrderId;
            decimal actualLongQuantity;
            try
            {
                if (isSpotFutures)
                {
                    // Place spot buy order
                    var (orderId, filledQty) = await longConnector.PlaceSpotBuyOrderAsync(
                        request.Symbol,
                        quantity
                    );
                    longOrderId = orderId;
                    actualLongQuantity = filledQty;

                    response.OrderIds.Add($"{request.LongExchange}:SPOT:{longOrderId}");
                    _logger.LogInformation("Placed spot buy order {OrderId} on {Exchange} for {Quantity} {Symbol} @ ${Price}",
                        longOrderId, request.LongExchange, actualLongQuantity, request.Symbol, longPrice);
                }
                else
                {
                    // Place perpetual long order
                    longOrderId = await longConnector.PlaceMarketOrderAsync(
                        request.Symbol,
                        PositionSide.Long,
                        quantity,
                        request.Leverage
                    );
                    actualLongQuantity = quantity; // For futures, assume full fill (we don't get actual fill from this method)

                    response.OrderIds.Add($"{request.LongExchange}:PERP:{longOrderId}");
                    _logger.LogInformation("Placed perpetual long order {OrderId} on {Exchange} for {Quantity} {Symbol} @ ${Price}",
                        longOrderId, request.LongExchange, quantity, request.Symbol, longPrice);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to place long order on {Exchange}", request.LongExchange);
                response.ErrorMessage = $"Failed to place long order on {request.LongExchange}: {ex.Message}";
                return response;
            }

            // Place short order on second exchange
            string shortOrderId;
            try
            {
                shortOrderId = await shortConnector.PlaceMarketOrderAsync(
                    request.Symbol,
                    PositionSide.Short,
                    quantity,
                    request.Leverage
                );
                response.OrderIds.Add($"{request.ShortExchange}:{shortOrderId}");
                _logger.LogInformation("Placed short order {OrderId} on {Exchange} for {Quantity} {Symbol} @ ${Price}",
                    shortOrderId, request.ShortExchange, quantity, request.Symbol, shortPrice);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to place short order on {Exchange}. Long order was placed on {LongExchange}: {LongOrderId}",
                    request.ShortExchange, request.LongExchange, longOrderId);
                response.ErrorMessage = $"Failed to place short order on {request.ShortExchange}: {ex.Message}. WARNING: Long position is open on {request.LongExchange}!";
                return response;
            }

            // Create Execution record for cross-exchange (shared across both positions)
            var execution = new Execution
            {
                Symbol = request.Symbol,
                Exchange = $"{request.LongExchange}/{request.ShortExchange}", // Both exchanges
                StartedAt = DateTime.UtcNow,
                State = ExecutionState.Running,
                FundingEarned = 0,
                PositionSizeUsd = isSpotFutures ? request.PositionSizeUsd : request.PositionSizeUsd * 2, // Spot/Futures uses single size, Futures/Futures uses both legs
                SpotOrderId = isSpotFutures ? longOrderId : null, // For Spot/Futures, store spot order ID here
                PerpOrderId = isSpotFutures ? shortOrderId : $"{longOrderId},{shortOrderId}" // Store both order IDs
            };

            _dbContext.Executions.Add(execution);
            await _dbContext.SaveChangesAsync(); // Save to get execution ID

            // Create Position records for both exchanges
            var longPosition = new Position
            {
                ExecutionId = execution.Id,
                Symbol = request.Symbol,
                Exchange = request.LongExchange,
                Type = isSpotFutures ? PositionType.Spot : PositionType.Perpetual, // Spot for Spot/Futures, Perpetual for Futures/Futures
                Side = PositionSide.Long,
                Status = PositionStatus.Open,
                EntryPrice = longPrice,
                Quantity = isSpotFutures ? actualLongQuantity : quantity, // Use actual filled quantity for spot
                Leverage = isSpotFutures ? 1m : request.Leverage, // Spot has no leverage
                InitialMargin = isSpotFutures ? request.PositionSizeUsd : request.PositionSizeUsd / request.Leverage,
                RealizedPnL = 0,
                UnrealizedPnL = 0,
                OrderId = longOrderId,
                OpenedAt = DateTime.UtcNow
            };

            var shortPosition = new Position
            {
                ExecutionId = execution.Id,
                Symbol = request.Symbol,
                Exchange = request.ShortExchange,
                Type = PositionType.Perpetual,
                Side = PositionSide.Short,
                Status = PositionStatus.Open,
                EntryPrice = shortPrice,
                Quantity = quantity,
                Leverage = request.Leverage,
                InitialMargin = request.PositionSizeUsd / request.Leverage,
                RealizedPnL = 0,
                UnrealizedPnL = 0,
                OrderId = shortOrderId,
                OpenedAt = DateTime.UtcNow
            };

            _dbContext.Positions.Add(longPosition);
            _dbContext.Positions.Add(shortPosition);
            await _dbContext.SaveChangesAsync();

            response.PositionIds.Add(longPosition.Id);
            response.PositionIds.Add(shortPosition.Id);
            response.TotalPositionSize = request.PositionSizeUsd * 2; // Both legs
            response.Success = true;
            response.Message = $"Successfully opened cross-exchange arbitrage for {request.Symbol}";

            _logger.LogInformation(
                "Cross-exchange arbitrage executed: {Symbol} LONG on {LongExchange} @ ${LongPrice}, SHORT on {ShortExchange} @ ${ShortPrice}, Execution ID: {Id}",
                request.Symbol, request.LongExchange, longPrice, request.ShortExchange, shortPrice, execution.Id);

            return response;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in cross-exchange execution");
            response.ErrorMessage = $"Execution error: {ex.Message}";
            return response;
        }
    }

    private async Task EnsureConnectedAsync(string exchangeName, IExchangeConnector connector)
    {
        // Get exchange credentials from configuration
        var exchange = _config.Exchanges.FirstOrDefault(e => e.Name == exchangeName);
        if (exchange == null)
        {
            throw new Exception($"Exchange '{exchangeName}' not found in configuration");
        }

        if (string.IsNullOrEmpty(exchange.ApiKey) || string.IsNullOrEmpty(exchange.ApiSecret))
        {
            throw new Exception($"Exchange '{exchangeName}' has no API credentials configured");
        }

        // Connect the exchange connector
        var connected = await connector.ConnectAsync(exchange.ApiKey, exchange.ApiSecret, exchange.UseDemoTrading);
        if (!connected)
        {
            throw new Exception($"Failed to connect to {exchangeName}");
        }

        var environment = exchange.UseDemoTrading ? "Demo Trading" : "Live";
        _logger.LogInformation("Connected to {Exchange} ({Environment}) for trade execution", exchangeName, environment);
    }

    private decimal RoundQuantity(decimal quantity, string symbol)
    {
        // Symbol-specific precision based on common exchange requirements
        // Most exchanges use standardized precision per asset class

        int decimals;

        // BTC and high-value coins (fractional quantities)
        if (symbol.StartsWith("BTC") || symbol.StartsWith("ETH") || symbol.StartsWith("BNB"))
        {
            if (quantity >= 1m)
                decimals = 3;   // e.g., 123.456 BTC
            else if (quantity >= 0.1m)
                decimals = 4;   // e.g., 0.1234 BTC
            else
                decimals = 5;   // e.g., 0.01234 BTC
        }
        // Mid-tier coins (10-1000 USD range)
        else if (quantity < 10m)
        {
            decimals = 2;       // e.g., 1.23 SOL
        }
        // Low-value coins or large quantities
        else if (quantity < 1000m)
        {
            decimals = 1;       // e.g., 123.4 DOGE
        }
        else
        {
            decimals = 0;       // e.g., 1234 SHIB (large quantities, no decimals)
        }

        return Math.Round(quantity, decimals, MidpointRounding.ToZero);
    }

    public async Task<CloseOpportunityResponse> StopExecutionAsync(int executionId)
    {
        var response = new CloseOpportunityResponse { ActiveOpportunityId = executionId };

        try
        {
            // Find the execution record
            var execution = await _dbContext.Executions
                .FirstOrDefaultAsync(e => e.Id == executionId);

            if (execution == null)
            {
                response.ErrorMessage = $"Execution {executionId} not found";
                return response;
            }

            if (execution.State != ExecutionState.Running)
            {
                response.ErrorMessage = $"Execution {executionId} is not running (state: {execution.State})";
                return response;
            }

            _logger.LogInformation("Stopping Execution {Id} for {Symbol} on {Exchange}",
                executionId, execution.Symbol, execution.Exchange);

            // Determine if this is a cross-exchange execution (format: "Binance/Bybit")
            bool isCrossExchange = execution.Exchange.Contains("/");

            if (isCrossExchange)
            {
                // Handle cross-exchange (cross-fut) strategy
                return await StopCrossExchangeExecutionAsync(execution, response);
            }
            else
            {
                // Handle single-exchange (spot-perpetual) strategy
                return await StopSingleExchangeExecutionAsync(execution, response);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping Execution {Id}", executionId);
            response.ErrorMessage = $"Failed to stop execution: {ex.Message}";
            return response;
        }
    }

    private async Task<CloseOpportunityResponse> StopSingleExchangeExecutionAsync(
        Execution execution,
        CloseOpportunityResponse response)
    {
        var connector = _exchangeConnectors[execution.Exchange];

        // Ensure connector is connected
        await EnsureConnectedAsync(execution.Exchange, connector);

            // IMPORTANT: For spot-perpetual strategy, we must close BOTH legs:
            // 1. Sell the spot asset (LONG position)
            // 2. Close the perpetual SHORT position
            // We do spot first so if it fails, we still have the hedge in place

            // Step 1: Get spot balance and sell ALL of the spot asset
            decimal spotQuantity = 0;
            string? spotSellOrderId = null;
            try
            {
                // Extract base asset from symbol (e.g., BTCUSDT -> BTC, ETHUSDT -> ETH)
                var baseAsset = execution.Symbol.Replace("USDT", "").Replace("BUSD", "").Replace("USDC", "");

                _logger.LogInformation("Checking spot balance for base asset: {Asset}", baseAsset);
                spotQuantity = await connector.GetSpotBalanceAsync(baseAsset);

                if (spotQuantity == 0)
                {
                    _logger.LogWarning("No spot balance found for {Asset}. This may indicate the position was already closed or there was an error during execution.", baseAsset);
                }
                else
                {
                    // Round the spot quantity to match the spot market's precision
                    decimal roundedSpotQuantity = spotQuantity;

                    if (connector is BybitConnector bybitConn)
                    {
                        var spotInstrumentInfo = await bybitConn.GetInstrumentInfoAsync(execution.Symbol, isSpot: true);
                        if (spotInstrumentInfo != null && spotInstrumentInfo.QtyStep > 0)
                        {
                            roundedSpotQuantity = bybitConn.ValidateAndAdjustQuantity(spotQuantity, spotInstrumentInfo);
                            if (roundedSpotQuantity != spotQuantity)
                            {
                                _logger.LogWarning("Adjusted spot sell quantity from {Original} to {Adjusted} to match instrument precision (step: {Step})",
                                    spotQuantity, roundedSpotQuantity, spotInstrumentInfo.QtyStep);
                            }
                        }
                        else
                        {
                            // Fallback: If instrument info is not available, use simple rounding
                            // This should rarely happen since we always fetch instrument info
                            roundedSpotQuantity = RoundQuantity(spotQuantity, execution.Symbol);
                            _logger.LogWarning("No instrument info available, using fallback rounding. Spot sell quantity from {Original} to {Adjusted}",
                                spotQuantity, roundedSpotQuantity);
                        }
                    }
                    else
                    {
                        roundedSpotQuantity = RoundQuantity(spotQuantity, execution.Symbol);
                    }

                    _logger.LogInformation("Found spot balance for {Asset}: {Quantity}. Placing sell order for {Symbol} with rounded quantity: {RoundedQty}",
                        baseAsset, spotQuantity, execution.Symbol, roundedSpotQuantity);

                    spotSellOrderId = await connector.PlaceSpotSellOrderAsync(
                        execution.Symbol,
                        roundedSpotQuantity
                    );

                    _logger.LogInformation(
                        "Successfully sold spot asset {Asset}, quantity: {Quantity}, order ID: {OrderId}",
                        baseAsset, roundedSpotQuantity, spotSellOrderId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to sell spot asset for {Symbol}. Will NOT close perpetual position to maintain hedge!",
                    execution.Symbol);
                response.ErrorMessage = $"Failed to sell spot asset: {ex.Message}. Perpetual position remains open to maintain hedge. Please close manually!";
                return response;
            }

            // Step 2: Close SHORT perpetual position using ClosePositionAsync
            try
            {
                _logger.LogInformation("Closing perpetual short position for {Symbol}", execution.Symbol);

                bool closedSuccessfully = await connector.ClosePositionAsync(execution.Symbol);

                if (closedSuccessfully)
                {
                    _logger.LogInformation("Successfully closed perpetual short position for {Symbol}", execution.Symbol);
                }
                else
                {
                    _logger.LogWarning("ClosePositionAsync returned false for {Symbol}, position may already be closed", execution.Symbol);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to close perpetual short position for {Symbol}. Spot asset was already sold!", execution.Symbol);
                response.ErrorMessage = $"Failed to close perpetual position: {ex.Message}. WARNING: Spot asset was already sold (order: {spotSellOrderId})!";
                return response;
            }

            // Update execution state to Stopped
            execution.State = ExecutionState.Stopped;
            execution.StoppedAt = DateTime.UtcNow;

            // Close related Position records
            var positions = await _dbContext.Positions
                .Where(p => p.ExecutionId == execution.Id && p.Status == PositionStatus.Open)
                .ToListAsync();

            foreach (var position in positions)
            {
                position.Status = PositionStatus.Closed;
                position.ClosedAt = DateTime.UtcNow;
                position.ExecutionId = null; // Detach from execution (historical record)
            }

            await _dbContext.SaveChangesAsync();

            _logger.LogInformation("Execution {Id} stopped successfully, closed {Count} positions, deleting execution record",
                execution.Id, positions.Count);

            // DELETE the Execution record (don't keep stopped records)
            _dbContext.Executions.Remove(execution);
            await _dbContext.SaveChangesAsync();

            _logger.LogInformation("Deleted Execution {Id} from database", execution.Id);

            response.Success = true;
            response.Message = $"Successfully stopped execution for {execution.Symbol}";

            return response;
    }

    private async Task<CloseOpportunityResponse> StopCrossExchangeExecutionAsync(
        Execution execution,
        CloseOpportunityResponse response)
    {
        // For cross-exchange strategies, we need to close positions on both exchanges
        // The execution.Exchange field contains "Exchange1/Exchange2" format

        // Get all positions for this execution to determine which exchanges are involved
        var positions = await _dbContext.Positions
            .Where(p => p.ExecutionId == execution.Id && p.Status == PositionStatus.Open)
            .ToListAsync();

        if (positions.Count == 0)
        {
            _logger.LogWarning("No open positions found for Execution {Id}", execution.Id);
            response.ErrorMessage = "No open positions found for this execution";
            return response;
        }

        _logger.LogInformation("Found {Count} open positions for cross-exchange execution {Id}",
            positions.Count, execution.Id);

        // Group positions by exchange
        var positionsByExchange = positions.GroupBy(p => p.Exchange).ToList();

        var closedExchanges = new List<string>();
        var failedExchanges = new List<string>();

        // Close positions on each exchange
        foreach (var exchangeGroup in positionsByExchange)
        {
            var exchangeName = exchangeGroup.Key;
            var exchangePositions = exchangeGroup.ToList();

            try
            {
                if (!_exchangeConnectors.TryGetValue(exchangeName, out var connector))
                {
                    _logger.LogError("Exchange connector not found for {Exchange}", exchangeName);
                    failedExchanges.Add(exchangeName);
                    continue;
                }

                _logger.LogInformation("Closing {Count} positions on {Exchange}",
                    exchangePositions.Count, exchangeName);

                // Ensure connector is connected
                await EnsureConnectedAsync(exchangeName, connector);

                // Close all perpetual positions for this exchange/symbol
                bool closedSuccessfully = await connector.ClosePositionAsync(execution.Symbol);

                if (closedSuccessfully)
                {
                    _logger.LogInformation("Successfully closed positions on {Exchange} for {Symbol}",
                        exchangeName, execution.Symbol);
                    closedExchanges.Add(exchangeName);
                }
                else
                {
                    _logger.LogWarning("ClosePositionAsync returned false for {Exchange}/{Symbol}, position may already be closed",
                        exchangeName, execution.Symbol);
                    closedExchanges.Add(exchangeName); // Still count as success
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to close positions on {Exchange} for {Symbol}",
                    exchangeName, execution.Symbol);
                failedExchanges.Add(exchangeName);
            }
        }

        // Check if we successfully closed positions on all exchanges
        if (failedExchanges.Count > 0)
        {
            var failedList = string.Join(", ", failedExchanges);
            var closedList = closedExchanges.Count > 0 ? string.Join(", ", closedExchanges) : "none";
            response.ErrorMessage = $"Failed to close positions on: {failedList}. Successfully closed on: {closedList}";
            return response;
        }

        // Update execution state to Stopped
        execution.State = ExecutionState.Stopped;
        execution.StoppedAt = DateTime.UtcNow;

        // Close all position records
        foreach (var position in positions)
        {
            position.Status = PositionStatus.Closed;
            position.ClosedAt = DateTime.UtcNow;
            position.ExecutionId = null; // Detach from execution (historical record)
        }

        await _dbContext.SaveChangesAsync();

        _logger.LogInformation("Cross-exchange execution {Id} stopped successfully, closed {Count} positions on {Exchanges}",
            execution.Id, positions.Count, string.Join(", ", closedExchanges));

        // DELETE the Execution record (don't keep stopped records)
        _dbContext.Executions.Remove(execution);
        await _dbContext.SaveChangesAsync();

        _logger.LogInformation("Deleted Execution {Id} from database", execution.Id);

        response.Success = true;
        response.Message = $"Successfully stopped cross-exchange execution for {execution.Symbol} on {string.Join(", ", closedExchanges)}";

        return response;
    }
}
