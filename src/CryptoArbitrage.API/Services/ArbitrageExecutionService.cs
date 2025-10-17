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
            var validationError = ValidateRequest(request);
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
                // COMMENTED OUT: Cross-exchange arbitrage not part of the refactoring scope
                return new ExecuteOpportunityResponse
                {
                    Success = false,
                    ErrorMessage = "Cross-exchange strategy is currently disabled"
                };
                // return await ExecuteCrossExchangeAsync(request);
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

    private string? ValidateRequest(ExecuteOpportunityRequest request)
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
        }

        return null;
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
            // Use the SAME quantity for both legs to maintain a proper hedge
            var baseQuantity = request.PositionSizeUsd / spotPrice;
            var quantity = RoundQuantity(baseQuantity, request.Symbol);

            var spotQuantity = quantity;
            var perpQuantity = quantity;

            // Step 1: Place spot buy order using PlaceSpotBuyOrderAsync
            string spotOrderId;
            try
            {
                spotOrderId = await connector.PlaceSpotBuyOrderAsync(
                    request.Symbol,
                    spotQuantity
                );
                response.OrderIds.Add(spotOrderId);
                _logger.LogInformation("Placed spot buy order {OrderId} for {Quantity} {Symbol} @ ${Price}",
                    spotOrderId, spotQuantity, request.Symbol, spotPrice);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to place spot buy order");
                response.ErrorMessage = $"Failed to place spot buy order: {ex.Message}";
                return response;
            }

            // Step 2: Place perpetual short order using PlaceMarketOrderAsync
            string perpOrderId;
            try
            {
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
                _logger.LogError(ex, "Failed to place perpetual short order. Spot order was placed: {SpotOrderId}", spotOrderId);
                response.ErrorMessage = $"Failed to place perpetual short order: {ex.Message}. WARNING: Spot position is open!";
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
                Quantity = perpQuantity,
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
                Quantity = spotQuantity,
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

    // COMMENTED OUT: Cross-exchange arbitrage not part of the refactoring scope
    // This method references ArbitrageOpportunities table and Positions table which no longer exist
    // private async Task<ExecuteOpportunityResponse> ExecuteCrossExchangeAsync(ExecuteOpportunityRequest request)
    // {
    //     if (string.IsNullOrEmpty(request.LongExchange) || string.IsNullOrEmpty(request.ShortExchange))
    //     {
    //         return new ExecuteOpportunityResponse
    //         {
    //             Success = false,
    //             ErrorMessage = "Long and short exchanges must be specified"
    //         };
    //     }
    //
    //     var longConnector = _exchangeConnectors[request.LongExchange];
    //     var shortConnector = _exchangeConnectors[request.ShortExchange];
    //     var response = new ExecuteOpportunityResponse();
    //
    //     try
    //     {
    //         // Ensure both connectors are connected
    //         await EnsureConnectedAsync(request.LongExchange, longConnector);
    //         await EnsureConnectedAsync(request.ShortExchange, shortConnector);
    //
    //         _logger.LogInformation(
    //             "Executing cross-exchange arbitrage for {Symbol}: LONG on {LongExchange}, SHORT on {ShortExchange}, size: ${Size}, leverage: {Leverage}x",
    //             request.Symbol, request.LongExchange, request.ShortExchange, request.PositionSizeUsd, request.Leverage);
    //
    //         // Get current prices from both exchanges
    //         var longPrices = await longConnector.GetPerpetualPricesAsync(new List<string> { request.Symbol });
    //         var shortPrices = await shortConnector.GetPerpetualPricesAsync(new List<string> { request.Symbol });
    //
    //         if (!longPrices.ContainsKey(request.Symbol) || !shortPrices.ContainsKey(request.Symbol))
    //         {
    //             response.ErrorMessage = $"Unable to fetch prices for {request.Symbol} from both exchanges";
    //             return response;
    //         }
    //
    //         var longPrice = longPrices[request.Symbol];
    //         var shortPrice = shortPrices[request.Symbol];
    //
    //         // Calculate quantities
    //         // Use the SAME quantity for both legs to maintain a proper hedge
    //         // Base calculation on the average price for better hedging
    //         var avgPrice = (longPrice + shortPrice) / 2;
    //         var baseQuantity = request.PositionSizeUsd / avgPrice;
    //         var quantity = RoundQuantity(baseQuantity, request.Symbol);
    //
    //         var longQuantity = quantity;
    //         var shortQuantity = quantity;
    //
    //         // Place long order on first exchange
    //         string longOrderId;
    //         try
    //         {
    //             longOrderId = await longConnector.PlaceMarketOrderAsync(
    //                 request.Symbol,
    //                 PositionSide.Long,
    //                 longQuantity,
    //                 request.Leverage
    //             );
    //             response.OrderIds.Add($"{request.LongExchange}:{longOrderId}");
    //         }
    //         catch (Exception ex)
    //         {
    //             _logger.LogError(ex, "Failed to place long order on {Exchange}", request.LongExchange);
    //             response.ErrorMessage = $"Failed to place long order on {request.LongExchange}: {ex.Message}";
    //             return response;
    //         }
    //
    //         // Place short order on second exchange
    //         string shortOrderId;
    //         try
    //         {
    //             shortOrderId = await shortConnector.PlaceMarketOrderAsync(
    //                 request.Symbol,
    //                 PositionSide.Short,
    //                 shortQuantity,
    //                 request.Leverage
    //             );
    //             response.OrderIds.Add($"{request.ShortExchange}:{shortOrderId}");
    //         }
    //         catch (Exception ex)
    //         {
    //             _logger.LogError(ex, "Failed to place short order on {Exchange}. Long order was placed on {LongExchange}: {LongOrderId}",
    //                 request.ShortExchange, request.LongExchange, longOrderId);
    //             response.ErrorMessage = $"Failed to place short order on {request.ShortExchange}: {ex.Message}. WARNING: Long position is open on {request.LongExchange}!";
    //             return response;
    //         }
    //
    //         // Get exchange IDs from database
    //         var longExchange = await _dbContext.Exchanges.FirstOrDefaultAsync(e => e.Name == request.LongExchange);
    //         var shortExchange = await _dbContext.Exchanges.FirstOrDefaultAsync(e => e.Name == request.ShortExchange);
    //
    //         if (longExchange == null || shortExchange == null)
    //         {
    //             response.ErrorMessage = $"Exchange(s) not found in database";
    //             return response;
    //         }
    //
    //         // Create ArbitrageOpportunity record (historical/detected opportunity)
    //         var arbitrageOpportunity = new ArbitrageOpportunity
    //         {
    //             Symbol = request.Symbol,
    //             LongExchangeId = longExchange.Id,
    //             ShortExchangeId = shortExchange.Id,
    //             LongFundingRate = request.LongFundingRate ?? 0,
    //             ShortFundingRate = request.ShortFundingRate ?? 0,
    //             SpreadRate = request.SpreadRate,
    //             AnnualizedSpread = request.AnnualizedSpread,
    //             EstimatedProfitPercentage = request.EstimatedProfitPercentage,
    //             Status = OpportunityStatus.Executed,
    //             DetectedAt = DateTime.UtcNow,
    //             ExecutedAt = DateTime.UtcNow,
    //             Notes = $"CrossExchange|{request.Strategy}"
    //         };
    //
    //         _dbContext.ArbitrageOpportunities.Add(arbitrageOpportunity);
    //         await _dbContext.SaveChangesAsync(); // Save to get the opportunity ID
    //
    //         // Create ActiveOpportunity record (currently executing opportunity)
    //         var activeOpportunity = new ActiveOpportunity
    //         {
    //             ArbitrageOpportunityId = arbitrageOpportunity.Id,
    //             ExecutedAt = DateTime.UtcNow,
    //             PositionSizeUsd = request.PositionSizeUsd,
    //             Leverage = request.Leverage,
    //             StopLossPercentage = request.StopLossPercentage,
    //             TakeProfitPercentage = request.TakeProfitPercentage,
    //             Strategy = ArbitrageStrategy.CrossExchange,
    //             Exchange = null, // CrossExchange doesn't have a single exchange
    //             IsActive = true,
    //             Notes = $"CrossExchange|SL:{request.StopLossPercentage}|TP:{request.TakeProfitPercentage}"
    //         };
    //
    //         _dbContext.ActiveOpportunities.Add(activeOpportunity);
    //         await _dbContext.SaveChangesAsync(); // Save to get the active opportunity ID
    //
    //         // Record positions in database and link to active opportunity
    //         var longPosition = new Position
    //         {
    //             ExchangeId = longExchange.Id,
    //             Symbol = request.Symbol,
    //             Side = PositionSide.Long,
    //             EntryPrice = longPrice,
    //             Quantity = longQuantity,
    //             Leverage = request.Leverage,
    //             Status = PositionStatus.Open,
    //             OpenedAt = DateTime.UtcNow,
    //             ActiveOpportunityId = activeOpportunity.Id,
    //             Notes = $"Long|{request.LongExchange}|CrossExchange"
    //         };
    //
    //         var shortPosition = new Position
    //         {
    //             ExchangeId = shortExchange.Id,
    //             Symbol = request.Symbol,
    //             Side = PositionSide.Short,
    //             EntryPrice = shortPrice,
    //             Quantity = shortQuantity,
    //             Leverage = request.Leverage,
    //             Status = PositionStatus.Open,
    //             OpenedAt = DateTime.UtcNow,
    //             ActiveOpportunityId = activeOpportunity.Id,
    //             Notes = $"Short|{request.ShortExchange}|CrossExchange"
    //         };
    //
    //         _dbContext.Positions.Add(longPosition);
    //         _dbContext.Positions.Add(shortPosition);
    //         await _dbContext.SaveChangesAsync();
    //
    //         response.PositionIds.Add(longPosition.Id);
    //         response.PositionIds.Add(shortPosition.Id);
    //         response.TotalPositionSize = request.PositionSizeUsd * 2; // Both legs
    //         response.Success = true;
    //         response.Message = $"Successfully opened cross-exchange arbitrage for {request.Symbol}";
    //
    //         _logger.LogInformation(
    //             "Cross-exchange arbitrage executed: {Symbol} LONG on {LongExchange} @ ${LongPrice}, SHORT on {ShortExchange} @ ${ShortPrice}",
    //             request.Symbol, request.LongExchange, longPrice, request.ShortExchange, shortPrice);
    //
    //         return response;
    //     }
    //     catch (Exception ex)
    //     {
    //         _logger.LogError(ex, "Error in cross-exchange execution");
    //         response.ErrorMessage = $"Execution error: {ex.Message}";
    //         return response;
    //     }
    // }

    private async Task EnsureConnectedAsync(string exchangeName, IExchangeConnector connector)
    {
        // Get exchange credentials from database
        var exchange = await _dbContext.Exchanges.FirstOrDefaultAsync(e => e.Name == exchangeName);
        if (exchange == null)
        {
            throw new Exception($"Exchange '{exchangeName}' not found in database");
        }

        if (string.IsNullOrEmpty(exchange.ApiKey) || string.IsNullOrEmpty(exchange.ApiSecret))
        {
            throw new Exception($"Exchange '{exchangeName}' has no API credentials configured");
        }

        // Connect the exchange connector
        var connected = await connector.ConnectAsync(exchange.ApiKey, exchange.ApiSecret);
        if (!connected)
        {
            throw new Exception($"Failed to connect to {exchangeName}");
        }

        _logger.LogInformation("Connected to {Exchange} for trade execution", exchangeName);
    }

    private decimal RoundQuantity(decimal quantity, string symbol)
    {
        // Different symbols have different precision requirements
        // BTC typically allows 5-6 decimals, altcoins may allow 0-3 decimals
        // For safety, use a reasonable default precision based on quantity size

        int decimals;
        if (quantity >= 1000m)
            decimals = 0;      // Large quantities: whole numbers
        else if (quantity >= 10m)
            decimals = 1;      // Medium quantities: 1 decimal
        else if (quantity >= 1m)
            decimals = 2;      // Small quantities: 2 decimals
        else if (quantity >= 0.01m)
            decimals = 3;      // Smaller quantities: 3 decimals
        else
            decimals = 5;      // Very small quantities (like BTC): 5 decimals

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

            var connector = _exchangeConnectors[execution.Exchange];

            // Ensure connector is connected
            await EnsureConnectedAsync(execution.Exchange, connector);

            // Step 1: Close SHORT perpetual position using ClosePositionAsync
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
                    _logger.LogWarning("ClosePositionAsync returned false for {Symbol}, but will continue", execution.Symbol);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to close perpetual short position for {Symbol}", execution.Symbol);
                response.ErrorMessage = $"Failed to close perpetual position: {ex.Message}";
                return response;
            }

            // Step 2: Get spot balance and sell ALL of the spot asset
            decimal spotQuantity;
            try
            {
                // Extract base asset from symbol (e.g., BTCUSDT -> BTC)
                var baseAsset = execution.Symbol.Replace("USDT", "").Replace("BUSD", "");
                spotQuantity = await connector.GetSpotBalanceAsync(baseAsset);

                if (spotQuantity == 0)
                {
                    _logger.LogWarning("No spot balance found for {Asset}, nothing to sell", baseAsset);
                }
                else
                {
                    _logger.LogInformation("Selling spot asset for {Symbol}, quantity: {Quantity}",
                        execution.Symbol, spotQuantity);

                    string spotSellOrderId = await connector.PlaceSpotSellOrderAsync(
                        execution.Symbol,
                        spotQuantity
                    );

                    _logger.LogInformation(
                        "Successfully sold spot asset for {Symbol}, order ID: {OrderId}",
                        execution.Symbol, spotSellOrderId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to sell spot asset for {Symbol}. Perpetual position was already closed!",
                    execution.Symbol);
                response.ErrorMessage = $"Failed to sell spot asset: {ex.Message}. WARNING: Perpetual position is already closed!";
                return response;
            }

            // Update execution state to Stopped
            execution.State = ExecutionState.Stopped;
            execution.StoppedAt = DateTime.UtcNow;

            // Close related Position records
            var positions = await _dbContext.Positions
                .Where(p => p.ExecutionId == executionId && p.Status == PositionStatus.Open)
                .ToListAsync();

            foreach (var position in positions)
            {
                position.Status = PositionStatus.Closed;
                position.ClosedAt = DateTime.UtcNow;
                position.ExecutionId = null; // Detach from execution (historical record)
            }

            await _dbContext.SaveChangesAsync();

            _logger.LogInformation("Execution {Id} stopped successfully, closed {Count} positions, deleting execution record",
                executionId, positions.Count);

            // DELETE the Execution record (don't keep stopped records)
            _dbContext.Executions.Remove(execution);
            await _dbContext.SaveChangesAsync();

            _logger.LogInformation("Deleted Execution {Id} from database", executionId);

            response.Success = true;
            response.Message = $"Successfully stopped execution for {execution.Symbol}";

            return response;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping Execution {Id}", executionId);
            response.ErrorMessage = $"Failed to stop execution: {ex.Message}";
            return response;
        }
    }
}
