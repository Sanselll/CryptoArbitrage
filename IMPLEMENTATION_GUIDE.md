# Trading Data Tabs - Implementation Guide

## Status: IN PROGRESS

### Completed ✅
1. Database Entities created (Order.cs, Trade.cs, Transaction.cs)
2. DTOs created (OrderDto.cs, TradeDto.cs, TransactionDto.cs)
3. IExchangeConnector interface updated with new methods

### Remaining Work

---

## Phase 1: Connector Implementations

### Step 4: BinanceConnector.cs
Location: `/Users/sansel/Projects/CryptoArbitrage/src/CryptoArbitrage.API/Services/Exchanges/BinanceConnector.cs`

Add these four methods to the class (add after line 1222):

```csharp
public async Task<List<OrderDto>> GetOpenOrdersAsync()
{
    if (_restClient == null)
        throw new InvalidOperationException("Not connected to Binance");

    var orders = new List<OrderDto>();

    try
    {
        var result = await _restClient.UsdFuturesApi.Trading.GetOpenOrdersAsync();

        if (result.Success && result.Data != null)
        {
            orders = result.Data.Select(o => new OrderDto
            {
                Exchange = ExchangeName,
                OrderId = o.Id.ToString(),
                ClientOrderId = o.ClientOrderId,
                Symbol = o.Symbol,
                Side = o.Side == Binance.Net.Enums.OrderSide.Buy ? Data.Entities.OrderSide.Buy : Data.Entities.OrderSide.Sell,
                Type = MapBinanceOrderType(o.Type),
                Status = MapBinanceOrderStatus(o.Status),
                TimeInForce = o.TimeInForce?.ToString(),
                Price = o.Price,
                AveragePrice = o.AveragePrice,
                StopPrice = o.StopPrice,
                Quantity = o.Quantity,
                FilledQuantity = o.QuantityFilled ?? 0,
                Fee = 0, // Not available in open orders
                FeeAsset = null,
                CreatedAt = o.CreateTime,
                UpdatedAt = o.UpdateTime ?? o.CreateTime,
                WorkingTime = o.WorkingTime,
                ReduceOnly = o.ReduceOnly?.ToString(),
                PostOnly = o.TimeInForce == Binance.Net.Enums.TimeInForce.PostOnly ? "true" : null
            }).ToList();
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error fetching open orders from Binance");
    }

    return orders;
}

public async Task<List<OrderDto>> GetOrderHistoryAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
{
    if (_restClient == null)
        throw new InvalidOperationException("Not connected to Binance");

    var orders = new List<OrderDto>();

    try
    {
        var result = await _restClient.UsdFuturesApi.Trading.GetOrdersAsync(
            startTime: startTime,
            endTime: endTime,
            limit: limit);

        if (result.Success && result.Data != null)
        {
            orders = result.Data.Select(o => new OrderDto
            {
                Exchange = ExchangeName,
                OrderId = o.Id.ToString(),
                ClientOrderId = o.ClientOrderId,
                Symbol = o.Symbol,
                Side = o.Side == Binance.Net.Enums.OrderSide.Buy ? Data.Entities.OrderSide.Buy : Data.Entities.OrderSide.Sell,
                Type = MapBinanceOrderType(o.Type),
                Status = MapBinanceOrderStatus(o.Status),
                TimeInForce = o.TimeInForce?.ToString(),
                Price = o.Price,
                AveragePrice = o.AveragePrice,
                StopPrice = o.StopPrice,
                Quantity = o.Quantity,
                FilledQuantity = o.QuantityFilled ?? 0,
                Fee = 0, // Use trades for fee info
                FeeAsset = null,
                CreatedAt = o.CreateTime,
                UpdatedAt = o.UpdateTime ?? o.CreateTime,
                WorkingTime = o.WorkingTime,
                ReduceOnly = o.ReduceOnly?.ToString(),
                PostOnly = o.TimeInForce == Binance.Net.Enums.TimeInForce.PostOnly ? "true" : null
            }).ToList();
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error fetching order history from Binance");
    }

    return orders;
}

public async Task<List<TradeDto>> GetUserTradesAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
{
    if (_restClient == null)
        throw new InvalidOperationException("Not connected to Binance");

    var trades = new List<TradeDto>();

    try
    {
        var result = await _restClient.UsdFuturesApi.Trading.GetUserTradesAsync(
            startTime: startTime,
            endTime: endTime,
            limit: limit);

        if (result.Success && result.Data != null)
        {
            trades = result.Data.Select(t => new TradeDto
            {
                Exchange = ExchangeName,
                TradeId = t.Id.ToString(),
                OrderId = t.OrderId.ToString(),
                Symbol = t.Symbol,
                Side = t.Side == Binance.Net.Enums.OrderSide.Buy ? Data.Entities.OrderSide.Buy : Data.Entities.OrderSide.Sell,
                Price = t.Price,
                Quantity = t.Quantity,
                QuoteQuantity = t.QuoteQuantity,
                Fee = t.Fee,
                FeeAsset = t.FeeAsset,
                Commission = t.Fee,
                CommissionAsset = t.FeeAsset,
                IsMaker = t.Maker,
                IsBuyer = t.Buyer,
                ExecutedAt = t.Timestamp,
                OrderType = null, // Not available
                PositionSide = t.PositionSide?.ToString()
            }).ToList();
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error fetching user trades from Binance");
    }

    return trades;
}

public async Task<List<TransactionDto>> GetTransactionsAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
{
    if (_restClient == null)
        throw new InvalidOperationException("Not connected to Binance");

    var transactions = new List<TransactionDto>();

    try
    {
        var result = await _restClient.UsdFuturesApi.Account.GetIncomeHistoryAsync(
            startTime: startTime,
            endTime: endTime,
            limit: limit);

        if (result.Success && result.Data != null)
        {
            transactions = result.Data.Select(t => new TransactionDto
            {
                Exchange = ExchangeName,
                TransactionId = t.TransactionId?.ToString() ?? t.Timestamp.Ticks.ToString(),
                TxHash = t.TransactionId?.ToString(),
                Type = MapBinanceIncomeType(t.IncomeType),
                Asset = t.Asset,
                Amount = t.Income,
                Status = Data.Entities.TransactionStatus.Confirmed,
                FromAddress = null,
                ToAddress = null,
                Network = null,
                Info = t.Info,
                Symbol = t.Symbol,
                TradeId = t.TradeId?.ToString(),
                Fee = 0,
                FeeAsset = null,
                CreatedAt = t.Timestamp,
                ConfirmedAt = t.Timestamp
            }).ToList();
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error fetching transactions from Binance");
    }

    return transactions;
}

// Helper methods
private Data.Entities.OrderType MapBinanceOrderType(Binance.Net.Enums.FuturesOrderType type)
{
    return type switch
    {
        Binance.Net.Enums.FuturesOrderType.Market => Data.Entities.OrderType.Market,
        Binance.Net.Enums.FuturesOrderType.Limit => Data.Entities.OrderType.Limit,
        Binance.Net.Enums.FuturesOrderType.Stop => Data.Entities.OrderType.StopMarket,
        Binance.Net.Enums.FuturesOrderType.StopMarket => Data.Entities.OrderType.StopMarket,
        Binance.Net.Enums.FuturesOrderType.TakeProfit => Data.Entities.OrderType.TakeProfitMarket,
        Binance.Net.Enums.FuturesOrderType.TakeProfitMarket => Data.Entities.OrderType.TakeProfitMarket,
        _ => Data.Entities.OrderType.Market
    };
}

private Data.Entities.OrderStatus MapBinanceOrderStatus(Binance.Net.Enums.OrderStatus status)
{
    return status switch
    {
        Binance.Net.Enums.OrderStatus.New => Data.Entities.OrderStatus.New,
        Binance.Net.Enums.OrderStatus.PartiallyFilled => Data.Entities.OrderStatus.PartiallyFilled,
        Binance.Net.Enums.OrderStatus.Filled => Data.Entities.OrderStatus.Filled,
        Binance.Net.Enums.OrderStatus.Canceled => Data.Entities.OrderStatus.Canceled,
        Binance.Net.Enums.OrderStatus.Rejected => Data.Entities.OrderStatus.Rejected,
        Binance.Net.Enums.OrderStatus.Expired => Data.Entities.OrderStatus.Expired,
        _ => Data.Entities.OrderStatus.New
    };
}

private Data.Entities.TransactionType MapBinanceIncomeType(Binance.Net.Enums.IncomeType type)
{
    return type switch
    {
        Binance.Net.Enums.IncomeType.RealizedPnl => Data.Entities.TransactionType.RealizedPnL,
        Binance.Net.Enums.IncomeType.FundingFee => Data.Entities.TransactionType.Funding,
        Binance.Net.Enums.IncomeType.Commission => Data.Entities.TransactionType.Commission,
        Binance.Net.Enums.IncomeType.ReferralKickback => Data.Entities.TransactionType.ReferralKickback,
        Binance.Net.Enums.IncomeType.InsuranceClear => Data.Entities.TransactionType.InsuranceClear,
        Binance.Net.Enums.IncomeType.Transfer => Data.Entities.TransactionType.Transfer,
        _ => Data.Entities.TransactionType.Other
    };
}
```

### Step 5: BybitConnector.cs
Location: `/Users/sansel/Projects/CryptoArbitrage/src/CryptoArbitrage.API/Services/Exchanges/BybitConnector.cs`

Add after line 1257 (after ConvertPrecisionStringToDecimals method):

```csharp
public async Task<List<OrderDto>> GetOpenOrdersAsync()
{
    if (_restClient == null)
        throw new InvalidOperationException("Not connected to Bybit");

    var orders = new List<OrderDto>();

    try
    {
        var result = await _restClient.V5Api.Trading.GetOrdersAsync(
            Category.Linear,
            settleCoin: "USDT");

        if (result.Success && result.Data?.List != null)
        {
            orders = result.Data.List
                .Where(o => o.OrderStatus == Bybit.Net.Enums.V5.OrderStatus.New ||
                           o.OrderStatus == Bybit.Net.Enums.V5.OrderStatus.PartiallyFilled)
                .Select(o => new OrderDto
                {
                    Exchange = ExchangeName,
                    OrderId = o.OrderId,
                    ClientOrderId = o.OrderLinkId,
                    Symbol = o.Symbol,
                    Side = o.Side == Bybit.Net.Enums.OrderSide.Buy ? Data.Entities.OrderSide.Buy : Data.Entities.OrderSide.Sell,
                    Type = MapBybitOrderType(o.OrderType),
                    Status = MapBybitOrderStatus(o.OrderStatus),
                    TimeInForce = o.TimeInForce?.ToString(),
                    Price = o.Price,
                    AveragePrice = o.AveragePrice > 0 ? o.AveragePrice : null,
                    StopPrice = o.TriggerPrice,
                    Quantity = o.Quantity,
                    FilledQuantity = o.QuantityFilled ?? 0,
                    Fee = o.CumulativeExecFee ?? 0,
                    FeeAsset = "USDT",
                    CreatedAt = o.CreateTime,
                    UpdatedAt = o.UpdateTime,
                    WorkingTime = o.CreateTime,
                    ReduceOnly = o.ReduceOnly?.ToString(),
                    PostOnly = o.TimeInForce == Bybit.Net.Enums.V5.TimeInForce.PostOnly ? "true" : null
                }).ToList();
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error fetching open orders from Bybit");
    }

    return orders;
}

public async Task<List<OrderDto>> GetOrderHistoryAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
{
    if (_restClient == null)
        throw new InvalidOperationException("Not connected to Bybit");

    var orders = new List<OrderDto>();

    try
    {
        var result = await _restClient.V5Api.Trading.GetOrderHistoryAsync(
            Category.Linear,
            startTime: startTime,
            endTime: endTime,
            limit: limit);

        if (result.Success && result.Data?.List != null)
        {
            orders = result.Data.List.Select(o => new OrderDto
            {
                Exchange = ExchangeName,
                OrderId = o.OrderId,
                ClientOrderId = o.OrderLinkId,
                Symbol = o.Symbol,
                Side = o.Side == Bybit.Net.Enums.OrderSide.Buy ? Data.Entities.OrderSide.Buy : Data.Entities.OrderSide.Sell,
                Type = MapBybitOrderType(o.OrderType),
                Status = MapBybitOrderStatus(o.OrderStatus),
                TimeInForce = o.TimeInForce?.ToString(),
                Price = o.Price,
                AveragePrice = o.AveragePrice > 0 ? o.AveragePrice : null,
                StopPrice = o.TriggerPrice,
                Quantity = o.Quantity,
                FilledQuantity = o.QuantityFilled ?? 0,
                Fee = o.CumulativeExecFee ?? 0,
                FeeAsset = "USDT",
                CreatedAt = o.CreateTime,
                UpdatedAt = o.UpdateTime,
                WorkingTime = o.CreateTime,
                ReduceOnly = o.ReduceOnly?.ToString(),
                PostOnly = o.TimeInForce == Bybit.Net.Enums.V5.TimeInForce.PostOnly ? "true" : null
            }).ToList();
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error fetching order history from Bybit");
    }

    return orders;
}

public async Task<List<TradeDto>> GetUserTradesAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
{
    if (_restClient == null)
        throw new InvalidOperationException("Not connected to Bybit");

    var trades = new List<TradeDto>();

    try
    {
        var result = await _restClient.V5Api.Trading.GetUserTradesAsync(
            Category.Linear,
            startTime: startTime,
            endTime: endTime,
            limit: limit);

        if (result.Success && result.Data?.List != null)
        {
            trades = result.Data.List.Select(t => new TradeDto
            {
                Exchange = ExchangeName,
                TradeId = t.ExecId,
                OrderId = t.OrderId,
                Symbol = t.Symbol,
                Side = t.Side == Bybit.Net.Enums.OrderSide.Buy ? Data.Entities.OrderSide.Buy : Data.Entities.OrderSide.Sell,
                Price = t.ExecPrice,
                Quantity = t.ExecQuantity,
                QuoteQuantity = t.ExecValue,
                Fee = t.ExecFee,
                FeeAsset = "USDT",
                Commission = t.FeeRate,
                CommissionAsset = "USDT",
                IsMaker = t.IsMaker,
                IsBuyer = t.Side == Bybit.Net.Enums.OrderSide.Buy,
                ExecutedAt = t.ExecTime,
                OrderType = t.OrderType?.ToString(),
                PositionSide = null
            }).ToList();
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error fetching user trades from Bybit");
    }

    return trades;
}

public async Task<List<TransactionDto>> GetTransactionsAsync(DateTime? startTime = null, DateTime? endTime = null, int limit = 100)
{
    if (_restClient == null)
        throw new InvalidOperationException("Not connected to Bybit");

    var transactions = new List<TransactionDto>();

    try
    {
        var result = await _restClient.V5Api.Account.GetTransactionHistoryAsync(
            AccountType.Unified,
            startTime: startTime,
            endTime: endTime,
            limit: limit);

        if (result.Success && result.Data?.List != null)
        {
            transactions = result.Data.List.Select(t => new TransactionDto
            {
                Exchange = ExchangeName,
                TransactionId = t.TransactionId ?? t.TransactionTime.Ticks.ToString(),
                TxHash = t.TransactionId,
                Type = MapBybitTransactionType(t.Type),
                Asset = t.Coin,
                Amount = t.CashFlow,
                Status = Data.Entities.TransactionStatus.Confirmed,
                FromAddress = null,
                ToAddress = null,
                Network = null,
                Info = t.Type,
                Symbol = null,
                TradeId = null,
                Fee = 0,
                FeeAsset = null,
                CreatedAt = t.TransactionTime,
                ConfirmedAt = t.TransactionTime
            }).ToList();
        }
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "Error fetching transactions from Bybit");
    }

    return transactions;
}

// Helper methods
private Data.Entities.OrderType MapBybitOrderType(Bybit.Net.Enums.V5.OrderType type)
{
    return type switch
    {
        Bybit.Net.Enums.V5.OrderType.Market => Data.Entities.OrderType.Market,
        Bybit.Net.Enums.V5.OrderType.Limit => Data.Entities.OrderType.Limit,
        _ => Data.Entities.OrderType.Market
    };
}

private Data.Entities.OrderStatus MapBybitOrderStatus(Bybit.Net.Enums.V5.OrderStatus status)
{
    return status switch
    {
        Bybit.Net.Enums.V5.OrderStatus.New => Data.Entities.OrderStatus.New,
        Bybit.Net.Enums.V5.OrderStatus.PartiallyFilled => Data.Entities.OrderStatus.PartiallyFilled,
        Bybit.Net.Enums.V5.OrderStatus.Filled => Data.Entities.OrderStatus.Filled,
        Bybit.Net.Enums.V5.OrderStatus.Cancelled => Data.Entities.OrderStatus.Canceled,
        Bybit.Net.Enums.V5.OrderStatus.Rejected => Data.Entities.OrderStatus.Rejected,
        _ => Data.Entities.OrderStatus.New
    };
}

private Data.Entities.TransactionType MapBybitTransactionType(string type)
{
    return type.ToLower() switch
    {
        "transfer" => Data.Entities.TransactionType.Transfer,
        "realized_pnl" => Data.Entities.TransactionType.RealizedPnL,
        "funding" => Data.Entities.TransactionType.Funding,
        "commission" => Data.Entities.TransactionType.Commission,
        _ => Data.Entities.TransactionType.Other
    };
}
```

---

## Important Next Steps

Due to the size of this implementation (30+ remaining files), I recommend:

1. **Complete connector implementations first** (Steps 4-5 above)
2. **Test connectors** - Verify they return data correctly
3. **Then continue with remaining phases** in separate sessions:
   - Phase 2: Data Collectors (4 files)
   - Phase 3: Background Services (4 files)
   - Phase 4: Repositories (4 files)
   - Phase 5: API Endpoints (1-2 files)
   - Phase 6: DI Registration & Config (2 files)
   - Phase 7: Frontend Components (6-7 files)
   - Phase 8: State Management & API Services (2 files)

Each phase can be implemented and tested independently.

Would you like me to continue with the next phase in a new conversation, or would you prefer to implement the connectors first and test them?
