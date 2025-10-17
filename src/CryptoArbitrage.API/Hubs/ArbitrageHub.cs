using Microsoft.AspNetCore.SignalR;
using CryptoArbitrage.API.Models;

namespace CryptoArbitrage.API.Hubs;

public class ArbitrageHub : Hub
{
    private readonly ILogger<ArbitrageHub> _logger;

    public ArbitrageHub(ILogger<ArbitrageHub> logger)
    {
        _logger = logger;
    }

    public override async Task OnConnectedAsync()
    {
        _logger.LogInformation("Client connected: {ConnectionId}", Context.ConnectionId);
        await base.OnConnectedAsync();
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        _logger.LogInformation("Client disconnected: {ConnectionId}", Context.ConnectionId);
        await base.OnDisconnectedAsync(exception);
    }

    // Methods for broadcasting data to clients
    public async Task SendFundingRates(List<FundingRateDto> fundingRates)
    {
        await Clients.All.SendAsync("ReceiveFundingRates", fundingRates);
    }

    public async Task SendPositions(List<PositionDto> positions)
    {
        await Clients.All.SendAsync("ReceivePositions", positions);
    }

    public async Task SendOpportunities(List<ArbitrageOpportunityDto> opportunities)
    {
        await Clients.All.SendAsync("ReceiveOpportunities", opportunities);
    }

    public async Task SendBalances(List<AccountBalanceDto> balances)
    {
        await Clients.All.SendAsync("ReceiveBalances", balances);
    }

    public async Task SendDashboardData(DashboardDataDto data)
    {
        await Clients.All.SendAsync("ReceiveDashboardData", data);
    }

    public async Task SendPnLUpdate(decimal totalPnL, decimal todayPnL)
    {
        await Clients.All.SendAsync("ReceivePnLUpdate", new { totalPnL, todayPnL });
    }

    public async Task SendAlert(string message, string severity)
    {
        await Clients.All.SendAsync("ReceiveAlert", new { message, severity, timestamp = DateTime.UtcNow });
    }
}
