import { Wallet, DollarSign, PieChart, TrendingUp } from 'lucide-react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { Card, CardContent } from './ui/Card';
import { Badge } from './ui/Badge';

export const BalanceWidget = () => {
  const { balances } = useArbitrageStore();

  const totalBalance = balances.reduce((sum, b) => sum + b.totalBalance, 0);
  const totalAvailable = balances.reduce((sum, b) => sum + b.availableBalance, 0);
  const totalMarginUsed = balances.reduce((sum, b) => sum + b.marginUsed, 0);
  const totalUnrealizedPnL = balances.reduce((sum, b) => sum + b.unrealizedPnL, 0);

  const marginUtilization = totalBalance > 0 ? (totalMarginUsed / totalBalance) * 100 : 0;
  const availablePercent = totalBalance > 0 ? (totalAvailable / totalBalance) * 100 : 0;
  const pnlPercent = totalBalance > 0 ? (totalUnrealizedPnL / totalBalance) * 100 : 0;

  const getMarginRiskLevel = (utilization: number) => {
    if (utilization > 75) return { label: 'High Risk', variant: 'danger' as const };
    if (utilization > 50) return { label: 'Medium', variant: 'warning' as const };
    return { label: 'Low Risk', variant: 'success' as const };
  };

  const riskLevel = getMarginRiskLevel(marginUtilization);

  return (
    <div className="grid grid-cols-4 gap-3">
      {/* Total Balance */}
      <Card className="card-hover">
        <CardContent className="p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-binance-text-secondary font-medium">Total Balance</span>
            <Wallet className="w-4 h-4 text-binance-text-muted" />
          </div>
          <div className="text-lg font-bold text-binance-text mb-1 font-mono">
            ${totalBalance.toFixed(2)}
          </div>
          <div className="text-xs text-binance-text-muted">≈ {totalBalance.toFixed(2)} USDT</div>
        </CardContent>
      </Card>

      {/* Available Balance */}
      <Card className="card-hover">
        <CardContent className="p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-binance-text-secondary font-medium">Available</span>
            <DollarSign className="w-4 h-4 text-binance-green" />
          </div>
          <div className="text-lg font-bold text-binance-text mb-2 font-mono">
            ${totalAvailable.toFixed(2)}
          </div>
          {/* Progress bar */}
          <div className="space-y-1">
            <div className="w-full bg-binance-bg-tertiary rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-binance-green h-full transition-all duration-300"
                style={{ width: `${Math.min(availablePercent, 100)}%` }}
              />
            </div>
            <div className="text-xs text-binance-text-muted">
              {availablePercent.toFixed(1)}% of total
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Margin Used */}
      <Card className="card-hover">
        <CardContent className="p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-binance-text-secondary font-medium">Margin Used</span>
            <div className="flex items-center gap-1">
              <PieChart className="w-3.5 h-3.5 text-binance-yellow" />
              <Badge variant={riskLevel.variant} size="sm">
                {riskLevel.label}
              </Badge>
            </div>
          </div>
          <div className="text-lg font-bold text-binance-text mb-2 font-mono">
            ${totalMarginUsed.toFixed(2)}
          </div>
          {/* Progress bar with color gradient based on risk */}
          <div className="space-y-1">
            <div className="w-full bg-binance-bg-tertiary rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-full transition-all duration-300 ${
                  marginUtilization > 75
                    ? 'bg-binance-red'
                    : marginUtilization > 50
                    ? 'bg-binance-yellow'
                    : 'bg-binance-green'
                }`}
                style={{ width: `${Math.min(marginUtilization, 100)}%` }}
              />
            </div>
            <div className="text-xs text-binance-text-muted">
              {marginUtilization.toFixed(1)}% utilized
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Unrealized P&L */}
      <Card className="card-hover">
        <CardContent className="p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-binance-text-secondary font-medium">Unrealized P&L</span>
            <TrendingUp
              className={`w-4 h-4 ${
                totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
              }`}
            />
          </div>
          <div
            className={`text-lg font-bold mb-1 font-mono ${
              totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
            }`}
          >
            {totalUnrealizedPnL >= 0 ? '+' : ''}${totalUnrealizedPnL.toFixed(2)}
          </div>
          <div
            className={`text-xs font-medium ${
              totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
            }`}
          >
            {totalUnrealizedPnL >= 0 ? '+' : ''}
            {pnlPercent.toFixed(2)}% ROE
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
