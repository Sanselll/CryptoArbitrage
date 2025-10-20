import { TrendingUp, Wallet, Lock, Settings } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { Card, CardContent } from './ui/Card';
import { Badge } from './ui/Badge';
import { ExchangeBadge } from './ui/ExchangeBadge';
import { Button } from './ui/Button';

interface BalanceWidgetProps {
  supportedExchanges: string[];
  connectedExchanges: string[];
}

export const BalanceWidget = ({ supportedExchanges, connectedExchanges }: BalanceWidgetProps) => {
  const { balances } = useArbitrageStore();
  const navigate = useNavigate();

  const getMarginRiskLevel = (utilization: number) => {
    if (utilization > 75) return { label: 'High Risk', variant: 'danger' as const };
    if (utilization > 50) return { label: 'Medium', variant: 'warning' as const };
    return { label: 'Low Risk', variant: 'success' as const };
  };

  return (
    <div className="flex gap-2">
      {supportedExchanges.map((exchange) => {
        // Check if this exchange has API keys connected
        const hasKeys = connectedExchanges.includes(exchange);

        // Find balance data for this exchange if keys are connected
        const balance = balances.find((b) => b.exchange === exchange);

        // If no keys, show placeholder card
        if (!hasKeys) {
          return (
            <Card key={exchange} className="card-hover w-[240px]">
              <CardContent className="p-2 flex flex-col items-center justify-center min-h-[150px]">
                <ExchangeBadge exchange={exchange} className="mb-3" />
                <div className="text-center mb-3">
                  <Settings className="w-8 h-8 text-binance-text-secondary mx-auto mb-2" />
                  <p className="text-xs text-binance-text-secondary mb-1">
                    No API keys configured
                  </p>
                </div>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => navigate('/profile')}
                  className="gap-1 text-xs"
                >
                  <Settings className="w-3 h-3" />
                  Configure Keys
                </Button>
              </CardContent>
            </Card>
          );
        }

        // If keys connected but no balance data yet, show loading state
        if (!balance) {
          return (
            <Card key={exchange} className="card-hover w-[240px]">
              <CardContent className="p-2 flex items-center justify-center min-h-[150px]">
                <div className="text-xs text-binance-text-secondary">Loading...</div>
              </CardContent>
            </Card>
          );
        }

        // Show balance data (existing logic)
        // Total capital = all assets (spot + futures)
        const totalCapital = balance.totalBalance || 0;
        const totalUnrealizedPnL = balance.unrealizedPnL;

        const spotBalanceUsd = balance.spotBalanceUsd || 0;
        const spotAvailableUsd = balance.spotAvailableUsd || 0;
        const futuresBalance = balance.futuresBalanceUsd || 0;
        const futuresAvailable = balance.futuresAvailableUsd || 0;
        const totalMarginUsed = balance.marginUsed;

        // Available Capital = USDT available to deploy in new trades
        // Binance: Spot Available USDT + Futures Available USDT
        // Bybit: Available USDT (AvailableToWithdraw from unified account)
        const availableCapital = balance.exchange === 'Binance'
          ? spotAvailableUsd + futuresAvailable
          : futuresAvailable;

        // In Positions = coins locked in spot positions + margin used in futures
        // spotBalanceUsd - spotAvailableUsd = coins locked in spot
        const coinsInSpotPositions = spotBalanceUsd - spotAvailableUsd;
        const inPositions = coinsInSpotPositions + totalMarginUsed;

        // Margin % (only when positions exist)
        const marginUtilization = futuresBalance > 0 ? (totalMarginUsed / futuresBalance) * 100 : 0;
        const hasPositions = totalMarginUsed > 0 || coinsInSpotPositions > 0;
        const riskLevel = getMarginRiskLevel(marginUtilization);

        const pnlPercent = totalCapital > 0 ? (totalUnrealizedPnL / totalCapital) * 100 : 0;

        return (
          <Card key={exchange} className="card-hover w-[240px]">
            <CardContent className="p-2">
              {/* Exchange Header with P&L */}
              <div className="flex items-center justify-between mb-1.5 pb-1 border-b border-binance-border/30">
                <ExchangeBadge exchange={exchange} />
                <div className="flex items-center gap-1.5">
                  <TrendingUp
                    className={`w-2.5 h-2.5 ${
                      totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                    }`}
                  />
                  <div
                    className={`text-xs font-bold font-mono ${
                      totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                    }`}
                  >
                    {totalUnrealizedPnL >= 0 ? '+' : ''}${totalUnrealizedPnL.toFixed(2)}
                  </div>
                  <div
                    className={`text-[9px] font-medium ${
                      totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                    }`}
                  >
                    ({totalUnrealizedPnL >= 0 ? '+' : ''}{pnlPercent.toFixed(1)}%)
                  </div>
                </div>
              </div>

              {/* Capital Total */}
              <div className="mb-1.5 pb-1 border-b border-binance-border/30">
                <div className="text-[9px] text-binance-text-secondary font-bold mb-0.5">Capital</div>
                <div className="text-base font-bold text-binance-text font-mono">
                  ${totalCapital.toFixed(0)}
                </div>
              </div>

              {/* Column Headers */}
              <div className="grid grid-cols-2 gap-2 mb-1">
                <div className="text-[9px] text-binance-text-secondary font-bold">Available Capital</div>
                <div className="text-[9px] text-binance-text-secondary font-bold">In Positions</div>
              </div>

              {/* Grid Layout with 2 columns */}
              <div className="grid grid-cols-2 gap-2">
                {/* Column 1: Available Capital */}
                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <Wallet className="w-2.5 h-2.5 text-binance-text-secondary" />
                    <span className="text-sm font-bold text-binance-green font-mono">
                      ${availableCapital.toFixed(0)}
                    </span>
                  </div>
                  <div className="text-[9px] text-binance-text-muted">
                    Free to trade
                  </div>
                </div>

                {/* Column 2: In Positions */}
                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <Lock className="w-2.5 h-2.5 text-binance-text-secondary" />
                    <span className="text-sm font-bold text-binance-text font-mono">
                      ${inPositions.toFixed(0)}
                    </span>
                  </div>
                  {hasPositions ? (
                    <div className="flex items-center gap-1">
                      <span className="text-[9px] text-binance-text-secondary">Margin</span>
                      <Badge variant={riskLevel.variant} size="sm" className="text-[8px] px-1 py-0">
                        {marginUtilization.toFixed(0)}%
                      </Badge>
                    </div>
                  ) : (
                    <div className="text-[9px] text-binance-text-muted">No positions</div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
};
