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
              <CardContent className="p-1.5 flex flex-col items-center justify-center min-h-[75px]">
                <ExchangeBadge exchange={exchange} className="mb-2" />
                <div className="text-center mb-2">
                  <Settings className="w-6 h-6 text-binance-text-secondary mx-auto mb-1" />
                  <p className="text-[10px] text-binance-text-secondary mb-1">
                    No API keys configured
                  </p>
                </div>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => navigate('/profile')}
                  className="gap-1 text-[10px]"
                >
                  <Settings className="w-2.5 h-2.5" />
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
              <CardContent className="p-1.5 flex items-center justify-center min-h-[75px]">
                <div className="text-[10px] text-binance-text-secondary">Loading...</div>
              </CardContent>
            </Card>
          );
        }

        // Show balance data (new compact logic)
        const totalUnrealizedPnL = balance.unrealizedPnL;

        const spotBalanceUsd = balance.spotBalanceUsd || 0;
        const spotAvailableUsd = balance.spotAvailableUsd || 0;
        const spotAssets = spotBalanceUsd - spotAvailableUsd; // Other currencies in USD
        const futuresBalance = balance.futuresBalanceUsd || 0;
        const futuresAvailable = balance.futuresAvailableUsd || 0;
        const totalMarginUsed = balance.marginUsed;

        // Margin % of futures balance
        const marginUtilization = futuresBalance > 0 ? (totalMarginUsed / futuresBalance) * 100 : 0;
        const riskLevel = getMarginRiskLevel(marginUtilization);

        const pnlPercent = futuresBalance > 0 ? (totalUnrealizedPnL / futuresBalance) * 100 : 0;

        return (
          <Card key={exchange} className="card-hover w-[240px]">
            <CardContent className="p-1.5">
              {/* Exchange Header with P&L */}
              <div className="flex items-center justify-between mb-1 pb-0.5 border-b border-binance-border/30">
                <ExchangeBadge exchange={exchange} />
                <div className="flex items-center gap-1">
                  <TrendingUp
                    className={`w-2.5 h-2.5 ${
                      totalUnrealizedPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
                    }`}
                  />
                  <div
                    className={`text-[10px] font-bold font-mono ${
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

              {/* Two Column Layout: SPOT | FUTURES */}
              <div className="grid grid-cols-2 gap-2 mt-1">
                {/* SPOT Column */}
                <div className="border-r border-binance-border/30 pr-2">
                  <div className="text-[9px] text-binance-text-secondary font-bold mb-0.5">SPOT</div>
                  <div className="grid grid-cols-2 gap-1">
                    <div>
                      <div className="text-[8px] text-binance-text-muted">USDT</div>
                      <div className="text-[11px] font-bold text-binance-text font-mono">
                        ${spotAvailableUsd.toFixed(0)}
                      </div>
                    </div>
                    <div>
                      <div className="text-[8px] text-binance-text-muted">Assets</div>
                      <div className="text-[11px] font-bold text-binance-text font-mono">
                        ${spotAssets.toFixed(0)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* FUTURES Column */}
                <div className="pl-2">
                  <div className="text-[9px] text-binance-text-secondary font-bold mb-0.5">FUTURES</div>
                  <div className="grid grid-cols-2 gap-1">
                    <div>
                      <div className="text-[8px] text-binance-text-muted">Available</div>
                      <div className="text-[11px] font-bold text-binance-green font-mono">
                        ${futuresAvailable.toFixed(0)}
                      </div>
                    </div>
                    <div>
                      <div className="text-[8px] text-binance-text-muted">Margin</div>
                      <div className="flex items-center gap-0.5">
                        <div className="text-[11px] font-bold text-binance-text font-mono">
                          ${totalMarginUsed.toFixed(0)}
                        </div>
                        {totalMarginUsed > 0 && (
                          <Badge variant={riskLevel.variant} size="sm" className="text-[7px] px-0.5 py-0">
                            {marginUtilization.toFixed(0)}%
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
};
