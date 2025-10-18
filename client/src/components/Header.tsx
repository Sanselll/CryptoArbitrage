import { Activity, TrendingUp, TrendingDown, Wifi, WifiOff } from 'lucide-react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { Badge } from './ui/Badge';

export const Header = () => {
  const { totalPnL, todayPnL, isConnected } = useArbitrageStore();

  const totalPnLPercent = 0; // Calculate based on initial capital if available
  const todayPnLPercent = 0;

  return (
    <header className="h-10 bg-binance-bg-secondary border-b border-binance-border px-4 flex items-center justify-between">
      {/* Left Section - Branding & Status */}
      <div className="flex items-center gap-4">
        {/* Logo/Brand */}
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-binance-yellow" />
          <h1 className="text-sm font-bold text-binance-yellow">Crypto Arbitrage</h1>
        </div>

        {/* Connection Status Badge */}
        <Badge
          variant={isConnected ? 'success' : 'danger'}
          size="sm"
          className="gap-1"
        >
          {isConnected ? (
            <Wifi className="w-3 h-3" />
          ) : (
            <WifiOff className="w-3 h-3" />
          )}
          {isConnected ? 'Live' : 'Offline'}
        </Badge>
      </div>

      {/* Right Section - P&L Stats */}
      <div className="flex items-center gap-6">
        {/* Total P&L */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-binance-text-secondary font-medium">Total P&L</span>
          <div className="flex items-center gap-1">
            {totalPnL >= 0 ? (
              <TrendingUp className="w-3.5 h-3.5 text-binance-green" />
            ) : (
              <TrendingDown className="w-3.5 h-3.5 text-binance-red" />
            )}
            <span
              className={`text-sm font-bold font-mono ${
                totalPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
              }`}
            >
              {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Divider */}
        <div className="w-px h-6 bg-binance-border"></div>

        {/* Today P&L */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-binance-text-secondary font-medium">Today P&L</span>
          <div className="flex items-center gap-1">
            {todayPnL >= 0 ? (
              <TrendingUp className="w-3.5 h-3.5 text-binance-green" />
            ) : (
              <TrendingDown className="w-3.5 h-3.5 text-binance-red" />
            )}
            <span
              className={`text-sm font-bold font-mono ${
                todayPnL >= 0 ? 'text-binance-green' : 'text-binance-red'
              }`}
            >
              {todayPnL >= 0 ? '+' : ''}${todayPnL.toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
