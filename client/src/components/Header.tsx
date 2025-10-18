import { Activity, TrendingUp, TrendingDown, Wifi, WifiOff, User, Settings, LogOut } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { useAuthStore } from '../stores/authStore';
import { Badge } from './ui/Badge';

export const Header = () => {
  const { totalPnL, todayPnL, isConnected, disconnect } = useArbitrageStore();
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();
  const [showMenu, setShowMenu] = useState(false);

  const handleLogout = () => {
    disconnect();
    logout();
    navigate('/login');
  };

  const handleProfile = () => {
    navigate('/profile');
    setShowMenu(false);
  };

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

      {/* Right Section - P&L Stats & Profile */}
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

        {/* Divider */}
        <div className="w-px h-6 bg-binance-border"></div>

        {/* Profile Menu */}
        <div className="relative">
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="flex items-center gap-1.5 hover:bg-binance-bg-hover px-2 py-1 rounded transition-colors"
          >
            <User className="w-3 h-3 text-binance-text-secondary" />
            <span className="text-xs text-binance-text-secondary max-w-[150px] truncate">{user?.email}</span>
          </button>

          {showMenu && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setShowMenu(false)}
              />
              <div className="absolute right-0 mt-1 w-40 bg-binance-bg-secondary border border-binance-border rounded shadow-xl z-20">
                <button
                  onClick={handleProfile}
                  className="w-full flex items-center gap-1.5 px-3 py-1.5 text-xs text-binance-text hover:bg-binance-bg-tertiary transition-colors"
                >
                  <Settings className="w-3 h-3" />
                  API Keys
                </button>
                <button
                  onClick={handleLogout}
                  className="w-full flex items-center gap-1.5 px-3 py-1.5 text-xs text-binance-red hover:bg-binance-bg-tertiary transition-colors border-t border-binance-border"
                >
                  <LogOut className="w-3 h-3" />
                  Logout
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </header>
  );
};
