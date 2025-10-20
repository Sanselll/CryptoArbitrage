import { TrendingUp, TrendingDown, Wifi, WifiOff, User, Settings, LogOut } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { useAuthStore } from '../stores/authStore';
import { Badge } from './ui/Badge';
import apiClient from '../services/apiClient';

export const Header = () => {
  const { totalPnL, todayPnL, isConnected, disconnect } = useArbitrageStore();
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();
  const [showMenu, setShowMenu] = useState(false);
  const [environmentMode, setEnvironmentMode] = useState<string>('Demo');

  useEffect(() => {
    // Get mode from sessionStorage instead of API call
    const mode = sessionStorage.getItem('trading_mode') || 'Demo';
    setEnvironmentMode(mode);
  }, []);

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
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="url(#activity-gradient)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-4 h-4"
          >
            <defs>
              <linearGradient id="activity-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style={{ stopColor: '#FFFF00', stopOpacity: 1 }} />
                <stop offset="100%" style={{ stopColor: '#FFA500', stopOpacity: 1 }} />
              </linearGradient>
            </defs>
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
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

        {/* Environment Mode Badge */}
        <Badge
          variant={environmentMode === 'Real' ? 'warning' : 'info'}
          size="sm"
        >
          {environmentMode === 'Real' ? 'Real Trading' : 'Demo Trading'}
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
