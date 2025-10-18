import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { Header } from '../components/Header';
import { BalanceWidget } from '../components/BalanceWidget';
import { OpportunitiesList } from '../components/OpportunitiesList';
import { PositionsGrid } from '../components/PositionsGrid';
import { Shield, Settings, ArrowRight } from 'lucide-react';
import apiClient from '../services/apiClient';
import { Button } from '../components/ui/Button';

export function Dashboard() {
  const connect = useArbitrageStore((state) => state.connect);
  const disconnect = useArbitrageStore((state) => state.disconnect);
  const navigate = useNavigate();
  const [hasApiKeys, setHasApiKeys] = useState<boolean | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user has API keys configured
    const checkApiKeys = async () => {
      try {
        const response = await apiClient.get('/user/apikeys');
        setHasApiKeys(response.data.length > 0);
      } catch (error) {
        console.error('Error checking API keys:', error);
        setHasApiKeys(false);
      } finally {
        setIsLoading(false);
      }
    };

    checkApiKeys();
  }, []);

  useEffect(() => {
    // Only connect to SignalR if user has API keys
    if (hasApiKeys) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [hasApiKeys, connect, disconnect]);

  return (
    <div className="h-screen flex flex-col bg-binance-bg">
      <Header />

      <main className="flex-1 overflow-hidden p-4">
        {isLoading ? (
          // Loading state
          <div className="h-full flex items-center justify-center">
            <div className="text-binance-text-secondary text-sm">Loading...</div>
          </div>
        ) : !hasApiKeys ? (
          // Empty state - No API keys configured
          <div className="h-full flex items-center justify-center">
            <div className="max-w-md w-full text-center">
              <div className="bg-binance-bg-secondary border border-binance-border rounded-lg p-8">
                <div className="flex justify-center mb-6">
                  <div className="w-20 h-20 rounded-full bg-binance-yellow/10 flex items-center justify-center">
                    <Shield className="w-10 h-10 text-binance-yellow" />
                  </div>
                </div>

                <h2 className="text-xl font-bold text-binance-text mb-3">
                  Welcome to Crypto Arbitrage
                </h2>

                <p className="text-sm text-binance-text-secondary mb-6 leading-relaxed">
                  To start finding arbitrage opportunities and trading, you need to configure your exchange API keys first.
                </p>

                <div className="bg-binance-bg-tertiary border border-binance-border rounded p-4 mb-6 text-left">
                  <h3 className="text-xs font-semibold text-binance-text mb-2 flex items-center gap-2">
                    <Settings className="w-3 h-3" />
                    What you'll need:
                  </h3>
                  <ul className="text-xs text-binance-text-secondary space-y-1.5 list-disc list-inside">
                    <li>API keys from Binance and/or Bybit</li>
                    <li>Keys should have trading permissions enabled</li>
                    <li>Your keys are encrypted and stored securely</li>
                  </ul>
                </div>

                <Button
                  variant="primary"
                  size="lg"
                  onClick={() => navigate('/profile')}
                  className="w-full gap-2 text-sm font-semibold"
                >
                  Configure API Keys
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
        ) : (
          // Normal dashboard with data
          <div className="h-full flex flex-col gap-3">
            {/* Balance Overview - Top Section */}
            <div className="flex-shrink-0">
              <BalanceWidget />
            </div>

            {/* Opportunities List - Full Width */}
            <div className="flex-1 min-h-0">
              <OpportunitiesList />
            </div>

            {/* Positions Grid - Bottom Section */}
            <div className="flex-shrink-0 h-80">
              <PositionsGrid />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
