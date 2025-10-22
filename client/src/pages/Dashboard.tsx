import { useEffect, useState } from 'react';
import { useArbitrageStore } from '../stores/arbitrageStore';
import { Header } from '../components/Header';
import { BalanceWidget } from '../components/BalanceWidget';
import { OpportunitiesList } from '../components/OpportunitiesList';
import { TradingDataTabs } from '../components/TradingDataTabs';
import apiClient from '../services/apiClient';
import axios from 'axios';

export function Dashboard() {
  const connect = useArbitrageStore((state) => state.connect);
  const disconnect = useArbitrageStore((state) => state.disconnect);
  const [connectedExchanges, setConnectedExchanges] = useState<string[]>([]);
  const [supportedExchanges, setSupportedExchanges] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isBackendOffline, setIsBackendOffline] = useState(false);

  useEffect(() => {
    // Fetch both supported exchanges and user's connected exchanges
    const fetchExchangeData = async () => {
      try {
        // Fetch supported exchanges from configuration
        const supportedResponse = await apiClient.get('/environment/exchanges');
        const supported = supportedResponse.data.exchanges || [];
        setSupportedExchanges(supported);

        // Fetch user's connected exchanges
        const userKeysResponse = await apiClient.get('/user/apikeys');
        const connected = userKeysResponse.data
          .filter((key: any) => key.isEnabled)
          .map((key: any) => key.exchangeName);
        setConnectedExchanges(connected);
        setIsBackendOffline(false);
      } catch (error) {
        console.error('Error fetching exchange data:', error);

        // Check if it's a network error (backend is down)
        if (axios.isAxiosError(error) && (!error.response || error.code === 'ERR_NETWORK')) {
          setIsBackendOffline(true);
        } else {
          // Other error - user likely has no API keys, but we still have supported exchanges
          setConnectedExchanges([]);
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchExchangeData();
  }, []);

  useEffect(() => {
    // Always connect to SignalR (for opportunities data)
    // Connect even without API keys so user can see available opportunities
    if (!isBackendOffline) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [isBackendOffline, connect, disconnect]);

  return (
    <div className="h-screen flex flex-col bg-binance-bg">
      <Header />

      <main className="flex-1 overflow-hidden p-4">
        {isLoading ? (
          // Loading state
          <div className="h-full flex items-center justify-center">
            <div className="text-binance-text-secondary text-sm">Loading...</div>
          </div>
        ) : (
          // Always show dashboard - BalanceWidget will handle display of exchanges with/without keys
          <div className="h-full flex flex-col gap-3">
            {/* Balance Overview - Top Section */}
            <div className="flex-shrink-0">
              <BalanceWidget
                supportedExchanges={supportedExchanges}
                connectedExchanges={connectedExchanges}
              />
            </div>

            {/* Opportunities List - Full Width */}
            <div className="flex-1 min-h-0">
              <OpportunitiesList />
            </div>

            {/* Trading Data Tabs - Bottom Section */}
            <div className="flex-shrink-0 h-80">
              <TradingDataTabs />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
