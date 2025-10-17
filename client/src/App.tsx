import { useEffect } from 'react';
import { signalRService } from './services/signalRService';
import { useArbitrageStore } from './stores/arbitrageStore';
import { Header } from './components/Header';
import { BalanceWidget } from './components/BalanceWidget';
import { OpportunitiesList } from './components/OpportunitiesList';
import { PositionsGrid } from './components/PositionsGrid';

function App() {
  const {
    setFundingRates,
    setPositions,
    setOpportunities,
    setBalances,
    setPnL,
    setConnected,
  } = useArbitrageStore();

  useEffect(() => {
    // Connect to SignalR
    signalRService.connect();

    // Setup event listeners
    const unsubFundingRates = signalRService.onFundingRates((data) => {
      setFundingRates(data);
    });

    const unsubPositions = signalRService.onPositions((data) => {
      setPositions(data);
    });

    const unsubOpportunities = signalRService.onOpportunities((data) => {
      setOpportunities(data);
    });

    const unsubBalances = signalRService.onBalances((data) => {
      setBalances(data);
    });

    const unsubPnL = signalRService.onPnLUpdate((data) => {
      setPnL(data.totalPnL, data.todayPnL);
    });

    const unsubAlert = signalRService.onAlert((data) => {
      console.log('Alert:', data);
      // You could integrate a toast notification library here
    });

    // Check connection status periodically
    const connectionCheck = setInterval(() => {
      const state = signalRService.getConnectionState();
      setConnected(state === 'Connected');
    }, 1000);

    return () => {
      unsubFundingRates();
      unsubPositions();
      unsubOpportunities();
      unsubBalances();
      unsubPnL();
      unsubAlert();
      clearInterval(connectionCheck);
      signalRService.disconnect();
    };
  }, [setFundingRates, setPositions, setOpportunities, setBalances, setPnL, setConnected]);

  return (
    <div className="h-screen flex flex-col bg-binance-bg">
      <Header />

      <main className="flex-1 overflow-hidden p-4">
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
      </main>
    </div>
  );
}

export default App;
