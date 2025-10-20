import { create } from 'zustand';
import type { FundingRate, Position, ArbitrageOpportunity, AccountBalance } from '../types/index';
import { signalRService } from '../services/signalRService';

interface ArbitrageState {
  fundingRates: FundingRate[];
  positions: Position[];
  opportunities: ArbitrageOpportunity[];
  balances: AccountBalance[];
  totalPnL: number;
  todayPnL: number;
  isConnected: boolean;
  unsubscribe: (() => void)[];

  setFundingRates: (rates: FundingRate[]) => void;
  setPositions: (positions: Position[]) => void;
  setOpportunities: (opportunities: ArbitrageOpportunity[]) => void;
  setBalances: (balances: AccountBalance[]) => void;
  setPnL: (totalPnL: number, todayPnL: number) => void;
  setConnected: (connected: boolean) => void;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  reset: () => void;
}

export const useArbitrageStore = create<ArbitrageState>((set, get) => ({
  fundingRates: [],
  positions: [],
  opportunities: [],
  balances: [],
  totalPnL: 0,
  todayPnL: 0,
  isConnected: false,
  unsubscribe: [],

  setFundingRates: (rates) => set({ fundingRates: rates }),
  setPositions: (positions) => set({ positions }),
  setOpportunities: (opportunities) => set({ opportunities }),
  setBalances: (balances) => set({ balances }),
  setPnL: (totalPnL, todayPnL) => set({ totalPnL, todayPnL }),
  setConnected: (connected) => set({ isConnected: connected }),

  connect: async () => {
    try {
      // Connect to SignalR
      await signalRService.connect();

      // Subscribe to all events and store cleanup functions
      const state = get();
      const cleanupFunctions = [
        signalRService.onFundingRates((data) => {
          useArbitrageStore.getState().setFundingRates(data);
        }),
        signalRService.onOpportunities((data) => {
          useArbitrageStore.getState().setOpportunities(data);
        }),
        signalRService.onPositions((data) => {
          useArbitrageStore.getState().setPositions(data);
        }),
        signalRService.onBalances((data) => {
          useArbitrageStore.getState().setBalances(data);
        }),
        signalRService.onPnLUpdate((data) => {
          useArbitrageStore.getState().setPnL(data.totalPnL, data.todayPnL);
        }),
      ];

      set({ isConnected: true, unsubscribe: cleanupFunctions });
    } catch (error) {
      console.error('Failed to connect to SignalR:', error);
      set({ isConnected: false });
    }
  },

  disconnect: async () => {
    // Unsubscribe from all events
    const state = get();
    state.unsubscribe.forEach(cleanup => cleanup());

    // Disconnect from SignalR
    signalRService.disconnect();

    set({ isConnected: false, unsubscribe: [] });
  },

  reset: () => {
    // Disconnect first if connected
    const state = get();
    if (state.isConnected) {
      state.unsubscribe.forEach(cleanup => cleanup());
      signalRService.disconnect();
    }

    // Reset all state to initial values
    set({
      fundingRates: [],
      positions: [],
      opportunities: [],
      balances: [],
      totalPnL: 0,
      todayPnL: 0,
      isConnected: false,
      unsubscribe: [],
    });
  },
}));
