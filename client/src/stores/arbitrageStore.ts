import { create } from 'zustand';
import type { FundingRate, Position, ArbitrageOpportunity, AccountBalance, Order, Trade, Transaction } from '../types/index';
import { signalRService } from '../services/signalRService';

interface ArbitrageState {
  fundingRates: FundingRate[];
  positions: Position[];
  opportunities: ArbitrageOpportunity[];
  balances: AccountBalance[];
  openOrders: Order[];
  orderHistory: Order[];
  tradeHistory: Trade[];
  transactionHistory: Transaction[];
  totalPnL: number;
  todayPnL: number;
  isConnected: boolean;
  unsubscribe: (() => void)[];

  setFundingRates: (rates: FundingRate[]) => void;
  setPositions: (positions: Position[]) => void;
  setOpportunities: (opportunities: ArbitrageOpportunity[]) => void;
  setBalances: (balances: AccountBalance[]) => void;
  setOpenOrders: (orders: Order[]) => void;
  setOrderHistory: (orders: Order[]) => void;
  setTradeHistory: (trades: Trade[]) => void;
  setTransactionHistory: (transactions: Transaction[]) => void;
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
  openOrders: [],
  orderHistory: [],
  tradeHistory: [],
  transactionHistory: [],
  totalPnL: 0,
  todayPnL: 0,
  isConnected: false,
  unsubscribe: [],

  setFundingRates: (rates) => set({ fundingRates: rates }),
  setPositions: (positions) => set({ positions }),
  setOpportunities: (opportunities) => set({ opportunities }),
  setBalances: (balances) => set({ balances }),

  // Merge open orders instead of replacing (using exchange+orderId as unique key)
  setOpenOrders: (orders) => set((state) => {
    const existingMap = new Map(
      state.openOrders.map(o => [`${o.exchange}-${o.orderId}`, o])
    );
    orders.forEach(o => {
      existingMap.set(`${o.exchange}-${o.orderId}`, o);
    });
    return { openOrders: Array.from(existingMap.values()) };
  }),

  // Merge order history instead of replacing (using exchange+orderId as unique key)
  setOrderHistory: (orders) => set((state) => {
    const existingMap = new Map(
      state.orderHistory.map(o => [`${o.exchange}-${o.orderId}`, o])
    );
    orders.forEach(o => {
      existingMap.set(`${o.exchange}-${o.orderId}`, o);
    });
    return { orderHistory: Array.from(existingMap.values()) };
  }),

  // Merge trade history instead of replacing (using exchange+tradeId as unique key)
  setTradeHistory: (trades) => set((state) => {
    const existingMap = new Map(
      state.tradeHistory.map(t => [`${t.exchange}-${t.tradeId}`, t])
    );
    trades.forEach(t => {
      existingMap.set(`${t.exchange}-${t.tradeId}`, t);
    });
    return { tradeHistory: Array.from(existingMap.values()) };
  }),

  // Merge transaction history instead of replacing (using exchange+transactionId as unique key)
  setTransactionHistory: (transactions) => set((state) => {
    const existingMap = new Map(
      state.transactionHistory.map(t => [`${t.exchange}-${t.transactionId}`, t])
    );
    transactions.forEach(t => {
      existingMap.set(`${t.exchange}-${t.transactionId}`, t);
    });
    return { transactionHistory: Array.from(existingMap.values()) };
  }),

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
        signalRService.onOpenOrders((data) => {
          useArbitrageStore.getState().setOpenOrders(data);
        }),
        signalRService.onOrderHistory((data) => {
          useArbitrageStore.getState().setOrderHistory(data);
        }),
        signalRService.onTradeHistory((data) => {
          useArbitrageStore.getState().setTradeHistory(data);
        }),
        signalRService.onTransactionHistory((data) => {
          useArbitrageStore.getState().setTransactionHistory(data);
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
      openOrders: [],
      orderHistory: [],
      tradeHistory: [],
      transactionHistory: [],
      totalPnL: 0,
      todayPnL: 0,
      isConnected: false,
      unsubscribe: [],
    });
  },
}));
