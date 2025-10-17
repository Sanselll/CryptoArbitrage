import { create } from 'zustand';
import type { FundingRate, Position, ArbitrageOpportunity, AccountBalance } from '../types/index';

interface ArbitrageState {
  fundingRates: FundingRate[];
  positions: Position[];
  opportunities: ArbitrageOpportunity[];
  balances: AccountBalance[];
  totalPnL: number;
  todayPnL: number;
  isConnected: boolean;

  setFundingRates: (rates: FundingRate[]) => void;
  setPositions: (positions: Position[]) => void;
  setOpportunities: (opportunities: ArbitrageOpportunity[]) => void;
  setBalances: (balances: AccountBalance[]) => void;
  setPnL: (totalPnL: number, todayPnL: number) => void;
  setConnected: (connected: boolean) => void;
}

export const useArbitrageStore = create<ArbitrageState>((set) => ({
  fundingRates: [],
  positions: [],
  opportunities: [],
  balances: [],
  totalPnL: 0,
  todayPnL: 0,
  isConnected: false,

  setFundingRates: (rates) => set({ fundingRates: rates }),
  setPositions: (positions) => set({ positions }),
  setOpportunities: (newOpportunities) => set((state) => {
    // Create a map of existing opportunities by uniqueKey
    const existingMap = new Map(
      state.opportunities.map(opp => [(opp as any).uniqueKey || `${opp.symbol}-${(opp as any).exchange || (opp as any).longExchange}`, opp])
    );

    // Merge new opportunities, updating existing ones while preserving detectedAt timestamp
    newOpportunities.forEach(newOpp => {
      const key = (newOpp as any).uniqueKey || `${newOpp.symbol}-${(newOpp as any).exchange || (newOpp as any).longExchange}`;
      const existing = existingMap.get(key);

      // If opportunity already exists, preserve its original detectedAt timestamp
      if (existing) {
        existingMap.set(key, {
          ...newOpp,
          detectedAt: existing.detectedAt // Preserve original detection time
        });
      } else {
        existingMap.set(key, newOpp);
      }
    });

    return { opportunities: Array.from(existingMap.values()) };
  }),
  setBalances: (balances) => set({ balances }),
  setPnL: (totalPnL, todayPnL) => set({ totalPnL, todayPnL }),
  setConnected: (connected) => set({ isConnected: connected }),
}));
