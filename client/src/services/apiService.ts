import type { Position } from '../types/index';

const API_BASE_URL = 'http://localhost:5052';

interface ExecuteOpportunityRequest {
  symbol: string;
  strategy: number; // 1 = SpotPerpetual, 2 = CrossExchange
  exchange: string;
  longExchange?: string;
  shortExchange?: string;
  positionSizeUsd: number;
  leverage: number;
  stopLossPercentage?: number;
  takeProfitPercentage?: number;
}

interface ExecuteOpportunityResponse {
  success: boolean;
  message?: string;
  positionIds: number[];
  orderIds: string[];
  totalPositionSize: number;
  errorMessage?: string;
}

interface CloseOpportunityResponse {
  success: boolean;
  message?: string;
  activeOpportunityId: number;
  closedPositionIds: number[];
  finalPnL: number;
  netFundingFees: number;
  errorMessage?: string;
}

interface ExecutionBalances {
  exchange: string;
  spotUsdtAvailable: number;
  futuresAvailable: number;
  totalAvailable: number;
  isUnifiedAccount: boolean;
  marginUsagePercent: number;
  maxPositionSize: number;
}

export const apiService = {
  async getPositions(): Promise<any[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/position`);

      if (!response.ok) {
        throw new Error('Failed to fetch positions');
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching positions:', error);
      throw error;
    }
  },

  async executeOpportunity(request: ExecuteOpportunityRequest): Promise<ExecuteOpportunityResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/opportunity/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.errorMessage || 'Execution failed');
      }

      return data;
    } catch (error) {
      console.error('Error executing opportunity:', error);
      throw error;
    }
  },

  async closeOpportunity(activeOpportunityId: number): Promise<CloseOpportunityResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/opportunity/close/${activeOpportunityId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.errorMessage || 'Close failed');
      }

      return data;
    } catch (error) {
      console.error('Error closing opportunity:', error);
      throw error;
    }
  },

  async stopExecution(executionId: number): Promise<CloseOpportunityResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/opportunity/stop/${executionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.errorMessage || 'Stop execution failed');
      }

      return data;
    } catch (error) {
      console.error('Error stopping execution:', error);
      throw error;
    }
  },

  async getExecutionBalances(exchange: string, maxLeverage: number = 5): Promise<ExecutionBalances> {
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/opportunity/execution-balances?exchange=${encodeURIComponent(exchange)}&maxLeverage=${maxLeverage}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch execution balances');
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching execution balances:', error);
      throw error;
    }
  },
};

export type { ExecuteOpportunityRequest, ExecuteOpportunityResponse, CloseOpportunityResponse, ExecutionBalances };
