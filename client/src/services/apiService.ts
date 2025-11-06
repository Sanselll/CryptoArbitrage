import type { Position } from '../types/index';
import { getAuthToken } from './authUtils';
import { getApiBaseUrl } from './apiClient';

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

interface UserApiKey {
  id: number;
  exchangeName: string;
  isEnabled: boolean;
}

// Agent types
interface AgentConfig {
  maxLeverage: number;
  targetUtilization: number;
  maxPositions: number;
  predictionIntervalSeconds: number;
}

interface AgentStats {
  totalDecisions: number;
  holdDecisions: number;
  enterDecisions: number;
  exitDecisions: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  sessionPnLUsd: number;
  sessionPnLPct: number;
  activePositions: number;
  maxActivePositions: number;
}

interface AgentStatus {
  status: string;
  startedAt?: string;
  pausedAt?: string;
  durationSeconds?: number;
  errorMessage?: string;
  totalPredictions: number;
  config?: AgentConfig;
  stats?: AgentStats;
}

interface AgentDecision {
  timestamp: string;
  action: string;
  symbol?: string; // Changed from opportunitySymbol to match backend
  opportunitySymbol?: string; // Keep for backwards compatibility
  confidence?: string;
  enterProbability?: number;
  exitProbability?: number;
  stateValue?: number;
  numOpportunities: number;
  numPositions: number;
  reasoning?: string;

  // Execution result fields
  executionStatus?: string; // "success" | "failed"
  errorMessage?: string;

  // ENTER specific fields
  amountUsd?: number;
  executionId?: number;

  // EXIT specific fields
  profitUsd?: number;
  profitPct?: number;
  durationHours?: number;
}

export const apiService = {
  // ============================================================================
  // AGENT APIS
  // ============================================================================

  async startAgent(config?: AgentConfig): Promise<AgentStatus> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/agent/start`, {
        method: 'POST',
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
        body: config ? JSON.stringify(config) : undefined,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to start agent');
      }

      return await response.json();
    } catch (error) {
      console.error('Error starting agent:', error);
      throw error;
    }
  },

  async stopAgent(): Promise<void> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/agent/stop`, {
        method: 'POST',
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to stop agent');
      }
    } catch (error) {
      console.error('Error stopping agent:', error);
      throw error;
    }
  },

  async pauseAgent(): Promise<void> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/agent/pause`, {
        method: 'POST',
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to pause agent');
      }
    } catch (error) {
      console.error('Error pausing agent:', error);
      throw error;
    }
  },

  async resumeAgent(): Promise<void> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/agent/resume`, {
        method: 'POST',
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to resume agent');
      }
    } catch (error) {
      console.error('Error resuming agent:', error);
      throw error;
    }
  },

  async getAgentStatus(): Promise<AgentStatus> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/agent/status`, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch agent status');
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching agent status:', error);
      throw error;
    }
  },

  async getAgentConfig(): Promise<AgentConfig> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/agent/config`, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch agent config');
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching agent config:', error);
      throw error;
    }
  },

  async updateAgentConfig(config: Partial<AgentConfig>): Promise<AgentConfig> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/agent/config`, {
        method: 'PUT',
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to update agent config');
      }

      return await response.json();
    } catch (error) {
      console.error('Error updating agent config:', error);
      throw error;
    }
  },

  async getAgentStats(): Promise<AgentStats> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/agent/stats`, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch agent stats');
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching agent stats:', error);
      throw error;
    }
  },

  // ============================================================================
  // EXISTING APIS
  // ============================================================================

  async getUserApiKeys(): Promise<UserApiKey[]> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/user/apikeys`, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch user API keys');
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching user API keys:', error);
      throw error;
    }
  },

  async getPositions(): Promise<any[]> {
    try {
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/position`, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'application/json',
        },
      });

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
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/opportunity/execute`, {
        method: 'POST',
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
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
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/opportunity/close/${activeOpportunityId}`, {
        method: 'POST',
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
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
      const token = getAuthToken();
      const response = await fetch(`${getApiBaseUrl()}/opportunity/stop/${executionId}`, {
        method: 'POST',
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
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
      const token = getAuthToken();
      const response = await fetch(
        `${getApiBaseUrl()}/opportunity/execution-balances?exchange=${encodeURIComponent(exchange)}&maxLeverage=${maxLeverage}`,
        {
          headers: {
            'Authorization': token ? `Bearer ${token}` : '',
            'Content-Type': 'application/json',
          },
        }
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

export type {
  ExecuteOpportunityRequest,
  ExecuteOpportunityResponse,
  CloseOpportunityResponse,
  ExecutionBalances,
  AgentConfig,
  AgentStats,
  AgentStatus,
  AgentDecision,
};
