import * as signalR from '@microsoft/signalr';
import type { FundingRate, Position, ArbitrageOpportunity, AccountBalance, Notification, Order, Trade, Transaction, ExecutionHistory } from '../types/index';
import { notificationService } from './notificationService.tsx';
import type { AgentStats, AgentDecision } from './apiService';

type TradingMode = 'Demo' | 'Real';

// Agent event types
interface AgentStatusEvent {
  status: string;
  durationSeconds?: number;
  isRunning: boolean;
  errorMessage?: string;
  timestamp: string;
}

interface AgentErrorEvent {
  error: string;
  timestamp: string;
}

// Get Hub URL based on trading mode from sessionStorage
const getHubUrl = (): string => {
  const mode = sessionStorage.getItem('trading_mode') as TradingMode | null;

  const apiBaseUrl = mode === 'Real'
    ? import.meta.env.VITE_API_BASE_URL_REAL || 'http://localhost:5053/api'
    : import.meta.env.VITE_API_BASE_URL_DEMO || 'http://localhost:5052/api';

  return apiBaseUrl.replace('/api', '/hubs/arbitrage');
};

class SignalRService {
  private connection: signalR.HubConnection | null = null;
  private connecting: boolean = false;
  private callbacks = {
    onFundingRates: [] as ((data: FundingRate[]) => void)[],
    onPositions: [] as ((data: Position[]) => void)[],
    onOpportunities: [] as ((data: ArbitrageOpportunity[]) => void)[],
    onBalances: [] as ((data: AccountBalance[]) => void)[],
    onOpenOrders: [] as ((data: Order[]) => void)[],
    onOrderHistory: [] as ((data: Order[]) => void)[],
    onTradeHistory: [] as ((data: Trade[]) => void)[],
    onTransactionHistory: [] as ((data: Transaction[]) => void)[],
    onPnLUpdate: [] as ((data: { totalPnL: number; todayPnL: number }) => void)[],
    onAlert: [] as ((data: { message: string; severity: string; timestamp: string }) => void)[],
    onNotification: [] as ((data: Notification) => void)[],
    onAgentStatus: [] as ((data: AgentStatusEvent) => void)[],
    onAgentStats: [] as ((data: AgentStats) => void)[],
    onAgentDecision: [] as ((data: AgentDecision) => void)[],
    onAgentError: [] as ((data: AgentErrorEvent) => void)[],
    onExecutionHistory: [] as ((data: ExecutionHistory[]) => void)[],
  };

  async connect() {
    // Prevent concurrent connection attempts
    if (this.connecting) {
      console.log('Connection already in progress');
      return;
    }

    if (this.connection?.state === signalR.HubConnectionState.Connected) {
      console.log('Already connected');
      return;
    }

    this.connecting = true;

    try {
      // Clean up old connection if it exists
      if (this.connection) {
        await this.connection.stop();
        this.connection = null;
      }

      // Use single jwt_token key for all modes
      const token = localStorage.getItem('jwt_token');
      if (!token) {
        console.error('No authentication token available');
        this.connecting = false;
        throw new Error('Not authenticated');
      }

      const url = getHubUrl();
      console.log('Connecting to SignalR hub:', url);

      this.connection = new signalR.HubConnectionBuilder()
        .withUrl(url, {
          // CRITICAL: Send JWT token with connection
          accessTokenFactory: () => token
        })
        .withAutomaticReconnect()
        .configureLogging(signalR.LogLevel.Information)
        .build();

      this.setupEventHandlers();

      // Set up connection lifecycle handlers
      this.connection.onreconnecting(() => {
        console.log('SignalR Reconnecting...');
      });

      this.connection.onreconnected(() => {
        console.log('SignalR Reconnected');
      });

      this.connection.onclose(() => {
        console.log('SignalR Disconnected');
        setTimeout(() => this.connect(), 5000);
      });

      await this.connection.start();
      this.connecting = false;
      console.log('SignalR Connected to:', url);
    } catch (err) {
      this.connecting = false;
      console.error('SignalR Connection Error: ', err);
      setTimeout(() => this.connect(), 5000);
    }
  }

  private setupEventHandlers() {
    if (!this.connection) return;

    // Remove any existing handlers first
    this.connection.off('ReceiveFundingRates');
    this.connection.off('ReceivePositions');
    this.connection.off('ReceiveOpportunities');
    this.connection.off('ReceiveBalances');
    this.connection.off('ReceivePnLUpdate');
    this.connection.off('ReceiveAlert');
    this.connection.off('ReceiveNotification');
    this.connection.off('ReceiveOpenOrders');
    this.connection.off('ReceiveOrderHistory');
    this.connection.off('ReceiveTradeHistory');
    this.connection.off('ReceiveTransactionHistory');
    this.connection.off('ReceiveAgentStatus');
    this.connection.off('ReceiveAgentStats');
    this.connection.off('ReceiveAgentDecision');
    this.connection.off('ReceiveAgentError');
    this.connection.off('ReceiveExecutionHistory');

    // Register new handlers
    this.connection.on('ReceiveFundingRates', (data: FundingRate[]) => {
      this.callbacks.onFundingRates.forEach(cb => cb(data));
    });

    this.connection.on('ReceivePositions', (data: Position[]) => {
      this.callbacks.onPositions.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveOpportunities', (data: ArbitrageOpportunity[]) => {
      this.callbacks.onOpportunities.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveBalances', (data: AccountBalance[]) => {
      this.callbacks.onBalances.forEach(cb => cb(data));
    });

    this.connection.on('ReceivePnLUpdate', (data: { totalPnL: number; todayPnL: number }) => {
      this.callbacks.onPnLUpdate.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveAlert', (data: { message: string; severity: string; timestamp: string }) => {
      this.callbacks.onAlert.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveNotification', (data: Notification) => {
      console.log('[SignalR] ReceiveNotification event:', {
        id: data.id,
        type: data.type,
        severity: data.severity,
        title: data.title,
        timestamp: new Date().toISOString()
      });
      notificationService.showNotification(data);
      this.callbacks.onNotification.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveOpenOrders', (data: Order[]) => {
      this.callbacks.onOpenOrders.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveOrderHistory', (data: Order[]) => {
      this.callbacks.onOrderHistory.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveTradeHistory', (data: Trade[]) => {
      this.callbacks.onTradeHistory.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveTransactionHistory', (data: Transaction[]) => {
      this.callbacks.onTransactionHistory.forEach(cb => cb(data));
    });

    // Agent events
    this.connection.on('ReceiveAgentStatus', (data: AgentStatusEvent) => {
      this.callbacks.onAgentStatus.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveAgentStats', (data: AgentStats) => {
      this.callbacks.onAgentStats.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveAgentDecision', (data: AgentDecision) => {
      console.log('[SignalR] ReceiveAgentDecision - callbacks count:', this.callbacks.onAgentDecision.length);
      this.callbacks.onAgentDecision.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveAgentError', (data: AgentErrorEvent) => {
      this.callbacks.onAgentError.forEach(cb => cb(data));
    });

    this.connection.on('ReceiveExecutionHistory', (data: ExecutionHistory[]) => {
      this.callbacks.onExecutionHistory.forEach(cb => cb(data));
    });
  }

  onFundingRates(callback: (data: FundingRate[]) => void) {
    this.callbacks.onFundingRates.push(callback);
    return () => {
      this.callbacks.onFundingRates = this.callbacks.onFundingRates.filter(cb => cb !== callback);
    };
  }

  onPositions(callback: (data: Position[]) => void) {
    this.callbacks.onPositions.push(callback);
    return () => {
      this.callbacks.onPositions = this.callbacks.onPositions.filter(cb => cb !== callback);
    };
  }

  onOpportunities(callback: (data: ArbitrageOpportunity[]) => void) {
    this.callbacks.onOpportunities.push(callback);
    return () => {
      this.callbacks.onOpportunities = this.callbacks.onOpportunities.filter(cb => cb !== callback);
    };
  }

  onBalances(callback: (data: AccountBalance[]) => void) {
    this.callbacks.onBalances.push(callback);
    return () => {
      this.callbacks.onBalances = this.callbacks.onBalances.filter(cb => cb !== callback);
    };
  }

  onPnLUpdate(callback: (data: { totalPnL: number; todayPnL: number }) => void) {
    this.callbacks.onPnLUpdate.push(callback);
    return () => {
      this.callbacks.onPnLUpdate = this.callbacks.onPnLUpdate.filter(cb => cb !== callback);
    };
  }

  onAlert(callback: (data: { message: string; severity: string; timestamp: string }) => void) {
    this.callbacks.onAlert.push(callback);
    return () => {
      this.callbacks.onAlert = this.callbacks.onAlert.filter(cb => cb !== callback);
    };
  }

  onNotification(callback: (data: Notification) => void) {
    this.callbacks.onNotification.push(callback);
    return () => {
      this.callbacks.onNotification = this.callbacks.onNotification.filter(cb => cb !== callback);
    };
  }

  onOpenOrders(callback: (data: Order[]) => void) {
    this.callbacks.onOpenOrders.push(callback);
    return () => {
      this.callbacks.onOpenOrders = this.callbacks.onOpenOrders.filter(cb => cb !== callback);
    };
  }

  onOrderHistory(callback: (data: Order[]) => void) {
    this.callbacks.onOrderHistory.push(callback);
    return () => {
      this.callbacks.onOrderHistory = this.callbacks.onOrderHistory.filter(cb => cb !== callback);
    };
  }

  onTradeHistory(callback: (data: Trade[]) => void) {
    this.callbacks.onTradeHistory.push(callback);
    return () => {
      this.callbacks.onTradeHistory = this.callbacks.onTradeHistory.filter(cb => cb !== callback);
    };
  }

  onTransactionHistory(callback: (data: Transaction[]) => void) {
    this.callbacks.onTransactionHistory.push(callback);
    return () => {
      this.callbacks.onTransactionHistory = this.callbacks.onTransactionHistory.filter(cb => cb !== callback);
    };
  }

  onAgentStatus(callback: (data: AgentStatusEvent) => void) {
    this.callbacks.onAgentStatus.push(callback);
    return () => {
      this.callbacks.onAgentStatus = this.callbacks.onAgentStatus.filter(cb => cb !== callback);
    };
  }

  onAgentStats(callback: (data: AgentStats) => void) {
    this.callbacks.onAgentStats.push(callback);
    return () => {
      this.callbacks.onAgentStats = this.callbacks.onAgentStats.filter(cb => cb !== callback);
    };
  }

  onAgentDecision(callback: (data: AgentDecision) => void) {
    this.callbacks.onAgentDecision.push(callback);
    console.log('[SignalR] Registered onAgentDecision callback. Total callbacks:', this.callbacks.onAgentDecision.length);
    return () => {
      this.callbacks.onAgentDecision = this.callbacks.onAgentDecision.filter(cb => cb !== callback);
      console.log('[SignalR] Unregistered onAgentDecision callback. Remaining callbacks:', this.callbacks.onAgentDecision.length);
    };
  }

  onAgentError(callback: (data: AgentErrorEvent) => void) {
    this.callbacks.onAgentError.push(callback);
    return () => {
      this.callbacks.onAgentError = this.callbacks.onAgentError.filter(cb => cb !== callback);
    };
  }

  onExecutionHistory(callback: (data: ExecutionHistory[]) => void) {
    this.callbacks.onExecutionHistory.push(callback);
    return () => {
      this.callbacks.onExecutionHistory = this.callbacks.onExecutionHistory.filter(cb => cb !== callback);
    };
  }

  disconnect() {
    this.connection?.stop();
  }

  getConnectionState() {
    return this.connection?.state;
  }
}

export const signalRService = new SignalRService();
