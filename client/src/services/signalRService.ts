import * as signalR from '@microsoft/signalr';
import type { FundingRate, Position, ArbitrageOpportunity, AccountBalance, Notification } from '../types/index';
import { notificationService } from './notificationService.tsx';

type TradingMode = 'Demo' | 'Real';

// Get Hub URL based on trading mode from sessionStorage
const getHubUrl = (): string => {
  const mode = sessionStorage.getItem('trading_mode') as TradingMode | null;

  const apiBaseUrl = mode === 'Real'
    ? import.meta.env.VITE_API_BASE_URL_REAL || 'http://localhost:5053/api'
    : import.meta.env.VITE_API_BASE_URL_DEMO || 'http://localhost:5052/api';

  return apiBaseUrl.replace('/api', '/hubs/arbitrage');
};

// Get mode-specific JWT token key
const getTokenKey = (): string => {
  const mode = sessionStorage.getItem('trading_mode') as TradingMode | null;
  return mode === 'Real' ? 'jwt_token_real' : 'jwt_token_demo';
};

class SignalRService {
  private connection: signalR.HubConnection | null = null;
  private callbacks = {
    onFundingRates: [] as ((data: FundingRate[]) => void)[],
    onPositions: [] as ((data: Position[]) => void)[],
    onOpportunities: [] as ((data: ArbitrageOpportunity[]) => void)[],
    onBalances: [] as ((data: AccountBalance[]) => void)[],
    onPnLUpdate: [] as ((data: { totalPnL: number; todayPnL: number }) => void)[],
    onAlert: [] as ((data: { message: string; severity: string; timestamp: string }) => void)[],
    onNotification: [] as ((data: Notification) => void)[],
  };

  async connect() {
    if (this.connection?.state === signalR.HubConnectionState.Connected) {
      console.log('Already connected');
      return;
    }

    const tokenKey = getTokenKey();
    const token = localStorage.getItem(tokenKey);
    if (!token) {
      console.error('No authentication token available');
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

    try {
      await this.connection.start();
      console.log('SignalR Connected to:', url);
    } catch (err) {
      console.error('SignalR Connection Error: ', err);
      setTimeout(() => this.connect(), 5000);
    }

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
  }

  private setupEventHandlers() {
    if (!this.connection) return;

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
      notificationService.showNotification(data);
      this.callbacks.onNotification.forEach(cb => cb(data));
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

  disconnect() {
    this.connection?.stop();
  }

  getConnectionState() {
    return this.connection?.state;
  }
}

export const signalRService = new SignalRService();
