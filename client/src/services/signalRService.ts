import * as signalR from '@microsoft/signalr';
import type { FundingRate, Position, ArbitrageOpportunity, AccountBalance } from '../types/index';

class SignalRService {
  private connection: signalR.HubConnection | null = null;
  private callbacks = {
    onFundingRates: [] as ((data: FundingRate[]) => void)[],
    onPositions: [] as ((data: Position[]) => void)[],
    onOpportunities: [] as ((data: ArbitrageOpportunity[]) => void)[],
    onBalances: [] as ((data: AccountBalance[]) => void)[],
    onPnLUpdate: [] as ((data: { totalPnL: number; todayPnL: number }) => void)[],
    onAlert: [] as ((data: { message: string; severity: string; timestamp: string }) => void)[],
  };

  async connect(url: string = 'http://localhost:5052/arbitragehub') {
    if (this.connection?.state === signalR.HubConnectionState.Connected) {
      console.log('Already connected');
      return;
    }

    this.connection = new signalR.HubConnectionBuilder()
      .withUrl(url)
      .withAutomaticReconnect()
      .configureLogging(signalR.LogLevel.Information)
      .build();

    this.setupEventHandlers();

    try {
      await this.connection.start();
      console.log('SignalR Connected');
    } catch (err) {
      console.error('SignalR Connection Error: ', err);
      setTimeout(() => this.connect(url), 5000);
    }

    this.connection.onreconnecting(() => {
      console.log('SignalR Reconnecting...');
    });

    this.connection.onreconnected(() => {
      console.log('SignalR Reconnected');
    });

    this.connection.onclose(() => {
      console.log('SignalR Disconnected');
      setTimeout(() => this.connect(url), 5000);
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

  disconnect() {
    this.connection?.stop();
  }

  getConnectionState() {
    return this.connection?.state;
  }
}

export const signalRService = new SignalRService();
