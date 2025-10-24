export type TradingMode = 'Demo' | 'Real';

/**
 * Get the current trading mode from sessionStorage
 */
export const getTradingMode = (): TradingMode => {
  const mode = sessionStorage.getItem('trading_mode');
  return (mode === 'Real' ? 'Real' : 'Demo') as TradingMode;
};

/**
 * Get the current authentication token (single token for all modes)
 */
export const getAuthToken = (): string | null => {
  return localStorage.getItem('jwt_token');
};

/**
 * Set the trading mode in sessionStorage
 */
export const setTradingMode = (mode: TradingMode): void => {
  sessionStorage.setItem('trading_mode', mode);
};

/**
 * Set the authentication token
 */
export const setAuthToken = (token: string): void => {
  localStorage.setItem('jwt_token', token);
};

/**
 * Completely clear ALL application data
 * This includes:
 * - All localStorage keys
 * - All sessionStorage keys
 * - Zustand stores will be reset by the caller
 * - SignalR will be disconnected by the caller
 */
export const clearAllAppData = (): void => {
  // Clear all localStorage
  localStorage.clear();

  // Clear all sessionStorage
  sessionStorage.clear();

  console.log('All application data cleared');
};
