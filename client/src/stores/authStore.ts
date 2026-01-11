import { create } from 'zustand';
import { setApiBaseUrl, getApiBaseUrl } from '../services/apiClient';
import { clearAllAppData, setAuthToken, setTradingMode, getAuthToken, getTradingMode } from '../services/authUtils';
import { useArbitrageStore } from './arbitrageStore';

export type TradingMode = 'Demo' | 'Real';

interface User {
  id: string;
  email: string;
  createdAt: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  login: (googleToken: string, mode: TradingMode) => Promise<void>;
  devLogin: (email: string, password: string, mode: TradingMode) => Promise<void>;
  logout: () => void;
  checkAuth: () => void;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: getAuthToken(),
  isAuthenticated: !!getAuthToken(),
  isLoading: false,
  error: null,

  login: async (googleToken: string, mode: TradingMode) => {
    set({ isLoading: true, error: null });
    try {
      // Set the API base URL for the selected mode
      setApiBaseUrl(mode);
      const apiBaseUrl = getApiBaseUrl();

      console.log('Login attempt with mode:', mode);
      console.log('Connecting to:', apiBaseUrl);

      const response = await fetch(`${apiBaseUrl}/auth/google-signin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ idToken: googleToken })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Login failed');
      }

      const { token, user } = await response.json();

      // Store new token and trading mode (overwrites any old data)
      setAuthToken(token);
      setTradingMode(mode);

      console.log('Stored token in jwt_token');
      console.log('Stored trading mode:', mode);

      // Update auth state
      set({ token, user, isAuthenticated: true, isLoading: false, error: null });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Login failed';
      set({ isLoading: false, error: message, isAuthenticated: false });
      throw error;
    }
  },

  devLogin: async (email: string, password: string, mode: TradingMode) => {
    set({ isLoading: true, error: null });
    try {
      setApiBaseUrl(mode);
      const apiBaseUrl = getApiBaseUrl();

      console.log('Dev login attempt with mode:', mode);
      console.log('Connecting to:', apiBaseUrl);

      const response = await fetch(`${apiBaseUrl}/auth/dev-signin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Dev login failed');
      }

      const { token, user } = await response.json();

      setAuthToken(token);
      setTradingMode(mode);

      console.log('Dev login successful for:', email);

      set({ token, user, isAuthenticated: true, isLoading: false, error: null });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Dev login failed';
      set({ isLoading: false, error: message, isAuthenticated: false });
      throw error;
    }
  },

  logout: () => {
    console.log('Logging out - clearing all application data...');

    // Reset arbitrage store state (disconnect SignalR, clear state)
    useArbitrageStore.getState().reset();

    // Clear ALL application data
    clearAllAppData();

    // Reset auth state
    set({ user: null, token: null, isAuthenticated: false, error: null });
  },

  checkAuth: () => {
    const token = getAuthToken();
    set({ isAuthenticated: !!token, token });
  },

  clearError: () => {
    set({ error: null });
  }
}));
