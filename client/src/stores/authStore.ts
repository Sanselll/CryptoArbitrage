import { create } from 'zustand';

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

  login: (googleToken: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => void;
  clearError: () => void;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5052/api';

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: localStorage.getItem('jwt_token'),
  isAuthenticated: !!localStorage.getItem('jwt_token'),
  isLoading: false,
  error: null,

  login: async (googleToken: string) => {
    set({ isLoading: true, error: null });
    try {
      const response = await fetch(`${API_BASE_URL}/auth/google-signin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ idToken: googleToken })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Login failed');
      }

      const { token, user } = await response.json();
      localStorage.setItem('jwt_token', token);
      set({ token, user, isAuthenticated: true, isLoading: false, error: null });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Login failed';
      set({ isLoading: false, error: message, isAuthenticated: false });
      throw error;
    }
  },

  logout: () => {
    localStorage.removeItem('jwt_token');
    set({ user: null, token: null, isAuthenticated: false, error: null });
  },

  checkAuth: () => {
    const token = localStorage.getItem('jwt_token');
    set({ isAuthenticated: !!token, token });
  },

  clearError: () => {
    set({ error: null });
  }
}));
