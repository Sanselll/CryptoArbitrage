import axios, { AxiosInstance, AxiosError, AxiosResponse } from 'axios';

type TradingMode = 'Demo' | 'Real';

// Get base URL based on trading mode from sessionStorage
const getApiBaseUrl = (): string => {
  const mode = sessionStorage.getItem('trading_mode') as TradingMode | null;

  if (mode === 'Real') {
    return import.meta.env.VITE_API_BASE_URL_REAL || 'http://localhost:5053/api';
  }

  // Default to Demo
  return import.meta.env.VITE_API_BASE_URL_DEMO || 'http://localhost:5052/api';
};

// Get mode-specific JWT token key
const getTokenKey = (): string => {
  const mode = sessionStorage.getItem('trading_mode') as TradingMode | null;
  return mode === 'Real' ? 'jwt_token_real' : 'jwt_token_demo';
};

const apiClient: AxiosInstance = axios.create({
  baseURL: getApiBaseUrl(),
  headers: {
    'Content-Type': 'application/json'
  }
});

// Function to update the base URL (called when mode changes)
export const setApiBaseUrl = (mode: TradingMode) => {
  sessionStorage.setItem('trading_mode', mode);
  const newBaseUrl = mode === 'Real'
    ? import.meta.env.VITE_API_BASE_URL_REAL || 'http://localhost:5053/api'
    : import.meta.env.VITE_API_BASE_URL_DEMO || 'http://localhost:5052/api';
  apiClient.defaults.baseURL = newBaseUrl;
  console.log(`API client updated to ${mode} mode:`, newBaseUrl);
};

// CRITICAL: Attach JWT token from localStorage to every request using mode-specific key
apiClient.interceptors.request.use((config) => {
  const tokenKey = getTokenKey();
  const token = localStorage.getItem(tokenKey);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 Unauthorized - redirect to login
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Clear both mode-specific tokens
      localStorage.removeItem('jwt_token_demo');
      localStorage.removeItem('jwt_token_real');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default apiClient;
