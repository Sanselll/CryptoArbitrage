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
};

// CRITICAL: Attach JWT token from localStorage to every request
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('jwt_token');
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
      localStorage.removeItem('jwt_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default apiClient;
