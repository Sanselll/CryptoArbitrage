import { Building2, Wallet, CircleDollarSign } from 'lucide-react';

export interface ExchangeConfig {
  color: string;
  icon: typeof Wallet;
  badgeVariant: 'info' | 'success' | 'warning' | 'danger' | 'secondary';
  faviconUrl?: string;
  borderColor: string;
  bgColor: string;
}

const exchangeConfigs: Record<string, ExchangeConfig> = {
  binance: {
    color: 'text-binance-yellow',
    icon: Wallet,
    badgeVariant: 'warning',
    faviconUrl: 'https://bin.bnbstatic.com/static/images/common/favicon.ico',
    borderColor: 'border-binance-yellow',
    bgColor: 'bg-binance-yellow/10',
  },
  bybit: {
    color: 'text-orange-400',
    icon: Building2,
    badgeVariant: 'warning',
    faviconUrl: 'https://www.bybit.com/favicon.ico',
    borderColor: 'border-orange-400',
    bgColor: 'bg-orange-400/10',
  },
  // Add more exchanges as needed
  // coinbase: {
  //   color: 'text-blue-500',
  //   icon: CircleDollarSign,
  //   badgeVariant: 'info',
  //   faviconUrl: 'https://www.coinbase.com/favicon.ico',
  //   borderColor: 'border-blue-500',
  //   bgColor: 'bg-blue-500/10',
  // },
};

const defaultConfig: ExchangeConfig = {
  color: 'text-binance-green',
  icon: Wallet,
  badgeVariant: 'secondary',
  borderColor: 'border-binance-border',
  bgColor: 'bg-binance-bg-secondary',
};

export const getExchangeConfig = (exchange: string): ExchangeConfig => {
  const normalized = exchange.toLowerCase();
  return exchangeConfigs[normalized] || defaultConfig;
};

export const getExchangeColor = (exchange: string): string => {
  return getExchangeConfig(exchange).color;
};

export const getExchangeIcon = (exchange: string): typeof Wallet => {
  return getExchangeConfig(exchange).icon;
};

export const getExchangeBadgeVariant = (exchange: string): 'info' | 'success' | 'warning' | 'danger' | 'secondary' => {
  return getExchangeConfig(exchange).badgeVariant;
};

export const getExchangeFaviconUrl = (exchange: string): string | undefined => {
  return getExchangeConfig(exchange).faviconUrl;
};

export const getExchangeBorderColor = (exchange: string): string => {
  return getExchangeConfig(exchange).borderColor;
};

export const getExchangeBgColor = (exchange: string): string => {
  return getExchangeConfig(exchange).bgColor;
};
