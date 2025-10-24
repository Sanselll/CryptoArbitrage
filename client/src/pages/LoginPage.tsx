import { GoogleLogin } from '@react-oauth/google';
import { useAuthStore, type TradingMode } from '../stores/authStore';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { Activity, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '../components/ui/Card';

export const LoginPage = () => {
  const login = useAuthStore((state) => state.login);
  const error = useAuthStore((state) => state.error);
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState<TradingMode>('Demo');

  const handleGoogleSuccess = async (credentialResponse: any) => {
    try {
      setIsLoading(true);
      await login(credentialResponse.credential, selectedMode);
      navigate('/');
    } catch (err) {
      console.error('Login failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-binance-bg px-4">
      <Card className="w-full max-w-md">
        <CardContent className="p-6">
          {/* Header */}
          <div className="flex flex-col items-center mb-6">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-6 h-6 text-binance-yellow" />
              <h1 className="text-xl font-bold text-binance-yellow">
                Crypto Arbitrage
              </h1>
            </div>
            <p className="text-xs text-binance-text-secondary text-center">
              Sign in with your authorized Google account
            </p>
          </div>

          {/* Trading Mode Selector */}
          <div className="mb-6">
            <p className="text-xs text-binance-text-secondary mb-3 text-center">
              Select Trading Mode
            </p>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setSelectedMode('Demo')}
                className={`p-3 rounded border transition-all ${
                  selectedMode === 'Demo'
                    ? 'border-binance-blue bg-binance-blue/10 text-binance-blue'
                    : 'border-binance-border bg-binance-bg-tertiary text-binance-text-secondary hover:border-binance-blue/50'
                }`}
              >
                <div className="font-semibold text-sm">Demo Trading</div>
                <div className="text-[10px] mt-1 opacity-80">Practice Mode</div>
              </button>
              <button
                onClick={() => setSelectedMode('Real')}
                className={`p-3 rounded border transition-all ${
                  selectedMode === 'Real'
                    ? 'border-binance-yellow bg-binance-yellow/10 text-binance-yellow'
                    : 'border-binance-border bg-binance-bg-tertiary text-binance-text-secondary hover:border-binance-yellow/50'
                }`}
              >
                <div className="font-semibold text-sm">Real Trading</div>
                <div className="text-[10px] mt-1 opacity-80">Live Market</div>
              </button>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-binance-red/10 border border-binance-red/20 rounded p-3 mb-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-binance-red flex-shrink-0 mt-0.5" />
                <p className="text-xs text-binance-red">{error}</p>
              </div>
            </div>
          )}

          {/* Google Login Button */}
          <div className="mb-4">
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => console.error('Login failed')}
            />
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="text-center py-2">
              <p className="text-xs text-binance-text-secondary">Authenticating...</p>
            </div>
          )}

          {/* Footer Info */}
          <div className="mt-6 pt-4 border-t border-binance-border">
            <p className="text-[10px] text-binance-text-muted text-center">
              This platform requires whitelisted email access. Contact your administrator for access.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
