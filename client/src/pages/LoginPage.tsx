import { GoogleLogin } from '@react-oauth/google';
import { useAuthStore } from '../stores/authStore';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { Activity, AlertCircle } from 'lucide-react';
import { Card, CardContent } from '../components/ui/Card';

export const LoginPage = () => {
  const login = useAuthStore((state) => state.login);
  const error = useAuthStore((state) => state.error);
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const handleGoogleSuccess = async (credentialResponse: any) => {
    try {
      setIsLoading(true);
      // Always login to Real trading mode (Demo disabled)
      await login(credentialResponse.credential, 'Real');
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
