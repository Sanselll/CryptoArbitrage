import { Loader2 } from 'lucide-react';

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
}

export const LoadingOverlay = ({ isLoading, message = 'Processing...' }: LoadingOverlayProps) => {
  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-binance-bg-secondary border border-binance-border rounded-lg p-8 flex flex-col items-center gap-4 shadow-2xl">
        <Loader2 className="w-12 h-12 text-binance-yellow animate-spin" />
        <p className="text-binance-text font-medium text-lg">{message}</p>
      </div>
    </div>
  );
};
