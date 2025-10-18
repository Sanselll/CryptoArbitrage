import { ReactNode } from 'react';
import { X, AlertCircle, CheckCircle, AlertTriangle, Info } from 'lucide-react';
import { Button } from './Button';

interface DialogProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  variant?: 'default' | 'success' | 'danger' | 'warning' | 'info';
  showCloseButton?: boolean;
}

export const Dialog = ({
  isOpen,
  onClose,
  title,
  children,
  variant = 'default',
  showCloseButton = true,
}: DialogProps) => {
  if (!isOpen) return null;

  const getIcon = () => {
    switch (variant) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-binance-green" />;
      case 'danger':
        return <AlertCircle className="w-4 h-4 text-binance-red" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-binance-yellow" />;
      case 'info':
        return <Info className="w-4 h-4 text-binance-blue" />;
      default:
        return null;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm">
      <div className="bg-binance-bg-secondary border border-binance-border rounded-lg shadow-2xl w-full max-w-md m-4">
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b border-binance-border">
          <div className="flex items-center gap-2">
            {getIcon()}
            <h2 className="text-base font-bold text-binance-text">{title}</h2>
          </div>
          {showCloseButton && (
            <button
              onClick={onClose}
              className="text-binance-text-secondary hover:text-binance-text transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Content */}
        <div className="p-4">
          {children}
        </div>
      </div>
    </div>
  );
};

interface AlertDialogProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  message: string;
  variant?: 'default' | 'success' | 'danger' | 'warning' | 'info';
  actionText?: string;
  onAction?: () => void;
}

export const AlertDialog = ({
  isOpen,
  onClose,
  title,
  message,
  variant = 'info',
  actionText,
  onAction,
}: AlertDialogProps) => {
  const handleAction = () => {
    if (onAction) {
      onAction();
    }
    onClose();
  };

  return (
    <Dialog isOpen={isOpen} onClose={onClose} title={title} variant={variant} showCloseButton={false}>
      <div className="space-y-4">
        <p className="text-sm text-binance-text whitespace-pre-line">{message}</p>
        <div className="flex gap-2 justify-end">
          {actionText && onAction ? (
            <>
              <Button
                variant="secondary"
                size="sm"
                onClick={onClose}
                className="min-w-[80px]"
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleAction}
                className="min-w-[120px]"
              >
                {actionText}
              </Button>
            </>
          ) : (
            <Button
              variant="primary"
              size="sm"
              onClick={onClose}
              className="min-w-[80px]"
            >
              OK
            </Button>
          )}
        </div>
      </div>
    </Dialog>
  );
};

interface ConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  variant?: 'default' | 'success' | 'danger' | 'warning' | 'info';
}

export const ConfirmDialog = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'warning',
}: ConfirmDialogProps) => {
  const handleConfirm = () => {
    onConfirm();
    onClose();
  };

  return (
    <Dialog isOpen={isOpen} onClose={onClose} title={title} variant={variant} showCloseButton={false}>
      <div className="space-y-4">
        <p className="text-sm text-binance-text whitespace-pre-line">{message}</p>
        <div className="flex gap-2 justify-end">
          <Button
            variant="secondary"
            size="sm"
            onClick={onClose}
            className="min-w-[80px]"
          >
            {cancelText}
          </Button>
          <Button
            variant={variant === 'danger' ? 'danger' : 'primary'}
            size="sm"
            onClick={handleConfirm}
            className="min-w-[80px]"
          >
            {confirmText}
          </Button>
        </div>
      </div>
    </Dialog>
  );
};
