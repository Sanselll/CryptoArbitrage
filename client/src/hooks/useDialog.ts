import { useState, useCallback } from 'react';

interface AlertDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  variant: 'default' | 'success' | 'danger' | 'warning' | 'info';
}

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  variant: 'default' | 'success' | 'danger' | 'warning' | 'info';
  confirmText: string;
  cancelText: string;
  onConfirm: () => void;
}

export const useDialog = () => {
  const [alertState, setAlertState] = useState<AlertDialogState>({
    isOpen: false,
    title: '',
    message: '',
    variant: 'info',
  });

  const [confirmState, setConfirmState] = useState<ConfirmDialogState>({
    isOpen: false,
    title: '',
    message: '',
    variant: 'warning',
    confirmText: 'Confirm',
    cancelText: 'Cancel',
    onConfirm: () => {},
  });

  const showAlert = useCallback(
    (
      message: string,
      title: string = 'Alert',
      variant: 'default' | 'success' | 'danger' | 'warning' | 'info' = 'info'
    ) => {
      setAlertState({
        isOpen: true,
        title,
        message,
        variant,
      });
    },
    []
  );

  const showSuccess = useCallback((message: string, title: string = 'Success') => {
    showAlert(message, title, 'success');
  }, [showAlert]);

  const showError = useCallback((message: string, title: string = 'Error') => {
    showAlert(message, title, 'danger');
  }, [showAlert]);

  const showWarning = useCallback((message: string, title: string = 'Warning') => {
    showAlert(message, title, 'warning');
  }, [showAlert]);

  const showInfo = useCallback((message: string, title: string = 'Information') => {
    showAlert(message, title, 'info');
  }, [showAlert]);

  const closeAlert = useCallback(() => {
    setAlertState((prev) => ({ ...prev, isOpen: false }));
  }, []);

  const showConfirm = useCallback(
    (
      message: string,
      onConfirm: () => void,
      options?: {
        title?: string;
        confirmText?: string;
        cancelText?: string;
        variant?: 'default' | 'success' | 'danger' | 'warning' | 'info';
      }
    ): Promise<boolean> => {
      return new Promise((resolve) => {
        setConfirmState({
          isOpen: true,
          title: options?.title || 'Confirm',
          message,
          variant: options?.variant || 'warning',
          confirmText: options?.confirmText || 'Confirm',
          cancelText: options?.cancelText || 'Cancel',
          onConfirm: () => {
            onConfirm();
            resolve(true);
          },
        });
      });
    },
    []
  );

  const closeConfirm = useCallback(() => {
    setConfirmState((prev) => ({ ...prev, isOpen: false }));
  }, []);

  return {
    // Alert dialog
    alertState,
    showAlert,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    closeAlert,
    // Confirm dialog
    confirmState,
    showConfirm,
    closeConfirm,
  };
};
