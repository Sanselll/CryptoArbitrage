import { toast, ToastOptions } from 'react-toastify';
import type { Notification } from '../types/index';
import { NotificationSeverity } from '../types/index';
import { AlertTriangle, AlertCircle, CheckCircle, Info } from 'lucide-react';

class NotificationService {
  private hasPermission = false;

  constructor() {
    this.requestBrowserPermission();
  }

  /**
   * Request permission for browser notifications
   */
  async requestBrowserPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
      try {
        const permission = await Notification.requestPermission();
        this.hasPermission = permission === 'granted';
      } catch (error) {
        console.warn('Browser notifications not supported or permission denied', error);
      }
    } else if ('Notification' in window) {
      this.hasPermission = Notification.permission === 'granted';
    }
  }

  /**
   * Show a notification using react-toastify
   */
  showNotification(notification: Notification) {
    console.log('[NotificationService] showNotification called:', {
      id: notification.id,
      type: notification.type,
      severity: notification.severity,
      title: notification.title,
      timestamp: new Date().toISOString()
    });

    const options: ToastOptions = {
      autoClose: notification.autoClose ? (notification.autoCloseDelay || 5000) : false,
      closeButton: true,
      position: 'top-right',
      hideProgressBar: false,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
    };

    const message = (
      <div className="flex flex-col gap-1">
        <div className="font-semibold">{notification.title}</div>
        <div className="text-sm">{notification.message}</div>
      </div>
    );

    switch (notification.severity) {
      case NotificationSeverity.Success:
        toast.success(message, {
          ...options,
          icon: () => <CheckCircle className="w-5 h-5 text-green-500" />,
        });
        break;
      case NotificationSeverity.Warning:
        toast.warning(message, {
          ...options,
          icon: () => <AlertTriangle className="w-5 h-5 text-yellow-500" />,
        });
        // Send browser notification for warnings if critical
        if (this.hasPermission && !notification.autoClose) {
          this.sendBrowserNotification(notification);
        }
        break;
      case NotificationSeverity.Error:
        toast.error(message, {
          ...options,
          icon: () => <AlertCircle className="w-5 h-5 text-red-500" />,
        });
        // Send browser notification for errors
        if (this.hasPermission) {
          this.sendBrowserNotification(notification);
        }
        break;
      case NotificationSeverity.Info:
      default:
        toast.info(message, {
          ...options,
          icon: () => <Info className="w-5 h-5 text-blue-500" />,
        });
        break;
    }
  }

  /**
   * Send a browser notification (when tab is not focused)
   */
  private sendBrowserNotification(notification: Notification) {
    if (!this.hasPermission || document.hasFocus()) {
      return; // Only send if tab is not focused
    }

    try {
      const browserNotification = new Notification(notification.title, {
        body: notification.message,
        icon: '/favicon.ico',
        tag: notification.id,
        requireInteraction: !notification.autoClose,
      });

      browserNotification.onclick = () => {
        window.focus();
        browserNotification.close();
      };
    } catch (error) {
      console.error('Failed to send browser notification', error);
    }
  }
}

export const notificationService = new NotificationService();
