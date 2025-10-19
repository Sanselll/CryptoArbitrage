import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Plus, Trash2, TestTube, AlertCircle, CheckCircle, Shield, X } from 'lucide-react';
import apiClient from '../services/apiClient';
import { useAuthStore } from '../stores/authStore';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { ConfirmDialog, AlertDialog } from '../components/ui/Dialog';

interface ApiKey {
  id: number;
  exchangeName: string;
  apiKey: string;
  isEnabled: boolean;
  createdAt: string;
  lastTestedAt?: string;
  lastTestResult?: string;
}

export const ProfileSettings = () => {
  const user = useAuthStore((state) => state.user);
  const navigate = useNavigate();
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [isAdding, setIsAdding] = useState(false);
  const [isTesting, setIsTesting] = useState<number | null>(null);
  const [newKey, setNewKey] = useState({
    exchangeName: 'Binance',
    apiKey: '',
    apiSecret: ''
  });
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<{ isOpen: boolean; id: number | null }>({ isOpen: false, id: null });
  const [alertDialog, setAlertDialog] = useState<{ isOpen: boolean; title: string; message: string; variant: 'success' | 'danger' | 'warning' | 'info' }>({
    isOpen: false,
    title: '',
    message: '',
    variant: 'info'
  });

  useEffect(() => {
    loadApiKeys();
  }, []);

  const loadApiKeys = async () => {
    try {
      const response = await apiClient.get('/user/apikeys');
      setApiKeys(response.data);
    } catch (error) {
      console.error('Error loading API keys:', error);
      setMessage({ type: 'error', text: 'Failed to load API keys' });
    }
  };

  const addApiKey = async () => {
    if (!newKey.apiKey || !newKey.apiSecret) {
      setMessage({ type: 'error', text: 'API key and secret are required' });
      return;
    }

    try {
      setMessage({ type: 'success', text: 'Validating API credentials...' });

      const response = await apiClient.post('/user/apikeys', newKey);

      // If we reach here, validation succeeded and key was saved
      setAlertDialog({
        isOpen: true,
        title: 'API Key Added Successfully',
        message: `Successfully connected to ${newKey.exchangeName}! Your API credentials have been validated and saved.`,
        variant: 'success'
      });

      setNewKey({ exchangeName: 'Binance', apiKey: '', apiSecret: '' });
      setIsAdding(false);
      setMessage(null);

      // Reload keys to show the new key
      await loadApiKeys();
    } catch (error: any) {
      const errorMsg = error.response?.data?.error || 'Failed to add API key';
      setMessage(null);

      // Handle session expiration (401 Unauthorized)
      if (error.response?.status === 401) {
        setAlertDialog({
          isOpen: true,
          title: 'Session Expired',
          message: `${errorMsg} You will be logged out automatically.`,
          variant: 'warning'
        });

        // Log out after showing the message
        setTimeout(() => {
          useAuthStore.getState().logout();
          navigate('/login');
        }, 3000);
      } else {
        setAlertDialog({
          isOpen: true,
          title: 'Failed to Add API Key',
          message: errorMsg,
          variant: 'danger'
        });
      }
    }
  };

  const deleteApiKey = async (id: number) => {
    try {
      await apiClient.delete(`/user/apikeys/${id}`);
      setMessage({ type: 'success', text: 'API key deleted successfully' });
      setDeleteConfirm({ isOpen: false, id: null });
      loadApiKeys();
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to delete API key' });
      setDeleteConfirm({ isOpen: false, id: null });
    }
  };

  const testApiKey = async (id: number) => {
    setIsTesting(id);
    try {
      const response = await apiClient.post(`/user/apikeys/${id}/test`);

      setAlertDialog({
        isOpen: true,
        title: response.data.success ? 'Connection Successful' : 'Connection Failed',
        message: response.data.message,
        variant: response.data.success ? 'success' : 'danger'
      });

      loadApiKeys();
    } catch (error: any) {
      setAlertDialog({
        isOpen: true,
        title: 'Test Failed',
        message: error.response?.data?.error || 'Failed to test API key',
        variant: 'danger'
      });
    } finally {
      setIsTesting(null);
    }
  };

  return (
    <div className="min-h-screen bg-binance-bg">
      {/* Header */}
      <div className="h-10 bg-binance-bg-secondary border-b border-binance-border px-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/')}
            className="text-binance-text-secondary hover:text-binance-text transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <h1 className="text-sm font-bold text-binance-text">Profile Settings</h1>
        </div>
        <div className="text-xs text-binance-text-secondary">{user?.email}</div>
      </div>

      {/* Content */}
      <div className="p-4 max-w-5xl mx-auto">
        {/* Messages */}
        {message && (
          <div
            className={`p-3 rounded mb-3 border ${
              message.type === 'success'
                ? 'bg-binance-green/10 border-binance-green/20'
                : 'bg-binance-red/10 border-binance-red/20'
            }`}
          >
            <div className="flex items-start gap-2">
              {message.type === 'success' ? (
                <CheckCircle className="w-4 h-4 text-binance-green flex-shrink-0 mt-0.5" />
              ) : (
                <AlertCircle className="w-4 h-4 text-binance-red flex-shrink-0 mt-0.5" />
              )}
              <p className={`text-xs ${message.type === 'success' ? 'text-binance-green' : 'text-binance-red'}`}>
                {message.text}
              </p>
              <button
                onClick={() => setMessage(null)}
                className="ml-auto text-binance-text-secondary hover:text-binance-text"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}

        {/* API Keys Card */}
        <Card>
          <CardHeader className="p-3">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Shield className="w-4 h-4 text-binance-yellow" />
                Exchange API Keys
              </CardTitle>
              {!isAdding && (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => setIsAdding(true)}
                  className="gap-1"
                >
                  <Plus className="w-3 h-3" />
                  Add Key
                </Button>
              )}
            </div>
          </CardHeader>

          <CardContent className="p-3 pt-0">
            {/* Add Form */}
            {isAdding && (
              <div className="bg-binance-bg-tertiary p-3 rounded mb-3 border border-binance-border">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xs font-semibold text-binance-text">Add New API Key</h3>
                  <button
                    onClick={() => setIsAdding(false)}
                    className="text-binance-text-secondary hover:text-binance-text"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
                <div className="space-y-2">
                  <div>
                    <label className="text-[10px] text-binance-text-secondary block mb-1">Exchange</label>
                    <select
                      value={newKey.exchangeName}
                      onChange={(e) => setNewKey({ ...newKey, exchangeName: e.target.value })}
                      className="w-full bg-binance-bg border border-binance-border text-binance-text rounded text-xs px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-binance-yellow"
                    >
                      <option>Binance</option>
                      <option>Bybit</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-binance-text-secondary block mb-1">API Key</label>
                    <input
                      type="text"
                      value={newKey.apiKey}
                      onChange={(e) => setNewKey({ ...newKey, apiKey: e.target.value })}
                      className="w-full bg-binance-bg border border-binance-border text-binance-text rounded text-xs px-2 py-1.5 font-mono focus:outline-none focus:ring-1 focus:ring-binance-yellow"
                      placeholder="Paste your API key"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-binance-text-secondary block mb-1">API Secret</label>
                    <input
                      type="password"
                      value={newKey.apiSecret}
                      onChange={(e) => setNewKey({ ...newKey, apiSecret: e.target.value })}
                      className="w-full bg-binance-bg border border-binance-border text-binance-text rounded text-xs px-2 py-1.5 font-mono focus:outline-none focus:ring-1 focus:ring-binance-yellow"
                      placeholder="Paste your API secret"
                    />
                  </div>
                  <div className="flex gap-2 pt-1">
                    <Button
                      variant="success"
                      size="sm"
                      onClick={addApiKey}
                      className="text-xs"
                    >
                      Save
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => setIsAdding(false)}
                      className="text-xs"
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              </div>
            )}

            {/* API Keys List */}
            <div className="space-y-2">
              {apiKeys.map((key) => (
                <div key={key.id} className="bg-binance-bg-tertiary p-3 rounded border border-binance-border">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1.5">
                        <h3 className="text-xs font-semibold text-binance-text">{key.exchangeName}</h3>
                        <Badge
                          variant={key.isEnabled ? 'success' : 'secondary'}
                          size="sm"
                          className="text-[10px]"
                        >
                          {key.isEnabled ? 'Enabled' : 'Disabled'}
                        </Badge>
                      </div>
                      <div className="text-[10px] text-binance-text-muted font-mono truncate">
                        {key.apiKey}
                      </div>
                      {key.lastTestResult && (
                        <div className="text-[10px] text-binance-text-secondary mt-1.5">
                          <span className="text-binance-text-muted">Last Test:</span> {key.lastTestResult}
                          {key.lastTestedAt && (
                            <span className="text-binance-text-muted">
                              {' '}({new Date(key.lastTestedAt).toLocaleString()})
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    <div className="flex gap-1.5 ml-3 flex-shrink-0">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => testApiKey(key.id)}
                        disabled={isTesting === key.id}
                        className="gap-1 text-xs"
                      >
                        <TestTube className="w-3 h-3" />
                        {isTesting === key.id ? 'Testing...' : 'Test'}
                      </Button>
                      <Button
                        variant="danger"
                        size="sm"
                        onClick={() => setDeleteConfirm({ isOpen: true, id: key.id })}
                        className="gap-1 text-xs"
                      >
                        <Trash2 className="w-3 h-3" />
                        Delete
                      </Button>
                    </div>
                  </div>
                </div>
              ))}

              {apiKeys.length === 0 && !isAdding && (
                <div className="text-center py-8 text-binance-text-muted">
                  <Shield className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-xs">No API keys configured</p>
                  <p className="text-[10px] mt-1">Add one to start trading</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Info Box */}
        <div className="bg-binance-blue/10 border border-binance-blue/20 rounded p-3 mt-3">
          <div className="flex items-start gap-2">
            <Shield className="w-4 h-4 text-binance-blue flex-shrink-0 mt-0.5" />
            <p className="text-[10px] text-binance-blue">
              API keys are encrypted and stored securely. Your keys are never logged or exposed in plaintext.
            </p>
          </div>
        </div>
      </div>

      {/* Confirm Delete Dialog */}
      <ConfirmDialog
        isOpen={deleteConfirm.isOpen}
        onClose={() => setDeleteConfirm({ isOpen: false, id: null })}
        onConfirm={() => deleteConfirm.id && deleteApiKey(deleteConfirm.id)}
        title="Delete API Key"
        message="Are you sure you want to delete this API key? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
      />

      {/* Alert Dialog */}
      <AlertDialog
        isOpen={alertDialog.isOpen}
        onClose={() => setAlertDialog({ ...alertDialog, isOpen: false })}
        title={alertDialog.title}
        message={alertDialog.message}
        variant={alertDialog.variant}
      />
    </div>
  );
};
