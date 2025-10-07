import React, { useState, useEffect } from 'react';
import {
  Clock, Plus, Play, Pause, Trash2, Eye, Calendar, Settings,
  CheckCircle, XCircle, AlertCircle, RefreshCw, X, Edit
} from 'lucide-react';
import { ApiService, ScheduledForecast, ForecastExecution, ForecastConfig } from '../services/api';

// Utility function to format date/time for display
function formatDateTime(dateString: string | Date): string {
  const date = typeof dateString === "string" ? new Date(dateString) : dateString;
  if (isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

interface ScheduledForecastManagerProps {
  isOpen: boolean;
  onClose: () => void;
  currentConfig?: ForecastConfig;
  productOptions: string[];
  customerOptions: string[];
  locationOptions: string[];
}

export const ScheduledForecastManager: React.FC<ScheduledForecastManagerProps> = ({
  isOpen,
  onClose,
  currentConfig,
  productOptions,
  customerOptions,
  locationOptions
}) => {
  const [scheduledForecasts, setScheduledForecasts] = useState<ScheduledForecast[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showExecutionHistory, setShowExecutionHistory] = useState<number | null>(null);
  const [executions, setExecutions] = useState<ForecastExecution[]>([]);
  const [schedulerStatus, setSchedulerStatus] = useState<{ running: boolean; check_interval: number } | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadScheduledForecasts();
      loadSchedulerStatus();
    }
  }, [isOpen]);

  const loadScheduledForecasts = async () => {
    setLoading(true);
    setError(null);
    try {
      const forecasts = await ApiService.getScheduledForecasts();
      setScheduledForecasts(forecasts);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load scheduled forecasts');
    } finally {
      setLoading(false);
    }
  };

  const loadSchedulerStatus = async () => {
    try {
      const status = await ApiService.getSchedulerStatus();
      setSchedulerStatus(status);
    } catch (err) {
      console.error('Failed to load scheduler status:', err);
    }
  };

  const handleDeleteForecast = async (id: number, name: string) => {
    if (!confirm(`Are you sure you want to delete the scheduled forecast "${name}"?`)) {
      return;
    }

    try {
      await ApiService.deleteScheduledForecast(id);
      await loadScheduledForecasts();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete scheduled forecast');
    }
  };

  const handleToggleStatus = async (forecast: ScheduledForecast) => {
    const newStatus = forecast.status === 'active' ? 'paused' : 'active';
    
    try {
      await ApiService.updateScheduledForecast(forecast.id, { status: newStatus });
      await loadScheduledForecasts();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update forecast status');
    }
  };

  const loadExecutionHistory = async (forecastId: number) => {
    try {
      const executions = await ApiService.getForecastExecutions(forecastId);
      setExecutions(executions);
      setShowExecutionHistory(forecastId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load execution history');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'paused':
        return <Pause className="w-4 h-4 text-yellow-500" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-blue-500" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800';
      case 'completed':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getFrequencyDisplay = (frequency: string) => {
    return frequency.charAt(0).toUpperCase() + frequency.slice(1);
  };

  const getSuccessRate = (forecast: ScheduledForecast) => {
    if (forecast.run_count === 0) return 0;
    return Math.round((forecast.success_count / forecast.run_count) * 100);
  };

  const getForecastConfigSummary = (config: ForecastConfig) => {
    if (config.selectedProduct && config.selectedCustomer && config.selectedLocation) {
      return `${config.selectedProduct} → ${config.selectedCustomer} → ${config.selectedLocation}`;
    } else if (config.multiSelect) {
      const dimensions = [];
      if (config.selectedProducts && config.selectedProducts.length > 0) {
        dimensions.push(`${config.selectedProducts.length} Products`);
      }
      if (config.selectedCustomers && config.selectedCustomers.length > 0) {
        dimensions.push(`${config.selectedCustomers.length} Customers`);
      }
      if (config.selectedLocations && config.selectedLocations.length > 0) {
        dimensions.push(`${config.selectedLocations.length} Locations`);
      }
      return dimensions.join(' × ') || 'Multi-select';
    } else if (config.selectedItems && config.selectedItems.length > 1) {
      return `${config.selectedItems.length} ${config.forecastBy}s`;
    } else if (config.selectedItem) {
      return `${config.forecastBy}: ${config.selectedItem}`;
    }
    return 'Not configured';
  };



  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-7xl mx-4 h-[90vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <Clock className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">Scheduled Forecasts</h2>
            <span className="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full">
              {scheduledForecasts.length} schedules
            </span>
            {schedulerStatus && (
              <span className={`text-sm px-3 py-1 rounded-full ${
                schedulerStatus.running 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                Scheduler: {schedulerStatus.running ? 'Running' : 'Stopped'}
              </span>
            )}
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Schedule Forecast</span>
            </button>
            
            <button
              onClick={loadScheduledForecasts}
              disabled={loading}
              className="flex items-center space-x-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
            
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <AlertCircle className="w-4 h-4 text-red-500 mr-2" />
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
            </div>
          ) : scheduledForecasts.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Clock className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Scheduled Forecasts</h3>
                <p className="text-gray-600 mb-4">
                  Create your first scheduled forecast to automate your forecasting process
                </p>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Schedule Your First Forecast
                </button>
              </div>
            </div>
          ) : (
            <div className="overflow-y-auto h-full">
              <div className="p-6 space-y-4">
                {scheduledForecasts.map((forecast) => (
                  <div
                    key={forecast.id}
                    className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <h3 className="font-semibold text-gray-900">{forecast.name}</h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(forecast.status)}`}>
                            <div className="flex items-center space-x-1">
                              {getStatusIcon(forecast.status)}
                              <span>{forecast.status.charAt(0).toUpperCase() + forecast.status.slice(1)}</span>
                            </div>
                          </span>
                          <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                            {getFrequencyDisplay(forecast.frequency)}
                          </span>
                        </div>
                        
                        {forecast.description && (
                          <p className="text-sm text-gray-600 mb-3">{forecast.description}</p>
                        )}
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500">Next Run:</span>
                            <p className="font-medium text-gray-900">
                              {formatDateTime(forecast.next_run)}
                            </p>
                          </div>
                          <div>
                            <span className="text-gray-500">Last Run:</span>
                            <p className="font-medium text-gray-900">
                              {forecast.last_run ? formatDateTime(forecast.last_run) : 'Never'}
                            </p>
                          </div>
                          <div>
                            <span className="text-gray-500">Configuration:</span>
                            <p className="font-medium text-gray-900">
                              {getForecastConfigSummary(forecast.forecast_config)}
                            </p>
                          </div>
                          <div>
                            <span className="text-gray-500">Success Rate:</span>
                            <p className="font-medium text-gray-900">
                              {getSuccessRate(forecast)}% ({forecast.success_count}/{forecast.run_count})
                            </p>
                          </div>
                        </div>
                        
                        <div className="mt-2 text-xs text-gray-500">
                          <span className="font-medium">Algorithm:</span> {forecast.forecast_config.algorithm.replace('_', ' ')}
                          <span className="ml-4 font-medium">Periods:</span> {forecast.forecast_config.historicPeriod}H / {forecast.forecast_config.forecastPeriod}F
                          <span className="ml-4 font-medium">Interval:</span> {forecast.forecast_config.interval}
                        </div>
                        
                        {forecast.last_error && (
                          <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm">
                            <span className="font-medium text-red-800">Last Error:</span>
                            <p className="text-red-700 mt-1">{forecast.last_error}</p>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex items-center space-x-2 ml-4">
                        <button
                          onClick={() => loadExecutionHistory(forecast.id)}
                          className="flex items-center space-x-1 px-3 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                          <span>History</span>
                        </button>
                        
                        <button
                          onClick={() => handleToggleStatus(forecast)}
                          className={`flex items-center space-x-1 px-3 py-2 rounded-lg transition-colors ${
                            forecast.status === 'active'
                              ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                              : 'bg-green-600 hover:bg-green-700 text-white'
                          }`}
                        >
                          {forecast.status === 'active' ? (
                            <>
                              <Pause className="w-4 h-4" />
                              <span>Pause</span>
                            </>
                          ) : (
                            <>
                              <Play className="w-4 h-4" />
                              <span>Resume</span>
                            </>
                          )}
                        </button>
                        
                        <button
                          onClick={() => handleDeleteForecast(forecast.id, forecast.name)}
                          className="flex items-center space-x-1 px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                          <span>Delete</span>
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Create Modal */}
        {showCreateModal && (
          <CreateScheduledForecastModal
            isOpen={showCreateModal}
            onClose={() => setShowCreateModal(false)}
            onSuccess={() => {
              setShowCreateModal(false);
              loadScheduledForecasts();
            }}
            currentConfig={currentConfig}
            productOptions={productOptions}
            customerOptions={customerOptions}
            locationOptions={locationOptions}
          />
        )}

        {/* Execution History Modal */}
        {showExecutionHistory && (
          <ExecutionHistoryModal
            isOpen={!!showExecutionHistory}
            onClose={() => setShowExecutionHistory(null)}
            executions={executions}
            forecastName={scheduledForecasts.find(f => f.id === showExecutionHistory)?.name || ''}
          />
        )}
      </div>
    </div>
  );
};

// Create Scheduled Forecast Modal Component
interface CreateScheduledForecastModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  currentConfig?: ForecastConfig;
  productOptions: string[];
  customerOptions: string[];
  locationOptions: string[];
}

const CreateScheduledForecastModal: React.FC<CreateScheduledForecastModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
  currentConfig,
  productOptions,
  customerOptions,
  locationOptions
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [frequency, setFrequency] = useState<'daily' | 'weekly' | 'monthly'>('daily');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  
  // Forecast configuration state
  const [forecastBy, setForecastBy] = useState<'product' | 'customer' | 'location'>('product');
  const [selectedItem, setSelectedItem] = useState('');
  const [selectedProduct, setSelectedProduct] = useState('');
  const [selectedCustomer, setSelectedCustomer] = useState('');
  const [selectedLocation, setSelectedLocation] = useState('');
  const [algorithm, setAlgorithm] = useState('best_fit');
  const [interval, setInterval] = useState('month');
  const [historicPeriod, setHistoricPeriod] = useState(12);
  const [forecastPeriod, setForecastPeriod] = useState(6);
  const [advancedMode, setAdvancedMode] = useState(false);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      // Set default start date to tomorrow
      const tomorrow = new Date();
      tomorrow.setDate(tomorrow.getDate() + 1);
      setStartDate(tomorrow.toISOString().slice(0, 16));
      
      // Initialize forecast configuration from currentConfig if provided
      if (currentConfig) {
        setForecastBy(currentConfig.forecastBy as 'product' | 'customer' | 'location');
        setSelectedItem(currentConfig.selectedItem || '');
        setSelectedProduct(currentConfig.selectedProduct || '');
        setSelectedCustomer(currentConfig.selectedCustomer || '');
        setSelectedLocation(currentConfig.selectedLocation || '');
        setAlgorithm(currentConfig.algorithm || 'best_fit');
        setInterval(currentConfig.interval || 'month');
        setHistoricPeriod(currentConfig.historicPeriod || 12);
        setForecastPeriod(currentConfig.forecastPeriod || 6);
        setAdvancedMode(!!(currentConfig.selectedProduct && currentConfig.selectedCustomer && currentConfig.selectedLocation));
      }
    }
  }, [isOpen]);

  const getOptionsForForecastBy = () => {
    switch (forecastBy) {
      case 'product':
        return productOptions;
      case 'customer':
        return customerOptions;
      case 'location':
        return locationOptions;
      default:
        return [];
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name.trim()) {
      setError('Schedule name is required');
      return;
    }

    if (!startDate) {
      setError('Start date is required');
      return;
    }

    // Validate forecast configuration
    if (advancedMode) {
      if (!selectedProduct || !selectedCustomer || !selectedLocation) {
        setError('Please select Product, Customer, and Location for advanced mode');
        return;
      }
    } else {
      if (!selectedItem) {
        setError('Please select an item to forecast');
        return;
      }
    }

    setLoading(true);
    setError(null);

    try {
      // Build forecast configuration
      const forecastConfig: ForecastConfig = {
        forecastBy,
        algorithm,
        interval,
        historicPeriod,
        forecastPeriod,
        multiSelect: false,
        ...(advancedMode ? {
          selectedProduct,
          selectedCustomer,
          selectedLocation
        } : {
          selectedItem
        })
      };

      await ApiService.createScheduledForecast({
        name: name.trim(),
        description: description.trim() || undefined,
        forecast_config: forecastConfig,
        frequency,
        start_date: startDate,
        end_date: endDate || undefined
      });

      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create scheduled forecast');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-md mx-4 shadow-2xl flex flex-col" style={{ maxHeight: '90vh' }}>
        <div className="p-6 overflow-y-auto" style={{ maxHeight: '80vh' }}>
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900">Schedule Forecast</h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center">
                <AlertCircle className="w-4 h-4 text-red-500 mr-2" />
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Schedule Name *
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., Daily Product A Forecast"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of this scheduled forecast..."
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Frequency *
              </label>
              <select
                value={frequency}
                onChange={(e) => setFrequency(e.target.value as any)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>

            {/* Forecast Configuration Section */}
            <div className="border-t border-gray-200 pt-4">
              <h4 className="font-medium text-gray-900 mb-4">Forecast Configuration</h4>
              
              <div className="space-y-4">
                {/* Forecast By */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Forecast By *
                  </label>
                  <select
                    value={forecastBy}
                    onChange={(e) => {
                      setForecastBy(e.target.value as 'product' | 'customer' | 'location');
                      setSelectedItem(''); // Reset selection when changing forecast by
                    }}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    required
                  >
                    <option value="product">Product</option>
                    <option value="customer">Customer</option>
                    <option value="location">Location</option>
                  </select>
                </div>

                {/* Advanced Mode Toggle */}
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="advancedMode"
                    checked={advancedMode}
                    onChange={(e) => setAdvancedMode(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <label htmlFor="advancedMode" className="text-sm font-medium text-gray-700">
                    Advanced Mode (Select Product + Customer + Location)
                  </label>
                </div>

                {/* Selection Fields */}
                {advancedMode ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Product *
                      </label>
                      <select
                        value={selectedProduct}
                        onChange={(e) => setSelectedProduct(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        required
                      >
                        <option value="">Select Product</option>
                        {productOptions.map(option => (
                          <option key={option} value={option}>{option}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Customer *
                      </label>
                      <select
                        value={selectedCustomer}
                        onChange={(e) => setSelectedCustomer(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        required
                      >
                        <option value="">Select Customer</option>
                        {customerOptions.map(option => (
                          <option key={option} value={option}>{option}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Location *
                      </label>
                      <select
                        value={selectedLocation}
                        onChange={(e) => setSelectedLocation(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        required
                      >
                        <option value="">Select Location</option>
                        {locationOptions.map(option => (
                          <option key={option} value={option}>{option}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                ) : (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Select {forecastBy.charAt(0).toUpperCase() + forecastBy.slice(1)} *
                    </label>
                    <select
                      value={selectedItem}
                      onChange={(e) => setSelectedItem(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      required
                    >
                      <option value="">Select {forecastBy}</option>
                      {getOptionsForForecastBy().map(option => (
                        <option key={option} value={option}>{option}</option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Algorithm */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Algorithm
                  </label>
                  <select
                    value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="best_fit">Best Fit (Automatic)</option>
                    <option value="linear_regression">Linear Regression</option>
                    <option value="polynomial_regression">Polynomial Regression</option>
                    <option value="exponential_smoothing">Exponential Smoothing</option>
                    <option value="holt_winters">Holt-Winters</option>
                    <option value="arima">ARIMA</option>
                    <option value="random_forest">Random Forest</option>
                    <option value="seasonal_decomposition">Seasonal Decomposition</option>
                    <option value="moving_average">Moving Average</option>
                  </select>
                </div>

                {/* Time Configuration */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Interval
                    </label>
                    <select
                      value={interval}
                      onChange={(e) => setInterval(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="week">Weekly</option>
                      <option value="month">Monthly</option>
                      <option value="year">Yearly</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Historic Periods
                    </label>
                    <input
                      type="number"
                      value={historicPeriod}
                      onChange={(e) => setHistoricPeriod(parseInt(e.target.value) || 12)}
                      min="1"
                      max="100"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Forecast Periods
                    </label>
                    <input
                      type="number"
                      value={forecastPeriod}
                      onChange={(e) => setForecastPeriod(parseInt(e.target.value) || 6)}
                      min="1"
                      max="50"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Start Date & Time *
              </label>
              <input
                type="datetime-local"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                End Date & Time (Optional)
              </label>
              <input
                type="datetime-local"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">
                Leave empty to run indefinitely
              </p>
            </div>

            <div className="flex items-center justify-end space-x-3 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                    <span>Creating...</span>
                  </div>
                ) : (
                  'Create Schedule'
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

// Execution History Modal Component
interface ExecutionHistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  executions: ForecastExecution[];
  forecastName: string;
}

const ExecutionHistoryModal: React.FC<ExecutionHistoryModalProps> = ({
  isOpen,
  onClose,
  executions,
  forecastName
}) => {
  const getExecutionStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'running':
        return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-4xl mx-4 h-[80vh] flex flex-col shadow-2xl">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">
            Execution History: {forecastName}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {executions.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Clock className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Execution History</h3>
                <p className="text-gray-600 mb-4">
                  This scheduled forecast has not been executed yet
                </p>
                <button
                  onClick={onClose}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          ) : (
            <div className="p-6 space-y-4">
              {executions.map((execution) => (
                <div
                  key={execution.id}
                  className="border border-gray-200 rounded-lg p-4"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        {getExecutionStatusIcon(execution.status)}
                        <span className="font-medium text-gray-900">
                          {formatDateTime(execution.execution_time)}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          execution.status === 'success' ? 'bg-green-100 text-green-800' :
                          execution.status === 'failed' ? 'bg-red-100 text-red-800' :
                          'bg-blue-100 text-blue-800'
                        }`}>
                          {execution.status.charAt(0).toUpperCase() + execution.status.slice(1)}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Duration:</span>
                          <p className="font-medium text-gray-900">
                            {formatDuration(execution.duration_seconds)}
                          </p>
                        </div>
                        {execution.result_summary && (
                          <div>
                            <span className="text-gray-500">Result:</span>
                            <p className="font-medium text-gray-900">
                              {execution.result_summary.type === 'multi_forecast' 
                                ? `${execution.result_summary.successful}/${execution.result_summary.total_combinations} successful`
                                : `${execution.result_summary.accuracy}% accuracy`
                              }
                            </p>
                          </div>
                        )}
                      </div>
                      
                      {execution.error_message && (
                        <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm">
                          <span className="font-medium text-red-800">Error:</span>
                          <p className="text-red-700 mt-1">{execution.error_message}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};