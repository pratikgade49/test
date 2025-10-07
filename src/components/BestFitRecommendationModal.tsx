import React from 'react';
import { X, Brain, Target, Clock, TrendingUp, CheckCircle, AlertCircle } from 'lucide-react';
import { BestFitRecommendationResponse } from '../services/api';

interface BestFitRecommendationModalProps {
  isOpen: boolean;
  onClose: () => void;
  recommendation: BestFitRecommendationResponse;
  onApplyRecommendation: (algorithm: string) => void;
}

export const BestFitRecommendationModal: React.FC<BestFitRecommendationModalProps> = ({
  isOpen,
  onClose,
  recommendation,
  onApplyRecommendation
}) => {
  if (!isOpen) return null;

  const getAlgorithmDisplayName = (algorithm: string) => {
    const algorithmMap: Record<string, string> = {
      'linear_regression': 'Linear Regression',
      'polynomial_regression': 'Polynomial Regression',
      'exponential_smoothing': 'Exponential Smoothing',
      'holt_winters': 'Holt-Winters',
      'arima': 'ARIMA',
      'random_forest': 'Random Forest',
      'seasonal_decomposition': 'Seasonal Decomposition',
      'moving_average': 'Moving Average',
      'sarima': 'SARIMA',
      'prophet_like': 'Prophet-like Forecasting',
      'lstm_like': 'LSTM-like',
      'xgboost': 'XGBoost',
      'svr': 'Support Vector Regression',
      'knn': 'K-Nearest Neighbors',
      'gaussian_process': 'Gaussian Process',
      'neural_network': 'Neural Network',
      'theta_method': 'Theta Method',
      'croston': 'Croston\'s Method',
      'ses': 'Simple Exponential Smoothing',
      'damped_trend': 'Damped Trend',
      'naive_seasonal': 'Naive Seasonal',
      'drift_method': 'Drift Method'
    };
    return algorithmMap[algorithm] || algorithm;
  };

  const getConfidenceColor = (confidence?: number) => {
    if (!confidence) return 'text-gray-600';
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceIcon = (confidence?: number) => {
    if (!confidence) return <AlertCircle className="w-5 h-5 text-gray-500" />;
    if (confidence >= 80) return <CheckCircle className="w-5 h-5 text-green-500" />;
    if (confidence >= 60) return <Target className="w-5 h-5 text-yellow-500" />;
    return <AlertCircle className="w-5 h-5 text-red-500" />;
  };

  const handleApply = () => {
    if (recommendation.recommended_algorithm) {
      onApplyRecommendation(recommendation.recommended_algorithm);
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl w-full max-w-md mx-4 shadow-2xl">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Brain className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-semibold text-gray-900">Algorithm Recommendation</h2>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {recommendation.recommended_algorithm ? (
            <div className="space-y-4">
              {/* Recommendation Card */}
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center space-x-3 mb-3">
                  <Target className="w-6 h-6 text-blue-600" />
                  <h3 className="font-semibold text-blue-900">Recommended Algorithm</h3>
                </div>
                <p className="text-lg font-bold text-blue-800 mb-2">
                  {getAlgorithmDisplayName(recommendation.recommended_algorithm)}
                </p>
                <p className="text-sm text-blue-700">
                  {recommendation.message}
                </p>
              </div>

              {/* Performance Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-green-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-green-600">Last Accuracy</p>
                      <p className="text-xl font-bold text-green-900">
                        {recommendation.last_accuracy}%
                      </p>
                    </div>
                    <TrendingUp className="w-6 h-6 text-green-500" />
                  </div>
                </div>

                <div className="bg-purple-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-purple-600">Confidence</p>
                      <p className={`text-xl font-bold ${getConfidenceColor(recommendation.confidence)}`}>
                        {recommendation.confidence}%
                      </p>
                    </div>
                    {getConfidenceIcon(recommendation.confidence)}
                  </div>
                </div>
              </div>

              {/* Last Run Info */}
              {recommendation.last_run_date && (
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="flex items-center space-x-2 text-sm text-gray-600">
                    <Clock className="w-4 h-4" />
                    <span>Last run: {new Date(recommendation.last_run_date).toLocaleDateString()}</span>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex items-center justify-end space-x-3 pt-4">
                <button
                  onClick={onClose}
                  className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  Keep Best Fit
                </button>
                <button
                  onClick={handleApply}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Use Recommendation
                </button>
              </div>

              {/* Info Note */}
              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-xs text-blue-800">
                  <strong>Note:</strong> This recommendation is based on previous "best fit" runs with similar configurations. 
                  Using the recommended algorithm will skip the comparison of all 23 algorithms and run only the suggested one.
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* No Recommendation Available */}
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-center space-x-3 mb-3">
                  <AlertCircle className="w-6 h-6 text-yellow-600" />
                  <h3 className="font-semibold text-yellow-900">No Recommendation Available</h3>
                </div>
                <p className="text-sm text-yellow-800">
                  {recommendation.message}
                </p>
              </div>

              {/* Action Button */}
              <div className="flex items-center justify-end pt-4">
                <button
                  onClick={onClose}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Continue with Best Fit
                </button>
              </div>

              {/* Info Note */}
              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-xs text-blue-800">
                  <strong>Tip:</strong> After running "best fit" for the first time with this configuration, 
                  future runs will show algorithm recommendations based on past performance.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};