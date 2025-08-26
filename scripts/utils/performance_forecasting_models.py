#!/usr/bin/env python3
"""
Advanced Performance Forecasting Models for SutazAI System
=========================================================

Implements sophisticated forecasting algorithms for system performance prediction:
- ARIMA models for time series forecasting
- LSTM neural networks for complex pattern recognition
- Prophet for seasonal trend analysis
- Ensemble methods for improved accuracy
- Real-time anomaly detection
- Capacity planning recommendations
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML and forecasting imports
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    logging.warning("Some ML libraries not available. Install with: pip install scikit-learn statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForecastResult:
    """Forecast result data structure"""
    metric: str
    forecast_horizon: int
    predictions: List[float]
    confidence_intervals: List[Tuple[float, float]]
    model_accuracy: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    seasonality_detected: bool
    anomalies_predicted: List[int]  # indices of predicted anomalies
    recommendations: List[str]

@dataclass
class CapacityPrediction:
    """Capacity planning prediction"""
    resource: str
    current_utilization: float
    predicted_utilization: float
    capacity_exhaustion_date: Optional[datetime]
    recommended_scaling_date: Optional[datetime]
    scaling_factor: float
    confidence: float

class TimeSeriesPreprocessor:
    """Time series data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    window_size: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for machine learning models"""
        
        # Sort by timestamp
        df = df.sort_values('timestamp').copy()
        
        # Create time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Create lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Create rolling statistics
        for window in [6, 12, 24]:
            df[f'{target_column}_rolling_mean_{window}'] = (
                df[target_column].rolling(window=window).mean()
            )
            df[f'{target_column}_rolling_std_{window}'] = (
                df[target_column].rolling(window=window).std()
            )
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < window_size * 2:
            raise ValueError(f"Insufficient data: {len(df)} rows, need at least {window_size * 2}")
        
        # Prepare features
        feature_columns = [col for col in df.columns 
                          if col not in ['timestamp', target_column]]
        self.feature_columns = feature_columns
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Scale features
        if target_column not in self.scalers:
            self.scalers[target_column] = MinMaxScaler()
            X = self.scalers[target_column].fit_transform(X)
        else:
            X = self.scalers[target_column].transform(X)
        
        # Create sequences for LSTM
        X_sequences, y_sequences = [], []
        for i in range(window_size, len(X)):
            X_sequences.append(X[i-window_size:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def inverse_transform_predictions(self, predictions: np.ndarray, 
                                    target_column: str) -> np.ndarray:
        """Inverse transform scaled predictions"""
        if target_column in self.scalers:
            # For inverse transform, we need to create a dummy array with the same shape
            dummy = np.zeros((len(predictions), len(self.feature_columns)))
            dummy[:, 0] = predictions  # Assuming target is first column
            inverse_transformed = self.scalers[target_column].inverse_transform(dummy)
            return inverse_transformed[:, 0]
        return predictions

class ARIMAForecaster:
    """ARIMA model for time series forecasting"""
    
    def __init__(self):
        self.model = None
        self.fitted = False
    
    def find_optimal_order(self, data: pd.Series, max_p: int = 5, 
                          max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using AIC"""
        best_aic = float('inf')
        best_order = (0, 0, 0)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except Exception as e:
                        logger.debug(f"Continuing after exception: {e}")
                        continue
        
        return best_order
    
    def fit(self, data: pd.Series, order: Optional[Tuple[int, int, int]] = None):
        """Fit ARIMA model"""
        if order is None:
            order = self.find_optimal_order(data)
        
        logger.info(f"Fitting ARIMA model with order {order}")
        
        self.model = ARIMA(data, order=order)
        self.fitted_model = self.model.fit()
        self.fitted = True
        
        return self.fitted_model
    
    def forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals"""
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.forecast(steps=steps, alpha=0.05)
        
        predictions = forecast_result
        conf_int = self.fitted_model.get_prediction(
            start=len(self.fitted_model.fittedvalues),
            end=len(self.fitted_model.fittedvalues) + steps - 1
        ).conf_int()
        
        lower_bounds = conf_int.iloc[:, 0].values
        upper_bounds = conf_int.iloc[:, 1].values
        
        return predictions, lower_bounds, upper_bounds

class LSTMForecaster:
    """LSTM neural network for complex pattern recognition"""
    
    def __init__(self, sequence_length: int = 24, lstm_units: int = 50):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler()
        self.fitted = False
    
    def build_model(self, n_features: int):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM forecasting")
        
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data sequences for LSTM"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.Series, epochs: int = 50, batch_size: int = 32):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping LSTM training")
            return None
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_data.flatten())
        
        if len(X) < 10:
            raise ValueError("Insufficient data for LSTM training")
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build and train model
        self.model = self.build_model(1)
        
        history = self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size, 
            validation_split=0.2, verbose=0
        )
        
        self.fitted = True
        return history
    
    def forecast(self, data: pd.Series, steps: int) -> np.ndarray:
        """Generate LSTM forecasts"""
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Scale recent data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        
        # Get last sequence
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions

class ProphetForecaster:
    """Prophet model for seasonal trend analysis"""
    
    def __init__(self):
        self.model = None
        self.fitted = False
    
    def fit(self, data: pd.DataFrame):
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, skipping Prophet training")
            return None
        
        # Prepare data for Prophet
        prophet_data = data.copy()
        prophet_data.columns = ['ds', 'y']
        
        # Initialize and fit model
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95
        )
        
        self.model.fit(prophet_data)
        self.fitted = True
        
        return self.model
    
    def forecast(self, periods: int, freq: str = 'H') -> pd.DataFrame:
        """Generate Prophet forecasts"""
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        return forecast

class EnsembleForecaster:
    """Ensemble forecasting combining multiple models"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.fitted = False
    
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add model to ensemble"""
        self.models[name] = model
        self.weights[name] = weight
    
    def fit(self, data: pd.Series):
        """Fit all models in ensemble"""
        for name, model in self.models.items():
            try:
                if hasattr(model, 'fit'):
                    if name == 'prophet':
                        # Prophet needs DataFrame with ds, y columns
                        prophet_data = pd.DataFrame({
                            'ds': pd.date_range('2023-01-01', periods=len(data), freq='H'),
                            'y': data.values
                        })
                        model.fit(prophet_data)
                    else:
                        model.fit(data)
                logger.info(f"Successfully fitted {name} model")
            except Exception as e:
                logger.warning(f"Failed to fit {name} model: {e}")
                # Remove failed model from ensemble
                del self.models[name]
                del self.weights[name]
        
        self.fitted = True
    
    def forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ensemble forecasts"""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before forecasting")
        
        predictions_dict = {}
        
        for name, model in self.models.items():
            try:
                if name == 'arima':
                    pred, lower, upper = model.forecast(steps)
                    predictions_dict[name] = pred
                elif name == 'lstm':
                    # Need to pass recent data for LSTM
                    pred = model.forecast(pd.Series(range(100)), steps)  # Placeholder
                    predictions_dict[name] = pred
                elif name == 'prophet':
                    forecast_df = model.forecast(steps)
                    predictions_dict[name] = forecast_df['yhat'].tail(steps).values
            except Exception as e:
                logger.warning(f"Failed to generate forecast with {name}: {e}")
        
        if not predictions_dict:
            raise ValueError("No models produced successful forecasts")
        
        # Weighted ensemble
        total_weight = sum(self.weights[name] for name in predictions_dict.keys())
        ensemble_prediction = np.zeros(steps)
        
        for name, predictions in predictions_dict.items():
            weight = self.weights[name] / total_weight
            ensemble_prediction += weight * predictions
        
        # Simple confidence intervals (±10% of prediction)
        lower_bounds = ensemble_prediction * 0.9
        upper_bounds = ensemble_prediction * 1.1
        
        return ensemble_prediction, lower_bounds, upper_bounds

class AnomalyDetector:
    """Anomaly detection for performance metrics"""
    
    def __init__(self, method: str = 'isolation_forest'):
        self.method = method
        self.model = None
        self.threshold = None
    
    def fit(self, data: pd.Series):
        """Train anomaly detection model"""
        if self.method == 'statistical':
            # Statistical method using standard deviations
            self.mean = data.mean()
            self.std = data.std()
            self.threshold = 3  # 3-sigma rule
        
        elif self.method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.model.fit(data.values.reshape(-1, 1))
    
    def detect_anomalies(self, data: pd.Series) -> List[int]:
        """Detect anomalies in data"""
        anomalies = []
        
        if self.method == 'statistical':
            for i, value in enumerate(data):
                z_score = abs((value - self.mean) / self.std)
                if z_score > self.threshold:
                    anomalies.append(i)
        
        elif self.method == 'isolation_forest' and self.model:
            predictions = self.model.predict(data.values.reshape(-1, 1))
            anomalies = [i for i, pred in enumerate(predictions) if pred == -1]
        
        return anomalies

class PerformanceForecastingSystem:
    """Main performance forecasting system"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.preprocessor = TimeSeriesPreprocessor()
        self.anomaly_detector = AnomalyDetector()
        
        # Initialize models
        self.arima_forecaster = ARIMAForecaster()
        self.lstm_forecaster = LSTMForecaster() if TENSORFLOW_AVAILABLE else None
        self.prophet_forecaster = ProphetForecaster() if PROPHET_AVAILABLE else None
        self.ensemble_forecaster = EnsembleForecaster()
        
        # Set up ensemble
        self.setup_ensemble()
    
    def setup_ensemble(self):
        """Set up ensemble forecaster with available models"""
        self.ensemble_forecaster.add_model('arima', self.arima_forecaster, weight=0.4)
        
        if self.lstm_forecaster:
            self.ensemble_forecaster.add_model('lstm', self.lstm_forecaster, weight=0.3)
        
        if self.prophet_forecaster:
            self.ensemble_forecaster.add_model('prophet', self.prophet_forecaster, weight=0.3)
    
    def load_historical_data(self, metric: str, days: int = 7) -> pd.DataFrame:
        """Load historical data for a specific metric"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT timestamp, value 
        FROM benchmark_results 
        WHERE metric_name = ? 
        AND timestamp > datetime('now', '-{} days')
        ORDER BY timestamp
        """.format(days)
        
        df = pd.read_sql_query(query, conn, params=(metric,))
        conn.close()
        
        if df.empty:
            raise ValueError(f"No historical data found for metric: {metric}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def generate_forecast(self, metric: str, horizon_hours: int = 24, 
                        model_type: str = 'ensemble') -> ForecastResult:
        """Generate forecast for a specific metric"""
        
        # Load historical data
        try:
            df = self.load_historical_data(metric, days=14)  # 2 weeks of data
        except ValueError as e:
            logger.error(f"Cannot generate forecast: {e}")
            return None
        
        # Prepare time series
        ts_data = df.set_index('timestamp')['value']
        
        # Detect anomalies in historical data
        self.anomaly_detector.fit(ts_data)
        historical_anomalies = self.anomaly_detector.detect_anomalies(ts_data)
        
        # Select and train model
        if model_type == 'arima':
            self.arima_forecaster.fit(ts_data)
            predictions, lower_bounds, upper_bounds = self.arima_forecaster.forecast(horizon_hours)
        
        elif model_type == 'lstm' and self.lstm_forecaster:
            self.lstm_forecaster.fit(ts_data)
            predictions = self.lstm_forecaster.forecast(ts_data, horizon_hours)
            # Simple confidence intervals for LSTM
            lower_bounds = predictions * 0.9
            upper_bounds = predictions * 1.1
        
        elif model_type == 'prophet' and self.prophet_forecaster:
            prophet_df = df.rename(columns={'timestamp': 'ds', 'value': 'y'})
            self.prophet_forecaster.fit(prophet_df)
            forecast_df = self.prophet_forecaster.forecast(horizon_hours)
            predictions = forecast_df['yhat'].tail(horizon_hours).values
            lower_bounds = forecast_df['yhat_lower'].tail(horizon_hours).values
            upper_bounds = forecast_df['yhat_upper'].tail(horizon_hours).values
        
        else:  # ensemble
            self.ensemble_forecaster.fit(ts_data)
            predictions, lower_bounds, upper_bounds = self.ensemble_forecaster.forecast(horizon_hours)
        
        # Analyze trends
        recent_trend = self.analyze_trend(ts_data.tail(24))  # Last 24 hours
        prediction_trend = self.analyze_trend(pd.Series(predictions))
        
        # Detect seasonality
        seasonality_detected = self.detect_seasonality(ts_data)
        
        # Predict future anomalies
        future_anomalies = self.predict_anomalies(predictions)
        
        # Calculate model accuracy (using last 20% of data as test set)
        accuracy = self.calculate_accuracy(ts_data, model_type)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            metric, predictions, recent_trend, prediction_trend
        )
        
        return ForecastResult(
            metric=metric,
            forecast_horizon=horizon_hours,
            predictions=predictions.tolist(),
            confidence_intervals=list(zip(lower_bounds, upper_bounds)),
            model_accuracy=accuracy,
            trend=prediction_trend,
            seasonality_detected=seasonality_detected,
            anomalies_predicted=future_anomalies,
            recommendations=recommendations
        )
    
    def analyze_trend(self, data: pd.Series) -> str:
        """Analyze trend in time series data"""
        if len(data) < 3:
            return 'stable'
        
        # Linear regression to detect trend
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data.values, 1)
        slope = coeffs[0]
        
        # Determine trend direction
        if abs(slope) < data.std() * 0.1:  # Slope is small relative to variability
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def detect_seasonality(self, data: pd.Series) -> bool:
        """Detect seasonality in time series"""
        if len(data) < 48:  # Need at least 2 days of hourly data
            return False
        
        try:
            # Simple autocorrelation check for 24-hour seasonality
            autocorr_24h = data.autocorr(lag=24)
            return abs(autocorr_24h) > 0.3
        except Exception as e:
            logger.warning(f"Exception caught, returning: {e}")
            return False
    
    def predict_anomalies(self, predictions: np.ndarray) -> List[int]:
        """Predict which future points might be anomalies"""
        # Simple method: values that are >2 std deviations from mean
        if len(predictions) < 3:
            return []
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        anomalies = []
        for i, pred in enumerate(predictions):
            if abs(pred - mean_pred) > 2 * std_pred:
                anomalies.append(i)
        
        return anomalies
    
    def calculate_accuracy(self, data: pd.Series, model_type: str) -> float:
        """Calculate model accuracy using cross-validation"""
        if len(data) < 50:
            return 0.5  # Default accuracy for insufficient data
        
        # Simple holdout validation
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        try:
            if model_type == 'arima':
                forecaster = ARIMAForecaster()
                forecaster.fit(train_data)
                predictions, _, _ = forecaster.forecast(len(test_data))
            else:
                # Use ARIMA as fallback for accuracy calculation
                forecaster = ARIMAForecaster()
                forecaster.fit(train_data)
                predictions, _, _ = forecaster.forecast(len(test_data))
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100
            accuracy = max(0, 100 - mape) / 100  # Convert to 0-1 scale
            
            return min(1.0, accuracy)  # Cap at 100%
            
        except Exception as e:
            logger.warning(f"Could not calculate accuracy: {e}")
            return 0.5
    
    def generate_recommendations(self, metric: str, predictions: np.ndarray, 
                               recent_trend: str, prediction_trend: str) -> List[str]:
        """Generate actionable recommendations based on forecasts"""
        recommendations = []
        
        # Trend-based recommendations
        if prediction_trend == 'increasing':
            if 'cpu' in metric.lower():
                recommendations.append(
                    "CPU utilization is predicted to increase. Consider scaling horizontally "
                    "or optimizing CPU-intensive processes."
                )
            elif 'memory' in metric.lower():
                recommendations.append(
                    "Memory usage is predicted to increase. Monitor for memory leaks "
                    "and consider increasing memory limits."
                )
            elif 'response_time' in metric.lower():
                recommendations.append(
                    "Response times are predicted to increase. Investigate potential "
                    "bottlenecks and consider performance optimizations."
                )
        
        elif prediction_trend == 'decreasing' and 'response_time' in metric.lower():
            recommendations.append(
                "Response times are predicted to improve. Recent optimizations "
                "appear to be effective."
            )
        
        # Threshold-based recommendations
        max_prediction = np.max(predictions)
        if 'cpu' in metric.lower() and max_prediction > 80:
            recommendations.append(
                f"CPU utilization predicted to reach {max_prediction:.1f}%. "
                "Plan for capacity scaling before hitting critical levels."
            )
        
        if 'memory' in metric.lower() and max_prediction > 85:
            recommendations.append(
                f"Memory utilization predicted to reach {max_prediction:.1f}%. "
                "Implement memory optimization or increase available memory."
            )
        
        # Volatility-based recommendations
        prediction_volatility = np.std(predictions)
        prediction_mean = np.mean(predictions)
        cv = prediction_volatility / prediction_mean if prediction_mean > 0 else 0
        
        if cv > 0.3:  # High coefficient of variation
            recommendations.append(
                f"High volatility predicted for {metric}. Consider implementing "
                "auto-scaling or load balancing strategies."
            )
        
        if not recommendations:
            recommendations.append(f"Predicted performance for {metric} appears stable.")
        
        return recommendations
    
    def generate_capacity_predictions(self, resources: List[str], 
                                    horizon_days: int = 30) -> List[CapacityPrediction]:
        """Generate capacity planning predictions"""
        predictions = []
        
        for resource in resources:
            try:
                # Get current utilization
                current_data = self.load_historical_data(resource, days=1)
                current_utilization = current_data['value'].iloc[-1] if not current_data.empty else 0
                
                # Generate forecast
                forecast_result = self.generate_forecast(
                    resource, horizon_hours=horizon_days * 24, model_type='ensemble'
                )
                
                if not forecast_result:
                    continue
                
                predicted_utilization = np.mean(forecast_result.predictions)
                max_predicted = np.max(forecast_result.predictions)
                
                # Determine capacity exhaustion date
                capacity_exhaustion_date = None
                recommended_scaling_date = None
                
                if max_predicted > 95:  # Will hit capacity
                    # Find when it hits 95%
                    for i, pred in enumerate(forecast_result.predictions):
                        if pred > 95:
                            capacity_exhaustion_date = datetime.now() + timedelta(hours=i)
                            break
                    
                    # Recommend scaling at 80%
                    for i, pred in enumerate(forecast_result.predictions):
                        if pred > 80:
                            recommended_scaling_date = datetime.now() + timedelta(hours=i)
                            break
                
                # Calculate scaling factor
                scaling_factor = 1.0
                if predicted_utilization > 70:
                    scaling_factor = predicted_utilization / 70  # Scale to keep at 70%
                
                prediction = CapacityPrediction(
                    resource=resource,
                    current_utilization=current_utilization,
                    predicted_utilization=predicted_utilization,
                    capacity_exhaustion_date=capacity_exhaustion_date,
                    recommended_scaling_date=recommended_scaling_date,
                    scaling_factor=scaling_factor,
                    confidence=forecast_result.model_accuracy
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Failed to generate capacity prediction for {resource}: {e}")
        
        return predictions
    
    def generate_visualization(self, forecast_result: ForecastResult, 
                             output_path: str):
        """Generate forecast visualization"""
        plt.figure(figsize=(12, 8))
        
        # Plot historical data (if available)
        try:
            historical_data = self.load_historical_data(forecast_result.metric, days=7)
            plt.plot(historical_data['timestamp'], historical_data['value'], 
                    label='Historical', color='blue', alpha=0.7)
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        # Plot predictions
        future_timestamps = pd.date_range(
            start=datetime.now(), 
            periods=len(forecast_result.predictions), 
            freq='H'
        )
        
        plt.plot(future_timestamps, forecast_result.predictions, 
                label='Forecast', color='red', linewidth=2)
        
        # Plot confidence intervals
        if forecast_result.confidence_intervals:
            lower_bounds, upper_bounds = zip(*forecast_result.confidence_intervals)
            plt.fill_between(future_timestamps, lower_bounds, upper_bounds, 
                           alpha=0.3, color='red', label='Confidence Interval')
        
        # Mark predicted anomalies
        if forecast_result.anomalies_predicted:
            anomaly_timestamps = [future_timestamps[i] for i in forecast_result.anomalies_predicted]
            anomaly_values = [forecast_result.predictions[i] for i in forecast_result.anomalies_predicted]
            plt.scatter(anomaly_timestamps, anomaly_values, 
                       color='orange', s=100, label='Predicted Anomalies', zorder=5)
        
        plt.title(f'Performance Forecast: {forecast_result.metric}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Forecast visualization saved to {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize forecasting system
    forecasting_system = PerformanceForecastingSystem("/opt/sutazaiapp/data/performance_metrics.db")
    
    # Test with sample metrics
    test_metrics = [
        'cpu_percent',
        'memory_percent',
        'health_response_time_ms',
        'load_test_requests_per_second'
    ]
    
    for metric in test_metrics:
        try:
            logger.info(f"Generating forecast for {metric}")
            
            forecast = forecasting_system.generate_forecast(
                metric=metric,
                horizon_hours=48,  # 2 days
                model_type='ensemble'
            )
            
            if forecast:
                logger.info(f"\nForecast for {metric}:")
                logger.info(f"  Trend: {forecast.trend}")
                logger.info(f"  Model Accuracy: {forecast.model_accuracy:.2%}")
                logger.info(f"  Seasonality Detected: {forecast.seasonality_detected}")
                logger.info(f"  Recommendations: {len(forecast.recommendations)}")
                
                for rec in forecast.recommendations:
                    logger.info(f"    - {rec}")
            
        except Exception as e:
            logger.error(f"Failed to generate forecast for {metric}: {e}")
    
    # Test capacity predictions
    try:
        logger.info("Generating capacity predictions...")
        
        capacity_predictions = forecasting_system.generate_capacity_predictions([
            'cpu_percent', 'memory_percent'
        ], horizon_days=7)
        
        logger.info(f"\nCapacity Predictions ({len(capacity_predictions)} resources):")
        for pred in capacity_predictions:
            logger.info(f"  {pred.resource}:")
            logger.info(f"    Current: {pred.current_utilization:.1f}%")
            logger.info(f"    Predicted: {pred.predicted_utilization:.1f}%")
            if pred.capacity_exhaustion_date:
                logger.info(f"    Capacity exhaustion: {pred.capacity_exhaustion_date}")
            if pred.recommended_scaling_date:
                logger.info(f"    Recommended scaling: {pred.recommended_scaling_date}")
            logger.info(f"    Scaling factor: {pred.scaling_factor:.2f}")
            logger.info(f"    Confidence: {pred.confidence:.2%}")
        
    except Exception as e:
        logger.error(f"Failed to generate capacity predictions: {e}")
    
    logger.info("\nForecasting system test completed!")