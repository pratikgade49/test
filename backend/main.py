#!/usr/bin/env python3
"""
Advanced Multi-variant Forecasting API with MySQL Database Integration
Now includes automated forecast scheduling and modular router structure
"""

from database import get_db, init_database, ForecastData, User, ExternalFactorData, ForecastConfiguration
from database import ProductDimension, CustomerDimension, LocationDimension, DimensionManager
from model_persistence import ModelPersistenceManager, SavedModel, ModelAccuracyHistory
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import warnings
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct, and_, or_
from database import SavedForecastResult
from auth import create_access_token, get_current_user, get_current_user_optional
from validation import DateRangeValidator
import requests
import logging
from scheduler import (
    ScheduledForecast, ForecastExecution, ScheduleFrequency, ScheduleStatus,
    start_scheduler, stop_scheduler, get_scheduler_status
)
from sqlalchemy.exc import IntegrityError

warnings.filterwarnings('ignore')

app = FastAPI(title="Multi-variant Forecasting API with MySQL", version="3.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import and mount AI Chat router
# try:
#     from ai_chat.router import router as ai_chat_router
#     app.include_router(ai_chat_router)
# except Exception as e:
#     print(f"⚠️  AI Chat router not mounted: {e}")

# FRED API Configuration
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY", "82a8e6191d71f41b22cf33bf73f7a0c2")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Multi-variant Forecasting API...")
    start_scheduler()
    logger.info("Forecast scheduler started")
    
    if init_database():
        print("✅ Database initialization successful!")
        try:
            from database import engine
            SavedModel.metadata.create_all(bind=engine)
            ModelAccuracyHistory.metadata.create_all(bind=engine)
            print("✅ Model persistence tables initialized!")
        except Exception as e:
            print(f"⚠️  Model persistence table creation failed: {e}")
    else:
        print("⚠️  Database initialization failed - some features may not work")
        print("Please run: python setup_database.py")

@app.on_event("shutdown")
async def shutdown_event():
    stop_scheduler()
    logger.info("Forecast scheduler stopped")

# Import routers
from routes import (
    auth as auth_router,
    forecasting as forecasting_router,
    data_management as data_management_router,
    configurations as configurations_router,
    scheduled_forecasts as scheduled_forecasts_router,
    external_factors as external_factors_router,
    admin as admin_router,
    saved_forecasts as saved_forecasts_router,
    exports as exports_router,
    model_cache as model_cache_router
)

# Include routers
app.include_router(auth_router.router, prefix="/auth", tags=["Authentication"])
app.include_router(forecasting_router.router, tags=["Forecasting"])
app.include_router(data_management_router.router, prefix="/database", tags=["Data Management"])
app.include_router(configurations_router.router, prefix="/configurations", tags=["Configurations"])
app.include_router(scheduled_forecasts_router.router, prefix="/scheduled_forecasts", tags=["Scheduled Forecasts"])
app.include_router(external_factors_router.router, tags=["External Factors"])
app.include_router(admin_router.router, prefix="/admin", tags=["Admin"])
app.include_router(saved_forecasts_router.router, prefix="/saved_forecasts", tags=["Saved Forecasts"])
app.include_router(exports_router.router, tags=["Exports"])
app.include_router(model_cache_router.router, tags=["Model Cache"])

# Keep only the essential models and ForecastingEngine class here
class DataPoint(BaseModel):
    date: str
    quantity: float
    period: str

class AlgorithmResult(BaseModel):
    algorithm: str
    accuracy: float
    mae: float
    rmse: float
    historicData: List[DataPoint]
    forecastData: List[DataPoint]
    trend: str

class ForecastResult(BaseModel):
    combination: Optional[Dict[str, str]] = None
    selectedAlgorithm: str
    accuracy: float
    mae: float
    rmse: float
    historicData: List[DataPoint]
    forecastData: List[DataPoint]
    trend: str
    allAlgorithms: Optional[List[AlgorithmResult]] = None
    processLog: Optional[List[str]] = None
    configHash: Optional[str] = None

class MultiForecastResult(BaseModel):
    results: List[ForecastResult]
    totalCombinations: int
    summary: Dict[str, Any]
    processLog: Optional[List[str]] = None

class ForecastConfig(BaseModel):
    forecastBy: str
    selectedItem: Optional[str] = None
    selectedProduct: Optional[str] = None
    selectedCustomer: Optional[str] = None
    selectedLocation: Optional[str] = None
    selectedProducts: Optional[List[str]] = None
    selectedCustomers: Optional[List[str]] = None
    selectedLocations: Optional[List[str]] = None
    selectedItems: Optional[List[str]] = None
    algorithm: str = "linear_regression"
    interval: str = "month"
    historicPeriod: int = 12
    forecastPeriod: int = 6
    multiSelect: bool = False
    advancedMode: bool = False
    externalFactors: Optional[List[str]] = None

# Keep the ForecastingEngine class here (with all its methods)
# This is the core forecasting logic that routers will use
class ForecastingEngine:
    """Advanced forecasting engine with multiple algorithms"""

    ALGORITHMS = {
        "linear_regression": "Linear Regression",
        "polynomial_regression": "Polynomial Regression",
        "exponential_smoothing": "Exponential Smoothing",
        "holt_winters": "Holt-Winters",
        "arima": "ARIMA (Simple)",
        "random_forest": "Random Forest",
        "seasonal_decomposition": "Seasonal Decomposition",
        "moving_average": "Moving Average",
        "sarima": "SARIMA (Seasonal ARIMA)",
        "prophet_like": "Prophet-like Forecasting",
        "lstm_like": "Simple LSTM-like",
        "xgboost": "XGBoost Regression",
        "svr": "Support Vector Regression",
        "knn": "K-Nearest Neighbors",
        "gaussian_process": "Gaussian Process",
        "neural_network": "Neural Network (MLP)",
        "theta_method": "Theta Method",
        "croston": "Croston's Method",
        "ses": "Simple Exponential Smoothing",
        "damped_trend": "Damped Trend Method",
        "naive_seasonal": "Naive Seasonal",
        "drift_method": "Drift Method",
        "best_fit": "Best Fit (Auto-Select)",
        "best_statistical": "Best Statistical Method",
        "best_ml": "Best Machine Learning Method",
        "best_specialized": "Best Specialized Method"
    }

    @staticmethod
    def load_data_from_db(db: Session, config: ForecastConfig) -> pd.DataFrame:
        """Load data from MySQL database based on forecast configuration"""
        query = db.query(ForecastData)

        # Convert selected names to IDs for filtering
        product_ids = []
        customer_ids = []
        location_ids = []

        if config.multiSelect:
            if config.selectedProducts:
                product_ids = [DimensionManager.get_dimension_id(db, 'product', p) for p in config.selectedProducts if DimensionManager.get_dimension_id(db, 'product', p) is not None]
            if config.selectedCustomers:
                customer_ids = [DimensionManager.get_dimension_id(db, 'customer', c) for c in config.selectedCustomers if DimensionManager.get_dimension_id(db, 'customer', c) is not None]
            if config.selectedLocations:
                location_ids = [DimensionManager.get_dimension_id(db, 'location', l) for l in config.selectedLocations if DimensionManager.get_dimension_id(db, 'location', l) is not None]
        else: # Single selection mode
            if config.selectedItem:
                if config.forecastBy == 'product':
                    product_ids = [DimensionManager.get_dimension_id(db, 'product', config.selectedItem)]
                elif config.forecastBy == 'customer':
                    customer_ids = [DimensionManager.get_dimension_id(db, 'customer', config.selectedItem)]
                elif config.forecastBy == 'location':
                    location_ids = [DimensionManager.get_dimension_id(db, 'location', config.selectedItem)]
            
            if config.selectedProduct:
                product_ids = [DimensionManager.get_dimension_id(db, 'product', config.selectedProduct)]
            if config.selectedCustomer:
                customer_ids = [DimensionManager.get_dimension_id(db, 'customer', config.selectedCustomer)]
            if config.selectedLocation:
                location_ids = [DimensionManager.get_dimension_id(db, 'location', config.selectedLocation)]

        # Apply filters using surrogate IDs
        if product_ids:
            query = query.filter(ForecastData.product_id.in_(product_ids))
        if customer_ids:
            query = query.filter(ForecastData.customer_id.in_(customer_ids))
        if location_ids:
            query = query.filter(ForecastData.location_id.in_(location_ids))

        results = query.all()

        if not results:
            raise ValueError("No data found for the selected configuration")

        # Convert to DataFrame, including original string names for aggregation/display
        data = []
        for record in results:
            data.append({
                'date': record.date,
                'quantity': float(record.quantity),
                'product': record.product, # Keep original string for now
                'customer': record.customer, # Keep original string for now
                'location': record.location # Keep original string for now
            })

        df = pd.DataFrame(data)

        # Load and merge external factors if specified
        if config.externalFactors and len(config.externalFactors) > 0:
            df = ForecastingEngine.merge_external_factors(db, df, config.externalFactors)

        return df

    @staticmethod
    def merge_external_factors(db: Session, main_df: pd.DataFrame, external_factors: List[str]) -> pd.DataFrame:
        """Merge external factor data with main forecast data"""
        try:
            # Get date range from main data
            min_date = main_df['date'].min()
            max_date = main_df['date'].max()

            # Query external factor data
            external_query = db.query(ExternalFactorData).filter(
                ExternalFactorData.factor_name.in_(external_factors),
                ExternalFactorData.date >= min_date,
                ExternalFactorData.date <= max_date
            )

            external_results = external_query.all()
            print(f"External factor query results: {external_results}")

            if not external_results:
                print(f"No external factor data found for factors: {external_factors}")
                return main_df

            # Convert external factor data to DataFrame
            external_data = []
            for record in external_results:
                external_data.append({
                    'date': record.date,
                    'factor_name': record.factor_name,
                    'factor_value': float(record.factor_value)
                })

            external_df = pd.DataFrame(external_data)

            # Pivot external factors so each factor becomes a column
            external_pivot = external_df.pivot(index='date', columns='factor_name', values='factor_value')
            external_pivot.reset_index(inplace=True)

            # Ensure date columns are the same type
            main_df['date'] = pd.to_datetime(main_df['date']).dt.date
            external_pivot['date'] = pd.to_datetime(external_pivot['date']).dt.date

            # Merge with main data
            merged_df = pd.merge(main_df, external_pivot, on='date', how='left')
            print(f"Merged DataFrame columns: {merged_df.columns.tolist()}")

            # Forward fill missing external factor values
            for factor in external_factors:
                if factor in merged_df.columns:
                    merged_df[factor] = merged_df[factor].fillna(method='ffill').fillna(method='bfill')
            print(merged_df.head())
            print(f"Successfully merged external factors: {external_factors}")
            print(f"Merged data shape: {merged_df.shape}")

            return merged_df

        except Exception as e:
            print(f"Error merging external factors: {e}")
            return main_df

    @staticmethod
    def time_based_split(data: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
        """Proper time-based train/test split for time series"""
        n = len(data)
        if n < 6:
            return data.copy(), None

        split_idx = int(n * (1 - test_ratio))
        train = data.iloc[:split_idx].copy()
        test = data.iloc[split_idx:].copy()

        print(f"Time-based split: Train={len(train)} records, Test={len(test)} records")
        print(f"Train period: {train['date'].min()} to {train['date'].max()}")
        print(f"Test period: {test['date'].min()} to {test['date'].max()}")

        return train, test

    @staticmethod
    def forecast_external_factors(data: pd.DataFrame, external_factor_cols: List[str], periods: int) -> Dict[str, np.ndarray]:
        """Forecast external factors using simple trend analysis

        WARNING: This is a simplified approach for external factor forecasting.
        In production, external factors should be provided by the user or forecasted
        using specialized models. Current approach may lead to reduced accuracy.
        """
        future_factors = {}

        if external_factor_cols:
            print("⚠️  WARNING: External factors are being forecasted using simple trend analysis.")
            print("   For better accuracy, consider providing future external factor values.")
            print("   Current forecasting method may reduce model performance.")

        for col in external_factor_cols:
            if col not in data.columns:
                continue

            values = data[col].dropna().values
            if len(values) < 2:
                # Use last known value if insufficient data
                future_factors[col] = np.full(periods, values[-1] if len(values) > 0 else 0)
                print(f"External factor '{col}': Using last known value (insufficient data)")
                continue

            try:
                # Simple linear trend forecasting
                x = np.arange(len(values))

                # Fit linear trend
                slope, intercept = np.polyfit(x, values, 1)

                # Generate future values
                future_x = np.arange(len(values), len(values) + periods)
                future_values = slope * future_x + intercept

                # Add some uncertainty by using trend confidence
                trend_strength = abs(slope) / (np.std(values) + 1e-8)
                if trend_strength < 0.1:  # Weak trend, use last value
                    future_values = np.full(periods, values[-1])
                    print(f"External factor '{col}': Using last value (weak trend)")
                else:
                    print(f"External factor '{col}': Using linear trend (slope={slope:.4f})")

                future_factors[col] = future_values

            except Exception as e:
                # Fallback to last known value
                future_factors[col] = np.full(periods, values[-1])
                print(f"External factor '{col}': Fallback to last value due to error: {e}")

        return future_factors



    @staticmethod
    def aggregate_by_period(df: pd.DataFrame, interval: str, config: ForecastConfig = None) -> pd.DataFrame:
        """Aggregate data by time period"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Identify external factor columns
        external_factor_cols = [col for col in df.columns if col not in ['quantity', 'product', 'customer', 'location']]
        print(f"External factor columns found in aggregate_by_period: {external_factor_cols}")

        # Determine grouping columns based on configuration
        group_cols = ['date']
        if config and config.multiSelect:
            # For multi-select mode, determine which dimensions to group by
            selected_dimensions = []
            if config.selectedProducts and len(config.selectedProducts) > 0:
                selected_dimensions.append('product')
            if config.selectedCustomers and len(config.selectedCustomers) > 0:
                selected_dimensions.append('customer')
            if config.selectedLocations and len(config.selectedLocations) > 0:
                selected_dimensions.append('location')

            # If 2 or more dimensions are selected, group by all selected dimensions plus date
            if len(selected_dimensions) >= 2:
                group_cols = selected_dimensions + ['date']
        elif config and config.selectedItems and len(config.selectedItems) > 1:
            # Simple mode multi-select - group by the selected dimension plus date
            group_cols = [config.forecastBy, 'date']

        # Aggregate by period
        df_reset = df.reset_index()
        if interval == 'week':
            df_reset['period_group'] = df_reset['date'].dt.to_period('W-MON')
        elif interval == 'month':
            df_reset['period_group'] = df_reset['date'].dt.to_period('M')
        elif interval == 'year':
            df_reset['period_group'] = df_reset['date'].dt.to_period('Y')
        else:
            df_reset['period_group'] = df_reset['date'].dt.to_period('M')

        # Group by period and selected dimensions
        if len(group_cols) > 1:
            # Multi-dimensional grouping
            group_cols_with_period = [col for col in group_cols if col != 'date'] + ['period_group']

            # Aggregate quantity (sum) and external factors (mean)
            agg_dict = {'quantity': 'sum'}
            for col in external_factor_cols:
                if col in df_reset.columns:
                    agg_dict[col] = 'mean'  # Use mean for external factors

            aggregated = df_reset.groupby(group_cols_with_period).agg(agg_dict).reset_index()

            # Convert period back to timestamp for the first date of each period
            aggregated['date'] = aggregated['period_group'].dt.start_time
        else:
            # Single dimension grouping (original behavior)
            agg_dict = {'quantity': 'sum'}
            for col in external_factor_cols:
                if col in df_reset.columns:
                    agg_dict[col] = 'mean'  # Use mean for external factors

            aggregated = df_reset.groupby('period_group').agg(agg_dict).reset_index()
            aggregated['date'] = aggregated['period_group'].dt.start_time

        # Add period labels
        aggregated['period'] = aggregated['date'].apply(
            lambda x: ForecastingEngine.format_period(pd.Timestamp(x), interval)
        )

        # Clean up
        aggregated = aggregated.drop('period_group', axis=1)

        print(f"Aggregated DataFrame columns: {aggregated.columns.tolist()}")
        print(f"Sample aggregated data with external factors:")
        print(aggregated.head())

        return aggregated.reset_index(drop=True)

    @staticmethod
    def aggregate_advanced_mode(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Aggregate data for advanced mode (Product-Customer-Location-Date combinations)
        No aggregation across dimensions - keeps unique combinations intact
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Identify external factor columns
        external_factor_cols = [col for col in df.columns if col not in ['quantity', 'product', 'customer', 'location', 'date']]
        print(f"External factor columns found in aggregate_advanced_mode: {external_factor_cols}")

        # Create period groupings without aggregating across dimensions
        if interval == 'week':
            df['period_group'] = df['date'].dt.to_period('W-MON')
        elif interval == 'month':
            df['period_group'] = df['date'].dt.to_period('M')
        elif interval == 'year':
            df['period_group'] = df['date'].dt.to_period('Y')
        else:
            df['period_group'] = df['date'].dt.to_period('M')

        # Group by all dimensions plus period to preserve unique combinations
        # This ensures Product-Customer-Location-Date combinations remain distinct
        group_cols = ['product', 'customer', 'location', 'period_group']

        # Aggregate quantity (sum) and external factors (mean)
        agg_dict = {'quantity': 'sum'}
        for col in external_factor_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'  # Use mean for external factors

        aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()

        # Convert period back to timestamp
        aggregated['date'] = aggregated['period_group'].dt.start_time

        # Add period labels
        aggregated['period'] = aggregated['date'].apply(
            lambda x: ForecastingEngine.format_period(pd.Timestamp(x), interval)
        )

        # Clean up
        aggregated = aggregated.drop('period_group', axis=1)

        print(f"Advanced mode aggregated DataFrame columns: {aggregated.columns.tolist()}")
        print(f"Sample advanced mode aggregated data:")
        print(aggregated.head())

        return aggregated.reset_index(drop=True)

    @staticmethod
    def generate_simple_multi_forecast(db: Session, config: ForecastConfig, process_log: List[str] = None) -> MultiForecastResult:
        """Generate forecasts for multiple items in simple mode"""
        if process_log is not None:
            process_log.append("Generating simple multi-forecast...")
            process_log.append(f"Selected items: {config.selectedItems}")

        if not config.selectedItems or len(config.selectedItems) == 0:
            raise ValueError(f"Please select at least one {config.forecastBy} for multi-selection forecasting")

        results = []
        successful_combinations = 0
        failed_combinations = []

        for item in config.selectedItems:
            try:
                if process_log is not None:
                    process_log.append(f"Processing item: {item}")

                # Create a single-item config
                single_config = ForecastConfig(
                    forecastBy=config.forecastBy,
                    selectedItem=item,
                    algorithm=config.algorithm,
                    interval=config.interval,
                    historicPeriod=config.historicPeriod,
                    forecastPeriod=config.forecastPeriod,
                    externalFactors=config.externalFactors
                )

                # Load and aggregate data for this item
                if process_log is not None:
                    process_log.append("Loading data from database...")

                df = ForecastingEngine.load_data_from_db(db, single_config)

                if process_log is not None:
                    process_log.append(f"Data loaded: {len(df)} records")
                    process_log.append("Aggregating data by period...")

                aggregated_df = ForecastingEngine.aggregate_by_period(df, config.interval, single_config)

                if process_log is not None:
                    process_log.append(f"Data aggregated: {len(aggregated_df)} records")

                if len(aggregated_df) < 2:
                    if process_log is not None:
                        process_log.append("Insufficient data for forecasting")

                    failed_combinations.append({
                        'combination': item,
                        'error': 'Insufficient data'
                    })
                    continue

                if config.algorithm in ["best_fit", "best_statistical", "best_ml", "best_specialized"]:
                    if process_log is not None:
                        process_log.append(f"Running {config.algorithm} algorithm selection...")

                    # Define algorithm categories
                    statistical_algorithms = [
                        "linear_regression", "polynomial_regression", "exponential_smoothing", 
                        "holt_winters", "arima", "sarima", "ses", "damped_trend", 
                        "theta_method", "drift_method", "naive_seasonal", "prophet_like"
                    ]

                    ml_algorithms = [
                        "random_forest", "xgboost", "svr", "knn", "gaussian_process", 
                        "neural_network", "lstm_like"
                    ]

                    specialized_algorithms = [
                        "seasonal_decomposition", "moving_average", "croston"
                    ]

                    # Select algorithms based on category
                    if config.algorithm == "best_statistical":
                        algorithms = statistical_algorithms
                    elif config.algorithm == "best_ml":
                        algorithms = ml_algorithms
                    elif config.algorithm == "best_specialized":
                        algorithms = specialized_algorithms
                    else:  # best_fit
                        algorithms = [alg for alg in ForecastingEngine.ALGORITHMS.keys() 
                                    if alg not in ["best_fit", "best_statistical", "best_ml", "best_specialized"]]

                    algorithm_results = []  # Initialize before parallel execution

                    # Use parallel execution for algorithm evaluation
                    max_workers = min(len(algorithms), os.cpu_count() or 4)
                    if process_log is not None:
                        process_log.append(f"Running parallel evaluation with {max_workers} workers...")

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_algorithm = {
                            executor.submit(ForecastingEngine.run_algorithm, algorithm, aggregated_df, single_config, save_model=False): algorithm
                            for algorithm in algorithms
                        }

                        for future in as_completed(future_to_algorithm):
                            algorithm_name = future_to_algorithm[future]
                            try:
                                result = future.result()
                                if result:  # Check if result is not None
                                    algorithm_results.append(result)
                                    if process_log is not None:
                                        process_log.append(f"✅ Algorithm {algorithm_name} completed for {item}")
                            except Exception as exc:
                                if process_log is not None:
                                    process_log.append(f"❌ Algorithm {algorithm_name} failed for {item}: {str(exc)}")
                                continue

                    if not algorithm_results:
                        if process_log is not None:
                            process_log.append(f"No algorithms produced valid results for {item}")

                        failed_combinations.append({
                            'combination': item,
                            'error': 'No algorithms produced valid results'
                        })
                        continue

                    best_result = max(algorithm_results, key=lambda x: x.accuracy)
                    combination_dict = {config.forecastBy: item}
                    
                    forecast_result = ForecastResult(
                        combination=combination_dict,
                        selectedAlgorithm=f"{best_result.algorithm} (Best {config.algorithm.split('_')[1].capitalize()})",
                        accuracy=best_result.accuracy,
                        mae=best_result.mae,
                        rmse=best_result.rmse,
                        historicData=best_result.historicData,
                        forecastData=best_result.forecastData,
                        trend=best_result.trend,
                        allAlgorithms=algorithm_results,
                        processLog=process_log.copy() if process_log is not None else None
                    )
                else:
                    if process_log is not None:
                        process_log.append(f"Running algorithm: {config.algorithm}")

                    result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, single_config)
                    combination_dict = {config.forecastBy: item}
                    
                    forecast_result = ForecastResult(
                        combination=combination_dict,
                        selectedAlgorithm=result.algorithm,
                        accuracy=result.accuracy,
                        mae=result.mae,
                        rmse=result.rmse,
                        historicData=result.historicData,
                        forecastData=result.forecastData,
                        trend=result.trend,
                        processLog=process_log.copy() if process_log is not None else None
                    )

                results.append(forecast_result)
                successful_combinations += 1

                if process_log is not None:
                    process_log.append(f"Item {item} processed successfully")

            except Exception as e:
                if process_log is not None:
                    process_log.append(f"Error processing item {item}: {str(e)}")

                failed_combinations.append({
                    'combination': item,
                    'error': str(e)
                })

        if not results:
            raise ValueError("No valid forecasts could be generated for any item")

        # Calculate summary statistics
        avg_accuracy = np.mean([r.accuracy for r in results])
        best_combination = max(results, key=lambda x: x.accuracy)
        worst_combination = min(results, key=lambda x: x.accuracy)

        summary = {
            'averageAccuracy': round(avg_accuracy, 2),
            'bestCombination': {
                'combination': best_combination.combination,
                'accuracy': best_combination.accuracy
            },
            'worstCombination': {
                'combination': worst_combination.combination,
                'accuracy': worst_combination.accuracy
            },
            'successfulCombinations': successful_combinations,
            'failedCombinations': len(failed_combinations),
            'failedDetails': failed_combinations
        }

        if process_log is not None:
            process_log.append(f"Multi-forecast completed. Successful: {successful_combinations}, Failed: {len(failed_combinations)}")

        return MultiForecastResult(
            results=results,
            totalCombinations=len(config.selectedItems),
            summary=summary,
            processLog=process_log
        )

    @staticmethod
    def load_data_for_combination(db: Session, product_name: str, customer_name: str, location_name: str) -> pd.DataFrame:
        """Load data from MySQL database for a specific combination using surrogate keys"""
        product_id = DimensionManager.get_dimension_id(db, 'product', product_name)
        customer_id = DimensionManager.get_dimension_id(db, 'customer', customer_name)
        location_id = DimensionManager.get_dimension_id(db, 'location', location_name)

        if product_id is None or customer_id is None or location_id is None:
            raise ValueError(f"One or more dimension names not found: Product={product_name}, Customer={customer_name}, Location={location_name}")

        query = db.query(ForecastData).filter(
            ForecastData.product_id == product_id,
            ForecastData.customer_id == customer_id,
            ForecastData.location_id == location_id
        )

        results = query.all()

        if not results:
            raise ValueError(f"No data found for combination: {product_name} + {customer_name} + {location_name}")

        # Convert to DataFrame, keeping original string names for aggregation/display
        data = []
        for record in results:
            data.append({
                'date': record.date,
                'quantity': float(record.quantity),
                'product': record.product,
                'customer': record.customer,
                'location': record.location
            })

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def format_period(date: pd.Timestamp, interval: str) -> str:
        """Format period for display"""
        if interval == 'week':
            return f"Week of {date.strftime('%b %d, %Y')}"
        elif interval == 'month':
            return date.strftime('%b %Y')
        elif interval == 'year':
            return date.strftime('%Y')
        else:
            return date.strftime('%b %Y')

    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # Calculate accuracy as percentage
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        accuracy = max(0, 100 - mape)

        return {
            'accuracy': min(accuracy, 99.9),
            'mae': mae,
            'rmse': rmse
        }

    @staticmethod
    def calculate_trend(data: np.ndarray) -> str:
        """Calculate trend direction"""
        if len(data) < 2:
            return 'stable'

        # Linear regression to find trend
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)

        threshold = np.mean(data) * 0.02  # 2% threshold

        if slope > threshold:
            return 'increasing'
        elif slope < -threshold:
            return 'decreasing'
        else:
            return 'stable'

    # Algorithm implementations (keeping all existing algorithms)
    @staticmethod
    def linear_regression_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Linear regression forecasting with feature engineering"""
        y = data['quantity'].values
        n = len(y)

        # Feature engineering: create lag features and time index
        window = min(5, n - 1)

        # Get external factor columns
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        print(f"External factor columns: {external_factor_cols}")
        if window < 1:
            # Not enough data for feature engineering, fallback
            x = np.arange(n).reshape(-1, 1)
            if external_factor_cols:
                x = np.hstack([x, data[external_factor_cols].values])
            model = LinearRegression()
            model.fit(x, y)
            future_x = np.arange(n, n + periods).reshape(-1, 1)

            if external_factor_cols:
                # For simplicity, assume external factors remain at their last known value
                last_factors = data[external_factor_cols].iloc[-1].values
                future_factors = np.tile(last_factors, (periods, 1))
                future_x = np.hstack([future_x, future_factors])
            forecast = model.predict(future_x)
            forecast = np.maximum(forecast, 0)
            predicted = model.predict(x)
            metrics = ForecastingEngine.calculate_metrics(y, predicted)
            return forecast, metrics

        X = []
        y_target = []
        for i in range(window, n):
            lags = y[i-window:i]
            time_idx = i
            features = list(lags) + [time_idx]
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[i].values)
            X.append(features)
            y_target.append(y[i])

        X = np.array(X)
        y_target = np.array(y_target)

        # Debug print feature engineered data
        print(f"\nFeature engineered data for linear_regression:")
        print("Features (X) first 5 rows:")
        print(X[:5])
        print("Targets (y) first 5 values:")
        print(y_target[:5])

        model = LinearRegression()
        model.fit(X, y_target)

        # Forecast
        forecast = []
        recent_lags = list(y[-window:])
        for i in range(periods):
            features = recent_lags + [n + i]
            if external_factor_cols:
                # For simplicity, assume external factors remain at their last known value
                last_factors = data[external_factor_cols].iloc[-1].values
                features.extend(last_factors)

            pred = model.predict([features])[0]
            pred = max(0, pred)
            forecast.append(pred)
            recent_lags = recent_lags[1:] + [pred]

        forecast = np.array(forecast)

        # Calculate metrics on training data
        predicted = model.predict(X)
        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)

        return forecast, metrics

    @staticmethod
    def polynomial_regression_forecast(data: pd.DataFrame, periods: int, degree: int = 2) -> tuple:
        """Polynomial regression forecasting with feature engineering and external factors"""
        y = data['quantity'].values
        n = len(y)
        window = min(5, n - 1)
        # Get external factor columns
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if window < 1:
            x = np.arange(n).reshape(-1, 1)
            if external_factor_cols:
                x = np.hstack([x, data[external_factor_cols].values])
            best_metrics = None
            best_forecast = None
            for d in [2, 3]:
                coeffs = np.polyfit(np.arange(n), y, d)
                poly_func = np.poly1d(coeffs)
                future_x = np.arange(n, n + periods).reshape(-1, 1)
                if external_factor_cols:
                    last_factors = data[external_factor_cols].iloc[-1].values
                    future_factors = np.tile(last_factors, (periods, 1))
                    future_x = np.hstack([future_x, future_factors])
                forecast = poly_func(np.arange(n, n + periods))
                forecast = np.maximum(forecast, 0)
                predicted = poly_func(np.arange(n))
                metrics = ForecastingEngine.calculate_metrics(y, predicted)
                if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                    best_metrics = metrics
                    best_forecast = forecast
            return best_forecast, best_metrics
        X = []
        y_target = []
        for i in range(window, n):
            lags = y[i-window:i]
            time_idx = i
            features = list(lags) + [time_idx]
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[i].values)
            X.append(features)
            y_target.append(y[i])
        X = np.array(X)
        y_target = np.array(y_target)
        best_metrics = None
        best_forecast = None
        for d in [2, 3]:
            coeffs = np.polyfit(np.arange(len(y_target)), y_target, d)
            poly_func = np.poly1d(coeffs)
            future_x = np.arange(len(y_target), len(y_target) + periods)
            forecast = poly_func(future_x)
            forecast = np.maximum(forecast, 0)
            predicted = poly_func(np.arange(len(y_target)))
            metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
            if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                best_metrics = metrics
                best_forecast = forecast
        return best_forecast, best_metrics


    @staticmethod
    def exponential_smoothing_forecast(data: pd.DataFrame, periods: int, alphas: list = [0.1,0.3,0.5]) -> tuple:
        """Enhanced exponential smoothing with external factors integration"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 3:
            return np.full(periods, y[-1] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}

        best_metrics = None
        best_forecast = None

        for alpha in alphas:
            print(f"Running Exponential Smoothing with alpha={alpha}")

            if external_factor_cols:
                # Use regression-based exponential smoothing with external factors
                window = min(5, n - 1)
                X, y_target = [], []

                for i in range(window, n):
                    # Exponentially weighted historical values
                    weights = np.array([alpha * (1 - alpha) ** j for j in range(window)])
                    weights = weights / weights.sum()
                    weighted_history = np.sum(weights * y[i-window:i])

                    features = [weighted_history, i]  # Smoothed value + trend
                    if external_factor_cols:
                        features.extend(data[external_factor_cols].iloc[i].values)

                    X.append(features)
                    y_target.append(y[i])

                if len(X) > 1:
                    X = np.array(X)
                    y_target = np.array(y_target)

                    # Fit linear model with smoothed features
                    model = LinearRegression()
                    model.fit(X, y_target)

                    # Forecast with external factors
                    forecast = []
                    last_values = y[-window:]

                    for i in range(periods):
                        weights = np.array([alpha * (1 - alpha) ** j for j in range(len(last_values))])
                        weights = weights / weights.sum()
                        weighted_history = np.sum(weights * last_values)

                        features = [weighted_history, n + i]
                        if external_factor_cols:
                            # Use last known external factor values
                            features.extend(data[external_factor_cols].iloc[-1].values)

                        pred = model.predict([features])[0]
                        pred = max(0, pred)
                        forecast.append(pred)

                        # Update last_values for next prediction
                        last_values = np.append(last_values[1:], pred)

                    predicted = model.predict(X)
                    metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
                else:
                    # Fallback to simple smoothing
                    smoothed = pd.Series(y).ewm(alpha=alpha).mean().values
                    forecast = np.full(periods, smoothed[-1])
                    metrics = ForecastingEngine.calculate_metrics(y[1:], smoothed[1:])
            else:
                # Traditional exponential smoothing without external factors
                smoothed = pd.Series(y).ewm(alpha=alpha).mean().values
                forecast = np.full(periods, smoothed[-1])
                metrics = ForecastingEngine.calculate_metrics(y[1:], smoothed[1:])

            print(f"Alpha={alpha}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, Accuracy={metrics['accuracy']:.2f}")

            if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                best_metrics = metrics
                best_forecast = forecast

        return np.array(best_forecast), best_metrics

    @staticmethod
    def holt_winters_forecast(data: pd.DataFrame, periods: int, season_length: int = 12) -> tuple:
        """Enhanced Holt-Winters with external factors integration"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 2 * season_length:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        if external_factor_cols:
            # Enhanced Holt-Winters with external factor regression
            # First decompose the series using traditional Holt-Winters
            alpha, beta, gamma = 0.3, 0.1, 0.1

            level = np.mean(y[:season_length])
            trend = (np.mean(y[season_length:2*season_length]) - np.mean(y[:season_length])) / season_length
            seasonal = y[:season_length] - level

            levels = [level]
            trends = [trend]
            seasonals = list(seasonal)
            fitted = []
            residuals = []

            # Apply Holt-Winters to get base forecast
            for i in range(len(y)):
                if i == 0:
                    forecast_val = level + trend + seasonal[i % season_length]
                    fitted.append(forecast_val)
                    residuals.append(y[i] - forecast_val)
                else:
                    level = alpha * (y[i] - seasonals[i % season_length]) + (1 - alpha) * (levels[-1] + trends[-1])
                    trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
                    if len(seasonals) > i:
                        seasonals[i % season_length] = gamma * (y[i] - level) + (1 - gamma) * seasonals[i % season_length]

                    levels.append(level)
                    trends.append(trend)
                    forecast_val = level + trend + seasonals[i % season_length]
                    fitted.append(forecast_val)
                    residuals.append(y[i] - forecast_val)

            # Use external factors to model residuals
            window = min(3, len(residuals) - 1)
            if window > 0 and len(residuals) > window:
                X, y_residual = [], []
                for i in range(window, len(residuals)):
                    features = list(residuals[i-window:i])  # Lag residuals
                    features.extend(data[external_factor_cols].iloc[i].values)
                    X.append(features)
                    y_residual.append(residuals[i])

                if len(X) > 1:
                    X = np.array(X)
                    y_residual = np.array(y_residual)

                    # Model residuals with external factors
                    residual_model = LinearRegression()
                    residual_model.fit(X, y_residual)

                    # Forecast with external factors
                    forecast = []
                    recent_residuals = residuals[-window:]

                    for i in range(periods):
                        # Base Holt-Winters forecast
                        hw_forecast = level + (i + 1) * trend + seasonals[(len(y) + i) % season_length]

                        # Residual correction using external factors
                        features = list(recent_residuals)
                        features.extend(data[external_factor_cols].iloc[-1].values)

                        residual_correction = residual_model.predict([features])[0]
                        final_forecast = hw_forecast + residual_correction
                        final_forecast = max(0, final_forecast)

                        forecast.append(final_forecast)
                        recent_residuals = recent_residuals[1:] + [residual_correction]

                    # Calculate metrics on corrected fitted values
                    fitted_corrected = np.array(fitted) + residual_model.predict(X)
                    metrics = ForecastingEngine.calculate_metrics(y[window:], fitted_corrected)

                    return np.array(forecast), metrics

        # Traditional Holt-Winters without external factors
        alpha, beta, gamma = 0.3, 0.1, 0.1

        level = np.mean(y[:season_length])
        trend = (np.mean(y[season_length:2*season_length]) - np.mean(y[:season_length])) / season_length
        seasonal = y[:season_length] - level

        levels = [level]
        trends = [trend]
        seasonals = list(seasonal)
        fitted = []

        for i in range(len(y)):
            if i == 0:
                fitted.append(level + trend + seasonal[i % season_length])
            else:
                level = alpha * (y[i] - seasonals[i % season_length]) + (1 - alpha) * (levels[-1] + trends[-1])
                trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
                if len(seasonals) > i:
                    seasonals[i % season_length] = gamma * (y[i] - level) + (1 - gamma) * seasonals[i % season_length]

                levels.append(level)
                trends.append(trend)
                fitted.append(level + trend + seasonals[i % season_length])

        forecast = []
        for i in range(periods):
            forecast_value = level + (i + 1) * trend + seasonals[(len(y) + i) % season_length]
            forecast.append(max(0, forecast_value))

        metrics = ForecastingEngine.calculate_metrics(y, fitted)

        return np.array(forecast), metrics

    @staticmethod
    def arima_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Actual ARIMA/ARIMAX forecasting using statsmodels"""
        from statsmodels.tsa.arima.model import ARIMA
        from pmdarima import auto_arima # type: ignore
        import warnings

        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if len(y) < 10:  # Need sufficient data for ARIMA
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if external_factor_cols:
                    # ARIMAX: ARIMA with external regressors
                    print(f"Running ARIMAX with external factors: {external_factor_cols}")

                    # Prepare external regressors
                    exog = data[external_factor_cols].values

                    # Auto-select ARIMA parameters with external regressors
                    try:
                        auto_model = auto_arima(
                            y, 
                            exogenous=exog,
                            start_p=0, start_q=0, max_p=3, max_q=3, max_d=2,
                            seasonal=False,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )

                        # Forecast external factors properly
                        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
                        future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

                        # Create future exogenous matrix
                        future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
                        forecast = auto_model.predict(n_periods=periods, exogenous=future_exog)
                        forecast = np.maximum(forecast, 0)

                        # Calculate metrics
                        fitted = auto_model.fittedvalues()
                        if len(fitted) == len(y):
                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
                        else:
                            # Handle case where fitted values might be shorter due to differencing
                            start_idx = len(y) - len(fitted)
                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                        print(f"ARIMAX model: {auto_model.order}, AIC: {auto_model.aic():.2f}")
                        return forecast, metrics

                    except Exception as e:
                        print(f"Auto ARIMAX failed: {e}, trying manual ARIMAX")

                        # Fallback to manual ARIMAX
                        for order in [(1,1,1), (2,1,1), (1,1,2), (0,1,1)]:
                            try:
                                model = ARIMA(y, exog=exog, order=order)
                                fitted_model = model.fit()

                                # Use forecasted external factors
                                future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
                                forecast = fitted_model.forecast(steps=periods, exog=future_exog)
                                forecast = np.maximum(forecast, 0)

                                fitted = fitted_model.fittedvalues
                                if len(fitted) == len(y):
                                    metrics = ForecastingEngine.calculate_metrics(y, fitted)
                                else:
                                    start_idx = len(y) - len(fitted)
                                    metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                                print(f"Manual ARIMAX model: {order}, AIC: {fitted_model.aic:.2f}")
                                return forecast, metrics

                            except:
                                continue

                else:
                    # Traditional ARIMA without external factors
                    print("Running ARIMA without external factors")

                    try:
                        # Auto-select ARIMA parameters
                        auto_model = auto_arima(
                            y,
                            start_p=0, start_q=0, max_p=3, max_q=3, max_d=2,
                            seasonal=False,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )

                        forecast = auto_model.predict(n_periods=periods)
                        forecast = np.maximum(forecast, 0)

                        # Calculate metrics
                        fitted = auto_model.fittedvalues()
                        if len(fitted) == len(y):
                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
                        else:
                            start_idx = len(y) - len(fitted)
                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                        print(f"ARIMA model: {auto_model.order}, AIC: {auto_model.aic():.2f}")
                        return forecast, metrics

                    except Exception as e:
                        print(f"Auto ARIMA failed: {e}, trying manual ARIMA")

                        # Fallback to manual ARIMA
                        for order in [(1,1,1), (2,1,1), (1,1,2), (0,1,1), (1,0,1)]:
                            try:
                                model = ARIMA(y, order=order)
                                fitted_model = model.fit()

                                forecast = fitted_model.forecast(steps=periods)
                                forecast = np.maximum(forecast, 0)

                                fitted = fitted_model.fittedvalues
                                if len(fitted) == len(y):
                                    metrics = ForecastingEngine.calculate_metrics(y, fitted)
                                else:
                                    start_idx = len(y) - len(fitted)
                                    metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                                print(f"Manual ARIMA model: {order}, AIC: {fitted_model.aic:.2f}")
                                return forecast, metrics

                            except:
                                continue

        except Exception as e:
            print(f"ARIMA forecasting failed: {e}")

        # Final fallback to exponential smoothing
        print("ARIMA failed, falling back to exponential smoothing")
        return ForecastingEngine.exponential_smoothing_forecast(data, periods)

    @staticmethod
    def random_forest_forecast(data: pd.DataFrame, periods: int, n_estimators_list: list = [50, 100, 200], max_depth_list: list = [3, 5, None]) -> tuple:
        """Random Forest regression forecasting with hyperparameter tuning"""
        y = data['quantity'].values
        dates = pd.to_datetime(data['date'])

        # Get external factor columns
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        # Create features
        features = []
        targets = []
        window = min(5, len(y) - 1)

        for i in range(window, len(y)):
            lags = y[i-window:i]
            trend = i
            seasonal = i % 12
            month = dates.iloc[i].month
            quarter = dates.iloc[i].quarter
            feature_vector = list(lags) + [trend, seasonal, month, quarter]
            if external_factor_cols:
                feature_vector.extend(data[external_factor_cols].iloc[i].values)
            features.append(feature_vector)
            targets.append(y[i])

        if len(features) < 3:
            return ForecastingEngine.linear_regression_forecast(data, periods)

        features = np.array(features)
        targets = np.array(targets)

        best_metrics = None
        best_forecast = None

        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                print(f"Running Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(features, targets)

                # Forecast
                forecast = []
                recent_values = list(y[-window:])
                last_date = dates.iloc[-1]

                for i in range(periods):
                    trend_val = len(y) + i
                    seasonal_val = (len(y) + i) % 12
                    next_date = last_date + pd.DateOffset(months=i+1)
                    month_val = next_date.month
                    quarter_val = next_date.quarter
                    feature_vector = recent_values + [trend_val, seasonal_val, month_val, quarter_val]
                    if external_factor_cols:
                        # Use forecasted external factors
                        if 'future_factors' not in locals():
                            future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)
                        forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
                        feature_vector.extend(forecasted_factors)

                    next_value = model.predict([feature_vector])[0]
                    next_value = max(0, next_value)
                    forecast.append(next_value)
                    recent_values = recent_values[1:] + [next_value]

                predicted = model.predict(features)
                metrics = ForecastingEngine.calculate_metrics(targets, predicted)
                print(f"n_estimators={n_estimators}, max_depth={max_depth}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, Accuracy={metrics['accuracy']:.2f}")

                if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                    best_metrics = metrics
                    best_forecast = forecast

        return np.array(best_forecast), best_metrics

    @staticmethod
    def seasonal_decomposition_forecast(data: pd.DataFrame, periods: int, season_length: int = 12) -> tuple:
        """Enhanced seasonal decomposition with external factors"""
        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if len(y) < 2 * season_length:
            return ForecastingEngine.linear_regression_forecast(data, periods)

        # Simple seasonal decomposition
        trend = np.convolve(y, np.ones(season_length)/season_length, mode='same')

        # Calculate seasonal component
        detrended = y - trend
        seasonal_pattern = []
        for i in range(season_length):
            seasonal_values = [detrended[j] for j in range(i, len(detrended), season_length)]
            seasonal_pattern.append(np.mean(seasonal_values))

        if external_factor_cols:
            # Use external factors to enhance trend forecasting
            x = np.arange(len(trend))
            valid_trend = ~np.isnan(trend)

            if np.sum(valid_trend) > season_length:
                # Prepare features for trend modeling
                X_trend, y_trend = [], []
                for i in range(season_length, len(y)):
                    if not np.isnan(trend[i]):
                        features = [i]  # Time index
                        features.extend(data[external_factor_cols].iloc[i].values)
                        X_trend.append(features)
                        y_trend.append(trend[i])

                if len(X_trend) > 1:
                    X_trend = np.array(X_trend)
                    y_trend = np.array(y_trend)

                    # Model trend with external factors
                    trend_model = LinearRegression()
                    trend_model.fit(X_trend, y_trend)

                    # Forecast trend with external factors
                    future_trend = []
                    for i in range(periods):
                        features = [len(y) + i]
                        features.extend(data[external_factor_cols].iloc[-1].values)
                        trend_forecast = trend_model.predict([features])[0]
                        future_trend.append(trend_forecast)

                    # Forecast seasonal (repeat pattern)
                    future_seasonal = [seasonal_pattern[(len(y) + i) % season_length] for i in range(periods)]

                    # Combine forecast
                    forecast = np.array(future_trend) + np.array(future_seasonal)
                    forecast = np.maximum(forecast, 0)

                    # Calculate metrics using model predictions
                    trend_predicted = trend_model.predict(X_trend)
                    seasonal_full = np.tile(seasonal_pattern, len(y) // season_length + 1)[:len(y)]
                    fitted = np.full(len(y), np.nan)

                    for i, idx in enumerate(range(season_length, len(y))):
                        if not np.isnan(trend[idx]):
                            fitted[idx] = trend_predicted[X_trend[:, 0] == idx][0] if np.any(X_trend[:, 0] == idx) else trend[idx]
                            fitted[idx] += seasonal_full[idx]

                    valid_fitted = ~np.isnan(fitted)
                    if np.sum(valid_fitted) > 0:
                        metrics = ForecastingEngine.calculate_metrics(y[valid_fitted], fitted[valid_fitted])
                    else:
                        metrics = {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}

                    return forecast, metrics

        # Traditional seasonal decomposition without external factors
        x = np.arange(len(trend))
        valid_trend = ~np.isnan(trend)
        if np.sum(valid_trend) > 1:
            slope, intercept, _, _, _ = stats.linregress(x[valid_trend], trend[valid_trend])
            future_trend = [slope * (len(y) + i) + intercept for i in range(periods)]
        else:
            future_trend = [np.nanmean(trend)] * periods

        future_seasonal = [seasonal_pattern[(len(y) + i) % season_length] for i in range(periods)]

        forecast = np.array(future_trend) + np.array(future_seasonal)
        forecast = np.maximum(forecast, 0)

        seasonal_full = np.tile(seasonal_pattern, len(y) // season_length + 1)[:len(y)]
        fitted = trend + seasonal_full
        valid_fitted = ~np.isnan(fitted)
        if np.sum(valid_fitted) > 0:
            metrics = ForecastingEngine.calculate_metrics(y[valid_fitted], fitted[valid_fitted])
        else:
            metrics = {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}

        return forecast, metrics

    @staticmethod
    def moving_average_forecast(data: pd.DataFrame, periods: int, window: int = 3) -> tuple:
        """Enhanced moving average forecasting with external factors"""
        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        window = min(window, len(y))

        if external_factor_cols and len(y) > window + 2:
            # Moving average with external factor regression
            ma_values = []
            for i in range(len(y)):
                start_idx = max(0, i - window + 1)
                ma_values.append(np.mean(y[start_idx:i+1]))

            # Use external factors to adjust moving average
            X, y_target = [], []
            for i in range(window, len(y)):
                features = [ma_values[i]]  # Moving average as base feature
                features.extend(data[external_factor_cols].iloc[i].values)  # External factors
                X.append(features)
                y_target.append(y[i])

            if len(X) > 1:
                X = np.array(X)
                y_target = np.array(y_target)

                # Fit model to adjust moving average with external factors
                model = LinearRegression()
                model.fit(X, y_target)

                # Forecast
                forecast = []
                for i in range(periods):
                    # Calculate moving average based on recent values
                    if i == 0:
                        recent_avg = np.mean(y[-window:])
                    else:
                        # Use forecasted values for moving average
                        recent_values = list(y[-window+i:]) + forecast[:i]
                        recent_avg = np.mean(recent_values[-window:])

                    # Apply external factors
                    features = [recent_avg]
                    features.extend(data[external_factor_cols].iloc[-1].values)

                    adjusted_forecast = model.predict([features])[0]
                    adjusted_forecast = max(0, adjusted_forecast)
                    forecast.append(adjusted_forecast)

                # Calculate metrics
                predicted = model.predict(X)
                metrics = ForecastingEngine.calculate_metrics(y_target, predicted)

                return np.array(forecast), metrics

        # Traditional moving average without external factors
        moving_avg = []
        for i in range(len(y)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(np.mean(y[start_idx:i+1]))

        last_avg = np.mean(y[-window:])
        forecast = np.full(periods, last_avg)

        metrics = ForecastingEngine.calculate_metrics(y[window-1:], moving_avg[window-1:])

        return forecast, metrics

    @staticmethod
    def sarima_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Actual SARIMA/SARIMAX forecasting using statsmodels"""
        from statsmodels.tsa.arima.model import ARIMA
        from pmdarima import auto_arima # type: ignore
        import warnings

        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if len(y) < 24:  # Need at least 2 seasons for SARIMA
            return ForecastingEngine.arima_forecast(data, periods)

        # Determine seasonal period based on data frequency
        # Assume monthly data, so seasonal period = 12
        seasonal_period = 12
        if len(y) < 2 * seasonal_period:
            seasonal_period = max(4, len(y) // 3)  # Quarterly or adaptive

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if external_factor_cols:
                    # SARIMAX: SARIMA with external regressors
                    print(f"Running SARIMAX with external factors: {external_factor_cols}, seasonal_period: {seasonal_period}")

                    # Prepare external regressors
                    exog = data[external_factor_cols].values

                    # Auto-select SARIMA parameters with external regressors
                    try:
                        auto_model = auto_arima(
                            y, 
                            exogenous=exog,
                            start_p=0, start_q=0, max_p=2, max_q=2, max_d=1,
                            start_P=0, start_Q=0, max_P=2, max_Q=2, max_D=1,
                            seasonal=True, m=seasonal_period,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )

                        # Forecast external factors properly
                        future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

                        # Create future exogenous matrix
                        future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
                        forecast = auto_model.predict(n_periods=periods, exogenous=future_exog)
                        forecast = np.maximum(forecast, 0)

                        # Calculate metrics
                        fitted = auto_model.fittedvalues()
                        if len(fitted) == len(y):
                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
                        else:
                            start_idx = len(y) - len(fitted)
                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                        print(f"SARIMAX model: {auto_model.order}x{auto_model.seasonal_order}, AIC: {auto_model.aic():.2f}")
                        return forecast, metrics

                    except Exception as e:
                        print(f"Auto SARIMAX failed: {e}, trying manual SARIMAX")

                        # Fallback to manual SARIMAX
                        seasonal_orders = [
                            (1, 1, 1, seasonal_period),
                            (0, 1, 1, seasonal_period),
                            (1, 1, 0, seasonal_period),
                            (2, 1, 1, seasonal_period)
                        ]

                        for seasonal_order in seasonal_orders:
                            for order in [(1,1,1), (0,1,1), (1,1,0)]:
                                try:
                                    model = ARIMA(y, exog=exog, order=order, seasonal_order=seasonal_order)
                                    fitted_model = model.fit()

                                    # Use forecasted external factors
                                    future_exog = np.column_stack([future_factors[col] for col in external_factor_cols])
                                    forecast = fitted_model.forecast(steps=periods, exog=future_exog)
                                    forecast = np.maximum(forecast, 0)

                                    fitted = fitted_model.fittedvalues
                                    if len(fitted) == len(y):
                                        metrics = ForecastingEngine.calculate_metrics(y, fitted)
                                    else:
                                        start_idx = len(y) - len(fitted)
                                        metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                                    print(f"Manual SARIMAX model: {order}x{seasonal_order}, AIC: {fitted_model.aic:.2f}")
                                    return forecast, metrics

                                except:
                                    continue

                else:
                    # Traditional SARIMA without external factors
                    print(f"Running SARIMA without external factors, seasonal_period: {seasonal_period}")

                    try:
                        # Auto-select SARIMA parameters
                        auto_model = auto_arima(
                            y,
                            start_p=0, start_q=0, max_p=2, max_q=2, max_d=1,
                            start_P=0, start_Q=0, max_P=2, max_Q=2, max_D=1,
                            seasonal=True, m=seasonal_period,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )

                        forecast = auto_model.predict(n_periods=periods)
                        forecast = np.maximum(forecast, 0)

                        # Calculate metrics
                        fitted = auto_model.fittedvalues()
                        if len(fitted) == len(y):
                            metrics = ForecastingEngine.calculate_metrics(y, fitted)
                        else:
                            start_idx = len(y) - len(fitted)
                            metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                        print(f"SARIMA model: {auto_model.order}x{auto_model.seasonal_order}, AIC: {auto_model.aic():.2f}")
                        return forecast, metrics

                    except Exception as e:
                        print(f"Auto SARIMA failed: {e}, trying manual SARIMA")

                        # Fallback to manual SARIMA
                        seasonal_orders = [
                            (1, 1, 1, seasonal_period),
                            (0, 1, 1, seasonal_period),
                            (1, 1, 0, seasonal_period),
                            (2, 1, 1, seasonal_period)
                        ]

                        for seasonal_order in seasonal_orders:
                            for order in [(1,1,1), (0,1,1), (1,1,0)]:
                                try:
                                    model = ARIMA(y, order=order, seasonal_order=seasonal_order)
                                    fitted_model = model.fit()

                                    forecast = fitted_model.forecast(steps=periods)
                                    forecast = np.maximum(forecast, 0)

                                    fitted = fitted_model.fittedvalues
                                    if len(fitted) == len(y):
                                        metrics = ForecastingEngine.calculate_metrics(y, fitted)
                                    else:
                                        start_idx = len(y) - len(fitted)
                                        metrics = ForecastingEngine.calculate_metrics(y[start_idx:], fitted)

                                    print(f"Manual SARIMA model: {order}x{seasonal_order}, AIC: {fitted_model.aic:.2f}")
                                    return forecast, metrics

                                except:
                                    continue

        except Exception as e:
            print(f"SARIMA forecasting failed: {e}")

        # Final fallback to ARIMA
        print("SARIMA failed, falling back to ARIMA")
        return ForecastingEngine.arima_forecast(data, periods)

    @staticmethod
    def prophet_like_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Actual Facebook Prophet forecasting with external regressors support"""
        from prophet import Prophet # type: ignore
        import warnings

        y = data['quantity'].values
        dates = pd.to_datetime(data['date'])
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if len(y) < 10:  # Need sufficient data for Prophet
            return ForecastingEngine.linear_regression_forecast(data, periods)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Prepare data for Prophet (requires 'ds' and 'y' columns)
                prophet_data = pd.DataFrame({
                    'ds': dates,
                    'y': y
                })

                # Add external regressors if available
                if external_factor_cols:
                    print(f"Running Prophet with external regressors: {external_factor_cols}")
                    for col in external_factor_cols:
                        prophet_data[col] = data[col].values

                # Initialize Prophet model
                model = Prophet(
                    yearly_seasonality=True if len(y) >= 24 else False,
                    weekly_seasonality=False,  # Assuming monthly data
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    interval_width=0.8
                )

                # Add external regressors to the model
                if external_factor_cols:
                    for col in external_factor_cols:
                        model.add_regressor(col)

                # Fit the model
                model.fit(prophet_data)

                # Create future dataframe
                future = model.make_future_dataframe(periods=periods, freq='MS')  # Monthly start

                # Add external regressor values for future periods
                if external_factor_cols:
                    # Forecast external factors properly
                    future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

                    for col in external_factor_cols:
                        # Fill historical values
                        future[col] = np.nan
                        future.loc[:len(prophet_data)-1, col] = prophet_data[col].values
                        # Use forecasted values for future periods
                        future.loc[len(prophet_data):, col] = future_factors[col]

                # Make predictions
                forecast_df = model.predict(future)

                # Extract forecast values
                forecast = forecast_df['yhat'].iloc[-periods:].values
                forecast = np.maximum(forecast, 0)

                # Calculate metrics using fitted values
                fitted = forecast_df['yhat'].iloc[:len(y)].values
                metrics = ForecastingEngine.calculate_metrics(y, fitted)

                print(f"Prophet model fitted successfully with {len(external_factor_cols)} external regressors")
                return forecast, metrics

        except Exception as e:
            print(f"Prophet forecasting failed: {e}")

            # Fallback to simplified Prophet without external regressors
            try:
                print("Trying Prophet without external regressors...")

                prophet_data = pd.DataFrame({
                    'ds': dates,
                    'y': y
                })

                model = Prophet(
                    yearly_seasonality=True if len(y) >= 24 else False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.05
                )

                model.fit(prophet_data)
                future = model.make_future_dataframe(periods=periods, freq='MS')
                forecast_df = model.predict(future)

                forecast = forecast_df['yhat'].iloc[-periods:].values
                forecast = np.maximum(forecast, 0)

                fitted = forecast_df['yhat'].iloc[:len(y)].values
                metrics = ForecastingEngine.calculate_metrics(y, fitted)

                print("Prophet model fitted successfully without external regressors")
                return forecast, metrics

            except Exception as e2:
                print(f"Prophet fallback also failed: {e2}")

        # Final fallback to linear regression
        print("Prophet failed, falling back to linear regression")
        return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def lstm_simple_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Actual LSTM forecasting using TensorFlow/Keras with external factors support"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential # type: ignore
        from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
        from tensorflow.keras.optimizers import Adam # type: ignore
        from sklearn.preprocessing import MinMaxScaler
        import warnings

        y = data['quantity'].values
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        n = len(y)

        if n < 20:  # Need sufficient data for LSTM
            return ForecastingEngine.linear_regression_forecast(data, periods)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Set random seeds for reproducibility
                tf.random.set_seed(42)
                np.random.seed(42)

                # Prepare data
                window_size = min(10, n // 3)  # Use larger window for LSTM

                # Scale the data
                scaler_y = MinMaxScaler()
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

                # Handle external factors
                external_data = None
                scaler_external = None
                if external_factor_cols:
                    print(f"Running LSTM with external factors: {external_factor_cols}")
                    external_data = data[external_factor_cols].values
                    scaler_external = MinMaxScaler()
                    external_data = scaler_external.fit_transform(external_data)

                # Create sequences
                X, y_target = [], []
                for i in range(window_size, n):
                    # Time series features
                    sequence = y_scaled[i-window_size:i]

                    if external_factor_cols:
                        # Add external factors to each time step
                        external_seq = external_data[i-window_size:i]
                        # Combine time series with external factors
                        combined_seq = np.column_stack([sequence.reshape(-1, 1), external_seq])
                        X.append(combined_seq)
                    else:
                        X.append(sequence.reshape(-1, 1))

                    y_target.append(y_scaled[i])

                X = np.array(X)
                y_target = np.array(y_target)

                if len(X) < 10:  # Need minimum samples for LSTM
                    return ForecastingEngine.linear_regression_forecast(data, periods)

                # Determine input shape
                if external_factor_cols:
                    input_shape = (window_size, 1 + len(external_factor_cols))
                else:
                    input_shape = (window_size, 1)

                print(f"LSTM input shape: {input_shape}, samples: {len(X)}")

                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=input_shape),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])

                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

                # Train the model
                history = model.fit(
                    X, y_target,
                    epochs=50,
                    batch_size=min(32, len(X) // 2),
                    verbose=0,
                    validation_split=0.2 if len(X) > 20 else 0
                )

                # Make predictions for training data (for metrics)
                predicted_scaled = model.predict(X, verbose=0).flatten()
                predicted = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
                actual_for_metrics = scaler_y.inverse_transform(y_target.reshape(-1, 1)).flatten()

                # Generate forecasts
                forecast = []
                current_sequence = X[-1].copy()  # Last sequence from training data

                for i in range(periods):
                    # Predict next value
                    next_pred_scaled = model.predict(current_sequence.reshape(1, window_size, -1), verbose=0)[0, 0]
                    next_pred = scaler_y.inverse_transform([[next_pred_scaled]])[0, 0]
                    next_pred = max(0, next_pred)
                    forecast.append(next_pred)

                    # Update sequence for next prediction
                    if external_factor_cols:
                        # Use last known external factor values
                        last_external = external_data[-1]
                        next_scaled = scaler_y.transform([[next_pred]])[0, 0]
                        next_combined = np.array([next_scaled] + list(last_external))

                        # Shift sequence and add new prediction
                        current_sequence = np.roll(current_sequence, -1, axis=0)
                        current_sequence[-1] = next_combined
                    else:
                        next_scaled = scaler_y.transform([[next_pred]])[0, 0]
                        current_sequence = np.roll(current_sequence, -1, axis=0)
                        current_sequence[-1, 0] = next_scaled

                # Calculate metrics
                metrics = ForecastingEngine.calculate_metrics(actual_for_metrics, predicted)

                print(f"LSTM model trained successfully. Final loss: {history.history['loss'][-1]:.4f}")
                return np.array(forecast), metrics

        except Exception as e:
            print(f"LSTM forecasting failed: {e}")

            # Fallback to neural network (MLP)
            try:
                print("Trying MLP as LSTM fallback...")

                window_size = min(5, n - 1)
                X, y_target = [], []

                for i in range(window_size, n):
                    features = list(y[i-window_size:i])
                    if external_factor_cols:
                        features.extend(data[external_factor_cols].iloc[i].values)
                    X.append(features)
                    y_target.append(y[i])

                X = np.array(X)
                y_target = np.array(y_target)

                if len(X) >= 3:
                    model = MLPRegressor(
                        hidden_layer_sizes=(50, 25), 
                        max_iter=500, 
                        random_state=42, 
                        alpha=0.01
                    )
                    model.fit(X, y_target)

                    # Forecast
                    forecast = []
                    recent_values = list(y[-window_size:])

                    for _ in range(periods):
                        features = list(recent_values)
                        if external_factor_cols:
                            features.extend(data[external_factor_cols].iloc[-1].values)

                        next_pred = model.predict([features])[0]
                        next_pred = max(0, next_pred)
                        forecast.append(next_pred)
                        recent_values = recent_values[1:] + [next_pred]

                    predicted = model.predict(X)
                    metrics = ForecastingEngine.calculate_metrics(y_target, predicted)

                    print("MLP fallback successful")
                    return np.array(forecast), metrics

            except Exception as e2:
                print(f"MLP fallback also failed: {e2}")

        # Final fallback to linear regression
        print("LSTM and MLP failed, falling back to linear regression")
        return ForecastingEngine.linear_regression_forecast(data, periods)

    @staticmethod
    def xgboost_forecast(data: pd.DataFrame, periods: int, n_estimators_list: list = [50, 100], learning_rate_list: list = [0.05, 0.1, 0.2], max_depth_list: list = [3, 4, 5]) -> tuple:
        """XGBoost-like forecasting with hyperparameter tuning and external factors"""
        y = data['quantity'].values
        dates = pd.to_datetime(data['date'])
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if n < 6:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        features = []
        targets = []
        window = min(4, n - 1)
        for i in range(window, n):
            lags = list(y[i-window:i])
            date = dates.iloc[i]
            time_features = [
                i,
                date.month,
                date.quarter,
                date.dayofyear % 7,
                i % 12,
            ]
            recent_mean = np.mean(y[max(0, i-3):i])
            recent_std = np.std(y[max(0, i-3):i]) if i > 3 else 0
            feature_vector = lags + time_features + [recent_mean, recent_std]
            if external_factor_cols:
                feature_vector.extend(data[external_factor_cols].iloc[i].values)
            features.append(feature_vector)
            targets.append(y[i])
        if len(features) < 3:
            return ForecastingEngine.random_forest_forecast(data, periods)
        features = np.array(features)
        targets = np.array(targets)
        best_metrics = None
        best_forecast = None
        for n_estimators in n_estimators_list:
            for learning_rate in learning_rate_list:
                for max_depth in max_depth_list:
                    print(f"Running XGBoost with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
                    try:
                        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                        model.fit(features, targets)
                        forecast = []
                        recent_values = list(y[-window:])
                        last_date = dates.iloc[-1]
                        for i in range(periods):
                            next_date = last_date + pd.DateOffset(months=i+1)
                            time_features = [
                                n + i,
                                next_date.month,
                                next_date.quarter,
                                next_date.dayofyear % 7,
                                (n + i) % 12,
                            ]
                            recent_mean = np.mean(recent_values[-3:])
                            recent_std = np.std(recent_values[-3:]) if len(recent_values) > 1 else 0
                            feature_vector = recent_values + time_features + [recent_mean, recent_std]
                            if external_factor_cols:
                                last_factors = data[external_factor_cols].iloc[-1].values
                                feature_vector = list(feature_vector) + list(last_factors)
                            next_pred = model.predict([feature_vector])[0]
                            next_pred = max(0, next_pred)
                            forecast.append(next_pred)
                            recent_values = recent_values[1:] + [next_pred]
                        predicted = model.predict(features)
                        metrics = ForecastingEngine.calculate_metrics(targets, predicted)
                        print(f"n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, Accuracy={metrics['accuracy']:.2f}")
                        if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                            best_metrics = metrics
                            best_forecast = forecast
                    except Exception as e:
                        print(f"Error running XGBoost with params: {e}")
                        continue
        return np.array(best_forecast), best_metrics

    @staticmethod
    def svr_forecast(data: pd.DataFrame, periods: int, C_list: list = [1, 10, 100], epsilon_list: list = [0.1, 0.2]) -> tuple:
        """Support Vector Regression forecasting with hyperparameter tuning and external factors"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if n < 4:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        window = min(3, n - 1)
        X, y_target = [], []
        for i in range(window, n):
            features = [i] + list(y[i-window:i])
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[i].values)
            X.append(features)
            y_target.append(y[i])
        if len(X) < 2:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        X = np.array(X)
        y_target = np.array(y_target)
        X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X_norm = (X - X_mean) / X_std
        param_grid = {'C': C_list, 'epsilon': epsilon_list}
        model = SVR(kernel='rbf', gamma='scale')
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_norm, y_target)
        best_model = grid_search.best_estimator_
        print(f"Best SVR params: {grid_search.best_params_}")
        forecast = []
        recent_values = list(y[-window:])
        # Forecast external factors if needed
        if external_factor_cols:
            future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

        for i in range(periods):
            features = [n + i] + recent_values
            if external_factor_cols:
                forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
                features.extend(forecasted_factors)
            features_norm = (np.array(features) - X_mean) / X_std
            next_pred = best_model.predict([features_norm])[0]
            next_pred = max(0, next_pred)
            forecast.append(next_pred)
            recent_values = recent_values[1:] + [next_pred]
        predicted = best_model.predict(X_norm)
        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
        return np.array(forecast), metrics

    @staticmethod
    def knn_forecast(data: pd.DataFrame, periods: int, n_neighbors_list: list = [7, 10]) -> tuple:
        """K-Nearest Neighbors forecasting with hyperparameter tuning and external factors"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if n < 6:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        window = min(4, n - 1)
        X, y_target = [], []
        for i in range(window, n):
            features = list(y[i-window:i])
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[i].values)
            X.append(features)
            y_target.append(y[i])
        if len(X) < 3:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        X = np.array(X)
        y_target = np.array(y_target)
        param_grid = {'n_neighbors': n_neighbors_list}
        model = KNeighborsRegressor(weights='distance')
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X, y_target)
        best_model = grid_search.best_estimator_
        print(f"Best KNN params: {grid_search.best_params_}")
        forecast = []
        current_window = list(y[-window:])

        # Forecast external factors if needed
        if external_factor_cols:
            future_factors = ForecastingEngine.forecast_external_factors(data, external_factor_cols, periods)

        for i in range(periods):
            features = list(current_window)
            if external_factor_cols:
                forecasted_factors = [future_factors[col][i] for col in external_factor_cols]
                features.extend(forecasted_factors)
            next_pred = best_model.predict([features])[0]
            next_pred = max(0, next_pred)
            forecast.append(next_pred)
            current_window = current_window[1:] + [next_pred]
        predicted = best_model.predict(X)
        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
        return np.array(forecast), metrics

    @staticmethod
    def gaussian_process_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Improved Gaussian Process Regression forecasting with hyperparameter tuning and scaling"""
        y = data['quantity'].values
        n = len(y)

        if n < 4:
            return ForecastingEngine.linear_regression_forecast(data, periods)

        # Create time features
        X = np.arange(n).reshape(-1, 1)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            # Define kernel with initial parameters
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

            # Create GP model
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42, normalize_y=True)

            # Hyperparameter tuning for kernel parameters
            param_grid = {
                "kernel__k1__constant_value": [0.1, 1, 10],
                "kernel__k2__length_scale": [0.1, 1, 10]
            }
            grid_search = GridSearchCV(gp, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_scaled, y)
            best_model = grid_search.best_estimator_

            # Forecast
            future_X = np.arange(n, n + periods).reshape(-1, 1)
            future_X_scaled = scaler.transform(future_X)
            forecast, _ = best_model.predict(future_X_scaled, return_std=True)
            forecast = np.maximum(forecast, 0)

            # Calculate metrics
            predicted, _ = best_model.predict(X_scaled, return_std=True)
            metrics = ForecastingEngine.calculate_metrics(y, predicted)

        except Exception as e:
            print(f"Error in Gaussian Process forecasting: {e}")
            return ForecastingEngine.linear_regression_forecast(data, periods)

        return forecast, metrics

    @staticmethod
    def neural_network_forecast(data: pd.DataFrame, periods: int, hidden_layer_sizes_list: list = [(10,), (20, 10)], alpha_list: list = [0.001, 0.01]) -> tuple:
        """Multi-layer Perceptron Neural Network forecasting with hyperparameter tuning and external factors"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]
        if n < 6:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        window = min(5, n - 1)
        X, y_target = [], []
        for i in range(window, n):
            lags = list(y[i-window:i])
            trend = i / n
            seasonal = np.sin(2 * np.pi * i / 12)
            features = lags + [trend, seasonal]
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[i].values)
            X.append(features)
            y_target.append(y[i])
        if len(X) < 3:
            return ForecastingEngine.linear_regression_forecast(data, periods)
        X = np.array(X)
        y_target = np.array(y_target)
        param_grid = {'hidden_layer_sizes': hidden_layer_sizes_list, 'alpha': alpha_list}
        model = MLPRegressor(activation='relu', solver='adam', max_iter=1000, random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X, y_target)
        best_model = grid_search.best_estimator_
        print(f"Best MLP params: {grid_search.best_params_}")
        forecast = []
        recent_values = list(y[-window:])
        for i in range(periods):
            trend = (n + i) / n
            seasonal = np.sin(2 * np.pi * (n + i) / 12)
            features = recent_values + [trend, seasonal]
            if external_factor_cols:
                features.extend(data[external_factor_cols].iloc[-1].values)
            next_pred = best_model.predict([features])[0]
            next_pred = max(0, next_pred)
            forecast.append(next_pred)
            recent_values = recent_values[1:] + [next_pred]
        predicted = best_model.predict(X)
        metrics = ForecastingEngine.calculate_metrics(y_target, predicted)
        return np.array(forecast), metrics

    @staticmethod
    def theta_method_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Enhanced Theta method with external factors integration"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 3:
            return ForecastingEngine.linear_regression_forecast(data, periods)

        # Decompose series into two theta lines
        theta1 = 0
        theta2 = 2

        t = np.arange(n)
        mean_y = np.mean(y)
        theta_line1 = mean_y + (y - mean_y) * theta1
        theta_line2 = mean_y + (y - mean_y) * theta2

        if external_factor_cols:
            # Enhanced theta method with external factors
            # Forecast theta_line1 with external factors
            X1 = np.column_stack([t, data[external_factor_cols].values])
            model1 = LinearRegression()
            model1.fit(X1, theta_line1)

            future_external = np.tile(data[external_factor_cols].iloc[-1].values, (periods, 1))
            future_t = np.arange(n, n + periods)
            future_X1 = np.column_stack([future_t, future_external])
            forecast1 = model1.predict(future_X1)

            # Forecast theta_line2 with external factor-adjusted smoothing
            window = min(3, n - 1)
            if window > 0:
                X2, y2_target = [], []
                for i in range(window, n):
                    features = [theta_line2[i-window:i]]  # Historical smoothed values
                    features.extend(data[external_factor_cols].iloc[i].values)
                    X2.append(features)
                    y2_target.append(theta_line2[i])

                if len(X2) > 1:
                    X2 = np.array(X2)
                    y2_target = np.array(y2_target)

                    model2 = LinearRegression()
                    model2.fit(X2, y2_target)

                    # Forecast theta_line2
                    forecast2 = []
                    recent_smoothed = list(theta_line2[-window:])

                    for i in range(periods):
                        features = list(recent_smoothed)
                        features.extend(data[external_factor_cols].iloc[-1].values)

                        next_smoothed = model2.predict([features])[0]
                        forecast2.append(next_smoothed)
                        recent_smoothed = recent_smoothed[1:] + [next_smoothed]

                    forecast2 = np.array(forecast2)
                else:
                    # Fallback to simple smoothing
                    alpha = 0.3
                    smoothed = [theta_line2[0]]
                    for i in range(1, n):
                        smoothed.append(alpha * theta_line2[i] + (1 - alpha) * smoothed[i-1])
                    forecast2 = np.full(periods, smoothed[-1])
            else:
                # Simple smoothing fallback
                alpha = 0.3
                smoothed = [theta_line2[0]]
                for i in range(1, n):
                    smoothed.append(alpha * theta_line2[i] + (1 - alpha) * smoothed[i-1])
                forecast2 = np.full(periods, smoothed[-1])

            # Combine forecasts
            forecast = (forecast1 + forecast2) / 2
            forecast = np.maximum(forecast, 0)

            # Calculate metrics
            fitted1 = model1.predict(X1)
            if len(X2) > 1:
                fitted2_partial = model2.predict(X2)
                fitted2 = np.full(n, np.mean(theta_line2))
                fitted2[window:] = fitted2_partial
            else:
                fitted2 = theta_line2

            fitted = (fitted1 + fitted2) / 2
            metrics = ForecastingEngine.calculate_metrics(y, fitted)

            return forecast, metrics

        # Traditional theta method without external factors
        X = t.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, theta_line1)
        future_t = np.arange(n, n + periods).reshape(-1, 1)
        forecast1 = model.predict(future_t)

        alpha = 0.3
        smoothed = [theta_line2[0]]
        for i in range(1, n):
            smoothed.append(alpha * theta_line2[i] + (1 - alpha) * smoothed[i-1])
        last_smoothed = smoothed[-1]
        forecast2 = np.full(periods, last_smoothed)

        forecast = (forecast1 + forecast2) / 2
        forecast = np.maximum(forecast, 0)

        fitted = (model.predict(X) + smoothed) / 2
        metrics = ForecastingEngine.calculate_metrics(y, fitted)

        return forecast, metrics

    @staticmethod
    def croston_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Enhanced Croston's method with simplified, more accurate approach"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 3:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        # Identify non-zero demands and their timing
        non_zero_indices = np.where(y > 0)[0]
        
        if len(non_zero_indices) < 2:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        non_zero_values = y[non_zero_indices]
        
        # Calculate intervals between non-zero demands
        if len(non_zero_indices) > 1:
            intervals = np.diff(non_zero_indices)
        else:
            intervals = np.array([1])

        # Determine optimal smoothing parameter using simple cross-validation
        alphas = [0.1, 0.2, 0.3, 0.4]
        best_alpha = 0.2
        best_error = float('inf')
        
        for alpha in alphas:
            # Test forecast accuracy with this alpha
            if len(non_zero_values) > 3:
                # Use last 20% of data for validation
                split_point = max(2, int(len(non_zero_values) * 0.8))
                train_demands = non_zero_values[:split_point]
                train_intervals = intervals[:split_point-1] if len(intervals) >= split_point-1 else intervals
                test_demands = non_zero_values[split_point:]
                
                # Simple Croston calculation for validation
                if len(train_demands) > 1 and len(train_intervals) > 0:
                    smoothed_demand = train_demands[0]
                    smoothed_interval = train_intervals[0] if len(train_intervals) > 0 else 1
                    
                    for i in range(1, len(train_demands)):
                        smoothed_demand = alpha * train_demands[i] + (1 - alpha) * smoothed_demand
                        if i-1 < len(train_intervals):
                            smoothed_interval = alpha * train_intervals[i-1] + (1 - alpha) * smoothed_interval
                    
                    predicted_rate = smoothed_demand / max(1, smoothed_interval)
                    actual_rate = np.mean(test_demands) / np.mean(intervals[-len(test_demands):]) if len(test_demands) > 0 else 0
                    
                    error = abs(predicted_rate - actual_rate)
                    if error < best_error:
                        best_error = error
                        best_alpha = alpha

        # Apply Croston's method with optimal alpha
        if len(external_factor_cols) > 0 and len(non_zero_values) > 5:
            # Enhanced version with external factors (simplified)
            try:
                # Prepare regression features
                X_features = []
                y_target = []
                
                for i in range(1, min(len(non_zero_indices), len(data))):
                    idx = non_zero_indices[i]
                    if idx < len(data):
                        # Simple feature set: previous demand, time trend, external factors
                        features = [
                            non_zero_values[i-1],  # Previous demand
                            i / len(non_zero_values),  # Normalized time trend
                        ]
                        
                        # Add external factors (take mean if multiple values)
                        ext_vals = data[external_factor_cols].iloc[idx].values
                        features.extend(ext_vals)
                        
                        X_features.append(features)
                        y_target.append(non_zero_values[i])
                
                if len(X_features) > 2:
                    X_features = np.array(X_features)
                    y_target = np.array(y_target)
                    
                    # Fit simple model with regularization
                    model = Ridge(alpha=1.0)  # Higher regularization for stability
                    model.fit(X_features, y_target)
                    
                    # Generate forecast
                    forecast = []
                    last_demand = non_zero_values[-1]
                    
                    for i in range(periods):
                        # Predict next demand
                        time_factor = (len(non_zero_values) + i) / len(non_zero_values)
                        features = [last_demand, time_factor]
                        features.extend(data[external_factor_cols].iloc[-1].values)
                        
                        predicted_demand = model.predict([features])[0]
                        predicted_demand = max(0, predicted_demand)
                        
                        # Apply Croston logic: demand / average interval
                        avg_interval = np.mean(intervals[-min(3, len(intervals)):])  # Use recent intervals
                        forecast_value = predicted_demand / max(1, avg_interval)
                        
                        forecast.append(forecast_value)
                        last_demand = predicted_demand * 0.9 + last_demand * 0.1  # Smooth update
                    
                    # Calculate fitted values for metrics
                    fitted_values = []
                    for i in range(n):
                        if i in non_zero_indices:
                            fitted_values.append(y[i])
                        else:
                            # Use interpolation between non-zero values
                            prev_nz = non_zero_indices[non_zero_indices <= i]
                            next_nz = non_zero_indices[non_zero_indices > i]
                            
                            if len(prev_nz) > 0 and len(next_nz) > 0:
                                prev_idx = prev_nz[-1]
                                next_idx = next_nz[0]
                                weight = (i - prev_idx) / (next_idx - prev_idx)
                                interpolated = y[prev_idx] * (1 - weight) + y[next_idx] * weight
                                fitted_values.append(interpolated * 0.5)  # Conservative for zero periods
                            elif len(prev_nz) > 0:
                                fitted_values.append(y[prev_nz[-1]] * 0.3)
                            else:
                                fitted_values.append(0)
                    
                    metrics = ForecastingEngine.calculate_metrics(y, np.array(fitted_values))
                    return np.array(forecast), metrics
                    
            except Exception:
                pass  # Fall back to standard Croston

        # Standard Croston method (simplified and more stable)
        # Initialize with first values
        smoothed_demand = non_zero_values[0]
        smoothed_interval = intervals[0] if len(intervals) > 0 else 1
        
        # Update smoothed values
        for i in range(1, len(non_zero_values)):
            smoothed_demand = best_alpha * non_zero_values[i] + (1 - best_alpha) * smoothed_demand
            
            if i-1 < len(intervals):
                new_interval = max(1, intervals[i-1])  # Ensure interval >= 1
                smoothed_interval = best_alpha * new_interval + (1 - best_alpha) * smoothed_interval
        
        # Calculate base forecast rate
        base_rate = smoothed_demand / max(1, smoothed_interval)
        
        # Add simple trend based on recent demands
        if len(non_zero_values) >= 3:
            recent_trend = (non_zero_values[-1] - non_zero_values[-3]) / 2
            trend_factor = recent_trend / max(1, smoothed_interval)
        else:
            trend_factor = 0
        
        # Generate forecast with dampened trend
        forecast = []
        for i in range(periods):
            # Apply trend with exponential dampening
            dampening = np.exp(-i * 0.2)  # Dampen trend effect over time
            trend_component = trend_factor * dampening
            
            forecast_value = max(0, base_rate + trend_component)
            forecast.append(forecast_value)
        
        # Create simple fitted values for metrics calculation
        fitted_values = np.zeros(n)
        last_nonzero_value = 0
        
        for i in range(n):
            if y[i] > 0:
                last_nonzero_value = y[i]
                fitted_values[i] = y[i]
            else:
                # For zero periods, use a fraction of last non-zero demand
                fitted_values[i] = last_nonzero_value * base_rate * 0.5
        
        metrics = ForecastingEngine.calculate_metrics(y, fitted_values)
        
        return np.array(forecast), metrics


    @staticmethod
    def ses_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Enhanced Simple Exponential Smoothing with external factors"""
        y = data['quantity'].values
        n = len(y)
        external_factor_cols = [col for col in data.columns if col not in ['date', 'quantity', 'period', 'product', 'customer', 'location']]

        if n < 2:
            return np.full(periods, y[0] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': 0, 'rmse': 0}

        if external_factor_cols and n > 5:
            # SES with external factor enhancement
            alphas = [0.1, 0.3, 0.5, 0.7]
            best_metrics = None
            best_forecast = None

            for alpha in alphas:
                # Traditional SES for base smoothing
                smoothed = [y[0]]
                for i in range(1, n):
                    smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[i-1])

                # Use external factors to adjust smoothed values
                window = min(3, n - 1)
                X, y_target = [], []

                for i in range(window, n):
                    features = [smoothed[i]]  # Base smoothed value
                    features.extend(data[external_factor_cols].iloc[i].values)
                    X.append(features)
                    y_target.append(y[i])

                if len(X) > 1:
                    X = np.array(X)
                    y_target = np.array(y_target)

                    # Model to adjust SES with external factors
                    model = LinearRegression()
                    model.fit(X, y_target)

                    # Forecast
                    forecast = []
                    current_smoothed = smoothed[-1]

                    for i in range(periods):
                        features = [current_smoothed]
                        features.extend(data[external_factor_cols].iloc[-1].values)

                        adjusted_forecast = model.predict([features])[0]
                        adjusted_forecast = max(0, adjusted_forecast)
                        forecast.append(adjusted_forecast)

                        # Update smoothed value for next prediction
                        current_smoothed = alpha * adjusted_forecast + (1 - alpha) * current_smoothed

                    # Calculate metrics
                    predicted = model.predict(X)
                    metrics = ForecastingEngine.calculate_metrics(y_target, predicted)

                    if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                        best_metrics = metrics
                        best_forecast = forecast

            if best_forecast is not None:
                return np.array(best_forecast), best_metrics

        # Traditional SES without external factors using statsmodels
        try:
            import warnings
            best_model = None
            best_aic = float('inf')
            best_forecast = None

            seasonal_periods_options = [None, 4, 6, 12]
            trend_options = [None, 'add', 'mul']
            seasonal_options = [None, 'add', 'mul']

            for seasonal_periods in seasonal_periods_options:
                for trend in trend_options:
                    for seasonal in seasonal_options:
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                model = ExponentialSmoothing(
                                    y,
                                    trend=trend,
                                    seasonal=seasonal,
                                    seasonal_periods=seasonal_periods,
                                    initialization_method="estimated"
                                )
                                fit = model.fit(optimized=True)
                            aic = fit.aic
                            if aic < best_aic:
                                best_aic = aic
                                best_model = fit
                                best_forecast = fit.forecast(periods)
                        except Exception:
                            continue

            if best_forecast is not None:
                forecast = np.maximum(best_forecast, 0)
                fitted = best_model.fittedvalues
                metrics = ForecastingEngine.calculate_metrics(y, fitted)
                return forecast, metrics
        except:
            pass

        # Simple fallback SES
        alpha = 0.3
        smoothed = [y[0]]
        for i in range(1, n):
            smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[i-1])

        forecast = np.full(periods, smoothed[-1])
        metrics = ForecastingEngine.calculate_metrics(y[1:], smoothed[1:])

        return forecast, metrics

    @staticmethod
    def damped_trend_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Damped trend exponential smoothing"""
        y = data['quantity'].values
        n = len(y)

        if n < 3:
            return ForecastingEngine.exponential_smoothing_forecast(data, periods)

        # Parameters
        alpha, beta, phi = 0.3, 0.1, 0.8  # phi is damping parameter

        # Initialize
        level = y[0]
        trend = y[1] - y[0] if n > 1 else 0

        levels = [level]
        trends = [trend]
        fitted = [level + trend]

        # Apply damped trend smoothing
        for i in range(1, n):
            level = alpha * y[i] + (1 - alpha) * (levels[-1] + phi * trends[-1])
            trend = beta * (level - levels[-1]) + (1 - beta) * phi * trends[-1]

            levels.append(level)
            trends.append(trend)
            fitted.append(level + trend)

        # Forecast with damping
        forecast = []
        for h in range(1, periods + 1):
            damped_trend = trend * sum(phi**i for i in range(1, h + 1))
            forecast_value = level + damped_trend
            forecast.append(max(0, forecast_value))

        # Calculate metrics
        metrics = ForecastingEngine.calculate_metrics(y[1:], fitted[1:])

        return np.array(forecast), metrics

    @staticmethod
    def naive_seasonal_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Naive seasonal forecasting"""
        y = data['quantity'].values
        n = len(y)

        if n < 2:
            return np.full(periods, y[0] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': 0, 'rmse': 0}

        # Determine season length
        season_length = min(12, n)

        # Forecast by repeating seasonal pattern
        forecast = []
        for i in range(periods):
            seasonal_index = (n + i) % season_length
            if seasonal_index < n:
                forecast.append(y[-(season_length - seasonal_index)])
            else:
                forecast.append(y[-1])

        # Calculate metrics using naive forecast on historical data
        if n > season_length:
            predicted = []
            for i in range(season_length, n):
                predicted.append(y[i - season_length])
            actual = y[season_length:]
            metrics = ForecastingEngine.calculate_metrics(actual, predicted)
        else:
            metrics = {'accuracy': 60.0, 'mae': np.std(y), 'rmse': np.std(y)}

        return np.array(forecast), metrics

    @staticmethod
    def drift_method_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """Improved Drift method forecasting with linear regression trend"""
        y = data['quantity'].values
        n = len(y)

        if n < 2:
            return np.full(periods, y[0] if len(y) > 0 else 0), {'accuracy': 50.0, 'mae': 0, 'rmse': 0}

        # Use linear regression to estimate trend and intercept
        X = np.arange(n).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)

        # Forecast
        future_X = np.arange(n, n + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        forecast = np.maximum(forecast, 0)

        # Calculate metrics on training data
        predicted = model.predict(X)
        metrics = ForecastingEngine.calculate_metrics(y, predicted)

        return forecast, metrics

    @staticmethod
    def generate_forecast_dates(last_date: pd.Timestamp, periods: int, interval: str) -> List[pd.Timestamp]:
        """Generate future dates for forecast"""
        dates = []
        current_date = last_date

        for i in range(periods):
            if interval == 'week':
                current_date = current_date + timedelta(weeks=1)
            elif interval == 'month':
                current_date = current_date + pd.DateOffset(months=1)
            elif interval == 'year':
                current_date = current_date + pd.DateOffset(years=1)
            else:
                current_date = current_date + pd.DateOffset(months=1)

            dates.append(current_date)

        return dates

    @staticmethod
    def run_algorithm(algorithm: str, data: pd.DataFrame, config: ForecastConfig, save_model: bool = True) -> AlgorithmResult:
        """Run a specific forecasting algorithm"""
        from database import SessionLocal
        db = SessionLocal()
        try:
            # Print first 5 rows of data fed to algorithm
            print(f"\nData fed to algorithm '{algorithm}':")
            print(data.head(5))

            # Check for cached model
            training_data = data['quantity'].values
            if training_data is not None and len(training_data) > 0:
                model_hash = ModelPersistenceManager.find_cached_model(db, algorithm, config.dict(), training_data)
            else:
                model_hash = None

            if model_hash:
                print(f"Using cached model for {algorithm}")
                cached_model = ModelPersistenceManager.load_model(db, model_hash)
                if cached_model:
                    # Use cached model for prediction
                    # Note: This is a simplified example - in practice, you'd need to adapt this
                    # based on the specific algorithm and model type
                    pass

            # Time-based train/test split for realistic metrics
            train, test = ForecastingEngine.time_based_split(data, test_ratio=0.2)

            # Train model on train set
            model = None
            if algorithm == "linear_regression":
                forecast, metrics = ForecastingEngine.linear_regression_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = LinearRegression().fit(np.arange(len(train)).reshape(-1, 1), train['quantity'].values)
            elif algorithm == "polynomial_regression":
                forecast, metrics = ForecastingEngine.polynomial_regression_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "exponential_smoothing":
                forecast, metrics = ForecastingEngine.exponential_smoothing_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "holt_winters":
                forecast, metrics = ForecastingEngine.holt_winters_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "arima":
                forecast, metrics = ForecastingEngine.arima_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "random_forest":
                forecast, metrics = ForecastingEngine.random_forest_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "seasonal_decomposition":
                forecast, metrics = ForecastingEngine.seasonal_decomposition_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "moving_average":
                forecast, metrics = ForecastingEngine.moving_average_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "sarima":
                forecast, metrics = ForecastingEngine.sarima_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "prophet_like":
                forecast, metrics = ForecastingEngine.prophet_like_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "lstm_like":
                forecast, metrics = ForecastingEngine.lstm_simple_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "xgboost":
                forecast, metrics = ForecastingEngine.xgboost_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "svr":
                forecast, metrics = ForecastingEngine.svr_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "knn":
                forecast, metrics = ForecastingEngine.knn_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "gaussian_process":
                forecast, metrics = ForecastingEngine.gaussian_process_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "neural_network":
                forecast, metrics = ForecastingEngine.neural_network_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "theta_method":
                forecast, metrics = ForecastingEngine.theta_method_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "croston":
                forecast, metrics = ForecastingEngine.croston_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "ses":
                forecast, metrics = ForecastingEngine.ses_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "damped_trend":
                forecast, metrics = ForecastingEngine.damped_trend_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "naive_seasonal":
                forecast, metrics = ForecastingEngine.naive_seasonal_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            elif algorithm == "drift_method":
                forecast, metrics = ForecastingEngine.drift_method_forecast(train, len(test) if test is not None else config.forecastPeriod)
                model = None  # No explicit model object to save
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Save model/configuration to cache
            if ENABLE_MODEL_CACHE and save_model:
                try:
                    print(f"Debug run_algorithm: Saving model/configuration for algorithm {algorithm}")
                    ModelPersistenceManager.save_model(
                        db, model, algorithm, config.dict(), 
                        training_data, metrics, {'data_shape': training_data.shape}
                    )
                    print(f"Debug run_algorithm: Model/configuration saved successfully for algorithm {algorithm}")
                except Exception as e:
                    print(f"Failed to save model/configuration to cache: {e}")

            # Compute test metrics
            if test is not None and len(test) > 0:
                actual = test['quantity'].values
                predicted = forecast[:len(test)]
                metrics = ForecastingEngine.calculate_metrics(actual, predicted)
            else:
                # Fallback to training metrics
                y = train['quantity'].values
                x = np.arange(len(y)).reshape(-1, 1)
                if algorithm == "linear_regression":
                    model = LinearRegression().fit(x, y)
                    predicted = model.predict(x)
                elif algorithm == "polynomial_regression":
                    coeffs = np.polyfit(np.arange(len(y)), y, 2)
                    poly_func = np.poly1d(coeffs)
                    predicted = poly_func(np.arange(len(y)))
                elif algorithm == "exponential_smoothing" or algorithm == "ses":
                    # Use simple exponential smoothing for fallback
                    alpha = 0.3
                    smoothed = [y[0]]
                    for i in range(1, len(y)):
                        smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[i-1])
                    predicted = smoothed
                else:
                    predicted = y
                metrics = ForecastingEngine.calculate_metrics(y, predicted)

            # Prepare output
            last_date = data['date'].iloc[-1]
            forecast_dates = ForecastingEngine.generate_forecast_dates(last_date, config.forecastPeriod, config.interval)

            historic_data = []
            historic_subset = data.tail(config.historicPeriod)
            for _, row in historic_subset.iterrows():
                historic_data.append(DataPoint(
                    date=row['date'].strftime('%Y-%m-%d'),
                    quantity=float(row['quantity']),
                    period=row['period']
                ))

            forecast_data = []
            for i, (date, quantity) in enumerate(zip(forecast_dates, forecast)):
                forecast_data.append(DataPoint(
                    date=date.strftime('%Y-%m-%d'),
                    quantity=float(quantity),
                    period=ForecastingEngine.format_period(date, config.interval)
                ))

            trend = ForecastingEngine.calculate_trend(data['quantity'].values)

            return AlgorithmResult(
                algorithm=ForecastingEngine.ALGORITHMS[algorithm],
                accuracy=round(metrics['accuracy'], 1),
                mae=round(metrics['mae'], 2),
                rmse=round(metrics['rmse'], 2),
                historicData=historic_data,
                forecastData=forecast_data,
                trend=trend
            )
        except Exception as e:
            print(f"Error in {algorithm}: {str(e)}")
            return AlgorithmResult(
                algorithm=ForecastingEngine.ALGORITHMS[algorithm],
                accuracy=0.0,
                mae=999.0,
                rmse=999.0,
                historicData=[],
                forecastData=[],
                trend='stable'
            )
        finally:
            db.close()

    @staticmethod
    def generate_forecast(db: Session, config: ForecastConfig, process_log: List[str] = None) -> ForecastResult:
        """Generate forecast using data from database"""
        if process_log is not None:
            process_log.append("Loading data from database...")

        df = ForecastingEngine.load_data_from_db(db, config)

        if process_log is not None:
            process_log.append(f"Data loaded: {len(df)} records")
            process_log.append("Aggregating data by period...")

        aggregated_df = ForecastingEngine.aggregate_by_period(df, config.interval, config)

        if process_log is not None:
            process_log.append(f"Data aggregated: {len(aggregated_df)} records")

        if len(aggregated_df) < 2:
            raise ValueError("Insufficient data for forecasting")

        if config.algorithm in ["best_fit", "best_statistical", "best_ml", "best_specialized"]:
            if process_log is not None:
                if config.algorithm == "best_fit":
                    process_log.append("Running best fit algorithm selection...")
                elif config.algorithm == "best_statistical":
                    process_log.append("Running best statistical method selection...")
                elif config.algorithm == "best_ml":
                    process_log.append("Running best machine learning method selection...")
                elif config.algorithm == "best_specialized":
                    process_log.append("Running best specialized method selection...")

            # Define algorithm categories
            statistical_algorithms = [
                "linear_regression", "polynomial_regression", "exponential_smoothing", 
                "holt_winters", "arima", "sarima", "ses", "damped_trend", 
                "theta_method", "drift_method", "naive_seasonal", "prophet_like"
            ]

            ml_algorithms = [
                "random_forest", "xgboost", "svr", "knn", "gaussian_process", 
                "neural_network", "lstm_like"
            ]

            specialized_algorithms = [
                "seasonal_decomposition", "moving_average", "croston"
            ]

            # Select algorithms based on category
            if config.algorithm == "best_statistical":
                algorithms = statistical_algorithms
            elif config.algorithm == "best_ml":
                algorithms = ml_algorithms
            elif config.algorithm == "best_specialized":
                algorithms = specialized_algorithms
            else:  # best_fit
                algorithms = [alg for alg in ForecastingEngine.ALGORITHMS.keys() 
                            if alg not in ["best_fit", "best_statistical", "best_ml", "best_specialized"]]

            algorithm_results = []
            best_model = None
            best_algorithm = None
            best_metrics = None

            # Use ThreadPoolExecutor for parallel execution
            max_workers = min(len(algorithms), os.cpu_count() or 4)
            if process_log is not None:
                process_log.append(f"Starting parallel execution with {max_workers} workers for {len(algorithms)} algorithms...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all algorithm tasks
                future_to_algorithm = {
                    executor.submit(ForecastingEngine.run_algorithm, algorithm, aggregated_df, config, save_model=False): algorithm
                    for algorithm in algorithms
                }

                # Collect results as they complete
                for future in as_completed(future_to_algorithm):
                    algorithm_name = future_to_algorithm[future]
                    try:
                        result = future.result()
                        algorithm_results.append(result)
                        if process_log is not None:
                            process_log.append(f"✅ Algorithm {algorithm_name} completed with accuracy: {result.accuracy:.2f}%")

                        # Track best performing algorithm
                        if best_metrics is None or result.accuracy > best_metrics['accuracy']:
                            best_metrics = {
                                'accuracy': result.accuracy,
                                'mae': result.mae,
                                'rmse': result.rmse
                            }
                            best_model = result
                            best_algorithm = algorithm_name

                    except Exception as exc:
                        if process_log is not None:
                            process_log.append(f"❌ Algorithm {algorithm_name} failed: {str(exc)}")
                        # Create a dummy result for failed algorithms to maintain structure
                        algorithm_results.append(AlgorithmResult(
                            algorithm=ForecastingEngine.ALGORITHMS[algorithm_name],
                            accuracy=0.0,
                            mae=999.0,
                            rmse=999.0,
                            historicData=[],
                            forecastData=[],
                            trend='stable'
                        ))

            # Filter out failed results for ensemble calculation
            successful_results = [res for res in algorithm_results if res.accuracy > 0]
            if not successful_results:
                raise ValueError("All algorithms failed to produce valid results")

            if process_log is not None:
                process_log.append(f"Parallel execution completed. {len(successful_results)} algorithms succeeded, {len(algorithm_results) - len(successful_results)} failed.")

            if not algorithm_results:
                raise ValueError("No algorithms produced valid results")

            # Ensemble: average forecast of top 3 algorithms by accuracy
            top3 = sorted(successful_results, key=lambda x: -x.accuracy)[:3]
            if len(top3) >= 2:
                if process_log is not None:
                    process_log.append(f"Creating ensemble from top {len(top3)} algorithms...")

                n_forecast = len(top3[0].forecastData) if top3[0].forecastData else 0
                avg_forecast = []
                for i in range(n_forecast):
                    quantities = [algo.forecastData[i].quantity for algo in top3 if algo.forecastData]
                    if not quantities:
                        avg_qty = 0
                    else:
                        avg_qty = np.mean(quantities)
                    avg_forecast.append(DataPoint(
                        date=top3[0].forecastData[i].date,
                        quantity=avg_qty,
                        period=top3[0].forecastData[i].period
                    ))

                ensemble_result = AlgorithmResult(
                    algorithm="Ensemble (Top 3 Avg)",
                    accuracy=np.mean([algo.accuracy for algo in top3]),
                    mae=np.mean([algo.mae for algo in top3]),
                    rmse=np.mean([algo.rmse for algo in top3]),
                    historicData=top3[0].historicData,
                    forecastData=avg_forecast,
                    trend=top3[0].trend
                )
                algorithm_results.append(ensemble_result)

            # Save the best_fit model configuration
            try:
                if process_log is not None:
                    process_log.append(f"Saving best fit model configuration (best algorithm: {best_algorithm})...")

                ModelPersistenceManager.save_model(
                    db, None, best_algorithm, config.dict(),
                    aggregated_df['quantity'].values, 
                    {'accuracy': ensemble_result.accuracy, 'mae': ensemble_result.mae, 'rmse': ensemble_result.rmse},
                    {'data_shape': aggregated_df.shape, 'best_algorithm': best_algorithm}
                )
            except Exception as e:
                print(f"Failed to save best_fit model: {e}")
                if process_log is not None:
                    process_log.append(f"Warning: Failed to save best_fit model: {e}")

            # Generate config hash for tracking
            config_hash = ModelPersistenceManager.generate_config_hash(config.dict())

            return ForecastResult(
                selectedAlgorithm=f"{best_model.algorithm} (Best Fit)",
                accuracy=best_model.accuracy,
                mae=best_model.mae,
                rmse=best_model.rmse,
                historicData=best_model.historicData,
                forecastData=best_model.forecastData,
                trend=best_model.trend,
                allAlgorithms=algorithm_results,
                configHash=config_hash,
                processLog=process_log
            )
        else:
            if process_log is not None:
                process_log.append(f"Running algorithm: {config.algorithm}")

            result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, config, save_model=True)
            config_hash = ModelPersistenceManager.generate_config_hash(config.dict())
            return ForecastResult(
                selectedAlgorithm=result.algorithm,
                combination=None,
                accuracy=result.accuracy,
                mae=result.mae,
                rmse=result.rmse,
                historicData=result.historicData,
                forecastData=result.forecastData,
                trend=result.trend,
                configHash=config_hash,
                processLog=process_log
            )

    @staticmethod
    def generate_forecast_three_dimensions(db: Session, config: ForecastConfig, products: list, customers: list, locations: list, process_log: List[str] = None) -> MultiForecastResult:
        """Generate forecasts for all three dimensions selected"""
        from itertools import product as itertools_product
        combinations = [(p, c, l) for p, c, l in itertools_product(products, customers, locations)]

        if process_log is not None:
            process_log.append(f"Generated {len(combinations)} combinations for three dimensions")

        results = []
        successful_combinations = 0
        failed_combinations = []

        for product, customer, location in combinations:
            try:
                if process_log is not None:
                    process_log.append(f"Processing combination: {product} + {customer} + {location}")

                # Load data for this specific combination
                df = ForecastingEngine.load_data_for_combination(db, product, customer, location)
                aggregated_df = ForecastingEngine.aggregate_by_period(df, config.interval, config)

                if len(aggregated_df) < 2:
                    if process_log is not None:
                        process_log.append("Insufficient data for forecasting")

                    failed_combinations.append({
                        'combination': f"{product} + {customer} + {location}",
                        'error': 'Insufficient data'
                    })
                    continue

                if config.algorithm in ["best_fit", "best_statistical", "best_ml", "best_specialized"]:
                    if process_log is not None:
                        process_log.append(f"Running {config.algorithm} algorithm selection...")
                    
                    # Define algorithm categories
                    statistical_algorithms = [
                        "linear_regression", "polynomial_regression", "exponential_smoothing", 
                        "holt_winters", "arima", "sarima", "ses", "damped_trend", 
                        "theta_method", "drift_method", "naive_seasonal", "prophet_like"
                    ]

                    ml_algorithms = [
                        "random_forest", "xgboost", "svr", "knn", "gaussian_process", 
                        "neural_network", "lstm_like"
                    ]

                    specialized_algorithms = [
                        "seasonal_decomposition", "moving_average", "croston"
                    ]

                    # Select algorithms based on category
                    if config.algorithm == "best_statistical":
                        algorithms = statistical_algorithms
                    elif config.algorithm == "best_ml":
                        algorithms = ml_algorithms
                    elif config.algorithm == "best_specialized":
                        algorithms = specialized_algorithms
                    else:  # best_fit
                        algorithms = [alg for alg in ForecastingEngine.ALGORITHMS.keys() 
                                    if alg not in ["best_fit", "best_statistical", "best_ml", "best_specialized"]]
                    algorithm_results = []

                    # Use parallel execution for three-dimension best fit
                    max_workers = min(len(algorithms), os.cpu_count() or 4)
                    if process_log is not None:
                        process_log.append(f"Running parallel best fit with {max_workers} workers...")

                    # Create a copy of config for each algorithm
                    single_config = ForecastConfig(
                        forecastBy=config.forecastBy,
                        selectedProducts=[product],
                        selectedCustomers=[customer],
                        selectedLocations=[location],
                        algorithm=config.algorithm,
                        interval=config.interval,
                        historicPeriod=config.historicPeriod,
                        forecastPeriod=config.forecastPeriod,
                        multiSelect=False,
                        externalFactors=config.externalFactors
                    )

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_algorithm = {
                            executor.submit(ForecastingEngine.run_algorithm, algorithm, aggregated_df, single_config, save_model=False): algorithm
                            for algorithm in algorithms
                        }

                        for future in as_completed(future_to_algorithm):
                            algorithm_name = future_to_algorithm[future]
                            try:
                                result = future.result()
                                algorithm_results.append(result)
                                if process_log is not None:
                                    process_log.append(f"✅ Algorithm {algorithm_name} completed for {product}+{customer}+{location}")
                            except Exception as exc:
                                if process_log is not None:
                                    process_log.append(f"❌ Algorithm {algorithm_name} failed for {product}+{customer}+{location}: {str(exc)}")
                                continue

                    if not algorithm_results:
                        if process_log is not None:
                            process_log.append("No algorithms produced valid results")

                        failed_combinations.append({
                            'combination': f"{product} + {customer} + {location}",
                            'error': 'No algorithms produced valid results'
                        })
                        continue
                    best_result = max(algorithm_results, key=lambda x: x.accuracy)
                    forecast_result = ForecastResult(
                    combination={"product": product, "customer": customer, "location": location},
                    selectedAlgorithm=f"{best_result.algorithm} (Best Fit)",
                    accuracy=best_result.accuracy,
                    mae=best_result.mae,
                    rmse=best_result.rmse,
                    historicData=best_result.historicData,
                    forecastData=best_result.forecastData,
                    trend=best_result.trend,
                    allAlgorithms=algorithm_results,
                    processLog=process_log if process_log is not None else []
                )
                else:
                    if process_log is not None:
                        process_log.append(f"Running algorithm: {config.algorithm}")

                    result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, config)
                    forecast_result = ForecastResult(
                        combination={"product": product, "customer": customer, "location": location},
                        selectedAlgorithm=result.algorithm,
                        accuracy=result.accuracy,
                        mae=result.mae,
                        rmse=result.rmse,
                        historicData=result.historicData,
                        forecastData=result.forecastData,
                        trend=result.trend,
                        processLog=process_log if process_log is not None else []
                    )
                results.append(forecast_result)
                successful_combinations += 1

                if process_log is not None:
                    process_log.append(f"Combination {product} + {customer} + {location} processed successfully")

            except Exception as e:
                if process_log is not None:
                    process_log.append(f"Error processing combination {product} + {customer} + {location}: {str(e)}")

                failed_combinations.append({
                    'combination': f"{product} + {customer} + {location}",
                    'error': str(e)
                })

        if not results:
            raise ValueError("No valid forecasts could be generated for any combination")

        avg_accuracy = np.mean([r.accuracy for r in results])
        best_combination = max(results, key=lambda x: x.accuracy)
        worst_combination = min(results, key=lambda x: x.accuracy)

        summary = {
            'averageAccuracy': round(avg_accuracy, 2),
            'bestCombination': {
                'combination': best_combination.combination,
                'accuracy': best_combination.accuracy
            },
            'worstCombination': {
                'combination': worst_combination.combination,
                'accuracy': worst_combination.accuracy
            },
            'successfulCombinations': successful_combinations,
            'failedCombinations': len(failed_combinations),
            'failedDetails': failed_combinations
        }

        if process_log is not None:
            process_log.append(f"Three-dimension forecast completed. Successful: {successful_combinations}, Failed: {len(failed_combinations)}")

        return MultiForecastResult(
            results=results,
            totalCombinations=len(combinations),
            summary=summary,
            processLog=process_log if process_log is not None else []
        )


    @staticmethod
    def aggregate_two_dimensions(df: pd.DataFrame, interval: str, combination_type: str, dim1_value: str, dim2_value: str) -> pd.DataFrame:
        """
        Aggregate data specifically for two-dimensional combinations
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Filter data for the specific combination
        if combination_type == 'product_customer':
            # Filter for specific product and customer, aggregate across all locations
            filtered_df = df[(df['product'] == dim1_value) & (df['customer'] == dim2_value)]
            group_cols = ['product', 'customer']
        elif combination_type == 'product_location':
            # Filter for specific product and location, aggregate across all customers
            filtered_df = df[(df['product'] == dim1_value) & (df['location'] == dim2_value)]
            group_cols = ['product', 'location']
        elif combination_type == 'customer_location':
            # Filter for specific customer and location, aggregate across all products
            filtered_df = df[(df['customer'] == dim1_value) & (df['location'] == dim2_value)]
            group_cols = ['customer', 'location']
        else:
            raise ValueError(f"Unknown combination type: {combination_type}")

        if len(filtered_df) == 0:
            raise ValueError(f"No data found for combination: {dim1_value} + {dim2_value}")

        print(f"Filtered data shape: {filtered_df.shape}")
        print(f"Sample filtered data:\n{filtered_df.head()}")

        # Create period groupings
        if interval == 'week':
            filtered_df['period_group'] = filtered_df['date'].dt.to_period('W-MON')
        elif interval == 'month':
            filtered_df['period_group'] = filtered_df['date'].dt.to_period('M')
        elif interval == 'year':
            filtered_df['period_group'] = filtered_df['date'].dt.to_period('Y')
        else:
            filtered_df['period_group'] = filtered_df['date'].dt.to_period('M')

        # Group by the two dimensions and period, then sum quantities
        group_cols_with_period = group_cols + ['period_group']
        aggregated = filtered_df.groupby(group_cols_with_period)['quantity'].sum().reset_index()

        # Convert period back to timestamp
        aggregated['date'] = aggregated['period_group'].dt.start_time

        # Add period labels
        aggregated['period'] = aggregated['date'].apply(
            lambda x: ForecastingEngine.format_period(pd.Timestamp(x), interval)
        )

        # Keep essential columns plus external factors for forecasting
        external_factor_cols = [col for col in aggregated.columns if col not in ['product', 'customer', 'location', 'date', 'period', 'quantity', 'period_group']]
        result_cols = group_cols + ['date', 'period', 'quantity'] + external_factor_cols
        aggregated = aggregated[result_cols]

        print(f"Aggregated data shape: {aggregated.shape}")
        print(f"Sample aggregated data:\n{aggregated.head()}")

        return aggregated.reset_index(drop=True)

    @staticmethod
    def generate_forecast_two_dimensions(db: Session, config: ForecastConfig, products: list, customers: list, locations: list, process_log: List[str] = None) -> MultiForecastResult:
        """Generate forecasts for two dimension combinations with dedicated aggregation"""
        if process_log is not None:
            process_log.append("Generating two-dimension forecast...")

        combinations = []

        # Determine which two dimensions are selected and create combinations accordingly
        if products and customers:
            # Product + Customer selected, aggregate across all locations
            for p in products:
                for c in customers:
                    combinations.append(('product_customer', p, c))
        elif products and locations:
            # Product + Location selected, aggregate across all customers
            for p in products:
                for l in locations:
                    combinations.append(('product_location', p, l))
        elif customers and locations:
            # Customer + Location selected, aggregate across all products
            for c in customers:
                for l in locations:
                    combinations.append(('customer_location', c, l))

        if process_log is not None:
            process_log.append(f"Generated {len(combinations)} two-dimension combinations")

        # Load all data once (without filtering by specific combinations)
        base_config = ForecastConfig(
            forecastBy=config.forecastBy,
            selectedProducts=products,
            selectedCustomers=customers,
            selectedLocations=locations,
            algorithm=config.algorithm,
            interval=config.interval,
            historicPeriod=config.historicPeriod,
            forecastPeriod=config.forecastPeriod,
            multiSelect=False,  # Load all data
            externalFactors=config.externalFactors
        )

        # Load the full dataset once
        if process_log is not None:
            process_log.append("Loading full dataset...")

        full_df = ForecastingEngine.load_data_from_db(db, base_config)

        if process_log is not None:
            process_log.append(f"Full dataset loaded: {full_df.shape}")

        results = []
        successful_combinations = 0
        failed_combinations = []

        for combination_type, dim1_value, dim2_value in combinations:
            try:
                if process_log is not None:
                    process_log.append(f"\nProcessing combination: {combination_type} - {dim1_value} + {dim2_value}")

                # Use dedicated two-dimensional aggregation
                aggregated_df = ForecastingEngine.aggregate_two_dimensions(
                    full_df, config.interval, combination_type, dim1_value, dim2_value
                )

                if len(aggregated_df) < 2:
                    if process_log is not None:
                        process_log.append("Insufficient data after aggregation")

                    failed_combinations.append({
                        'combination': f"{combination_type}: {dim1_value} + {dim2_value}",
                        'error': 'Insufficient data after aggregation'
                    })
                    continue

                # Create combination dictionary for result
                if combination_type == 'product_customer':
                    combination_dict = {"product": dim1_value, "customer": dim2_value}
                elif combination_type == 'product_location':
                    combination_dict = {"product": dim1_value, "location": dim2_value}
                elif combination_type == 'customer_location':
                    combination_dict = {"customer": dim1_value, "location": dim2_value}

                # Create a simplified config for the algorithm (it doesn't need the multi-select complexity)
                algo_config = ForecastConfig(
                    forecastBy=config.forecastBy,
                    selectedProducts=None,
                    selectedCustomers=None,
                    selectedLocations=None,
                    algorithm=config.algorithm,
                    interval=config.interval,
                    historicPeriod=config.historicPeriod,
                    forecastPeriod=config.forecastPeriod,
                    multiSelect=False,
                    externalFactors=config.externalFactors
                )

                # Run forecasting algorithm
                if config.algorithm in ["best_fit", "best_statistical", "best_ml", "best_specialized"]:
                    if process_log is not None:
                        if config.algorithm == "best_fit":
                            process_log.append("Running best fit algorithm selection...")
                        elif config.algorithm == "best_statistical":
                            process_log.append("Running best statistical method selection...")
                        elif config.algorithm == "best_ml":
                            process_log.append("Running best machine learning method selection...")
                        elif config.algorithm == "best_specialized":
                            process_log.append("Running best specialized method selection...")

                    # Define algorithm categories
                    statistical_algorithms = [
                        "linear_regression", "polynomial_regression", "exponential_smoothing", 
                        "holt_winters", "arima", "sarima", "ses", "damped_trend", 
                        "theta_method", "drift_method", "naive_seasonal", "prophet_like"
                    ]

                    ml_algorithms = [
                        "random_forest", "xgboost", "svr", "knn", "gaussian_process", 
                        "neural_network", "lstm_like"
                    ]

                    specialized_algorithms = [
                        "seasonal_decomposition", "moving_average", "croston"
                    ]

                    # Select algorithms based on category
                    if config.algorithm == "best_statistical":
                        algorithms = statistical_algorithms
                    elif config.algorithm == "best_ml":
                        algorithms = ml_algorithms
                    elif config.algorithm == "best_specialized":
                        algorithms = specialized_algorithms
                    else:  # best_fit
                        algorithms = [alg for alg in ForecastingEngine.ALGORITHMS.keys() 
                                    if alg not in ["best_fit", "best_statistical", "best_ml", "best_specialized"]]

                    algorithm_results = []
                    # Use parallel execution for two-dimension best fit
                    max_workers = min(len(algorithms), os.cpu_count() or 4)
                    if process_log is not None:
                        process_log.append(f"Running parallel best fit with {max_workers} workers...")

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_algorithm = {
                            executor.submit(ForecastingEngine.run_algorithm, algorithm, aggregated_df, algo_config, save_model=False): algorithm
                            for algorithm in algorithms
                        }

                        for future in as_completed(future_to_algorithm):
                            algorithm_name = future_to_algorithm[future]
                            try:
                                result = future.result()
                                algorithm_results.append(result)
                                if process_log is not None:
                                    process_log.append(f"✅ Algorithm {algorithm_name} completed for {combination_type}: {dim1_value}+{dim2_value}")
                            except Exception as exc:
                                if process_log is not None:
                                    process_log.append(f"❌ Algorithm {algorithm_name} failed for {combination_type}: {dim1_value}+{dim2_value}: {str(exc)}")
                                continue

                    if not algorithm_results:
                        if process_log is not None:
                            process_log.append("No algorithms produced valid results")

                        failed_combinations.append({
                            'combination': f"{combination_type}: {dim1_value} + {dim2_value}",
                            'error': 'No algorithms produced valid results'
                        })
                        continue
                    best_result = max(algorithm_results, key=lambda x: x.accuracy)
                    forecast_result = ForecastResult(
                        combination=combination_dict,
                        selectedAlgorithm=f"{best_result.algorithm} (Best Fit)",
                        accuracy=best_result.accuracy,
                        mae=best_result.mae,
                        rmse=best_result.rmse,
                        historicData=best_result.historicData,
                        forecastData=best_result.forecastData,
                        trend=best_result.trend,
                        allAlgorithms=algorithm_results,
                        processLog=process_log if process_log is not None else []
                    )
                else:
                    if process_log is not None:
                        process_log.append(f"Running algorithm: {config.algorithm}")

                    result = ForecastingEngine.run_algorithm(config.algorithm, aggregated_df, algo_config)
                    forecast_result = ForecastResult(
                        combination=combination_dict,
                        selectedAlgorithm=result.algorithm,
                        accuracy=result.accuracy,
                        mae=result.mae,
                        rmse=result.rmse,
                        historicData=result.historicData,
                        forecastData=result.forecastData,
                        trend=result.trend,
                        processLog=process_log if process_log is not None else []
                    )

                results.append(forecast_result)
                successful_combinations += 1

                if process_log is not None:
                    process_log.append(f"Successfully processed: {combination_type} - {dim1_value} + {dim2_value}")

            except Exception as e:
                if process_log is not None:
                    process_log.append(f"Error processing combination {dim1_value} + {dim2_value}: {e}")

                failed_combinations.append({
                    'combination': f"{combination_type}: {dim1_value} + {dim2_value}",
                    'error': str(e)
                })

        if not results:
            raise ValueError("No valid forecasts could be generated for any combination")

        avg_accuracy = np.mean([r.accuracy for r in results])
        best_combination = max(results, key=lambda x: x.accuracy)
        worst_combination = min(results, key=lambda x: x.accuracy)

        summary = {
            'averageAccuracy': round(avg_accuracy, 2),
            'bestCombination': {
                'combination': best_combination.combination,
                'accuracy': best_combination.accuracy
            },
            'worstCombination': {
                'combination': worst_combination.combination,
                'accuracy': worst_combination.accuracy
            },
            'successfulCombinations': successful_combinations,
            'failedCombinations': len(failed_combinations),
            'failedDetails': failed_combinations
        }

        if process_log is not None:
            process_log.append(f"Two-dimension forecast completed. Successful: {successful_combinations}, Failed: {len(failed_combinations)}")

        return MultiForecastResult(
            results=results,
            totalCombinations=len(combinations),
            summary=summary,
            processLog=process_log if process_log is not None else []
        )

    @staticmethod
    def generate_multi_forecast(db: Session, config: ForecastConfig, process_log: List[str] = None) -> MultiForecastResult:
        """Generate forecasts for multiple combinations"""
        if process_log is not None:
            process_log.append("Generating multi-forecast...")

        products = config.selectedProducts or []
        customers = config.selectedCustomers or []
        locations = config.selectedLocations or []

        if process_log is not None:
            process_log.append(f"Multi-forecast: products={len(products)}, customers={len(customers)}, locations={len(locations)}")

        selected_dimensions = 0
        if products: selected_dimensions += 1
        if customers: selected_dimensions += 1
        if locations: selected_dimensions += 1

        if selected_dimensions < 2:
            raise ValueError("Please select at least two dimensions (Product, Customer, or Location) for multi-selection forecasting")

        if selected_dimensions == 3:
            if process_log is not None:
                process_log.append("Running three-dimension forecast")
                # Check if all three dimensions are selected for advanced mode
                if not (config.selectedProducts and config.selectedCustomers and config.selectedLocations):
                    process_log.append("ERROR: Advanced mode requires all three dimensions to be selected")
                    raise ValueError("Advanced mode requires selection of Products, Customers, and Locations")
                
            return ForecastingEngine.generate_forecast_three_dimensions(db, config, 
                config.selectedProducts, config.selectedCustomers, config.selectedLocations, process_log)
        else:
            if process_log is not None:
                process_log.append("Running two-dimension forecast")
            return ForecastingEngine.generate_forecast_two_dimensions(db, config, products, customers, locations, process_log)


# Add global flag for model caching
ENABLE_MODEL_CACHE = True

# Health check endpoint
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "message": "Advanced Multi-variant Forecasting API with MySQL is running",
        "algorithms": list(ForecastingEngine.ALGORITHMS.values())
    }

# Internal wrapper for scheduler
def generate_forecast_internal(forecast_config: dict, db_session):
    """
    Internal wrapper function for the scheduler to call the existing forecast engine
    """
    try:
        if forecast_config.get('selectedProduct') and forecast_config.get('selectedCustomer') and forecast_config.get('selectedLocation'):
            config = ForecastConfig(
                forecastBy=forecast_config.get('forecastBy', 'product'),
                selectedProduct=forecast_config.get('selectedProduct'),
                selectedCustomer=forecast_config.get('selectedCustomer'),
                selectedLocation=forecast_config.get('selectedLocation'),
                algorithm=forecast_config.get('algorithm', 'best_fit'),
                interval=forecast_config.get('interval', 'month'),
                historicPeriod=forecast_config.get('historicPeriod', 12),
                forecastPeriod=forecast_config.get('forecastPeriod', 6),
                multiSelect=False
            )
        elif forecast_config.get('multiSelect'):
            config = ForecastConfig(
                forecastBy=forecast_config.get('forecastBy', 'product'),
                selectedProducts=forecast_config.get('selectedProducts', []),
                selectedCustomers=forecast_config.get('selectedCustomers', []),
                selectedLocations=forecast_config.get('selectedLocations', []),
                algorithm=forecast_config.get('algorithm', 'best_fit'),
                interval=forecast_config.get('interval', 'month'),
                historicPeriod=forecast_config.get('historicPeriod', 12),
                forecastPeriod=forecast_config.get('forecastPeriod', 6),
                multiSelect=True
            )
        else:
            config = ForecastConfig(
                forecastBy=forecast_config.get('forecastBy', 'product'),
                selectedItem=forecast_config.get('selectedItem'),
                algorithm=forecast_config.get('algorithm', 'best_fit'),
                interval=forecast_config.get('interval', 'month'),
                historicPeriod=forecast_config.get('historicPeriod', 12),
                forecastPeriod=forecast_config.get('forecastPeriod', 6),
                multiSelect=False
            )
        
        process_log = []
        forecast_result = ForecastingEngine.generate_forecast(db_session, config, process_log)
        
        result_dict = {
            "selectedAlgorithm": forecast_result.selectedAlgorithm,
            "accuracy": forecast_result.accuracy,
            "mae": forecast_result.mae,
            "rmse": forecast_result.rmse,
            "trend": forecast_result.trend,
            "configHash": forecast_result.configHash,
            "processLog": forecast_result.processLog,
            "historicData": [
                {
                    "date": point.date.isoformat() if hasattr(point.date, 'isoformat') else str(point.date),
                    "quantity": point.quantity,
                    "period": point.period
                }
                for point in forecast_result.historicData
            ],
            "forecastData": [
                {
                    "date": point.date.isoformat() if hasattr(point.date, 'isoformat') else str(point.date),
                    "quantity": point.quantity,
                    "period": point.period
                }
                for point in forecast_result.forecastData
            ]
        }
        
        if hasattr(forecast_result, 'combination') and forecast_result.combination:
            result_dict["combination"] = forecast_result.combination
        
        if config.multiSelect:
            result_dict = {
                "results": [result_dict],
                "totalCombinations": 1,
                "summary": {
                    "successfulCombinations": 1,
                    "failedCombinations": 0,
                    "averageAccuracy": forecast_result.accuracy
                }
            }
        
        return result_dict
        
    except Exception as e:
        error_msg = f"Forecast generation failed: {str(e)}"
        print(f"Error in generate_forecast_internal: {error_msg}")
        raise Exception(error_msg)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Advanced Multi-variant Forecasting API with MySQL")
    print("📊 23 Algorithms + Best Fit Available")
    print("🗄️  MySQL Database Integration")
    print("🌐 Server starting on http://localhost:8000")
    print("📈 Frontend should be available on http://localhost:5173")
    print("ℹ️  Press Ctrl+C to stop the server\n")

    start_scheduler()
    uvicorn.run(app, host="0.0.0.0", port=8000)