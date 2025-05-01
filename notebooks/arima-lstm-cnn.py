#!/usr/bin/env python
# coding: utf-8

# --- Standard Libraries ---
import copy
from datetime import datetime, timedelta
import warnings
import logging
import os
import sys

# --- Data Manipulation ---
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- Machine Learning ---
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

# --- Time Series Modeling ---
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Weather Data ---
from meteostat import Stations, Hourly

# --- Configure Logging ---
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)
forecast_dir = '../forecast_files'
os.makedirs(forecast_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'arima_lstm_cnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Configure Warnings and Plot Style ---
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

# --- Check for GPU Availability ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


def load_and_preprocess_data():
    """Load and preprocess energy and weather data"""
    logging.info("Loading PJME hourly energy consumption data...")
    pjme = (
        pd.read_csv(
            '../data/3/PJME_hourly.csv',
            index_col='Datetime',
            parse_dates=['Datetime']
        )
        .sort_index()
        .loc['2003-01-01':'2018-08-02']
    )
    
    logging.info("Adding calendar features and holidays...")
    # --- Feature Engineering: Holidays and Calendar Features ---
    cal = USFederalHolidayCalendar()
    fed_hols = cal.holidays(start=pjme.index.min(), end=pjme.index.max())
    extended_hols = set(fed_hols)
    for year in range(pjme.index.year.min(), pjme.index.year.max() + 1):
        july4 = datetime(year, 7, 4)
        wd = july4.weekday()
        if wd == 1: extended_hols.add(july4 - timedelta(days=1))
        elif wd == 2: extended_hols.add(july4 + timedelta(days=1)); extended_hols.add(july4 + timedelta(days=2))
        elif wd == 3: extended_hols.add(july4 + timedelta(days=1))
    all_hols = pd.DatetimeIndex(sorted(extended_hols))
    pjme['is_holiday'] = pjme.index.normalize().isin(all_hols)
    pjme['is_weekend'] = pjme.index.weekday >= 5
    pjme['is_dayoff'] = pjme['is_holiday'] | pjme['is_weekend']
    pjme.drop(columns=['is_holiday', 'is_weekend'], inplace=True)
    # pjme['hour']       = pjme.index.hour
    # pjme['dayofweek']  = pjme.index.weekday
    # pjme['month']      = pjme.index.month
    # pjme['year']       = pjme.index.year
    # pjme['dayofyear']  = pjme.index.dayofyear

    logging.info("Fetching and processing weather data...")
    # --- Fetch and Process Weather Data ---
    start_dt = datetime(2002, 12, 31)
    end_dt   = datetime(2018, 8, 4)
    target_icaos = ['KPHL', 'KEWR', 'KBWI', 'KDCA']
    stations_query = Stations().nearby(39.95, -75.17).inventory('hourly', (start_dt, end_dt))
    target_stations_df = stations_query.fetch()[lambda x: x['icao'].isin(target_icaos)]
    station_ids = target_stations_df.index.tolist()
    logging.info(f"Using weather stations: {target_stations_df['name'].tolist()}")
    weather_all = Hourly(station_ids, start_dt, end_dt).fetch()
    weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wspd']
    average_weather = weather_all.groupby(level='time').mean(numeric_only=True)[weather_cols].ffill()

    logging.info("Combining energy and weather data...")
    # --- Combine Energy and Weather Data ---
    pjme_weather = pjme.join(average_weather, how='left').dropna()

    logging.info(f'Combined data shape: {pjme_weather.shape}')
    logging.info(f'Index monotonic? {pjme_weather.index.is_monotonic_increasing}')

    # --- Handle DST Duplicates ---
    if not pjme_weather.index.is_unique:
        logging.warning("Duplicate timestamps found in pjme_weather index! (Likely DST)")
        logging.info("Resolving by averaging values for duplicate timestamps...")
        # Group by index, calculate mean for all columns, then sort index again
        pjme_weather = pjme_weather.groupby(pjme_weather.index).mean()
        pjme_weather = pjme_weather.sort_index() # Ensure index is sorted after grouping
        logging.info(f"Shape after averaging duplicates: {pjme_weather.shape}")
        logging.info(f"Index is now unique: {pjme_weather.index.is_unique}")
    
    return pjme_weather


def split_data(df, train_pct=0.8, val_pct=0.1):
    """Split data into train, validation, and test sets"""
    logging.info("Splitting data into train, validation, and test sets...")
    total_hours = len(df)
    test_split_idx = int(total_hours * (train_pct + val_pct))
    val_split_idx = int(total_hours * train_pct)

    train_df = df.iloc[:val_split_idx].copy()
    val_df = df.iloc[val_split_idx:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()

    logging.info(f"Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    logging.info(f"Val  : {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    logging.info(f"Test : {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
    
    return train_df, val_df, test_df


def check_stationarity(series):
    """Check stationarity of time series using ADF test"""
    logging.info("Performing ADF test for stationarity...")
    adf_result = adfuller(series)
    logging.info(f'ADF Statistic: {adf_result[0]}')
    logging.info(f'p-value: {adf_result[1]}')
    logging.info('Critical Values:')
    for key, value in adf_result[4].items():
        logging.info(f'\t{key}: {value}')

    if adf_result[1] > 0.05:
        logging.info("Series is likely non-stationary. Consider differencing (d=1).")
        return False
    else:
        logging.info("Series is likely stationary. We can proceed without differencing.")
        return True


def plot_acf_pacf(series, lags=40):
    """Plot ACF and PACF for time series analysis"""
    logging.info(f"Plotting ACF and PACF with {lags} lags...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=lags, ax=axes[0], title='Autocorrelation Function (ACF)')
    plot_pacf(series, lags=lags, ax=axes[1], title='Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    plt.savefig(os.path.join(forecast_dir, 'acf_pacf_plot.png'))
    logging.info(f"ACF/PACF plot saved to {os.path.join(forecast_dir, 'acf_pacf_plot.png')}")
    plt.close()


def create_diagnostic_plots(train_df, val_df, test_df, target_col):
    """Create and save diagnostic plots for model evaluation"""
    logging.info("Creating diagnostic plots...")
    
    # 1. Actual vs Predicted Time Series Plot
    for df_name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        if 'ARIMA_pred' in df.columns:
            plt.figure(figsize=(15, 6))
            
            # Plot only a subset for clarity if dataset is large
            if len(df) > 1000:
                sample_size = 1000
                logging.info(f"Dataset too large, plotting last {sample_size} points for {df_name}")
                plot_df = df.iloc[-sample_size:]
            else:
                plot_df = df
                
            plt.plot(plot_df.index, plot_df[target_col], label='Actual', alpha=0.7)
            plt.plot(plot_df.index, plot_df['ARIMA_pred'], label='Predicted', alpha=0.7)
            plt.title(f'{df_name} Set: Actual vs Predicted Energy Consumption')
            plt.xlabel('Date')
            plt.ylabel('Energy Consumption (MW)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(forecast_dir, f'{df_name.lower()}_actual_vs_predicted.png'))
            logging.info(f"Saved {df_name} actual vs predicted plot")
            plt.close()
    
    # 2. Residual Analysis Plots
    for df_name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        if 'Residual' in df.columns:
            # 2.1 Residual Time Series
            plt.figure(figsize=(15, 6))
            plt.plot(df.index, df['Residual'], color='blue', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title(f'{df_name} Set: Residual Time Series')
            plt.xlabel('Date')
            plt.ylabel('Residual (Actual - Predicted)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(forecast_dir, f'{df_name.lower()}_residual_time_series.png'))
            logging.info(f"Saved {df_name} residual time series plot")
            plt.close()
            
            # 2.2 Residual Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df['Residual'], kde=True, bins=50)
            plt.title(f'{df_name} Set: Residual Distribution')
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(forecast_dir, f'{df_name.lower()}_residual_distribution.png'))
            logging.info(f"Saved {df_name} residual distribution plot")
            plt.close()
            
            # 2.3 QQ Plot of Residuals
            plt.figure(figsize=(10, 6))
            sm.qqplot(df['Residual'].dropna(), line='45', fit=True)
            plt.title(f'{df_name} Set: Q-Q Plot of Residuals')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(forecast_dir, f'{df_name.lower()}_residual_qq_plot.png'))
            logging.info(f"Saved {df_name} residual QQ plot")
            plt.close()
    
    # 3. Forecast Error Analysis
    # 3.1 Error by Hour of Day
    for df_name, df in [("Validation", val_df), ("Test", test_df)]:
        if 'Residual' in df.columns and 'hour' in df.columns:
            hourly_error = df.groupby('hour')['Residual'].agg(['mean', 'std'])
            
            plt.figure(figsize=(12, 6))
            plt.errorbar(hourly_error.index, hourly_error['mean'], 
                         yerr=hourly_error['std'], fmt='o-', capsize=5)
            plt.title(f'{df_name} Set: Mean Forecast Error by Hour of Day')
            plt.xlabel('Hour of Day')
            plt.ylabel('Mean Error (with std dev)')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24))
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(forecast_dir, f'{df_name.lower()}_error_by_hour.png'))
            logging.info(f"Saved {df_name} error by hour plot")
            plt.close()
    
    # 3.2 Error by Day of Week
    for df_name, df in [("Validation", val_df), ("Test", test_df)]:
        if 'Residual' in df.columns and 'dayofweek' in df.columns:
            daily_error = df.groupby('dayofweek')['Residual'].agg(['mean', 'std'])
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            plt.figure(figsize=(12, 6))
            plt.errorbar(daily_error.index, daily_error['mean'], 
                         yerr=daily_error['std'], fmt='o-', capsize=5)
            plt.title(f'{df_name} Set: Mean Forecast Error by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Mean Error (with std dev)')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 7), day_names, rotation=45)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(forecast_dir, f'{df_name.lower()}_error_by_day.png'))
            logging.info(f"Saved {df_name} error by day plot")
            plt.close()
    
    # 3.3 Error by Month
    for df_name, df in [("Validation", val_df), ("Test", test_df)]:
        if 'Residual' in df.columns and 'month' in df.columns:
            monthly_error = df.groupby('month')['Residual'].agg(['mean', 'std'])
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            plt.figure(figsize=(12, 6))
            plt.errorbar(monthly_error.index, monthly_error['mean'], 
                         yerr=monthly_error['std'], fmt='o-', capsize=5)
            plt.title(f'{df_name} Set: Mean Forecast Error by Month')
            plt.xlabel('Month')
            plt.ylabel('Mean Error (with std dev)')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(1, 13), month_names, rotation=45)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(forecast_dir, f'{df_name.lower()}_error_by_month.png'))
            logging.info(f"Saved {df_name} error by month plot")
            plt.close()
    
    # 4. Prediction Accuracy Scatter Plot
    for df_name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        if 'ARIMA_pred' in df.columns:
            plt.figure(figsize=(10, 10))
            plt.scatter(df[target_col], df['ARIMA_pred'], alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(df[target_col].min(), df['ARIMA_pred'].min())
            max_val = max(df[target_col].max(), df['ARIMA_pred'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title(f'{df_name} Set: Actual vs Predicted Scatter Plot')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(forecast_dir, f'{df_name.lower()}_scatter_plot.png'))
            logging.info(f"Saved {df_name} scatter plot")
            plt.close()
    
    logging.info("All diagnostic plots created and saved successfully")


def fit_arima_model(train_df, target_col, arima_order, seasonal_order):
    """Fit ARIMA/SARIMA model to training data"""
    logging.info(f"Fitting SARIMA{arima_order}×{seasonal_order} model...")
    try:
        arima_model = ARIMA(
            train_df[target_col],
            order=arima_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        arima_results = arima_model.fit()
        logging.info("ARIMA model fitting successful")
        logging.info(f"AIC: {arima_results.aic}, BIC: {arima_results.bic}")
        
        # Get one-step-ahead predictions for training data
        logging.info("Generating one-step-ahead forecasts for training data...")
        train_pred = arima_results.get_prediction(
            start=train_df.index[0],
            end=train_df.index[-1],
            dynamic=False
        )
        train_df['ARIMA_pred'] = train_pred.predicted_mean
        train_df.dropna(subset=['ARIMA_pred'], inplace=True)
        
        # Calculate training metrics
        train_rmse = np.sqrt(mean_squared_error(train_df[target_col], train_df['ARIMA_pred']))
        train_mae = mean_absolute_error(train_df[target_col], train_df['ARIMA_pred'])
        logging.info(f"Training RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
        
        return arima_results
    
    except Exception as e:
        logging.error(f"Error fitting ARIMA model: {str(e)}")
        return None


def forecast_validation_set(arima_results, train_df, val_df, target_col, error_on_fallback=True):
    """Generate forecasts for validation set using rolling window approach"""
    logging.info("Generating rolling forecasts for validation set...")
    val_predictions = []
    buffer = []
    current_model = arima_results  # Start with the trained model
    
    # Determine appropriate update interval based on data density
    # For sparse data (sampled), use a smaller update interval
    if len(val_df) < 200:  # If we're using a small sample
        update_interval = max(1, min(24, len(val_df) // 10))  # Update at least once, at most every 24 points
        logging.info(f"Using reduced update interval of {update_interval} for sparse validation data")
    else:
        update_interval = 168  # Weekly updates (24*7 hours) for full data
    
    window_size = max(update_interval * 4, 24)  # Keep at least 4 update intervals of history
    
    logging.info(f"Validation forecast settings: update_interval={update_interval}, window_size={window_size}")
    
    for i in range(len(val_df)):
        if i % max(1, len(val_df) // 10) == 0:
            logging.info(f"Validation forecasting progress: {i}/{len(val_df)} steps")
            
        # 1. Forecast next step
        fc = current_model.get_forecast(steps=1)
        val_predictions.append(fc.predicted_mean.iloc[0])
        
        # 2. Store actual value in FIFO buffer
        buffer.append(val_df[target_col].iloc[i])
        if len(buffer) > window_size:
            buffer.pop(0)  # Maintain fixed window size
            
        # 3. Update model periodically with most recent data
        if (i + 1) % update_interval == 0 and len(buffer) >= update_interval:
            # Revert to re-fitting the model with combined historical and new data
            logging.info(f"Attempting model re-fit at validation step {i+1}")
            
            try:
                # Get the most recent chunk of actual data observed
                recent_buffer = buffer[-update_interval:]
                recent_index = val_df.index[max(0, i+1-update_interval):i+1]
                
                # Ensure lengths match exactly
                if len(recent_buffer) != len(recent_index):
                    min_len = min(len(recent_buffer), len(recent_index))
                    recent_buffer = recent_buffer[-min_len:]
                    recent_index = recent_index[-min_len:]
                
                update_data = pd.Series(recent_buffer, index=recent_index)

                # Combine original training data with all validation data up to this point
                # This ensures a contiguous index for re-fitting
                combined_series = pd.concat([train_df[target_col], val_df[target_col].iloc[:i+1]])
                
                # Ensure no duplicate indices (preferring the latest data)
                combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
                
                # Re-fit the model
                try:
                    # Get the original model orders
                    model_order = arima_results.model.order
                    seasonal_order = arima_results.model.seasonal_order
                    
                    current_model = ARIMA(
                        combined_series,
                        order=model_order,
                        seasonal_order=seasonal_order
                    ).fit()
                    logging.info(f"Model re-fit successful at validation step {i+1}")
                except Exception as e:
                    logging.error(f"Validation model re-fit failed at step {i}: {str(e)}")
                    if error_on_fallback:
                        raise RuntimeError(f"Model re-fit failed and error_on_fallback=True: {str(e)}")
                    else:
                        logging.warning(f"Continuing with the model from step {i+1-update_interval} due to re-fit failure.")
                        # Continue with the existing model if fallback is allowed
            
            except Exception as e:
                logging.error(f"Unexpected error during model update preparation at step {i}: {str(e)}")
                # Continue with the existing model if an error occurs during data prep

    val_df['ARIMA_pred'] = val_predictions
    
    # Calculate validation metrics
    val_rmse = np.sqrt(mean_squared_error(val_df[target_col], val_df['ARIMA_pred']))
    val_mae = mean_absolute_error(val_df[target_col], val_df['ARIMA_pred'])
    logging.info(f"Validation RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}")
    
    return True


def forecast_test_set(arima_results, train_val_df, test_df, target_col, error_on_fallback=True):
    """Generate forecasts for test set using rolling window approach"""
    logging.info("Generating rolling forecasts for test set...")
    test_predictions = []
    buffer = []
    current_model = arima_results  # Start with the trained model
    
    # Determine appropriate update interval based on data density
    # For sparse data (sampled), use a smaller update interval
    if len(test_df) < 200:  # If we're using a small sample
        update_interval = max(1, min(24, len(test_df) // 10))  # Update at least once, at most every 24 points
        logging.info(f"Using reduced update interval of {update_interval} for sparse test data")
    else:
        update_interval = 168  # Weekly updates (24*7 hours) for full data
    
    window_size = max(update_interval * 4, 24)  # Keep at least 4 update intervals of history
    
    logging.info(f"Test forecast settings: update_interval={update_interval}, window_size={window_size}")
    
    for i in range(len(test_df)):
        if i % max(1, len(test_df) // 10) == 0:
            logging.info(f"Test forecasting progress: {i}/{len(test_df)} steps")
            
        # 1. Forecast next step
        fc = current_model.get_forecast(steps=1)
        test_predictions.append(fc.predicted_mean.iloc[0])
        
        # 2. Store actual value in FIFO buffer
        buffer.append(test_df[target_col].iloc[i])
        if len(buffer) > window_size:
            buffer.pop(0)  # Maintain fixed window size
            
        # 3. Update model periodically with most recent data
        if (i + 1) % update_interval == 0 and len(buffer) >= update_interval:
            # Revert to re-fitting the model with combined historical and new data
            logging.info(f"Attempting model re-fit at test step {i+1}")
            
            try:
                # Combine original train+validation data with all test data up to this point
                # This ensures a contiguous index for re-fitting
                combined_series = pd.concat([train_val_df[target_col], test_df[target_col].iloc[:i+1]])
                
                # Ensure no duplicate indices (preferring the latest data)
                combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
                
                # Re-fit the model
                try:
                    # Get the original model orders
                    model_order = arima_results.model.order
                    seasonal_order = arima_results.model.seasonal_order
                    
                    current_model = ARIMA(
                        combined_series,
                        order=model_order,
                        seasonal_order=seasonal_order
                    ).fit()
                    logging.info(f"Model re-fit successful at test step {i+1}")
                except Exception as e:
                    logging.error(f"Test model re-fit failed at step {i}: {str(e)}")
                    if error_on_fallback:
                        raise RuntimeError(f"Model re-fit failed and error_on_fallback=True: {str(e)}")
                    else:
                        logging.warning(f"Continuing with the model from step {i+1-update_interval} due to re-fit failure.")
                        # Continue with the existing model if fallback is allowed
            
            except Exception as e:
                logging.error(f"Unexpected error during model update preparation at step {i}: {str(e)}")
                # Continue with the existing model if an error occurs during data prep

    test_df['ARIMA_pred'] = test_predictions
    
    # Calculate test metrics
    test_rmse = np.sqrt(mean_squared_error(test_df[target_col], test_df['ARIMA_pred']))
    test_mae = mean_absolute_error(test_df[target_col], test_df['ARIMA_pred'])
    logging.info(f"Test RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")
    
    return True


def compute_residuals(train_df, val_df, test_df, target_col):
    """Compute residuals for all datasets"""
    logging.info("Computing residuals...")
    for df_name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        if 'ARIMA_pred' in df.columns:
            df['Residual'] = df[target_col] - df['ARIMA_pred']
            residual_std = df['Residual'].std()
            logging.info(f"{df_name} residual std: {residual_std:.2f}")
        else:
            logging.warning(f"No ARIMA predictions found for {df_name} set")


def fallback_naive_forecast(train_df, val_df, test_df, target_col):
    """Fallback to naive 1-lag shift forecast if ARIMA fails"""
    logging.warning("Using fallback naive 1-lag shift forecast")
    for df_name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        df['ARIMA_pred'] = df[target_col].shift(1).fillna(method='bfill')
        if not df['ARIMA_pred'].isna().any():
            rmse = np.sqrt(mean_squared_error(df[target_col], df['ARIMA_pred']))
            mae = mean_absolute_error(df[target_col], df['ARIMA_pred'])
            logging.info(f"{df_name} naive forecast - RMSE: {rmse:.2f}, MAE: {mae:.2f}")


def main(use_sample=False, sample_fraction=0.01, error_on_fallback=True, sample_method='window'):
    """Main function to run the ARIMA modeling pipeline
    
    Parameters:
    -----------
    use_sample : bool, default=False
        Whether to use a sample of the data for prototyping
    sample_fraction : float, default=0.01
        Fraction of data to use if use_sample=True (1/100 by default)
    error_on_fallback : bool, default=True
        Whether to raise an error when model update fails instead of falling back to re-fitting
    sample_method : str, default='window'
        Method to use for sampling: 'window' for a contiguous time window, 
        'resample' for lower frequency resampling
    """
    logging.info("Starting ARIMA modeling pipeline")
    logging.info(f"Settings: use_sample={use_sample}, sample_fraction={sample_fraction}, "
                f"error_on_fallback={error_on_fallback}, sample_method={sample_method}")
    
    # Load and preprocess data
    pjme_weather = load_and_preprocess_data()
    
    # Use a sample of the data for prototyping if requested
    if use_sample:
        logging.info(f"Using {sample_fraction*100:.2f}% of data for prototyping")
        
        if sample_method == 'window':
            # Use a contiguous time window to preserve hourly granularity
            total_days = (pjme_weather.index[-1] - pjme_weather.index[0]).days
            window_days = int(total_days * sample_fraction)
            # Ensure we have at least 30 days of data (or the entire dataset if smaller)
            window_days = max(30, min(window_days, total_days))
            
            # Select a window from the middle of the dataset for more representative data
            middle_idx = len(pjme_weather) // 2
            start_idx = middle_idx - (window_days * 24) // 2
            end_idx = start_idx + (window_days * 24)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(pjme_weather), end_idx)
            
            pjme_weather = pjme_weather.iloc[start_idx:end_idx].copy()
            logging.info(f"Using contiguous window of {window_days} days "
                        f"({pjme_weather.index[0]} to {pjme_weather.index[-1]})")
            
        elif sample_method == 'resample':
            # Determine appropriate resampling frequency based on sample_fraction
            if sample_fraction <= 0.01:
                # For 1% or less, use daily averages (24x reduction)
                resample_freq = '24H'
                seasonal_adjustment = 1  # Adjust seasonal period from 24 to 1
            elif sample_fraction <= 0.05:
                # For 5% or less, use 6-hourly averages (6x reduction)
                resample_freq = '6H'
                seasonal_adjustment = 4  # Adjust seasonal period from 24 to 4
            elif sample_fraction <= 0.1:
                # For 10% or less, use 3-hourly averages (3x reduction)
                resample_freq = '3H'
                seasonal_adjustment = 8  # Adjust seasonal period from 24 to 8
            else:
                # For larger samples, use 2-hourly averages (2x reduction)
                resample_freq = '2H'
                seasonal_adjustment = 12  # Adjust seasonal period from 24 to 12
            
            # Resample the data to lower frequency
            pjme_weather = pjme_weather.resample(resample_freq).mean()
            
            # Ensure the frequency is properly set
            pjme_weather = pjme_weather.asfreq(resample_freq)
            
            logging.info(f"Resampled data to {resample_freq} frequency, shape: {pjme_weather.shape}")
            logging.info(f"Seasonal adjustment factor: {seasonal_adjustment} (adjust seasonal_order accordingly)")
        else:
            # Original method (not recommended) - taking every Nth point
            logging.warning("Using original sampling method which breaks time series frequency")
            sample_step = int(1/sample_fraction)
            pjme_weather = pjme_weather.iloc[::sample_step].copy()
            logging.info(f"Sampled data shape: {pjme_weather.shape}")
    
    # Split data
    TARGET = 'PJME_MW'
    train_df, val_df, test_df = split_data(pjme_weather)
    
    # Check stationarity
    is_stationary = check_stationarity(train_df[TARGET])
    
    # Plot ACF and PACF
    plot_acf_pacf(train_df[TARGET])
    
    # Define ARIMA orders based on analysis
    arima_order = (2, 0, 1)
    
    # Adjust seasonal order if we've resampled the data
    if use_sample and sample_method == 'resample':
        # Get the original seasonal period
        original_seasonal_period = 24  # daily seasonality
        
        # Calculate the new seasonal period based on resampling
        if sample_fraction <= 0.01:
            new_seasonal_period = 1  # Daily data, so period is 1 day
        elif sample_fraction <= 0.05:
            new_seasonal_period = 4  # 6-hourly data, so period is 4 per day
        elif sample_fraction <= 0.1:
            new_seasonal_period = 8  # 3-hourly data, so period is 8 per day
        else:
            new_seasonal_period = 12  # 2-hourly data, so period is 12 per day
        
        seasonal_order = (1, 0, 0, new_seasonal_period)
        logging.info(f"Adjusted seasonal period from {original_seasonal_period} to {new_seasonal_period} due to resampling")
    else:
        seasonal_order = (1, 0, 0, 24)  # daily seasonality
    
    # Fit ARIMA model
    arima_results = fit_arima_model(train_df, TARGET, arima_order, seasonal_order)
    
    if arima_results is not None:
        # Generate forecasts for validation set
        try:
            forecast_validation_set(arima_results, train_df, val_df, TARGET, error_on_fallback=error_on_fallback)
            
            # Combine train and validation for test forecasting
            train_val_df = pd.concat([train_df, val_df])
            
            # Generate forecasts for test set
            forecast_test_set(arima_results, train_val_df, test_df, TARGET, error_on_fallback=error_on_fallback)
            
            # Compute residuals
            compute_residuals(train_df, val_df, test_df, TARGET)
            
            # Create diagnostic plots
            create_diagnostic_plots(train_df, val_df, test_df, TARGET)
            
            # Save results
            logging.info("Saving results...")
            results_dir = '../results'
            os.makedirs(results_dir, exist_ok=True)
            train_df.to_csv(os.path.join(results_dir, 'train_with_predictions.csv'))
            val_df.to_csv(os.path.join(results_dir, 'val_with_predictions.csv'))
            test_df.to_csv(os.path.join(results_dir, 'test_with_predictions.csv'))
            
        except Exception as e:
            logging.error(f"Forecasting failed with error: {str(e)}")
            logging.info("Falling back to naive forecast for comparison")
            fallback_naive_forecast(train_df, val_df, test_df, TARGET)
    else:
        logging.error("ARIMA model fitting failed. Using naive forecast instead.")
        fallback_naive_forecast(train_df, val_df, test_df, TARGET)
    
    logging.info("ARIMA modeling pipeline completed")


if __name__ == "__main__":
    # For prototyping, use a small sample of the data and make the function error out
    # when model update fails instead of falling back to re-fitting
    main(use_sample=True, sample_fraction=0.1, error_on_fallback=True, sample_method='window')

