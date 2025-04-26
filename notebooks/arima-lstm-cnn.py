#!/usr/bin/env python
# coding: utf-8

# --- Standard Libraries ---
import copy
import time
import os
import warnings
from datetime import datetime, timedelta
import logging

# --- Data Manipulation ---
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm # For Q-Q plot

# --- Machine Learning ---
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# --- Configuration ---
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
TARGET = 'PJME_MW'
DATA_PATH = '../data/3/PJME_hourly.csv' # Adjust if needed
OUTPUT_DIR = 'forecasting_output' # Directory to save logs and plots
N_STEPS = 24 # Sequence length for LSTM/CNN
BATCH_SIZE = 64
EPOCHS = 50 # Max epochs for LSTM/CNN training
PATIENCE_ES = 10 # Early stopping patience
LR = 5e-4
WEIGHT_DECAY = 1e-5
ARIMA_ORDER = (2, 0, 1)
SEASONAL_ORDER = (1, 0, 0, 24)

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Check for GPU Availability ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- Start Script Timer ---
script_start_time = time.time()

# ==============================================================================
# 1. Load and Preprocess Data (Energy + Weather)
# ==============================================================================
def load_and_preprocess_data(file_path):
    """Loads energy data, engineers features, fetches weather data, and merges."""
    logging.info("Loading and preprocessing data...")
    start = time.time()

    # Load PJME hourly energy consumption data
    try:
        pjme = (
            pd.read_csv(
                file_path,
                index_col='Datetime',
                parse_dates=['Datetime']
            )
            .sort_index()
            .loc['2003-01-01':'2018-08-02'] # Adjust date range if needed
        )
    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}")
        raise

    # --- Feature Engineering: Holidays and Calendar Features ---
    cal = USFederalHolidayCalendar()
    fed_hols = cal.holidays(start=pjme.index.min(), end=pjme.index.max())
    # Simple holiday flag (can be enhanced as per original notes)
    pjme['is_holiday'] = pjme.index.normalize().isin(fed_hols)
    pjme['is_weekend'] = pjme.index.weekday >= 5
    pjme['is_dayoff'] = pjme['is_holiday'] | pjme['is_weekend']
    pjme.drop(columns=['is_holiday', 'is_weekend'], inplace=True)
    pjme['hour']        = pjme.index.hour
    pjme['dayofweek']   = pjme.index.weekday
    pjme['month']       = pjme.index.month
    pjme['year']        = pjme.index.year
    pjme['dayofyear']   = pjme.index.dayofyear

    # --- Fetch and Process Weather Data ---
    # Define representative coordinates and stations for PJME region (example: Philly area)
    # Adjust coordinates and ICAOs if needed for better representation
    lat, lon = 39.95, -75.17
    start_dt = pjme.index.min() - timedelta(days=1) # Fetch slightly earlier for join buffer
    end_dt   = pjme.index.max() + timedelta(days=1)
    target_icaos = ['KPHL', 'KEWR', 'KBWI', 'KDCA'] # Example major airports in PJME

    try:
        stations_query = Stations().nearby(lat, lon).inventory('hourly', (start_dt, end_dt))
        target_stations_df = stations_query.fetch()[lambda x: x['icao'].isin(target_icaos)]
        station_ids = target_stations_df.index.tolist()

        if not station_ids:
            logging.warning("No target weather stations found. Proceeding without weather data.")
            average_weather = pd.DataFrame(index=pjme.index) # Empty df to allow join
        else:
            weather_all = Hourly(station_ids, start_dt, end_dt).fetch()
            weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wspd']
            # Aggregate weather, forward fill missing values
            average_weather = weather_all.groupby(level='time').mean(numeric_only=True)[weather_cols].ffill()

    except Exception as e:
        logging.warning(f"Could not fetch or process weather data: {e}. Proceeding without weather features.")
        average_weather = pd.DataFrame(index=pjme.index) # Empty df

    # --- Combine Energy and Weather Data ---
    pjme_weather = pjme.join(average_weather, how='left')
    # Fill any remaining NaNs (e.g., start/end of weather data)
    pjme_weather = pjme_weather.ffill().bfill()

    logging.info(f"Combined data shape: {pjme_weather.shape}")
    logging.info(f"Index monotonic? {pjme_weather.index.is_monotonic_increasing}")
    logging.info(f"Data loading and preprocessing took {time.time() - start:.2f} seconds.")
    return pjme_weather

# ==============================================================================
# 2. Train/Validation/Test Split
# ==============================================================================
def split_data(df, val_ratio=0.1, test_ratio=0.1):
    """Splits data chronologically into train, validation, and test sets."""
    logging.info("Splitting data...")
    total_len = len(df)
    test_split_idx = int(total_len * (1 - test_ratio))
    val_split_idx = int(total_len * (1 - test_ratio - val_ratio))

    train_df = df.iloc[:val_split_idx].copy()
    val_df = df.iloc[val_split_idx:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()

    logging.info(f"Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    logging.info(f"Val  : {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    logging.info(f"Test : {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
    return train_df, val_df, test_df

# ==============================================================================
# 3. ARIMA Modeling
# ==============================================================================
def plot_acf_pacf(series, lags, filename):
    """Plots and saves ACF and PACF plots."""
    logging.info("Generating ACF/PACF plots...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], title='Autocorrelation Function (ACF)')
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title='Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"ACF/PACF plots saved to {filename}")
    plt.close(fig)

def check_stationarity(series):
    """Performs ADF test and logs results."""
    logging.info("Checking stationarity using Augmented Dickey-Fuller test...")
    adf_result = adfuller(series.dropna())
    logging.info(f'ADF Statistic: {adf_result[0]}')
    logging.info(f'p-value: {adf_result[1]}')
    if adf_result[1] > 0.05:
        logging.warning("Series is likely non-stationary (p > 0.05). Consider differencing (d=1 in ARIMA).")
    else:
        logging.info("Series is likely stationary (p <= 0.05).")

def fit_arima_predict(train_df, val_df, test_df, target_col, arima_order, seasonal_order):
    """Fits SARIMA model and generates predictions for all sets."""
    logging.info(f"Fitting SARIMA{arima_order}x{seasonal_order} model...")
    start = time.time()

    # Create copies to avoid SettingWithCopyWarning
    train_df_res = train_df.copy()
    val_df_res = val_df.copy()
    test_df_res = test_df.copy()

    try:
        # Fit the SARIMA model on the training data
        arima_model = ARIMA(
            train_df_res[target_col],
            order=arima_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        arima_results = arima_model.fit()
        logging.info(arima_results.summary())

        # --- Get predictions ---
        # Train predictions (in-sample, non-dynamic)
        train_pred = arima_results.get_prediction(
            start=train_df_res.index[0],
            end=train_df_res.index[-1],
            dynamic=False
        )
        train_df_res['ARIMA_pred'] = train_pred.predicted_mean

        # Validation predictions (out-of-sample, 1-step ahead using training model state)
        # Extend the model state to include validation period for smoother 1-step forecasts
        extended_model = arima_results.extend(val_df_res[target_col])
        val_pred = extended_model.get_prediction(
            start=val_df_res.index[0],
             end=val_df_res.index[-1],
             dynamic=False # Still 1-step ahead
        )
        val_df_res['ARIMA_pred'] = val_pred.predicted_mean

        # Test predictions (multi-step forecast from end of validation)
        # We forecast from the state *after* observing validation data
        n_test = len(test_df_res)
        test_fc = extended_model.get_forecast(steps=n_test)
        test_df_res['ARIMA_pred'] = test_fc.predicted_mean.values # Ensure correct length alignment

        logging.info("ARIMA fitting and forecasting complete.")

    except Exception as e:
        logging.error(f"Error fitting/predicting with ARIMA: {e}. Falling back to naive persistence forecast.")
        # Fallback: Use previous hour's value as prediction
        for df in (train_df_res, val_df_res, test_df_res):
            df['ARIMA_pred'] = df[target_col].shift(1)
        # Handle initial NaN
        train_df_res['ARIMA_pred'].fillna(method='bfill', inplace=True)
        val_df_res['ARIMA_pred'].fillna(train_df_res['ARIMA_pred'].iloc[-1], inplace=True) # Use last train value
        test_df_res['ARIMA_pred'].fillna(val_df_res['ARIMA_pred'].iloc[-1], inplace=True)   # Use last val value

    # Compute residuals
    train_df_res['Residual'] = train_df_res[target_col] - train_df_res['ARIMA_pred']
    val_df_res['Residual']   = val_df_res[target_col] - val_df_res['ARIMA_pred']
    test_df_res['Residual']  = test_df_res[target_col] - test_df_res['ARIMA_pred']

    # Drop any NaNs potentially introduced by ARIMA start or fallback
    train_df_res.dropna(subset=['ARIMA_pred', 'Residual'], inplace=True)
    val_df_res.dropna(subset=['ARIMA_pred', 'Residual'], inplace=True)
    test_df_res.dropna(subset=['ARIMA_pred', 'Residual'], inplace=True)

    logging.info(f"ARIMA modeling took {time.time() - start:.2f} seconds.")
    return train_df_res, val_df_res, test_df_res

# ==============================================================================
# 4. LSTM-CNN Data Preparation
# ==============================================================================
def prepare_lstm_cnn_data(train_df, val_df, test_df, n_steps, batch_size):
    """Scales features, creates sequences, and prepares DataLoaders."""
    logging.info("Preparing data for LSTM-CNN...")
    start = time.time()

    # Identify features (all columns except target and residuals)
    features_exog = [col for col in train_df.columns if col not in [TARGET, 'ARIMA_pred', 'Residual']]
    logging.info(f"Using exogenous features: {features_exog}")

    # Ensure all required weather columns exist, fill with 0 if missing (due to earlier warning)
    weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wspd']
    for df in [train_df, val_df, test_df]:
        for col in weather_cols:
            if col not in df.columns:
                logging.warning(f"Weather feature '{col}' not found. Filling with 0.")
                df[col] = 0

    # Extract raw arrays
    X_train_raw = train_df[features_exog].values
    X_val_raw   = val_df[features_exog].values
    X_test_raw  = test_df[features_exog].values

    y_train_raw = train_df['Residual'].values.reshape(-1, 1)
    y_val_raw   = val_df['Residual'].values.reshape(-1, 1)
    y_test_raw  = test_df['Residual'].values.reshape(-1, 1)

    # Fit scalers ONLY on training data
    scaler_X = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train_raw)

    # Transform all sets
    X_train_scaled = scaler_X.transform(X_train_raw)
    X_val_scaled   = scaler_X.transform(X_val_raw)
    X_test_scaled  = scaler_X.transform(X_test_raw)

    y_train_scaled = scaler_y.transform(y_train_raw).flatten()
    y_val_scaled   = scaler_y.transform(y_val_raw).flatten()
    y_test_scaled  = scaler_y.transform(y_test_raw).flatten()

    # Build sliding windows
    def create_sequences(X, y, n_steps):
        Xs, ys = [], []
        for i in range(len(X) - n_steps):
            Xs.append(X[i : i + n_steps])
            ys.append(y[i + n_steps]) # Target is the residual *after* the sequence
        return np.array(Xs), np.array(ys)

    X_train, y_train_res = create_sequences(X_train_scaled, y_train_scaled, n_steps)
    X_val,   y_val_res   = create_sequences(X_val_scaled,   y_val_scaled,   n_steps)
    X_test,  y_test_res  = create_sequences(X_test_scaled,  y_test_scaled,  n_steps)

    logging.info(f"Sequence shapes: Train X:{X_train.shape}, y:{y_train_res.shape} | Val X:{X_val.shape}, y:{y_val_res.shape} | Test X:{X_test.shape}, y:{y_test_res.shape}")

    # Convert numpy arrays to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_res, dtype=torch.float32).unsqueeze(-1)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val_res,   dtype=torch.float32).unsqueeze(-1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_res,  dtype=torch.float32).unsqueeze(-1)

    # Build TensorDatasets
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t,   y_val_t)
    test_ds  = TensorDataset(X_test_t,  y_test_t)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    logging.info(f"LSTM-CNN data preparation took {time.time() - start:.2f} seconds.")
    return train_loader, val_loader, test_loader, scaler_X, scaler_y, X_test.shape[2], len(features_exog) # Return n_features

# ==============================================================================
# 5. LSTM-CNN Model Definition (PyTorch)
# ==============================================================================
class LSTMCNN(nn.Module):
    def __init__(
        self,
        input_dim,       # Number of features
        n_steps,         # Sequence length
        hidden_dim=128,
        cnn_filters=64,
        kernel_size=3,
        pool_size=2,
        dropout_rate=0.3,
        num_lstm_layers=2,
        output_dim=1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # Conv blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters, kernel_size, padding='same'), # Use 'same' padding
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.MaxPool1d(pool_size)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding='same'), # Use 'same' padding
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.MaxPool1d(pool_size)
        )

        # LSTM layer
        # Input size to LSTM is the number of filters from the last CNN layer
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0 # Add dropout between LSTM layers if more than 1
        )

        # Output layer
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input x shape: (batch_size, seq_len, n_features)
        x = x.permute(0, 2, 1)  # -> (batch_size, n_features, seq_len) for Conv1D
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.permute(0, 2, 1)  # -> (batch_size, new_seq_len, cnn_filters) for LSTM

        # LSTM part
        # Initialize hidden and cell states (optional, default to zeros)
        # h0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_dim).to(x.device)
        # _, (hn, _) = self.lstm(x, (h0, c0)) # Pass initialized states

        lstm_out, (hn, _) = self.lstm(x) # Default zero states

        # Use the hidden state from the last time step of the last layer
        last_hidden = hn[-1] # Shape: (batch_size, hidden_dim)

        # Apply LayerNorm and Dropout
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        # Fully connected layer
        out = self.fc(last_hidden) # Shape: (batch_size, output_dim)
        return out

# ==============================================================================
# 6. LSTM-CNN Training (PyTorch)
# ==============================================================================
def train_lstm_cnn(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs, patience_es, device, model_save_path, loss_plot_path):
    """Trains the LSTM-CNN model with early stopping and saves the best model."""
    logging.info("Starting LSTM-CNN training...")
    start = time.time()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    history = {'train': [], 'val': []}

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        # --- Train ---
        model.train()
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            running_train_loss += loss.item() * xb.size(0)
        train_loss = running_train_loss / len(train_loader.dataset)
        history['train'].append(train_loss)

        # --- Validate ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                running_val_loss += criterion(preds, yb).item() * xb.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)
        history['val'].append(val_loss)

        epoch_duration = time.time() - epoch_start
        logging.info(f"Epoch {epoch}/{n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.1e}, Time: {epoch_duration:.2f}s")

        # --- LR Scheduling & Early Stopping ---
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, model_save_path) # Save best model state
            logging.info(f"Validation loss improved. Best model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_es:
                logging.info(f"Early stopping triggered after {epoch} epochs.")
                break

    logging.info(f"LSTM-CNN training finished in {time.time() - start:.2f} seconds.")

    # Restore best model if early stopping occurred
    if best_state:
        model.load_state_dict(best_state)
        logging.info("Loaded best model state for final evaluation.")

    # Plot and save loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['train'], label='Train Loss')
    ax.plot(history['val'],   label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('LSTM-CNN Training and Validation Loss')
    ax.legend()
    plt.savefig(loss_plot_path)
    logging.info(f"Training loss plot saved to {loss_plot_path}")
    plt.close(fig)

    return model # Return the best trained model

# ==============================================================================
# 7. Generate Final Predictions
# ==============================================================================
def generate_final_predictions(model, test_loader, scaler_y, test_df_orig, n_steps, device):
    """Generates LSTM-CNN residual predictions and combines with ARIMA."""
    logging.info("Generating final hybrid predictions...")
    start = time.time()
    model.eval()
    all_preds_scaled = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds_scaled = model(xb).cpu().numpy()
            all_preds_scaled.extend(preds_scaled.flatten())

    # Inverse transform the scaled residual predictions
    residual_preds = scaler_y.inverse_transform(np.array(all_preds_scaled).reshape(-1, 1)).flatten()

    # Align predictions with the original test dataframe
    # The predictions correspond to the time steps *after* the n_steps lookback period
    test_df_aligned = test_df_orig.iloc[n_steps:].copy()

    # Ensure lengths match before assignment
    if len(residual_preds) != len(test_df_aligned):
         logging.warning(f"Mismatch in prediction length ({len(residual_preds)}) and aligned test df length ({len(test_df_aligned)}). Truncating predictions.")
         min_len = min(len(residual_preds), len(test_df_aligned))
         test_df_aligned = test_df_aligned.iloc[:min_len]
         residual_preds = residual_preds[:min_len]


    test_df_aligned['Residual_pred'] = residual_preds
    test_df_aligned['Combined_pred'] = test_df_aligned['ARIMA_pred'] + test_df_aligned['Residual_pred']

    logging.info(f"Final prediction generation took {time.time() - start:.2f} seconds.")
    return test_df_aligned

# ==============================================================================
# 8. Evaluation
# ==============================================================================
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE, handling zero values in y_true."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask): # Handle case where all true values are zero
        return 0.0 if np.allclose(y_pred, 0) else np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(y_true, y_pred, model_name):
    """Calculates and logs RMSE, MAE, and MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    logging.info(f"--- {model_name} Test Set Error Metrics ---")
    logging.info(f"RMSE: {rmse:,.2f} MW")
    logging.info(f"MAE:  {mae:,.2f} MW")
    logging.info(f"MAPE: {mape:.2f}%")

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# ==============================================================================
# 9. Visualization
# ==============================================================================
# ==============================================================================
# 9. Visualization
# ==============================================================================
def plot_forecasts(df, target_col, pred_cols, title, filename, zoom_days=7):
    """Plots and saves actual vs. predicted values."""
    logging.info(f"Generating forecast plot: {title}")
    fig, ax = plt.subplots(figsize=(15, 6))
    df[target_col].plot(ax=ax, label='Actual', style='.', alpha=0.7)
    # Use consistent styles for the main plot
    main_styles = ['-', '--']
    for i, col in enumerate(pred_cols):
        style = main_styles[i % len(main_styles)] # Cycle through styles if more preds than styles
        if col in df.columns:
            df[col].plot(ax=ax, label=col.replace('_', ' ').title(), style=style, alpha=0.9)

    ax.set_title(title)
    ax.set_ylabel('MW')
    ax.legend()
    plt.savefig(filename)
    logging.info(f"Forecast plot saved to {filename}")
    plt.close(fig)

    # Zoomed plot
    if zoom_days and len(df) > zoom_days * 24:
        # Ensure start_date allows for a full zoom_days period if possible
        max_start_index = max(0, len(df) - zoom_days * 24)
        start_index = np.random.randint(0, max_start_index + 1) if max_start_index > 0 else 0
        start_date = df.index[start_index]
        end_date = start_date + pd.Timedelta(days=zoom_days)
        # Adjust end_date if it exceeds the data range
        end_date = min(end_date, df.index[-1])
        zoom_df = df.loc[start_date:end_date]


        if not zoom_df.empty:
            fig_zoom, ax_zoom = plt.subplots(figsize=(15, 6))
            # Plot actuals with line and marker for zoom
            zoom_df[target_col].plot(ax=ax_zoom, label='Actual', style='.-')
            # Use the same base styles but add markers for predictions
            for i, col in enumerate(pred_cols):
                 style = main_styles[i % len(main_styles)] # Use the same base style as main plot
                 if col in zoom_df.columns:
                    # Use original style but add a marker for zoomed plot clarity
                    zoom_df[col].plot(ax=ax_zoom, label=col.replace('_', ' ').title(), style=style, marker='.') # CORRECTED LINE

            ax_zoom.set_title(f'{title} (Zoomed: {start_date.date()} to {end_date.date()})')
            ax_zoom.set_ylabel('MW')
            ax_zoom.legend()
            zoom_filename = filename.replace('.png', '_zoom.png')
            plt.savefig(zoom_filename)
            logging.info(f"Zoomed forecast plot saved to {zoom_filename}")
            plt.close(fig_zoom)
        else:
            logging.warning(f"Could not create zoomed plot for {title}, possibly due to slicing issues.")
# ==============================================================================
# 10. Log Results
# ==============================================================================
def log_results(metrics, runtime, filename):
    """Saves evaluation metrics and runtime to a text file."""
    logging.info(f"Saving results to {filename}")
    with open(filename, 'w') as f:
        f.write("--- Evaluation Metrics ---\n")
        for model_name, model_metrics in metrics.items():
            f.write(f"\n{model_name}:\n")
            for metric_name, value in model_metrics.items():
                 if isinstance(value, float):
                    f.write(f"  {metric_name}: {value:.4f}\n")
                 else:
                    f.write(f"  {metric_name}: {value}\n")

        f.write("\n--- Runtime ---\n")
        f.write(f"Total Script Runtime: {runtime:.2f} seconds\n")
    logging.info("Results saved.")

# ==============================================================================
# Main Script Execution
# ==============================================================================
if __name__ == "__main__":
    # 1. Load Data
    full_data = load_and_preprocess_data(DATA_PATH)

    # 2. Split Data
    train_data, val_data, test_data = split_data(full_data)

    # 3. ARIMA Modeling
    check_stationarity(train_data[TARGET])
    plot_acf_pacf(train_data[TARGET], lags=48, filename=os.path.join(OUTPUT_DIR, 'acf_pacf_raw.png'))
    # Optional: Plot ACF/PACF on differenced data if non-stationary
    # plot_acf_pacf(train_data[TARGET].diff().dropna(), lags=48, filename=os.path.join(OUTPUT_DIR, 'acf_pacf_diff1.png'))

    train_data, val_data, test_data = fit_arima_predict(
        train_data, val_data, test_data, TARGET, ARIMA_ORDER, SEASONAL_ORDER
    )

    # Optional: Plot ARIMA residuals ACF/PACF
    if 'Residual' in train_data.columns:
        plot_acf_pacf(train_data['Residual'], lags=48, filename=os.path.join(OUTPUT_DIR, 'acf_pacf_arima_residuals.png'))
        # Q-Q plot of residuals
        fig_qq = sm.qqplot(train_data['Residual'].dropna(), line='s')
        plt.title('Q-Q Plot of ARIMA Residuals')
        plt.savefig(os.path.join(OUTPUT_DIR, 'qqplot_arima_residuals.png'))
        plt.close(fig_qq)


    # 4. Prepare LSTM-CNN Data
    train_loader, val_loader, test_loader, scaler_X, scaler_y, n_features, _ = prepare_lstm_cnn_data(
        train_data, val_data, test_data, N_STEPS, BATCH_SIZE
    )

    # 5. Define LSTM-CNN Model
    lstm_cnn_model = LSTMCNN(
        input_dim=n_features,
        n_steps=N_STEPS, # Pass sequence length
        # Other hyperparameters can be adjusted here or kept as defaults
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(lstm_cnn_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE_ES//2) # Scheduler patience < early stopping

    logging.info(f"LSTM-CNN Model Architecture:\n{lstm_cnn_model}")

    # 6. Train LSTM-CNN Model
    best_lstm_cnn_model = train_lstm_cnn(
        model=lstm_cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=EPOCHS,
        patience_es=PATIENCE_ES,
        device=device,
        model_save_path=os.path.join(OUTPUT_DIR, 'best_lstm_cnn_model.pt'),
        loss_plot_path=os.path.join(OUTPUT_DIR, 'lstm_cnn_training_loss.png')
    )

    # 7. Generate Final Predictions
    test_results_df = generate_final_predictions(
        model=best_lstm_cnn_model,
        test_loader=test_loader,
        scaler_y=scaler_y,
        test_df_orig=test_data, # Use the original test_data before sequence creation
        n_steps=N_STEPS,
        device=device
    )

    # 8. Evaluate Models
    all_metrics = {}
    y_true_test = test_results_df[TARGET]

    # Evaluate ARIMA-only
    if 'ARIMA_pred' in test_results_df.columns:
         all_metrics['ARIMA_Only'] = evaluate_predictions(
            y_true_test, test_results_df['ARIMA_pred'], 'ARIMA Only'
        )

    # Evaluate Combined Hybrid Model
    if 'Combined_pred' in test_results_df.columns:
        all_metrics['ARIMA_LSTM_CNN_Hybrid'] = evaluate_predictions(
            y_true_test, test_results_df['Combined_pred'], 'Combined Hybrid'
        )

    # 9. Visualize Results
    plot_forecasts(
        df=test_results_df,
        target_col=TARGET,
        pred_cols=['ARIMA_pred', 'Combined_pred'],
        title='Test Set: Actual vs. Forecasts',
        filename=os.path.join(OUTPUT_DIR, 'test_forecast_comparison.png')
    )

    # --- Error Analysis (Optional - uncomment if needed) ---
    test_results_df['error'] = test_results_df[TARGET] - test_results_df['Combined_pred']
    test_results_df['abs_error'] = test_results_df['error'].abs()
    error_by_day = test_results_df.groupby(test_results_df.index.date).agg(
        mean_abs_error=('abs_error', 'mean')
    )
    worst_day_date = error_by_day['mean_abs_error'].idxmax()
    best_day_date = error_by_day['mean_abs_error'].idxmin()
    logging.info(f"Worst prediction day (MAE): {worst_day_date} ({error_by_day.loc[worst_day_date].values[0]:.2f} MW)")
    logging.info(f"Best prediction day (MAE): {best_day_date} ({error_by_day.loc[best_day_date].values[0]:.2f} MW)")
    plot_forecasts(test_results_df.loc[str(worst_day_date)], TARGET, ['Combined_pred'], f'Worst Day ({worst_day_date})', os.path.join(OUTPUT_DIR,'worst_day_forecast.png'), zoom_days=None)
    plot_forecasts(test_results_df.loc[str(best_day_date)], TARGET, ['Combined_pred'], f'Best Day ({best_day_date})', os.path.join(OUTPUT_DIR, 'best_day_forecast.png'), zoom_days=None)

    # 10. Log Results & Runtime
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time
    logging.info(f"Total script execution time: {total_runtime:.2f} seconds.")

    log_results(
        metrics=all_metrics,
        runtime=total_runtime,
        filename=os.path.join(OUTPUT_DIR, 'evaluation_summary.txt')
    )

    logging.info("Script finished successfully.")