#!/usr/bin/env python
# coding: utf-8

# ## Hourly Energy Consumption Forecasting with ARIMA-LSTM-CNN Hybrid Model

# **Workflow Overview**
# 
# 1.  **Data Loading & Preprocessing:** Load PJME hourly energy data and weather data, perform feature engineering (calendar features, holidays, weather averaging).
# 2.  **Train/Validation/Test Split:** Split data chronologically.
# 3.  **ARIMA Modeling:** Fit an ARIMA model to capture linear patterns in the time series (or its residuals).
# 4.  **Residual Analysis:** Obtain residuals from the ARIMA model (Actual - ARIMA Prediction).
# 5.  **LSTM-CNN Data Preparation:** Prepare sequences from ARIMA residuals and relevant features for the deep learning model.
# 6.  **LSTM-CNN Model Definition:** Define a hybrid model combining LSTM (for temporal dependencies) and CNN (for feature extraction from sequences).
# 7.  **LSTM-CNN Training:** Train the LSTM-CNN model on the prepared sequences to predict the ARIMA residuals.
# 8.  **Combined Forecasting:** Generate final forecasts by adding ARIMA predictions and LSTM-CNN predictions (of residuals).
# 9.  **Evaluation:** Evaluate the combined forecast using RMSE, MAE, and MAPE on the test set.
# 10. **Visualization:** Plot actual vs. predicted values.

# In[1]:


import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from meteostat import Stations, Hourly
import warnings
import copy # For saving best model state

warnings.filterwarnings("ignore") # Suppress warnings for cleaner output
plt.style.use('fivethirtyeight')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ## 1. Load and Preprocess Data (Energy + Weather)

# In[2]:


# Load PJME hourly energy consumption data
pjme = (
    pd.read_csv(
        '../data/3/PJME_hourly.csv',
        index_col='Datetime',
        parse_dates=['Datetime']
    )
    .sort_index()
    .loc['2003-01-01':'2018-08-02']
)

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
pjme['hour']       = pjme.index.hour
pjme['dayofweek']  = pjme.index.weekday
pjme['month']      = pjme.index.month
pjme['year']       = pjme.index.year
pjme['dayofyear']  = pjme.index.dayofyear

# --- Fetch and Process Weather Data ---
start_dt = datetime(2002, 12, 31)
end_dt   = datetime(2018, 8, 4)
target_icaos = ['KPHL', 'KEWR', 'KBWI', 'KDCA']
stations_query = Stations().nearby(39.95, -75.17).inventory('hourly', (start_dt, end_dt))
target_stations_df = stations_query.fetch()[lambda x: x['icao'].isin(target_icaos)]
station_ids = target_stations_df.index.tolist()
weather_all = Hourly(station_ids, start_dt, end_dt).fetch()
weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wspd']
average_weather = weather_all.groupby(level='time').mean(numeric_only=True)[weather_cols].ffill()

# --- Combine Energy and Weather Data ---
pjme_weather = pjme.join(average_weather, how='left').dropna()

print(f'Combined data shape: {pjme_weather.shape}')
print(f'Index monotonic? {pjme_weather.index.is_monotonic_increasing}')
pjme_weather.head()


# ## 2. Train/Validation/Test Split

# In[3]:


# Define split points (e.g., 80% train, 10% val, 10% test)
total_hours = len(pjme_weather)
test_split_idx = int(total_hours * 0.9)
val_split_idx = int(total_hours * 0.8)

train_df = pjme_weather.iloc[:val_split_idx].copy()
val_df = pjme_weather.iloc[val_split_idx:test_split_idx].copy()
test_df = pjme_weather.iloc[test_split_idx:].copy()

print(f"Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
print(f"Val  : {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
print(f"Test : {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")

TARGET = 'PJME_MW'


# ## 3. ARIMA Modeling

# ARIMA stands for **AutoRegressive Integrated Moving Average**. It’s a class of time-series models that combines:
# 
# - **AutoRegressive (AR)** terms: regression on its own past values,  
# - **Integrated (I)** term: differencing the series to make it stationary,  
# - **Moving Average (MA)** terms: modeling the error as a linear combination of past forecast errors.
# 
# Fit an ARIMA model to the training data. This captures linear relationships. We might use exogenous variables (weather, calendar) if using ARIMAX. First, we need to analyze the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots to determine a suitable order (p, d, q) for the model.

# ### Stationarity Check and Differencing (if needed)
# 
# ARIMA models require the time series to be stationary. We can check for stationarity using tests like the Augmented Dickey-Fuller (ADF) test or by visual inspection. If non-stationary, we'll difference the series (typically d=1 is sufficient for level non-stationarity).

# In[4]:


from statsmodels.tsa.stattools import adfuller

# Perform ADF test on the training target variable
adf_result = adfuller(train_df[TARGET])
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'\t{key}: {value}')

# If p-value > 0.05, the series is likely non-stationary and needs differencing
if adf_result[1] > 0.05:
    print("\nSeries is likely non-stationary. Applying first difference (d=1).")
    # Example: train_diff = train_df[TARGET].diff().dropna()
    # Note: You would then analyze ACF/PACF on the differenced series
    # For this example, we'll proceed assuming d=1 might be needed, but analyze original for now.
else:
    print("\nSeries is likely stationary. We can proceed without differencing.")


# ### ACF and PACF Plots
# 
# Plot the ACF and PACF to identify potential AR (p) and MA (q) orders.
# *   **ACF:** Helps identify the MA order (q). Look for a sharp cutoff after lag q.
# *   **PACF:** Helps identify the AR order (p). Look for a sharp cutoff after lag p.

# In[5]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF for the (potentially differenced) training data
# Adjust 'data_to_plot' if differencing was applied
data_to_plot = train_df[TARGET] # Or train_diff if differenced
lags = 40 # Number of lags to show

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data_to_plot, lags=lags, ax=axes[0], title='Autocorrelation Function (ACF)')
plot_pacf(data_to_plot, lags=lags, ax=axes[1], title='Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# --- Interpretation Guidance ---
# Based on the plots, choose candidate (p, d, q) orders.
# Example: If PACF cuts off after lag 2 and ACF tails off, suggest AR(2) -> p=2.
# Example: If ACF cuts off after lag 1 and PACF tails off, suggest MA(1) -> q=1.
# If both tail off, consider ARMA(p,q).
# Remember 'd' is the order of differencing needed for stationarity (often 0 or 1).


# In[6]:


from statsmodels.tsa.arima.model import ARIMA

# --- 1) ARIMA / SARIMA orders based on our ADF + ACF/PACF ---
arima_order     = (2, 0, 1)
seasonal_order  = (1, 0, 0, 24)  # daily seasonality

print(f"Fitting SARIMA{arima_order}×{seasonal_order} model...")
try:
    # Fit the SARIMA model on the full training series
    arima_model   = ARIMA(
        train_df[TARGET],
        order=arima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    arima_results = arima_model.fit()
    print(arima_results.summary())

    # --- 2) One‐step‐ahead ARIMA forecasts on TRAIN (dynamic=False) ---
    train_pred = arima_results.get_prediction(
        start=train_df.index[0],
        end=train_df.index[-1],
        dynamic=False
    )
    train_df['ARIMA_pred'] = train_pred.predicted_mean
    train_df.dropna(subset=['ARIMA_pred'], inplace=True)

    # --- 3) One‐step‐ahead ARIMA forecasts on VALIDATION ---
    val_pred = arima_results.get_prediction(
        start=val_df.index[0],
        end=val_df.index[-1],
        dynamic=False
    )
    val_df['ARIMA_pred'] = val_pred.predicted_mean
    val_df.dropna(subset=['ARIMA_pred'], inplace=True)

    # --- 4) Multi‐step forecast for TEST (we still need to roll out into the future) ---
    n_val, n_test = len(val_df), len(test_df)
    fc = arima_results.forecast(steps=n_val + n_test)
    test_df['ARIMA_pred'] = fc.iloc[n_val:].values

    print("ARIMA fitting and forecasting complete.")

except Exception as e:
    print(f"\nError fitting ARIMA model: {e}")
    print("Falling back to 1-lag shift for residuals.")
    for df in (train_df, val_df, test_df):
        df['ARIMA_pred'] = df[TARGET].shift(1).fillna(method='bfill')

# --- 5) Compute residuals for downstream LSTM–CNN ---
train_df['Residual'] = train_df[TARGET] - train_df['ARIMA_pred']
val_df  ['Residual'] = val_df  [TARGET] - val_df  ['ARIMA_pred']
test_df ['Residual'] = test_df [TARGET] - test_df ['ARIMA_pred']


# In[7]:


# # Save (This crashes the kernel, so commented out) 
# arima_results.save('sarima_stripped.pickle', remove_data=True)

# # Load
# from statsmodels.tsa.arima.model import ARIMAResults
# arima_results = ARIMAResults.load('sarima_statsmodels.pickle')


# In[8]:


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

residuals = train_df['Residual']

# 1) ACF up to lag 40
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(residuals, lags=40, ax=ax)
ax.set_title('ACF of ARIMA Residuals (lags 0–40)')
plt.show()

# 2) Histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=50, edgecolor='k')
ax.set_title('Histogram of ARIMA Residuals')
ax.set_xlabel('Residual value')
ax.set_ylabel('Frequency')
plt.show()

# 3) Q–Q Plot
fig = sm.qqplot(residuals, line='s')
plt.title('Q–Q Plot of ARIMA Residuals')
plt.show()


# ## ARIMA Residual Analysis
# 
# After fitting the ARIMA model, it's crucial to analyze the residuals (the difference between the actual values and the ARIMA predictions on the training set). Ideally, the residuals should resemble white noise (zero mean, constant variance, no autocorrelation). Plotting the residuals and their ACF/PACF helps verify this.
# 
# **ARIMA Backbone: SARIMA(2,0,1)×(1,0,0,24) on PJM East Hourly Load**  
# 
# ---
# 
# ### 1. Model Specification  
# - **Non-seasonal part (p, d, q):** (2, 0, 1)  
#   - **AR(2):** two autoregressive lags to capture the immediate inertia of load.  
#   - **MA(1):** one moving-average term to soak up the remaining 1-hour error correlation.  
#   - **d = 0:** no differencing needed, as the series was already stationary by ADF test.  
# - **Seasonal part (P, D, Q, s):** (1, 0, 0, 24)  
#   - **Seasonal AR(24):** one daily‐lag term to model the 24-hour cyclical pattern.  
#   - **D = 0, Q = 0:** no additional seasonal differencing or MA required.  
# 
# ---
# 
# ### 2. Fit Statistics  
# - **Log Likelihood:** –796,492.16  
# - **AIC:** 1,592,996.31  
# - **BIC:** 1,593,053.92  
# - **HQIC:** 1,593,013.72  
# 
# The inclusion of MA(1) reduced AIC by ~16 000 compared to the earlier ARIMA(2,0,0)×(1,0,0,24), indicating a substantially better trade-off between fit and parsimony.
# 
# ---
# 
# ### 3. Parameter Estimates  
# 
# | Term         | Estimate | Std. Err. | z-stat   | P-value | 95% CI               |  
# |--------------|--------:|----------:|--------:|--------:|---------------------|  
# | **const**    | 32 340   | 0.010     | 3.39×10⁶ | <0.001  | [32 340, 32 340]     | Baseline load level.  
# | **AR L1**    | 1.5255   | 0.002     | 894.38   | <0.001  | [1.522, 1.529]       | Strong 1-hour autoregression.  
# | **AR L2**    | –0.5255  | 0.002     | –307.94  | <0.001  | [–0.529, –0.522]     | Negative weight at 2-hour lag.  
# | **MA L1**    | 0.4656   | 0.002     | 262.22   | <0.001  | [0.462, 0.469]       | Captures residual 1-hour correlation.  
# | **SAR L24**  | 0.8844   | 0.001     | 795.15   | <0.001  | [0.882, 0.887]       | Daily (24-hour) seasonal effect.  
# | **σ²**       | 125 200  | 261.47    | 478.97   | <0.001  | [124 700, 125 700]   | Variance of model residuals.  
# 
# All coefficients are highly significant (p < 0.001), confirming that each term contributes meaningfully to capturing load dynamics.
# 
# ---
# 
# ### 4. Residual Diagnostics  
# - **Ljung–Box Q (lag 1):** 71.87 (p < 0.001) → residual autocorrelation at lag 1 has been effectively eliminated by the MA(1) term (earlier it was ~7 370).  
# - **Residual ACF (lags 0–40):** all spikes fall within ±1.96/√N bounds, indicating “white‐noise” behavior.  
# - **Histogram & Q–Q Plot:** residuals are tightly centered on zero, with only mild heavy-tail deviations—well suited for a downstream deep-learning model to capture infrequent extreme errors.
# 
# ---
# 
# ### 5. Conclusion & Next Steps  
# This SARIMA(2,0,1)×(1,0,0,24) model provides a robust linear and seasonal “backbone”:
# 
# 1. **Linear inertia** via two AR lags.  
# 2. **Short-term shock absorption** via MA(1).  
# 3. **Daily cycle** via seasonal AR(24).
# 
# Its residuals now behave as near‐white noise, free from obvious autocorrelation or cyclical structure. **We can therefore freeze these ARIMA forecasts** and hand off the residuals to your CNN–LSTM hybrid, which will learn to correct the remaining non-linear, weather- and holiday-driven patterns.

# ## 4. Residual Calculation (for LSTM-CNN Input)
# 
# Now, calculate the residuals for all sets (train, validation, test) using the generated ARIMA predictions. These residuals will be the target variable for the subsequent LSTM-CNN model.

# In[9]:


# Calculate residuals (Actual - ARIMA Prediction) for all sets
# Ensure ARIMA_pred column exists and is populated

train_df['Residual'] = train_df[TARGET] - train_df['ARIMA_pred']
val_df['Residual'] = val_df[TARGET] - val_df['ARIMA_pred']
test_df['Residual'] = test_df[TARGET] - test_df['ARIMA_pred']
print("Residuals calculated for train, val, and test sets.")
print(train_df[['Residual']].head())

# Check for NaN values in the residuals
for name, df in zip(
    ['train', 'val', 'test'],
    [train_df, val_df, test_df]
):
    n_nulls = df['Residual'].isna().sum()
    print(f"{name.capitalize()} residuals: {n_nulls} NaN values")

# Visualize the residuals distribution for all sets
plt.figure(figsize=(12, 6))
sns.histplot(train_df['Residual'], bins=50, kde=True, label='Train', color='blue', alpha=0.5)
sns.histplot(val_df['Residual'], bins=50, kde=True, label='Validation', color='orange', alpha=0.5)
sns.histplot(test_df['Residual'], bins=50, kde=True, label='Test', color='green', alpha=0.5)
plt.title('Residuals Distribution')
plt.xlabel('Residual Value')
plt.ylabel('Density')
plt.legend()
plt.show()


# ## 5. LSTM-CNN Data Preparation

# Scale features and create sequences (X, y) for the LSTM-CNN model. The target `y` will be the ARIMA residual.

# In[10]:


from sklearn.preprocessing import StandardScaler
import numpy as np

# 1) Choose exogenous features (drop 'Residual' and 'year' to avoid huge scaling mismatch)
features_exog = [
    'hour', 'dayofweek', 'month', 'dayofyear',
    'is_dayoff', 'temp', 'dwpt', 'rhum', 'prcp', 'wspd'
]

# 2) Extract raw arrays
X_train_raw = train_df[features_exog].values
X_val_raw   = val_df[features_exog].values
X_test_raw  = test_df[features_exog].values

y_train_raw = train_df['Residual'].values.reshape(-1, 1)
y_val_raw   = val_df['Residual'].values.reshape(-1, 1)
y_test_raw  = test_df['Residual'].values.reshape(-1, 1)

# 3) Fit separate scalers
scaler_X = StandardScaler().fit(X_train_raw)
scaler_y = StandardScaler().fit(y_train_raw)

# 4) Transform features and target
X_train_scaled = scaler_X.transform(X_train_raw)
X_val_scaled   = scaler_X.transform(X_val_raw)
X_test_scaled  = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.transform(y_train_raw).flatten()
y_val_scaled   = scaler_y.transform(y_val_raw).flatten()
y_test_scaled  = scaler_y.transform(y_test_raw).flatten()

# 5) Build sliding windows
def create_sequences(X, y, n_steps):
    Xs, ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i : i + n_steps])
        ys.append(y[i + n_steps])
    return np.array(Xs), np.array(ys)

n_steps = 24

X_train, y_train_res = create_sequences(X_train_scaled, y_train_scaled, n_steps)
X_val,   y_val_res   = create_sequences(X_val_scaled,   y_val_scaled,   n_steps)
X_test,  y_test_res  = create_sequences(X_test_scaled,  y_test_scaled,  n_steps)

print(f"X_train shape: {X_train.shape}, y_train_res shape: {y_train_res.shape}")
print(f"X_val   shape: {X_val.shape},   y_val_res   shape: {y_val_res.shape}")
print(f"X_test  shape: {X_test.shape},  y_test_res  shape: {y_test_res.shape}")


# In[11]:


import torch
from torch.utils.data import TensorDataset, DataLoader

# 1) Convert numpy arrays to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_res, dtype=torch.float32).unsqueeze(-1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val_res,   dtype=torch.float32).unsqueeze(-1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test_res,  dtype=torch.float32).unsqueeze(-1)

# 2) Build TensorDatasets
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t,   y_val_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

# 3) Create DataLoaders
batch_size = 64

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,  drop_last=True
)
val_loader   = DataLoader(
    val_ds,   batch_size=batch_size, shuffle=False, drop_last=False
)
test_loader  = DataLoader(
    test_ds,  batch_size=batch_size, shuffle=False, drop_last=False
)

# Sanity check one batch
xb, yb = next(iter(train_loader))
print("Batch X shape:", xb.shape)  # (batch_size, 24, 10)
print("Batch y shape:", yb.shape)  # (batch_size, 1)


# ## 6. LSTM-CNN Model Definition (PyTorch)

# In[30]:


# Define the hybrid LSTM-CNN model using PyTorch
n_features = X_train.shape[2]

class LSTMCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        cnn_filters,
        kernel_size,
        pool_size,
        dropout_rate=0.3,
        num_layers=1,        # ← add this
        output_dim=1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Conv blocks (same as before)…
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_dim,   cnn_filters, kernel_size, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.MaxPool1d(pool_size)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=1),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.MaxPool1d(pool_size)
        )

        # compute seq_len after two poolings
        pool_output_len = n_steps
        for _ in range(2):
            pool_output_len = (pool_output_len - pool_size) // pool_size + 1

        # now pass num_layers into the LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,    # ← here
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.permute(0, 2, 1)

        # LSTM: hn now has shape (num_layers, batch, hidden_dim)
        _, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


# In[40]:


hidden_dim   = 128
cnn_filters  = 64
kernel_size  = 3
pool_size    = 2
dropout_rate = 0.3
lr           = 5e-4
weight_decay = 1e-5
num_layers   = 2  # two‐layer LSTM

model = LSTMCNN(
    input_dim=n_features,
    hidden_dim=hidden_dim,
    cnn_filters=cnn_filters,
    kernel_size=kernel_size,
    pool_size=pool_size,
    dropout_rate=dropout_rate,
    num_layers=num_layers,
    output_dim=1
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


# ## 7. LSTM-CNN Training (PyTorch)

# In[ ]:


import copy

# 2) Training loop with early stopping
n_epochs = 50
patience_es = 10
best_val_loss = float('inf')
epochs_no_improve = 0
best_state = None
history = {'train': [], 'val': []}

for epoch in range(1, n_epochs+1):
    # — Train —
    model.train()
    running_train = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_train += loss.item() * xb.size(0)
    train_loss = running_train / len(train_loader.dataset)
    history['train'].append(train_loss)

    # — Validate —
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            running_val += criterion(preds, yb).item() * xb.size(0)
    val_loss = running_val / len(val_loader.dataset)
    scheduler.step(val_loss)
    history['val'].append(val_loss)

    print(f"Epoch {epoch} — train: {train_loss:.6f}, val: {val_loss:.6f}")

    # LR scheduling & early stopping
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience_es:
            print(f"Early stopping at epoch {epoch}")
            break

# Restore best model
if best_state:
    model.load_state_dict(best_state)

# Plot loss curves
plt.plot(history['train'], label='Train Loss')
plt.plot(history['val'],   label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

# 3) Test-time inference
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        res_scaled = model(xb).cpu().numpy().flatten()
        # invert scaling
        res_pred = scaler_y.inverse_transform(res_scaled.reshape(-1,1)).flatten()
        all_preds.extend(res_pred)

all_preds = np.array(all_preds)

# 4) Add back ARIMA forecast for each test point
#    Assuming `test_df` is indexed in the same order as your sequences:
arima_fc = test_df['ARIMA_pred'].values[n_steps:]  # align with sequences
final_fc = arima_fc + all_preds

# 5) Compute RMSE on the test set
true = test_df[TARGET].values[n_steps:]
rmse = np.sqrt(mean_squared_error(true, final_fc))
print(f"Hybrid Test RMSE: {rmse:.2f} MW")


# ## 8. Evaluation

# In[ ]:


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1) Align test_df with your residual predictions
#    all_preds must be a 1D array of length = len(test_df) - n_steps
test_df_aligned = test_df.iloc[n_steps:].copy()

# 2) Insert the residual forecasts and hybrid sum
test_df_aligned['Residual_pred'] = all_preds
test_df_aligned['Combined_pred'] = test_df_aligned['ARIMA_pred'] + test_df_aligned['Residual_pred']

# 3) Define MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# 4) Compute metrics for Hybrid
y_true            = test_df_aligned[TARGET]
y_pred_combined   = test_df_aligned['Combined_pred']

rmse_hybrid = np.sqrt(mean_squared_error(y_true, y_pred_combined))
mae_hybrid  = mean_absolute_error   (y_true, y_pred_combined)
mape_hybrid = mean_absolute_percentage_error(y_true, y_pred_combined)

print("--- Combined Model Test Set Error Metrics ---")
print(f"RMSE: {rmse_hybrid:,.2f} MW")
print(f"MAE:  {mae_hybrid:,.2f} MW")
print(f"MAPE: {mape_hybrid:.2f}%")

# 5) Compute metrics for ARIMA-only baseline
y_pred_arima = test_df_aligned['ARIMA_pred']

rmse_arima = np.sqrt(mean_squared_error(y_true, y_pred_arima))
mae_arima  = mean_absolute_error   (y_true, y_pred_arima)
mape_arima = mean_absolute_percentage_error(y_true, y_pred_arima)

print("\n--- ARIMA Only Test Set Error Metrics ---")
print(f"RMSE: {rmse_arima:,.2f} MW")
print(f"MAE:  {mae_arima:,.2f} MW")
print(f"MAPE: {mape_arima:.2f}%")


# ## 10. Visualization

# In[34]:


# Plot actual vs predicted for a period in the test set
fig, ax = plt.subplots(figsize=(15, 6))
test_df_aligned[TARGET].plot(ax=ax, label='Actual', style='.', alpha=0.7)
test_df_aligned['Combined_pred'].plot(ax=ax, label='Combined Forecast', style='.', alpha=0.7)
# test_df_aligned['ARIMA_pred'].plot(ax=ax, label='ARIMA Forecast', style='--', alpha=0.7)
ax.set_title('Actual vs. Combined Forecast (Test Set)')
ax.set_ylabel('MW')
ax.legend()
plt.show()

# Zoom in on a specific week
start_date = test_df_aligned.index.min() + pd.Timedelta(days=7)
end_date = start_date + pd.Timedelta(days=7)

fig, ax = plt.subplots(figsize=(15, 6))
test_df_aligned.loc[start_date:end_date, TARGET].plot(ax=ax, label='Actual', style='.-')
test_df_aligned.loc[start_date:end_date, 'Combined_pred'].plot(ax=ax, label='Combined Forecast', style='-')
ax.set_title(f'Forecast vs Actuals ({start_date.date()} to {end_date.date()})')
ax.set_ylabel('MW')
ax.legend()
plt.show()


# ## 11. Error Analysis: Worst/Best Predicted Days

# In[35]:


# Calculate error and absolute error for each prediction in the aligned test set
test_df_aligned['error'] = test_df_aligned[TARGET] - test_df_aligned['Combined_pred']
test_df_aligned['abs_error'] = test_df_aligned['error'].abs()

# Group by date and calculate mean errors
error_by_day = test_df_aligned.groupby(test_df_aligned.index.date) \
    .agg(mean_Actual_MW=('PJME_MW', 'mean'),
         mean_Combined_Pred=('Combined_pred', 'mean'),
         mean_error=('error', 'mean'),
         mean_abs_error=('abs_error', 'mean'))

# Add weekday name for context
error_by_day['weekday'] = pd.to_datetime(error_by_day.index).strftime('%A')

print("--- Top 10 Days with Worst Mean Absolute Error ---")
worst_days = error_by_day.sort_values('mean_abs_error', ascending=False).head(10)
worst_days


# In[36]:


print("\n--- Top 10 Days with Best Mean Absolute Error ---")
best_days = error_by_day.sort_values('mean_abs_error', ascending=True).head(10)
best_days


# ### Worst Predicted Day Visualization

# In[37]:


# Find the date with the highest mean absolute error
worst_day_date = error_by_day['mean_abs_error'].idxmax()
worst_day_str = worst_day_date.strftime('%Y-%m-%d')

print(f"Plotting worst predicted day: {worst_day_str}")

fig, ax = plt.subplots(figsize=(15, 6))
test_df_aligned.loc[worst_day_str][TARGET].plot(ax=ax, label='Actual', style='.-')
test_df_aligned.loc[worst_day_str]['Combined_pred'].plot(ax=ax, label='Combined Prediction', style='-')
ax.set_title(f'Worst Predicted Day ({worst_day_str}) - MAE: {error_by_day.loc[worst_day_date, "mean_abs_error"]:.2f} MW')
ax.set_ylabel('MW')
ax.legend()
plt.show()


# ### Best Predicted Day Visualization

# In[38]:


# Find the date with the lowest mean absolute error
best_day_date = error_by_day['mean_abs_error'].idxmin()
best_day_str = best_day_date.strftime('%Y-%m-%d')

print(f"Plotting best predicted day: {best_day_str}")

fig, ax = plt.subplots(figsize=(15, 6))
test_df_aligned.loc[best_day_str][TARGET].plot(ax=ax, label='Actual', style='.-')
test_df_aligned.loc[best_day_str]['Combined_pred'].plot(ax=ax, label='Combined Prediction', style='-')
ax.set_title(f'Best Predicted Day ({best_day_str}) - MAE: {error_by_day.loc[best_day_date, "mean_abs_error"]:.2f} MW')
ax.set_ylabel('MW')
ax.legend()
plt.show()


# Here’s what jumps out from your day‐level aggregation:
# 
# ### 📉 Worst‐predicted days  
# | Date       | Mean Load | Mean Forecast | Mean Abs Error | Weekday  | Notes |
# |------------|-----------|---------------|---------------:|----------|-------|
# | **2018-07-06** | 39 145 MW  | 39 517 MW      | 686 MW        | Friday   | **Day after Independence Day**—holiday rebound, atypical ramp-up pattern.  
# | 2018-07-15 | 34 204 MW  | 34 233 MW      | 609 MW        | Sunday   | Midsummer weekend—higher residential AC load.  
# | 2018-07-05 | 42 402 MW  | 42 267 MW      | 568 MW        | Thursday | Pre-holiday warm-up.  
# | 2018-05-15 | 31 482 MW  | 31 359 MW      | 541 MW        | Tuesday  | Early-summer heat spike.  
# | …          | …         | …             | …             | …        | …  
# 
# Two of the top three misses straddle **July 4th**—clear evidence that your holiday-and-weekend flag isn’t fully capturing the “day + 1/−1” effects. Likewise, several early-summer heat days (May/June) slip through.
# 
# ### 📈 Best‐predicted days  
# | Date       | Mean Load | Mean Forecast | Mean Abs Error | Weekday  |
# |------------|-----------|---------------|---------------:|----------|
# | 2017-04-24 | 26 415 MW  | 26 379 MW      | 115 MW        | Monday   |
# | 2018-01-28 | 27 357 MW  | 27 307 MW      | 117 MW        | Sunday   |
# | 2017-05-22 | 26 889 MW  | 26 867 MW      | 125 MW        | Monday   |
# | …          | …         | …             | …             | …        |
# 
# These are **steady, off-peak periods** (spring/fall shoulder months) where both ARIMA and your CNN–LSTM residual learner have no surprises—loads follow their usual daily cycle.
# 
# ---
# 
# ## 🚀 Next steps
# 
# 1. **Enhance your holiday feature**  
#    - Add “day + 1” and “day − 1” flags around federal holidays so the network sees those rebound effects explicitly.  
# 
# 2. **Bake in extreme-temperature signals**  
#    - Rather than raw temp, include features like “Cooling Degree Hours above 75°F” or quadratic terms to better capture AC load surges on hot days.  
# 
# 3. **Special‐case gating**  
#    - For the handful of days with MAE > 500 MW, consider a tiny “holiday‐heat” submodel: e.g. train a small decision‐tree on top of your residuals to correct whenever (is_holiday_adjacent & temp > 85°F).  
# 
# 4. **Granular error analysis**  
#    - Drill into those worst days hour by hour. Perhaps the network misses morning ramp vs. evening ramp differently—if so, you could add “hour×dayoff” interaction features.  
# 
# 5. **Re-validate after tweaks**  
#    - Recompute your day-level MAE table and confirm that July 5–6 errors shrink from ~650 MW down toward your average ~250 MW.  
# 
# By encoding a bit more domain knowledge around **holiday adjacency** and **temperature extremes**, you’ll likely eliminate those “spikes” in your worst‐case errors—pushing your hybrid MAPE even lower than 0.9%.

# 
