#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TimeXer Time-Series Forecasting Script

This script refactors the PJM Hourly Load Forecasting notebook into a
command-line executable Python script. It implements the TimeXer model
for short-term forecasting of PJME grid load, incorporating weather features.

Workflow:
1. Setup environment, logging, and configurations.
2. Load and preprocess data (energy + weather), including feature engineering.
3. Split data into training, validation, and test sets.
4. Scale features and target variable.
5. Define TimeXer model architecture components.
6. Define PyTorch Dataset and DataLoaders.
7. Train the model with early stopping (initial training or final retraining).
8. Evaluate the model using various metrics and diagnostic plots.
9. (Optional) Perform hyperparameter tuning using Optuna.
10. Save evaluation results, plots, best parameters, and the final model.
"""

# --- Standard Libraries ---
import os
import time
import copy
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# --- Data Manipulation ---
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Machine Learning ---
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib  # For saving scalers

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Weather Data ---
# Ensure meteostat is installed: pip install meteostat
try:
    from meteostat import Stations, Hourly
except ImportError:
    print("Meteostat library not found. Weather data fetching will be skipped.")
    print("To include weather data, please install it: pip install meteostat")
    Stations = None  # Set to None to indicate unavailability
    Hourly = None
    # Handle this potentially missing dependency later in the data loading phase

# --- Hyperparameter Tuning (Optional) ---
try:
    import optuna
except ImportError:
    print("Optuna library not found. Hyperparameter tuning will be skipped.")
    print("To enable tuning, please install it: pip install optuna")
    optuna = None # Set to None to indicate unavailability

# === Configuration & Setup ===

# --- Script Constants ---
SCRIPT_NAME = "timexer"
BASE_DIR = Path(__file__).resolve().parent.parent # Assumes script is in a 'src' or similar subdir
LOG_DIR = BASE_DIR / "log"
OUTPUT_DIR = BASE_DIR / f"{SCRIPT_NAME}_files"
DATA_DIR = BASE_DIR / "data" / "3" # Adjust if data path is different
PJME_CSV_FILENAME = "PJME_hourly.csv"
BEST_PARAMS_FILENAME = "best_hyperparams.json"
BEST_MODEL_FILENAME = "best_model.pt"
FEATURE_SCALER_FILENAME = "feature_scaler.joblib"
TARGET_SCALER_FILENAME = "target_scaler.joblib"

# --- Create Output Directories ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Setup Logging ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"{SCRIPT_NAME}_{TIMESTAMP}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() # Also print logs to console
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Starting {SCRIPT_NAME} script run.")
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# --- Configure Environment ---
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight') # Consistent plotting style

# --- Check for GPU Availability ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU.")

# === Helper Functions & Classes ===

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_mask = y_true != 0
    # Handle cases where all true values are zero
    if not np.any(non_zero_mask):
        logger.warning("MAPE calculation encountered all zero true values. Returning 0.0.")
        return 0.0
    # Calculate MAPE only for non-zero true values
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    # Handle potential NaNs or Infs resulting from edge cases (though ideally prevented)
    if np.isnan(mape) or np.isinf(mape):
        logger.warning(f"MAPE calculation resulted in NaN or Inf. Returning large value (1e9). Check inputs.")
        return 1e9 # Return a large number to indicate failure if needed
    return mape

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a
    given patience. Saves the best model checkpoint.
    """
    def __init__(self, patience=7, verbose=True, delta=0,
                 path='checkpoint.pt', trace_func=logger.info):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str or Path): Path for the checkpoint to be saved to.
            trace_func (function): Function to use for logging messages.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = Path(path) # Ensure path is a Path object
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        """Call method to update early stopping status."""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0 # Reset counter

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        try:
            # Ensure the directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
        except Exception as e:
            self.trace_func(f"Error saving checkpoint to {self.path}: {e}")


class Configs:
    """Simple class to hold configuration parameters."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)


# === Model Building Blocks ===
# (Adapted from the notebook, with type hints and minor style adjustments)

class PositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
             raise ValueError(f"d_model must be even for PositionalEmbedding, got {d_model}")
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Simplified div_term calculation
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Keep pe shape as [max_len, d_model] for easier slicing
        self.register_buffer('pe', pe, persistent=False) # Not part of state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor to determine sequence length (e.g., [B, L, D]).

        Returns:
            torch.Tensor: Positional encoding slice [seq_len, d_model].
        """
        # x is used to determine seq_len (dim 1)
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum positional embedding length {self.pe.size(0)}")
        # Return the slice corresponding to the sequence length
        # The calling function needs to handle batch expansion if necessary
        return self.pe[:seq_len, :]


class TimeFeatureEmbedding(nn.Module):
    """Embeds time features derived from timestamps."""
    def __init__(self, d_model: int, freq: str = 'h'):
        super().__init__()
        # Determine input dimension based on frequency and features created in dataset
        # Assuming features for 'h': hour, dayofweek, day, dayofyear, month, weekofyear
        if freq == 'h':
            d_inp = 6
        # Add other frequencies if needed ('t' for minutely, 'd' for daily, etc.)
        # elif freq == 'd': d_inp = ...
        else:
            logger.warning(f"Frequency '{freq}' not explicitly handled in TimeFeatureEmbedding. Using default d_inp=2.")
            d_inp = 2 # Default fallback, adjust as needed
        self.embed = nn.Linear(d_inp, d_model, bias=False)
        logger.debug(f"TimeFeatureEmbedding initialized with d_inp={d_inp}, d_model={d_model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Time features tensor [Batch, SeqLen, NumTimeFeatures].

        Returns:
            torch.Tensor: Embedded time features [Batch, SeqLen, d_model].
        """
        return self.embed(x)


class TokenEmbedding(nn.Module):
    """Embeds input feature values using a linear layer."""
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        # Using Linear layer is common and simpler than Conv1d for basic token embedding
        self.value_embedding = nn.Linear(c_in, d_model, bias=False)
        logger.debug(f"TokenEmbedding initialized with c_in={c_in}, d_model={d_model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features tensor [Batch, SeqLen, Features].

        Returns:
            torch.Tensor: Embedded features [Batch, SeqLen, d_model].
        """
        return self.value_embedding(x)


class DataEmbeddingExog(nn.Module):
    """
    Combined embedding for exogenous variables, incorporating value, temporal,
    and positional information.
    """
    def __init__(self, c_in_exog: int, num_time_features: int, d_model: int,
                 freq: str = 'h', dropout: float = 0.1):
        super().__init__()
        logger.debug(f"Initializing DataEmbeddingExog: c_in_exog={c_in_exog}, "
                     f"num_time_features={num_time_features}, d_model={d_model}, freq='{freq}'")

        self.value_embedding = TokenEmbedding(c_in=c_in_exog, d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        # Positional embedding (sinusoidal) - Added here for consistency
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_exog: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_exog (torch.Tensor): Exogenous variable values [Batch, SeqLen, NumExogVars (c_in_exog)].
            x_mark (torch.Tensor): Time features [Batch, SeqLen, NumTimeFeatures].

        Returns:
            torch.Tensor: Combined embeddings [Batch, SeqLen, d_model].
        """
        if x_exog.size(1) != x_mark.size(1):
             raise ValueError(f"Sequence length mismatch between x_exog ({x_exog.size(1)}) and x_mark ({x_mark.size(1)})")
        if x_mark.size(2) != self.temporal_embedding.embed.in_features:
             raise ValueError(f"Number of time features in x_mark ({x_mark.size(2)}) does not match "
                              f"TimeFeatureEmbedding input dim ({self.temporal_embedding.embed.in_features})")

        # Get positional encoding slice [SeqLen, d_model]
        pos_enc_slice = self.position_embedding(x_exog) # Pass x_exog to get seq_len

        # Apply embeddings and sum them
        value_embed = self.value_embedding(x_exog)
        temporal_embed = self.temporal_embedding(x_mark)

        # Expand pos_enc_slice to [Batch, SeqLen, d_model] for addition
        # Use .unsqueeze(0) which is equivalent to None indexing for the first dim
        pos_embed_expanded = pos_enc_slice.unsqueeze(0).expand(x_exog.size(0), -1, -1)

        # Combine embeddings
        combined_embedding = value_embed + temporal_embed + pos_embed_expanded

        return self.dropout(combined_embedding)


# --- Standard Attention Layers (as provided in notebook) ---

class FullAttention(nn.Module):
    """Standard multi-head self-attention mechanism."""
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor | None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape # Note: keys shape is [B, S, H, E], values is [B, S, H, D]
        scale = self.scale or 1. / np.sqrt(E)

        # Calculate scores: [B, L, H, E] @ [B, S, H, E].transpose(-1,-2) -> [B, L, H, S]? No, einsum is better
        # Einsum: batch, query_len, heads, embed_dim ; batch, key_len, heads, embed_dim -> batch, heads, query_len, key_len
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is not None:
            # Ensure attn_mask is broadcastable to [B, H, L, S]
            if attn_mask.dim() == 2: # [L, S] -> [1, 1, L, S]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3: # [B, L, S] -> [B, 1, L, S]
                attn_mask = attn_mask.unsqueeze(1)
            # Expand mask to match heads dimension if necessary
            if attn_mask.shape[1] != H:
                attn_mask = attn_mask.expand(-1, H, -1, -1)

            # Fill positions with True (or 1) in the mask with -inf
            # Assuming attn_mask has 0 where attention is allowed, 1 where masked
            scores = scores.masked_fill(attn_mask.bool(), -float('inf')) # Use bool() for clarity

        # Calculate attention weights
        attn_weights = self.dropout(torch.softmax(scores * scale, dim=-1)) # Softmax over key sequence length (S)

        # Apply attention to values
        # Einsum: batch, heads, query_len, key_len ; batch, key_len, heads, value_dim -> batch, query_len, heads, value_dim
        context = torch.einsum("bhls,bshd->blhd", attn_weights, values).contiguous() # Use 'd' for value_dim

        # Reshape back to [B, L, H*D] = [B, L, d_model] if D = d_model // H
        context = context.view(B, L, -1)

        if self.output_attention:
            return context, attn_weights # Return computed context and attention weights
        else:
            return context, None # Return computed context only


class AttentionLayer(nn.Module):
    """Combines MultiHeadAttention with Residual connection and LayerNorm."""
    def __init__(self, attention_module: FullAttention, d_model: int, n_heads: int,
                 d_keys: int | None = None, d_values: int | None = None):
        super().__init__()
        if d_model % n_heads != 0:
            logger.warning(f"d_model ({d_model}) is not divisible by n_heads ({n_heads}). "
                           "This might lead to uneven head dimensions.")

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads) # Usually d_keys == d_values

        self.inner_attention = attention_module
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        # Output projection to combine heads
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: torch.Tensor | None, tau=None, delta=None):
        # Input shapes: queries [B, L, D], keys [B, S, D], values [B, S, D]
        # D = d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 1. Project queries, keys, values
        # Result shape: [B, SeqLen, H, HeadDim]
        q_proj = self.query_projection(queries).view(B, L, H, self.d_keys)
        k_proj = self.key_projection(keys).view(B, S, H, self.d_keys)
        v_proj = self.value_projection(values).view(B, S, H, self.d_values)

        # 2. Apply inner attention mechanism
        # The inner_attention expects inputs like [B, L, H, E] or [B, S, H, E/D]
        # Our FullAttention takes: [B, L, H, E], [B, S, H, E], [B, S, H, D]
        out, attn = self.inner_attention(
            q_proj, k_proj, v_proj,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        # Output 'out' from FullAttention is already reshaped to [B, L, d_model]

        # 3. Apply final output projection (potentially redundant if inner_attention already does it)
        # The provided FullAttention reshapes V to [B, L, -1] before returning.
        # This output projection is standard in transformer blocks.
        out = self.out_projection(out) # Shape [B, L, d_model]

        return out, attn # Return context and attention weights


# === TimeXer Core Model Components ===

class FlattenHead(nn.Module):
    """Flattens encoder output and projects to the prediction horizon."""
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.target_window = target_window
        # Flattens the last two dimensions (usually d_model and patch_num)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        logger.debug(f"FlattenHead initialized: n_vars={n_vars}, nf={nf}, "
                     f"target_window={target_window}, dropout={head_dropout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, e.g., [bs, n_vars, d_model, patch_num+1].

        Returns:
            torch.Tensor: Projected output [bs, n_vars, target_window].
        """
        # Input shape assumed: [bs, nvars, d_model, patch_num] or similar
        # Let's assume input x is [bs, nvars, d_model, N+1] (N=patch_num)
        # Flatten transforms [bs, nvars, d_model, N+1] -> [bs, nvars, d_model * (N+1)]
        x_flat = self.flatten(x)

        # Apply linear layer
        # Input shape: [bs, nvars, nf (=d_model * (N+1))]
        # Output shape: [bs, nvars, target_window]
        x_projected = self.linear(x_flat)
        x_out = self.dropout(x_projected)
        return x_out


# (Continuing from the previous 500 lines)

class EnEmbedding(nn.Module):
    """
    Embedding for the Endogenous variable(s).
    Applies patching, value embedding, positional embedding, and adds a global token.
    """
    def __init__(self, n_vars: int, d_model: int, patch_len: int, dropout: float):
        super().__init__()
        if patch_len <= 0:
            raise ValueError("patch_len must be positive.")
        self.patch_len = patch_len
        self.n_vars = n_vars
        self.d_model = d_model

        logger.debug(f"Initializing EnEmbedding: n_vars={n_vars}, d_model={d_model}, "
                     f"patch_len={patch_len}, dropout={dropout}")

        # Linear layer to embed each patch
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Learnable global token (one per endogenous variable)
        # Shape: [1, n_vars, 1, d_model] -> allows broadcasting/repeating later
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        # Positional embedding for patches (and potentially the global token)
        # max_len should accommodate max possible number of patches + 1 (for glb_token)
        # Calculate max_len needed based on typical sequence lengths if possible, or use default.
        self.position_embedding = PositionalEmbedding(d_model) # Uses default max_len=5000
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Args:
            x (torch.Tensor): Input tensor for endogenous variables [Batch, SeqLen, n_vars_in].
                              Expected n_vars_in to match self.n_vars.

        Returns:
            tuple[torch.Tensor, int]:
                - Embedded tokens (patches + global) reshaped for Transformer input
                  [Batch * n_vars, NumPatches + 1, d_model].
                - Number of endogenous variables processed (self.n_vars).
        """
        batch_size, seq_len, n_vars_in = x.shape
        if n_vars_in != self.n_vars:
             raise ValueError(f"Input n_vars ({n_vars_in}) doesn't match model EnEmbedding n_vars ({self.n_vars})")
        if seq_len % self.patch_len != 0:
            # If seq_len is not divisible by patch_len, padding might be needed beforehand,
            # or adjust the logic here (e.g., truncate last partial patch).
            # For simplicity, assuming seq_len is divisible or handled upstream.
            # Alternatively, use ceil for num_patches and pad the last patch.
            padding_size = self.patch_len - (seq_len % self.patch_len)
            x_padded = F.pad(x, (0, 0, 0, padding_size), mode='replicate') # Pad sequence dim
            seq_len = x_padded.shape[1] # Update seq_len
            x = x_padded
            logger.debug(f"Padded input sequence to length {seq_len} for patching.")


        # Transpose to [Batch, n_vars, SeqLen] for patching
        x = x.permute(0, 2, 1)

        # Patching using unfold
        # unfold(dimension, size, step)
        x_unfolded = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # Shape after unfold: [Batch, n_vars, NumPatches, patch_len]
        num_patches = x_unfolded.shape[2]

        # Reshape for linear embedding: [Batch * n_vars * NumPatches, patch_len]
        x_reshaped = x_unfolded.permute(0, 1, 2, 3).reshape(-1, self.patch_len)

        # Apply value embedding
        # Shape: [Batch * n_vars * NumPatches, d_model]
        x_embedded = self.value_embedding(x_reshaped)

        # Reshape back to [Batch, n_vars, NumPatches, d_model]
        x_embedded = x_embedded.reshape(batch_size, self.n_vars, num_patches, self.d_model)

        # --- Get and Add Positional Embedding ---
        # Get positional encoding slice [NumPatches, d_model]
        # Pass a dummy tensor with correct seq_len (num_patches) to positional embedding
        dummy_pos_input = torch.zeros(1, num_patches, self.d_model, device=x.device)
        pos_encoding_slice = self.position_embedding(dummy_pos_input) # Shape: [NumPatches, d_model]

        # Expand slice and add to x_embedded
        # Expand to [Batch, n_vars, NumPatches, d_model]
        pos_embed_expanded = pos_encoding_slice.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_vars, -1, -1)
        x_patched_embedded = x_embedded + pos_embed_expanded # Add positional embedding

        # Prepare global token: repeat for batch and concatenate
        # glb_token shape: [1, n_vars, 1, d_model]
        glb_repeated = self.glb_token.repeat(batch_size, 1, 1, 1) # Shape: [Batch, n_vars, 1, d_model]

        # Concatenate global token with patch tokens along the sequence (num_patches) dimension
        # Shape: [Batch, n_vars, NumPatches + 1, d_model]
        x_with_glb = torch.cat([x_patched_embedded, glb_repeated], dim=2)

        # Reshape for Transformer layers: [Batch * n_vars, NumPatches + 1, d_model]
        x_final = x_with_glb.reshape(batch_size * self.n_vars, num_patches + 1, self.d_model)

        return self.dropout(x_final), self.n_vars


class EncoderLayer(nn.Module):
    """
    Single layer of the TimeXer Encoder, including self-attention over endogenous
    tokens and cross-attention between the endogenous global token and exogenous tokens.
    """
    def __init__(self, self_attention: AttentionLayer, cross_attention: AttentionLayer,
                 d_model: int, n_vars_en: int, c_exog: int, d_ff: int | None = None,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model # Default feed-forward dimension
        self.n_vars_en = n_vars_en # Number of endogenous variables
        self.c_exog = c_exog       # Number of exogenous variables/tokens expected

        logger.debug(f"Initializing EncoderLayer: d_model={d_model}, n_vars_en={n_vars_en}, "
                     f"c_exog={c_exog}, d_ff={d_ff}, activation='{activation}'")

        self.self_attention = self_attention # Attends over endogenous patches + global token
        self.cross_attention = cross_attention # Attends global token (Q) to exogenous variate tokens (K, V)

        # Feed-Forward Network (using 1D convolutions as in original Transformer)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model) # After self-attention
        self.norm2 = nn.LayerNorm(d_model) # After cross-attention (applied only to global token)
        self.norm3 = nn.LayerNorm(d_model) # After feed-forward

        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x: torch.Tensor, cross: torch.Tensor | None,
                x_mask: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None,
                tau=None, delta=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Endogenous tokens (Patch + Global) [Batch * n_vars_en, N+1, D].
            cross (torch.Tensor | None): Exogenous tokens [Batch, C, D] or [Batch, SeqLen_exog, D].
                                        None if no exogenous features. C=num_exog_tokens.
            x_mask (torch.Tensor | None): Mask for self-attention.
            cross_mask (torch.Tensor | None): Mask for cross-attention.
            tau, delta: Optional parameters for attention variants (not used in standard attention).

        Returns:
            torch.Tensor: Output tensor [Batch * n_vars_en, N+1, D].
        """
        # --- 1. Self-Attention over Endogenous Tokens (Patches + Global) ---
        x_res = x # Store residual connection input
        x_attn, _ = self.self_attention(
            x, x, x, # Query, Key, Value are all from endogenous tokens
            attn_mask=x_mask,
            tau=tau, delta=None # Delta usually for causal attention, not needed here
        )
        x = x_res + self.dropout(x_attn) # Add residual
        x = self.norm1(x)
        # x shape: [B * n_vars_en, N+1, D]

        x_after_self_attn = x # Store for FFN residual connection

        # --- 2. Cross-Attention: Endogenous Global Token <-> Exogenous Tokens ---
        # Only perform cross-attention if exogenous tokens (`cross`) are provided
        if cross is not None and self.c_exog > 0:
            # Get current global token (output of norm1) from the *end* of the sequence
            x_glb_en = x[:, -1, :] # Shape: [B * n_vars_en, D]
            x_glb_en_query = x_glb_en.unsqueeze(1) # Shape: [B * n_vars_en, 1, D] - Query for cross-attn

            # Prepare exogenous 'cross' input (Keys and Values for cross-attn)
            # Expected cross shape: [B, C, D] or [B, L_exog, D]
            batch_size_x = x.shape[0]       # B * n_vars_en
            batch_size_cross = cross.shape[0] # B

            # Repeat 'cross' if needed to match the expanded batch dimension of 'x'
            if batch_size_x == batch_size_cross * self.n_vars_en:
                # Repeat each sample in 'cross' n_vars_en times
                cross_repeated = cross.repeat_interleave(self.n_vars_en, dim=0)
            elif batch_size_x == batch_size_cross:
                 # This case might happen if n_vars_en=1 or if dimensions already match
                 cross_repeated = cross
            else:
                raise ValueError(f"Batch size mismatch: x ({batch_size_x}) vs cross ({batch_size_cross}), "
                                 f"n_vars_en={self.n_vars_en}. Cannot align for cross-attention.")

            # Apply cross-attention: Query=GlobalEndogToken, Key=Value=ExogTokens
            x_glb_attn, _ = self.cross_attention(
                x_glb_en_query, cross_repeated, cross_repeated,
                attn_mask=cross_mask,
                tau=tau, delta=delta
            ) # Output shape: [B * n_vars_en, 1, D]

            # Add result back to original global token (residual connection for cross-attention)
            # Squeeze the sequence dimension (dim 1) from cross-attention output
            x_glb_updated = x_glb_en + self.dropout(x_glb_attn.squeeze(1))
            x_glb_updated = self.norm2(x_glb_updated) # Apply LayerNorm. Shape: [B * n_vars_en, D]

            # --- Combine Patches and Updated Global Token ---
            x_patches = x_after_self_attn[:, :-1, :] # Get patch tokens [B*n_vars_en, N, D]
            x_glb_updated_unsqueezed = x_glb_updated.unsqueeze(1) # Shape [B*n_vars_en, 1, D]
            # Create the new tensor 'x' for the FFN input
            x_ffn_input = torch.cat([x_patches, x_glb_updated_unsqueezed], dim=1) # Shape [B*n_vars_en, N+1, D]

        else:
            # If no cross-attention (no exogenous features), FFN input is just the output of self-attention + norm1
            x_ffn_input = x_after_self_attn
            if self.c_exog > 0:
                logger.warning("EncoderLayer has c_exog > 0 but 'cross' input was None. Skipping cross-attention.")


        # --- 3. Feed Forward Network ---
        # Apply FFN to the combined (or self-attended) tensor
        y = x_ffn_input # Residual connection for FFN starts here
        # Conv1d expects [Batch, Channels, Length] = [B*n_vars_en, D, N+1]
        y_ffn = self.conv1(y.transpose(-1, 1)) # Transpose: [B*n_vars_en, N+1, D] -> [B*n_vars_en, D, N+1]
        y_ffn = self.activation(y_ffn)
        y_ffn = self.dropout(y_ffn)
        y_ffn = self.conv2(y_ffn) # Output: [B*n_vars_en, D, N+1]
        y_ffn = y_ffn.transpose(-1, 1) # Transpose back: [B*n_vars_en, N+1, D]
        y_ffn = self.dropout(y_ffn)

        # Final residual connection and norm
        output = self.norm3(x_ffn_input + y_ffn) # Add residual from *before* FFN
        return output


class Encoder(nn.Module):
    """Stack of Encoder Layers."""
    def __init__(self, attn_layers: list[EncoderLayer], norm_layer: nn.Module | None = None):
        super().__init__()
        self.layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        logger.info(f"Initialized Encoder with {len(attn_layers)} layers.")

    def forward(self, x: torch.Tensor, cross: torch.Tensor | None,
                x_mask: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None,
                tau=None, delta=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Endogenous tokens [B*n_vars_en, N+1, D].
            cross (torch.Tensor | None): Exogenous tokens [B, C, D] or [B, L_exog, D].
            x_mask, cross_mask: Attention masks.
            tau, delta: Optional parameters for attention variants.

        Returns:
            torch.Tensor: Output tensor from the last layer [B*n_vars_en, N+1, D].
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        return x


# === TimeXer Full Model Definition ===

class Model(nn.Module):
    """
    TimeXer Model for Time Series Forecasting with Exogenous Variables.

    Combines endogenous embedding (patching + global token), exogenous embedding
    (value + temporal + positional), an encoder with cross-attention, and a
    prediction head.
    """
    def __init__(self, configs: Configs):
        super().__init__()
        self.configs = configs
        logger.info("Initializing TimeXer Model...")
        logger.info(f"Model Configs: {configs}")

        # --- Validate Configs ---
        if configs.d_model % configs.n_heads != 0:
            logger.warning(f"d_model ({configs.d_model}) not divisible by n_heads ({configs.n_heads}).")
        if configs.seq_len <= 0 or configs.pred_len <= 0 or configs.patch_len <= 0:
             raise ValueError("seq_len, pred_len, and patch_len must be positive.")

        # Store essential configs as attributes
        self.task_name = configs.task_name
        self.features = configs.features # 'M', 'S', or 'MS'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, 'use_norm', False) # Instance Norm flag
        self.patch_len = configs.patch_len
        # Calculate number of patches, potentially padding if seq_len % patch_len != 0
        self.patch_num = (self.seq_len + self.patch_len -1) // self.patch_len # Ceiling division
        self.padded_seq_len = self.patch_num * self.patch_len # Actual seq len after padding in EnEmbed
        if self.padded_seq_len != self.seq_len:
             logger.info(f"Sequence length {self.seq_len} is not divisible by patch length {self.patch_len}. "
                         f"Effective sequence length after padding will be {self.padded_seq_len}.")

        self.d_model = configs.d_model
        self.n_vars_en = configs.n_vars_en # Number of endogenous variables (target(s))
        self.c_exog = configs.c_exog       # Number of exogenous variables

        # --- 1. Embeddings ---
        # Endogenous Embedding (Patching + Global Token)
        self.en_embedding = EnEmbedding(
            n_vars=self.n_vars_en,
            d_model=configs.d_model,
            patch_len=self.patch_len,
            dropout=configs.dropout
        )

        # Exogenous Embedding (Handles time features and exogenous values)
        if self.c_exog > 0:
            # Determine number of time features based on frequency
            # TODO: This should ideally be passed via configs or derived more robustly
            if configs.freq == 'h':
                num_time_features = 6 # hour, dayofweek, day, dayofyear, month, weekofyear
            else:
                num_time_features = 2 # Default fallback
                logger.warning(f"Using default num_time_features={num_time_features} for freq='{configs.freq}'.")

            self.ex_embedding = DataEmbeddingExog(
                c_in_exog=self.c_exog,
                num_time_features=num_time_features,
                d_model=configs.d_model,
                freq=configs.freq,
                dropout=configs.dropout
            )
            logger.info(f"Initialized Exogenous Embedding for {self.c_exog} variables.")
        else:
            self.ex_embedding = None
            logger.info("No exogenous variables configured (c_exog=0). Skipping Exogenous Embedding.")


        # --- 2. Encoder ---
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    # Self-Attention for Endogenous patches+global
                    self_attention=AttentionLayer(
                        FullAttention(mask_flag=False, # No look-ahead mask needed for encoder self-attn
                                      factor=configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False), # Set True to visualize attention
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    # Cross-Attention for Endog Global <-> Exog Variate/Time Embeddings
                    cross_attention=AttentionLayer(
                        FullAttention(mask_flag=False, # No mask typically needed for cross-attn
                                      factor=configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    n_vars_en=self.n_vars_en,
                    c_exog=self.c_exog, # Inform layer about expected exog features
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers) # Stack multiple encoder layers
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model) if configs.e_layers > 0 else None
        )
        logger.info(f"Initialized Encoder with {configs.e_layers} layers.")

        # --- 3. Prediction Head ---
        # Number of features coming out of the encoder flatten operation
        # Each of the (patch_num + 1) tokens has d_model features
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(
            n_vars=self.n_vars_en,       # Predict for each endogenous variable
            nf=self.head_nf,            # Input features to the head's linear layer
            target_window=self.pred_len, # Output prediction length
            head_dropout=configs.dropout # Use same dropout rate or define separate head_dropout
        )
        logger.info(f"Initialized FlattenHead: n_vars={self.n_vars_en}, nf={self.head_nf}, target_window={self.pred_len}")


    def _apply_instance_norm(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies instance normalization to the target variable(s)."""
        # Assuming target variable(s) are the *last* self.n_vars_en columns
        target_data = x_enc[:, :, -self.n_vars_en:] # Shape [B, L, n_vars_en]

        means = target_data.mean(1, keepdim=True).detach() # Mean across seq_len [B, 1, n_vars_en]
        stdev = torch.sqrt(torch.var(target_data, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # [B, 1, n_vars_en]

        normalized_target = (target_data - means) / stdev

        # If needed, create the full normalized x_enc tensor
        if self.c_exog > 0:
            x_exog_part = x_enc[:, :, :-self.n_vars_en]
            x_enc_normalized = torch.cat((x_exog_part, normalized_target), dim=-1)
        else:
            x_enc_normalized = normalized_target

        # Squeeze means/stdev for easier denormalization later: [B, n_vars_en]
        means = means.squeeze(1)
        stdev = stdev.squeeze(1)

        return x_enc_normalized, means, stdev

    def _denormalize(self, dec_out: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
        """Applies inverse instance normalization."""
        # means/stdev shape: [B, n_vars_en]
        # dec_out shape: [B, pred_len, n_vars_en]
        # Unsqueeze means/stdev to broadcast: [B, 1, n_vars_en]
        means = means.unsqueeze(1)
        stdev = stdev.unsqueeze(1)
        dec_out = dec_out * stdev + means
        return dec_out

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                 x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> torch.Tensor:
        """
        Core forecasting logic using the encoder-only architecture.

        Args:
            x_enc (torch.Tensor): Input sequence data [Batch, SeqLen, NumFeatures (Exog+Endog)].
            x_mark_enc (torch.Tensor): Time features for input sequence [Batch, SeqLen, NumTimeFeatures].
            x_dec (torch.Tensor): Placeholder for decoder input (not used in this encoder-only model).
            x_mark_dec (torch.Tensor): Time features for prediction window (potentially used by exog embedding if needed).

        Returns:
            torch.Tensor: Forecasted sequence [Batch, PredLen, n_vars_en].
        """
        # --- Instance Normalization (Optional) ---
        if self.use_norm:
            x_enc_norm, means, stdev = self._apply_instance_norm(x_enc)
        else:
            x_enc_norm = x_enc
            means, stdev = None, None # Will not be used for denormalization

        # --- Prepare Endogenous and Exogenous Data ---
        # Assuming target(s) are the last n_vars_en columns
        x_endog = x_enc_norm[:, :, -self.n_vars_en:] # [B, L, n_vars_en]

        if self.c_exog > 0 and self.ex_embedding is not None:
            if x_enc.shape[2] != self.n_vars_en + self.c_exog:
                 raise ValueError(f"Input x_enc feature dim ({x_enc.shape[2]}) doesn't match "
                                  f"n_vars_en ({self.n_vars_en}) + c_exog ({self.c_exog}).")
            x_exog = x_enc_norm[:, :, :-self.n_vars_en] # [B, L, c_exog]
        else:
            x_exog = None

        # --- Embeddings ---
        # Endogenous Embedding (Patching + Global Token) -> [B * n_vars_en, N+1, D]
        # Note: EnEmbedding handles potential padding if seq_len % patch_len != 0
        en_embed, n_vars_processed = self.en_embedding(x_endog)
        if n_vars_processed != self.n_vars_en: # Sanity check
             logger.error(f"Mismatch: EnEmbedding processed {n_vars_processed} vars, expected {self.n_vars_en}")


        # Exogenous Embedding (Value + Time + Positional) -> [B, L_padded, D]
        if self.ex_embedding is not None and x_exog is not None:
            # Ensure exogenous inputs match the potentially padded sequence length
            if x_exog.size(1) != self.padded_seq_len:
                 # Pad exogenous features similarly to how endogenous might be padded
                 padding_size = self.padded_seq_len - x_exog.size(1)
                 if padding_size > 0 :
                      x_exog = F.pad(x_exog, (0, 0, 0, padding_size), mode='replicate')
                      x_mark_enc = F.pad(x_mark_enc, (0, 0, 0, padding_size), mode='replicate')
                 elif padding_size < 0: # Should not happen if padded_seq_len calculated correctly
                      x_exog = x_exog[:, :self.padded_seq_len, :]
                      x_mark_enc = x_mark_enc[:, :self.padded_seq_len, :]


            # Input shapes: x_exog [B, L_padded, C], x_mark_enc [B, L_padded, NumTimeFeatures]
            ex_embed = self.ex_embedding(x_exog, x_mark_enc) # [B, L_padded, D]
            # This ex_embed (time-step embeddings) will be used as K, V in cross-attention
            cross_input = ex_embed # Shape [B, L_padded, D]
        else:
            # Handle case with no exogenous vars - pass None to encoder
            cross_input = None

        # --- Encoder Forward Pass ---
        # Input: en_embed [B*n_vars_en, N+1, D], cross_input [B, L_padded, D] or None
        enc_out = self.encoder(en_embed, cross_input)
        # enc_out shape: [B * n_vars_en, N+1, D]

        # --- Prediction Head ---
        # Reshape encoder output for the head
        # Head expects [bs, nvars, d_model, patch_num+1] -> Flattens last two dims
        # Current enc_out: [B * n_vars_en, N+1, D]
        # Reshape to [B, n_vars_en, N+1, D] then permute to [B, n_vars_en, D, N+1]
        enc_out_reshaped = enc_out.view(x_enc.shape[0], self.n_vars_en, self.patch_num + 1, self.d_model)
        enc_out_reshaped = enc_out_reshaped.permute(0, 1, 3, 2) # [B, n_vars_en, D, N+1]

        # Apply head -> [B, n_vars_en, pred_len]
        dec_out = self.head(enc_out_reshaped)

        # Permute to standard [B, pred_len, n_vars_en]
        dec_out = dec_out.permute(0, 2, 1)

        # --- De-Normalization (Optional) ---
        if self.use_norm and means is not None and stdev is not None:
            dec_out = self._denormalize(dec_out, means, stdev)

        # Return only the target predictions
        return dec_out # Shape: [B, pred_len, n_vars_en]


    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                mask=None) -> torch.Tensor:
        """
        Main forward pass, routes to the appropriate method based on task.

        Args:
            x_enc (torch.Tensor): Encoder input data [B, L, Fin].
            x_mark_enc (torch.Tensor): Encoder time features [B, L, Ft].
            x_dec (torch.Tensor): Decoder input data (placeholder) [B, L_pred, Fout].
            x_mark_dec (torch.Tensor): Decoder time features [B, L_pred, Ft].
            mask: Optional mask (not typically used in this setup).

        Returns:
            torch.Tensor: Model output predictions [B, L_pred, n_vars_en].
        """
        # Currently only supporting forecasting tasks
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # Return the predictions matching pred_len
            # Shape is already [B, pred_len, n_vars_en] from forecast method
            return dec_out
        else:
            logger.error(f"Task name '{self.task_name}' not implemented.")
            raise NotImplementedError(f"Task '{self.task_name}' not implemented.")


# === Dataset Definition ===

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for creating time series sequences and targets.

    Generates batches containing:
    - x_enc: Encoder input sequence [SeqLen, NumFeatures]
    - x_mark_enc: Time features for encoder input [SeqLen, NumTimeFeatures]
    - x_dec: Decoder input (placeholder) [PredLen, NumFeatures]
    - x_mark_dec: Time features for decoder input/prediction window [PredLen, NumTimeFeatures]
    - y: Target sequence (actual values) [PredLen, NumTargetFeatures]
    """
    def __init__(self, data_df: pd.DataFrame, target_cols: list[str],
                 feature_cols: list[str], # All columns used in x_enc
                 sequence_length: int, prediction_length: int, freq: str = 'h',
                 instance_norm: bool = False):
        """
        Args:
            data_df (pd.DataFrame): DataFrame containing scaled time series data with DatetimeIndex.
            target_cols (list[str]): List of column names for the target variable(s).
            feature_cols (list[str]): List of all column names to be included in x_enc.
            sequence_length (int): Length of the input sequence (L).
            prediction_length (int): Length of the prediction horizon (H).
            freq (str): Frequency string ('h', 'd', 't', etc.) for time feature generation.
            instance_norm (bool): If True, instance normalization is applied *per sample*
                                  within __getitem__. If False, assumes data is already scaled globally.
                                  Note: Model also has internal instance norm flag, this is separate.
                                  Setting to False here is usually preferred if model handles it.
        """
        super().__init__()
        self.data_df = data_df.copy() # Work on a copy
        self.target_cols = target_cols
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.freq = freq
        self.instance_norm = instance_norm # Usually False if model does its own normalization

        if not isinstance(self.data_df.index, pd.DatetimeIndex):
             raise ValueError("data_df must have a DatetimeIndex.")

        logger.info(f"Initializing TimeSeriesDataset: SeqLen={sequence_length}, PredLen={prediction_length}, "
                    f"NumFeatures={len(feature_cols)}, NumTargets={len(target_cols)}, Freq='{freq}'")

        # Ensure target columns are present
        missing_targets = [col for col in target_cols if col not in data_df.columns]
        if missing_targets:
             raise ValueError(f"Target columns {missing_targets} not found in DataFrame.")
        # Ensure all feature columns are present
        missing_features = [col for col in feature_cols if col not in data_df.columns]
        if missing_features:
             raise ValueError(f"Feature columns {missing_features} not found in DataFrame.")


        # Convert relevant parts of DataFrame to numpy arrays for efficiency
        self.data_values = self.data_df[self.feature_cols].values.astype(np.float32)
        self.target_values = self.data_df[self.target_cols].values.astype(np.float32)
        self.index = self.data_df.index

        # Precompute time features for the entire dataset
        self.time_features = self._compute_time_features(self.index, freq=self.freq).astype(np.float32)
        self.num_time_features = self.time_features.shape[1]
        logger.info(f"Precomputed {self.num_time_features} time features for frequency '{freq}'.")

        # Calculate total number of samples possible
        self.num_samples = len(self.data_values) - self.sequence_length - self.prediction_length + 1
        if self.num_samples <= 0:
            logger.warning(f"Dataset length ({len(self.data_values)}) is too small for "
                           f"sequence_length ({self.sequence_length}) and prediction_length "
                           f"({self.prediction_length}). Resulting dataset will be empty.")
            self.num_samples = 0


# (Continuing from the previous lines, finishing TimeSeriesDataset and starting workflow functions)

    def _compute_time_features(self, dt_index: pd.DatetimeIndex, freq: str) -> np.ndarray:
        """
        Computes normalized time features based on the datetime index and frequency.

        Args:
            dt_index (pd.DatetimeIndex): The datetime index for which to compute features.
            freq (str): Frequency string ('h', 'd', 't', etc.).

        Returns:
            np.ndarray: Array of normalized time features [NumTimestamps, NumTimeFeatures].
        """
        if freq == 'h': # Hourly features
            features = [
                dt_index.hour.to_numpy(),
                dt_index.dayofweek.to_numpy(), # Monday=0, Sunday=6
                dt_index.day.to_numpy(),
                dt_index.dayofyear.to_numpy(),
                dt_index.month.to_numpy(),
                # dt_index.weekofyear is deprecated, use isocalendar().week
                dt_index.isocalendar().week.to_numpy(dtype=np.float32)
            ]
            feature_names = ['hour', 'dayofweek', 'dayofmonth', 'dayofyear', 'month', 'weekofyear']
        # Add elif blocks for other frequencies ('d', 't', 'm', etc.) if needed
        # elif freq == 'd':
        #     features = [...]
        #     feature_names = [...]
        else:
            # Default minimal features if frequency is unknown
            features = [dt_index.dayofyear.to_numpy(), dt_index.month.to_numpy()]
            feature_names = ['dayofyear', 'month']
            logger.warning(f"Using default time features {feature_names} for unknown frequency '{freq}'.")

        # Stack features and ensure float32 type
        time_features_raw = np.vstack(features).transpose().astype(np.float32)

        # Normalize features (simple min-max scaling to [0, 1] per feature)
        # This normalization is dataset-wide based on the provided dt_index
        min_vals = time_features_raw.min(axis=0)
        max_vals = time_features_raw.max(axis=0)
        range_vals = max_vals - min_vals
        # Avoid division by zero for constant features (e.g., month in a short dataset)
        range_vals[range_vals == 0] = 1.0

        time_features_normalized = (time_features_raw - min_vals) / range_vals

        logger.debug(f"Computed and normalized time features: {feature_names}")
        return time_features_normalized

    def __len__(self) -> int:
        """Returns the number of samples available in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieves a single sample (sequence, context, target) from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensors for the sample:
                'x_enc', 'x_mark_enc', 'x_dec', 'x_mark_dec', 'y'.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with size {self.num_samples}")

        # Calculate indices for the sequence and prediction windows
        seq_start = idx
        seq_end = idx + self.sequence_length
        pred_start = seq_end
        pred_end = pred_start + self.prediction_length

        # --- Input Data (Encoder) ---
        # x_enc: Input sequence data [SeqLen, NumFeatures]
        x_enc = self.data_values[seq_start:seq_end, :]
        # x_mark_enc: Time features for the input sequence [SeqLen, NumTimeFeatures]
        x_mark_enc = self.time_features[seq_start:seq_end, :]

        # --- Target Data (Ground Truth) ---
        # y: True target values for the prediction window [PredLen, NumTargetFeatures]
        y = self.target_values[pred_start:pred_end, :]
        # x_mark_dec: Time features for the prediction window [PredLen, NumTimeFeatures]
        x_mark_dec = self.time_features[pred_start:pred_end, :]

        # --- Decoder Input (Placeholder for Encoder-Only Models like TimeXer) ---
        # Often zeros or a concatenation of last part of input + zeros.
        # Using zeros matching the shape expected by the model's forward pass.
        # Shape: [PredLen, NumFeatures] (matching x_enc's feature dimension)
        x_dec = np.zeros((self.prediction_length, len(self.feature_cols)), dtype=np.float32)

                # --- Get Prediction Window Start Timestamp ---
        # <<< Modification Start >>>
        try:
            pred_start_time = self.index[pred_start] # Get the timestamp for the first point in the target window
        except IndexError:
             # Handle edge case where pred_start might be out of bounds if dataset is very small
             logger.error(f"IndexError getting pred_start_time at index {pred_start} for sample index {idx}.")
             # Assign a dummy value or raise error - let's assign NaT
             pred_start_time = pd.NaT

        # --- Apply per-sample instance normalization if requested ---
        # Note: This is generally *not* recommended if the model performs its own normalization.
        # It's included here for completeness if needed, but typically self.instance_norm should be False.
        if self.instance_norm:
             # Normalize target variable(s) within x_enc based on *this sample's* mean/std
             # Assuming target is the last N columns
             target_in_x_enc = x_enc[:, -len(self.target_cols):]
             mean = np.mean(target_in_x_enc, axis=0, keepdims=True)
             std = np.std(target_in_x_enc, axis=0, keepdims=True) + 1e-5
             x_enc[:, -len(self.target_cols):] = (target_in_x_enc - mean) / std
             # Note: This normalization is *not* easily reversible at evaluation time
             # unless means/stds are also returned, which complicates the batch structure.
             # Prefer using the model's internal normalization or global scaling.
             if idx == 0: # Log only once
                 logger.warning("Applying per-sample instance normalization in Dataset __getitem__. "
                                "This is generally not recommended if the model has internal normalization.")

        # Convert numpy arrays to PyTorch tensors
        sample = {
            'x_enc': torch.from_numpy(x_enc),
            'x_mark_enc': torch.from_numpy(x_mark_enc),
            'x_dec': torch.from_numpy(x_dec),
            'x_mark_dec': torch.from_numpy(x_mark_dec),
            'y': torch.from_numpy(y),
            # <<< Modification Start >>>
            'pred_start_time': pred_start_time # Add timestamp to the batch dictionary
            # <<< Modification End >>>
        }
        return sample

# === Data Loading and Preprocessing Function ===

def load_and_preprocess_data(csv_path: Path, start_date: str = '2003-01-01',
                             end_date: str = '2018-08-02',
                             weather_lat: float = 39.95, weather_lon: float = -75.17,
                             target_icaos: list[str] = ['KPHL', 'KEWR', 'KBWI', 'KDCA'],
                             fetch_weather: bool = True
                             ) -> pd.DataFrame | None:
    """
    Loads PJME energy data, performs feature engineering (calendar features),
    optionally fetches and integrates weather data, and handles missing values.

    Args:
        csv_path (Path): Path to the PJME_hourly.csv file.
        start_date (str): Start date for filtering data (inclusive).
        end_date (str): End date for filtering data (inclusive).
        weather_lat (float): Latitude for finding nearby weather stations.
        weather_lon (float): Longitude for finding nearby weather stations.
        target_icaos (list[str]): Preferred ICAO codes for weather stations.
        fetch_weather (bool): Whether to attempt fetching weather data.

    Returns:
        pd.DataFrame | None: Processed DataFrame with energy and optionally weather
                             features, or None if loading fails.
    """
    logger.info(f"Loading energy data from: {csv_path}")
    if not csv_path.is_file():
        logger.error(f"Energy data CSV file not found at {csv_path}")
        logger.error("Please download the PJME hourly energy data and place it correctly.")
        # Example download hint (requires wget or similar):
        # logger.error("Example: !wget https://path/to/your/data/PJME_hourly.csv -O {csv_path.name}")
        return None

    try:
        pjme = pd.read_csv(
            csv_path,
            index_col='Datetime',
            parse_dates=['Datetime']
        )
        pjme = pjme.sort_index()
        # Filter by date range
        pjme = pjme.loc[start_date:end_date]
        logger.info(f"Loaded energy data shape: {pjme.shape}, Index range: {pjme.index.min()} to {pjme.index.max()}")
    except Exception as e:
        logger.error(f"Error loading or parsing energy data CSV: {e}", exc_info=True)
        return None

    # --- Feature Engineering: Holidays and Calendar Features ---
    logger.info("Engineering calendar features...")
    try:
        cal = USFederalHolidayCalendar()
        fed_hols = cal.holidays(start=pjme.index.min(), end=pjme.index.max())

        # Logic to extend July 4th holiday (as in notebook)
        extended_hols = set(fed_hols)
        for year in range(pjme.index.year.min(), pjme.index.year.max() + 1):
            try:
                july4 = datetime(year, 7, 4)
                wd = july4.weekday() # Monday=0, Sunday=6
                if wd == 1:  # Tuesday -> add Monday
                    extended_hols.add(july4 - timedelta(days=1))
                elif wd == 2: # Wednesday -> add Thursday, Friday
                    extended_hols.add(july4 + timedelta(days=1))
                    extended_hols.add(july4 + timedelta(days=2))
                elif wd == 3: # Thursday -> add Friday
                    extended_hols.add(july4 + timedelta(days=1))
                extended_hols.add(july4) # Always add July 4th itself
            except ValueError: pass # Handle potential date issues

        all_hols_index = pd.DatetimeIndex(sorted(list(extended_hols)))

        pjme['is_holiday'] = pjme.index.normalize().isin(all_hols_index)
        pjme['is_weekend'] = pjme.index.weekday >= 5 # Saturday=5, Sunday=6
        pjme['is_dayoff'] = (pjme['is_holiday'] | pjme['is_weekend']).astype(int)
        pjme.drop(columns=['is_holiday', 'is_weekend'], inplace=True)

        # Standard Calendar Features
        pjme['hour']       = pjme.index.hour
        pjme['dayofweek']  = pjme.index.dayofweek
        pjme['dayofmonth'] = pjme.index.day
        pjme['month']      = pjme.index.month
        pjme['year']       = pjme.index.year
        pjme['dayofyear']  = pjme.index.dayofyear
        pjme['weekofyear'] = pjme.index.isocalendar().week.astype(int)
        logger.info("Calendar features engineered.")
    except Exception as e:
        logger.error(f"Error during calendar feature engineering: {e}", exc_info=True)
        # Continue without these features if error occurs? Or return None? Decide policy.
        # For now, let's try to continue.

    # --- Fetch and Process Weather Data (Optional) ---
    average_weather = None
    if fetch_weather:
        if Stations is None or Hourly is None:
            logger.warning("Meteostat library not available. Skipping weather data fetching.")
        else:
            logger.info("Fetching weather data...")
            # Define date range (slightly wider for coverage)
            start_dt = pjme.index.min() - timedelta(days=1)
            end_dt   = pjme.index.max() + timedelta(days=1)
            weather_cols = ['temp', 'dwpt', 'rhum', 'prcp', 'wspd'] # Temp(C), DewPt(C), Humidity(%), Precip(mm), Wind(km/h)

            try:
                stations_query = Stations()
                # Search stations with hourly data within the date range
                stations_query = stations_query.inventory('hourly', (start_dt, end_dt))
                # Find stations near the specified coordinates (e.g., 200km radius)
                # Note: .nearby() filters after .inventory(), might need adjustment based on Meteostat version/behavior.
                # Let's try filtering by ICAO first from the inventory.
                nearby_stations_df = stations_query.fetch()

                # Filter for target ICAOs present in the inventory result
                target_stations_df = nearby_stations_df[nearby_stations_df['icao'].isin(target_icaos)]

                station_ids_to_fetch = []
                if not target_stations_df.empty:
                    station_ids_to_fetch = target_stations_df.index.tolist()
                    logger.info(f"Found target weather stations: {target_stations_df['icao'].tolist()}")
                else:
                    logger.warning(f"None of the target ICAOs {target_icaos} found with data in range. "
                                   "Attempting to find closest stations.")
                    # Fallback: Use closest stations (requires calculating distance or using .nearby())
                    stations_query_nearby = Stations()
                    stations_query_nearby = stations_query_nearby.nearby(weather_lat, weather_lon) # Find stations sorted by distance
                    stations_query_nearby = stations_query_nearby.inventory('hourly', (start_dt, end_dt)) # Check if they have data
                    closest_stations_df = stations_query_nearby.fetch()

                    # Take top N available closest stations that were in the inventory
                    available_closest = closest_stations_df[closest_stations_df.index.isin(nearby_stations_df.index)]
                    if not available_closest.empty:
                         # Limit to a reasonable number, e.g., 4
                         station_ids_to_fetch = available_closest.head(4).index.tolist()
                         selected_icaos = available_closest.head(4)['icao'].tolist()
                         logger.info(f"Using closest available stations: {selected_icaos}")
                    else:
                         logger.warning("No suitable fallback weather stations found.")

                if station_ids_to_fetch:
                    logger.info(f"Fetching hourly data for station IDs: {station_ids_to_fetch}")
                    # Fetch hourly data for selected stations
                    weather_all = Hourly(station_ids_to_fetch, start_dt, end_dt).fetch()

                    # Select relevant columns, group by time, average, and fill NaNs
                    valid_cols = [col for col in weather_cols if col in weather_all.columns]
                    if not valid_cols:
                        logger.warning("No relevant weather columns found in fetched data.")
                    else:
                        logger.info(f"Averaging weather columns: {valid_cols}")
                        # Group by timestamp (level='time') and calculate mean across stations
                        average_weather = weather_all.groupby(level='time').mean(numeric_only=True)[valid_cols]
                        # Forward fill then backward fill NaNs
                        average_weather = average_weather.ffill().bfill()
                        logger.info(f"Weather data processed. Shape: {average_weather.shape}")
                else:
                    logger.warning("No station IDs selected for weather data fetching.")

            except Exception as e:
                logger.error(f"An error occurred during weather data fetching/processing: {e}", exc_info=True)
                logger.warning("Proceeding without weather data.")
                average_weather = None # Ensure it's None on error

    # --- Combine Energy and Weather Data ---
    if average_weather is not None:
        logger.info("Joining energy and weather data...")
        pjme_weather = pjme.join(average_weather, how='left')
        # Fill any remaining NaNs in weather columns (e.g., if join failed at edges)
        weather_cols_in_df = [col for col in weather_cols if col in pjme_weather.columns]
        pjme_weather[weather_cols_in_df] = pjme_weather[weather_cols_in_df].ffill().bfill()
        logger.info("Energy and weather data joined.")
    else:
        logger.warning("No weather data available to join. Using energy data only.")
        pjme_weather = pjme.copy()
        # Add placeholder columns if weather data was expected but failed
        if fetch_weather:
            logger.warning("Adding placeholder zero columns for missing weather features.")
            for col in weather_cols:
                if col not in pjme_weather.columns:
                    pjme_weather[col] = 0.0 # Fill with zeros or another default


    # --- Final NaN Check and Fill ---
    nan_counts = pjme_weather.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaNs found after processing & joining:\n{nan_counts[nan_counts > 0]}")
        logger.warning("Attempting final fill with 0 for any remaining NaNs...")
        pjme_weather.fillna(0, inplace=True) # Last resort fill
    else:
        logger.info("No NaNs found in the final processed DataFrame.")

    # --- Final Checks ---
    # Ensure index is sorted and monotonic
    pjme_weather = pjme_weather.sort_index()
    if not pjme_weather.index.is_monotonic_increasing:
         logger.warning("Index is not monotonic increasing after processing. Sorting again.")
         pjme_weather = pjme_weather.sort_index()

    logger.info(f"Final processed data shape: {pjme_weather.shape}")
    logger.debug(f"Final columns: {pjme_weather.columns.tolist()}")
    logger.debug(f"Data Head:\n{pjme_weather.head()}")
    logger.debug(f"Data Tail:\n{pjme_weather.tail()}")
    # Log memory usage
    try:
        memory_usage_mb = pjme_weather.memory_usage(deep=True).sum() / (1024**2)
        logger.info(f"Approximate memory usage of final DataFrame: {memory_usage_mb:.2f} MB")
    except Exception: # Sometime deep=True fails
        memory_usage_mb = pjme_weather.memory_usage().sum() / (1024**2)
        logger.info(f"Approximate memory usage of final DataFrame: {memory_usage_mb:.2f} MB")


    return pjme_weather


# === Data Splitting Function ===

def split_data(df: pd.DataFrame, val_split_ratio: float = 0.1, test_split_ratio: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame chronologically into train, validation, and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame with a DatetimeIndex.
        val_split_ratio (float): Proportion of data for the validation set.
        test_split_ratio (float): Proportion of data for the test set.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df.
    """
    if not (0 < val_split_ratio < 1 and 0 < test_split_ratio < 1 and val_split_ratio + test_split_ratio < 1):
        raise ValueError("Split ratios must be between 0 and 1, and their sum must be less than 1.")

    total_len = len(df)
    test_split_idx = int(total_len * (1 - test_split_ratio))
    val_split_idx = int(total_len * (1 - test_split_ratio - val_split_ratio))

    train_df = df.iloc[:val_split_idx].copy()
    val_df = df.iloc[val_split_idx:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()

    logger.info("Data split chronologically:")
    logger.info(f"Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Val  : {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    logger.info(f"Test : {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")

    # Basic check for empty dataframes
    if train_df.empty or val_df.empty or test_df.empty:
         logger.warning("One or more data splits are empty. Check split ratios and total data length.")

    return train_df, val_df, test_df


# === Data Scaling Function ===

def scale_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
               target_col: str, feature_cols: list[str],
               output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler]:
    """
    Applies StandardScaler to features and target variable separately.
    Fits scalers on the training set and transforms all sets. Saves the scalers.

    Args:
        train_df, val_df, test_df (pd.DataFrame): DataFrames for each split.
        target_col (str): Name of the target column to scale.
        feature_cols (list[str]): List of feature column names to scale.
        output_dir (Path): Directory to save the fitted scalers.

    Returns:
        tuple: (train_df_scaled, val_df_scaled, test_df_scaled, feature_scaler, target_scaler)
    """
    logger.info("Scaling features and target variable using StandardScaler...")

    # --- Scale Features ---
    feature_scaler = StandardScaler()
    # Ensure feature columns exist and handle potential missing columns if necessary
    train_features = train_df[feature_cols]
    val_features = val_df[feature_cols]
    test_features = test_df[feature_cols]

    # Fit on training data only
    logger.debug(f"Fitting feature scaler on {len(feature_cols)} features: {feature_cols}")
    feature_scaler.fit(train_features)

    # Transform all sets
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled[feature_cols] = feature_scaler.transform(train_features)
    val_df_scaled[feature_cols] = feature_scaler.transform(val_features)
    test_df_scaled[feature_cols] = feature_scaler.transform(test_features)
    logger.info("Features scaled.")

    # --- Scale Target Variable ---
    target_scaler = StandardScaler()
    # Reshape target column to 2D array as expected by scaler
    train_target = train_df[[target_col]]
    val_target = val_df[[target_col]]
    test_target = test_df[[target_col]]

    logger.debug(f"Fitting target scaler on column: {target_col}")
    target_scaler.fit(train_target)

    train_df_scaled[[target_col]] = target_scaler.transform(train_target)
    val_df_scaled[[target_col]] = target_scaler.transform(val_target)
    test_df_scaled[[target_col]] = target_scaler.transform(test_target)
    logger.info("Target variable scaled.")

    # --- Save Scalers ---
    try:
        feature_scaler_path = output_dir / FEATURE_SCALER_FILENAME
        target_scaler_path = output_dir / TARGET_SCALER_FILENAME
        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)
        logger.info(f"Feature scaler saved to: {feature_scaler_path}")
        logger.info(f"Target scaler saved to: {target_scaler_path}")
    except Exception as e:
        logger.error(f"Error saving scalers to {output_dir}: {e}", exc_info=True)

    logger.debug("Scaled Train DataFrame head:\n%s", train_df_scaled.head())

    return train_df_scaled, val_df_scaled, test_df_scaled, feature_scaler, target_scaler

# === DataLoader Creation Function ===

def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                       target_cols: list[str], feature_cols: list[str],
                       configs: Configs, batch_size: int, num_workers: int = 0
                       ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        train_df, val_df, test_df (pd.DataFrame): Scaled DataFrames for each split.
        target_cols (list[str]): Name of the target column(s).
        feature_cols (list[str]): List of all columns included in the scaled DataFrames.
        configs (Configs): Model/data configuration object (needs seq_len, pred_len, freq).
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for loading data.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train_loader, val_loader, test_loader.
    """
    logger.info(f"Creating DataLoaders with batch_size={batch_size}...")

    try:
        train_dataset = TimeSeriesDataset(
            train_df, target_cols=target_cols, feature_cols=feature_cols,
            sequence_length=configs.seq_len, prediction_length=configs.pred_len,
            freq=configs.freq
        )
        val_dataset = TimeSeriesDataset(
            val_df, target_cols=target_cols, feature_cols=feature_cols,
            sequence_length=configs.seq_len, prediction_length=configs.pred_len,
            freq=configs.freq
        )
        test_dataset = TimeSeriesDataset(
            test_df, target_cols=target_cols, feature_cols=feature_cols,
            sequence_length=configs.seq_len, prediction_length=configs.pred_len,
            freq=configs.freq
        )

        logger.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
             logger.warning("One or more datasets have zero length. DataLoaders might be empty.")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        logger.info("DataLoaders created successfully.")

        # Optional: Log a sample batch shape check
        try:
            batch = next(iter(train_loader))
            logger.debug("Sample batch shapes:")
            for key, value in batch.items():
                logger.debug(f"- {key}: {value.shape}")
        except StopIteration:
            logger.warning("Train loader is empty - cannot check batch shapes.")
        except Exception as e:
            logger.error(f"Error checking sample batch: {e}", exc_info=True)


        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {e}", exc_info=True)
        raise # Re-raise the exception to halt execution if loaders can't be created


# === Training Function ===

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                configs: Configs, learning_rate: float, num_epochs: int, patience: int,
                device: torch.device, output_dir: Path,
                checkpoint_filename: str = "checkpoint.pt"
                ) -> tuple[nn.Module, list[float], list[float]]:
    """
    Trains the TimeXer model using the provided data loaders and configurations.

    Args:
        model (nn.Module): The TimeXer model instance.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        configs (Configs): Model/training configuration object.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Maximum number of epochs to train.
        patience (int): Patience for early stopping.
        device (torch.device): Device to train on ('cuda' or 'cpu').
        output_dir (Path): Directory to save checkpoints and potentially loss plots.
        checkpoint_filename (str): Filename for saving the best model checkpoint.

    Returns:
        tuple[nn.Module, list[float], list[float]]:
            - The trained model (loaded with the best weights).
            - List of training losses per epoch.
            - List of validation losses per epoch.
    """
    logger.info("Starting model training...")
    model.to(device)

    # --- Setup Optimizer, Loss, Early Stopping ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # Standard L2 loss for regression
    checkpoint_path = output_dir / checkpoint_filename
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    train_losses = []
    val_losses = []
    total_training_start_time = time.time()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_batch_count = 0
        for i, batch in enumerate(train_loader):
            # Move batch to device
            try:
                x_enc = batch['x_enc'].to(device)
                x_mark_enc = batch['x_mark_enc'].to(device)
                x_dec = batch['x_dec'].to(device) # Placeholder
                x_mark_dec = batch['x_mark_dec'].to(device)
                y_true = batch['y'].to(device) # Target
            except KeyError as e:
                 logger.error(f"Missing key in batch: {e}. Check Dataset __getitem__.")
                 raise
            except Exception as e:
                 logger.error(f"Error moving batch {i} to device: {e}", exc_info=True)
                 # Decide whether to skip batch or raise error
                 continue # Skip this batch


# (Continuing from the previous lines, within the train_model function's training loop)

            optimizer.zero_grad()

            # --- Forward pass ---
            try:
                # Model expects: x_enc, x_mark_enc, x_dec, x_mark_dec
                outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                # Output shape: [B, PredLen, NumTargets]
                # Target shape: [B, PredLen, NumTargets]
                if outputs.shape != y_true.shape:
                     logger.warning(f"Shape mismatch in batch {i}: Output {outputs.shape}, Target {y_true.shape}. Skipping loss calculation.")
                     continue # Skip if shapes don't align

                loss = criterion(outputs, y_true)
            except Exception as e:
                 logger.error(f"Error during forward pass or loss calculation in training batch {i}: {e}", exc_info=True)
                 continue # Skip batch on error

            # --- Backward pass and optimize ---
            try:
                loss.backward()
                # Optional: Gradient clipping (helps prevent exploding gradients)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            except Exception as e:
                logger.error(f"Error during backward pass or optimizer step in training batch {i}: {e}", exc_info=True)
                continue # Skip batch optimization on error


            running_train_loss += loss.item()
            train_batch_count += 1

            # Optional: Log progress within epoch
            # if (i + 1) % 100 == 0: # Log every 100 batches
            #     logger.debug(f'  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Current Batch Loss: {loss.item():.6f}')

        # --- Calculate Average Train Loss for Epoch ---
        if train_batch_count > 0:
            avg_train_loss = running_train_loss / train_batch_count
            train_losses.append(avg_train_loss)
        else:
            logger.warning(f"Epoch {epoch+1}: No batches processed in training. Setting train loss to infinity.")
            avg_train_loss = float('inf')
            train_losses.append(avg_train_loss) # Append Inf or handle as error


        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad(): # Disable gradient calculations
            for i, batch in enumerate(val_loader):
                try:
                    x_enc = batch['x_enc'].to(device)
                    x_mark_enc = batch['x_mark_enc'].to(device)
                    x_dec = batch['x_dec'].to(device)
                    x_mark_dec = batch['x_mark_dec'].to(device)
                    y_true = batch['y'].to(device)

                    outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

                    if outputs.shape != y_true.shape:
                         logger.warning(f"Shape mismatch in validation batch {i}: Output {outputs.shape}, Target {y_true.shape}. Skipping loss calculation.")
                         continue

                    loss = criterion(outputs, y_true)
                    running_val_loss += loss.item()
                    val_batch_count += 1

                except Exception as e:
                    logger.error(f"Error during validation batch {i}: {e}", exc_info=True)
                    # Decide whether to skip batch or treat epoch as failed
                    continue

        # --- Calculate Average Validation Loss for Epoch ---
        if val_batch_count > 0:
            avg_val_loss = running_val_loss / val_batch_count
            val_losses.append(avg_val_loss)
        else:
            logger.warning(f"Epoch {epoch+1}: No batches processed in validation. Setting val loss to infinity.")
            avg_val_loss = float('inf')
            val_losses.append(avg_val_loss) # Append Inf or handle as error

        epoch_duration = time.time() - epoch_start_time
        logger.info(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | '
                    f'Val Loss: {avg_val_loss:.6f} | Time: {epoch_duration:.2f}s')

        # --- Early Stopping Check ---
        # Check based on validation loss
        if avg_val_loss == float('inf'):
             logger.warning(f"Epoch {epoch+1}: Skipping early stopping check due to invalid validation loss.")
        else:
             early_stopping(avg_val_loss, model)
             if early_stopping.early_stop:
                 logger.info("Early stopping triggered.")
                 break

    # --- End of Training ---
    total_training_time = time.time() - total_training_start_time
    logger.info(f'Training Finished. Total time: {total_training_time:.2f}s')

    # --- Load the best model ---
    logger.info(f"Loading best model weights from checkpoint: {checkpoint_path}")
    if checkpoint_path.is_file():
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info("Best model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}. "
                         "Using model from the last epoch.", exc_info=True)
    else:
        logger.warning(f"Checkpoint file {checkpoint_path} not found. "
                       "Using model from the last epoch.")

    return model, train_losses, val_losses


# === Evaluation Function ===

def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module,
                    target_scaler: StandardScaler, target_cols: list[str],
                    pred_len: int,
                    device: torch.device, output_dir: Path, prefix: str = "final") -> dict:
    """
    Evaluates the trained model on the test set, calculates metrics,
    and generates diagnostic plots.

    Args:
        model (nn.Module): The trained TimeXer model instance (with best weights loaded).
        test_loader (DataLoader): DataLoader for the test set.
        criterion (nn.Module): Loss function (e.g., MSELoss) to calculate test loss if needed.
        target_scaler (StandardScaler): Fitted scaler for the target variable(s) for inverse transform.
        target_cols (list[str]): Names of the target columns.
        device (torch.device): Device to run evaluation on.
        output_dir (Path): Directory to save evaluation results and plots.
        prefix (str): Prefix for saving plot filenames (e.g., 'initial', 'final', 'tuned').

    Returns:
        dict: Dictionary containing evaluation metrics (RMSE, MAE, MAPE, Test Loss).
    """
    logger.info(f"Starting model evaluation on the test set ({prefix} run)...")
    model.to(device)
    model.eval() # Set model to evaluation mode

    all_predictions = []
    all_actuals = []
    all_pred_start_times = [] # List to collect starting timestamps of prediction windows
    test_loss = 0.0
    test_batch_count = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                x_enc = batch['x_enc'].to(device)
                x_mark_enc = batch['x_mark_enc'].to(device)
                x_dec = batch['x_dec'].to(device)
                x_mark_dec = batch['x_mark_dec'].to(device)
                y_true = batch['y'].to(device) # Scaled actuals [B, PredLen, NumTargets]
                # Get timestamps from the batch (these are pd.Timestamp objects)
                pred_start_times_batch = batch['pred_start_time']

                outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec) # Scaled predictions [B, PredLen, NumTargets]

                if outputs.shape != y_true.shape:
                     logger.warning(f"Shape mismatch in test batch {i}: Output {outputs.shape}, Target {y_true.shape}.")
                     continue

                # Calculate loss on scaled data
                loss = criterion(outputs, y_true)
                test_loss += loss.item()
                test_batch_count += 1

                # Inverse transform predictions and actuals
                num_targets = y_true.shape[-1]
                # Reshape to 2D array: [B * PredLen, NumTargets]
                preds_reshaped = outputs.cpu().numpy().reshape(-1, num_targets)
                actuals_reshaped = y_true.cpu().numpy().reshape(-1, num_targets)

                preds_inv = target_scaler.inverse_transform(preds_reshaped)
                actuals_inv = target_scaler.inverse_transform(actuals_reshaped)

                all_predictions.append(preds_inv)
                all_actuals.append(actuals_inv)

                # Store timestamps, ensuring alignment with batch size (in case last batch is smaller)
                # Assumes timestamps correspond to the start of the window for each item in the batch
                # The number of items processed in this batch is outputs.shape[0]
                current_batch_size = outputs.shape[0]
                # Need to handle pd.NaT if it occurred in dataset creation
                valid_timestamps = [ts for ts in pred_start_times_batch[:current_batch_size] if pd.notna(ts)]
                all_pred_start_times.extend(valid_timestamps)

            except KeyError as e:
                 logger.error(f"Missing key '{e}' in evaluation batch {i}. Skipping batch.")
                 continue # Skip batch if timestamp or other key is missing
            except Exception as e:
                 logger.error(f"Error during evaluation batch {i}: {e}", exc_info=True)
                 continue

    if test_batch_count == 0:
        logger.error("No batches processed during evaluation. Cannot calculate metrics.")
        return {"error": "No batches processed"}

    # --- Combine results from all batches ---
    predictions_np = np.concatenate(all_predictions, axis=0)
    actuals_np = np.concatenate(all_actuals, axis=0)
    avg_test_loss = test_loss / test_batch_count

    # --- Reconstruct the full prediction index ---
    full_prediction_index = None
    logger.info("Reconstructing prediction timestamp index...")
    if all_pred_start_times:
        # Infer frequency from the first few timestamps
        # Use pd.infer_freq on the Timestamps directly
        time_freq = pd.infer_freq(pd.Series(all_pred_start_times[:min(len(all_pred_start_times), 10)]))
        if time_freq is None:
             logger.warning("Could not infer frequency from prediction start times. Assuming hourly ('H').")
             time_freq = 'H' # Default to hourly if inference fails
        else:
             logger.info(f"Inferred prediction frequency: {time_freq}")

        # Check consistency: Total predictions should equal num_windows * pred_len
        num_windows = len(all_pred_start_times)
        expected_predictions = num_windows * pred_len
        actual_predictions = len(predictions_np)

        if expected_predictions != actual_predictions:
            logger.error(f"Prediction count mismatch: Expected {expected_predictions} ({num_windows} windows * {pred_len} steps/window), "
                         f"but found {actual_predictions} total prediction steps. Cannot reliably reconstruct index.")
        else:
            reconstructed_index_list = []
            for start_time in all_pred_start_times:
                # Generate DatetimeIndex for the prediction window [start_time, start_time + pred_len steps)
                try:
                    # Note: freq=time_freq might need adjustment based on how date_range interprets it (e.g., 'H' vs 'h')
                    # Using offset aliases generally works well.
                    window_index = pd.date_range(start=start_time, periods=pred_len, freq=time_freq)
                    reconstructed_index_list.extend(window_index)
                except Exception as e:
                     logger.error(f"Error generating date range for start_time {start_time} with freq {time_freq}: {e}")
                     reconstructed_index_list = None # Mark as failed
                     break

            if reconstructed_index_list:
                full_prediction_index = pd.DatetimeIndex(reconstructed_index_list)
                logger.info(f"Reconstructed prediction index from {full_prediction_index.min()} to {full_prediction_index.max()} "
                            f"with {len(full_prediction_index)} entries.")
                # Final length check
                if len(full_prediction_index) != len(predictions_np):
                    logger.error(f"Length mismatch after reconstructing index ({len(full_prediction_index)}) and predictions ({len(predictions_np)}).")
                    full_prediction_index = None # Mark as invalid
    else:
        logger.warning("No valid prediction start times collected. Cannot reconstruct index.")




    logger.info(f"Evaluation completed. Average Test Loss (scaled): {avg_test_loss:.6f}")
    logger.info(f"Shape of inverse-scaled predictions array: {predictions_np.shape}")
    logger.info(f"Shape of inverse-scaled actuals array: {actuals_np.shape}")

    # --- Calculate Metrics ---
    # Calculate metrics for each target variable if multiple targets exist
    metrics = {}
    num_targets = actuals_np.shape[1]
    for i in range(num_targets):
        target_name = target_cols[i]
        actual = actuals_np[:, i]
        pred = predictions_np[:, i]

        # Ensure lengths match before calculating metrics
        if len(actual) != len(pred):
             logger.error(f"Length mismatch for target '{target_name}': Actuals {len(actual)}, Predictions {len(pred)}. Skipping metrics.")
             metrics[f'{target_name}_RMSE'] = np.nan
             metrics[f'{target_name}_MAE'] = np.nan
             metrics[f'{target_name}_MAPE'] = np.nan
             continue

        try:
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            mape = mean_absolute_percentage_error(actual, pred) # Uses helper function

            metrics[f'{target_name}_RMSE'] = rmse
            metrics[f'{target_name}_MAE'] = mae
            metrics[f'{target_name}_MAPE'] = mape

            logger.info(f"Metrics for target '{target_name}':")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE : {mae:.4f}")
            logger.info(f"  MAPE: {mape:.4f}%")

        except Exception as e:
            logger.error(f"Error calculating metrics for target '{target_name}': {e}", exc_info=True)
            metrics[f'{target_name}_RMSE'] = np.nan
            metrics[f'{target_name}_MAE'] = np.nan
            metrics[f'{target_name}_MAPE'] = np.nan


    metrics[f'{prefix}_Test_Loss_Scaled'] = avg_test_loss

    # --- Generate Diagnostic Plots ---
    logger.info("Generating diagnostic plots...")
    # Assuming single target for simplicity in plots for now. Adapt if multi-target.
    if num_targets == 1:
        target_name = target_cols[0]
        actual = actuals_np[:, 0]
        pred = predictions_np[:, 0]
        residuals = actual - pred

        # 1. Residuals vs. Predicted Plot
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=pred, y=residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'{prefix} - Residuals vs. Predicted Values ({target_name})')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True)
        plot_path = output_dir / f"{prefix}_residuals_vs_predicted.png"
        try:
             plt.savefig(plot_path, bbox_inches='tight')
             logger.info(f"Saved residuals vs predicted plot to {plot_path}")
        except Exception as e:
             logger.error(f"Failed to save plot {plot_path}: {e}", exc_info=True)
        plt.close() # Close figure to free memory

        # 2. Residuals Histogram
        plt.figure(figsize=(10, 5))
        sns.histplot(residuals, kde=True)
        plt.title(f'{prefix} - Histogram of Residuals ({target_name})')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plot_path = output_dir / f"{prefix}_residuals_histogram.png"
        try:
             plt.savefig(plot_path, bbox_inches='tight')
             logger.info(f"Saved residuals histogram plot to {plot_path}")
        except Exception as e:
             logger.error(f"Failed to save plot {plot_path}: {e}", exc_info=True)
        plt.close()

        # 3. Actual vs. Predicted Scatter Plot
        plt.figure(figsize=(8, 8))
        max_val = max(actual.max(), pred.max())
        min_val = min(actual.min(), pred.min())
        plt.scatter(actual, pred, alpha=0.3)
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
        plt.title(f'{prefix} - Actual vs. Predicted Values ({target_name})')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.grid(True)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box') # Make axes equal
        plot_path = output_dir / f"{prefix}_actual_vs_predicted.png"
        try:
             plt.savefig(plot_path, bbox_inches='tight')
             logger.info(f"Saved actual vs predicted plot to {plot_path}")
        except Exception as e:
             logger.error(f"Failed to save plot {plot_path}: {e}", exc_info=True)
        plt.close()
    else:
        logger.warning("Diagnostic plots are currently implemented for single-target prediction only.")


    # --- Save Metrics ---
    metrics_path = output_dir / f"{prefix}_evaluation_metrics.json"
    try:
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {k: (float(v) if isinstance(v, (np.number, np.bool_)) else v) for k, v in metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to: {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {metrics_path}: {e}", exc_info=True)


    # Store predictions, actuals, and the reconstructed index object
    metrics['predictions'] = predictions_np.tolist() # Convert for JSON compatibility
    metrics['actuals'] = actuals_np.tolist()
    metrics['prediction_index_obj'] = full_prediction_index # Keep actual DatetimeIndex object

    return metrics


# === Forecast Plotting Function ===

def plot_forecasts(results_df: pd.DataFrame, target_col: str,
                   output_dir: Path, prefix: str = "final",
                   sample_size: int = 24 * 7): # Plot one week by default
    """
    Plots actual vs predicted values for the test set predictions.

    Args:
        results_df (pd.DataFrame): DataFrame containing 'Actual' and 'Predicted' columns
                                  with a DatetimeIndex matching the prediction times.
        target_col (str): Name of the target column being plotted.
        output_dir (Path): Directory to save the plot.
        prefix (str): Prefix for saving the plot filename.
        sample_size (int): Number of initial points to plot for a sample view.
                          Set to 0 or None to plot the entire series.
    """
    logger.info("Generating forecast comparison plots...")

    if results_df.empty:
        logger.warning("Results DataFrame is empty. Skipping forecast plotting.")
        return

    # --- Plot a Sample Period ---
    if sample_size is not None and sample_size > 0 and sample_size <= len(results_df):
        plot_data_sample = results_df.head(sample_size)
        title_suffix = f"(First {sample_size} Points)"
        filename_suffix = "_sample"
    else:
        plot_data_sample = results_df
        title_suffix = "(Full Test Period)"
        filename_suffix = "_full"
        sample_size = len(results_df) # For logging message

    plt.figure(figsize=(18, 6))
    plt.plot(plot_data_sample.index, plot_data_sample['Actual'], label='Actual', marker='.', linestyle='-', markersize=4, alpha=0.7)
    plt.plot(plot_data_sample.index, plot_data_sample['Predicted'], label='Predicted', marker='.', linestyle='--', markersize=4, alpha=0.7)
    plt.title(f'{prefix.capitalize()} TimeXer Forecast vs Actuals for {target_col} {title_suffix}')
    plt.xlabel('Timestamp')
    plt.ylabel(f'{target_col}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45) # Improve date readability
    plt.tight_layout() # Adjust layout

    plot_path = output_dir / f"{prefix}_forecast_vs_actual{filename_suffix}.png"
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        logger.info(f"Saved forecast plot ({sample_size} points) to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_path}: {e}", exc_info=True)
    plt.close() # Close figure


# === Optuna Objective Function (if Optuna is available) ===

# Define objective function only if optuna was imported successfully
if optuna:
    def objective(trial: optuna.trial.Trial,
                  # Pass necessary data/configs as arguments - avoids globals
                  train_df_scaled: pd.DataFrame,
                  val_df_scaled: pd.DataFrame,
                  target_cols: list[str],
                  feature_cols: list[str],
                  base_configs: Configs, # Base configs with non-tuned params
                  device: torch.device,
                  output_dir: Path
                  ) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            train_df_scaled (pd.DataFrame): Scaled training data.
            val_df_scaled (pd.DataFrame): Scaled validation data.
            target_cols (list[str]): Target column name(s).
            feature_cols (list[str]): All feature column names.
            base_configs (Configs): Config object with fixed parameters.
            device (torch.device): Device to run training on.
            output_dir (Path): Base output directory (trial artifacts might go in subdirs).

        Returns:
            float: The metric to minimize (e.g., best validation MSE loss).
        """
        # --- 1. Suggest Hyperparameters ---
        try:
            # Clone base configs to modify for this trial
            trial_configs = copy.deepcopy(base_configs)

            # Suggest hyperparameters
            trial_configs.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            trial_configs.d_model = trial.suggest_categorical('d_model', [64, 128, 256])
            trial_configs.seq_len = trial.suggest_categorical('seq_len', [24*7, 24*10, 24*14]) # 7, 10, 14 days
            trial_configs.patch_len = trial.suggest_categorical('patch_len', [8, 12, 16, 24])
            trial_configs.dropout = trial.suggest_float('dropout', 0.05, 0.3)
            # trial_configs.e_layers = trial.suggest_int('e_layers', 1, 3) # Optionally tune layers
            # trial_configs.n_heads = trial.suggest_categorical('n_heads', [4, 8]) # Optionally tune heads

            # --- Update dependent parameters and validate ---
            trial_configs.d_ff = trial_configs.d_model * 4 # Keep d_ff relative
            if trial_configs.d_model % trial_configs.n_heads != 0:
                logger.warning(f"Trial {trial.number}: d_model {trial_configs.d_model} not divisible by "
                               f"n_heads {trial_configs.n_heads}. Pruning.")
                raise optuna.exceptions.TrialPruned(f"Incompatible d_model/n_heads")
            # seq_len must be >= patch_len? (Handled by EnEmbedding padding now)

            logger.info(f"--- Optuna Trial {trial.number} ---")
            logger.info(f"Params: lr={trial_configs.learning_rate:.6f}, d_model={trial_configs.d_model}, "
                        f"seq_len={trial_configs.seq_len}, patch_len={trial_configs.patch_len}, dropout={trial_configs.dropout:.3f}")


# (Continuing from the previous lines, within the Optuna objective function)

            # --- 2. Create Datasets & DataLoaders for this trial (using trial's seq_len) ---
            # Use a reasonable batch size for trials (could also be tuned)
            optuna_batch_size = 32 # Or base_configs.batch_size
            try:
                # Important: Use the pre-scaled dataframes passed as arguments
                train_dataset_trial = TimeSeriesDataset(
                    train_df_scaled, target_cols=target_cols, feature_cols=feature_cols,
                    sequence_length=trial_configs.seq_len, prediction_length=trial_configs.pred_len,
                    freq=trial_configs.freq
                )
                val_dataset_trial = TimeSeriesDataset(
                    val_df_scaled, target_cols=target_cols, feature_cols=feature_cols,
                    sequence_length=trial_configs.seq_len, prediction_length=trial_configs.pred_len,
                    freq=trial_configs.freq
                )

                # Check if datasets are too small for the chosen seq_len/pred_len
                if len(train_dataset_trial) == 0 or len(val_dataset_trial) == 0:
                    logger.warning(f"Trial {trial.number}: Dataset too small for seq_len={trial_configs.seq_len} "
                                   f"and pred_len={trial_configs.pred_len}. Pruning.")
                    raise optuna.exceptions.TrialPruned("Dataset empty for this sequence length")

                train_loader_trial = DataLoader(
                    train_dataset_trial, batch_size=optuna_batch_size, shuffle=True, num_workers=0 # Keep num_workers=0 for simplicity in trials
                )
                val_loader_trial = DataLoader(
                    val_dataset_trial, batch_size=optuna_batch_size, shuffle=False, num_workers=0
                )

            except optuna.exceptions.TrialPruned as e:
                 raise e # Re-raise prune exception
            except Exception as e:
                logger.error(f"Trial {trial.number}: Error creating datasets/dataloaders: {e}. Pruning.", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Dataset/Dataloader creation error: {e}")


            # --- 3. Initialize Model, Loss, Optimizer for this trial ---
            try:
                model_trial = Model(trial_configs).to(device)
                # Count parameters (optional, but informative)
                # total_params = sum(p.numel() for p in model_trial.parameters() if p.requires_grad)
                # logger.debug(f"Trial {trial.number}: Model parameters: {total_params:,}")

            except Exception as e:
                 logger.error(f"Trial {trial.number}: Error initializing model: {e}. Pruning.", exc_info=True)
                 raise optuna.exceptions.TrialPruned(f"Model initialization error: {e}")

            criterion_trial = nn.MSELoss()
            optimizer_trial = optim.Adam(model_trial.parameters(), lr=trial_configs.learning_rate)

            # --- 4. Training & Validation Loop (Condensed for Optuna) ---
            # Use fewer epochs and stricter patience for trials
            num_optuna_epochs = 15 # Example: Train for max 15 epochs per trial
            patience_optuna = 3    # Example: Stop trial if no improvement for 3 epochs
            best_val_loss_trial = float('inf')
            stopping_counter = 0

            logger.info(f"Trial {trial.number}: Starting training for max {num_optuna_epochs} epochs...")
            for epoch in range(num_optuna_epochs):
                # --- Trial Training Phase ---
                model_trial.train()
                running_train_loss_trial = 0.0
                train_batch_count_trial = 0
                for batch in train_loader_trial:
                    try:
                        x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = [b.to(device) for b in batch.values()]
                        optimizer_trial.zero_grad()
                        outputs = model_trial(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        loss = criterion_trial(outputs, y_true)
                        loss.backward()
                        optimizer_trial.step()
                        running_train_loss_trial += loss.item()
                        train_batch_count_trial += 1
                    except Exception as e:
                         # Catch potential OOM errors during training
                         if 'CUDA out of memory' in str(e):
                             logger.error(f"Trial {trial.number}: CUDA OOM Error during training epoch {epoch+1}. Pruning.")
                             raise optuna.exceptions.TrialPruned("CUDA OOM Error")
                         else:
                             logger.error(f"Trial {trial.number}: Error during training epoch {epoch+1}: {e}. Skipping batch.", exc_info=True)
                             continue # Skip batch

                avg_train_loss_trial = running_train_loss_trial / train_batch_count_trial if train_batch_count_trial > 0 else float('inf')

                # --- Trial Validation Phase ---
                model_trial.eval()
                running_val_loss_trial = 0.0
                val_batch_count_trial = 0
                with torch.no_grad():
                    for batch in val_loader_trial:
                         try:
                            x_enc, x_mark_enc, x_dec, x_mark_dec, y_true = [b.to(device) for b in batch.values()]
                            outputs = model_trial(x_enc, x_mark_enc, x_dec, x_mark_dec)
                            loss = criterion_trial(outputs, y_true)
                            running_val_loss_trial += loss.item()
                            val_batch_count_trial += 1
                         except Exception as e:
                             # Catch potential OOM errors during validation
                             if 'CUDA out of memory' in str(e):
                                 logger.error(f"Trial {trial.number}: CUDA OOM Error during validation epoch {epoch+1}. Pruning.")
                                 raise optuna.exceptions.TrialPruned("CUDA OOM Error")
                             else:
                                 logger.error(f"Trial {trial.number}: Error during validation epoch {epoch+1}: {e}. Skipping batch.", exc_info=True)
                                 continue # Skip batch

                avg_val_loss_trial = running_val_loss_trial / val_batch_count_trial if val_batch_count_trial > 0 else float('inf')

                logger.debug(f"  Trial {trial.number} Epoch {epoch+1}/{num_optuna_epochs} | Train Loss: {avg_train_loss_trial:.6f} | Val Loss: {avg_val_loss_trial:.6f}")

                # Update best validation loss for *this trial*
                if avg_val_loss_trial < best_val_loss_trial:
                    best_val_loss_trial = avg_val_loss_trial
                    stopping_counter = 0 # Reset counter
                else:
                    stopping_counter += 1

                # --- Optuna Pruning & Intermediate Reporting ---
                # Report the *current* epoch's validation loss to Optuna
                trial.report(avg_val_loss_trial, epoch)
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at epoch {epoch+1} based on intermediate value: {avg_val_loss_trial:.6f}")
                    raise optuna.exceptions.TrialPruned()

                # Simple early stopping within the trial based on patience
                if stopping_counter >= patience_optuna:
                    logger.info(f"Trial {trial.number} stopped early at epoch {epoch+1} due to lack of improvement.")
                    break

            # --- 5. Return Metric to Minimize ---
            # Return the best validation loss achieved *during this trial*
            if best_val_loss_trial == float('inf'):
                 logger.warning(f"Trial {trial.number} finished with invalid best validation loss. Returning Inf.")
            logger.info(f"Trial {trial.number} finished. Best Validation Loss: {best_val_loss_trial:.6f}")
            return best_val_loss_trial

        except optuna.exceptions.TrialPruned as e:
            # Important: Re-raise TrialPruned exceptions for Optuna to handle
            raise e
        except Exception as e:
            # Handle other unexpected errors during the trial
            logger.error(f"Trial {trial.number} failed with unexpected error: {e}", exc_info=True)
            # Return a large value or NaN to indicate failure to Optuna
            return float('inf')
# End of Optuna objective function definition

# === Hyperparameter Tuning Function ===

def run_hyperparameter_tuning(n_trials: int, timeout: int | None, study_name: str, storage: str | None,
                              train_df_scaled: pd.DataFrame, val_df_scaled: pd.DataFrame,
                              target_cols: list[str], feature_cols: list[str],
                              base_configs: Configs, device: torch.device, output_dir: Path
                              ) -> dict | None:
    """
    Sets up and runs an Optuna hyperparameter tuning study.

    Args:
        n_trials (int): Number of trials to run.
        timeout (int | None): Optional timeout for the study in seconds.
        study_name (str): Name for the Optuna study.
        storage (str | None): Optuna storage URL (e.g., for database persistence). None for in-memory.
        train_df_scaled, val_df_scaled: Scaled dataframes.
        target_cols, feature_cols: Column lists.
        base_configs (Configs): Base configurations with fixed parameters.
        device (torch.device): Device ('cuda' or 'cpu').
        output_dir (Path): Directory to save Optuna study results/plots.

    Returns:
        dict | None: Dictionary containing the best hyperparameters found, or None if tuning failed or was skipped.
    """
    if optuna is None:
        logger.warning("Optuna library not available. Skipping hyperparameter tuning.")
        return None

    logger.info(f"--- Starting Optuna Hyperparameter Tuning ---")
    logger.info(f"Study Name: {study_name}, Number of Trials: {n_trials}, Timeout: {timeout}s")

    # --- Create Optuna Study ---
    # Consider using a pruner (e.g., MedianPruner)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize', # Minimize validation loss
        storage=storage,      # Use None for in-memory storage
        pruner=pruner,
        load_if_exists=True   # Resume study if it exists (useful with DB storage)
    )

    # --- Define the objective function with fixed arguments ---
    # Use functools.partial or lambda to pass extra args to objective
    objective_with_args = lambda trial: objective(
        trial,
        train_df_scaled=train_df_scaled,
        val_df_scaled=val_df_scaled,
        target_cols=target_cols,
        feature_cols=feature_cols,
        base_configs=base_configs,
        device=device,
        output_dir=output_dir # Pass output dir if objective needs to save trial artifacts
    )

    # --- Run Optimization ---
    try:
        study.optimize(objective_with_args, n_trials=n_trials, timeout=timeout, gc_after_trial=True) # Enable garbage collection
    except KeyboardInterrupt:
        logger.warning("Optuna optimization stopped manually via KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Optuna optimization: {e}", exc_info=True)

    # --- Process and Save Results ---
    logger.info(f"Optuna study finished. Number of finished trials: {len(study.trials)}")

    best_hyperparams = None
    if study.best_trial:
        best_value = study.best_value
        best_hyperparams = study.best_params
        logger.info(f"Best trial finished with value (best validation loss): {best_value:.6f}")
        logger.info("Best hyperparameters found:")
        # Use json dumps for nice formatting in log
        logger.info(json.dumps(best_hyperparams, indent=4))

        # --- Save Best Hyperparameters ---
        params_path = output_dir / BEST_PARAMS_FILENAME
        try:
            with open(params_path, 'w') as f:
                json.dump(best_hyperparams, f, indent=4)
            logger.info(f"Best hyperparameters saved to: {params_path}")
        except Exception as e:
            logger.error(f"Error saving best hyperparameters to {params_path}: {e}", exc_info=True)

        # --- Generate and Save Optuna Plots ---
        logger.info("Generating Optuna visualization plots...")
        try:
            if optuna.visualization.is_available():
                fig_history = optuna.visualization.plot_optimization_history(study)
                fig_history.write_image(output_dir / f"{study_name}_optimization_history.png")
                logger.info(f"Saved optimization history plot.")

                fig_importance = optuna.visualization.plot_param_importances(study)
                fig_importance.write_image(output_dir / f"{study_name}_param_importances.png")
                logger.info(f"Saved parameter importances plot.")

                # Slice plot can be useful too
                # fig_slice = optuna.visualization.plot_slice(study)
                # fig_slice.write_image(output_dir / f"{study_name}_slice_plot.png")
                # logger.info(f"Saved slice plot.")
            else:
                 logger.warning("Optuna visualization is not available (plotly might be missing). Skipping plots.")
        except ImportError:
             logger.warning("Plotly is required for Optuna visualization. Skipping plots. Install with: pip install plotly")
        except Exception as e:
            logger.error(f"Could not generate or save Optuna plots: {e}", exc_info=True)

    else:
        logger.warning("Optuna study completed, but no best trial was found (e.g., all failed or pruned).")

    return best_hyperparams


# === Main Execution Block ===

def main():
    """
    Main function to orchestrate the TimeXer forecasting workflow.
    """
    logger.info("=================================================")
    logger.info(f" Starting TimeXer Workflow @ {datetime.now()} ")
    logger.info("=================================================")
    workflow_start_time = time.time()

    # --- Argument Parsing (Optional but Recommended) ---
    # Example using argparse:
    import argparse
    parser = argparse.ArgumentParser(description="TimeXer Time Series Forecasting Script")
    parser.add_argument('--data_path', type=str, default=str(DATA_DIR / PJME_CSV_FILENAME),
                        help='Path to the PJME hourly data CSV file.')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='Directory to save outputs (logs, models, plots, etc.).')
    parser.add_argument('--target_col', type=str, default='PJME_MW', help='Name of the target column.')
    # Add arguments for epochs, learning rate, seq_len, pred_len, patch_len, d_model etc.
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--seq_len', type=int, default=168, help='Input sequence length (e.g., 7 days).')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction horizon (e.g., 1 day).')
    parser.add_argument('--patch_len', type=int, default=24, help='Patch length for embedding.')
    parser.add_argument('--d_model', type=int, default=128, help='Model embedding dimension.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--use_norm', action='store_true', help='Use instance normalization in the model.') # Flag defaults to False
    parser.add_argument('--skip_weather', action='store_true', help='Skip fetching and using weather data.')
    parser.add_argument('--run_tuning', action='store_true', help='Run hyperparameter tuning with Optuna.')
    parser.add_argument('--tuning_trials', type=int, default=20, help='Number of Optuna trials.')
    parser.add_argument('--tuning_timeout', type=int, default=None, help='Optuna study timeout in seconds.')
    parser.add_argument('--tuning_study_name', type=str, default=f"{SCRIPT_NAME}_study", help='Optuna study name.')
    parser.add_argument('--tuning_storage', type=str, default=None, help='Optuna storage URL (e.g., sqlite:///optuna.db).')


    args = parser.parse_args()

    # Update output dir based on args
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    logger.info(f"Running with arguments: {args}")

    # --- Configuration Setup ---
    # Combine fixed settings and arguments into a Configs object
    # Start with base/default configuration values
    base_configs_dict = {
        'task_name': 'short_term_forecast', # Or 'long_term_forecast'
        'features': 'MS',           # Multivariate input, Single target ('PJME_MW') - Assumed for now
        'enc_in': -1,               # Placeholder, will be set after data loading
        'n_vars_en': 1,             # Assuming single target variable for now
        'c_exog': -1,               # Placeholder, will be set after data loading
        'd_ff': args.d_model * 4,   # Derived
        'activation': 'gelu',
        'embed': 'timeF',           # Type of time embedding
        'freq': 'h',                # Frequency of data ('h' for hourly)
        'factor': 3,                # Attention factor (for potential future use)

        # Parameters potentially tuned or set by args
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'patch_len': args.patch_len,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'e_layers': args.e_layers,
        'dropout': args.dropout,
        'use_norm': args.use_norm,
        # Add learning rate here as well if needed by base configs (though usually separate)
        # 'learning_rate': args.lr
    }
    # Convert dict to Configs object
    current_configs = Configs(**base_configs_dict)

# (Continuing from the previous lines, within the main function)

    # --- 1. Load and Preprocess Data ---
    try:
        processed_df = load_and_preprocess_data(
            csv_path=Path(args.data_path),
            fetch_weather=(not args.skip_weather)
            # Add args for start_date, end_date, weather params if needed
        )
        if processed_df is None:
            logger.error("Data loading and preprocessing failed. Exiting.")
            return # Exit script if data loading fails
    except Exception as e:
        logger.error(f"Unhandled error during data loading/preprocessing: {e}", exc_info=True)
        return

    # --- Update Configs with Data-Dependent Values ---
    all_feature_cols = processed_df.columns.tolist()
    # Remove target col to get exogenous features
    exog_feature_cols = [col for col in all_feature_cols if col != args.target_col]
    target_cols_list = [args.target_col] # Assuming single target for now

    current_configs.enc_in = len(all_feature_cols) # Total input features
    current_configs.c_exog = len(exog_feature_cols)  # Number of exogenous features
    current_configs.n_vars_en = len(target_cols_list) # Number of endogenous/target features

    # Log updated config
    logger.info("Updated configurations based on loaded data:")
    logger.info(f"  enc_in (total features): {current_configs.enc_in}")
    logger.info(f"  c_exog (exogenous features): {current_configs.c_exog}")
    logger.info(f"  n_vars_en (target features): {current_configs.n_vars_en}")


    # --- 2. Split Data ---
    try:
        train_raw_df, val_raw_df, test_raw_df = split_data(processed_df)
    except Exception as e:
        logger.error(f"Error during data splitting: {e}", exc_info=True)
        return


    # --- 3. Scale Data ---
    try:
        train_df, val_df, test_df, feature_scaler, target_scaler = scale_data(
            train_raw_df, val_raw_df, test_raw_df,
            target_col=args.target_col, # Pass single target column name
            feature_cols=exog_feature_cols, # Scale only exogenous features separately? No, scale all features together except target?
                                             # Let's stick to notebook: scale features (exog) and target separately.
                                             # Update: The notebook scales FEATURES (all non-target) and TARGET separately.
            output_dir=output_dir
        )
    except Exception as e:
        logger.error(f"Error during data scaling: {e}", exc_info=True)
        return

    # --- 4. Hyperparameter Tuning (Optional) ---
    best_hyperparams = None
    if args.run_tuning:
        if optuna:
             try:
                 # Prepare base configs for tuning (parameters not being tuned)
                 tuning_base_configs = Configs(**base_configs_dict)
                 # Update with data-dependent values
                 tuning_base_configs.enc_in = current_configs.enc_in
                 tuning_base_configs.c_exog = current_configs.c_exog
                 tuning_base_configs.n_vars_en = current_configs.n_vars_en
                 # Remove params that *will* be tuned from the base config passed to objective
                 delattr(tuning_base_configs, 'learning_rate', None) # Will be set by trial
                 delattr(tuning_base_configs, 'd_model', None)
                 delattr(tuning_base_configs, 'seq_len', None)
                 delattr(tuning_base_configs, 'patch_len', None)
                 delattr(tuning_base_configs, 'dropout', None)

                 best_hyperparams = run_hyperparameter_tuning(
                     n_trials=args.tuning_trials,
                     timeout=args.tuning_timeout,
                     study_name=args.tuning_study_name,
                     storage=args.tuning_storage,
                     train_df_scaled=train_df, # Pass scaled data
                     val_df_scaled=val_df,     # Pass scaled data
                     target_cols=target_cols_list,
                     feature_cols=all_feature_cols, # Pass all columns names
                     base_configs=tuning_base_configs,
                     device=DEVICE,
                     output_dir=output_dir
                 )
             except Exception as e:
                 logger.error(f"Error during hyperparameter tuning: {e}", exc_info=True)
                 logger.warning("Proceeding with default hyperparameters.")
                 best_hyperparams = None # Ensure it's None if tuning fails
        else:
            logger.warning("Optuna requested but library not available. Skipping tuning.")

    # --- 5. Final Configuration Setup ---
    logger.info("--- Setting up Final Model Configuration ---")
    final_configs = Configs(**base_configs_dict) # Start with base/args again
    # Update data-dependent values
    final_configs.enc_in = current_configs.enc_in
    final_configs.c_exog = current_configs.c_exog
    final_configs.n_vars_en = current_configs.n_vars_en

    final_learning_rate = args.lr # Default LR

    if best_hyperparams:
        logger.info("Applying best hyperparameters found by Optuna.")
        # Update final_configs and final_learning_rate with tuned values
        try:
             final_configs.seq_len = best_hyperparams.get('seq_len', final_configs.seq_len)
             final_configs.patch_len = best_hyperparams.get('patch_len', final_configs.patch_len)
             final_configs.d_model = best_hyperparams.get('d_model', final_configs.d_model)
             final_configs.dropout = best_hyperparams.get('dropout', final_configs.dropout)
             # Update other tuned params if any (e.g., e_layers, n_heads)
             # final_configs.e_layers = best_hyperparams.get('e_layers', final_configs.e_layers)
             # final_configs.n_heads = best_hyperparams.get('n_heads', final_configs.n_heads)

             # Update dependent config
             final_configs.d_ff = final_configs.d_model * 4

             # Update learning rate separately
             final_learning_rate = best_hyperparams.get('learning_rate', final_learning_rate)

             # Validate compatibility after update (e.g., d_model vs n_heads)
             if final_configs.d_model % final_configs.n_heads != 0:
                 logger.warning(f"Tuned d_model ({final_configs.d_model}) not divisible by n_heads ({final_configs.n_heads}). Check config/tuning ranges.")
                 # Potentially adjust n_heads or raise error
                 # raise ValueError("Incompatible tuned d_model and n_heads")

        except KeyError as e:
             logger.error(f"KeyError accessing tuned hyperparameter '{e}'. Check Optuna objective/results. Using defaults.")
             # Reset LR to default if key missing
             final_learning_rate = args.lr
        except Exception as e:
             logger.error(f"Error applying tuned hyperparameters: {e}. Using defaults.", exc_info=True)
             final_learning_rate = args.lr

    else:
        logger.info("No Optuna results available or tuning skipped. Using default/command-line hyperparameters.")

    logger.info("Final Configuration for Training:")
    logger.info(f"  Learning Rate: {final_learning_rate:.6f}")
    # Log the final config object
    for key, value in final_configs.__dict__.items():
        logger.info(f"  {key}: {value}")


    # --- 6. Create Final DataLoaders ---
    # Use the final configuration (especially seq_len)
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            target_cols=target_cols_list,
            feature_cols=all_feature_cols, # Pass all column names
            configs=final_configs, # Use final configs
            batch_size=args.batch_size,
            num_workers=0 # Adjust as needed, 0 often safer for debugging
        )
    except Exception as e:
        logger.error(f"Failed to create final DataLoaders: {e}", exc_info=True)
        return


    # --- 7. Initialize Final Model ---
    try:
        final_model = Model(final_configs)
        logger.info("Final TimeXer model initialized.")
        # Log parameter count
        total_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters in final model: {total_params:,}")
    except Exception as e:
        logger.error(f"Failed to initialize final model: {e}", exc_info=True)
        return


    # --- 8. Train Final Model ---
    try:
        trained_model, train_losses, val_losses = train_model(
            model=final_model,
            train_loader=train_loader,
            val_loader=val_loader,
            configs=final_configs,
            learning_rate=final_learning_rate,
            num_epochs=args.epochs,
            patience=args.patience,
            device=DEVICE,
            output_dir=output_dir,
            checkpoint_filename=BEST_MODEL_FILENAME # Use specific name for final model
        )
        # Plot training loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Final Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        loss_curve_path = output_dir / "final_training_loss_curve.png"
        try:
             plt.savefig(loss_curve_path, bbox_inches='tight')
             logger.info(f"Saved final training loss curve to {loss_curve_path}")
        except Exception as e:
             logger.error(f"Failed to save loss curve plot {loss_curve_path}: {e}", exc_info=True)
        plt.close()

    except Exception as e:
        logger.error(f"Error during final model training: {e}", exc_info=True)
        return


    # --- 9. Evaluate Final Model ---
    try:
        # Use MSE loss for evaluation consistency, though metrics don't depend on it
        evaluation_criterion = nn.MSELoss()
        eval_results = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            criterion=evaluation_criterion,
            target_scaler=target_scaler, # Pass the loaded/saved target scaler
            target_cols=target_cols_list,
            pred_len=final_configs.pred_len, # Pass pred_len here
            device=DEVICE,
            output_dir=output_dir,
            prefix="final" # Indicate this is the final evaluation
        )
        logger.info(f"Final evaluation results summary: "
                    f"RMSE={eval_results.get(f'{args.target_col}_RMSE', 'N/A'):.4f}, "
                    f"MAE={eval_results.get(f'{args.target_col}_MAE', 'N/A'):.4f}, "
                    f"MAPE={eval_results.get(f'{args.target_col}_MAPE', 'N/A'):.2f}%, "
                    f"TestLoss={eval_results.get('final_Test_Loss_Scaled', 'N/A'):.6f}")
    except Exception as e:
        logger.error(f"Error during final model evaluation: {e}", exc_info=True)
        return


    # --- 10. Plot Final Forecasts ---
    try:
    # Get data directly from the eval_results dictionary
        predictions_list = eval_results.get('predictions')
        actuals_list = eval_results.get('actuals')
        # Get the reconstructed DatetimeIndex object
        prediction_index = eval_results.get('prediction_index_obj')

        if predictions_list and actuals_list and isinstance(prediction_index, pd.DatetimeIndex):
            # Convert lists back to numpy arrays if needed for slicing etc.
            predictions_plot = np.array(predictions_list)
            actuals_plot = np.array(actuals_list)

            # Check length consistency one last time before creating DataFrame
            if len(prediction_index) == len(predictions_plot):
                logger.info(f"Creating results DataFrame for plotting with index length {len(prediction_index)}")
                results_plot_df = pd.DataFrame({
                    # Assuming single target for plot (index 0)
                    'Actual': actuals_plot[:, 0],
                    'Predicted': predictions_plot[:, 0]
                }, index=prediction_index) # Use the reconstructed index

                plot_forecasts(
                    results_df=results_plot_df,
                    target_col=args.target_col,
                    output_dir=output_dir,
                    prefix="final"
                    # sample_size can be adjusted via args if needed
                )
            else:
                # This log should now be less likely if index reconstruction is correct
                logger.warning(f"Length mismatch between reconstructed index ({len(prediction_index)}) "
                               f"and predictions ({len(predictions_plot)}). Skipping forecast plot.")

        elif not isinstance(prediction_index, pd.DatetimeIndex):
             logger.warning(f"Reconstructed prediction index is invalid or missing (type: {type(prediction_index)}). Skipping forecast plot.")
        else:
             logger.warning("No evaluation results (predictions/actuals) found in dictionary to plot forecasts.")

    except Exception as e:
        logger.error(f"Error during forecast plotting: {e}", exc_info=True)
        # Continue even if plotting fails


    # --- Workflow Complete ---
    workflow_duration = time.time() - workflow_start_time
    logger.info("=================================================")
    logger.info(f" TimeXer Workflow Completed @ {datetime.now()} ")
    logger.info(f" Total Execution Time: {workflow_duration:.2f} seconds ")
    logger.info("=================================================")


if __name__ == "__main__":
    # Execute the main workflow only when the script is run directly
    main()
