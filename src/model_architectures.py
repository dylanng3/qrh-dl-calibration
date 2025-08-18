"""
advanced_model_architectures.py
===============================
Advanced QRH Model Architectures with PCA-head, Residual blocks, and Sobolev regularization

Key improvements:
1. PCA-head: Predict K=12 PCA coefficients instead of 60 IV points directly
2. Residual-MLP: Skip connections with LayerNorm for better gradient flow  
3. Huber loss: Robust to IV outliers/extreme values
4. Sobolev penalty: Enforce smoothness along strike/maturity dimensions
5. Weighted loss: Emphasize important regions (ATM, short-tenor)

Usage:
    from src.advanced_model_architectures import build_resmlp_pca_model, compile_advanced_qrh_model
    model = build_resmlp_pca_model(K=12, width=128, n_blocks=8)
    model = compile_advanced_qrh_model(model, pca_components, learning_rate=1e-3)
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, Any


# ----------------------------------------------------------------------
# PCA Components Fitting
# ----------------------------------------------------------------------

def fit_pca_components(y_train_raw: np.ndarray, K: int = 12, 
                      use_scaler: bool = True) -> Dict[str, Any]:
    """
    Fit PCA components for IV surface dimensionality reduction.
    
    Args:
        y_train_raw: Raw IV training data (N_train, 60)  
        K: Number of PCA components (typically 10-14)
        use_scaler: Whether to standardize before PCA
        
    Returns:
        Dictionary containing PCA components, mean, scaler, and explained variance
    """
    print(f"Fitting PCA with K={K} components on IV surface data...")
    
    # Optional standardization
    if use_scaler:
        y_scaler = StandardScaler().fit(y_train_raw)
        Y = y_scaler.transform(y_train_raw)
    else:
        y_scaler = None
        Y = y_train_raw.copy()
    
    # Fit PCA
    pca = PCA(n_components=K, svd_solver="full")
    pca.fit(Y)
    
    # Extract components
    P = pca.components_.T  # (60, K)
    mu = pca.mean_         # (60,)
    explained_var_ratio = pca.explained_variance_ratio_
    total_explained = explained_var_ratio.sum()
    
    print(f"PCA explained variance: {total_explained:.4f} ({100*total_explained:.2f}%)")
    print(f"Top 5 components: {explained_var_ratio[:5]}")
    
    return {
        'pca': pca,
        'P': P,              # Components matrix (60, K)  
        'mu': mu,            # Mean vector (60,)
        'y_scaler': y_scaler,
        'explained_variance_ratio': explained_var_ratio,
        'total_explained': total_explained,
        'K': K
    }


def pca_transform_targets(y_data: np.ndarray, pca_info: Dict[str, Any]) -> np.ndarray:
    """Transform IV targets to PCA coefficient space."""
    if pca_info['y_scaler'] is not None:
        Y = pca_info['y_scaler'].transform(y_data)
    else:
        Y = y_data.copy()
    
    # Transform to PCA space: coeffs = (Y - mu) @ P
    Y_centered = Y - pca_info['mu']
    coeffs = Y_centered @ pca_info['P']  # (N, K)
    return coeffs


def pca_inverse_transform(coeffs: np.ndarray, pca_info: Dict[str, Any]) -> np.ndarray:
    """Reconstruct IV surface from PCA coefficients."""
    # Reconstruct: Y = mu + coeffs @ P.T  
    Y_reconstructed = pca_info['mu'] + coeffs @ pca_info['P'].T
    
    if pca_info['y_scaler'] is not None:
        y_reconstructed = pca_info['y_scaler'].inverse_transform(Y_reconstructed)
    else:
        y_reconstructed = Y_reconstructed
    
    return y_reconstructed


# ----------------------------------------------------------------------
# Advanced Residual MLP with PCA-head  
# ----------------------------------------------------------------------

def build_resmlp_pca_model(
    input_dim: int = 15,
    K: int = 12,
    width: int = 128,
    n_blocks: int = 8,
    neck_width: int = 64,
    dropout_rate: float = 0.0,
    l2_reg: float = 1e-5,
    name: str = "QRH_ResMLP_PCA"
) -> keras.Model:
    """
    Build Residual MLP with PCA-head for QRH IV surface prediction.
    
    Architecture:
    1. Encoder: Dense(width) -> SiLU
    2. N Residual blocks: Dense(width, no_bias) -> LayerNorm -> SiLU -> Add(skip)
    3. Neck: Dense(neck_width) -> SiLU  
    4. PCA-head: Dense(K, linear) - predicts PCA coefficients
    
    Args:
        input_dim: Input dimensions (15: 5 model params + 10 factors)
        K: Number of PCA components to predict
        width: Hidden layer width for residual blocks
        n_blocks: Number of residual blocks
        neck_width: Width before final PCA head
        dropout_rate: Dropout rate (0.0 for no dropout)
        l2_reg: L2 regularization strength
        name: Model name
        
    Returns:
        Keras model that outputs PCA coefficients
    """
    
    # Regularizer
    reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    # Input
    x_in = keras.Input(shape=(input_dim,), name="input_params")
    
    # Encoder 
    h = layers.Dense(width, kernel_regularizer=reg, name="encoder")(x_in)
    h = layers.Activation('swish')(h)
    
    if dropout_rate > 0:
        h = layers.Dropout(dropout_rate, name="encoder_dropout")(h)
    
    # Residual blocks
    for i in range(n_blocks):
        # Skip connection
        residual = h
        
        # Transform
        h = layers.Dense(width, use_bias=False, kernel_regularizer=reg, 
                        name=f"res_block_{i+1}_dense")(h)
        h = layers.LayerNormalization(name=f"res_block_{i+1}_ln")(h)
        h = layers.Activation('swish')(h)
        
        if dropout_rate > 0:
            h = layers.Dropout(dropout_rate, name=f"res_block_{i+1}_dropout")(h)
        
        # Add skip connection
        h = layers.Add(name=f"res_block_{i+1}_add")([residual, h])
    
    # Neck
    h = layers.Dense(neck_width, kernel_regularizer=reg, name="neck")(h) 
    h = layers.Activation('swish')(h)
    
    if dropout_rate > 0:
        h = layers.Dropout(dropout_rate, name="neck_dropout")(h)
    
    # PCA-head (linear output)
    pca_coeffs = layers.Dense(K, activation='linear', kernel_regularizer=reg,
                             name="pca_coeffs")(h)
    
    model = keras.Model(inputs=x_in, outputs=pca_coeffs, name=name)
    return model


# ----------------------------------------------------------------------
# Advanced Loss Functions
# ----------------------------------------------------------------------

def create_pca_reconstruction_layer(pca_info: Dict[str, Any]) -> layers.Layer:
    """Create TensorFlow layer for PCA reconstruction."""
    
    P_tf = tf.constant(pca_info['P'].T, dtype=tf.float32)  # (K, 60)
    mu_tf = tf.constant(pca_info['mu'], dtype=tf.float32)   # (60,)
    
    class PCAReconstructionLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
        def call(self, coeffs):
            # coeffs: (batch_size, K)
            # Reconstruct: Y = mu + coeffs @ P.T
            y_hat = tf.linalg.matmul(coeffs, P_tf) + mu_tf  # (batch_size, 60)
            return y_hat
    
    return PCAReconstructionLayer()


def huber_loss(y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 0.015) -> tf.Tensor:
    """Huber loss - robust to outliers."""
    error = y_pred - y_true
    abs_error = tf.abs(error)
    quadratic = tf.where(abs_error <= delta, 
                        0.5 * tf.square(error),
                        delta * (abs_error - 0.5 * delta))
    return tf.reduce_mean(quadratic)


def weighted_huber_loss(y_true: tf.Tensor, y_pred: tf.Tensor, 
                       weights: Optional[tf.Tensor] = None, delta: float = 0.015) -> tf.Tensor:
    """Weighted Huber loss for emphasizing important IV regions."""
    error = y_pred - y_true
    abs_error = tf.abs(error)
    quadratic = tf.where(abs_error <= delta,
                        0.5 * tf.square(error), 
                        delta * (abs_error - 0.5 * delta))
    
    if weights is not None:
        # Normalize weights to have mean=1
        weights_norm = weights / tf.reduce_mean(weights)
        quadratic = quadratic * weights_norm
    
    return tf.reduce_mean(quadratic)


def create_sobolev_penalty(grid_shape: Tuple[int, int] = (4, 15),
                          alpha: float = 0.1, beta: float = 0.05) -> callable:
    """
    Create Sobolev penalty for smoothness along strike (k) and maturity (tau) dimensions.
    
    Args:
        grid_shape: (n_maturities, n_strikes) - typically (4, 15)  
        alpha: Weight for strike smoothness
        beta: Weight for maturity smoothness
        
    Returns:
        Penalty function
    """
    n_tau, n_k = grid_shape
    n_total = n_tau * n_k  # Should be 60
    
    # Create finite difference matrices for strike direction (within each maturity)
    D_k_list = []
    for i in range(n_tau):  # For each maturity
        for j in range(n_k - 1):  # Adjacent strikes
            row = np.zeros(n_total)
            idx1 = i * n_k + j
            idx2 = i * n_k + j + 1
            row[idx1] = -1.0
            row[idx2] = 1.0
            D_k_list.append(row)
    
    D_k = np.array(D_k_list)  # (n_edges_k, 60)
    
    # Create finite difference matrices for maturity direction (within each strike)  
    D_tau_list = []
    for j in range(n_k):  # For each strike
        for i in range(n_tau - 1):  # Adjacent maturities
            row = np.zeros(n_total)
            idx1 = i * n_k + j  
            idx2 = (i + 1) * n_k + j
            row[idx1] = -1.0
            row[idx2] = 1.0
            D_tau_list.append(row)
    
    D_tau = np.array(D_tau_list)  # (n_edges_tau, 60)
    
    # Convert to TensorFlow constants
    D_k_tf = tf.constant(D_k, dtype=tf.float32)
    D_tau_tf = tf.constant(D_tau, dtype=tf.float32) 
    
    def sobolev_penalty(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """Compute Sobolev penalty for surface smoothness."""
        # Compute differences for predicted surface
        dk_pred = tf.linalg.matmul(y_pred, D_k_tf, transpose_b=True)
        dtau_pred = tf.linalg.matmul(y_pred, D_tau_tf, transpose_b=True)
        
        # Compute differences for true surface  
        dk_true = tf.linalg.matmul(y_true, D_k_tf, transpose_b=True)
        dtau_true = tf.linalg.matmul(y_true, D_tau_tf, transpose_b=True)
        
        # Penalties
        strike_penalty = alpha * tf.reduce_mean(tf.square(dk_pred - dk_true))
        maturity_penalty = beta * tf.reduce_mean(tf.square(dtau_pred - dtau_true))
        
        return strike_penalty + maturity_penalty
    
    return sobolev_penalty


def create_simple_pca_loss(pca_info: Dict[str, Any]) -> callable:
    """
    Create simple MSE loss for PCA coefficients training.
    Used for easier training and testing.
    
    Args:
        pca_info: PCA information dict
        
    Returns:
        Simple loss function for PCA coefficients
    """
    
    def simple_pca_loss(y_true_pca: tf.Tensor, y_pred_pca: tf.Tensor) -> tf.Tensor:
        """
        Simple MSE loss for PCA coefficients.
        
        Args:
            y_true_pca: True PCA coefficients (batch_size, K)
            y_pred_pca: Predicted PCA coefficients (batch_size, K)
            
        Returns:
            MSE loss
        """
        return tf.reduce_mean(tf.square(y_true_pca - y_pred_pca))
    
    return simple_pca_loss


def create_advanced_loss_function(pca_info: Dict[str, Any],
                                weights: Optional[np.ndarray] = None,
                                delta: float = 0.015,
                                alpha: float = 0.1, 
                                beta: float = 0.05,
                                grid_shape: Tuple[int, int] = (4, 15)) -> callable:
    """
    Create advanced loss function combining PCA reconstruction, Huber loss, and Sobolev penalty.
    
    Args:
        pca_info: PCA information dict from fit_pca_components
        weights: Optional weights for different IV points (60,)
        delta: Huber loss threshold  
        alpha, beta: Sobolev penalty weights for strike/maturity smoothness
        grid_shape: IV surface grid shape
        
    Returns:
        Loss function compatible with Keras
    """
    
    # Create PCA reconstruction
    pca_layer = create_pca_reconstruction_layer(pca_info)
    
    # Create Sobolev penalty
    sobolev_fn = create_sobolev_penalty(grid_shape, alpha, beta)
    
    # Prepare weights
    if weights is not None:
        weights_tf = tf.constant(weights, dtype=tf.float32)
    else:
        weights_tf = None
    
    def advanced_loss(y_true: tf.Tensor, pca_coeffs_pred: tf.Tensor) -> tf.Tensor:
        """
        Advanced loss function.
        
        Args:
            y_true: True IV surface (batch_size, 60)
            pca_coeffs_pred: Predicted PCA coefficients (batch_size, K)
            
        Returns:
            Total loss
        """
        # Reconstruct IV surface from PCA coefficients
        y_pred = pca_layer(pca_coeffs_pred)  # (batch_size, 60)
        
        # Base loss: Weighted Huber
        base_loss = weighted_huber_loss(y_true, y_pred, weights_tf, delta)
        
        # Sobolev smoothness penalty
        smoothness_loss = sobolev_fn(y_pred, y_true)
        
        return base_loss + smoothness_loss
    
    return advanced_loss


# ----------------------------------------------------------------------
# Model Compilation and Training
# ----------------------------------------------------------------------

def compile_advanced_qrh_model(model: keras.Model,
                              pca_info: Dict[str, Any],
                              learning_rate: float = 1e-3,
                              weights: Optional[np.ndarray] = None,
                              loss_params: Optional[Dict[str, float]] = None) -> keras.Model:
    """
    Compile advanced QRH model with custom loss function.
    
    Args:
        model: Built QRH model
        pca_info: PCA information dict
        learning_rate: Learning rate for optimizer
        weights: Optional weights for IV surface points
        loss_params: Loss function parameters
        
    Returns:
        Compiled model
    """
    
    # Default loss parameters
    if loss_params is None:
        loss_params = {
            'delta': 0.015,
            'alpha': 0.1, 
            'beta': 0.05,
            'grid_shape': (4, 15)
        }
    
    # Create loss function
    loss_fn = create_advanced_loss_function(
        pca_info=pca_info,
        weights=weights,
        **loss_params
    )
    
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['mae']  # Additional metrics
    )
    
    return model


def create_training_callbacks(patience: int = 10, 
                            reduce_lr_patience: int = 5,
                            min_lr: float = 1e-6,
                            factor: float = 0.5) -> list:
    """Create training callbacks for advanced model."""
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        ),
    ]
    
    return callbacks


# ----------------------------------------------------------------------
# Model Evaluation and Diagnostics
# ----------------------------------------------------------------------

def evaluate_by_buckets(y_true: np.ndarray, y_pred: np.ndarray,
                       grid_shape: Tuple[int, int] = (4, 15)) -> Dict[str, float]:
    """
    Evaluate model performance by different IV surface buckets.
    
    Args:
        y_true: True IV values (N, 60)
        y_pred: Predicted IV values (N, 60)  
        grid_shape: Grid shape (n_maturities, n_strikes)
        
    Returns:
        Dictionary of RMSE values for different buckets
    """
    
    n_tau, n_k = grid_shape
    center_k = n_k // 2  # ATM strike index (typically 7)
    
    # Reshape to (N, n_tau, n_k)
    y_true_grid = y_true.reshape(-1, n_tau, n_k)
    y_pred_grid = y_pred.reshape(-1, n_tau, n_k)
    
    results = {}
    
    # Overall RMSE
    results['overall'] = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # ATM (center strikes across all maturities)
    atm_true = y_true_grid[:, :, center_k].flatten()
    atm_pred = y_pred_grid[:, :, center_k].flatten()
    results['atm'] = np.sqrt(np.mean((atm_pred - atm_true) ** 2))
    
    # OTM puts (low strikes)
    otm_put_true = y_true_grid[:, :, :3].flatten()
    otm_put_pred = y_pred_grid[:, :, :3].flatten()
    results['otm_put'] = np.sqrt(np.mean((otm_put_pred - otm_put_true) ** 2))
    
    # OTM calls (high strikes)  
    otm_call_true = y_true_grid[:, :, -3:].flatten()
    otm_call_pred = y_pred_grid[:, :, -3:].flatten()
    results['otm_call'] = np.sqrt(np.mean((otm_call_pred - otm_call_true) ** 2))
    
    # Short tenor (first 2 maturities)
    short_true = y_true_grid[:, :2, :].flatten()
    short_pred = y_pred_grid[:, :2, :].flatten()  
    results['short_tenor'] = np.sqrt(np.mean((short_pred - short_true) ** 2))
    
    # Long tenor (last 2 maturities)
    long_true = y_true_grid[:, -2:, :].flatten()
    long_pred = y_pred_grid[:, -2:, :].flatten()
    results['long_tenor'] = np.sqrt(np.mean((long_pred - long_true) ** 2))
    
    return results


# ----------------------------------------------------------------------
# Main Functions for Easy Usage
# ----------------------------------------------------------------------

def build_and_compile_advanced_qrh_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_params: Optional[Dict[str, Any]] = None,
    pca_params: Optional[Dict[str, Any]] = None,
    loss_params: Optional[Dict[str, Any]] = None,
    learning_rate: float = 1e-3,
    weights: Optional[np.ndarray] = None
) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Build and compile advanced QRH model with PCA-head in one step.
    
    Args:
        X_train: Training input data (N, 15)
        y_train: Training target data (N, 60) - raw IV values
        model_params: Model architecture parameters
        pca_params: PCA fitting parameters  
        loss_params: Loss function parameters
        learning_rate: Learning rate
        weights: Optional weights for IV surface points
        
    Returns:
        Tuple of (compiled_model, pca_info)
    """
    
    # Default parameters
    if model_params is None:
        model_params = {
            'K': 12,
            'width': 128, 
            'n_blocks': 8,
            'neck_width': 64,
            'dropout_rate': 0.0,
            'l2_reg': 1e-5
        }
    
    if pca_params is None:
        pca_params = {
            'K': model_params['K'],
            'use_scaler': True
        }
    
    print("Step 1: Fitting PCA components...")
    pca_info = fit_pca_components(y_train, **pca_params)
    
    print("Step 2: Building Residual MLP model...")  
    model = build_resmlp_pca_model(
        input_dim=X_train.shape[1],
        **model_params
    )
    
    print("Step 3: Compiling with advanced loss function...")
    model = compile_advanced_qrh_model(
        model=model,
        pca_info=pca_info,
        learning_rate=learning_rate,
        weights=weights,
        loss_params=loss_params
    )
    
    print(f"Model summary:")
    print(f"  - Parameters: {model.count_params():,}")
    print(f"  - PCA components: {pca_info['K']} (explains {pca_info['total_explained']:.3f} variance)")
    print(f"  - Architecture: {model_params['n_blocks']} residual blocks × {model_params['width']} units")
    
    return model, pca_info


if __name__ == "__main__":
    # Example usage
    print("Advanced QRH Model Architectures")
    print("This module provides PCA-head Residual MLP with Sobolev regularization")
    print("Import and use build_and_compile_advanced_qrh_model() for easy setup")
