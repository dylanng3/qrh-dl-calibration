"""
training_utils.py
================
Training utilities for Quadratic Rough Heston model training.

Features:
- QRH-specific training configuration
- Learning rate scheduling (step decay as per specs)
- Early stopping and model checkpointing
- Training metrics and logging
- Data loading utilities for QRH dataset
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import datetime as dt

import numpy as np
import tensorflow as tf
import keras


# ----------------------------------------------------------------------
# QRH Training Configuration
# ----------------------------------------------------------------------

def get_qrh_training_config() -> Dict[str, Any]:
    """Get default training configuration for QRH model according to specs."""
    return {
        # Model architecture
        "input_dim": 15,
        "output_dim": 60,
        "hidden_layers": 7,
        "hidden_units": 25,
        "activation": "swish",
        "dropout_rate": 0.0,
        "use_batch_norm": False,
        
        # Training parameters according to QRH specs
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 100,
        "early_stopping_patience": 5,
        
        # Learning rate schedule: decay 1/2 every 10 epochs
        "lr_schedule": "step",
        "lr_decay_factor": 0.5,
        "lr_decay_epochs": 10,
        
        # Loss and metrics
        "loss": "mse",
        "metrics": ["mae", "mse"],
        
        # Data splits according to QRH specs
        "train_samples": 150000,
        "val_samples": 20000,
        "test_samples": 10000,
    }


# ----------------------------------------------------------------------
# Data Loading for QRH Dataset
# ----------------------------------------------------------------------

def load_qrh_dataset(data_dir: str, dataset_size: str = "150k") -> Tuple[
    Tuple[np.ndarray, np.ndarray],  # train
    Tuple[np.ndarray, np.ndarray],  # val  
    Tuple[np.ndarray, np.ndarray],  # test
    Tuple[Any, Any]  # scalers
]:
    """
    Load QRH dataset with train/validation/test splits.
    
    Args:
        data_dir: Path to data directory (e.g., "data/raw/data_150k")
        dataset_size: Size identifier (e.g., "150k")
    
    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test), (x_scaler, y_scaler))
    """
    data_path = Path(data_dir)
    
    # Load datasets
    train_data = np.load(data_path / f"train_{dataset_size}.npz")
    val_data = np.load(data_path / f"val_{dataset_size}.npz")
    test_data = np.load(data_path / f"test_{dataset_size}.npz")
    
    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]
    
    # Load scalers
    with open(data_path / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    with open(data_path / "y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)
    
    print(f"Loaded QRH dataset:")
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Val:   {X_val.shape[0]:,} samples") 
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print(f"  Input dim: {X_train.shape[1]}, Output dim: {y_train.shape[1]}")
    
    return ((X_train, y_train), (X_val, y_val), (X_test, y_test), (x_scaler, y_scaler))


def load_qrh_modular_dataset(data_dir: str) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                     Tuple[np.ndarray, np.ndarray], 
                                                     Tuple[Any, Any]]:
    """
    Load QRH dataset from modular X, y format.
    
    Args:
        data_dir: Directory containing train_X.npy, train_y.npy, val_X.npy, val_y.npy, scalers
        
    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (x_scaler, y_scaler))
    """
    data_path = Path(data_dir)
    
    # Check required files
    required_files = ["train_X.npy", "train_y.npy", "val_X.npy", "val_y.npy", 
                      "x_scaler.pkl", "y_scaler.pkl"]
    missing_files = [f for f in required_files if not (data_path / f).exists()]
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing modular files in {data_dir}: {missing_files}\n"
            f"Run converter first: python src/processing/converter.py --data_dir {data_dir}"
        )
    
    # Load data arrays
    X_train = np.load(data_path / "train_X.npy")
    y_train = np.load(data_path / "train_y.npy")
    X_val = np.load(data_path / "val_X.npy")
    y_val = np.load(data_path / "val_y.npy")
    
    # Load scalers
    with open(data_path / "x_scaler.pkl", 'rb') as f:
        x_scaler = pickle.load(f)
    with open(data_path / "y_scaler.pkl", 'rb') as f:
        y_scaler = pickle.load(f)
    
    print(f"Loaded QRH modular dataset:")
    print(f"  Train: X{X_train.shape}, y{y_train.shape}")
    print(f"  Val:   X{X_val.shape}, y{y_val.shape}")
    print(f"  Input dim: {X_train.shape[1]}, Output dim: {y_train.shape[1]}")
    
    return ((X_train, y_train), (X_val, y_val), (x_scaler, y_scaler))


# ----------------------------------------------------------------------
# Learning Rate Scheduling
# ----------------------------------------------------------------------

def create_step_lr_scheduler(
    initial_lr: float = 0.001,
    decay_factor: float = 0.5,
    decay_epochs: int = 10
) -> keras.callbacks.LearningRateScheduler:
    """
    Create step decay learning rate scheduler as per QRH specs:
    LR decreases by 1/2 every 10 epochs.
    """
    def schedule(epoch, lr):
        if epoch > 0 and epoch % decay_epochs == 0:
            new_lr = lr * decay_factor
            print(f"\nEpoch {epoch}: Reducing learning rate from {lr:.6f} to {new_lr:.6f}")
            return new_lr
        return lr
    
    return keras.callbacks.LearningRateScheduler(schedule, verbose=0)


# ----------------------------------------------------------------------
# Callbacks Setup
# ----------------------------------------------------------------------

def create_qrh_callbacks(
    model_save_path: str,
    log_dir: Optional[str] = None,
    patience: int = 5,
    monitor: str = "val_loss",
    lr_schedule_config: Optional[Dict] = None
) -> List[keras.callbacks.Callback]:
    """
    Create callbacks for QRH model training.
    
    Args:
        model_save_path: Path to save best model
        log_dir: Directory for TensorBoard logs
        patience: Early stopping patience
        monitor: Metric to monitor for early stopping
        lr_schedule_config: Learning rate schedule configuration
    
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    
    # Early stopping (5 epochs patience as per specs)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode='min'
    )
    callbacks.append(model_checkpoint)
    
    # Learning rate scheduler (step decay)
    if lr_schedule_config is None:
        lr_schedule_config = {"initial_lr": 0.001, "decay_factor": 0.5, "decay_epochs": 10}
    
    lr_scheduler = create_step_lr_scheduler(**lr_schedule_config)
    callbacks.append(lr_scheduler)
    
    # TensorBoard logging
    if log_dir:
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
    
    # Terminate on NaN
    callbacks.append(keras.callbacks.TerminateOnNaN())
    
    return callbacks


# ----------------------------------------------------------------------
# Training Utilities
# ----------------------------------------------------------------------

def train_qrh_model(
    model: keras.Model,
    data: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    config: Dict[str, Any],
    model_save_path: str,
    log_dir: Optional[str] = None,
    verbose: int = 1
) -> keras.callbacks.History:
    """
    Train QRH model with specified configuration.
    
    Args:
        model: Compiled QRH model
        data: ((X_train, y_train), (X_val, y_val))
        config: Training configuration
        model_save_path: Path to save best model
        log_dir: TensorBoard log directory
        verbose: Verbosity level
    
    Returns:
        Training history
    """
    (X_train, y_train), (X_val, y_val) = data
    
    # Create callbacks
    callbacks = create_qrh_callbacks(
        model_save_path=model_save_path,
        log_dir=log_dir,
        patience=config.get("early_stopping_patience", 5),
        lr_schedule_config={
            "initial_lr": config.get("learning_rate", 0.001),
            "decay_factor": config.get("lr_decay_factor", 0.5),
            "decay_epochs": config.get("lr_decay_epochs", 10)
        }
    )
    
    print(f"Starting QRH model training...")
    print(f"Train samples: {X_train.shape[0]:,}")
    print(f"Val samples: {X_val.shape[0]:,}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Max epochs: {config['epochs']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True
    )
    
    return history


def train_qrh_modular_model(
    model: keras.Model,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    config: Dict[str, Any],
    model_save_path: str,
    log_dir: Optional[str] = None,
    verbose: int = 1
) -> keras.callbacks.History:
    """
    Train QRH model using modular X, y format.
    
    Args:
        model: Compiled QRH model
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        config: Training configuration
        model_save_path: Path to save best model
        log_dir: TensorBoard log directory
        verbose: Verbosity level
        
    Returns:
        Training history
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    print(f"Starting QRH modular training...")
    print(f"Train samples: {len(X_train):,}")
    print(f"Val samples: {len(X_val):,}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Max epochs: {config['epochs']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    
    # Create callbacks
    callbacks = create_qrh_callbacks(
        model_save_path=model_save_path,
        log_dir=log_dir,
        patience=config["early_stopping_patience"],
        lr_schedule_config={
            "initial_lr": config["learning_rate"],
            "decay_factor": config.get("lr_decay_factor", 0.5),
            "decay_epochs": config.get("lr_decay_epochs", 10)
        }
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True
    )
    
    return history


# ----------------------------------------------------------------------
# Model Evaluation
# ----------------------------------------------------------------------

def evaluate_qrh_model(
    model: keras.Model,
    test_data: Tuple[np.ndarray, np.ndarray],
    y_scaler: Any,
    verbose: int = 1
) -> Dict[str, float]:
    """
    Evaluate QRH model on test data.
    
    Args:
        model: Trained QRH model
        test_data: (X_test, y_test)
        y_scaler: Output scaler for inverse transform
        verbose: Verbosity level
    
    Returns:
        Dictionary with evaluation metrics
    """
    X_test, y_test = test_data
    
    # Get predictions
    y_pred = model.predict(X_test, verbose=verbose)
    
    # Calculate metrics on normalized data
    mse_norm = np.mean((y_test - y_pred) ** 2)
    mae_norm = np.mean(np.abs(y_test - y_pred))
    
    # Calculate metrics on original scale
    y_test_orig = y_scaler.inverse_transform(y_test)
    y_pred_orig = y_scaler.inverse_transform(y_pred)
    
    mse_orig = np.mean((y_test_orig - y_pred_orig) ** 2)
    mae_orig = np.mean(np.abs(y_test_orig - y_pred_orig))
    rmse_orig = np.sqrt(mse_orig)
    
    # R² score
    ss_res = np.sum((y_test_orig - y_pred_orig) ** 2)
    ss_tot = np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    metrics = {
        "mse_normalized": mse_norm,
        "mae_normalized": mae_norm,
        "mse_original": mse_orig,
        "mae_original": mae_orig,
        "rmse_original": rmse_orig,
        "r2_score": r2_score,
        "test_samples": len(X_test)
    }
    
    if verbose:
        print(f"\nQRH Model Evaluation Results:")
        print(f"  Test samples: {metrics['test_samples']:,}")
        print(f"  MSE (normalized): {metrics['mse_normalized']:.6f}")
        print(f"  MAE (normalized): {metrics['mae_normalized']:.6f}")
        print(f"  MSE (original): {metrics['mse_original']:.6f}")
        print(f"  MAE (original): {metrics['mae_original']:.6f}")
        print(f"  RMSE (original): {metrics['rmse_original']:.6f}")
        print(f"  R² score: {metrics['r2_score']:.6f}")
    
    return metrics


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def save_training_config(config: Dict[str, Any], save_path: str) -> None:
    """Save training configuration to JSON file."""
    import json
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {save_path}")


def create_experiment_dir(base_dir: str = "experiments") -> str:
    """Create timestamped experiment directory."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"qrh_experiment_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    return str(exp_dir)
