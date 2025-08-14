"""
train_qrh_model.py
================
Main training script for Quadratic Rough Heston model.

Usage:
    python train_qrh_model.py --data_dir data/raw/data_0k --epochs 50 --batch_size 64
    python train_qrh_model.py --data_dir data/raw/data_150k --epochs 100 --batch_size 128 --test_run

Features:
- Complete QRH model training pipeline
- Automatic experiment directory creation
- Model checkpointing and logging
- Comprehensive evaluation and reporting
- Support for different dataset sizes
"""

import argparse
import os
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import tensorflow as tf
import keras

from src.model_architectures import create_qrh_model, compile_qrh_model, print_model_summary
from src.training_utils_qrh import (
    get_qrh_training_config,
    load_qrh_dataset,
    train_qrh_model,
    evaluate_qrh_model,
    save_training_config,
    create_experiment_dir
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Quadratic Rough Heston model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory (e.g., data/raw/data_150k)")
    parser.add_argument("--dataset_size", type=str, default=None,
                        help="Dataset size identifier (auto-detected from data_dir if not provided)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    
    # Model arguments
    parser.add_argument("--hidden_layers", type=int, default=7,
                        help="Number of hidden layers")
    parser.add_argument("--hidden_units", type=int, default=25,
                        help="Units per hidden layer")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--use_batch_norm", action="store_true",
                        help="Use batch normalization")
    
    # Experiment arguments
    parser.add_argument("--experiment_dir", type=str, default="experiments",
                        help="Base directory for experiments")
    parser.add_argument("--model_name", type=str, default="qrh_mlp",
                        help="Model name for saving")
    parser.add_argument("--test_run", action="store_true",
                        help="Run quick test with reduced epochs")
    
    # Hardware arguments
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    
    return parser.parse_args()


def setup_gpu(gpu_id: int, mixed_precision: bool = False):
    """Setup GPU configuration."""
    # Set GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Use specific GPU
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            print(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            print("Falling back to CPU")
    else:
        print("No GPUs available, using CPU")
    
    # Mixed precision
    if mixed_precision:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")


def auto_detect_dataset_size(data_dir: str) -> str:
    """Auto-detect dataset size from directory name."""
    data_path = Path(data_dir)
    dir_name = data_path.name
    
    # Extract size from directory name (e.g., data_150k -> 150k)
    if "_" in dir_name:
        size_part = dir_name.split("_")[-1]
        return size_part
    
    # Fallback: look for files and extract size
    npz_files = list(data_path.glob("train_*.npz"))
    if npz_files:
        filename = npz_files[0].stem
        if "_" in filename:
            return filename.split("_")[-1]
    
    raise ValueError(f"Cannot auto-detect dataset size from {data_dir}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup hardware
    setup_gpu(args.gpu, args.mixed_precision)
    
    # Auto-detect dataset size if not provided
    if args.dataset_size is None:
        args.dataset_size = auto_detect_dataset_size(args.data_dir)
        print(f"Auto-detected dataset size: {args.dataset_size}")
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.experiment_dir)
    
    # Get base configuration
    config = get_qrh_training_config()
    
    # Override with command line arguments
    config.update({
        "epochs": args.epochs if not args.test_run else min(10, args.epochs),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.patience,
        "hidden_layers": args.hidden_layers,
        "hidden_units": args.hidden_units,
        "dropout_rate": args.dropout_rate,
        "use_batch_norm": args.use_batch_norm,
        "data_dir": args.data_dir,
        "dataset_size": args.dataset_size,
        "model_name": args.model_name,
        "mixed_precision": args.mixed_precision,
        "test_run": args.test_run
    })
    
    print(f"\\n{'='*60}")
    print(f"QUADRATIC ROUGH HESTON MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Dataset: {args.data_dir}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Model: {config['hidden_layers']} layers × {config['hidden_units']} units")
    print(f"Experiment dir: {exp_dir}")
    if args.test_run:
        print(f"*** TEST RUN MODE ***")
    print(f"{'='*60}\\n")
    
    # Save configuration
    config_path = Path(exp_dir) / "config.json"
    save_training_config(config, str(config_path))
    
    # Load dataset
    print("Loading QRH dataset...")
    try:
        (train_data, val_data, test_data, scalers) = load_qrh_dataset(
            args.data_dir, args.dataset_size
        )
        x_scaler, y_scaler = scalers
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Make sure the dataset exists at {args.data_dir}")
        return 1
    
    # Create model
    print("\\nCreating QRH model...")
    model = create_qrh_model(
        model_type="qrh_mlp",
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        hidden_layers=config["hidden_layers"],
        hidden_units=config["hidden_units"],
        activation=config["activation"],
        dropout_rate=config["dropout_rate"],
        use_batch_norm=config["use_batch_norm"],
        name=f"{config['model_name']}_{args.dataset_size}"
    )
    
    # Compile model
    model = compile_qrh_model(
        model,
        learning_rate=config["learning_rate"],
        optimizer=config["optimizer"],
        loss=config["loss"],
        metrics=config["metrics"]
    )
    
    # Print model summary
    print_model_summary(model)
    
    # Setup paths
    model_save_path = Path(exp_dir) / "models" / f"{config['model_name']}_best.keras"
    log_dir = Path(exp_dir) / "logs"
    
    # Train model
    print(f"\\nStarting training...")
    try:
        history = train_qrh_model(
            model=model,
            data=(train_data, val_data),
            config=config,
            model_save_path=str(model_save_path),
            log_dir=str(log_dir),
            verbose=1
        )
        
        print(f"\\n✅ Training completed!")
        print(f"Best model saved to: {model_save_path}")
        
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        return 1
    
    # Load best model for evaluation
    print(f"\\nLoading best model for evaluation...")
    try:
        best_model = keras.models.load_model(str(model_save_path))
    except Exception as e:
        print(f"Warning: Could not load best model ({e}), using current model")
        best_model = model
    
    # Evaluate model
    print(f"\\nEvaluating model on test set...")
    metrics = evaluate_qrh_model(
        model=best_model,
        test_data=test_data,
        y_scaler=y_scaler,
        verbose=1
    )
    
    # Save results
    results = {
        "config": config,
        "metrics": metrics,
        "training_history": {
            "loss": history.history.get("loss", []),
            "val_loss": history.history.get("val_loss", []),
            "mae": history.history.get("mae", []),
            "val_mae": history.history.get("val_mae", [])
        }
    }
    
    results_path = Path(exp_dir) / "results" / "training_results.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n✅ Results saved to: {results_path}")
    print(f"\\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Final validation loss: {min(history.history.get('val_loss', [float('inf')])):.6f}")
    print(f"Test R² score: {metrics['r2_score']:.6f}")
    print(f"Test RMSE: {metrics['rmse_original']:.6f}")
    print(f"Model parameters: {best_model.count_params():,}")
    print(f"Training epochs: {len(history.history.get('loss', []))}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
