#!/usr/bin/env python3
"""
training.py
===========
Training script for advanced Quantitative Risk Heston (QRH) models with PCA-head, residual blocks, and Sobolev regularization.

Features:
- Supports modular and .npz data formats
- PCA-head for dimensionality reduction of targets
- Residual MLP architecture with configurable blocks and width
- Sobolev regularization for smoothness
- Early stopping, learning rate scheduling, and model checkpointing
- Integrated TensorBoard logging (logs in reports/tensorboard/)

Usage:
    python training.py --data_size 100k --epochs 200 --format modular --pca_components 30

Arguments:
    --data_size      Dataset size to use (e.g., 5k, 100k, 150k, ...)
    --format         Data format: npz or modular (default: modular)
    --epochs         Number of training epochs
    --batch_size     Batch size for training
    --learning_rate  Learning rate
    --patience       Early stopping patience
    --model_name     Model name for saving
    --pca_components Number of PCA components (K)
    --width          Hidden layer width
    --n_blocks       Number of residual blocks
    --dropout_rate   Dropout rate
    --l2_reg         L2 regularization strength
    --huber_delta    Huber loss threshold
    --sobolev_alpha  Strike smoothness weight
    --sobolev_beta   Maturity smoothness weight


All models and logs are saved in the appropriate experiment and report directories.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import datetime
import pickle
import random
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def main():
    """Main training function for advanced QRH model."""
    parser = argparse.ArgumentParser(description="Advanced QRH Model Training")
    
    # Data arguments
    parser.add_argument("--data_size", type=str, required=True,
                        help="Dataset size to use (e.g., 0k, 5k, 100k, 150k, ...)")
    parser.add_argument("--format", type=str, default="modular",
                        choices=["npz", "modular"], 
                        help="Data format: npz or modular (default: modular)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--model_name", type=str, default="qrh_advanced",
                        help="Model name for saving")

    
    # Advanced model parameters
    parser.add_argument("--pca_components", type=int, default=12,
                        help="Number of PCA components (K)")
    parser.add_argument("--width", type=int, default=128,
                        help="Hidden layer width")
    parser.add_argument("--n_blocks", type=int, default=8,
                        help="Number of residual blocks")
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--l2_reg", type=float, default=1e-5,
                        help="L2 regularization strength")
    
    # Loss function parameters
    parser.add_argument("--huber_delta", type=float, default=0.015,
                        help="Huber loss threshold")
    parser.add_argument("--sobolev_alpha", type=float, default=0.1,
                        help="Strike smoothness weight")
    parser.add_argument("--sobolev_beta", type=float, default=0.05,
                        help="Maturity smoothness weight")
    parser.add_argument("--otm_put_weight", type=float, default=2.0,
                        help="Weight for OTM Put region")
    
    args = parser.parse_args()
    
    # Construct data directory path
    data_dir = f"data/raw/data_{args.data_size}"
    
    print("Starting Advanced QRH Training...")
    print(f"Dataset: {data_dir}")
    print(f"Format: {args.format}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"PCA components: {args.pca_components}")
    print(f"Architecture: {args.n_blocks} residual blocks × {args.width} units")

    
    # Setup paths
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    
    # Add to Python path
    sys.path.insert(0, str(src_path))
    
    # Import after setting path
    try:
        print("Importing dependencies...")
        import tensorflow as tf
        print("  tensorflow")
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        import keras
        print("  keras")
        
        from src.model_architectures import (
            build_resmlp_pca_model,
            fit_pca_components,
            pca_transform_targets,
            pca_inverse_transform,
            evaluate_by_buckets,
            create_training_callbacks,
            compile_advanced_qrh_model
        )
        print("  advanced_model_architectures")
        
    except ImportError as e:
        print(f"Import error: {e}")
        return 1
    
    # Check if dataset exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Dataset directory not found: {data_dir}")
        print("Available datasets:")
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            for d in raw_dir.glob("data_*"):
                print(f"  - {d.name}")
        return 1
    
    # Load dataset
    print("Loading dataset...")
    try:
        if args.format == "modular":
            # Load modular format
            X_train = np.load(data_path / "train_X.npy")
            y_train = np.load(data_path / "train_y.npy")
            X_val = np.load(data_path / "val_X.npy")  
            y_val = np.load(data_path / "val_y.npy")
            
            # Load scalers
            with open(data_path / "x_scaler.pkl", 'rb') as f:
                x_scaler = pickle.load(f)
            with open(data_path / "y_scaler.pkl", 'rb') as f:
                y_scaler = pickle.load(f)
                
        else:  # NPZ format
            train_data = np.load(data_path / f"train_{args.data_size}.npz")
            val_data = np.load(data_path / f"val_{args.data_size}.npz") 
            
            X_train, y_train = train_data['X'], train_data['y']
            X_val, y_val = val_data['X'], val_data['y']
            
            # Load scalers  
            with open(data_path / "x_scaler.pkl", 'rb') as f:
                x_scaler = pickle.load(f)
            with open(data_path / "y_scaler.pkl", 'rb') as f:
                y_scaler = pickle.load(f)
        
        print(f"  Train samples: {len(X_train)}")
        print(f"  Val samples: {len(X_val)}")
        print(f"  Input dimensions: {X_train.shape[1]}")
        print(f"  Output dimensions: {y_train.shape[1]}")
            
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return 1
    
    # Get raw (unscaled) y data for PCA fitting
    print("Preparing data for PCA...")
    try:
        y_train_raw = y_scaler.inverse_transform(y_train)
        y_val_raw = y_scaler.inverse_transform(y_val)
        print(f"  Raw IV range: [{y_train_raw.min():.4f}, {y_train_raw.max():.4f}]")
    except Exception as e:
        print(f"Failed to prepare raw data: {e}")
        return 1
    
    # Model parameters
    model_params = {
        'K': args.pca_components,
        'width': args.width,
        'n_blocks': args.n_blocks,
        'neck_width': 64,
        'dropout_rate': args.dropout_rate,
        'l2_reg': args.l2_reg
    }
    
    # PCA parameters  
    pca_params = {
        'K': args.pca_components,
        'use_scaler': True
    }
    
    # Loss parameters
    loss_params = {
        'delta': args.huber_delta,
        'alpha': args.sobolev_alpha,
        'beta': args.sobolev_beta,
        'grid_shape': (4, 15)  # Standard QRH IV surface grid
    }
    
    # Build and compile model (using simple MSE for PCA coefficients)
    print("Building advanced QRH model...")
    try:
        # Build model architecture
        model_params = {
            'K': args.pca_components,
            'width': args.width,
            'n_blocks': args.n_blocks,
            'neck_width': 64,
            'dropout_rate': args.dropout_rate,
            'l2_reg': args.l2_reg
        }
        
        # Fit PCA components first
        pca_info = fit_pca_components(y_train_raw, K=args.pca_components, use_scaler=True)
        print(f"  PCA explained variance: {pca_info['total_explained']:.4f}")
        
        # Build model
        model = build_resmlp_pca_model(
            input_dim=X_train.shape[1],
            **model_params
        )
        
        # Advanced compilation with weighted loss for OTM Put improvement
        model = compile_advanced_qrh_model(
            model=model,
            pca_info=pca_info,
            learning_rate=args.learning_rate,
            otm_put_weight=args.otm_put_weight,
            loss_params={
                'delta': args.huber_delta,
                'alpha': args.sobolev_alpha,
                'beta': args.sobolev_beta,
                'grid_shape': (10, 6),  # Assuming 10x6 grid for 60 IV points
                'otm_put_weight': args.otm_put_weight
            }
        )
        
        total_params = model.count_params()
        print(f"  Model created with {total_params:,} parameters")
        print(f"  PCA components: {pca_info['K']} (explains {pca_info['total_explained']:.3f} variance)")
        print(f"  OTM Put weight: {args.otm_put_weight}")
        print("  Using advanced loss function (Huber + Sobolev + weighted)")
        
    except Exception as e:
        print(f"Failed to create model: {e}")
        return 1
    
    # Transform targets to PCA space
    print("Transforming targets to PCA coefficient space...")  
    try:
        y_train_pca = pca_transform_targets(y_train_raw, pca_info)
        y_val_pca = pca_transform_targets(y_val_raw, pca_info)
        
        print(f"  PCA coefficients shape: train{y_train_pca.shape}, val{y_val_pca.shape}")
        print(f"  PCA explained variance: {pca_info['total_explained']:.4f}")
        
    except Exception as e:
        print(f"Failed to transform targets: {e}")
        return 1
    
    # Create experiment directory
    exp_dir = Path("experiments") / f"advanced_qrh_{args.data_size}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    
    # Setup model saving (weights only to avoid custom loss serialization issues)
    weights_save_path = exp_dir / f"{args.model_name}_{args.data_size}.weights.h5"
    model_save_path = exp_dir / f"{args.model_name}_{args.data_size}.keras"
    
    # Training configuration
    epochs = args.epochs
    
    print(f"Training configuration:")
    print(f"  - Dataset size: {args.data_size}")
    print(f"  - Format: {args.format}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - PCA components: {args.pca_components}")
    print(f"  - Huber delta: {args.huber_delta}")
    print(f"  - Sobolev weights: α={args.sobolev_alpha}, β={args.sobolev_beta}")
    
    # Create callbacks
    callbacks = create_training_callbacks(
        patience=args.patience,
        reduce_lr_patience=5,
        min_lr=1e-6,
        factor=0.5
    )
    
    # Add model checkpoint callback (save weights only to avoid custom loss serialization)
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            str(weights_save_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    )

    # Add TensorBoard callback
    log_dir = os.path.join("reports", "tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks.append(
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    )
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Train model
    print("Starting training...")
    try:
        history = model.fit(
            X_train, y_train_pca,  # Note: using PCA coefficients as targets
            batch_size=args.batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val_pca),
            callbacks=callbacks,
            verbose=2
        )
        
        print("Training completed!")
        
        # Get final metrics
        final_loss = min(history.history.get('val_loss', []))
        print(f"Final validation loss: {final_loss:.6f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Load best model for evaluation
    print("Loading best model for evaluation...")
    # Recreate model structure
    best_model = build_resmlp_pca_model(
        input_dim=X_train.shape[1],
        **model_params
    )
    # Recompile with same advanced loss
    best_model = compile_advanced_qrh_model(
        model=best_model,
        pca_info=pca_info,
        learning_rate=args.learning_rate,
        otm_put_weight=args.otm_put_weight,
        loss_params={
            'delta': args.huber_delta,
            'alpha': args.sobolev_alpha,
            'beta': args.sobolev_beta,
            'grid_shape': (10, 6),
            'otm_put_weight': args.otm_put_weight
        }
    )
    # Load best weights
    best_model.load_weights(str(weights_save_path))
    
    # Save complete model for later use (evaluation script)
    print("Saving complete model...")
    # Create a simple model version for saving
    eval_model = build_resmlp_pca_model(
        input_dim=X_train.shape[1],
        **model_params
    )
    eval_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    eval_model.set_weights(best_model.get_weights())
    eval_model.save(str(model_save_path))
    print(f"Evaluation-ready model saved to: {model_save_path}")
    
    # Evaluation
    print("Evaluating model...")
    try:
        # Predict PCA coefficients
        y_val_pred_pca = best_model.predict(X_val)
        
        # Reconstruct IV surface
        y_val_pred_raw = pca_inverse_transform(y_val_pred_pca, pca_info)
        
        # Overall metrics (in original IV space)
        mse_original = np.mean((y_val_pred_raw - y_val_raw) ** 2)
        mae_original = np.mean(np.abs(y_val_pred_raw - y_val_raw))
        rmse_original = np.sqrt(mse_original)
        
        # R² score
        ss_res = np.sum((y_val_raw - y_val_pred_raw) ** 2)
        ss_tot = np.sum((y_val_raw - np.mean(y_val_raw)) ** 2)  
        r2_score = 1 - (ss_res / ss_tot)

        print(f"QRH Model Evaluation Results:")
        print(f"  Val samples: {len(y_val_raw):,}")
        print(f"  MSE (original): {mse_original:.6f}")
        print(f"  MAE (original): {mae_original:.6f}")
        print(f"  RMSE (original): {rmse_original:.6f}")
        print(f"  R² score: {r2_score:.6f}")
        
        # Bucket-wise evaluation
        print("\nBucket-wise RMSE:")
        bucket_rmse = evaluate_by_buckets(y_val_raw, y_val_pred_raw)
        for bucket, rmse_val in bucket_rmse.items():
            print(f"  {bucket}: {rmse_val:.6f}")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"Final Results:")
    print(f"  - R² score: {r2_score:.6f}")
    print(f"  - RMSE (original): {rmse_original:.6f}")
    print(f"  - MAE (original): {mae_original:.6f}")
    print(f"  - Training epochs: {len(history.history.get('loss', []))}")
    print(f"  - Model parameters: {total_params:,}")
    print(f"  - PCA components: {pca_info['K']} (explains {pca_info['total_explained']:.3f})")
    
    # Save training summary
    summary_path = exp_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"QRH Training Summary\n")
        f.write(f"============================\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Dataset size: {args.data_size}\n")
        f.write(f"Model parameters: {total_params:,}\n")
        f.write(f"Training epochs: {len(history.history.get('loss', []))}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"PCA components: {pca_info['K']}\n")
        f.write(f"PCA explained variance: {pca_info['total_explained']:.6f}\n")
        f.write(f"Architecture: {args.n_blocks} ResBlocks × {args.width} units\n")
        f.write(f"Final validation loss: {final_loss:.6f}\n")
        f.write(f"Test R² score: {r2_score:.6f}\n")
        f.write(f"Test RMSE: {rmse_original:.6f}\n")
        f.write(f"Test MAE: {mae_original:.6f}\n")
        f.write(f"\nBucket-wise RMSE:\n")
        for bucket, rmse_val in bucket_rmse.items():
            f.write(f"  {bucket}: {rmse_val:.6f}\n")
    
    # Save PCA info
    pca_info_path = exp_dir / "pca_info.pkl"
    with open(pca_info_path, 'wb') as f:
        pickle.dump(pca_info, f)
    
    print(f"Summary saved to: {summary_path}")
    print(f"PCA info saved to: {pca_info_path}")
    
    return 0


if __name__ == "__main__":
    import pandas as pd  # for timestamp
    exit_code = main()
    sys.exit(exit_code)
