#!/usr/bin/env python3
"""
training.py
===========
QRH training script with support for both NPZ and modular X,y formats.

Usage:
    python training.py --data_size 0k --epochs 50 --format modular
    python training.py --data_size 5k --epochs 100 --format npz
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    """Main training function with format selection."""
    parser = argparse.ArgumentParser(description="QRH Model Training")
    
    # Data arguments
    parser.add_argument("--data_size", type=str, required=True,
                        choices=["0k", "5k", "150k"],
                        help="Dataset size to use")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["npz", "modular", "auto"],
                        help="Data format: npz (train_0k.npz) or modular (train_X.npy)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--model_name", type=str, default="qrh_model",
                        help="Model name for saving")
    parser.add_argument("--test_run", action="store_true",
                        help="Quick test with reduced epochs")
    
    args = parser.parse_args()
    
    # Construct data directory path
    data_dir = f"data/raw/data_{args.data_size}"
    
    print("🚀 Starting QRH Training...")
    print(f"Dataset: {data_dir}")
    print(f"Format: {args.format}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    if args.test_run:
        print("Mode: TEST RUN")
    
    # Setup paths
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    
    # Add to Python path
    sys.path.insert(0, str(src_path))
    
    # Import after setting path
    try:
        print("📦 Importing dependencies...")
        import numpy as np
        print("  ✅ numpy")
        
        import tensorflow as tf
        print("  ✅ tensorflow")
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        import keras
        print("  ✅ keras")
        
        from modeling.model_architectures import create_qrh_model, compile_qrh_model
        print("  ✅ model_architectures")
        
        from modeling.training_utils_qrh import (
            get_qrh_training_config,
            load_qrh_dataset,
            load_qrh_modular_dataset,
            train_qrh_model,
            train_qrh_modular_model,
            evaluate_qrh_model,
            create_experiment_dir
        )
        print("  ✅ training_utils_qrh")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    
    # Check if dataset exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        print("Available datasets:")
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            for d in raw_dir.glob("data_*"):
                print(f"  - {d.name}")
        return 1
    
    # Auto-detect or verify format
    format_type = args.format
    if format_type == "auto":
        # Check which format is available
        modular_files = ["train_X.npy", "train_y.npy", "val_X.npy", "val_y.npy"]
        npz_files = [f"train_{args.data_size}.npz", f"test_{args.data_size}.npz", f"val_{args.data_size}.npz"]
        
        has_modular = all((data_path / f).exists() for f in modular_files)
        has_npz = all((data_path / f).exists() for f in npz_files)
        
        if has_modular:
            format_type = "modular"
            print(f"✓ Auto-detected modular format")
        elif has_npz:
            format_type = "npz"
            print(f"✓ Auto-detected NPZ format")
        else:
            print(f"❌ No recognized format found in {data_dir}")
            print("Available files:", list(f.name for f in data_path.glob("*")))
            return 1
    
    print(f"📊 Using format: {format_type}")
    
    # Create experiment directory
    try:
        exp_dir = create_experiment_dir("experiments")
        print(f"📁 Experiment directory: {exp_dir}")
    except Exception as e:
        print(f"❌ Failed to create experiment directory: {e}")
        return 1
    
    # Get configuration
    config = get_qrh_training_config()
    
    # Override with command line arguments
    epochs = min(10, args.epochs) if args.test_run else args.epochs
    config.update({
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.patience,
        "data_dir": data_dir,
        "dataset_size": args.data_size,
        "model_name": args.model_name
    })
    
    print(f"⚙️  Training configuration:")
    print(f"  - Dataset size: {args.data_size}")
    print(f"  - Format: {format_type}")
    print(f"  - Epochs: {config['epochs']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Early stopping patience: {config['early_stopping_patience']}")
    
    # Load dataset based on format
    print("📂 Loading dataset...")
    try:
        if format_type == "modular":
            train_data, val_data, scalers = load_qrh_modular_dataset(data_dir)
            test_data = None  # No separate test set in modular format
        else:  # npz format
            train_data, val_data, test_data, scalers = load_qrh_dataset(data_dir, args.data_size)
        
        print(f"  ✅ Train samples: {len(train_data[0])}")
        print(f"  ✅ Val samples: {len(val_data[0])}")
        if test_data:
            print(f"  ✅ Test samples: {len(test_data[0])}")
            
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Create model
    print("🧠 Creating model...")
    try:
        model = create_qrh_model("qrh_mlp")
        model = compile_qrh_model(model, learning_rate=config["learning_rate"])
        
        total_params = model.count_params()
        print(f"  ✅ Model created with {total_params:,} parameters")
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return 1
    
    # Setup paths
    model_save_path = Path(exp_dir) / f"{args.model_name}_{args.data_size}.keras"
    log_dir = Path(exp_dir) / "logs"
    
    print(f"💾 Model will be saved to: {model_save_path}")
    
    # Train model
    print("🏋️  Starting training...")
    try:
        if format_type == "modular":
            history = train_qrh_modular_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                config=config,
                model_save_path=str(model_save_path),
                log_dir=str(log_dir),
                verbose=1
            )
        else:  # npz format
            history = train_qrh_model(
                model=model,
                data=(train_data, val_data),
                config=config,
                model_save_path=str(model_save_path),
                log_dir=str(log_dir),
                verbose=1
            )
        
        print("✅ Training completed!")
        
        # Get final metrics
        final_loss = min(history.history.get('val_loss', []))
        print(f"📈 Final validation loss: {final_loss:.6f}")
        
        # Load best model for evaluation
        print("🔄 Loading best model for evaluation...")
        best_model = keras.models.load_model(str(model_save_path))
        
        # Evaluation
        x_scaler, y_scaler = scalers
        if test_data:
            # NPZ format has separate test set
            print("🧪 Evaluating on test set...")
            metrics = evaluate_qrh_model(
                model=best_model,
                test_data=test_data,
                y_scaler=y_scaler,
                verbose=1
            )
        else:
            # Modular format - evaluate on validation set
            print("🧪 Evaluating on validation set...")
            metrics = evaluate_qrh_model(
                model=best_model,
                test_data=val_data,
                y_scaler=y_scaler,
                verbose=1
            )
        
        print(f"📊 Final Results:")
        print(f"  - R² score: {metrics['r2_score']:.6f}")
        print(f"  - RMSE (original): {metrics['rmse_original']:.6f}")
        print(f"  - MAE (original): {metrics['mae_original']:.6f}")
        print(f"  - Training epochs: {len(history.history.get('loss', []))}")
        print(f"  - Model parameters: {total_params:,}")
        
        # Save training summary
        summary_path = Path(exp_dir) / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"QRH Production Training Summary\\n")
            f.write(f"================================\\n")
            f.write(f"Dataset: {data_dir}\\n")
            f.write(f"Dataset size: {args.data_size}\\n")
            f.write(f"Model parameters: {total_params:,}\\n")
            f.write(f"Training epochs: {len(history.history.get('loss', []))}\\n")
            f.write(f"Batch size: {config['batch_size']}\\n")
            f.write(f"Learning rate: {config['learning_rate']}\\n")
            f.write(f"Final validation loss: {final_loss:.6f}\\n")
            f.write(f"Test R² score: {metrics['r2_score']:.6f}\\n")
            f.write(f"Test RMSE: {metrics['rmse_original']:.6f}\\n")
            f.write(f"Test MAE: {metrics['mae_original']:.6f}\\n")
        
        print(f"📄 Summary saved to: {summary_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
