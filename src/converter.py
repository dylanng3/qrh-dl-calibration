"""
qrh_converter.py
===============
Simple QRH data converter: NPZ → modular X,y format

Converts QRH NPZ files (train_0k.npz, val_0k.npz) to modular format:
- train_X.npy, train_y.npy, val_X.npy, val_y.npy
- CSV previews for inspection
- Automatic QRH feature naming (15 features → 60 IV surface points)

Usage:
    python src/processing/qrh_converter.py --data_size 0k
    python src/processing/qrh_converter.py --data_size 5k --no-cleanup
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle


def convert_qrh_npz(data_dir: str, data_size: str) -> None:
    """Convert QRH NPZ files to modular X,y format."""
    data_path = Path(data_dir)
    
    # QRH NPZ files
    train_file = data_path / f"train_{data_size}.npz"
    val_file = data_path / f"val_{data_size}.npz"
    
    # Check files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    
    print(f"🔄 Converting QRH NPZ → modular format...")
    print(f"  📂 {train_file.name}")
    print(f"  📂 {val_file.name}")
    
    # Load NPZ data
    train_data = np.load(train_file)
    val_data = np.load(val_file)
    
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    
    print(f"  ✅ Train: X{X_train.shape}, y{y_train.shape}")
    print(f"  ✅ Val:   X{X_val.shape}, y{y_val.shape}")
    
    # Save modular format
    np.save(data_path / "train_X.npy", X_train)
    np.save(data_path / "train_y.npy", y_train)
    np.save(data_path / "val_X.npy", X_val)
    np.save(data_path / "val_y.npy", y_val)
    
    print(f"  💾 Saved: train_X.npy, train_y.npy, val_X.npy, val_y.npy")
    
    # Create CSV previews
    save_csv_previews(data_path, X_train, y_train)
    
    print(f"  📊 Created CSV previews")


def save_csv_previews(data_path: Path, X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Save CSV previews with proper QRH feature names."""
    
    # QRH feature names (15 features)
    feature_names = [
        "omega_lambda", "omega_eta", "omega_a", "omega_b", "omega_c",  # 5 omega params
        "z0_1", "z0_2", "z0_3", "z0_4", "z0_5",  # z0 factors 1-5
        "z0_6", "z0_7", "z0_8", "z0_9", "z0_10"  # z0 factors 6-10
    ]
    
    # IV surface column names (60 points)
    n_outputs = y_train.shape[1]
    iv_columns = [f"iv_point_{i:02d}" for i in range(n_outputs)]
    
    # Save X preview (first 100 rows)
    preview_size = min(100, len(X_train))
    X_preview = pd.DataFrame(X_train[:preview_size], columns=feature_names)
    X_preview.to_csv(data_path / "train_X.csv", index=False)
    
    # Save y preview (first 100 rows)
    y_preview = pd.DataFrame(y_train[:preview_size], columns=iv_columns)
    y_preview.to_csv(data_path / "train_y.csv", index=False)


def check_if_converted(data_dir: str) -> bool:
    """Check if data is already in modular format."""
    data_path = Path(data_dir)
    
    modular_files = ["train_X.npy", "train_y.npy", "val_X.npy", "val_y.npy"]
    return all((data_path / f).exists() for f in modular_files)


def main():
    """Main converter function."""
    parser = argparse.ArgumentParser(description="QRH NPZ → Modular Converter")
    
    parser.add_argument("--data_size", type=str, required=True,
                        choices=["0k", "5k", "150k"],
                        help="Dataset size (0k, 5k, 150k)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (default: data/raw/data_{size})")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't remove NPZ files after conversion")
    
    args = parser.parse_args()
    
    # Auto-construct data directory
    if args.data_dir is None:
        args.data_dir = f"data/raw/data_{args.data_size}"
    
    data_path = Path(args.data_dir)
    
    print(f"🚀 QRH Data Converter")
    print(f"📂 Directory: {data_path}")
    print(f"📊 Size: {args.data_size}")
    
    # Check directory exists
    if not data_path.exists():
        print(f"❌ Directory not found: {data_path}")
        return 1
    
    # Check if already converted
    if check_if_converted(args.data_dir):
        print(f"✅ Data already in modular format!")
        return 0
    
    # Convert
    try:
        convert_qrh_npz(args.data_dir, args.data_size)
        
        # Cleanup NPZ files if requested
        if not args.no_cleanup:
            npz_files = [f"train_{args.data_size}.npz", f"val_{args.data_size}.npz"]
            for npz_file in npz_files:
                npz_path = data_path / npz_file
                if npz_path.exists():
                    npz_path.unlink()
                    print(f"  🗑️  Removed: {npz_file}")
        
        print(f"\n✅ Conversion completed!")
        print(f"📊 Files created:")
        print(f"  - train_X.npy, train_y.npy")
        print(f"  - val_X.npy, val_y.npy") 
        print(f"  - train_X.csv, train_y.csv (previews)")
        
        # Check scalers
        scalers = ["x_scaler.pkl", "y_scaler.pkl"]
        missing_scalers = [s for s in scalers if not (data_path / s).exists()]
        if missing_scalers:
            print(f"⚠️  Missing scalers: {missing_scalers}")
        else:
            print(f"✅ Scalers found: x_scaler.pkl, y_scaler.pkl")
        
        return 0
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
