# Heston Surrogate Pricer Pipeline

## 📝 Overview

This project builds a pipeline for training and evaluating a Surrogate model for pricing and calibrating the Heston model. Instead of using the traditional FFT method (which is slow and resource-intensive), the pipeline creates a fast, accurate neural network to predict the implied volatility surface from Heston parameters.

### Theoretical Background

**Heston Model** describes the stochastic volatility of asset prices:
```
dS(t) = rS(t)dt + √v(t)S(t)dW₁(t)
dv(t) = κ(θ - v(t))dt + σ√v(t)dW₂(t)
```
With parameters: v₀ (initial volatility), κ (mean reversion), θ (long-term volatility), σ (vol-of-vol), ρ (correlation).

**Surrogate Model** learns the mapping:
```
f: (v₀, κ, θ, σ, ρ, r, strikes, tenors) → IV_surface
```
Instead of complex FFT computation, the model directly predicts implied volatility for the entire strikes × tenors grid.

---

## 📂 Project Structure

```
├── data/
│   ├── raw/             # Raw data (npz, csv, scaler, ...)
│   └── processed/       
│
├── experiments/         # Training results, checkpoints, PCA info, summary
│   └── experiments_...  # Each training run (auto-generated)
│
├── notebooks/           # Jupyter notebooks for exploratory analysis
│
├── reports/             # Model results
│└── experiments_...     # Each valuation (auto-generated)

├── references/          # Reference papers, manuals
│
├── scripts/
│   ├── training.py      # Model training script
│   └── evaluation.py    # Model evaluation script
│
├── src/                 # Core code: model, data, loss, utils
│   ├── model_architectures.py
│   ├── data_gen.py
│   ├── converter.py
│
├── tests/               # Unit tests for main modules
│
├── Makefile             # Utility commands (make train, make eval, ...)
├── requirements.txt     # Python environment
├── pyproject.toml       # Project metadata
├── README.md            # This documentation
└── .gitignore           # Files/directories excluded from git
```

---

## 🏗️ Model Architecture Details

### 1. **PCA-head Architecture**
Instead of predicting 60 IV values directly (10 strikes × 6 tenors), the model uses PCA:
```
Original_IV → PCA_transform → K_components (K << 60)
Model: Input(15) → ResidualMLP → PCA_coefficients(K)
PCA_coefficients → PCA_inverse → IV_surface(60)
```
**Advantages**: Reduces dimensionality, increases stability, preserves IV surface structure.

### 2. **Residual MLP**
```
x → Dense(width) + ReLU → ResBlock₁ → ResBlock₂ → ... → ResBlockₙ → Dense(K)
```
**ResBlock**: `x + Dense(ReLU(Dense(x)))`
- Enables deeper training, avoids gradient vanishing
- Width = 128, n_blocks = 8 (configurable)

### 3. **Advanced Loss Function**
Combines 3 components:
```
L_total = L_Huber + α·L_Sobolev_strike + β·L_Sobolev_tenor + W_OTM·L_weighted
```

**Huber Loss**: `L_Huber = Σ huber_δ(y_true - y_pred)`
- More robust than MSE, less sensitive to outliers

**Sobolev Regularization**: Penalty for non-smoothness
```
L_Sobolev_strike = Σ |∂²IV/∂K²|²  (smoothness along strikes)
L_Sobolev_tenor = Σ |∂²IV/∂T²|²   (smoothness along tenors)
```

**OTM Put Weighting**: 
```
W_OTM = otm_put_weight if strike ∈ first_third_strikes
W_OTM = 1.0 otherwise
```
Helps the model focus on improving the OTM Put region (usually the hardest).

## 📚 Module Details

### `src/model_architectures.py`
**Main functions**:
- `build_resmlp_pca_model()`: Build Residual MLP architecture
- `fit_pca_components()`: Fit PCA on training data, store explained variance
- `create_advanced_loss_function()`: Custom loss with Huber + Sobolev + Weighting
- `pca_transform_targets()`, `pca_inverse_transform()`: Forward/inverse PCA transforms

### `src/data_gen.py`
**Function**: Generate synthetic Heston data

### `scripts/training.py`
**Full pipeline**:
6. Save model, weights, PCA info, summary

### `scripts/evaluation.py`
**Comprehensive evaluation**:
1. Load model and PCA info
2. Predict on test set
3. Transform from PCA space to IV space
4. Compute metrics: R², RMSE, MAE overall and by buckets
5. Create visualization: IV surface comparison, error heatmap
6. Export results JSON, PNG

## 🚀 Main Features

### 1. Surrogate Model Training
The pipeline uses a **Residual MLP + PCA-head** architecture to reduce output dimensionality from 60 to K components (typically 12-30), enabling faster and more stable learning.

**Integrated loss function:**
- **Huber loss**: Robust to outliers
- **Sobolev regularization**: Ensures IV surface smoothness (∂²IV/∂K², ∂²IV/∂T²)
- **OTM Put weighting**: 2-5x weight for the hardest region

**Hyperparameters**: Number of residual blocks (8), width (128), PCA components (30), OTM Put weight (2.0), learning rate scheduling.

### 2. Comprehensive Model Evaluation
The evaluation script performs:
- **Prediction** of IV surface on independent test set
- **Transform** from PCA space to original IV space
- **Metrics calculation**: R², RMSE, MAE overall and by buckets:
  - **ATM**: |moneyness - 1.0| < 0.1
  - **OTM Put**: strikes < first_third_boundary
  - **OTM Call**: strikes > last_third_boundary
  - **Short tenor**: T ≤ median_tenor
  - **Long tenor**: T > median_tenor
- **Visualization**: IV surface comparison, absolute error heatmap
- **Export**: JSON results, PNG plots

### 3. Experiment Management & Reproducibility
Each training run automatically creates a timestamped directory in `experiments/`:
```
experiments/advanced_qrh_100k_20250819_184403/
├── qrh_advanced_100k.weights.h5    # Best model weights
├── qrh_advanced_100k.keras         # Full model (evaluation-ready)
├── pca_info.pkl                    # PCA transformer & explained variance
└── training_summary.txt            # Hyperparams, metrics, epoch info
```
**Reproducibility**: Fixed random seeds (42) for numpy, random, tensorflow.

### 4. TensorBoard Monitoring
Real-time tracking of:
- **Training**: loss, val_loss, mae, val_mae
- **Learning rate**: scheduling with ReduceLROnPlateau
- **Model architecture**: computation graph
- **Hyperparameters**: auto-logged args

```bash
tensorboard --logdir reports/tensorboard/
```

### 5. Modular & Extensible Design
- **Easy parameter tuning**: Command-line args for all hyperparameters
- **Pluggable architectures**: Easily switch from ResidualMLP to Attention, Transformer
- **Custom loss functions**: Modular, easy to add new regularization terms
- **Data format flexibility**: Supports both modular (.npy, .csv) and compressed (.npz)
- **GPU optimization**: Mixed precision, automatic XLA compilation

---

## 💻 Usage Guide

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Model Training

```bash
python scripts/training.py --data_size 100k --epochs 200 --format modular --pca_components 30 --otm_put_weight 2.0
```
- For other parameters, see `scripts/training.py` or use `--help`.

### 3. Model Evaluation

```bash
python scripts/evaluation.py --data_path data/raw/data_100k/test_100k.npz
```
- Results and plots will be saved in `reports/evaluation/`.

### 4. Monitor Training with TensorBoard

```bash
tensorboard --logdir reports/tensorboard/
```

---

## 📊 Benchmark Results

### Performance metrics (100K dataset, PCA-30, OTM weight=2.0)
```
Test Results:
├── R² Score:           0.9987
├── RMSE Overall:       0.0362
├── MAE Overall:        0.0106
└── Bucket RMSE:
    ├── ATM:            0.0266
    ├── OTM Put:        0.0388
    ├── OTM Call:       0.0382
    ├── Short Tenor:    0.0415
    └── Long Tenor:     0.0298

Training Info:
├── PCA Components:     30
├── Epochs:             (see training_summary.txt)
├── Parameters:         145,374
├── PCA Explained Var:  99.99%
├── Training Time:      ~3 minutes (RTX 2060S)
```

### Key improvements
- **OTM Put RMSE**: Remains competitive and much improved over baseline
- **Error distribution**: Even across the surface, no OTM Put bias
- **Model efficiency**: 30 PCA components retain 99.99% info with 145K params
- **Training stability**: Early stopping, automatic learning rate scheduling

---

## 📚 References & Theory

### Papers & References
- **Heston, S.L. (1993)**: "A Closed-Form Solution for Options with Stochastic Volatility"
- **Carr, P., & Madan, D. (1999)**: "Option valuation using the fast Fourier transform"
- **Ruf, J., & Wang, W. (2019)**: "Neural networks for option pricing and hedging"
- See more in `references/`

### Mathematical Foundation
**Problem Setup**: Instead of solving a complex integral
```
C(K,T) = e^(-rT) ∫∫ max(S_T - K, 0) p(S_T, v_T | S_0, v_0) dS_T dv_T
```
The model learns a direct mapping: **θ = (v₀, κ, θ, σ, ρ, r) → IV(K,T)**

**Objective**: Minimize weighted prediction error with smoothness constraints
```
L = Σᵢ W_OTM(i) · huber_δ(IV_true - IV_pred) +
    α Σⱼ |∂²IV/∂K²|² + β Σₖ |∂²IV/∂T²|²
```

**PCA Compression**: IV ∈ R⁶⁰ → PCA_coeff ∈ Rᴷ where K << 60
```
IV = μ + Σᵢ₌₁ᴷ αᵢ · PC_i,  explained_var = Σᵢ₌₁ᴷ λᵢ / Σⱼ₌₁⁶⁰ λⱼ ≥ 0.999
```

### Computational Benefits
- **Speed**: ~1000x faster than FFT (milliseconds vs seconds)
- **Scalability**: Batch processing, GPU acceleration
- **Calibration**: Gradient-based optimization directly on IV space
- **Risk Management**: Real-time Greeks calculation

---

## Contribution & Contact

- Contribution: Pull requests or issues on GitHub
- Contact: dgngn03.forwork.dta@gmail.com

---
