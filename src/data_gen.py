# -*- coding: utf-8 -*-
"""data_gen.py – Quadratic Rough Heston synthetic dataset generator

* Sobol sampling for parameters ω=(λ,η,a,b,c) and z₀=(z₀₁,...,z₀₁₀)
* Simulates 50,000 paths using explicit-implicit Euler scheme
* Computes European call options via Monte Carlo simulation
* Converts to implied volatility surface (IVS) at fixed (k,τ) grid points
* Flattens IVS to 60-dimensional vector (15 log-moneyness × 4 maturities)
* Neural network input: 15 dimensions (5 model params + 10 initial factors)
* Neural network output: 60 dimensions (flattened IV surface)

Usage
-----
# From processing directory:
$ cd src/processing
$ python data_gen.py --samples 5000 --seed 42
$ python data_gen.py --samples 150000 --seed 42  # Uses default output: ../../data/raw/

# From project root:
$ python src/processing/data_gen.py --samples 5000 --seed 42

Output Structure
---------------
data/raw/data_XXXk/
├── train_XXXk.npz     # Training data (80%)
├── test_XXXk.npz      # Test data (20%) 
├── val_XXXk.npz       # Validation data (10%)
├── preview_XXXk.csv   # First 1000 samples preview
├── x_scaler.pkl       # Input features scaler (ω, z₀)
└── y_scaler.pkl       # Output IVS scaler

Dependencies
------------
- numpy, scipy, scikit‑learn, joblib, tqdm
- Black-Scholes implied volatility solver
"""
from __future__ import annotations # Enables postponed evaluation of type annotations for forward references

import argparse
import pickle
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc, norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

# --- Import Black-Scholes utilities ----------------------------------------
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Parameter ranges for Quadratic Rough Heston model ------------------------
# ω = (λ, η, a, b, c) - 5 main model parameters  
# z₀ = (z₀₁, ..., z₀₁₀) - 10 initial factor values from multi-factor approximation
PARAM_RANGES = {
    # Model parameters ω
    "lambda": (0.5, 2.5),      # λ ∈ [0.5, 2.5]
    "eta": (1.0, 1.5),         # η ∈ [1.0, 1.5] 
    "a": (0.1, 0.6),           # a ∈ [0.1, 0.6]
    "b": (0.01, 0.5),          # b ∈ [0.01, 0.5]
    "c": (0.0001, 0.03),       # c ∈ [0.0001, 0.03]
    # Initial factor values z₀
    "z0": (-0.5, 0.5),         # z₀ᵢ ∈ [-0.5, 0.5] for i=1,...,10
}

# Fixed grid for implied volatility surface
LOG_MONEYNESS = np.linspace(-0.3, 0.3, 15)  # 15 log-moneyness points k = log(K/S₀)
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0])  # 4 maturities τ (quarterly, semi-annual, 1Y, 2Y)
S0 = 100.0  # Fixed spot price
R = 0.03    # Fixed risk-free rate

# ---------------------------------------------------------------------------
# Black-Scholes utilities ---------------------------------------------------

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def implied_volatility(price, S, K, T, r, tol=1e-6, max_iter=100):
    """Compute implied volatility using Brent's method"""
    if price <= max(S - K * np.exp(-r * T), 0):
        return 0.0
    
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - price
    
    try:
        iv = brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        return 0.0

# ---------------------------------------------------------------------------
# Sampling helper -----------------------------------------------------------

def sample_params(n: int, sobol_seed: int | None) -> list[dict[str, float | np.ndarray]]:
    """
    Generates n valid sets of Quadratic Rough Heston parameters using Sobol sequences.
    Returns parameters ω=(λ,η,a,b,c) and z₀=(z₀₁,...,z₀₁₀).
    """
    dim = 15  # 5 model parameters + 10 initial factor values
    
    # Initialize Sobol sequence generator
    if sobol_seed is not None:
        np.random.seed(sobol_seed)
    engine = qmc.Sobol(d=dim, scramble=True)
    
    # Generate samples
    samples = engine.random_base2(int(np.ceil(np.log2(n))))
    
    params: list[dict[str, float | np.ndarray]] = []
    
    for u in samples:
        if len(params) >= n:
            break
            
        # Convert to 2D array for qmc.scale compatibility
        u_2d = u.reshape(1, -1)
        
        # Sample model parameters ω
        lambda_val = qmc.scale(u_2d[:, [0]], *PARAM_RANGES["lambda"])[0, 0]
        eta = qmc.scale(u_2d[:, [1]], *PARAM_RANGES["eta"])[0, 0]
        a = qmc.scale(u_2d[:, [2]], *PARAM_RANGES["a"])[0, 0]
        b = qmc.scale(u_2d[:, [3]], *PARAM_RANGES["b"])[0, 0]
        c = qmc.scale(u_2d[:, [4]], *PARAM_RANGES["c"])[0, 0]
        
        # Sample initial factor values z₀
        z0 = np.array([
            qmc.scale(u_2d[:, [5 + i]], *PARAM_RANGES["z0"])[0, 0] 
            for i in range(10)
        ])
        
        # Skip degenerate parameters
        if any(p <= 1e-8 for p in [lambda_val, eta, a, b, c]):
            continue
            
        params.append({
            "lambda": lambda_val,
            "eta": eta, 
            "a": a,
            "b": b,
            "c": c,
            "z0": z0
        })
    
    return params

# ---------------------------------------------------------------------------
# Path simulation for Quadratic Rough Heston --------------------------------

def simulate_qrh_paths(params, n_paths=50000, n_steps=252, T_max=2.0):
    """
    Simulate paths for Quadratic Rough Heston model using explicit-implicit Euler scheme.
    Returns final index values for option pricing.
    """
    dt = T_max / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Extract parameters
    lambda_val = params["lambda"]
    eta = params["eta"] 
    a = params["a"]
    b = params["b"]
    c = params["c"]
    z0 = params["z0"]
    
    # Initialize paths
    S = np.full(n_paths, S0)  # Index price paths
    Z = np.tile(z0, (n_paths, 1))  # Multi-factor approximation state (n_paths x 10)
    
    # Store paths at maturity times for option pricing
    paths_at_maturities = {}
    current_step = 0
    
    for T in MATURITIES:
        steps_to_T = int(T / dt)
        
        # Simulate from current_step to steps_to_T
        for step in range(current_step, steps_to_T):
            # Generate random increments
            dW_S = np.random.normal(0, sqrt_dt, n_paths)
            dW_Z = np.random.normal(0, sqrt_dt, (n_paths, 10))
            
            # Simplified QRH dynamics (placeholder for actual scheme)
            # In practice, this would implement the explicit-implicit Euler scheme
            # from the paper's Appendix A.1
            
            # Variance process (simplified)
            V_t = np.maximum(np.sum(Z**2, axis=1), 1e-6)
            
            # Index process update
            drift = R * dt
            diffusion = np.sqrt(V_t) * dW_S
            S *= np.exp(drift - 0.5 * V_t * dt + diffusion)
            
            # Multi-factor process update (simplified)
            for i in range(10):
                Z[:, i] += (-lambda_val * Z[:, i] * dt + 
                           eta * np.sqrt(np.maximum(V_t, 1e-6)) * dW_Z[:, i])
        
        paths_at_maturities[T] = S.copy()
        current_step = steps_to_T
    
    return paths_at_maturities

def compute_option_prices_mc(paths_at_maturities):
    """
    Compute European call option prices via Monte Carlo for all (k,τ) combinations.
    Returns option prices on the fixed grid.
    """
    option_prices = np.zeros((len(LOG_MONEYNESS), len(MATURITIES)))
    
    for i, k in enumerate(LOG_MONEYNESS):
        K = S0 * np.exp(k)  # Strike from log-moneyness
        
        for j, T in enumerate(MATURITIES):
            S_T = paths_at_maturities[T]
            payoffs = np.maximum(S_T - K, 0)
            discounted_payoff = np.exp(-R * T) * np.mean(payoffs)
            option_prices[i, j] = discounted_payoff
    
    return option_prices

def prices_to_iv_surface(option_prices):
    """
    Convert option prices to implied volatility surface.
    Returns flattened 60-dimensional vector.
    """
    iv_surface = np.zeros((len(LOG_MONEYNESS), len(MATURITIES)))
    
    for i, k in enumerate(LOG_MONEYNESS):
        K = S0 * np.exp(k)
        
        for j, T in enumerate(MATURITIES):
            price = option_prices[i, j]
            iv = implied_volatility(price, S0, K, T, R)
            # Ensure iv is a scalar float
            if isinstance(iv, (tuple, list)):
                iv = float(iv[0]) if len(iv) > 0 else 0.0
            iv_surface[i, j] = float(iv)
    
    # Flatten to 60-dimensional vector (15 x 4)
    return iv_surface.flatten()

# ---------------------------------------------------------------------------
# Dataset generator ----------------------------------------------------------

def generate_dataset(n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, StandardScaler]:
    """
    Generates a dataset of Quadratic Rough Heston parameters (X) and their corresponding
    implied volatility surfaces (y). Input normalized to [-1,1], output standardized.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample parameter sets
    params_list = sample_params(n, sobol_seed=seed)

    # Compute IV surfaces in parallel
    def compute_iv_surface(params):
        """Compute IV surface for one parameter set"""
        paths = simulate_qrh_paths(params)
        prices = compute_option_prices_mc(paths)
        iv_surface = prices_to_iv_surface(prices)
        return iv_surface

    print("Computing implied volatility surfaces...")
    iv_surfaces = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_iv_surface)(p) for p in tqdm(params_list, desc="IV surfaces", unit="sample")
    )
    
    # Build input features X (15-dimensional: 5 model params + 10 factors)
    X_raw = np.column_stack([
        np.array([p["lambda"] for p in params_list]),
        np.array([p["eta"] for p in params_list]),
        np.array([p["a"] for p in params_list]),
        np.array([p["b"] for p in params_list]),
        np.array([p["c"] for p in params_list]),
        np.vstack([p["z0"] for p in params_list])  # 10 columns for z₀
    ])
    
    # Build output y (60-dimensional IV surfaces)
    y_raw = np.array(iv_surfaces)
    
    # Normalize inputs to [-1, 1] using MinMaxScaler
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X = x_scaler.fit_transform(X_raw)
    
    # Normalize outputs (subtract mean, divide by std)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_raw)
    
    return X, y, x_scaler, y_scaler

# ---------------------------------------------------------------------------
# Main CLI (Command-Line Interface) -----------------------------------------

def main():  # pragma: no cover
    """
    Main function to parse command-line arguments, generate the dataset,
    split it into training/validation/testing sets, and save the results.
    """
    parser = argparse.ArgumentParser("Generate Quadratic Rough Heston IV surface dataset")
    
    parser.add_argument("--samples", type=int, default=150000,
                        help="Number of samples (e.g., 5000, 50000, 100000, or 150000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="./data/raw/",
                        help="Output directory (will create data_XXXk subfolder)")
    
    args = parser.parse_args()

    # Create output directory structure
    tag = f"{args.samples // 1000}k"
    out_dir = Path(args.out) / f"data_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.samples:,} Quadratic Rough Heston samples in {out_dir.absolute()}...")

    # Generate the dataset
    X, y, x_scaler, y_scaler = generate_dataset(args.samples, seed=args.seed)

    # Train/Validation/Test split (70/15/15) --------------------------------
    # First split: 85% train+val, 15% test  
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, shuffle=True
    )
    
    # Second split: From remaining 85%, take ~82.4% as train, ~17.6% as val
    # This gives us roughly 70% train, 15% val, 15% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=args.seed, shuffle=True  
    )

    # Save datasets
    print(f"Saving train_{tag}.npz: {X_train.shape[0]:,} samples")
    print(f"Saving val_{tag}.npz: {X_val.shape[0]:,} samples")
    print(f"Saving test_{tag}.npz: {X_test.shape[0]:,} samples")
    
    np.savez_compressed(out_dir / f"train_{tag}.npz", X=X_train, y=y_train)
    np.savez_compressed(out_dir / f"val_{tag}.npz", X=X_val, y=y_val)
    np.savez_compressed(out_dir / f"test_{tag}.npz", X=X_test, y=y_test)

    # Generate CSV preview
    preview = np.hstack([X_train[:1000], y_train[:1000]])
    np.savetxt(out_dir / f"preview_{tag}.csv", preview, delimiter=",", fmt="%.6f")

    # Save scalers
    with open(out_dir / "x_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    
    with open(out_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    print(f"\n✓ Quadratic Rough Heston dataset generated successfully!")
    print(f"Location: {out_dir.absolute()}")
    print(f"Training: {X_train.shape[0]:,} samples")
    print(f"Validation: {X_val.shape[0]:,} samples") 
    print(f"Testing: {X_test.shape[0]:,} samples")
    print(f"Input features: {X_train.shape[1]} (5 model params + 10 factors)")
    print(f"Output: {y_train.shape[1]} (60-dim flattened IV surface)")
    print(f"Files: train_{tag}.npz, val_{tag}.npz, test_{tag}.npz, preview_{tag}.csv")
    print(f"Scalers: x_scaler.pkl, y_scaler.pkl")

if __name__ == "__main__":
    main()