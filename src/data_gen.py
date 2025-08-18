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
├── x_scaler.pkl       # Input features scaler (ω, z0)
└── y_scaler.pkl       # Output IVS scaler

Dependencies
------------
- numpy, scipy, scikit‑learn, joblib, tqdm
- Black-Scholes implied volatility solver
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
 
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc, norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

# GPU acceleration imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 CuPy detected - GPU acceleration enabled")
    
    def to_numpy(arr):
        """Convert CuPy array to numpy array"""
        return cp.asnumpy(arr)
        
except ImportError:
    cp = np  # Fallback to numpy if CuPy not available
    GPU_AVAILABLE = False
    print("⚠️  CuPy not found - falling back to CPU (numpy)")
    
    def to_numpy(arr):
        """Identity function for numpy arrays"""
        return arr

import time

# --- Import Black-Scholes utilities ----------------------------------------
from scipy.optimize import brentq

# Parameter ranges for Quadratic Rough Heston model:
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
    "z0": (-0.5, 0.5),         # z0i ∈ [-0.5, 0.5] for i=1,...,10
}

# Fixed grid for implied volatility surface
MATURITIES = [0.25, 0.5, 1.0, 2.0]  # T ∈ {0.25, 0.5, 1, 2} years
LOG_MONEYNESS = np.linspace(-0.5, 0.5, 15)  # k ∈ [-0.5, 0.5], 15 points

# Global constants for QRH model
RISK_FREE_RATE = 0.05  # Single source of truth for r
S0 = 100.0  # Initial spot price

# Factor-specific parameters for multi-factor approximation
# Separate roles: aggregation weights vs diffusion coefficients
GAMMA_FACTORS = np.array([0.08, 0.12, 0.10, 0.15, 0.09, 0.11, 0.13, 0.07, 0.14, 0.06])  # γi mean-reversion rates

# Aggregation weights: cᵢ for Z_t = Σᵢ cᵢ * Y^(i) (kernel approximation)
C_FACTORS = np.array([0.25, 0.35, 0.30, 0.40, 0.28, 0.32, 0.38, 0.22, 0.42, 0.20])      # ci aggregation weights

# Leverage effect: correlation between stock price and volatility factors
# For equity indices (SPX, STOXX, etc.), ρ should be strongly negative
# Industry standard for equity: ρ ∈ [-0.9, -0.4], typical around -0.7
LEVERAGE_CORRELATION = -0.7  # Baseline for equity indices
LEVERAGE_RANGE = (-0.9, -0.4)  # Range for data generation if using random sampling
USE_RANDOM_LEVERAGE = False  # Set True for production robustness, False for research baseline

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

# Path simulation for Quadratic Rough Heston --------------------------------

def simulate_qrh_paths(params, n_paths=30000, n_steps=252, T_max=2.0, use_gpu=None, leverage_rho=None):
    """
    Simulate Quadratic Rough Heston model paths using Monte Carlo with GPU acceleration.
    
    Mathematical Formulation:
    ========================
    Stock Price SDE:
        dS_t = r S_t dt + √V_t S_t dB_S
    
    Variance Process:
        V_t = λ + a(Z_t - b)² + c
        where Z_t = Σᵢ cᵢ * Y^(i)_t  (multi-factor aggregation)
    
    Factor SDEs (with drift coupling):
        dY^(i)_t = (-γᵢ Y^(i)_t - λ Z_t) dt + η √V_t dB_v
    
    Correlation Structure:
        dB_S = √dt * u
        dB_v = √dt * (ρ u + √(1-ρ²) v)
        where u, v ~ N(0,1) independent, ρ = leverage correlation
    
    Parameters:
    ===========
    params : dict
        - 'lambda' : float, variance level parameter λ
        - 'eta' : float, diffusion coefficient η  
        - 'a' : float, quadratic coefficient a
        - 'b' : float, centering parameter b
        - 'c' : float, linear coefficient c
        - 'z0' : array(10), initial factor values [z₀₁, ..., z₀₁₀]
    
    n_paths : int, default=30000
        Number of Monte Carlo simulation paths
    
    n_steps : int, default=252  
        Number of time discretization steps (252 = trading days/year)
    
    T_max : float, default=2.0
        Maximum simulation time in years
    
    use_gpu : bool or None, default=None
        - True: Force GPU usage (requires CuPy)
        - False: Force CPU usage
        - None: Auto-detect (GPU if available, CPU fallback)
    
    leverage_rho : float or None, default=None
        Override leverage correlation ρ ∈ [-1,1]
        If None, uses global LEVERAGE_CORRELATION = -0.7
    
    Returns:
    ========
    dict
        Simulated stock prices at fixed maturities:
        - 'T_0.25' : array(n_paths), prices at T=0.25 years
        - 'T_0.5'  : array(n_paths), prices at T=0.5 years  
        - 'T_1.0'  : array(n_paths), prices at T=1.0 years
        - 'T_2.0'  : array(n_paths), prices at T=2.0 years
    
    Notes:
    ======
    - Uses explicit Euler-Maruyama scheme for SDE discretization
    - Automatically captures snapshots at exact maturity times
    - GPU acceleration via CuPy (if available) for 10x+ speedup
    - Reproducible results via fixed random seeds (CPU: np.random.seed, GPU: cp.random.seed)
    - Risk-free rate r = 0.05, initial spot S₀ = 100 (global constants)
    """
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    elif use_gpu and not GPU_AVAILABLE:
        print("GPU requested but CuPy not available, falling back to CPU")
        use_gpu = False
    
    # Use appropriate library (optimized for performance)
    xp = cp if use_gpu else np
    
    # Set random seed for reproducibility
    if use_gpu:
        cp.random.seed(42)  # GPU seeding
    else:
        np.random.seed(42)  # CPU seeding
    
    # Start timing
    start_time = time.time()
    
    # Extract parameters (use float32 for GPU efficiency)
    lambda_ = float(params["lambda"])
    eta = float(params["eta"]) 
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    z0 = xp.array(params["z0"], dtype=xp.float32)  # Shape: (10,)
    
    # Fixed parameters
    r = RISK_FREE_RATE  # Use global constant
    S0_local = S0  # Initial spot price
    
    # Convert factor-specific parameters to GPU if needed
    gamma_factors = xp.array(GAMMA_FACTORS, dtype=xp.float32)  # γᵢ: mean-reversion rates
    c_factors = xp.array(C_FACTORS, dtype=xp.float32)          # cᵢ: aggregation weights
    
    # Determine leverage correlation to use
    if leverage_rho is not None:
        current_rho = leverage_rho  # Override provided
    elif USE_RANDOM_LEVERAGE:
        current_rho = np.random.uniform(LEVERAGE_RANGE[0], LEVERAGE_RANGE[1])  # Random sampling
    else:
        current_rho = LEVERAGE_CORRELATION  # Fixed baseline
    
    dt = T_max / n_steps
    sqrt_dt = xp.sqrt(dt)
    
    # Maturity times to capture (no pre-computed indices to avoid off-by-one errors)
    maturities = [0.25, 0.5, 1.0, 2.0]  # T ∈ {0.25, 0.5, 1, 2}
    
    # Initialize arrays on GPU with optimal dtype
    S = xp.full(n_paths, S0_local, dtype=xp.float32)  # Index price paths
    Z = xp.tile(z0, (n_paths, 1))  # Multi-factor approximation state (n_paths x 10)
    
    # Store results at maturities
    results = {}
    
    # Time stepping loop
    for step in range(n_steps):
        t = step * dt
        t_next = (step + 1) * dt
        
        if step < n_steps - 1:  # Don't evolve after last maturity
            # Generate 2 independent standard normal random variables (Heston style)
            u = xp.random.normal(0, 1, n_paths).astype(xp.float32)  # For stock price
            v = xp.random.normal(0, 1, n_paths).astype(xp.float32)  # For volatility
            
            # Construct correlated Brownian increments (standard Heston approach)
            dB_S = sqrt_dt * u
            dB_v = sqrt_dt * (current_rho * u + xp.sqrt(1 - current_rho * current_rho) * v)
            
            # Compute volatility from multi-factor approximation (QRH specification)
            # Step 1: Compute aggregate factor using AGGREGATION WEIGHTS cᵢ
            # Zt = Σᵢ cᵢ * Y^(i) (kernel approximation weights)
            Z_aggregate = xp.sum(c_factors.reshape(1, -1) * Z, axis=1)  # Shape: (n_paths,)
            
            # Step 2: QRH variance formula - Centered form: V = a(Z-b)² + c
            # Alternative to a + bZ + cZ²: both are quadratic but centered form more natural
            Z_centered = Z_aggregate - b  # Center around b
            V_raw = a * Z_centered * Z_centered + c  # a(Z-b)² + c
            V = lambda_ + xp.maximum(V_raw, 0.0)  # Apply shift and positivity  
            V = xp.maximum(V, 1e-8)  # Numerical stability floor
            
            # Update stock price (vectorized exp operation)
            log_return = (r - 0.5 * V) * dt + xp.sqrt(V) * dB_S
            S *= xp.exp(log_return)
            
            # Update Z factors with correct SDE
            # dZ^(i) = (-γᵢ Z^(i) - λ Zt) dt + η √Vt dB_v  [Proper formulation]
            drift_factor = -gamma_factors.reshape(1, -1) * Z  # -γᵢ Z^(i) term
            drift_coupling = -lambda_ * Z_aggregate.reshape(-1, 1)  # -λ Zt coupling (same for all factors)
            drift_Z = (drift_factor + drift_coupling) * dt  # Combined drift
            
            # Diffusion: η √Vt dB_v (common scaling by √Vt, not individual dᵢ)
            diffusion_Z = (eta * xp.sqrt(V) * dB_v).reshape(-1, 1)  # Common √Vt scaling
            Z += drift_Z + diffusion_Z
        
        # Check if any maturity times match current time (after evolution step)
        for T_maturity in maturities:
            if abs(t_next - T_maturity) < 1e-6:  # Floating point tolerance
                # Store paths at this maturity
                if GPU_AVAILABLE and isinstance(S, cp.ndarray):
                    results[f"T_{T_maturity}"] = to_numpy(S.copy())
                else:
                    results[f"T_{T_maturity}"] = np.array(S.copy())
    
    # Ensure all maturities are captured (final check)
    for T_maturity in maturities:
        if f"T_{T_maturity}" not in results:
            # Use final state if maturity was missed
            if GPU_AVAILABLE and isinstance(S, cp.ndarray):
                results[f"T_{T_maturity}"] = to_numpy(S.copy())
            else:
                results[f"T_{T_maturity}"] = np.array(S.copy())
    
    # Timing info
    elapsed = time.time() - start_time
    device = "GPU" if use_gpu else "CPU"
    if elapsed > 1.0:  # Only print for slow operations
        print(f"  📊 Simulated {n_paths:,} paths on {device} in {elapsed:.2f}s")
    
    return results

def compute_option_prices_mc(paths_at_maturities):
    """
    Compute European call option prices via Monte Carlo for all (k,τ) combinations.
    Returns option prices on the fixed grid.
    """
    option_prices = np.zeros((len(LOG_MONEYNESS), len(MATURITIES)))
    
    for i, k in enumerate(LOG_MONEYNESS):
        K = S0 * np.exp(k)  # Strike from log-moneyness
        
        for j, T in enumerate(MATURITIES):
            T_key = f"T_{T}"
            if T_key in paths_at_maturities:
                S_T = paths_at_maturities[T_key]
                payoffs = np.maximum(S_T - K, 0)
                discounted_payoff = np.exp(-RISK_FREE_RATE * T) * np.mean(payoffs)
                option_prices[i, j] = discounted_payoff
            else:
                # Fallback: set a reasonable default price (lower threshold)
                option_prices[i, j] = max(S0 - K * np.exp(-RISK_FREE_RATE * T), 1e-6)
    
    return option_prices

def prices_to_iv_surface(option_prices):
    """
    Convert option prices to implied volatility surface.
    Returns flattened 60-dimensional vector with IV clipping for realism.
    """
    iv_surface = np.zeros((len(LOG_MONEYNESS), len(MATURITIES)))
    
    for i, k in enumerate(LOG_MONEYNESS):
        K = S0 * np.exp(k)
        
        for j, T in enumerate(MATURITIES):
            price = option_prices[i, j]
            iv = implied_volatility(price, S0, K, T, RISK_FREE_RATE)
            # Ensure iv is a scalar float
            if isinstance(iv, (tuple, list)):
                iv = float(iv[0]) if len(iv) > 0 else 0.0
            
            # Apply realistic IV clipping [0.01, 2.0] for market realism
            iv = np.clip(float(iv), 0.01, 2.0)
            iv_surface[i, j] = iv
    
    # Flatten to 60-dimensional vector (15 x 4)
    return iv_surface.flatten()

# Dataset generator ----------------------------------------------------------

def generate_dataset(n: int, seed: int | None = None) -> tuple[
    tuple[np.ndarray, np.ndarray],  # (X_train, y_train)
    tuple[np.ndarray, np.ndarray],  # (X_val, y_val)  
    tuple[np.ndarray, np.ndarray],  # (X_test, y_test)
    MinMaxScaler,                   # x_scaler
    StandardScaler                  # y_scaler
]:
    """
    Generates a dataset of Quadratic Rough Heston parameters (X) and their corresponding
    implied volatility surfaces (y). Splits into train/val/test and fits scalers only on training data.
    """
    if seed is not None:
        np.random.seed(seed)
        random_seed = seed
    else:
        random_seed = 42  # default seed for reproducible splits
    
    # Sample parameter sets
    params_list = sample_params(n, sobol_seed=seed)

    # Compute IV surfaces in parallel
    def compute_iv_surface(params):
        """Compute IV surface for one parameter set using GPU sequential processing"""
        paths = simulate_qrh_paths(params, use_gpu=GPU_AVAILABLE)
        prices = compute_option_prices_mc(paths)
        iv_surface = prices_to_iv_surface(prices)
        return iv_surface

    print("Computing implied volatility surfaces...")
    
    # GPU sequential to avoid memory conflicts (optimal for strong GPU)
    iv_surfaces = []
    for params in tqdm(params_list, desc="IV surfaces", unit="sample"):
        iv_surface = compute_iv_surface(params)
        iv_surfaces.append(iv_surface)
    
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
    
    # IMPORTANT: Split data BEFORE fitting scalers to avoid data leakage
    # First split: 10% for test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y_raw, test_size=0.1, random_state=random_seed
    )
    # Second split: 10% for val (so 80% train, 10% val, 10% test)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=random_seed + 1
    )
    
    # Fit scalers ONLY on training data to prevent data leakage
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = StandardScaler()
    
    # Fit on train, transform all sets
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)
    
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), x_scaler, y_scaler

# Main CLI (Command-Line Interface) -----------------------------------------

def main(): 
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

    # Generate the dataset with proper train/val/test split and no data leakage
    (X_train, y_train), (X_val, y_val), (X_test, y_test), x_scaler, y_scaler = generate_dataset(
        args.samples, seed=args.seed
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