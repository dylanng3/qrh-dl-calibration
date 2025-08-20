#!/usr/bin/env python3
"""
evaluation.py
=============
Comprehensive evaluation script for Quantitative Risk Heston (QRH) models.

This script provides a unified suite for evaluating trained QRH models, including:
- Loading and testing models (with or without PCA heads)
- Computing detailed performance metrics (R², RMSE, MAE, etc.)
- Residual and bucket-wise analysis
- Visualization: prediction scatter, residuals, error distributions, IV surface comparison, and more
- Exporting evaluation results and plots for reporting

Usage:
    python3 evaluation.py --data_path path/to/test_data.npz [--model_path path/to/model.keras]
    (python3 scripts/evaluation.py --data_path data/raw/data_100k/test_100k.npz)

Arguments:
    --data_path   Path to the test .npz file (required)
    --model_path  Path to the trained model (.keras) (optional; uses latest if not specified)

All results and plots are saved in the reports/evaluation/ directory.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error, median_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import jarque_bera
import tensorflow as tf
import keras
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend and style
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# QRH Model Evaluator Class
# =============================================================================

class QRHModelEvaluator:
    """Comprehensive evaluation suite for QRH models."""
    
    def __init__(self, model=None, pca_model=None, scaler=None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained neural network model
            pca_model: Fitted PCA model (if using PCA-head)
            scaler: Data scaler for inverse transformation
        """
        self.model = model
        self.pca_model = pca_model
        self.scaler = scaler
        self.results = {}
        
    def evaluate_model(self, X_test, y_test, X_val=None, y_val=None, 
                      strike_grid=None, tenor_grid=None, verbose=True):
        """
        Comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            strike_grid: Strike values for surface plotting
            tenor_grid: Tenor values for surface plotting
            verbose: Print results
            
        Returns:
            dict: Comprehensive evaluation results
        """
        print("Starting comprehensive model evaluation...")
        
        # Make predictions
        if self.pca_model is not None:
            # For PCA-head models
            print("Using PCA model for predictions...")
            y_pred_pca = self.model.predict(X_test, verbose=0) # type: ignore
            y_pred = self.pca_model.inverse_transform(y_pred_pca)
            print(f"Transformed from PCA space: {y_pred_pca.shape} -> {y_pred.shape}")
        else:
            # For standard models
            y_pred = self.model.predict(X_test, verbose=0) # type: ignore
            
        # Basic metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Bucket-wise analysis
        bucket_metrics = self._bucket_analysis(X_test, y_test, y_pred, 
                                             strike_grid, tenor_grid)
        
        # Residual analysis
        residual_stats = self._residual_analysis(y_test, y_pred)
        
        # Store results
        self.results = {
            'metrics': metrics,
            'bucket_metrics': bucket_metrics,
            'residual_stats': residual_stats,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test
        }
        
        if verbose:
            self._print_results()
            
        return self.results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate basic performance metrics."""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred)),
            'median_ae': np.median(np.abs(y_true - y_pred))
        }
    
    def _bucket_analysis(self, X_test, y_test, y_pred, strike_grid, tenor_grid):
        """Analyze performance by different buckets (ATM, OTM, tenor groups)."""
        if strike_grid is None or tenor_grid is None:
            return {}
            
        n_strikes, n_tenors = len(strike_grid), len(tenor_grid)
        expected_size = n_strikes * n_tenors
        actual_size = y_test.shape[1]
        
        if actual_size != expected_size:
            return {}
            
        # Reshape for analysis
        y_test_surf = y_test.reshape(-1, n_strikes, n_tenors)
        y_pred_surf = y_pred.reshape(-1, n_strikes, n_tenors)
        
        buckets = {}
        
        # ATM (middle strikes)
        atm_idx = n_strikes // 2
        atm_true = y_test_surf[:, atm_idx, :].flatten()
        atm_pred = y_pred_surf[:, atm_idx, :].flatten()
        buckets['atm'] = self._calculate_metrics(atm_true, atm_pred)
        
        # OTM Put (low strikes)
        otm_put_idx = slice(0, n_strikes // 3)
        otm_put_true = y_test_surf[:, otm_put_idx, :].flatten()
        otm_put_pred = y_pred_surf[:, otm_put_idx, :].flatten()
        buckets['otm_put'] = self._calculate_metrics(otm_put_true, otm_put_pred)
        
        # OTM Call (high strikes)
        otm_call_idx = slice(2 * n_strikes // 3, n_strikes)
        otm_call_true = y_test_surf[:, otm_call_idx, :].flatten()
        otm_call_pred = y_pred_surf[:, otm_call_idx, :].flatten()
        buckets['otm_call'] = self._calculate_metrics(otm_call_true, otm_call_pred)
        
        # Short tenor
        short_tenor_idx = slice(0, n_tenors // 2)
        short_true = y_test_surf[:, :, short_tenor_idx].flatten()
        short_pred = y_pred_surf[:, :, short_tenor_idx].flatten()
        buckets['short_tenor'] = self._calculate_metrics(short_true, short_pred)
        
        # Long tenor
        long_tenor_idx = slice(n_tenors // 2, n_tenors)
        long_true = y_test_surf[:, :, long_tenor_idx].flatten()
        long_pred = y_pred_surf[:, :, long_tenor_idx].flatten()
        buckets['long_tenor'] = self._calculate_metrics(long_true, long_pred)
        
        return buckets
    
    def _residual_analysis(self, y_true, y_pred):
        """Statistical analysis of residuals."""
        residuals = y_true.flatten() - y_pred.flatten()
        
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'jarque_bera': stats.jarque_bera(residuals),
            'shapiro_wilk': stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals),
            'percentiles': np.percentile(residuals, [5, 25, 50, 75, 95])
        }
    
    def _print_results(self):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Basic metrics
        metrics = self.results['metrics']
        print(f"\nOverall Performance:")
        print(f"  R² Score:     {metrics['r2_score']:.6f}")
        print(f"  RMSE:         {metrics['rmse']:.6f}")
        print(f"  MAE:          {metrics['mae']:.6f}")
        print(f"  Max Error:    {metrics['max_error']:.6f}")
        print(f"  Median AE:    {metrics['median_ae']:.6f}")
        
        # Bucket analysis
        if self.results['bucket_metrics']:
            print(f"\nBucket-wise RMSE:")
            for bucket_name, bucket_metrics in self.results['bucket_metrics'].items():
                print(f"  {bucket_name:<12}: {bucket_metrics['rmse']:.6f}")
        
        # Residual statistics
        residual_stats = self.results['residual_stats']
        print(f"\nResidual Statistics:")
        print(f"  Mean:         {residual_stats['mean']:.8f}")
        print(f"  Std Dev:      {residual_stats['std']:.6f}")
        print(f"  Skewness:     {residual_stats['skewness']:.4f}")
        print(f"  Kurtosis:     {residual_stats['kurtosis']:.4f}")
        
        jb_stat, jb_pvalue = residual_stats['jarque_bera']
        print(f"  Jarque-Bera:  {jb_stat:.4f} (p={jb_pvalue:.4f})")
        
    def plot_comprehensive_evaluation(self, figsize=(20, 16), save_path=None):
        """Create comprehensive evaluation plots."""
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate_model() first.")
            
        fig = plt.figure(figsize=figsize)
        
        # 1. Scatter plot: Predicted vs Actual
        ax1 = plt.subplot(3, 4, 1)
        self._plot_prediction_scatter(ax1)
        
        # 2. Residual plot
        ax2 = plt.subplot(3, 4, 2)
        self._plot_residuals(ax2)
        
        # 3. Residual histogram
        ax3 = plt.subplot(3, 4, 3)
        self._plot_residual_histogram(ax3)
        
        # 4. Q-Q plot
        ax4 = plt.subplot(3, 4, 4)
        self._plot_qq_plot(ax4)
        
        # 5. Bucket-wise RMSE
        ax5 = plt.subplot(3, 4, 5)
        self._plot_bucket_rmse(ax5)
        
        # 6. Error distribution by percentiles
        ax6 = plt.subplot(3, 4, 6)
        self._plot_error_percentiles(ax6)
        
        # 7. Residuals vs Fitted
        ax7 = plt.subplot(3, 4, 7)
        self._plot_residuals_vs_fitted(ax7)
        
        # 8. PCA explained variance (if available)
        ax8 = plt.subplot(3, 4, 8)
        if self.pca_model is not None:
            self._plot_pca_variance(ax8)
        else:
            ax8.text(0.5, 0.5, 'PCA not used', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('PCA Analysis')
        
        # 9-12: More detailed plots
        ax9 = plt.subplot(3, 4, 9)
        self._plot_absolute_errors(ax9)
        
        ax10 = plt.subplot(3, 4, 10)
        self._plot_prediction_intervals(ax10)
        
        ax11 = plt.subplot(3, 4, 11)
        self._plot_error_autocorr(ax11)
        
        ax12 = plt.subplot(3, 4, 12)
        self._plot_model_summary(ax12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to: {save_path}")
            
        plt.show()
        
    def _plot_prediction_scatter(self, ax):
        """Scatter plot of predictions vs actual values."""
        y_true = self.results['y_test'].flatten()
        y_pred = self.results['y_pred'].flatten()
        
        # Sample for plotting if too many points
        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            y_true, y_pred = y_true[idx], y_pred[idx]
            
        ax.scatter(y_true, y_pred, alpha=0.6, s=1)
        
        # Perfect prediction line
        lims = [np.min([y_true, y_pred]), np.max([y_true, y_pred])]
        ax.plot(lims, lims, 'r--', alpha=0.8, zorder=0)
        
        ax.set_xlabel('Actual IV')
        ax.set_ylabel('Predicted IV')
        ax.set_title('Predicted vs Actual')
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = self.results['metrics']['r2_score']
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_residuals(self, ax):
        """Plot residuals vs actual values."""
        y_true = self.results['y_test'].flatten()
        y_pred = self.results['y_pred'].flatten()
        residuals = y_true - y_pred
        
        # Sample for plotting
        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            y_true, residuals = y_true[idx], residuals[idx]
            
        ax.scatter(y_true, residuals, alpha=0.6, s=1)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Actual IV')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
    
    def _plot_residual_histogram(self, ax):
        """Histogram of residuals."""
        residuals = (self.results['y_test'] - self.results['y_pred']).flatten()
        
        ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal fit')
        
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_qq_plot(self, ax):
        """Q-Q plot for residuals."""
        residuals = (self.results['y_test'] - self.results['y_pred']).flatten()
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normality Check)')
        ax.grid(True, alpha=0.3)
    
    def _plot_bucket_rmse(self, ax):
        """Bar plot of bucket-wise RMSE."""
        if not self.results['bucket_metrics']:
            ax.text(0.5, 0.5, 'Bucket analysis\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Bucket-wise RMSE')
            return
            
        buckets = list(self.results['bucket_metrics'].keys())
        rmse_values = [self.results['bucket_metrics'][bucket]['rmse'] for bucket in buckets]
        
        bars = ax.bar(buckets, rmse_values)
        ax.set_ylabel('RMSE')
        ax.set_title('Bucket-wise RMSE')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_error_percentiles(self, ax):
        """Box plot of error percentiles."""
        residuals = (self.results['y_test'] - self.results['y_pred']).flatten()
        abs_errors = np.abs(residuals)
        
        ax.boxplot(abs_errors, vert=True)
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_residuals_vs_fitted(self, ax):
        """Residuals vs fitted values."""
        y_pred = self.results['y_pred'].flatten()
        residuals = (self.results['y_test'] - self.results['y_pred']).flatten()
        
        # Sample for plotting
        if len(y_pred) > 10000:
            idx = np.random.choice(len(y_pred), 10000, replace=False)
            y_pred, residuals = y_pred[idx], residuals[idx]
            
        ax.scatter(y_pred, residuals, alpha=0.6, s=1)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted')
        ax.grid(True, alpha=0.3)
    
    def _plot_pca_variance(self, ax):
        """PCA explained variance plot."""
        if self.pca_model is None:
            ax.text(0.5, 0.5, 'PCA not used', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('PCA Analysis')
            return
            
        explained_var = self.pca_model.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        x = range(1, len(explained_var) + 1)
        ax.plot(x, explained_var, 'bo-', label='Individual')
        ax.plot(x, cumsum_var, 'ro-', label='Cumulative')
        
        ax.set_xlabel('PCA Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add total variance text
        total_var = cumsum_var[-1]
        ax.text(0.7, 0.3, f'Total: {total_var:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_absolute_errors(self, ax):
        """Plot absolute errors over data points."""
        abs_errors = np.abs(self.results['y_test'] - self.results['y_pred']).flatten()
        
        # Sample for plotting
        if len(abs_errors) > 2000:
            idx = np.random.choice(len(abs_errors), 2000, replace=False)
            abs_errors = abs_errors[idx]
            
        ax.plot(abs_errors, alpha=0.7)
        ax.axhline(y=np.mean(abs_errors), color='r', linestyle='--', 
                  label=f'Mean: {np.mean(abs_errors):.4f}')
        
        ax.set_xlabel('Data Point')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Absolute Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_intervals(self, ax):
        """Plot prediction intervals."""
        y_true = self.results['y_test'].flatten()
        y_pred = self.results['y_pred'].flatten()
        residuals = y_true - y_pred
        
        # Calculate prediction intervals (±2σ)
        sigma = np.std(residuals)
        
        # Sample for plotting
        if len(y_true) > 1000:
            idx = np.random.choice(len(y_true), 1000, replace=False)
            y_true, y_pred = y_true[idx], y_pred[idx]
            
        sorted_idx = np.argsort(y_pred)
        y_true_sorted = y_true[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]
        
        ax.scatter(y_pred_sorted, y_true_sorted, alpha=0.6, s=2)
        ax.plot(y_pred_sorted, y_pred_sorted, 'r-', label='Perfect prediction')
        ax.fill_between(y_pred_sorted, y_pred_sorted - 2*sigma, y_pred_sorted + 2*sigma,
                       alpha=0.2, label='95% PI')
        
        ax.set_xlabel('Predicted IV')
        ax.set_ylabel('Actual IV')
        ax.set_title('Prediction Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_error_autocorr(self, ax):
        """Plot error autocorrelation (simple lag-1)."""
        residuals = (self.results['y_test'] - self.results['y_pred']).flatten()
        
        # Calculate lag-1 autocorrelation
        if len(residuals) > 1000:
            residuals = residuals[:1000]  # Sample for speed
            
        lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        
        ax.scatter(residuals[:-1], residuals[1:], alpha=0.6, s=1)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Residual(t)')
        ax.set_ylabel('Residual(t+1)')
        ax.set_title(f'Error Autocorr (r={lag1_corr:.3f})')
        ax.grid(True, alpha=0.3)
    
    def _plot_model_summary(self, ax):
        """Model summary text."""
        ax.axis('off')
        
        # Collect key metrics
        metrics = self.results['metrics']
        residual_stats = self.results['residual_stats']
        
        summary_text = f"""
MODEL SUMMARY
{'='*20}
R² Score: {metrics['r2_score']:.6f}
RMSE:     {metrics['rmse']:.6f}
MAE:      {metrics['mae']:.6f}
Max Err:  {metrics['max_error']:.4f}

RESIDUAL STATS
{'='*20}
Mean:     {residual_stats['mean']:.2e}
Std Dev:  {residual_stats['std']:.6f}
Skewness: {residual_stats['skewness']:.4f}
Kurtosis: {residual_stats['kurtosis']:.4f}

NORMALITY TESTS
{'='*20}
Jarque-Bera p: {residual_stats['jarque_bera'][1]:.4f}
Shapiro p:     {residual_stats['shapiro_wilk'][1]:.4f}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    def plot_iv_surface_comparison(self, sample_idx=0, strike_grid=None, tenor_grid=None, 
                                  figsize=(15, 5), save_path=None):
        """Plot IV surface comparison (actual vs predicted)."""
        if strike_grid is None or tenor_grid is None:
            print("Strike and tenor grids required for surface plotting")
            return
            
        n_strikes, n_tenors = len(strike_grid), len(tenor_grid)
        
        y_true_surf = self.results['y_test'][sample_idx].reshape(n_strikes, n_tenors)
        y_pred_surf = self.results['y_pred'][sample_idx].reshape(n_strikes, n_tenors)
        error_surf = np.abs(y_true_surf - y_pred_surf)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Actual surface
        im1 = axes[0].contourf(tenor_grid, strike_grid, y_true_surf, levels=20, cmap='viridis')
        axes[0].set_title('Actual IV Surface')
        axes[0].set_xlabel('Tenor')
        axes[0].set_ylabel('Strike')
        plt.colorbar(im1, ax=axes[0])
        
        # Predicted surface
        im2 = axes[1].contourf(tenor_grid, strike_grid, y_pred_surf, levels=20, cmap='viridis')
        axes[1].set_title('Predicted IV Surface')
        axes[1].set_xlabel('Tenor')
        axes[1].set_ylabel('Strike')
        plt.colorbar(im2, ax=axes[1])
        
        # Error surface
        im3 = axes[2].contourf(tenor_grid, strike_grid, error_surf, levels=20, cmap='Reds')
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('Tenor')
        axes[2].set_ylabel('Strike')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"IV surface comparison saved to: {save_path}")
            
        plt.show()
    
    def export_results(self, filepath):
        """Export evaluation results to file."""
        # Prepare exportable results
        export_data = {
            'metrics': self.results['metrics'],
            'bucket_metrics': self.results['bucket_metrics'],
            'residual_stats': {
                k: v if not isinstance(v, tuple) else list(v) 
                for k, v in self.results['residual_stats'].items()
            }
        }
        
        # Convert numpy arrays to lists
        for key in ['percentiles']:
            if key in export_data['residual_stats']:
                export_data['residual_stats'][key] = export_data['residual_stats'][key].tolist()
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Evaluation results exported to: {filepath}")

# =============================================================================
# Helper Functions  
# =============================================================================

def test_evaluation_module():
    """Test the evaluation module with dummy data."""
    print("Testing model evaluation module...")
    
    try:
        # QRHModelEvaluator is now defined in this same file, no import needed
        print("QRHModelEvaluator class is available in this file")
        
        # Create dummy data
        n_samples, n_features, n_targets = 1000, 6, 54
        X_test = np.random.randn(n_samples, n_features)
        y_test = np.random.rand(n_samples, n_targets) * 0.5 + 0.2  # IV range [0.2, 0.7]
        y_pred = y_test + np.random.normal(0, 0.05, y_test.shape)  # Add some noise
        
        print(f"Created dummy data: X_test {X_test.shape}, y_test {y_test.shape}")
        
        # Create evaluator (without actual model)
        evaluator = QRHModelEvaluator()
        print("Successfully created QRHModelEvaluator instance")
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

def load_test_data(data_size="100k", data_path=None): # type: ignore
    """Load test data for evaluation."""
    if not data_path:
        raise ValueError("You must provide --data_path to the test .npz file (e.g. --data_path data/raw/data_100k/test_100k.npz)")
    print(f"Loading test data from: {data_path}")
    data = np.load(data_path)
    # Support both npz and npy
    if isinstance(data, np.lib.npyio.NpzFile):
        # Try common key patterns
        if 'X_test' in data and 'y_test' in data:
            X_test = data['X_test']
            y_test = data['y_test']
        elif 'test_X' in data and 'test_y' in data:
            X_test = data['test_X']
            y_test = data['test_y']
        elif 'X' in data and 'y' in data:
            X_test = data['X']
            y_test = data['y']
        else:
            available_keys = list(data.keys())
            raise ValueError(f"Could not find test data keys. Available keys: {available_keys}. Expected: X_test/y_test, test_X/test_y, or X/y")
        x_scaler = data.get('x_scaler', None)
        y_scaler = data.get('y_scaler', None)
    else:
        raise ValueError("Only .npz format is supported for data_path")
    
    print(f"Loaded data: X_test {X_test.shape}, y_test {y_test.shape}")
    return X_test, y_test, x_scaler, y_scaler

def find_latest_model(model_path=None): # type: ignore
    """Find the latest trained model."""
    if model_path:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path, model_path.parent
    
    # Look for models in experiments folder
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        raise FileNotFoundError("No experiments directory found")
        
    experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and 'advanced_qrh' in d.name]
    
    if not experiment_dirs:
        raise FileNotFoundError("No advanced QRH experiment directories found")
        
    # Get the most recent experiment
    latest_experiment = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Using experiment: {latest_experiment}")
    
    # Look for model file in the experiment
    model_files = list(latest_experiment.glob("*.keras"))
    if not model_files:
        raise FileNotFoundError(f"No .keras model files found in {latest_experiment}")
        
    model_path = model_files[0]
    return model_path, latest_experiment

def load_model_and_pca(model_path, experiment_dir):
    """Load trained model and PCA components if available."""
    try:
        import tensorflow as tf
        import keras
    except ImportError:
        print("TensorFlow not available, trying to import keras directly...")
        import keras
        tf = None
    
    # Load main model
    model = keras.models.load_model(str(model_path))
    print(f"Loaded model from: {model_path}")
    print(f"Model parameters: {model.count_params():,}") # type: ignore
    
    # Look for PCA model in the same experiment directory
    pca_model = None
    pca_info_path = experiment_dir / "pca_info.pkl"
    if pca_info_path.exists():
        with open(pca_info_path, "rb") as f:
            pca_info = pickle.load(f)
            pca_model = pca_info.get('pca')  # Key is 'pca', not 'pca_model'
            if pca_model is not None:
                pca_components = pca_info.get('K', len(pca_model.components_))
                explained_var = pca_info.get('total_explained', sum(pca_model.explained_variance_ratio_))
                print(f"Loaded PCA model with {pca_components} components (explained variance: {explained_var:.6f})")
                
    # Fallback: try to find PCA model files
    if pca_model is None:
        for pca_components in [30, 20, 15, 12]:
            pca_path = experiment_dir / f"pca_model_{pca_components}components.pkl"
            if pca_path.exists():
                with open(pca_path, "rb") as f:
                    pca_model = pickle.load(f)
                print(f"Loaded PCA model with {pca_components} components")
                break
    
    return model, pca_model

def get_strike_tenor_grids(grid_size=60):
    """Get strike and tenor grids for surface plotting."""
    if grid_size == 60:
        # 10×6 grid
        strikes = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])  # 10 strikes  
        tenors = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 5.0])  # 6 tenors
    elif grid_size == 54:
        # 9×6 grid
        strikes = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])  # 9 strikes
        tenors = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 5.0])  # 6 tenors
    else:
        # Approximate grid
        n_strikes = int(np.sqrt(grid_size * 1.5))  # Assume more strikes than tenors
        n_tenors = grid_size // n_strikes
        strikes = np.linspace(0.7, 1.5, n_strikes)
        tenors = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 5.0][:n_tenors])
    
    return strikes, tenors

def load_test_data(data_path):
    """Load test data from file."""
    print(f"Loading test data from: {data_path}")
    
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        
        # Check available keys
        available_keys = list(data.keys())
        print(f"Available keys: {available_keys}")
        
        # Try different key combinations
        if 'X' in data and 'y' in data:
            X_test, y_test = data['X'], data['y']
        elif 'X_test' in data and 'y_test' in data:
            X_test, y_test = data['X_test'], data['y_test']
        else:
            raise KeyError(f"Expected 'X'/'y' or 'X_test'/'y_test' keys, found: {available_keys}")
            
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Loaded data: X_test {X_test.shape}, y_test {y_test.shape}")
    return X_test, y_test

def find_latest_model(base_dir="experiments"):
    """Find the latest trained model."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Experiments directory not found: {base_dir}")
    
    # Look for experiment directories
    experiment_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in {base_dir}")
    
    # Sort by modification time (most recent first)
    experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_dir = experiment_dirs[0]
    
    print(f"Using experiment: {latest_dir}")
    
    # Look for model file
    model_files = list(latest_dir.glob("*.keras"))
    if not model_files:
        model_files = list(latest_dir.glob("*.h5"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {latest_dir}")
    
    model_path = model_files[0]  # Take the first one found
    
    # Look for PCA model
    pca_path = latest_dir / "pca_info.pkl"  # Fixed filename
    pca_model = None
    if pca_path.exists():
        print(f"Loading PCA model from: {pca_path}")
        with open(pca_path, 'rb') as f:
            pca_info = pickle.load(f)
            pca_model = pca_info['pca']  # Extract PCA from pca_info dict
        print(f"Loaded PCA model with {pca_model.n_components_} components "
              f"(explained variance: {np.sum(pca_model.explained_variance_ratio_):.6f})")
    
    return model_path, pca_model, latest_dir

def create_evaluation_report(evaluator, model_name, save_dir="reports/evaluation"):
    """Create comprehensive evaluation report."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_path = save_path / f"{model_name}_evaluation.png"
    evaluator.plot_comprehensive_evaluation(save_path=plot_path)
    
    # Generate surface comparison if possible
    try:
        # Create default grids
        strike_grid = np.linspace(0.8, 1.2, 10)
        tenor_grid = np.linspace(0.1, 2.0, 6)
        
        surface_path = save_path / f"{model_name}_surface_comparison.png"
        evaluator.plot_iv_surface_comparison(
            strike_grid=strike_grid, 
            tenor_grid=tenor_grid,
            save_path=surface_path
        )
    except Exception as e:
        print(f"Could not generate surface comparison: {e}")
    
    # Export results
    results_path = save_path / f"{model_name}_results.json"
    evaluator.export_results(results_path)
    
    return save_path

def run_comprehensive_evaluation(model_path=None, data_path=None):
    """Run comprehensive evaluation for QRH model."""
    print("Running comprehensive evaluation for QRH Advanced Model...")
    
    try:
        # Load test data
        X_test, y_test = load_test_data(data_path)
        
        # Load model
        if model_path:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            model_path = Path(model_path)
            pca_model = None
            experiment_dir = model_path.parent
            print(f"Using specified model: {model_path}")
        else:
            model_path, pca_model, experiment_dir = find_latest_model()
            print(f"Found PCA model: {pca_model is not None}")
        
        # Load the model
        print(f"Loading model from: {model_path}")
        model = keras.models.load_model(str(model_path))
        print(f"Model parameters: {model.count_params():,}") # type: ignore
        
        # Load PCA model if not already loaded
        if pca_model is None:
            pca_path = experiment_dir / "pca_info.pkl"  # Fixed filename
            if pca_path.exists():
                print(f"Loading PCA model from: {pca_path}")
                with open(pca_path, 'rb') as f:
                    pca_info = pickle.load(f)
                    pca_model = pca_info['pca']  # Extract PCA from dict
                print(f"Loaded PCA model with {pca_model.n_components_} components "
                      f"(explained variance: {np.sum(pca_model.explained_variance_ratio_):.6f})")
        
        print(f"Final PCA model status: {pca_model is not None}")
        
        # Create grids for analysis
        strike_grid = np.linspace(0.8, 1.2, 10)
        tenor_grid = np.linspace(0.1, 2.0, 6)
        print(f"Using grids: {len(strike_grid)} strikes × {len(tenor_grid)} tenors = {len(strike_grid) * len(tenor_grid)} points")
        
        # Create evaluator and run evaluation
        evaluator = QRHModelEvaluator(model=model, pca_model=pca_model)
        
        print("\nRunning comprehensive evaluation...")
        results = evaluator.evaluate_model(
            X_test, y_test,
            strike_grid=strike_grid,
            tenor_grid=tenor_grid,
            verbose=True
        )
        
        # Generate model name
        model_name = f"QRH_Advanced_100k"
        if pca_model:
            model_name += f"_PCA{pca_model.n_components_}"
        
        # Create comprehensive report
        save_dir = create_evaluation_report(evaluator, model_name)
        
        print("\nComplete evaluation report generated!")
        print(f"Files saved in: {save_dir}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 60)
        
        metrics = results['metrics']
        print("Model Performance:")
        print(f"   R² Score:      {metrics['r2_score']:.6f}")
        print(f"   RMSE:          {metrics['rmse']:.6f}")
        print(f"   MAE:           {metrics['mae']:.6f}")
        print(f"   Max Error:     {metrics['max_error']:.6f}")
        print(f"   Median AE:     {metrics['median_ae']:.6f}")
        
        if pca_model:
            print("\nPCA Analysis:")
            print(f"   Components:    {pca_model.n_components_}")
            print(f"   Explained Var: {np.sum(pca_model.explained_variance_ratio_):.6f}")
        
        print("\nModel Complexity:")
        print(f"   Parameters:    {model.count_params():,}") # type: ignore
        
        if results['bucket_metrics']:
            print("\nBucket Performance (RMSE):")
            bucket_rmse = {k: v['rmse'] for k, v in results['bucket_metrics'].items()}
            for bucket, rmse in bucket_rmse.items():
                print(f"   {bucket:<12}: {rmse:.6f}")
            
            best_bucket = min(bucket_rmse.keys(), key=lambda k: bucket_rmse[k])
            worst_bucket = max(bucket_rmse.keys(), key=lambda k: bucket_rmse[k])
            print(f"\n   Best bucket:  {best_bucket} ({bucket_rmse[best_bucket]:.6f})")
            print(f"   Worst bucket: {worst_bucket} ({bucket_rmse[worst_bucket]:.6f})")
        
        residual_stats = results['residual_stats']
        jb_stat, jb_pvalue = residual_stats['jarque_bera']
        print("\nQuality Assessment:")
        print(f"   Residual Mean: {residual_stats['mean']:.2e}")
        print(f"   Residual Std:  {residual_stats['std']:.6f}")
        print(f"   Normality p:   {jb_pvalue:.4f}")
        
        print("\nQuality Checks:")
        if jb_pvalue < 0.05:
            print("   Residuals may not be normally distributed")
        else:
            print("   Residuals appear normally distributed")
            
        if metrics['r2_score'] > 0.99:
            print("   Excellent model fit (R² > 0.99)")
        elif metrics['r2_score'] > 0.95:
            print("   Very good model fit (R² > 0.95)")
        else:
            print("   Model fit could be improved")
        
        if abs(residual_stats['mean']) < 0.001:
            print("   Low bias in predictions")
        else:
            print("   Some bias detected in predictions")
        
        print(f"\nAll evaluation reports saved in: {save_dir}/")
        print("Comprehensive evaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive QRH Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 comprehensive_evaluation.py --data_path data/raw/data_100k/test_100k.npz
  python3 comprehensive_evaluation.py --model_path experiments/model.keras --data_path data/raw/data_100k/test_100k.npz
        """
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to specific model file (.keras)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test .npz file (e.g. data/raw/data_100k/test_100k.npz)")
    
    args = parser.parse_args()

    print("QRH Comprehensive Model Evaluation")
    print("="*50)
    print("Running model evaluation...")
    print(f"Model: {args.model_path or 'latest'}")
    print(f"Test data: {args.data_path}")
    
    success = run_comprehensive_evaluation(
        model_path=args.model_path,
        data_path=args.data_path
    )
    
    if success:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
