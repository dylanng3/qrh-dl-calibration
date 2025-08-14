"""
Heston Calibration Project
=========================

A professional-grade implementation for Heston stochastic volatility model
option pricing using neural networks and hyperparameter optimization.

Package Structure:
- processing: Data pipeline (generation, conversion, pricing)
- models: ML models (architectures, training, evaluation)  
- optimization: HPO system (objectives, strategies, analysis)
- pricing: Option pricing methods (heston, analytics)
- utils: Common utilities (config, visualization)
"""

__version__ = "0.1.0"
__author__ = "Heston Calibration Project"

# Import main packages - Commented out to avoid circular imports
# from . import processing
# from . import models  # Commented out to avoid circular import
# from . import optimization  
# from . import pricing
# from . import utils

# Quick access to commonly used modules - Commented out to avoid circular imports
# from .processing.data_gen import *
# from .processing.heston_fft import *
# from .processing.converter import *

__all__ = [
    # "processing",
    # "models",  # Commented out to avoid circular import
    # "optimization",
    # "pricing",
    # "utils"
]
