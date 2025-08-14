"""
model_architectures.py
=====================
Neural Network Architecture for Quadratic Rough Heston Model

Target: Build NN_Index_MtP: (ω, z₀) → IVS
- Input: 15 dimensions (5 model params + 10 initial factors)
- Output: 60 dimensions (flattened IV surface 15×4)
- Architecture: MLP with 7 hidden layers × 25 nodes each
- Activation: SiLU (Swish)
- Input normalization: [-1, 1]
- Output normalization: StandardScaler (mean=0, std=1)

Key Design Principles:
✓ Simple MLP architecture for stable training
✓ Consistent layer size (25 nodes) for uniform gradient flow
✓ SiLU activation for smooth gradients
✓ Suitable for L-BFGS-B optimization with autograd
"""

import tensorflow as tf
import keras
from typing import Optional, Dict, Any
import numpy as np


# ----------------------------------------------------------------------
# Quadratic Rough Heston MLP Architecture
# ----------------------------------------------------------------------

def build_qrh_mlp(
    input_dim: int = 15,
    output_dim: int = 60,
    hidden_layers: int = 7,
    hidden_units: int = 25,
    activation: str = "swish",  # SiLU/Swish
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False,
    name: str = "QRH_MLP"
) -> keras.Model:
    """
    Build MLP for Quadratic Rough Heston pricing function.
    
    Args:
        input_dim: Input dimensions (15: 5 model params + 10 factors)
        output_dim: Output dimensions (60: flattened IV surface)
        hidden_layers: Number of hidden layers (7)
        hidden_units: Units per hidden layer (25)
        activation: Activation function ("swish" for SiLU)
        dropout_rate: Dropout rate (0.0 for no dropout)
        use_batch_norm: Whether to use batch normalization
        name: Model name
    
    Returns:
        Keras model
    """
    # Input layer
    inputs = keras.layers.Input(shape=(input_dim,), name="input_params")
    
    x = inputs
    
    # Hidden layers
    for i in range(hidden_layers):
        x = keras.layers.Dense(
            hidden_units,
            activation=activation,
            kernel_initializer='he_normal',
            name=f"hidden_{i+1}"
        )(x)
        
        if use_batch_norm:
            x = keras.layers.BatchNormalization(name=f"bn_{i+1}")(x)
            
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)
    
    # Output layer (linear activation for regression)
    outputs = keras.layers.Dense(
        output_dim,
        activation='linear',
        kernel_initializer='glorot_normal',
        name="output_iv_surface"
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    return model


# ----------------------------------------------------------------------
# Alternative Architectures for Comparison
# ----------------------------------------------------------------------

def build_wider_mlp(
    input_dim: int = 15,
    output_dim: int = 60,
    hidden_layers: int = 5,
    hidden_units: int = 64,
    activation: str = "swish",
    dropout_rate: float = 0.1,
    name: str = "Wider_MLP"
) -> keras.Model:
    """Build wider but shallower MLP for comparison"""
    inputs = keras.layers.Input(shape=(input_dim,), name="input_params")
    
    x = inputs
    
    for i in range(hidden_layers):
        # Progressive width reduction
        units = hidden_units // (1 + i // 2)
        x = keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer='he_normal',
            name=f"hidden_{i+1}"
        )(x)
        
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)
    
    outputs = keras.layers.Dense(
        output_dim,
        activation='linear',
        kernel_initializer='glorot_normal',
        name="output_iv_surface"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def build_deeper_mlp(
    input_dim: int = 15,
    output_dim: int = 60,
    hidden_layers: int = 10,
    hidden_units: int = 32,
    activation: str = "swish",
    dropout_rate: float = 0.05,
    use_residual: bool = True,
    name: str = "Deeper_MLP"
) -> keras.Model:
    """Build deeper MLP with optional residual connections"""
    inputs = keras.layers.Input(shape=(input_dim,), name="input_params")
    
    # First layer to match hidden dimensions
    x = keras.layers.Dense(
        hidden_units,
        activation=activation,
        kernel_initializer='he_normal',
        name="input_projection"
    )(inputs)
    
    # Hidden layers with optional residual connections
    for i in range(hidden_layers):
        residual = x
        
        x = keras.layers.Dense(
            hidden_units,
            activation=activation,
            kernel_initializer='he_normal',
            name=f"hidden_{i+1}"
        )(x)
        
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)
        
        # Add residual connection every 2 layers
        if use_residual and i % 2 == 1:
            x = keras.layers.Add(name=f"residual_{i+1}")([x, residual])
    
    outputs = keras.layers.Dense(
        output_dim,
        activation='linear',
        kernel_initializer='glorot_normal',
        name="output_iv_surface"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


# ----------------------------------------------------------------------
# Model Factory
# ----------------------------------------------------------------------

def create_qrh_model(
    model_type: str = "qrh_mlp",
    input_dim: int = 15,
    output_dim: int = 60,
    **kwargs
) -> keras.Model:
    """
    Factory function to create different model architectures.
    
    Args:
        model_type: Type of model ("qrh_mlp", "wider_mlp", "deeper_mlp")
        input_dim: Input dimensions
        output_dim: Output dimensions
        **kwargs: Additional arguments for specific architectures
    
    Returns:
        Keras model
    """
    if model_type == "qrh_mlp":
        return build_qrh_mlp(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif model_type == "wider_mlp":
        return build_wider_mlp(input_dim=input_dim, output_dim=output_dim, **kwargs)
    elif model_type == "deeper_mlp":
        return build_deeper_mlp(input_dim=input_dim, output_dim=output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------------------------------------------------
# Model Configuration for Training
# ----------------------------------------------------------------------

def get_model_config() -> Dict[str, Any]:
    """Get default model configuration for QRH MLP"""
    return {
        "model_type": "qrh_mlp",
        "input_dim": 15,
        "output_dim": 60,
        "hidden_layers": 7,
        "hidden_units": 25,
        "activation": "swish",
        "dropout_rate": 0.0,
        "use_batch_norm": False,
        
        # Training configuration
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 100,
        "early_stopping_patience": 5,
        
        # Learning rate schedule
        "lr_schedule": "step",
        "lr_decay_factor": 0.5,
        "lr_decay_epochs": 10,
        
        # Loss function
        "loss": "mse",
        "metrics": ["mae", "mse"]
    }


def compile_qrh_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    loss: str = "mse",
    metrics: list = None
) -> keras.Model:
    """
    Compile QRH model with appropriate settings.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate
        optimizer: Optimizer type
        loss: Loss function
        metrics: List of metrics to track
    
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ["mae", "mse"]
    
    # Setup optimizer
    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "adamw":
        opt = keras.optimizers.AdamW(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    
    return model


# ----------------------------------------------------------------------
# Model Summary and Visualization
# ----------------------------------------------------------------------

def print_model_summary(model: keras.Model, input_shape: tuple = (15,)) -> None:
    """Print detailed model summary"""
    print("="*60)
    print(f"Model: {model.name}")
    print("="*60)
    model.summary()
    print("="*60)
    print(f"Total parameters: {model.count_params():,}")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {model.output_shape}")
    print("="*60)


# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Create and display the main QRH MLP model
    print("Creating Quadratic Rough Heston MLP...")
    
    model = create_qrh_model("qrh_mlp")
    model = compile_qrh_model(model)
    
    print_model_summary(model)
    
    # Test with dummy data
    import numpy as np
    dummy_input = np.random.randn(32, 15)  # Batch of 32 samples
    dummy_output = model(dummy_input)
    
    print(f"\nTest with dummy data:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print(f"Output range: [{dummy_output.numpy().min():.3f}, {dummy_output.numpy().max():.3f}]")
