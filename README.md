# Qwen3 From Scratch

A PyTorch implementation of the Qwen3 language model built from scratch, featuring modern transformer architecture components including Grouped Query Attention (GQA), RoPE (Rotary Position Encoding), and RMSNorm.

## Features

- **Grouped Query Attention (GQA)**: Efficient attention mechanism with 16 query heads and 8 key-value heads
- **RoPE (Rotary Position Encoding)**: Advanced positional encoding for better long-range dependencies
- **RMSNorm**: Root Mean Square normalization for improved training stability
- **SwiGLU FFN**: Swish-Gated Linear Unit feed-forward network
- **Causal Masking**: Proper autoregressive language modeling setup

## Model Architecture

### Configuration
- **Model Size**: ~0.6B parameters
- **Layers**: 28 transformer blocks
- **Embedding Dimension**: 1024
- **Vocabulary Size**: 151,936 tokens
- **Attention Heads**: 16 query heads, 8 key-value heads
- **Head Dimension**: 128
- **Hidden Dimension**: 3072
- **Max Context Length**: 40,96 tokens
- **Base Frequency**: 1,000,000.0 (for RoPE)

### Key Components

#### 1. **Config Class**
Centralized configuration management for all model hyperparameters.

#### 2. **RoPE (Rotary Position Encoding)**
- `calculate_sin_cos()`: Precomputes sine and cosine values for position encoding
- `rotate()`: Applies rotary position encoding to query and key vectors

#### 3. **RMSNORM**
Root Mean Square normalization layer with learnable scaling parameter (alpha).

#### 4. **FFN (Feed Forward Network)**
SwiGLU-based feed-forward network with:
- Two input projections (gate and up)
- SiLU activation function
- Output projection back to embedding dimension

#### 5. **GQA (Grouped Query Attention)**
Advanced attention mechanism featuring:
- Separate Q, K, V projections
- Q and K normalization
- KV head repetition for grouped attention
- Causal masking for autoregressive generation

#### 6. **Block (Transformer Layer)**
Complete transformer block with:
- Pre-norm architecture (RMSNorm before attention and FFN)
- Residual connections
- Shared RoPE calculations

#### 7. **QWEN (Main Model)**
Complete language model with:
- Token embeddings
- Stack of transformer blocks
- Final layer normalization
- Language modeling head

## Usage

### Basic Usage

```python
import torch
from main import QWEN

# Initialize model
model = QWEN()

# Create input tokens (example)
input_tokens = torch.tensor([[1, 2, 3, 4, 5]])  # Shape: (batch_size, seq_len)

# Forward pass
with torch.no_grad():
    logits = model(input_tokens)
    print(f"Output shape: {logits.shape}")  # (batch_size, seq_len, vocab_size)
```

### Model Information

```python
# Get total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Get unique parameters (accounting for weight tying)
unique_params = total_params - model.embeddings.weight.numel()
print(f"Unique parameters: {unique_params:,}")
```

## Architecture Details

### Attention Mechanism
- **Multi-Head Attention**: 16 attention heads for rich representation learning
- **Grouped Query Attention**: Efficient KV caching with 8 KV heads shared across query heads
- **Causal Masking**: Upper triangular mask prevents information leakage from future tokens

### Position Encoding
- **RoPE**: Rotary position encoding applied to query and key vectors
- **Frequency Calculation**: Uses base frequency of 1M for improved extrapolation
- **Angle Duplication**: Properly handles head dimension for rotation

### Normalization
- **RMSNorm**: More stable than LayerNorm, especially for large models
- **Q/K Normalization**: Additional normalization on query and key projections
- **Pre-Norm Architecture**: Normalization applied before each sub-layer

### Feed-Forward Network
- **SwiGLU**: Combines Swish activation with gating mechanism
- **Expansion Ratio**: ~3x expansion (1024 → 3072 → 1024)
- **No Bias**: Bias-free linear layers for efficiency

## Implementation Notes

1. **Memory Efficiency**: Uses `torch.float32` by default but can be configured for different precisions
2. **Gradient Checkpointing**: Not implemented but can be added for memory savings during training
3. **KV Caching**: Structure supports efficient KV caching for inference
4. **Flash Attention**: Can be integrated by replacing the attention computation
5. **Weight Initialization**: Uses PyTorch defaults (can be customized for better training)

## Getting Started

1. Ensure you have PyTorch installed:
   ```bash
   pip install torch
   ```
    or

    ```bash
   uv add torch
   ```

2. Run the model directly:
   ```bash
   python main.py
   ```
   or

   ```bash
   uv run main.py
   ```

3. This will output the model architecture and a sample forward pass result.

