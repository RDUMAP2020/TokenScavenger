# TokenScavenger

**Author:** Maulidi Barasa  
**Bismillahi Rahmani Rahim**

## Overview

TokenScavenger is a reinforcement learning trainer that implements **Adaptive Trajectory-Level Advantage Shaping for GRPO** with the following innovations:

## Key Features

### TokenScavenger: Two-Level Credit Assignment
- **Micro-advantage**: Normalises within each latent group
- **Macro-advantage**: Normalises across groups
- Combines latent group advantages for improved policy gradient estimation

### Carousel Memory Alignment
- MSE loss between current and historical log-probabilities on replay sequences
- Prevents policy drift from historically successful trajectories

### Adaptive Entropy Control
- Prevents entropy collapse during training
- Target entropy: ~1.5 nats (≈15% of log vocabulary size)

### Shaped Rewards
- **Whitespace penalty**: Discourages excessive whitespace (>50%)
- **Diversity bonus**: Rewards unique token ratio
- **Length bonus**: Saturates at 20 tokens

### Differentiable Memory with MoE Routing
- Mixture-of-Experts (MoE) with 4 experts, 2 active
- Memory slots with key-value attention
- Load balancing for expert stability

### Weight-Encoded Knowledge Distillation
- **Hybrid encoder**: Combines sparse top-K and PCA embeddings
- **Global weight statistics**: 5-dimensional feature vector
- Supports teacher models: DeepSeek-Coder-1.3B and 33B

## Architecture

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TEACHER_ARCH` | "1.3b" | Teacher model size |
| `HIDDEN_DIM` | 1024 | Student hidden dimension |
| `NUM_LAYERS` | 6 | Number of transformer layers |
| `NUM_HEADS` | 4 | Number of attention heads |
| `LATENT_DIM` | 128 | Latent dimension |
| `NUM_EXPERTS` | 4 | Total MoE experts |
| `ACTIVE_EXPERTS` | 2 | Active experts per forward pass |
| `MEMORY_SLOTS` | 64 | Memory capacity |
| `LEARNING_RATE` | 5.0e-4 | Base learning rate |
| `BATCH_SIZE` | 1 | Training batch size |

### Weight Encoders

#### SparseWeightEncoder
Extracts top-K magnitude elements and projects through a gated MLP.

#### PCAWeightEncoder
Encodes weight matrices via truncated SVD (PCA) of random sub-blocks.

#### HybridWeightEncoder
Fuses sparse and PCA embeddings with global statistics.

## Training Parameters

### Loss Weights
- `CE_LOSS_WEIGHT`: 1.0 (Cross-entropy anchor)
- `GRPO_LOSS_WEIGHT`: 0.25 (GRPO improvement)
- `HIDDEN_LOSS_WEIGHT`: 0.275 (Hidden state alignment)

### RL Warmup
- Pure CE for first 100 steps
- GRPO weight ramps to 0.2 after warmup

### GRPO Settings
- `BETA_GRPO`: 0.0 (KL penalty coefficient)
- `ADVANTAGE_CLAMP_MAX`: 5.0
- `ROLLOUT_GROUP_SIZE`: 4

## Usage

```python
from TokenScavenger import Config, DeepMimeTrainer

# Initialize trainer
trainer = DeepMimeTrainer(
    config=Config,
    teacher_model_id=Config.get_teacher_model_id(),
    student_model=None,  # Auto-creates student
)

# Training loop
for step in range(num_steps):
    trainer.train_step()
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- NumPy

## File Structure

```
TokenScavenger.py    # Main trainer implementation
├── Config             # Hyperparameter configuration
├── Weight Encoders    # Sparse, PCA, Hybrid encoders
├── DeepMimeTrainer    # Main training class
└── Utility Functions  # Reward shaping, metrics
```

## License

This code is provided for research purposes.
