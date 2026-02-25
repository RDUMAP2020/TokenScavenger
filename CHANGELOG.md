# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-25

### Added
- Initial release of DeepMime Trainer
- LAA-GiGPO implementation with two-level credit assignment
- Carousel Memory Alignment for experience replay
- Adaptive entropy control to prevent collapse
- Shaped rewards (whitespace penalty, diversity bonus, length bonus)
- Differentiable memory with MoE routing
- Weight-encoded teacher-student knowledge distillation
- Hybrid weight encoder (sparse + PCA)
- Support for DeepSeek-Coder-1.3b and 33b teacher models

### Features
- Dr. GRPO (unbiased constant normalisation)
- DAPO (asymmetric clip-higher bounds)
- MC-GRPO (median-centred baseline)
- RLEP (experience replay buffer)
- Constrained GRPO (Lagrangian constraint scalarisation)
- LUMOS-GRPO approximation (adaptive termination penalty)

## [Unreleased]

### Planned
- Additional teacher model support
- Enhanced evaluation metrics
- Pre-trained model checkpoints
