"""
Example Usage - TokenScavenger
================================

This script demonstrates how to use the TokenScavenger trainer with ATLAS-GRPO integration.
"""

import torch
from TokenScavenger import Config, DeepMimeTrainer

def main():
    """Main training example."""
    
    # Print configuration
    print("=" * 60)
    print("TokenScavenger Trainer Configuration")
    print("=" * 60)
    print(f"Teacher Model: {Config.get_teacher_model_id()}")
    print(f"Teacher Hidden Dim: {Config.get_teacher_hidden_dim()}")
    print(f"Teacher Layers: {Config.get_teacher_num_layers()}")
    print(f"Vocab Size: {Config.get_vocab_size()}")
    print(f"Student Hidden Dim: {Config.HIDDEN_DIM}")
    print(f"Student Layers: {Config.NUM_LAYERS}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Device: {Config.DEVICE}")
    print("=" * 60)
    
    # Check device availability
    if Config.DEVICE == "cuda":
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize trainer
    # Note: You need to load actual teacher/student models for real training
    print("\nInitializing trainer...")
    
    try:
        trainer = DeepMimeTrainer(
            config=Config,
            teacher_model_id=Config.get_teacher_model_id(),
            student_model=None,  # Will auto-create student model
        )
        print("Trainer initialized successfully!")
        
        # Example: Print model architecture summary
        print("\nModel Architecture Summary:")
        print(f"  - Student layers: {Config.NUM_LAYERS}")
        print(f"  - Hidden dimension: {Config.HIDDEN_DIM}")
        print(f"  - Number of heads: {Config.NUM_HEADS}")
        print(f"  - Latent dimension: {Config.LATENT_DIM}")
        print(f"  - MoE experts: {Config.NUM_EXPERTS} (active: {Config.ACTIVE_EXPERTS})")
        print(f"  - Memory slots: {Config.MEMORY_SLOTS}")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        print("\nNote: This is expected if model weights are not downloaded.")
        print("The example demonstrates the configuration structure.")

if __name__ == "__main__":
    main()
