"""
utils.py — Shared utility functions for the ABSA pipeline
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[utils] Seed set to {seed}")


def get_device():
    """Return the best available device (CUDA > CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[utils] Using device: {device}")
    if device.type == "cuda":
        print(f"        GPU: {torch.cuda.get_device_name(0)}")
    return device
