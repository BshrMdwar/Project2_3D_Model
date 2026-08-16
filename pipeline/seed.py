"""
seed.py

Reproducibility manager. Sets all relevant RNG seeds (Python's random, numpy,
torch CPU, torch CUDA if available) from a single config value.

MUST be called at the start of any training run or experiment, BEFORE any
dataset shuffling, model weight initialization, or augmentation happens.
Without this, comparing two experiments is unreliable -- any difference in
results could be pure randomness, not the actual change being tested.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """
    Set all RNG seeds for full reproducibility.

    Args:
        seed: the single seed value to use everywhere.
        deterministic_cudnn: if True, forces cuDNN into deterministic mode
            (torch.backends.cudnn.deterministic = True, benchmark = False).
            This costs some GPU speed but guarantees identical results across
            runs/machines (important when later comparing Windows-local vs
            Kaggle GPU runs of the same experiment). Set to False if you only
            care about statistical reproducibility, not bit-for-bit identical
            runs, and want the cuDNN autotuner's speed benefit.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic_cudnn:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True
    else:
        logger.warning(
            "torch is not installed in this environment -- only random/numpy/"
            "PYTHONHASHSEED were seeded. This is expected if you're running "
            "unit tests without the full training environment installed."
        )

    logger.info(f"Global seed set to {seed} (deterministic_cudnn={deterministic_cudnn})")


def seed_worker(worker_id: int) -> None:
    """
    Pass this as `worker_init_fn` to torch.utils.data.DataLoader when num_workers > 0.
    Without this, each DataLoader worker process has its own independent (and
    non-reproducible) numpy/random state, which silently breaks augmentation
    reproducibility even if set_seed() was called in the main process.
    """
    worker_seed = (torch.initial_seed() if _TORCH_AVAILABLE else 0) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
