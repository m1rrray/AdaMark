import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Seed Python, NumPy and PyTorch RNGs for reproducibility

    cudnn.benchmark stays on: we trade strict determinism for throughput.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
